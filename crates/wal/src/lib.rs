use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

use gpu_db_types::{EngineError, TxnId};

// E1 step 1: the optional FUA fence-pool durability backend (unix-only — it drives a
// `gpu_db_write_conveyor::FuaFrameLog`, an `O_DIRECT|O_DSYNC` construct). Default is OFF; the
// serial `WalDurableCore` path above is unchanged.
#[cfg(unix)]
mod fua;
#[cfg(unix)]
pub use fua::{fua_wal_segments_exist, recover_fua_wal_records};

// E2.5a (Variant 2): N independent ordered WAL lanes whose records carry EXPLICIT global commit
// seqs, with a cross-lane contiguous durable cut and merge recovery. Self-contained here; the
// engine consumer is a later slice. Unix-only (drives the FUA fence-pool lanes).
#[cfg(unix)]
mod fua_lanes;
#[cfg(unix)]
pub use fua_lanes::{encode_lane_frame_payload, recover_lanes, FuaWalLaneSet};

const WAL_SEGMENT_MAGIC: &[u8; 10] = b"GPUDBWAL1\n";
const WAL_CONTROL_MAGIC: &str = "GPUDBWALCONTROL1";
const WAL_ARCHIVE_MANIFEST_MAGIC: &str = "GPUDBWALARCHIVE1";
const WAL_ARCHIVE_TIMELINE_MAGIC: &str = "GPUDBWALTIMELINE1";
const WAL_ARCHIVE_TIMELINE_REGISTRY_MAGIC: &str = "GPUDBWALTIMELINEREGISTRY1";
const WAL_ARCHIVE_OBJECT_BACKUP_MAGIC: &str = "GPUDBWALOBJECTBACKUP1";
const WAL_RECORD_HEADER_LEN: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
    pub txn_id: TxnId,
    /// W1a: shared with the replication log entry + the commit-wave item (one allocation per
    /// statement, refcounted; was a fresh `Vec` copy per record on the commit hot path).
    pub payload: std::sync::Arc<[u8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalCheckpointMeta {
    pub durable_record_count: usize,
    pub last_durable_txn_id: Option<TxnId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalControlFile {
    pub segment_path: PathBuf,
    pub checkpoint: WalCheckpointMeta,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveSegment {
    pub segment_path: PathBuf,
    pub record_count: usize,
    pub first_txn_id: Option<TxnId>,
    pub last_txn_id: Option<TxnId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveManifest {
    pub segments: Vec<WalArchiveSegment>,
    pub checkpoint: WalCheckpointMeta,
    pub record_timestamps: Vec<WalArchiveRecordTimestamp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalArchiveRecoveryTarget {
    pub target_txn_id: TxnId,
    pub recovered_record_count: usize,
    pub last_recovered_txn_id: TxnId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalArchiveRecordTimestamp {
    pub txn_id: TxnId,
    pub timestamp_micros: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalArchiveTimestampRecoveryTarget {
    pub target_timestamp_micros: u64,
    pub target_txn_id: TxnId,
    pub recovered_record_count: usize,
    pub last_recovered_txn_id: TxnId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveRetentionPlan {
    pub target_txn_id: TxnId,
    pub retained_record_count: usize,
    pub removed_record_count: usize,
    pub retained_manifest: WalArchiveManifest,
    pub removed_segments: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimeline {
    pub timeline_id: String,
    pub parent_timeline_id: Option<String>,
    pub fork_txn_id: TxnId,
    pub fork_timestamp_micros: Option<u64>,
    pub source_manifest_path: PathBuf,
    pub branch_manifest_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimelineBranch {
    pub timeline: WalArchiveTimeline,
    pub manifest: WalArchiveManifest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimelineRegistryEntry {
    pub timeline_id: String,
    pub parent_timeline_id: Option<String>,
    pub fork_txn_id: TxnId,
    pub fork_timestamp_micros: Option<u64>,
    pub timeline_path: PathBuf,
    pub branch_manifest_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimelineRegistry {
    pub timelines: Vec<WalArchiveTimelineRegistryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimelineSelection {
    pub entry: WalArchiveTimelineRegistryEntry,
    pub timeline: WalArchiveTimeline,
    pub manifest: WalArchiveManifest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveTimelinePrunePlan {
    pub retained_timeline_id: String,
    pub retained_timeline_ids: Vec<String>,
    pub removed_timeline_ids: Vec<String>,
    pub retained_registry: WalArchiveTimelineRegistry,
    pub removed_timeline_paths: Vec<PathBuf>,
    pub removed_branch_manifest_paths: Vec<PathBuf>,
    pub removed_segment_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveObject {
    pub source_path: PathBuf,
    pub object_path: PathBuf,
    pub byte_len: u64,
    pub checksum: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalArchiveObjectBackup {
    pub archive_manifest: WalArchiveManifest,
    pub objects: Vec<WalArchiveObject>,
}

/// The durable backing for a [`WalBuffer`]: an **append-only** segment writer whose mutable
/// state sits behind its OWN small lock, separate from whatever outer lock guards the buffer
/// (in the engine: the commit_mutex).
///
/// The segment file is opened (or created) once; every flush serializes ONLY the currently-
/// unflushed record tail, appends it with a single `write_all`, and `fdatasync`s it — O(new
/// records) per commit. The parent directory is fsynced once, when the file is first created.
///
/// The split lock is what makes GROUP COMMIT real: [`WalBuffer::begin_group_flush`] snapshots
/// the unflushed tail under the outer lock and hands back a [`WalGroupFlushJob`]; the job's
/// `write_all` + fsync then run with NO lock held at all, so other committers keep appending
/// (forming the next group) while the disk works; [`WalGroupFlushJob::commit`] finishes by
/// taking only THIS core's lock — never the outer one — so completion cannot deadlock against
/// an outer-lock holder waiting for the in-flight IO to drain.
///
/// Because appends are not atomic, a crash mid-append can leave a torn record tail. Recovery
/// ([`recover_wal_segment`]) distinguishes a torn tail from bit rot of acknowledged data via the
/// **durable tail-offset sidecar** (`<segment>.tail`): an invalid region at or beyond the recorded
/// offset was never acknowledged and is safely truncated; corruption below it fails loudly. The
/// sidecar is advisory (a lower bound) and is written only at cheap points — segment creation,
/// recovery install, prefix truncation, and clean shutdown — never on the per-commit path.
#[derive(Debug)]
struct WalDurableCore {
    segment_path: PathBuf,
    state: Mutex<WalDurableState>,
    /// Signals `io_in_flight` clearing (a group job completed or was abandoned), so an inline
    /// `flush_all` / prefix truncation waiting for the disk can proceed.
    cv: Condvar,
}

/// W4a — WAL segment PREALLOCATION chunk. Appending into a growing file forces the filesystem
/// to journal a size-change on EVERY `fdatasync` (measured on this box: 2.46ms p50 append-grow
/// vs 0.84ms p50 inside preallocated+zeroed extents — 3.4x, the standard Postgres/etcd WAL
/// discipline). Segments are zero-filled ahead in chunks of this size and all record IO is
/// POSITIONAL (`write_all_at` at the logical tail); the zero tail is unambiguous end-of-log to
/// both readers (an all-zero record header can never be valid: the FNV checksum of a zero
/// header is nonzero — test-asserted). Override with `GPU_DB_WAL_PREALLOC_BYTES` (min 1MB) for
/// growth tests.
fn wal_prealloc_chunk_bytes() -> u64 {
    static CHUNK: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("GPU_DB_WAL_PREALLOC_BYTES")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v.max(1024 * 1024))
            .unwrap_or(64 * 1024 * 1024)
    })
}

/// Zero-fill `[from, to)` of `file` and `sync_all` (the size/extent change is metadata — a full
/// fsync persists it so later group syncs can stay `fdatasync`-fast inside written extents).
fn zero_fill_extend(file: &File, from: u64, to: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    const ZEROS: [u8; 1024 * 1024] = [0; 1024 * 1024];
    let mut offset = from;
    while offset < to {
        let n = ((to - offset) as usize).min(ZEROS.len());
        file.write_all_at(&ZEROS[..n], offset)?;
        offset += n as u64;
    }
    file.sync_all()
}

#[derive(Debug)]
struct WalDurableState {
    /// Open write handle for POSITIONAL record IO (`write_all_at` at the logical tail
    /// `durable_bytes` — W4a; was `O_APPEND`, incompatible with preallocation because appends
    /// would land after the zero fill). `None` until the first durable flush. Behind an `Arc`
    /// so a [`WalGroupFlushJob`] can perform its IO after the lock is released.
    file: Option<Arc<File>>,
    /// W4a: physical zero-filled length. Records live in `[0, durable_bytes)`; zeros in
    /// `[durable_bytes, prealloc_bytes)`. Writes never grow the file inside this region, so
    /// `fdatasync` skips the filesystem's size-change journaling.
    prealloc_bytes: u64,
    /// Valid, fsynced byte length of the live segment (magic + serialized flushed records).
    durable_bytes: u64,
    /// Durable watermark: how many of the owning buffer's records are fsynced. Lives HERE (not
    /// in the buffer) so a group flush can advance it without the buffer's outer lock.
    flushed_records: usize,
    /// Records `[0, segment_base_records)` of the owning buffer are durable in an external
    /// checkpoint segment, not in this file (set by [`WalBuffer::truncate_durable_segment_prefix`]).
    segment_base_records: usize,
    /// Last tail offset written to the sidecar, to skip redundant rewrites.
    tail_offset_recorded: u64,
    /// A group flush job's IO is running WITHOUT the lock; nothing else may touch the file (or
    /// start a second write) until it completes and clears this.
    io_in_flight: bool,
    /// A failed append or fsync left the on-disk tail state unknown — fail closed on later
    /// flushes rather than append past a possibly-torn region (restart recovery repairs it).
    poisoned: Option<String>,
    /// Group-commit accounting (one group per real fsync, inline or via a job).
    stats: WalGroupCommitStats,
}

impl WalDurableCore {
    fn fresh(segment_path: PathBuf) -> Self {
        Self {
            segment_path,
            state: Mutex::new(WalDurableState {
                file: None,
                prealloc_bytes: 0,
                durable_bytes: 0,
                flushed_records: 0,
                segment_base_records: 0,
                tail_offset_recorded: 0,
                io_in_flight: false,
                poisoned: None,
                stats: WalGroupCommitStats::default(),
            }),
            cv: Condvar::new(),
        }
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, WalDurableState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Lock the state and wait out any in-flight group IO (used by the inline flush and the
    /// admin ops, which must not overlap a running `write_all`/fsync on the same file).
    fn lock_state_idle(&self) -> std::sync::MutexGuard<'_, WalDurableState> {
        let mut state = self.lock_state();
        while state.io_in_flight {
            state = self
                .cv
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        state
    }

    fn poisoned_error(&self, reason: &str) -> EngineError {
        EngineError::Durability(format!(
            "WAL segment {} is poisoned by an earlier flush failure ({reason}); restart to \
             recover from the durable prefix",
            self.segment_path.display()
        ))
    }

    /// Best-effort sidecar update; errors are reported but tolerable (the sidecar is a lower
    /// bound — a stale value only widens the tolerated torn-tail window, never loses data).
    fn record_tail_offset(&self, state: &mut WalDurableState) -> Result<(), EngineError> {
        if state.tail_offset_recorded == state.durable_bytes {
            return Ok(());
        }
        write_wal_tail_offset(&self.segment_path, state.durable_bytes)?;
        state.tail_offset_recorded = state.durable_bytes;
        Ok(())
    }

    /// First durable use: create (or clobber — the fresh-database constructor semantic) the
    /// segment with the magic header, fsync it, fsync the parent directory so the file's
    /// existence is itself crash-durable, and keep an `O_APPEND` handle. Any stale tail-offset
    /// sidecar from a previous database at this path is removed FIRST so a crash mid-clobber
    /// cannot pair the new (short) file with the old (large) recorded tail and read as loud
    /// corruption of a database that no longer exists.
    fn ensure_created(&self, state: &mut WalDurableState) -> Result<(), EngineError> {
        if state.file.is_some() {
            return Ok(());
        }
        let _ = fs::remove_file(wal_tail_offset_path(&self.segment_path));
        if let Some(parent) = self
            .segment_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|err| {
                EngineError::Durability(format!(
                    "failed to create WAL segment directory {}: {err}",
                    parent.display()
                ))
            })?;
        }
        {
            let mut file = File::create(&self.segment_path).map_err(|err| {
                EngineError::Durability(format!(
                    "failed to create WAL segment {}: {err}",
                    self.segment_path.display()
                ))
            })?;
            file.write_all(WAL_SEGMENT_MAGIC)
                .and_then(|_| file.sync_all())
                .map_err(|err| {
                    EngineError::Durability(format!(
                        "failed to write WAL segment {}: {err}",
                        self.segment_path.display()
                    ))
                })?;
        }
        sync_segment_parent_dir(&self.segment_path)?;
        let file = fs::OpenOptions::new()
            .write(true)
            .open(&self.segment_path)
            .map_err(|err| {
                EngineError::Durability(format!(
                    "failed to reopen WAL segment for positional IO {}: {err}",
                    self.segment_path.display()
                ))
            })?;
        // W4a: zero-fill the first preallocation chunk so per-group fdatasyncs never pay the
        // filesystem's size-change journaling (one-time cost at creation).
        let prealloc_to = wal_prealloc_chunk_bytes();
        zero_fill_extend(&file, WAL_SEGMENT_MAGIC.len() as u64, prealloc_to).map_err(|err| {
            EngineError::Durability(format!(
                "failed to preallocate WAL segment {}: {err}",
                self.segment_path.display()
            ))
        })?;
        state.prealloc_bytes = prealloc_to;
        state.file = Some(Arc::new(file));
        state.durable_bytes = WAL_SEGMENT_MAGIC.len() as u64;
        self.record_tail_offset(state)?;
        Ok(())
    }

    /// Record a successful fsync of `group_size` records ending at `target_records`.
    fn note_group(state: &mut WalDurableState, group_size: usize, target_records: usize) {
        state.flushed_records = target_records;
        state.stats.flush_groups += 1;
        state.stats.durable_records += group_size as u64;
        state.stats.max_group_size = state.stats.max_group_size.max(group_size);
    }
}

impl Drop for WalDurableCore {
    fn drop(&mut self) {
        // Clean-shutdown tail-offset record: after this, ANY invalid byte in the segment is
        // detected loudly at recovery (nothing beyond the recorded offset remains tolerable).
        let mut state = std::mem::replace(
            self.state
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            WalDurableState {
                file: None,
                prealloc_bytes: 0,
                durable_bytes: 0,
                flushed_records: 0,
                segment_base_records: 0,
                tail_offset_recorded: 0,
                io_in_flight: false,
                poisoned: None,
                stats: WalGroupCommitStats::default(),
            },
        );
        if state.tail_offset_recorded != state.durable_bytes {
            let _ = write_wal_tail_offset(&self.segment_path, state.durable_bytes);
            state.tail_offset_recorded = state.durable_bytes;
        }
    }
}

/// The outcome of [`WalBuffer::begin_group_flush`]: either an IO job to run lock-free, or the
/// news that nothing was unflushed (with the current durable watermark).
pub enum WalGroupFlushBegin {
    /// Unflushed records were snapshotted; run [`WalGroupFlushJob::commit`] to make them durable.
    Job(WalGroupFlushJob),
    /// Nothing to flush — every appended record is already durable up to `flushed_records`.
    Clean { flushed_records: usize },
}

/// A snapshotted group flush whose expensive durability step — one `write_all` + fsync for the
/// serial backend, or a frame publish + fence-pool durable-cut wait for the FUA backend — runs in
/// [`WalGroupFlushJob::commit`] with NO lock held, so appenders keep working (and the next group
/// keeps forming) while the disk syncs. The public shape (`begin_group_flush` -> `Job(job)` ->
/// `job.commit()`) is identical across backends; only the private [`WalGroupFlushJobKind`]
/// differs. The `Option` is the consume-or-abandon latch: `commit` takes it, `Drop` abandons
/// whatever is left (a caller that panicked between begin and commit), which fail-closes the
/// backing so no later flush appends past a possibly-torn region / a never-published frame gap.
pub struct WalGroupFlushJob {
    kind: Option<WalGroupFlushJobKind>,
}

enum WalGroupFlushJobKind {
    /// The serial-fdatasync backend: one positional `write_all` + `fdatasync` on the live segment.
    Serial(SerialFlushJob),
    /// The FUA fence-pool backend (E1 step 1): publish the group's frame and wait for the
    /// contiguous durable cut to cover it, allowing MULTIPLE groups durable in flight.
    #[cfg(unix)]
    Fua(fua::FuaFlushJob),
}

impl WalGroupFlushJob {
    /// Make the snapshotted group durable — call with NO locks held. For the serial backend this
    /// is one `write_all` + `fdatasync`; for the FUA backend it publishes the frame and spins on
    /// the fence pool's durable cut. Returns the new durable watermark (record count). On failure
    /// the backing is POISONED fail-closed (the group's members may already have applied their
    /// deltas; see the engine's group-commit wedge semantics).
    pub fn commit(mut self) -> Result<usize, EngineError> {
        match self.kind.take().expect("group flush job already consumed") {
            WalGroupFlushJobKind::Serial(job) => job.commit(),
            #[cfg(unix)]
            WalGroupFlushJobKind::Fua(job) => job.commit(),
        }
    }
}

impl Drop for WalGroupFlushJob {
    fn drop(&mut self) {
        // `commit` took the kind out; anything left is an abandoned-mid-flight job.
        match self.kind.take() {
            None => {}
            Some(WalGroupFlushJobKind::Serial(job)) => job.abandon(),
            #[cfg(unix)]
            Some(WalGroupFlushJobKind::Fua(job)) => job.abandon(),
        }
    }
}

/// The serial-fdatasync group flush: the serialized unflushed tail plus the open segment handle.
/// While it is outstanding the core's `io_in_flight` excludes every other writer of the file
/// (inline flushes and admin ops wait on the condvar).
struct SerialFlushJob {
    core: Arc<WalDurableCore>,
    file: Arc<File>,
    /// W4a: the logical tail this group writes at (positional IO inside preallocated extents).
    offset: u64,
    /// W4a: the zero-filled frontier at snapshot time; the job extends it lock-free if needed
    /// (`io_in_flight` already excludes every other writer of the file).
    prealloc_end: u64,
    bytes: Vec<u8>,
    target_records: usize,
    group_size: usize,
}

impl SerialFlushJob {
    /// Perform the group's IO (one `write_all`, one `fdatasync`) then complete under the durable
    /// core's own lock: advance the watermark + stats and wake waiters. On IO failure the backing
    /// is POISONED fail-closed and waiters are still woken.
    fn commit(self) -> Result<usize, EngineError> {
        use std::os::unix::fs::FileExt;
        // W4a: extend the zero-filled frontier lock-free if this group crosses it (rare — once
        // per chunk; `io_in_flight` excludes every other writer), then write POSITIONALLY at
        // the snapshotted logical tail so `sync_data` never pays size-change journaling.
        let write_end = self.offset + self.bytes.len() as u64;
        let mut new_prealloc_end = self.prealloc_end;
        let io_result = if write_end > self.prealloc_end {
            new_prealloc_end = write_end.max(self.prealloc_end + wal_prealloc_chunk_bytes());
            zero_fill_extend(&self.file, self.prealloc_end, new_prealloc_end)
        } else {
            Ok(())
        }
        .and_then(|_| self.file.write_all_at(&self.bytes, self.offset))
        .and_then(|_| self.file.sync_data());
        let mut state = self.core.lock_state();
        state.io_in_flight = false;
        let outcome = match io_result {
            Ok(()) => {
                state.prealloc_bytes = state.prealloc_bytes.max(new_prealloc_end);
                state.durable_bytes += self.bytes.len() as u64;
                WalDurableCore::note_group(&mut state, self.group_size, self.target_records);
                Ok(self.target_records)
            }
            Err(err) => {
                // A partial positional write leaves garbage inside the preallocated region past
                // `durable_bytes` (no size rewind — it would chop the preallocation, and after
                // a failed fsync the page-cache state is unknowable anyway). Poison the backing:
                // the group's records may back already-applied deltas, so nothing may ever
                // append past this point until restart recovery truncates the torn tail.
                state.poisoned = Some(format!("group flush failed ({err})"));
                Err(EngineError::Durability(format!(
                    "failed to flush WAL segment group {}: {err}",
                    self.core.segment_path.display()
                )))
            }
        };
        drop(state);
        self.core.cv.notify_all();
        outcome
    }

    /// The flusher died between begin and commit: the file may hold a partial write. Fail closed
    /// and wake anyone waiting for the IO to drain.
    fn abandon(self) {
        let mut state = self.core.lock_state();
        state.io_in_flight = false;
        state.poisoned = Some("group flush abandoned mid-IO".to_string());
        drop(state);
        self.core.cv.notify_all();
    }
}

/// The result of tolerantly reading a live (append-only) WAL segment at recovery time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalSegmentRecovery {
    /// Every record whose bytes were fully written and CRC-valid, in log order.
    pub records: Vec<WalRecord>,
    /// Byte length of the valid prefix (magic + `records`); the file is truncated to this on
    /// recovery install.
    pub valid_bytes: u64,
    /// Bytes discarded beyond the valid prefix (a torn tail from a crash mid-append). Zero on a
    /// clean segment.
    pub discarded_torn_bytes: u64,
}

impl WalSegmentRecovery {
    /// A missing / empty / never-created segment: a fresh durable database.
    pub fn empty() -> Self {
        Self {
            records: Vec::new(),
            valid_bytes: 0,
            discarded_torn_bytes: 0,
        }
    }
}

/// Group-commit accounting for a [`WalBuffer`].
///
/// Each `flush_all` that performs a real fsync batches **all** currently-unflushed records into a
/// single segment write / single fsync — that batch is one *group*. The serialized commit path
/// flushes one record at a time (size-1 groups); the engine's concurrent DML path elects a
/// designated flusher whose [`WalGroupFlushJob`] IO runs lock-free, so committers that append
/// while a group's fsync is in flight coalesce into the NEXT group (`mean_group_size` grows with
/// write concurrency). These counters expose that batching ratio
/// (`durable_records / flush_groups`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WalGroupCommitStats {
    /// Number of `flush_all` calls that performed a real durable fsync (one group each).
    pub flush_groups: u64,
    /// Total records made durable across all groups.
    pub durable_records: u64,
    /// Largest single group (records fsynced by one `flush_all`).
    pub max_group_size: usize,
}

impl WalGroupCommitStats {
    /// Mean records-per-fsync (the group-commit amortization ratio). `0.0` before any flush.
    pub fn mean_group_size(&self) -> f64 {
        if self.flush_groups == 0 {
            0.0
        } else {
            self.durable_records as f64 / self.flush_groups as f64
        }
    }
}

#[derive(Debug, Default)]
pub struct WalBuffer {
    records: Vec<WalRecord>,
    /// Durable watermark for the IN-MEMORY mode only (`durable: None`). The durable mode's
    /// watermark lives in [`WalDurableState::flushed_records`] so a group flush can advance it
    /// under the core's own lock, without the buffer's outer lock (the engine commit_mutex).
    flushed_memory: usize,
    fail_next_flush: bool,
    durable: Option<Arc<WalDurableCore>>,
    /// E1 step 1: the optional FUA fence-pool durability backend (default OFF). When present it
    /// REPLACES `durable`: `flush_all` / `begin_group_flush` publish frames into a pipelined
    /// fence pool whose contiguous durable cut is the record watermark, so MULTIPLE groups can
    /// be durable in flight at once (unlike the serial single-slot `durable` core). Gated behind
    /// `#[cfg(unix)]` because the underlying `FuaFrameLog` is a unix `O_DIRECT|O_DSYNC` construct.
    #[cfg(unix)]
    fua: Option<Arc<fua::FuaWalBackend>>,
}

/// Which durability backend a durable [`WalBuffer`] uses. Default is the existing single-slot
/// serial `write_all` + `fdatasync` path; the FUA fence pool is opt-in (E1 step 1) and, once the
/// engine is wired to it in step 2, env-gated via [`WalDurability::from_env`] (default OFF).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum WalDurability {
    /// The existing behavior: one `write_all` + `fdatasync` per group, at most one IO in flight.
    #[default]
    SerialFdatasync,
    /// The FUA fence-pool backend: `lanes` concurrent FUA-write fence lanes over pre-written,
    /// epoch-stamped frame-log segments of `segment_bytes` data capacity each; the contiguous
    /// durable cut advances the record watermark, allowing multiple groups durable in flight.
    FuaFencePool { lanes: usize, segment_bytes: usize },
}

impl WalDurability {
    /// Suggested fence-pool depth when the env leaves it unset (measured fast-mode flip is
    /// qd 16-48 on the reference NVMe; 32 is a safe midpoint — re-probe per device at bring-up).
    pub const DEFAULT_FUA_LANES: usize = 32;
    /// Suggested per-segment data capacity when the env leaves it unset (64MiB, matching the
    /// serial preallocation chunk).
    pub const DEFAULT_FUA_SEGMENT_BYTES: usize = 64 * 1024 * 1024;

    /// Read the durability backend from the environment (the step-2 engine flag surface).
    /// `GPU_DB_WAL_DURABILITY=fua` selects the FUA fence pool (with `GPU_DB_WAL_FUA_LANES` and
    /// `GPU_DB_WAL_FUA_SEGMENT_BYTES` overrides); anything else — including unset — is the serial
    /// default. Kept here so the engine's later wiring reads ONE authority for the gate.
    pub fn from_env() -> Self {
        let selected = std::env::var("GPU_DB_WAL_DURABILITY")
            .map(|v| v.eq_ignore_ascii_case("fua"))
            .unwrap_or(false);
        if !selected {
            return Self::SerialFdatasync;
        }
        let lanes = std::env::var("GPU_DB_WAL_FUA_LANES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(Self::DEFAULT_FUA_LANES);
        let segment_bytes = std::env::var("GPU_DB_WAL_FUA_SEGMENT_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(Self::DEFAULT_FUA_SEGMENT_BYTES);
        Self::FuaFencePool {
            lanes,
            segment_bytes,
        }
    }
}

impl WalBuffer {
    /// An in-memory WAL buffer with no durable backing (the default — `flush_all` only advances the
    /// in-memory durable watermark). Used by ephemeral engines and the bulk of the test suite.
    pub fn new() -> Self {
        Self::default()
    }

    /// A WAL buffer backed by a real, fsync-durable segment file at `segment_path` — a FRESH
    /// durable database (an existing file at that path is clobbered on the first flush).
    ///
    /// `flush_all` appends the unflushed record tail to that file and fsyncs it before advancing
    /// the durable watermark. Recovery reads the segment back with [`recover_wal_segment`].
    pub fn with_durable_segment(segment_path: impl Into<PathBuf>) -> Self {
        Self {
            durable: Some(Arc::new(WalDurableCore::fresh(segment_path.into()))),
            ..Self::default()
        }
    }

    /// A WAL buffer backed by the FUA fence-pool durability backend (E1 step 1) — a FRESH durable
    /// database. `flush_all` / `begin_group_flush` publish each group's encoded record run as ONE
    /// frame into a pipelined fence pool; the contiguous durable cut is the record watermark, so
    /// multiple groups can be durable in flight at once. Segment files are created next to
    /// `segment_path` (named `<segment_path>.fua.<segment_id>`); recovery reads them back with
    /// [`recover_fua_wal_records`]. `lanes` is the fence-pool depth (see
    /// [`WalDurability::DEFAULT_FUA_LANES`]); `segment_bytes` is the per-segment data capacity.
    ///
    /// The frame payload bytes are byte-identical to what the serial backend's group flush would
    /// have written for the same records (the [`encode_record_into`] run), so the two backends
    /// recover to the same logical `WalRecord`s.
    #[cfg(unix)]
    pub fn with_fua_durable_segment(
        segment_path: impl Into<PathBuf>,
        lanes: usize,
        segment_bytes: usize,
    ) -> Result<Self, EngineError> {
        Ok(Self {
            fua: Some(Arc::new(fua::FuaWalBackend::create(
                segment_path.into(),
                lanes,
                segment_bytes,
            )?)),
            ..Self::default()
        })
    }

    /// A WAL buffer installed over a segment just read back by [`recover_wal_segment`], seeded
    /// with the full recovered record history and positioned to keep APPENDING to the same file.
    ///
    /// `records` is the buffer's complete logical history; its first `records.len() -
    /// recovery.records.len()` entries are the checkpoint-covered prefix that lives in an external
    /// checkpoint segment (empty for a plain single-segment recovery), and its tail must be
    /// exactly `recovery.records`. The segment file is truncated to `recovery.valid_bytes`
    /// (discarding any torn tail durably) and the tail-offset sidecar is re-recorded.
    pub fn with_recovered_durable_segment(
        segment_path: impl Into<PathBuf>,
        records: Vec<WalRecord>,
        recovery: &WalSegmentRecovery,
    ) -> Result<Self, EngineError> {
        let segment_path = segment_path.into();
        debug_assert!(records.len() >= recovery.records.len());
        debug_assert!(records.ends_with(&recovery.records));
        let segment_base_records = records.len() - recovery.records.len();
        let core = WalDurableCore::fresh(segment_path);
        {
            let mut state = core.lock_state();
            if recovery.valid_bytes > 0 {
                let file = fs::OpenOptions::new()
                    .write(true)
                    .open(&core.segment_path)
                    .map_err(|err| {
                        EngineError::Durability(format!(
                            "failed to open WAL segment for positional IO {}: {err}",
                            core.segment_path.display()
                        ))
                    })?;
                // Durably drop the torn tail (if any) so appends resume at the valid boundary,
                // then re-establish the zero-filled preallocation window (W4a) the truncate
                // chopped — recovery is the one-time place to pay it.
                let prealloc_to = recovery
                    .valid_bytes
                    .max(wal_prealloc_chunk_bytes())
                    .max(recovery.valid_bytes + wal_prealloc_chunk_bytes() / 2);
                file.set_len(recovery.valid_bytes)
                    .and_then(|_| file.sync_all())
                    .and_then(|_| zero_fill_extend(&file, recovery.valid_bytes, prealloc_to))
                    .map_err(|err| {
                        EngineError::Durability(format!(
                            "failed to truncate/preallocate WAL segment tail {}: {err}",
                            core.segment_path.display()
                        ))
                    })?;
                state.prealloc_bytes = prealloc_to;
                state.file = Some(Arc::new(file));
                state.durable_bytes = recovery.valid_bytes;
                core.record_tail_offset(&mut state)?;
            }
            state.segment_base_records = segment_base_records;
            state.flushed_records = records.len();
        }
        Ok(Self {
            records,
            flushed_memory: 0,
            fail_next_flush: false,
            durable: Some(Arc::new(core)),
            #[cfg(unix)]
            fua: None,
        })
    }

    /// REOPEN a FUA-durable database (E1 step 3) whose retained `<segment_path>.fua.*` segments were
    /// already scan-recovered into `records` by [`recover_fua_wal_records`] (the caller replayed
    /// them). The buffer is seeded with the full recovered history and positioned to keep APPENDING
    /// in a FRESH segment above the highest existing id — the old segments are retained, never
    /// appended into, so a crash can only tear the newest segment's tail. `records.len()` is the
    /// durable/published watermark: `flushed_count()` reports it immediately and the first new
    /// frame's `first_seq` continues the contiguous log (see [`fua::FuaWalBackend::reopen`]).
    #[cfg(unix)]
    pub fn with_recovered_fua_durable_segment(
        segment_path: impl Into<PathBuf>,
        records: Vec<WalRecord>,
        lanes: usize,
        segment_bytes: usize,
    ) -> Result<Self, EngineError> {
        let recovered = records.len();
        let backend =
            fua::FuaWalBackend::reopen(segment_path.into(), lanes, segment_bytes, recovered)?;
        Ok(Self {
            records,
            flushed_memory: 0,
            fail_next_flush: false,
            durable: None,
            fua: Some(Arc::new(backend)),
        })
    }

    /// FUA-backend PACING signal (E1 step 3): free fence lanes in the active segment's pool, or
    /// `None` for any non-FUA backend. The engine's concurrent-durability wait uses this at the
    /// engine seam — a committer begins its own group flush ONLY while a lane is free, so backlog
    /// forms a LARGER next group behind the busy lanes instead of collapsing the pool to tiny
    /// per-arrival frames (the population-share anti-convoy law, applied above the WAL).
    pub fn fua_free_fence_slots(&self) -> Option<usize> {
        #[cfg(unix)]
        if let Some(fua) = self.fua.as_ref() {
            return Some(fua.free_fence_slots());
        }
        None
    }

    /// The durable segment path, if this buffer is backed by one. For the FUA backend this is the
    /// base path the per-segment files (`<path>.fua.<segment_id>`) sit beside.
    pub fn durable_segment_path(&self) -> Option<&Path> {
        #[cfg(unix)]
        if let Some(fua) = self.fua.as_ref() {
            return Some(fua.base_path());
        }
        self.durable
            .as_ref()
            .map(|core| core.segment_path.as_path())
    }

    /// Whether `flush_all` performs a real fsync (vs. in-memory watermark advance only).
    pub fn is_durable(&self) -> bool {
        #[cfg(unix)]
        if self.fua.is_some() {
            return true;
        }
        self.durable.is_some()
    }

    /// Whether this buffer's durability backend supports MULTIPLE concurrent group flushes in
    /// flight (E1 step 2). True only for the FUA fence-pool backend, where `begin_group_flush`
    /// snapshots and advances the `published` cursor under the caller's outer lock (so frames stay
    /// totally ordered) while the returned job's fence-pool durable-cut wait runs off-lock and
    /// overlaps every other FUA job. The serial `write_all` + `fdatasync` backend requires the
    /// caller to elect a SINGLE flusher (its `io_in_flight` slot admits at most one IO), so it
    /// returns false — the engine keeps the flusher-election on that path and skips it on this one.
    pub fn durability_is_concurrent(&self) -> bool {
        #[cfg(unix)]
        if self.fua.is_some() {
            return true;
        }
        false
    }

    pub fn append(&mut self, rec: WalRecord) {
        self.records.push(rec);
    }

    /// Seed the buffer with records already known to be durable (e.g. recovered from a segment),
    /// marking them as the flushed prefix WITHOUT performing any I/O. Only meaningful on an
    /// in-memory buffer (a durable recovery installs via
    /// [`WalBuffer::with_recovered_durable_segment`], which also positions the append handle).
    /// Must be called on an otherwise-empty buffer.
    pub fn reinstate_durable_records(&mut self, records: Vec<WalRecord>) {
        debug_assert!(
            self.records.is_empty(),
            "reinstate_durable_records on a non-empty WAL buffer"
        );
        debug_assert!(
            self.durable.is_none(),
            "durable buffers are recovered via with_recovered_durable_segment"
        );
        self.flushed_memory = records.len();
        self.records = records;
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn truncate(&mut self, len: usize) {
        if len >= self.records.len() {
            return;
        }
        // Commit-path rollback only ever truncates the just-appended UNFLUSHED tail (the callers
        // capture `wal.len()` before appending, and any group flush that could cover the region
        // being cut would have had to `begin` inside this holder's outer critical section — it
        // cannot have). If a future caller cuts below the flushed watermark, physically rewind
        // the segment too so the file never replays records the buffer disowned.
        #[cfg(unix)]
        if let Some(fua) = self.fua.as_ref() {
            // The FUA backend advances its `published` cursor only inside `begin_group_flush`,
            // which runs under this same outer lock; so a commit-path rollback cutting the tail
            // it just appended is always ABOVE `published` and needs no frame rewind. A cut BELOW
            // `published` would disown records already handed to (possibly already-durable) frames
            // — the frame log cannot un-publish, so fail closed (defensive; never hit on the
            // commit path) and still drop the logical tail so the buffer's history stays coherent.
            if len < fua.published_records() {
                fua.set_poison("WAL truncate below the published FUA frame watermark");
            }
            self.records.truncate(len);
            return;
        }
        match self.durable.as_ref() {
            None => {
                self.records.truncate(len);
                if self.flushed_memory > self.records.len() {
                    self.flushed_memory = self.records.len();
                }
            }
            Some(core) => {
                let mut state = core.lock_state();
                if len < state.flushed_records {
                    // Never reached by the commit-path rollbacks; wait out any in-flight group
                    // IO before touching the file (defensive path only).
                    while state.io_in_flight {
                        state = core
                            .cv
                            .wait(state)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                    let disowned_bytes: u64 = self.records
                        [len.max(state.segment_base_records)..state.flushed_records]
                        .iter()
                        .map(encoded_record_len)
                        .sum();
                    if disowned_bytes > 0 {
                        if let Some(file) = state.file.clone() {
                            let target = state.durable_bytes.saturating_sub(disowned_bytes);
                            if let Err(err) = file.set_len(target).and_then(|_| file.sync_all()) {
                                state.poisoned =
                                    Some(format!("durable-prefix truncate rewind failed ({err})"));
                            } else {
                                state.durable_bytes = target;
                                // W4a: the set_len chopped the zero fill; account for it so the
                                // next write re-extends before writing.
                                state.prealloc_bytes = target;
                            }
                        }
                    }
                    state.flushed_records = len;
                }
                drop(state);
                self.records.truncate(len);
            }
        }
    }

    /// Make every appended record durable.
    ///
    /// In-memory mode: advances the durable watermark to the full record count.
    ///
    /// Durable mode: this is the **commit fsync** and the group-commit point. It serializes ONLY
    /// the currently-unflushed record tail, appends it to the open segment with a single
    /// `write_all`, and `fdatasync`s it — O(group) per flush, so total WAL work over N commits is
    /// O(N), not the O(N²) of a rewrite-per-commit scheme. The parent directory is fsynced once,
    /// when the segment file is first created. Only after the fsync succeeds is the in-memory
    /// durable watermark advanced — so a caller that gates visibility on `flushed_count` can never
    /// publish a record whose WAL bytes are not yet on disk (the WAL-before-visibility
    /// invariant). On any I/O error the watermark is left untouched and the error is returned, so
    /// the caller can roll back the in-flight commit before it becomes visible; a failure that
    /// leaves the on-disk tail state unknowable poisons the backing (fail-closed until restart
    /// recovery truncates the torn tail at [`recover_wal_segment`] time).
    pub fn flush_all(&mut self) -> Result<(), EngineError> {
        // FUA backend: the inline serial-path flush is just a group flush that also WAITS for the
        // durable cut. Delegating keeps one publish/wait path (and one `fail_next_flush`
        // consumption, handled by `begin_group_flush`).
        #[cfg(unix)]
        if self.fua.is_some() {
            let target = self.records.len();
            match self.begin_group_flush()? {
                WalGroupFlushBegin::Clean { .. } => {}
                WalGroupFlushBegin::Job(job) => {
                    job.commit()?;
                }
            }
            // Ensure durability up to the full record count: a `Clean` return means nothing NEW to
            // publish, but concurrently-published frames may not have reached the durable cut yet.
            if let Some(fua) = self.fua.as_ref() {
                fua.wait_durable(target)?;
            }
            return Ok(());
        }
        if self.fail_next_flush {
            self.fail_next_flush = false;
            return Err(EngineError::Durability(
                "simulated wal flush failure".to_string(),
            ));
        }
        let target = self.records.len();
        let Some(core) = self.durable.clone() else {
            self.flushed_memory = target;
            return Ok(());
        };
        // Inline (serialized-path) flush: wait out any in-flight group IO, then write + fsync
        // while holding the core lock (the caller already holds the outer lock and expects a
        // synchronous durable-or-error answer with clean rollback semantics).
        let mut state = core.lock_state_idle();
        if let Some(reason) = state.poisoned.clone() {
            return Err(core.poisoned_error(&reason));
        }
        let group_size = target.saturating_sub(state.flushed_records);
        if group_size == 0 {
            return Ok(());
        }
        core.ensure_created(&mut state)?;
        let mut tail = Vec::new();
        for record in &self.records[state.flushed_records..target] {
            encode_record_into(&mut tail, record)?;
        }
        let file = state.file.clone().expect("write handle present");
        // W4a: keep the write inside zero-filled extents (extend by whole chunks, rare) and
        // write POSITIONALLY at the logical tail — the file size never changes on the hot
        // path, so `sync_data` skips the filesystem's size-change journaling.
        let write_end = state.durable_bytes + tail.len() as u64;
        if write_end > state.prealloc_bytes {
            let new_prealloc = write_end
                .max(state.prealloc_bytes + wal_prealloc_chunk_bytes())
                .max(wal_prealloc_chunk_bytes());
            if let Err(err) = zero_fill_extend(&file, state.prealloc_bytes, new_prealloc) {
                state.poisoned = Some(format!("preallocation extend failed ({err})"));
                return Err(EngineError::Durability(format!(
                    "failed to extend WAL segment preallocation {}: {err}",
                    core.segment_path.display()
                )));
            }
            state.prealloc_bytes = new_prealloc;
        }
        {
            use std::os::unix::fs::FileExt;
            if let Err(err) = file.write_all_at(&tail, state.durable_bytes) {
                // A partial positional write leaves garbage INSIDE the preallocated region past
                // `durable_bytes`; the next successful write overwrites it and recovery's
                // checksum walk truncates it — no size rewind needed (or wanted: it would chop
                // the preallocation).
                state.poisoned = Some(format!("append write failed ({err})"));
                return Err(EngineError::Durability(format!(
                    "failed to append WAL segment {}: {err}",
                    core.segment_path.display()
                )));
            }
        }
        if let Err(err) = file.sync_data() {
            // After a failed fsync the page-cache state is unknowable (fsyncgate): the kernel may
            // have marked dirty pages clean without persisting them, so neither a retry nor a
            // rewind can be trusted. Fail closed; restart recovery truncates the torn tail.
            state.poisoned = Some(format!("fsync failed ({err})"));
            return Err(EngineError::Durability(format!(
                "failed to fsync WAL segment {}: {err}",
                core.segment_path.display()
            )));
        }
        state.durable_bytes += tail.len() as u64;
        // Watermark advances only after the fsync has succeeded.
        WalDurableCore::note_group(&mut state, group_size, target);
        Ok(())
    }

    /// Begin a GROUP flush (the concurrent commit path's designated-flusher protocol): snapshot
    /// the unflushed record tail and hand back a [`WalGroupFlushJob`] whose `write_all` + fsync
    /// run with NO lock held — the caller drops the buffer's outer lock (the engine commit_mutex)
    /// before [`WalGroupFlushJob::commit`], so other committers keep appending (forming the next
    /// group) while this group's disk IO is in flight. Returns
    /// [`WalGroupFlushBegin::Clean`] when everything appended is already durable.
    ///
    /// In-memory mode: advances the watermark (no IO exists to defer) and reports `Clean`.
    ///
    /// The serial backend requires the caller to serialize group flushes (at most one outstanding
    /// job — the engine's flusher-election does this); the job's `io_in_flight` mark excludes the
    /// INLINE [`WalBuffer::flush_all`] path in the meantime. The FUA backend has NO such single-slot
    /// exclusion: it snapshots + advances its `published` cursor under this outer lock (so frames
    /// stay totally ordered) and the returned job's fence-pool wait runs concurrently with any
    /// other FUA job — multiple groups may be durable in flight at once.
    pub fn begin_group_flush(&mut self) -> Result<WalGroupFlushBegin, EngineError> {
        if self.fail_next_flush {
            self.fail_next_flush = false;
            return Err(EngineError::Durability(
                "simulated wal flush failure".to_string(),
            ));
        }
        let target = self.records.len();
        // FUA backend: snapshot the not-yet-published tail as ONE frame payload (its bytes are the
        // exact serial-encoded record run), advance the `published` cursor under this outer lock
        // so frame order is total, and hand back a job that publishes + waits for the durable cut.
        #[cfg(unix)]
        if let Some(fua) = self.fua.as_ref() {
            if let Some(reason) = fua.poison_reason() {
                return Err(fua.poison_error(&reason));
            }
            let published = fua.published_records();
            let group_size = target.saturating_sub(published);
            if group_size == 0 {
                return Ok(WalGroupFlushBegin::Clean {
                    flushed_records: fua.durable_records(),
                });
            }
            let seq_count = u32::try_from(group_size).map_err(|_| {
                EngineError::Durability(
                    "FUA WAL group exceeds u32 records; split the commit batch".to_string(),
                )
            })?;
            let mut payload = Vec::new();
            for record in &self.records[published..target] {
                encode_record_into(&mut payload, record)?;
            }
            // Allocate the ticket and advance the published cursor together under this outer lock
            // so tickets and `first_seq` ranges are assigned in one total order.
            let ticket = fua.next_ticket();
            fua.set_published(target);
            return Ok(WalGroupFlushBegin::Job(WalGroupFlushJob {
                kind: Some(WalGroupFlushJobKind::Fua(fua::FuaFlushJob::new(
                    Arc::clone(fua),
                    ticket,
                    payload,
                    published as u64,
                    seq_count,
                    target,
                ))),
            }));
        }
        let Some(core) = self.durable.as_ref() else {
            self.flushed_memory = target;
            return Ok(WalGroupFlushBegin::Clean {
                flushed_records: target,
            });
        };
        let mut state = core.lock_state();
        debug_assert!(
            !state.io_in_flight,
            "at most one outstanding group flush job (the caller elects a single flusher)"
        );
        if let Some(reason) = state.poisoned.clone() {
            return Err(core.poisoned_error(&reason));
        }
        let group_size = target.saturating_sub(state.flushed_records);
        if group_size == 0 {
            return Ok(WalGroupFlushBegin::Clean {
                flushed_records: state.flushed_records,
            });
        }
        core.ensure_created(&mut state)?;
        let mut tail = Vec::new();
        for record in &self.records[state.flushed_records..target] {
            encode_record_into(&mut tail, record)?;
        }
        let file = state.file.clone().expect("write handle present");
        state.io_in_flight = true;
        Ok(WalGroupFlushBegin::Job(WalGroupFlushJob {
            kind: Some(WalGroupFlushJobKind::Serial(SerialFlushJob {
                core: Arc::clone(core),
                file,
                offset: state.durable_bytes,
                prealloc_end: state.prealloc_bytes,
                bytes: tail,
                target_records: target,
                group_size,
            })),
        }))
    }

    /// Valid, fsynced byte length of the live durable segment (0 for an in-memory buffer or
    /// before the first durable flush). The size-bound input for checkpoint/rotation policy. The
    /// FUA backend self-rolls its own segments, so it reports 0 (the outer rotation policy does not
    /// drive it — step 2 exposes FUA-native size/retention introspection).
    pub fn durable_segment_bytes(&self) -> u64 {
        #[cfg(unix)]
        if self.fua.is_some() {
            return 0;
        }
        self.durable
            .as_ref()
            .map_or(0, |core| core.lock_state().durable_bytes)
    }

    /// How many of the buffer's records are durable in an external checkpoint segment rather than
    /// the live segment file (see [`WalBuffer::truncate_durable_segment_prefix`]). Always 0 for the
    /// FUA backend (no external checkpoint segment in step 1).
    pub fn durable_segment_base_records(&self) -> usize {
        #[cfg(unix)]
        if self.fua.is_some() {
            return 0;
        }
        self.durable
            .as_ref()
            .map_or(0, |core| core.lock_state().segment_base_records)
    }

    /// Discard the live segment's prefix up to `base` (a record index into this buffer) — the
    /// checkpoint-truncation half of D2. The caller must FIRST have made records `[0, base)`
    /// durable elsewhere (a checkpoint segment + control file); this rewrites the live segment to
    /// contain only `[base, flushed)` via an atomic temp-write + rename + parent-dir fsync, then
    /// reopens a positional write handle on the rewritten file and re-establishes the W4a
    /// preallocation frontier. The buffer's in-memory records and all
    /// logical counters are unchanged — only the FILE is trimmed, so a long-lived database's live
    /// segment stays bounded by the checkpoint cadence instead of growing forever.
    pub fn truncate_durable_segment_prefix(&mut self, base: usize) -> Result<(), EngineError> {
        // The FUA backend's segment lifecycle (roll + recycle + retention) is step 2; it has no
        // external checkpoint segment to trim against in step 1.
        #[cfg(unix)]
        if self.fua.is_some() {
            let _ = base;
            return Err(EngineError::Durability(
                "prefix truncation is not supported by the FUA WAL backend in E1 step 1"
                    .to_string(),
            ));
        }
        let core = self.durable.as_ref().ok_or_else(|| {
            EngineError::Durability(
                "cannot truncate the segment prefix of an in-memory WAL buffer".to_string(),
            )
        })?;
        // Wait out any in-flight group IO: the rewrite below replaces the file wholesale.
        let mut state = core.lock_state_idle();
        let flushed = state.flushed_records;
        if base > flushed {
            return Err(EngineError::Durability(format!(
                "WAL segment prefix truncation boundary {base} exceeds the durable watermark \
                 {flushed}"
            )));
        }
        if base < state.segment_base_records {
            return Err(EngineError::Durability(format!(
                "WAL segment prefix truncation boundary {base} precedes the existing checkpoint \
                 base {}",
                state.segment_base_records
            )));
        }
        // Close the old handle first: the rename below unlinks the inode it points at.
        // W1b audit fix 3: any failure past this point leaves `file = None`, and the next
        // flush's ensure_created would CLOBBER the live segment with fresh-database semantics —
        // poison the backing instead so the half-rotated state is fail-closed until restart
        // recovery (which reads the on-disk files, not this handle).
        state.file = None;
        let retained = &self.records[base..flushed];
        if let Err(err) = write_wal_segment(&core.segment_path, retained) {
            state.poisoned = Some(format!("prefix-truncation rewrite failed ({err})"));
            return Err(err);
        }
        if let Err(err) = sync_segment_parent_dir(&core.segment_path) {
            state.poisoned = Some(format!("prefix-truncation dir fsync failed ({err})"));
            return Err(err);
        }
        // AUDIT 9d6e9f96 BLOCKER: the reopen MUST be a plain write handle — on Linux, pwrite on
        // an O_APPEND fd IGNORES the offset and appends at EOF, so every positional record write
        // (and worse, a frontier-crossing zero_fill_extend) after a rotation would silently
        // append past the logical tail: acknowledged records stranded behind a 64MB zero hole
        // that recovery either rejects loudly (clean shutdown) or truncates silently (crash).
        let file = match fs::OpenOptions::new().write(true).open(&core.segment_path) {
            Ok(file) => file,
            Err(err) => {
                state.poisoned = Some(format!("post-truncation reopen failed ({err})"));
                return Err(EngineError::Durability(format!(
                    "failed to reopen WAL segment after prefix truncation {}: {err}",
                    core.segment_path.display()
                )));
            }
        };
        let durable_bytes =
            WAL_SEGMENT_MAGIC.len() as u64 + retained.iter().map(encoded_record_len).sum::<u64>();
        // AUDIT 9d6e9f96 BLOCKER (part 2): the rewrite produced a COMPACT file — the old
        // `prealloc_bytes` frontier is stale and must be re-established here (rotation is
        // already a heavy, rare operation; paying the zero fill now keeps every subsequent
        // group fdatasync on the fast no-size-change path).
        let prealloc_to = durable_bytes.max(wal_prealloc_chunk_bytes());
        if let Err(err) = zero_fill_extend(&file, durable_bytes, prealloc_to) {
            state.poisoned = Some(format!("post-truncation preallocation failed ({err})"));
            return Err(EngineError::Durability(format!(
                "failed to re-preallocate WAL segment after prefix truncation {}: {err}",
                core.segment_path.display()
            )));
        }
        state.prealloc_bytes = prealloc_to;
        state.file = Some(Arc::new(file));
        state.durable_bytes = durable_bytes;
        state.segment_base_records = base;
        state.poisoned = None;
        core.record_tail_offset(&mut state)?;
        Ok(())
    }

    pub fn flushed_count(&self) -> usize {
        // FUA backend: the durable watermark is the contiguous durable cut of the fence pool.
        #[cfg(unix)]
        if let Some(fua) = self.fua.as_ref() {
            return fua.durable_records();
        }
        match self.durable.as_ref() {
            Some(core) => core.lock_state().flushed_records,
            None => self.flushed_memory,
        }
    }

    pub fn flushed_records(&self) -> &[WalRecord] {
        &self.records[..self.flushed_count()]
    }

    pub fn unflushed_count(&self) -> usize {
        self.records.len().saturating_sub(self.flushed_count())
    }

    /// Group-commit accounting (fsync groups, durable records, largest group). See
    /// [`WalGroupCommitStats`].
    pub fn group_commit_stats(&self) -> WalGroupCommitStats {
        #[cfg(unix)]
        if let Some(fua) = self.fua.as_ref() {
            return fua.group_commit_stats();
        }
        self.durable
            .as_ref()
            .map_or(WalGroupCommitStats::default(), |core| {
                core.lock_state().stats
            })
    }

    pub fn checkpoint_meta(&self) -> WalCheckpointMeta {
        let flushed = self.flushed_count();
        WalCheckpointMeta {
            durable_record_count: flushed,
            last_durable_txn_id: self.records[..flushed].last().map(|record| record.txn_id),
        }
    }

    pub fn fail_next_flush(&mut self) {
        self.fail_next_flush = true;
    }
}

/// Fsync the parent directory of `segment_path` so a freshly-`rename`d segment file's directory
/// entry is durable across a crash (POSIX: an fsync of the file does not guarantee the containing
/// directory entry is persisted). A best-effort no-op on platforms / filesystems that refuse to
/// open a directory for fsync is intentionally NOT done — a hard error here means the existence of
/// the just-written WAL could be lost on crash, which would violate durability, so it propagates.
/// W1b audit fix 1: crash-durability for the checkpoint/control RENAMES — a rename is not
/// durable until the parent directory is fsynced; the rotation must do this BEFORE truncating
/// the live segment, or a strict-POSIX crash can lose the control file (and with it the entire
/// checkpointed prefix, silently) after the truncation survived.
pub fn sync_wal_parent_dir(path: &Path) -> Result<(), EngineError> {
    sync_segment_parent_dir(path)
}

fn sync_segment_parent_dir(segment_path: &Path) -> Result<(), EngineError> {
    let parent = segment_path.parent().filter(|p| !p.as_os_str().is_empty());
    let Some(parent) = parent else {
        // No parent component (e.g. a bare relative file name) — the current working directory is
        // the container; there is nothing portable to fsync, so treat as durable.
        return Ok(());
    };
    let dir = File::open(parent).map_err(|err| {
        EngineError::Durability(format!(
            "failed to open WAL segment directory for fsync {}: {err}",
            parent.display()
        ))
    })?;
    dir.sync_all().map_err(|err| {
        EngineError::Durability(format!(
            "failed to fsync WAL segment directory {}: {err}",
            parent.display()
        ))
    })
}

pub fn write_wal_segment(path: impl AsRef<Path>, records: &[WalRecord]) -> Result<(), EngineError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL segment directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let tmp_path = temporary_segment_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL segment {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(WAL_SEGMENT_MAGIC).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL segment header {}: {err}",
                tmp_path.display()
            ))
        })?;
        for record in records {
            write_record(&mut file, record)?;
        }
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL segment {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL segment {}: {err}",
            path.display()
        ))
    })
}

pub fn read_wal_segment(path: impl AsRef<Path>) -> Result<Vec<WalRecord>, EngineError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to open WAL segment {}: {err}",
            path.display()
        ))
    })?;
    let mut magic = [0_u8; WAL_SEGMENT_MAGIC.len()];
    file.read_exact(&mut magic).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL segment header {}: {err}",
            path.display()
        ))
    })?;
    if &magic != WAL_SEGMENT_MAGIC {
        return Err(EngineError::Durability(format!(
            "invalid WAL segment header {}",
            path.display()
        )));
    }

    let mut records = Vec::new();
    loop {
        let mut header = [0_u8; WAL_RECORD_HEADER_LEN];
        match file.read(&mut header[..1]) {
            Ok(0) => break,
            Ok(1) => {
                file.read_exact(&mut header[1..]).map_err(|err| {
                    EngineError::Durability(format!(
                        "failed to read WAL segment record header {}: {err}",
                        path.display()
                    ))
                })?;
                let txn_id = u64::from_le_bytes(header[0..8].try_into().expect("txn id bytes"));
                let payload_len =
                    u64::from_le_bytes(header[8..16].try_into().expect("payload len bytes"));
                let expected_checksum =
                    u64::from_le_bytes(header[16..24].try_into().expect("checksum bytes"));
                // W4a: an ALL-ZERO header is the preallocated zero tail — clean end-of-log. It
                // can never be a real record: the FNV checksum of a zero header is nonzero
                // (test-asserted), so (0, 0, 0) is unrepresentable by any valid record.
                if txn_id == 0 && payload_len == 0 && expected_checksum == 0 {
                    break;
                }
                let payload_len = usize::try_from(payload_len).map_err(|_| {
                    EngineError::Durability(format!(
                        "WAL segment {} record payload length is too large",
                        path.display()
                    ))
                })?;
                let mut payload = vec![0_u8; payload_len];
                file.read_exact(&mut payload).map_err(|err| {
                    EngineError::Durability(format!(
                        "failed to read WAL segment payload {}: {err}",
                        path.display()
                    ))
                })?;
                let actual_checksum = wal_record_checksum(txn_id, payload_len as u64, &payload);
                if actual_checksum != expected_checksum {
                    return Err(EngineError::Durability(format!(
                        "WAL segment {} record checksum mismatch for txn {}",
                        path.display(),
                        txn_id
                    )));
                }
                records.push(WalRecord {
                    txn_id,
                    payload: payload.into(),
                });
            }
            Ok(_) => unreachable!("one-byte read returned more than one byte"),
            Err(err) => {
                return Err(EngineError::Durability(format!(
                    "failed to read WAL segment record {}: {err}",
                    path.display()
                )));
            }
        }
    }
    Ok(records)
}

/// W1b — the AUTO-CHECKPOINT path convention: for a live segment `P`, the rolling checkpoint
/// control file is `P.control` and the checkpoint segment is `P.checkpoint`. The engine's
/// auto-rotation writes with these paths and the checkpoint-aware open detects `P.control` to
/// recover checkpoint-then-suffix; a database that never checkpointed has no control file and
/// opens exactly as before.
pub fn wal_checkpoint_control_path(segment_path: &Path) -> PathBuf {
    let file_name = segment_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    segment_path.with_file_name(format!("{file_name}.control"))
}

/// See [`wal_checkpoint_control_path`].
pub fn wal_checkpoint_segment_path(segment_path: &Path) -> PathBuf {
    let file_name = segment_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    segment_path.with_file_name(format!("{file_name}.checkpoint"))
}

const WAL_TAIL_MAGIC: &str = "GPUDBWALTAIL1";

/// Path of the durable tail-offset sidecar for a live segment (`<segment>.tail`).
pub fn wal_tail_offset_path(segment_path: &Path) -> PathBuf {
    let file_name = segment_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    segment_path.with_file_name(format!("{file_name}.tail"))
}

/// Atomically record the segment's known-durable byte length. Advisory (a LOWER bound for
/// recovery's torn-tail tolerance window), so it is not fsynced — losing it merely widens the
/// window; it never loses data.
fn write_wal_tail_offset(segment_path: &Path, durable_bytes: u64) -> Result<(), EngineError> {
    let path = wal_tail_offset_path(segment_path);
    let tmp_path = temporary_control_path(&path);
    let body = format!("{WAL_TAIL_MAGIC}\n{durable_bytes}\n");
    fs::write(&tmp_path, body).map_err(|err| {
        EngineError::Durability(format!(
            "failed to write WAL tail-offset file {}: {err}",
            tmp_path.display()
        ))
    })?;
    fs::rename(&tmp_path, &path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL tail-offset file {}: {err}",
            path.display()
        ))
    })
}

/// The recorded durable tail offset, or 0 when the sidecar is missing or unreadable (pure
/// torn-tail-tolerant recovery — the safe direction for an advisory lower bound).
fn read_wal_tail_offset(segment_path: &Path) -> u64 {
    let path = wal_tail_offset_path(segment_path);
    let Ok(body) = fs::read_to_string(&path) else {
        return 0;
    };
    let mut lines = body.lines();
    if lines.next() != Some(WAL_TAIL_MAGIC) {
        return 0;
    }
    lines.next().and_then(|raw| raw.parse().ok()).unwrap_or(0)
}

/// Tolerantly read a LIVE (append-only) segment at recovery time.
///
/// Unlike the strict [`read_wal_segment`] (for checkpoint/archive segments, which are written
/// atomically and must be intact end-to-end), a live segment can legitimately end in a torn
/// record: a crash between an append's `write_all` and its fsync acknowledgment. Such a record
/// was never acknowledged as committed, so it is safe — and required — to truncate it away.
///
/// The durable tail-offset sidecar bounds how far that tolerance reaches: an invalid region
/// starting AT or BEYOND the recorded offset is a torn tail (recovered records so far are
/// returned, with `valid_bytes` marking the truncation boundary); an invalid record starting
/// BELOW it means acknowledged-durable data is damaged (bit rot, external truncation), which
/// fails loudly with the same error the strict reader would raise.
pub fn recover_wal_segment(path: impl AsRef<Path>) -> Result<WalSegmentRecovery, EngineError> {
    let path = path.as_ref();
    let recorded_tail = read_wal_tail_offset(path);
    // Every non-loud return must reach at least the recorded durable tail: a segment that ends
    // CLEANLY short of it (external truncation, a lost/foreign file next to a live sidecar) has
    // lost acknowledged-durable records and must fail loudly, exactly like below-tail corruption.
    let ends_short = |valid_bytes: u64| {
        EngineError::Durability(format!(
            "WAL segment {} ends at byte {valid_bytes}, before the recorded durable tail offset \
             {recorded_tail}",
            path.display()
        ))
    };
    if !path.exists() {
        if recorded_tail > 0 {
            return Err(ends_short(0));
        }
        return Ok(WalSegmentRecovery::empty());
    }
    let bytes = fs::read(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL segment {}: {err}",
            path.display()
        ))
    })?;
    if bytes.len() < WAL_SEGMENT_MAGIC.len() {
        // A crash during segment creation (mid-magic write) leaves a short prefix of the magic;
        // treat it as a fresh database. Anything else short is a foreign file — refuse to clobber.
        if WAL_SEGMENT_MAGIC.starts_with(bytes.as_slice()) {
            if recorded_tail > 0 {
                return Err(ends_short(0));
            }
            return Ok(WalSegmentRecovery {
                records: Vec::new(),
                valid_bytes: 0,
                discarded_torn_bytes: bytes.len() as u64,
            });
        }
        return Err(EngineError::Durability(format!(
            "invalid WAL segment header {}",
            path.display()
        )));
    }
    if &bytes[..WAL_SEGMENT_MAGIC.len()] != WAL_SEGMENT_MAGIC {
        return Err(EngineError::Durability(format!(
            "invalid WAL segment header {}",
            path.display()
        )));
    }
    let mut records = Vec::new();
    let mut offset = WAL_SEGMENT_MAGIC.len();
    while offset < bytes.len() {
        let record_start = offset;
        let torn = |records: Vec<WalRecord>| {
            Ok(WalSegmentRecovery {
                records,
                valid_bytes: record_start as u64,
                discarded_torn_bytes: (bytes.len() - record_start) as u64,
            })
        };
        let below_recorded_tail = (record_start as u64) < recorded_tail;
        if bytes.len() - record_start < WAL_RECORD_HEADER_LEN {
            if below_recorded_tail {
                return Err(EngineError::Durability(format!(
                    "failed to read WAL segment record header {}: acknowledged-durable record is \
                     truncated at byte {record_start}",
                    path.display()
                )));
            }
            return torn(records);
        }
        let header = &bytes[record_start..record_start + WAL_RECORD_HEADER_LEN];
        let txn_id = u64::from_le_bytes(header[0..8].try_into().expect("txn id bytes"));
        let payload_len = u64::from_le_bytes(header[8..16].try_into().expect("payload len bytes"));
        let expected_checksum =
            u64::from_le_bytes(header[16..24].try_into().expect("checksum bytes"));
        // W4a: an all-zero header starts the preallocated zero tail. If the ENTIRE remainder is
        // zeros this is a CLEAN end-of-log (discarded_torn_bytes = 0); any non-zero byte in the
        // remainder is a genuine torn tail and takes the torn path below. An all-zero header can
        // never be a real record (the FNV checksum of a zero header is nonzero, test-asserted),
        // and acknowledged-durable records live below `recorded_tail`, which the zero region
        // never reaches (`below_recorded_tail` would fail loudly first).
        if txn_id == 0
            && payload_len == 0
            && expected_checksum == 0
            && !below_recorded_tail
            && bytes[record_start..].iter().all(|&b| b == 0)
        {
            return Ok(WalSegmentRecovery {
                records,
                valid_bytes: record_start as u64,
                discarded_torn_bytes: 0,
            });
        }
        let payload_start = record_start + WAL_RECORD_HEADER_LEN;
        let payload_end = usize::try_from(payload_len)
            .ok()
            .and_then(|len| payload_start.checked_add(len));
        let Some(payload_end) = payload_end.filter(|end| *end <= bytes.len()) else {
            if below_recorded_tail {
                return Err(EngineError::Durability(format!(
                    "failed to read WAL segment payload {}: acknowledged-durable record is \
                     truncated at byte {record_start}",
                    path.display()
                )));
            }
            return torn(records);
        };
        let payload = &bytes[payload_start..payload_end];
        let actual_checksum = wal_record_checksum(txn_id, payload_len, payload);
        if actual_checksum != expected_checksum {
            if below_recorded_tail {
                return Err(EngineError::Durability(format!(
                    "WAL segment {} record checksum mismatch for txn {}",
                    path.display(),
                    txn_id
                )));
            }
            return torn(records);
        }
        records.push(WalRecord {
            txn_id,
            payload: payload.to_vec().into(),
        });
        offset = payload_end;
    }
    if (offset as u64) < recorded_tail {
        return Err(ends_short(offset as u64));
    }
    Ok(WalSegmentRecovery {
        records,
        valid_bytes: offset as u64,
        discarded_torn_bytes: 0,
    })
}

pub fn write_wal_control_file(
    path: impl AsRef<Path>,
    control: &WalControlFile,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL control directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let tmp_path = temporary_control_path(path);
    let last_txn = control
        .checkpoint
        .last_durable_txn_id
        .map(|txn_id| txn_id.to_string())
        .unwrap_or_else(|| "none".to_string());
    let body = format!(
        "{WAL_CONTROL_MAGIC}\nsegment={}\ndurable_record_count={}\nlast_durable_txn_id={last_txn}\n",
        control.segment_path.display(),
        control.checkpoint.durable_record_count,
    );

    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL control file {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL control file {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL control file {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL control file {}: {err}",
            path.display()
        ))
    })
}

pub fn read_wal_control_file(path: impl AsRef<Path>) -> Result<WalControlFile, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL control file {}: {err}",
            path.display()
        ))
    })?;
    let mut lines = body.lines();
    if lines.next() != Some(WAL_CONTROL_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid WAL control header {}",
            path.display()
        )));
    }

    let segment_path = parse_control_value(lines.next(), "segment", path).map(PathBuf::from)?;
    let durable_record_count = parse_control_value(lines.next(), "durable_record_count", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL control durable_record_count {}: {err}",
                path.display()
            ))
        })?;
    let last_durable_txn_id = match parse_control_value(lines.next(), "last_durable_txn_id", path)?
    {
        "none" => None,
        raw => Some(raw.parse().map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL control last_durable_txn_id {}: {err}",
                path.display()
            ))
        })?),
    };

    Ok(WalControlFile {
        segment_path,
        checkpoint: WalCheckpointMeta {
            durable_record_count,
            last_durable_txn_id,
        },
    })
}

pub fn read_wal_checkpoint(
    control_path: impl AsRef<Path>,
) -> Result<(WalControlFile, Vec<WalRecord>), EngineError> {
    let control_path = control_path.as_ref();
    let control = read_wal_control_file(control_path)?;
    let segment_path = if control.segment_path.is_absolute() {
        control.segment_path.clone()
    } else {
        control_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&control.segment_path)
    };
    let mut records = read_wal_segment(&segment_path)?;
    // W1b audit fix 2: the rotation renames the checkpoint segment BEFORE the control file; a
    // crash between them leaves a NEWER (longer) checkpoint paired with the previous control.
    // The CONTROL FILE is the commit point — the checkpoint's extra tail records were never
    // committed as a checkpoint, but every one of them is still covered by the (untruncated)
    // live segment, so truncating the LIST to the control's count recovers exactly the
    // committed state. A SHORTER checkpoint than the control commits to remains a loud error
    // (acknowledged checkpoint data is missing).
    if records.len() > control.checkpoint.durable_record_count {
        records.truncate(control.checkpoint.durable_record_count);
    }
    validate_checkpoint_control(control_path, &control, &records)?;
    Ok((control, records))
}

pub fn write_wal_archive(
    manifest_path: impl AsRef<Path>,
    segment_dir: impl AsRef<Path>,
    records: &[WalRecord],
    records_per_segment: usize,
) -> Result<WalArchiveManifest, EngineError> {
    write_wal_archive_with_timestamps(
        manifest_path,
        segment_dir,
        records,
        records_per_segment,
        &[],
    )
}

pub fn write_wal_archive_with_timestamps(
    manifest_path: impl AsRef<Path>,
    segment_dir: impl AsRef<Path>,
    records: &[WalRecord],
    records_per_segment: usize,
    record_timestamps: &[WalArchiveRecordTimestamp],
) -> Result<WalArchiveManifest, EngineError> {
    if records_per_segment == 0 {
        return Err(EngineError::Durability(
            "WAL archive records_per_segment must be non-zero".to_string(),
        ));
    }

    let manifest_path = manifest_path.as_ref();
    let segment_dir = segment_dir.as_ref();
    validate_timestamp_metadata(manifest_path, records, record_timestamps)?;
    fs::create_dir_all(segment_dir).map_err(|err| {
        EngineError::Durability(format!(
            "failed to create WAL archive segment directory {}: {err}",
            segment_dir.display()
        ))
    })?;

    let mut segments = Vec::new();
    for (index, chunk) in records.chunks(records_per_segment).enumerate() {
        let file_name = format!("segment-{:04}.wal", index + 1);
        let segment_path = segment_dir.join(&file_name);
        write_wal_segment(&segment_path, chunk)?;
        let manifest_segment_path = segment_path
            .strip_prefix(manifest_path.parent().unwrap_or_else(|| Path::new(".")))
            .unwrap_or(&segment_path)
            .to_path_buf();
        segments.push(WalArchiveSegment {
            segment_path: manifest_segment_path,
            record_count: chunk.len(),
            first_txn_id: chunk.first().map(|record| record.txn_id),
            last_txn_id: chunk.last().map(|record| record.txn_id),
        });
    }

    let manifest = WalArchiveManifest {
        segments,
        checkpoint: WalCheckpointMeta {
            durable_record_count: records.len(),
            last_durable_txn_id: records.last().map(|record| record.txn_id),
        },
        record_timestamps: record_timestamps.to_vec(),
    };
    write_wal_archive_manifest(manifest_path, &manifest)?;
    Ok(manifest)
}

pub fn append_wal_archive_segment(
    manifest_path: impl AsRef<Path>,
    segment_path: impl AsRef<Path>,
) -> Result<WalArchiveManifest, EngineError> {
    append_wal_archive_segment_with_timestamps(manifest_path, segment_path, &[])
}

pub fn append_wal_archive_segment_with_timestamps(
    manifest_path: impl AsRef<Path>,
    segment_path: impl AsRef<Path>,
    record_timestamps: &[WalArchiveRecordTimestamp],
) -> Result<WalArchiveManifest, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let segment_path = segment_path.as_ref();
    let (manifest, mut records) = read_wal_archive(manifest_path)?;
    let segment_records = read_wal_segment(segment_path)?;
    if segment_records.is_empty() {
        return Err(EngineError::Durability(format!(
            "WAL archive {} cannot ingest empty segment {}",
            manifest_path.display(),
            segment_path.display()
        )));
    }

    validate_archive_ingest_timestamps(
        manifest_path,
        &manifest,
        &segment_records,
        record_timestamps,
    )?;
    validate_archive_ingest_continuity(manifest_path, &manifest, &segment_records)?;

    let appended_record_count = segment_records.len();
    let appended_first_txn_id = segment_records.first().map(|record| record.txn_id);
    let appended_last_txn_id = segment_records.last().map(|record| record.txn_id);
    records.extend(segment_records);
    let mut combined_timestamps = manifest.record_timestamps.clone();
    combined_timestamps.extend_from_slice(record_timestamps);
    validate_timestamp_metadata(manifest_path, &records, &combined_timestamps)?;

    let manifest_segment_path = segment_path
        .strip_prefix(manifest_path.parent().unwrap_or_else(|| Path::new(".")))
        .unwrap_or(segment_path)
        .to_path_buf();
    let mut segments = manifest.segments;
    segments.push(WalArchiveSegment {
        segment_path: manifest_segment_path,
        record_count: appended_record_count,
        first_txn_id: appended_first_txn_id,
        last_txn_id: appended_last_txn_id,
    });

    let appended = WalArchiveManifest {
        segments,
        checkpoint: WalCheckpointMeta {
            durable_record_count: records.len(),
            last_durable_txn_id: records.last().map(|record| record.txn_id),
        },
        record_timestamps: combined_timestamps,
    };
    validate_archive_manifest_shape(manifest_path, &appended)?;
    validate_archive_records(manifest_path, &appended, &records)?;
    validate_archive_timestamps(manifest_path, &appended, &records)?;
    write_wal_archive_manifest(manifest_path, &appended)?;
    Ok(appended)
}

pub fn export_wal_archive_object_backup(
    manifest_path: impl AsRef<Path>,
    backup_manifest_path: impl AsRef<Path>,
    object_dir: impl AsRef<Path>,
) -> Result<WalArchiveObjectBackup, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let backup_manifest_path = backup_manifest_path.as_ref();
    let object_dir = object_dir.as_ref();
    let (archive_manifest, _records) = read_wal_archive(manifest_path)?;

    fs::create_dir_all(object_dir).map_err(|err| {
        EngineError::Durability(format!(
            "failed to create WAL archive object directory {}: {err}",
            object_dir.display()
        ))
    })?;

    let mut objects = Vec::with_capacity(archive_manifest.segments.len() + 1);
    objects.push(write_wal_archive_backup_object(
        backup_manifest_path,
        Path::new("MANIFEST"),
        manifest_path,
        &object_dir.join("archive-manifest.object"),
    )?);
    for (idx, segment) in archive_manifest.segments.iter().enumerate() {
        let segment_source_path = resolve_manifest_path(manifest_path, &segment.segment_path);
        let object_path = object_dir.join(format!("segment-{:04}.wal.object", idx + 1));
        objects.push(write_wal_archive_backup_object(
            backup_manifest_path,
            &segment.segment_path,
            &segment_source_path,
            &object_path,
        )?);
    }

    let backup = WalArchiveObjectBackup {
        archive_manifest,
        objects,
    };
    write_wal_archive_object_backup_manifest(backup_manifest_path, &backup)?;
    Ok(backup)
}

pub fn restore_wal_archive_object_backup(
    backup_manifest_path: impl AsRef<Path>,
    restored_manifest_path: impl AsRef<Path>,
    restored_segment_dir: impl AsRef<Path>,
) -> Result<WalArchiveManifest, EngineError> {
    let backup_manifest_path = backup_manifest_path.as_ref();
    let restored_manifest_path = restored_manifest_path.as_ref();
    let restored_segment_dir = restored_segment_dir.as_ref();
    let backup = read_wal_archive_object_backup_manifest(backup_manifest_path)?;
    validate_archive_manifest_shape(backup_manifest_path, &backup.archive_manifest)?;

    let manifest_object = backup
        .objects
        .iter()
        .find(|object| object.source_path == Path::new("MANIFEST"))
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive object backup {} has no manifest object",
                backup_manifest_path.display()
            ))
        })?;
    let manifest_bytes =
        read_verified_wal_archive_backup_object(backup_manifest_path, manifest_object)?;
    let expected_manifest_bytes =
        render_wal_archive_manifest_body(&backup.archive_manifest)?.into_bytes();
    if manifest_bytes != expected_manifest_bytes {
        return Err(EngineError::Durability(format!(
            "WAL archive object backup {} manifest object does not match backup manifest metadata",
            backup_manifest_path.display()
        )));
    }

    if restored_segment_dir.exists() {
        return Err(EngineError::Durability(format!(
            "restored WAL archive segment directory {} already exists",
            restored_segment_dir.display()
        )));
    }
    let restored_segment_parent = restored_segment_dir
        .parent()
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(restored_segment_parent).map_err(|err| {
        EngineError::Durability(format!(
            "failed to create restored WAL archive segment parent {}: {err}",
            restored_segment_parent.display()
        ))
    })?;
    let restored_segment_name = restored_segment_dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "segments".to_string());
    let staging_segment_dir = restored_segment_parent.join(format!(
        ".{restored_segment_name}.restore-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&staging_segment_dir);
    fs::create_dir_all(&staging_segment_dir).map_err(|err| {
        EngineError::Durability(format!(
            "failed to create staging WAL archive segment directory {}: {err}",
            staging_segment_dir.display()
        ))
    })?;

    let restore_result = (|| {
        let mut restored_segments = Vec::with_capacity(backup.archive_manifest.segments.len());
        for (idx, segment) in backup.archive_manifest.segments.iter().enumerate() {
            let object = backup
                .objects
                .iter()
                .find(|object| object.source_path == segment.segment_path)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "WAL archive object backup {} missing segment object {}",
                        backup_manifest_path.display(),
                        segment.segment_path.display()
                    ))
                })?;
            let bytes = read_verified_wal_archive_backup_object(backup_manifest_path, object)?;
            let final_segment_path =
                restored_segment_dir.join(format!("segment-{:04}.wal", idx + 1));
            let staging_segment_path =
                staging_segment_dir.join(format!("segment-{:04}.wal", idx + 1));
            write_verified_backup_bytes(&staging_segment_path, &bytes)?;
            let manifest_segment_path = final_segment_path
                .strip_prefix(
                    restored_manifest_path
                        .parent()
                        .unwrap_or_else(|| Path::new(".")),
                )
                .unwrap_or(&final_segment_path)
                .to_path_buf();
            restored_segments.push(WalArchiveSegment {
                segment_path: manifest_segment_path,
                record_count: segment.record_count,
                first_txn_id: segment.first_txn_id,
                last_txn_id: segment.last_txn_id,
            });
        }

        fs::rename(&staging_segment_dir, restored_segment_dir).map_err(|err| {
            EngineError::Durability(format!(
                "failed to install restored WAL archive segment directory {}: {err}",
                restored_segment_dir.display()
            ))
        })?;
        Ok::<_, EngineError>(restored_segments)
    })();
    let restored_segments = match restore_result {
        Ok(restored_segments) => restored_segments,
        Err(err) => {
            let _ = fs::remove_dir_all(&staging_segment_dir);
            return Err(err);
        }
    };

    let restored_manifest = WalArchiveManifest {
        segments: restored_segments,
        checkpoint: backup.archive_manifest.checkpoint,
        record_timestamps: backup.archive_manifest.record_timestamps,
    };
    write_wal_archive_manifest(restored_manifest_path, &restored_manifest)?;
    let (validated_manifest, _records) = read_wal_archive(restored_manifest_path)?;
    Ok(validated_manifest)
}

pub fn fork_wal_archive_timeline_to_txn(
    source_manifest_path: impl AsRef<Path>,
    branch_manifest_path: impl AsRef<Path>,
    branch_segment_dir: impl AsRef<Path>,
    timeline_path: impl AsRef<Path>,
    timeline_id: impl AsRef<str>,
    parent_timeline_id: Option<&str>,
    target_txn_id: TxnId,
) -> Result<WalArchiveTimelineBranch, EngineError> {
    let source_manifest_path = source_manifest_path.as_ref();
    let branch_manifest_path = branch_manifest_path.as_ref();
    let branch_segment_dir = branch_segment_dir.as_ref();
    let timeline_path = timeline_path.as_ref();
    let timeline = validate_timeline_identity(
        timeline_path,
        timeline_id.as_ref(),
        parent_timeline_id,
        source_manifest_path,
        branch_manifest_path,
        target_txn_id,
        None,
    )?;
    let (source_manifest, target, records) =
        read_wal_archive_to_txn(source_manifest_path, target_txn_id)?;
    let record_timestamps = retained_timestamps(&source_manifest, target.recovered_record_count);
    let records_per_segment = archive_records_per_segment(source_manifest_path, &source_manifest)?;
    let manifest = write_wal_archive_with_timestamps(
        branch_manifest_path,
        branch_segment_dir,
        &records,
        records_per_segment,
        record_timestamps,
    )?;
    write_wal_archive_timeline(timeline_path, &timeline)?;
    Ok(WalArchiveTimelineBranch { timeline, manifest })
}

pub fn fork_wal_archive_timeline_to_timestamp_micros(
    source_manifest_path: impl AsRef<Path>,
    branch_manifest_path: impl AsRef<Path>,
    branch_segment_dir: impl AsRef<Path>,
    timeline_path: impl AsRef<Path>,
    timeline_id: impl AsRef<str>,
    parent_timeline_id: Option<&str>,
    target_timestamp_micros: u64,
) -> Result<WalArchiveTimelineBranch, EngineError> {
    let source_manifest_path = source_manifest_path.as_ref();
    let branch_manifest_path = branch_manifest_path.as_ref();
    let branch_segment_dir = branch_segment_dir.as_ref();
    let timeline_path = timeline_path.as_ref();
    let (source_manifest, target, records) =
        read_wal_archive_to_timestamp_micros(source_manifest_path, target_timestamp_micros)?;
    let timeline = validate_timeline_identity(
        timeline_path,
        timeline_id.as_ref(),
        parent_timeline_id,
        source_manifest_path,
        branch_manifest_path,
        target.target_txn_id,
        Some(target_timestamp_micros),
    )?;
    let record_timestamps = retained_timestamps(&source_manifest, target.recovered_record_count);
    let records_per_segment = archive_records_per_segment(source_manifest_path, &source_manifest)?;
    let manifest = write_wal_archive_with_timestamps(
        branch_manifest_path,
        branch_segment_dir,
        &records,
        records_per_segment,
        record_timestamps,
    )?;
    write_wal_archive_timeline(timeline_path, &timeline)?;
    Ok(WalArchiveTimelineBranch { timeline, manifest })
}

pub fn write_wal_archive_timeline(
    path: impl AsRef<Path>,
    timeline: &WalArchiveTimeline,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    validate_timeline_value(path, "timeline_id", &timeline.timeline_id)?;
    if let Some(parent) = timeline.parent_timeline_id.as_ref() {
        validate_timeline_value(path, "parent_timeline_id", parent)?;
        if parent == &timeline.timeline_id {
            return Err(EngineError::Durability(format!(
                "WAL archive timeline {} cannot be its own parent",
                timeline.timeline_id
            )));
        }
    }
    validate_timeline_path(path, "source_manifest_path", &timeline.source_manifest_path)?;
    validate_timeline_path(path, "branch_manifest_path", &timeline.branch_manifest_path)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL timeline directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let body = format!(
        "{WAL_ARCHIVE_TIMELINE_MAGIC}\ntimeline_id={}\nparent_timeline_id={}\nfork_txn_id={}\nfork_timestamp_micros={}\nsource_manifest_path={}\nbranch_manifest_path={}\n",
        timeline.timeline_id,
        timeline
            .parent_timeline_id
            .as_deref()
            .unwrap_or("none"),
        timeline.fork_txn_id,
        format_optional_u64(timeline.fork_timestamp_micros),
        timeline.source_manifest_path.display(),
        timeline.branch_manifest_path.display()
    );

    let tmp_path = temporary_control_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive timeline {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL archive timeline {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL archive timeline {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL archive timeline {}: {err}",
            path.display()
        ))
    })
}

pub fn read_wal_archive_timeline(
    path: impl AsRef<Path>,
) -> Result<WalArchiveTimeline, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive timeline {}: {err}",
            path.display()
        ))
    })?;
    let mut lines = body.lines();
    if lines.next() != Some(WAL_ARCHIVE_TIMELINE_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid WAL archive timeline header {}",
            path.display()
        )));
    }

    let timeline_id = parse_control_value(lines.next(), "timeline_id", path)?.to_string();
    let parent_timeline_id = match parse_control_value(lines.next(), "parent_timeline_id", path)? {
        "none" => None,
        parent => Some(parent.to_string()),
    };
    let fork_txn_id = parse_control_value(lines.next(), "fork_txn_id", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive timeline fork transaction {}: {err}",
                path.display()
            ))
        })?;
    let fork_timestamp_micros = parse_optional_u64(
        parse_control_value(lines.next(), "fork_timestamp_micros", path)?,
        "fork timestamp",
        path,
    )?;
    let source_manifest_path = PathBuf::from(parse_control_value(
        lines.next(),
        "source_manifest_path",
        path,
    )?);
    let branch_manifest_path = PathBuf::from(parse_control_value(
        lines.next(),
        "branch_manifest_path",
        path,
    )?);
    let timeline = WalArchiveTimeline {
        timeline_id,
        parent_timeline_id,
        fork_txn_id,
        fork_timestamp_micros,
        source_manifest_path,
        branch_manifest_path,
    };
    validate_timeline_value(path, "timeline_id", &timeline.timeline_id)?;
    if let Some(parent) = timeline.parent_timeline_id.as_ref() {
        validate_timeline_value(path, "parent_timeline_id", parent)?;
        if parent == &timeline.timeline_id {
            return Err(EngineError::Durability(format!(
                "WAL archive timeline {} cannot be its own parent",
                timeline.timeline_id
            )));
        }
    }
    validate_timeline_path(path, "source_manifest_path", &timeline.source_manifest_path)?;
    validate_timeline_path(path, "branch_manifest_path", &timeline.branch_manifest_path)?;
    Ok(timeline)
}

pub fn register_wal_archive_timeline(
    registry_path: impl AsRef<Path>,
    timeline_path: impl AsRef<Path>,
) -> Result<WalArchiveTimelineRegistry, EngineError> {
    let registry_path = registry_path.as_ref();
    let timeline_path = timeline_path.as_ref();
    let timeline = read_wal_archive_timeline(timeline_path)?;
    let mut registry = if registry_path.exists() {
        read_wal_archive_timeline_registry(registry_path)?
    } else {
        WalArchiveTimelineRegistry {
            timelines: Vec::new(),
        }
    };
    if registry
        .timelines
        .iter()
        .any(|entry| entry.timeline_id == timeline.timeline_id)
    {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline registry {} already contains timeline {}",
            registry_path.display(),
            timeline.timeline_id
        )));
    }
    if let Some(parent) = timeline.parent_timeline_id.as_ref() {
        if !registry
            .timelines
            .iter()
            .any(|entry| &entry.timeline_id == parent)
        {
            return Err(EngineError::Durability(format!(
                "WAL archive timeline registry {} is missing parent timeline {} for child {}",
                registry_path.display(),
                parent,
                timeline.timeline_id
            )));
        }
    }

    let (_manifest, _records) = read_wal_archive(&timeline.branch_manifest_path)?;
    registry.timelines.push(WalArchiveTimelineRegistryEntry {
        timeline_id: timeline.timeline_id,
        parent_timeline_id: timeline.parent_timeline_id,
        fork_txn_id: timeline.fork_txn_id,
        fork_timestamp_micros: timeline.fork_timestamp_micros,
        timeline_path: timeline_path.to_path_buf(),
        branch_manifest_path: timeline.branch_manifest_path,
    });
    write_wal_archive_timeline_registry(registry_path, &registry)?;
    read_wal_archive_timeline_registry(registry_path)
}

pub fn select_wal_archive_timeline(
    registry_path: impl AsRef<Path>,
    timeline_id: impl AsRef<str>,
) -> Result<WalArchiveTimelineSelection, EngineError> {
    let registry_path = registry_path.as_ref();
    let timeline_id = timeline_id.as_ref();
    validate_timeline_value(registry_path, "timeline_id", timeline_id)?;
    let registry = read_wal_archive_timeline_registry(registry_path)?;
    let entry = registry
        .timelines
        .iter()
        .find(|entry| entry.timeline_id == timeline_id)
        .cloned()
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive timeline registry {} has no timeline {}",
                registry_path.display(),
                timeline_id
            ))
        })?;
    let timeline = read_wal_archive_timeline(&entry.timeline_path)?;
    if timeline.timeline_id != entry.timeline_id
        || timeline.parent_timeline_id != entry.parent_timeline_id
        || timeline.fork_txn_id != entry.fork_txn_id
        || timeline.fork_timestamp_micros != entry.fork_timestamp_micros
        || timeline.branch_manifest_path != entry.branch_manifest_path
    {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline registry {} entry {} does not match sidecar {}",
            registry_path.display(),
            entry.timeline_id,
            entry.timeline_path.display()
        )));
    }
    let (manifest, _records) = read_wal_archive(&timeline.branch_manifest_path)?;
    Ok(WalArchiveTimelineSelection {
        entry,
        timeline,
        manifest,
    })
}

pub fn plan_wal_archive_timeline_prune(
    registry_path: impl AsRef<Path>,
    retained_timeline_id: impl AsRef<str>,
) -> Result<WalArchiveTimelinePrunePlan, EngineError> {
    let registry_path = registry_path.as_ref();
    let retained_timeline_id = retained_timeline_id.as_ref();
    validate_timeline_value(registry_path, "timeline_id", retained_timeline_id)?;
    let registry = read_wal_archive_timeline_registry(registry_path)?;
    let mut retained = HashSet::new();
    let mut next_timeline_id = Some(retained_timeline_id.to_string());
    while let Some(timeline_id) = next_timeline_id {
        let entry = registry
            .timelines
            .iter()
            .find(|entry| entry.timeline_id == timeline_id)
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "WAL archive timeline registry {} has no timeline {}",
                    registry_path.display(),
                    timeline_id
                ))
            })?;
        retained.insert(entry.timeline_id.clone());
        next_timeline_id = entry.parent_timeline_id.clone();
    }

    for entry in &registry.timelines {
        validate_registered_timeline_entry(registry_path, entry)?;
    }

    let retained_registry = WalArchiveTimelineRegistry {
        timelines: registry
            .timelines
            .iter()
            .filter(|entry| retained.contains(&entry.timeline_id))
            .cloned()
            .collect(),
    };
    validate_timeline_registry_shape(registry_path, &retained_registry)?;
    let retained_timeline_ids = retained_registry
        .timelines
        .iter()
        .map(|entry| entry.timeline_id.clone())
        .collect();
    let removed_timeline_ids = registry
        .timelines
        .iter()
        .filter(|entry| !retained.contains(&entry.timeline_id))
        .map(|entry| entry.timeline_id.clone())
        .collect();

    let retained_timeline_paths: HashSet<PathBuf> = retained_registry
        .timelines
        .iter()
        .map(|entry| entry.timeline_path.clone())
        .collect();
    let retained_manifest_paths: HashSet<PathBuf> = retained_registry
        .timelines
        .iter()
        .map(|entry| entry.branch_manifest_path.clone())
        .collect();
    let mut retained_segment_paths = HashSet::new();
    for entry in &retained_registry.timelines {
        let (manifest, _records) = read_wal_archive(&entry.branch_manifest_path)?;
        for segment in &manifest.segments {
            retained_segment_paths.insert(resolve_manifest_path(
                &entry.branch_manifest_path,
                &segment.segment_path,
            ));
        }
    }

    let mut removed_timeline_paths = Vec::new();
    let mut removed_branch_manifest_paths = Vec::new();
    let mut removed_segment_paths = Vec::new();
    let mut seen_removed_timeline_paths = HashSet::new();
    let mut seen_removed_manifest_paths = HashSet::new();
    let mut seen_removed_segment_paths = HashSet::new();
    for entry in registry
        .timelines
        .iter()
        .filter(|entry| !retained.contains(&entry.timeline_id))
    {
        if !retained_timeline_paths.contains(&entry.timeline_path)
            && seen_removed_timeline_paths.insert(entry.timeline_path.clone())
        {
            removed_timeline_paths.push(entry.timeline_path.clone());
        }
        if !retained_manifest_paths.contains(&entry.branch_manifest_path)
            && seen_removed_manifest_paths.insert(entry.branch_manifest_path.clone())
        {
            removed_branch_manifest_paths.push(entry.branch_manifest_path.clone());
        }

        let (manifest, _records) = read_wal_archive(&entry.branch_manifest_path)?;
        for segment in &manifest.segments {
            let segment_path =
                resolve_manifest_path(&entry.branch_manifest_path, &segment.segment_path);
            if !retained_segment_paths.contains(&segment_path)
                && seen_removed_segment_paths.insert(segment_path.clone())
            {
                removed_segment_paths.push(segment_path);
            }
        }
    }

    Ok(WalArchiveTimelinePrunePlan {
        retained_timeline_id: retained_timeline_id.to_string(),
        retained_timeline_ids,
        removed_timeline_ids,
        retained_registry,
        removed_timeline_paths,
        removed_branch_manifest_paths,
        removed_segment_paths,
    })
}

pub fn apply_wal_archive_timeline_prune(
    registry_path: impl AsRef<Path>,
    retained_timeline_id: impl AsRef<str>,
) -> Result<WalArchiveTimelinePrunePlan, EngineError> {
    let registry_path = registry_path.as_ref();
    let plan = plan_wal_archive_timeline_prune(registry_path, retained_timeline_id)?;
    write_wal_archive_timeline_registry(registry_path, &plan.retained_registry)?;
    for segment_path in &plan.removed_segment_paths {
        remove_wal_archive_timeline_artifact("segment", segment_path)?;
    }
    for manifest_path in &plan.removed_branch_manifest_paths {
        remove_wal_archive_timeline_artifact("manifest", manifest_path)?;
    }
    for timeline_path in &plan.removed_timeline_paths {
        remove_wal_archive_timeline_artifact("sidecar", timeline_path)?;
    }
    Ok(plan)
}

pub fn write_wal_archive_timeline_registry(
    path: impl AsRef<Path>,
    registry: &WalArchiveTimelineRegistry,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    validate_timeline_registry_shape(path, registry)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive timeline registry directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let mut body = format!(
        "{WAL_ARCHIVE_TIMELINE_REGISTRY_MAGIC}\ntimeline_count={}\n",
        registry.timelines.len()
    );
    for entry in &registry.timelines {
        body.push_str(&format!(
            "timeline={}|{}|{}|{}|{}|{}\n",
            entry.timeline_id,
            entry.parent_timeline_id.as_deref().unwrap_or("none"),
            entry.fork_txn_id,
            format_optional_u64(entry.fork_timestamp_micros),
            entry.timeline_path.display(),
            entry.branch_manifest_path.display()
        ));
    }

    let tmp_path = temporary_control_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive timeline registry {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL archive timeline registry {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL archive timeline registry {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL archive timeline registry {}: {err}",
            path.display()
        ))
    })
}

pub fn read_wal_archive_timeline_registry(
    path: impl AsRef<Path>,
) -> Result<WalArchiveTimelineRegistry, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive timeline registry {}: {err}",
            path.display()
        ))
    })?;
    let mut lines = body.lines();
    if lines.next() != Some(WAL_ARCHIVE_TIMELINE_REGISTRY_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid WAL archive timeline registry header {}",
            path.display()
        )));
    }
    let expected_count: usize = parse_control_value(lines.next(), "timeline_count", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive timeline registry count {}: {err}",
                path.display()
            ))
        })?;
    let mut timelines = Vec::new();
    for line in lines {
        let raw = line.strip_prefix("timeline=").ok_or_else(|| {
            EngineError::Durability(format!(
                "invalid WAL archive timeline registry entry in {}",
                path.display()
            ))
        })?;
        let parts: Vec<&str> = raw.split('|').collect();
        if parts.len() != 6 {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive timeline registry entry in {}",
                path.display()
            )));
        }
        let parent_timeline_id = match parts[1] {
            "none" => None,
            parent => Some(parent.to_string()),
        };
        let entry = WalArchiveTimelineRegistryEntry {
            timeline_id: parts[0].to_string(),
            parent_timeline_id,
            fork_txn_id: parts[2].parse().map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive timeline registry fork transaction {}: {err}",
                    path.display()
                ))
            })?,
            fork_timestamp_micros: parse_optional_u64(
                parts[3],
                "timeline registry fork timestamp",
                path,
            )?,
            timeline_path: PathBuf::from(parts[4]),
            branch_manifest_path: PathBuf::from(parts[5]),
        };
        timelines.push(entry);
    }
    let registry = WalArchiveTimelineRegistry { timelines };
    if registry.timelines.len() != expected_count {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline registry {} expected {expected_count} timelines but found {}",
            path.display(),
            registry.timelines.len()
        )));
    }
    validate_timeline_registry_shape(path, &registry)?;
    Ok(registry)
}

pub fn write_wal_archive_object_backup_manifest(
    path: impl AsRef<Path>,
    backup: &WalArchiveObjectBackup,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    validate_archive_manifest_shape(path, &backup.archive_manifest)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive object backup directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let manifest = &backup.archive_manifest;
    let mut body = format!(
        "{WAL_ARCHIVE_OBJECT_BACKUP_MAGIC}\ndurable_record_count={}\nlast_durable_txn_id={}\nsegments={}\n",
        manifest.checkpoint.durable_record_count,
        format_optional_txn(manifest.checkpoint.last_durable_txn_id),
        manifest.segments.len()
    );
    for segment in &manifest.segments {
        validate_backup_path(path, "segment", &segment.segment_path)?;
        body.push_str(&format!(
            "segment={}|{}|{}|{}\n",
            segment.segment_path.display(),
            segment.record_count,
            format_optional_txn(segment.first_txn_id),
            format_optional_txn(segment.last_txn_id)
        ));
    }
    body.push_str(&format!(
        "record_timestamps={}\n",
        manifest.record_timestamps.len()
    ));
    for timestamp in &manifest.record_timestamps {
        body.push_str(&format!(
            "record_timestamp={}|{}\n",
            timestamp.txn_id, timestamp.timestamp_micros
        ));
    }
    body.push_str(&format!("objects={}\n", backup.objects.len()));
    for object in &backup.objects {
        validate_backup_path(path, "object source", &object.source_path)?;
        validate_backup_path(path, "object path", &object.object_path)?;
        body.push_str(&format!(
            "object={}|{}|{}|{}\n",
            object.source_path.display(),
            object.object_path.display(),
            object.byte_len,
            object.checksum
        ));
    }

    let tmp_path = temporary_control_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive object backup {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL archive object backup {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL archive object backup {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL archive object backup {}: {err}",
            path.display()
        ))
    })
}

pub fn read_wal_archive_object_backup_manifest(
    path: impl AsRef<Path>,
) -> Result<WalArchiveObjectBackup, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive object backup {}: {err}",
            path.display()
        ))
    })?;
    let mut lines = body.lines();
    if lines.next() != Some(WAL_ARCHIVE_OBJECT_BACKUP_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid WAL archive object backup header {}",
            path.display()
        )));
    }

    let durable_record_count = parse_control_value(lines.next(), "durable_record_count", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive object backup durable_record_count {}: {err}",
                path.display()
            ))
        })?;
    let last_durable_txn_id = parse_optional_txn(
        parse_control_value(lines.next(), "last_durable_txn_id", path)?,
        "last_durable_txn_id",
        path,
    )?;
    let segment_count: usize = parse_control_value(lines.next(), "segments", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive object backup segments {}: {err}",
                path.display()
            ))
        })?;
    let mut segments = Vec::with_capacity(segment_count);
    for _ in 0..segment_count {
        let raw = parse_control_value(lines.next(), "segment", path)?;
        let mut parts = raw.split('|');
        let segment_path = PathBuf::from(parts.next().ok_or_else(|| {
            EngineError::Durability(format!(
                "missing WAL archive object backup segment path in {}",
                path.display()
            ))
        })?);
        let record_count = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup segment count in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive object backup segment count {}: {err}",
                    path.display()
                ))
            })?;
        let first_txn_id = parse_optional_txn(
            parts.next().ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup segment first txn in {}",
                    path.display()
                ))
            })?,
            "segment first txn",
            path,
        )?;
        let last_txn_id = parse_optional_txn(
            parts.next().ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup segment last txn in {}",
                    path.display()
                ))
            })?,
            "segment last txn",
            path,
        )?;
        if parts.next().is_some() {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive object backup segment field count in {}",
                path.display()
            )));
        }
        segments.push(WalArchiveSegment {
            segment_path,
            record_count,
            first_txn_id,
            last_txn_id,
        });
    }

    let timestamp_count: usize = parse_control_value(lines.next(), "record_timestamps", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive object backup timestamp count {}: {err}",
                path.display()
            ))
        })?;
    let mut record_timestamps = Vec::with_capacity(timestamp_count);
    for _ in 0..timestamp_count {
        let raw = parse_control_value(lines.next(), "record_timestamp", path)?;
        let mut parts = raw.split('|');
        let txn_id = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup timestamp txn in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive object backup timestamp txn {}: {err}",
                    path.display()
                ))
            })?;
        let timestamp_micros = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup timestamp value in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive object backup timestamp value {}: {err}",
                    path.display()
                ))
            })?;
        if parts.next().is_some() {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive object backup timestamp field count in {}",
                path.display()
            )));
        }
        record_timestamps.push(WalArchiveRecordTimestamp {
            txn_id,
            timestamp_micros,
        });
    }

    let object_count: usize = parse_control_value(lines.next(), "objects", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive object backup object count {}: {err}",
                path.display()
            ))
        })?;
    let mut objects = Vec::with_capacity(object_count);
    for _ in 0..object_count {
        let raw = parse_control_value(lines.next(), "object", path)?;
        let mut parts = raw.split('|');
        let source_path = PathBuf::from(parts.next().ok_or_else(|| {
            EngineError::Durability(format!(
                "missing WAL archive object backup source path in {}",
                path.display()
            ))
        })?);
        let object_path = PathBuf::from(parts.next().ok_or_else(|| {
            EngineError::Durability(format!(
                "missing WAL archive object backup object path in {}",
                path.display()
            ))
        })?);
        let byte_len = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup byte length in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive object backup byte length {}: {err}",
                    path.display()
                ))
            })?;
        let checksum = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive object backup checksum in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive object backup checksum {}: {err}",
                    path.display()
                ))
            })?;
        if parts.next().is_some() {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive object backup object field count in {}",
                path.display()
            )));
        }
        objects.push(WalArchiveObject {
            source_path,
            object_path,
            byte_len,
            checksum,
        });
    }
    if lines.next().is_some() {
        return Err(EngineError::Durability(format!(
            "unexpected trailing WAL archive object backup data in {}",
            path.display()
        )));
    }

    let archive_manifest = WalArchiveManifest {
        segments,
        checkpoint: WalCheckpointMeta {
            durable_record_count,
            last_durable_txn_id,
        },
        record_timestamps,
    };
    validate_archive_manifest_shape(path, &archive_manifest)?;
    Ok(WalArchiveObjectBackup {
        archive_manifest,
        objects,
    })
}

pub fn write_wal_archive_manifest(
    path: impl AsRef<Path>,
    manifest: &WalArchiveManifest,
) -> Result<(), EngineError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive manifest directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let body = render_wal_archive_manifest_body(manifest)?;

    let tmp_path = temporary_control_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive manifest {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(body.as_bytes()).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL archive manifest {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL archive manifest {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();

    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL archive manifest {}: {err}",
            path.display()
        ))
    })
}

fn render_wal_archive_manifest_body(manifest: &WalArchiveManifest) -> Result<String, EngineError> {
    let mut body = format!(
        "{WAL_ARCHIVE_MANIFEST_MAGIC}\ndurable_record_count={}\nlast_durable_txn_id={}\nsegments={}\n",
        manifest.checkpoint.durable_record_count,
        format_optional_txn(manifest.checkpoint.last_durable_txn_id),
        manifest.segments.len()
    );
    for segment in &manifest.segments {
        if segment.segment_path.to_string_lossy().contains('|') {
            return Err(EngineError::Durability(format!(
                "WAL archive segment path contains unsupported delimiter: {}",
                segment.segment_path.display()
            )));
        }
        body.push_str(&format!(
            "segment={}|{}|{}|{}\n",
            segment.segment_path.display(),
            segment.record_count,
            format_optional_txn(segment.first_txn_id),
            format_optional_txn(segment.last_txn_id)
        ));
    }
    for timestamp in &manifest.record_timestamps {
        body.push_str(&format!(
            "record_timestamp={}|{}\n",
            timestamp.txn_id, timestamp.timestamp_micros
        ));
    }
    Ok(body)
}

pub fn read_wal_archive_manifest(
    path: impl AsRef<Path>,
) -> Result<WalArchiveManifest, EngineError> {
    let path = path.as_ref();
    let body = fs::read_to_string(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive manifest {}: {err}",
            path.display()
        ))
    })?;
    let mut lines = body.lines();
    if lines.next() != Some(WAL_ARCHIVE_MANIFEST_MAGIC) {
        return Err(EngineError::Durability(format!(
            "invalid WAL archive manifest header {}",
            path.display()
        )));
    }

    let durable_record_count = parse_control_value(lines.next(), "durable_record_count", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive durable_record_count {}: {err}",
                path.display()
            ))
        })?;
    let last_durable_txn_id = parse_optional_txn(
        parse_control_value(lines.next(), "last_durable_txn_id", path)?,
        "last_durable_txn_id",
        path,
    )?;
    let segment_count: usize = parse_control_value(lines.next(), "segments", path)?
        .parse()
        .map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive segments {}: {err}",
                path.display()
            ))
        })?;

    let mut segments = Vec::with_capacity(segment_count);
    for _ in 0..segment_count {
        let raw = parse_control_value(lines.next(), "segment", path)?;
        let mut parts = raw.split('|');
        let segment_path = parts.next().ok_or_else(|| {
            EngineError::Durability(format!("invalid WAL archive segment in {}", path.display()))
        })?;
        let record_count = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive segment record count in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive segment record count {}: {err}",
                    path.display()
                ))
            })?;
        let first_txn_id = parse_optional_txn(
            parts.next().ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive segment first txn in {}",
                    path.display()
                ))
            })?,
            "segment first txn",
            path,
        )?;
        let last_txn_id = parse_optional_txn(
            parts.next().ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive segment last txn in {}",
                    path.display()
                ))
            })?,
            "segment last txn",
            path,
        )?;
        if parts.next().is_some() {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive segment field count in {}",
                path.display()
            )));
        }
        segments.push(WalArchiveSegment {
            segment_path: PathBuf::from(segment_path),
            record_count,
            first_txn_id,
            last_txn_id,
        });
    }
    let mut record_timestamps = Vec::new();
    for line in lines {
        let raw = parse_control_value(Some(line), "record_timestamp", path)?;
        let mut parts = raw.split('|');
        let txn_id = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive record timestamp txn in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive record timestamp txn {}: {err}",
                    path.display()
                ))
            })?;
        let timestamp_micros = parts
            .next()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "missing WAL archive record timestamp value in {}",
                    path.display()
                ))
            })?
            .parse()
            .map_err(|err| {
                EngineError::Durability(format!(
                    "invalid WAL archive record timestamp value {}: {err}",
                    path.display()
                ))
            })?;
        if parts.next().is_some() {
            return Err(EngineError::Durability(format!(
                "invalid WAL archive record timestamp field count in {}",
                path.display()
            )));
        }
        record_timestamps.push(WalArchiveRecordTimestamp {
            txn_id,
            timestamp_micros,
        });
    }

    let manifest = WalArchiveManifest {
        segments,
        checkpoint: WalCheckpointMeta {
            durable_record_count,
            last_durable_txn_id,
        },
        record_timestamps,
    };
    validate_archive_manifest_shape(path, &manifest)?;
    Ok(manifest)
}

pub fn read_wal_archive(
    manifest_path: impl AsRef<Path>,
) -> Result<(WalArchiveManifest, Vec<WalRecord>), EngineError> {
    let manifest_path = manifest_path.as_ref();
    let manifest = read_wal_archive_manifest(manifest_path)?;
    let mut records = Vec::new();
    for segment in &manifest.segments {
        let segment_path = resolve_manifest_path(manifest_path, &segment.segment_path);
        let segment_records = read_wal_segment(&segment_path)?;
        validate_archive_segment(manifest_path, segment, &segment_records)?;
        records.extend(segment_records);
    }
    validate_archive_records(manifest_path, &manifest, &records)?;
    validate_archive_timestamps(manifest_path, &manifest, &records)?;
    Ok((manifest, records))
}

pub fn read_wal_archive_to_txn(
    manifest_path: impl AsRef<Path>,
    target_txn_id: TxnId,
) -> Result<(WalArchiveManifest, WalArchiveRecoveryTarget, Vec<WalRecord>), EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, records) = read_wal_archive(manifest_path)?;
    let Some(first_txn_id) = records.first().map(|record| record.txn_id) else {
        return Err(EngineError::Durability(format!(
            "WAL archive {} has no records for target transaction {}",
            manifest_path.display(),
            target_txn_id
        )));
    };
    if target_txn_id < first_txn_id {
        return Err(EngineError::Durability(format!(
            "WAL archive {} target transaction {} is before first archived transaction {}",
            manifest_path.display(),
            target_txn_id,
            first_txn_id
        )));
    }
    if target_txn_id > manifest.checkpoint.last_durable_txn_id.unwrap_or(0) {
        return Err(EngineError::Durability(format!(
            "WAL archive {} target transaction {} is beyond last durable transaction {:?}",
            manifest_path.display(),
            target_txn_id,
            manifest.checkpoint.last_durable_txn_id
        )));
    }

    let recovered_record_count = records
        .iter()
        .take_while(|record| record.txn_id <= target_txn_id)
        .count();
    let last_recovered_txn_id = records
        .get(recovered_record_count.saturating_sub(1))
        .map(|record| record.txn_id);
    if last_recovered_txn_id != Some(target_txn_id) {
        return Err(EngineError::Durability(format!(
            "WAL archive {} does not contain target transaction {}",
            manifest_path.display(),
            target_txn_id
        )));
    }

    let target = WalArchiveRecoveryTarget {
        target_txn_id,
        recovered_record_count,
        last_recovered_txn_id: target_txn_id,
    };
    Ok((
        manifest,
        target,
        records.into_iter().take(recovered_record_count).collect(),
    ))
}

pub fn read_wal_archive_to_timestamp_micros(
    manifest_path: impl AsRef<Path>,
    target_timestamp_micros: u64,
) -> Result<
    (
        WalArchiveManifest,
        WalArchiveTimestampRecoveryTarget,
        Vec<WalRecord>,
    ),
    EngineError,
> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, records) = read_wal_archive(manifest_path)?;
    if records.is_empty() {
        return Err(EngineError::Durability(format!(
            "WAL archive {} has no records for target timestamp {}",
            manifest_path.display(),
            target_timestamp_micros
        )));
    }
    if manifest.record_timestamps.is_empty() {
        return Err(EngineError::Durability(format!(
            "WAL archive {} has no timestamp metadata for target timestamp {}",
            manifest_path.display(),
            target_timestamp_micros
        )));
    }

    let first_timestamp = manifest
        .record_timestamps
        .first()
        .expect("non-empty timestamp metadata")
        .timestamp_micros;
    let last_timestamp = manifest
        .record_timestamps
        .last()
        .expect("non-empty timestamp metadata")
        .timestamp_micros;
    if target_timestamp_micros < first_timestamp {
        return Err(EngineError::Durability(format!(
            "WAL archive {} target timestamp {} is before first archived timestamp {}",
            manifest_path.display(),
            target_timestamp_micros,
            first_timestamp
        )));
    }
    if target_timestamp_micros > last_timestamp {
        return Err(EngineError::Durability(format!(
            "WAL archive {} target timestamp {} is beyond last durable timestamp {}",
            manifest_path.display(),
            target_timestamp_micros,
            last_timestamp
        )));
    }

    let matching_indexes = manifest
        .record_timestamps
        .iter()
        .enumerate()
        .filter_map(|(idx, timestamp)| {
            (timestamp.timestamp_micros == target_timestamp_micros).then_some(idx)
        })
        .collect::<Vec<_>>();
    match matching_indexes.as_slice() {
        [] => Err(EngineError::Durability(format!(
            "WAL archive {} target timestamp {} falls between archived transaction boundaries",
            manifest_path.display(),
            target_timestamp_micros
        ))),
        [_first, _second, ..] => Err(EngineError::Durability(format!(
            "WAL archive {} target timestamp {} is ambiguous across multiple transaction boundaries",
            manifest_path.display(),
            target_timestamp_micros
        ))),
        [idx] => {
            let recovered_record_count = idx + 1;
            let target_txn_id = records[*idx].txn_id;
            let target = WalArchiveTimestampRecoveryTarget {
                target_timestamp_micros,
                target_txn_id,
                recovered_record_count,
                last_recovered_txn_id: target_txn_id,
            };
            Ok((
                manifest,
                target,
                records.into_iter().take(recovered_record_count).collect(),
            ))
        }
    }
}

pub fn plan_wal_archive_retention_to_txn(
    manifest_path: impl AsRef<Path>,
    target_txn_id: TxnId,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, target, retained_records) =
        read_wal_archive_to_txn(manifest_path, target_txn_id)?;
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let retained_manifest = build_wal_archive_manifest(
        manifest_path,
        &segment_dir,
        &retained_records,
        records_per_segment,
        &manifest.record_timestamps[..target
            .recovered_record_count
            .min(manifest.record_timestamps.len())],
    );
    let retained_paths: HashSet<PathBuf> = retained_manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .collect();
    let removed_segments = manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .filter(|path| !retained_paths.contains(path))
        .collect();

    Ok(WalArchiveRetentionPlan {
        target_txn_id,
        retained_record_count: target.recovered_record_count,
        removed_record_count: manifest
            .checkpoint
            .durable_record_count
            .saturating_sub(target.recovered_record_count),
        retained_manifest,
        removed_segments,
    })
}

pub fn plan_wal_archive_retention_to_timestamp_micros(
    manifest_path: impl AsRef<Path>,
    target_timestamp_micros: u64,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, target, retained_records) =
        read_wal_archive_to_timestamp_micros(manifest_path, target_timestamp_micros)?;
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let retained_manifest = build_wal_archive_manifest(
        manifest_path,
        &segment_dir,
        &retained_records,
        records_per_segment,
        &manifest.record_timestamps[..target
            .recovered_record_count
            .min(manifest.record_timestamps.len())],
    );
    let retained_paths: HashSet<PathBuf> = retained_manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .collect();
    let removed_segments = manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .filter(|path| !retained_paths.contains(path))
        .collect();

    Ok(WalArchiveRetentionPlan {
        target_txn_id: target.target_txn_id,
        retained_record_count: target.recovered_record_count,
        removed_record_count: manifest
            .checkpoint
            .durable_record_count
            .saturating_sub(target.recovered_record_count),
        retained_manifest,
        removed_segments,
    })
}

pub fn plan_wal_archive_retention_from_txn(
    manifest_path: impl AsRef<Path>,
    base_txn_id: TxnId,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, records) = read_wal_archive(manifest_path)?;
    let first_txn = records.first().map(|record| record.txn_id).ok_or_else(|| {
        EngineError::Durability(format!(
            "WAL archive {} has no records for base transaction {}",
            manifest_path.display(),
            base_txn_id
        ))
    })?;
    if base_txn_id < first_txn {
        return Err(EngineError::Durability(format!(
            "WAL archive {} base transaction {} is before first archived transaction {}",
            manifest_path.display(),
            base_txn_id,
            first_txn
        )));
    }
    let last_txn = manifest.checkpoint.last_durable_txn_id.ok_or_else(|| {
        EngineError::Durability(format!(
            "WAL archive {} has no durable transaction for base transaction {}",
            manifest_path.display(),
            base_txn_id
        ))
    })?;
    if base_txn_id > last_txn {
        return Err(EngineError::Durability(format!(
            "WAL archive {} base transaction {} is beyond last durable transaction {}",
            manifest_path.display(),
            base_txn_id,
            last_txn
        )));
    }
    let start_index = records
        .iter()
        .position(|record| record.txn_id == base_txn_id)
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive {} does not contain base backup transaction boundary {}",
                manifest_path.display(),
                base_txn_id
            ))
        })?;
    let retained_records = records[start_index..].to_vec();
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let retained_timestamps = if manifest.record_timestamps.is_empty() {
        &[][..]
    } else {
        &manifest.record_timestamps[start_index..]
    };
    let retained_manifest = build_wal_archive_manifest(
        manifest_path,
        &segment_dir,
        &retained_records,
        records_per_segment,
        retained_timestamps,
    );
    let retained_paths: HashSet<PathBuf> = retained_manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .collect();
    let removed_segments = manifest
        .segments
        .iter()
        .map(|segment| resolve_manifest_path(manifest_path, &segment.segment_path))
        .filter(|path| !retained_paths.contains(path))
        .collect();

    Ok(WalArchiveRetentionPlan {
        target_txn_id: base_txn_id,
        retained_record_count: retained_records.len(),
        removed_record_count: start_index,
        retained_manifest,
        removed_segments,
    })
}

pub fn apply_wal_archive_retention_from_txn(
    manifest_path: impl AsRef<Path>,
    base_txn_id: TxnId,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, records) = read_wal_archive(manifest_path)?;
    let plan = plan_wal_archive_retention_from_txn(manifest_path, base_txn_id)?;
    let start_index = records
        .iter()
        .position(|record| record.txn_id == base_txn_id)
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive {} does not contain base backup transaction boundary {}",
                manifest_path.display(),
                base_txn_id
            ))
        })?;
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let retained_records = &records[start_index..];
    let retained_timestamps = if manifest.record_timestamps.is_empty() {
        &[][..]
    } else {
        &manifest.record_timestamps[start_index..]
    };
    let retained_manifest = write_wal_archive_with_timestamps(
        manifest_path,
        &segment_dir,
        retained_records,
        records_per_segment,
        retained_timestamps,
    )?;
    for removed_segment in &plan.removed_segments {
        match fs::remove_file(removed_segment) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(EngineError::Durability(format!(
                    "failed to remove obsolete WAL archive segment {}: {err}",
                    removed_segment.display()
                )));
            }
        }
    }

    Ok(WalArchiveRetentionPlan {
        retained_manifest,
        ..plan
    })
}

pub fn apply_wal_archive_retention_to_timestamp_micros(
    manifest_path: impl AsRef<Path>,
    target_timestamp_micros: u64,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, _target, retained_records) =
        read_wal_archive_to_timestamp_micros(manifest_path, target_timestamp_micros)?;
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let plan =
        plan_wal_archive_retention_to_timestamp_micros(manifest_path, target_timestamp_micros)?;

    let retained_timestamps = &manifest.record_timestamps[..plan
        .retained_record_count
        .min(manifest.record_timestamps.len())];
    let retained_manifest = write_wal_archive_with_timestamps(
        manifest_path,
        &segment_dir,
        &retained_records,
        records_per_segment,
        retained_timestamps,
    )?;
    for removed_segment in &plan.removed_segments {
        match fs::remove_file(removed_segment) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(EngineError::Durability(format!(
                    "failed to remove obsolete WAL archive segment {}: {err}",
                    removed_segment.display()
                )));
            }
        }
    }

    Ok(WalArchiveRetentionPlan {
        retained_manifest,
        ..plan
    })
}

pub fn apply_wal_archive_retention_to_txn(
    manifest_path: impl AsRef<Path>,
    target_txn_id: TxnId,
) -> Result<WalArchiveRetentionPlan, EngineError> {
    let manifest_path = manifest_path.as_ref();
    let (manifest, _target, retained_records) =
        read_wal_archive_to_txn(manifest_path, target_txn_id)?;
    let records_per_segment = archive_records_per_segment(manifest_path, &manifest)?;
    let segment_dir = archive_segment_dir(manifest_path, &manifest)?;
    let plan = plan_wal_archive_retention_to_txn(manifest_path, target_txn_id)?;

    let retained_timestamps = &manifest.record_timestamps[..plan
        .retained_record_count
        .min(manifest.record_timestamps.len())];
    let retained_manifest = write_wal_archive_with_timestamps(
        manifest_path,
        &segment_dir,
        &retained_records,
        records_per_segment,
        retained_timestamps,
    )?;
    for removed_segment in &plan.removed_segments {
        match fs::remove_file(removed_segment) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(EngineError::Durability(format!(
                    "failed to remove obsolete WAL archive segment {}: {err}",
                    removed_segment.display()
                )));
            }
        }
    }

    Ok(WalArchiveRetentionPlan {
        retained_manifest,
        ..plan
    })
}

fn validate_checkpoint_control(
    control_path: &Path,
    control: &WalControlFile,
    records: &[WalRecord],
) -> Result<(), EngineError> {
    if records.len() != control.checkpoint.durable_record_count {
        return Err(EngineError::Durability(format!(
            "WAL control {} expected {} durable records but segment contains {}",
            control_path.display(),
            control.checkpoint.durable_record_count,
            records.len()
        )));
    }
    let actual_last_txn = records.last().map(|record| record.txn_id);
    if actual_last_txn != control.checkpoint.last_durable_txn_id {
        return Err(EngineError::Durability(format!(
            "WAL control {} expected last durable txn {:?} but segment contains {:?}",
            control_path.display(),
            control.checkpoint.last_durable_txn_id,
            actual_last_txn
        )));
    }
    Ok(())
}

fn build_wal_archive_manifest(
    manifest_path: &Path,
    segment_dir: &Path,
    records: &[WalRecord],
    records_per_segment: usize,
    record_timestamps: &[WalArchiveRecordTimestamp],
) -> WalArchiveManifest {
    let mut segments = Vec::new();
    for (index, chunk) in records.chunks(records_per_segment).enumerate() {
        let file_name = format!("segment-{:04}.wal", index + 1);
        let segment_path = segment_dir.join(&file_name);
        let manifest_segment_path = segment_path
            .strip_prefix(manifest_path.parent().unwrap_or_else(|| Path::new(".")))
            .unwrap_or(&segment_path)
            .to_path_buf();
        segments.push(WalArchiveSegment {
            segment_path: manifest_segment_path,
            record_count: chunk.len(),
            first_txn_id: chunk.first().map(|record| record.txn_id),
            last_txn_id: chunk.last().map(|record| record.txn_id),
        });
    }
    WalArchiveManifest {
        segments,
        checkpoint: WalCheckpointMeta {
            durable_record_count: records.len(),
            last_durable_txn_id: records.last().map(|record| record.txn_id),
        },
        record_timestamps: record_timestamps.to_vec(),
    }
}

fn retained_timestamps(
    manifest: &WalArchiveManifest,
    retained_record_count: usize,
) -> &[WalArchiveRecordTimestamp] {
    &manifest.record_timestamps[..retained_record_count.min(manifest.record_timestamps.len())]
}

fn write_wal_archive_backup_object(
    backup_manifest_path: &Path,
    source_path: &Path,
    source_file_path: &Path,
    object_path: &Path,
) -> Result<WalArchiveObject, EngineError> {
    validate_backup_path(backup_manifest_path, "object source", source_path)?;
    validate_backup_path(backup_manifest_path, "object path", object_path)?;
    let bytes = fs::read(source_file_path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive backup source {}: {err}",
            source_file_path.display()
        ))
    })?;
    write_verified_backup_bytes(object_path, &bytes)?;
    let object_path = object_path
        .strip_prefix(
            backup_manifest_path
                .parent()
                .unwrap_or_else(|| Path::new(".")),
        )
        .unwrap_or(object_path)
        .to_path_buf();
    Ok(WalArchiveObject {
        source_path: source_path.to_path_buf(),
        object_path,
        byte_len: bytes.len() as u64,
        checksum: wal_object_checksum(&bytes),
    })
}

fn read_verified_wal_archive_backup_object(
    backup_manifest_path: &Path,
    object: &WalArchiveObject,
) -> Result<Vec<u8>, EngineError> {
    let object_path = resolve_manifest_path(backup_manifest_path, &object.object_path);
    let bytes = fs::read(&object_path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to read WAL archive backup object {}: {err}",
            object_path.display()
        ))
    })?;
    if bytes.len() as u64 != object.byte_len {
        return Err(EngineError::Durability(format!(
            "WAL archive backup object {} expected {} bytes but read {}",
            object_path.display(),
            object.byte_len,
            bytes.len()
        )));
    }
    let actual_checksum = wal_object_checksum(&bytes);
    if actual_checksum != object.checksum {
        return Err(EngineError::Durability(format!(
            "WAL archive backup object {} checksum mismatch",
            object_path.display()
        )));
    }
    Ok(bytes)
}

fn write_verified_backup_bytes(path: &Path, bytes: &[u8]) -> Result<(), EngineError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive backup object directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    let tmp_path = temporary_control_path(path);
    let write_result = (|| {
        let mut file = File::create(&tmp_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to create WAL archive backup object {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.write_all(bytes).map_err(|err| {
            EngineError::Durability(format!(
                "failed to write WAL archive backup object {}: {err}",
                tmp_path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            EngineError::Durability(format!(
                "failed to sync WAL archive backup object {}: {err}",
                tmp_path.display()
            ))
        })?;
        Ok::<_, EngineError>(())
    })();
    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        EngineError::Durability(format!(
            "failed to install WAL archive backup object {}: {err}",
            path.display()
        ))
    })
}

fn validate_backup_path(path: &Path, field: &str, value: &Path) -> Result<(), EngineError> {
    let rendered = value.to_string_lossy();
    if rendered.is_empty()
        || rendered.contains('|')
        || rendered.contains('\n')
        || rendered.contains('\r')
    {
        return Err(EngineError::Durability(format!(
            "WAL archive object backup {field} contains unsupported path in {}",
            path.display()
        )));
    }
    Ok(())
}

fn archive_records_per_segment(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
) -> Result<usize, EngineError> {
    manifest
        .segments
        .first()
        .map(|segment| segment.record_count)
        .filter(|record_count| *record_count > 0)
        .ok_or_else(|| {
            EngineError::Durability(format!(
                "WAL archive {} has no segment sizing for retention",
                manifest_path.display()
            ))
        })
}

fn archive_segment_dir(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
) -> Result<PathBuf, EngineError> {
    let first_segment = manifest.segments.first().ok_or_else(|| {
        EngineError::Durability(format!(
            "WAL archive {} has no segment directory for retention",
            manifest_path.display()
        ))
    })?;
    let first_segment_path = resolve_manifest_path(manifest_path, &first_segment.segment_path);
    Ok(first_segment_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf())
}

fn validate_archive_manifest_shape(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
) -> Result<(), EngineError> {
    let segment_records: usize = manifest
        .segments
        .iter()
        .map(|segment| segment.record_count)
        .sum();
    if segment_records != manifest.checkpoint.durable_record_count {
        return Err(EngineError::Durability(format!(
            "WAL archive {} expected {} durable records but manifest segments describe {}",
            manifest_path.display(),
            manifest.checkpoint.durable_record_count,
            segment_records
        )));
    }
    if manifest.segments.is_empty() && manifest.checkpoint.last_durable_txn_id.is_some() {
        return Err(EngineError::Durability(format!(
            "WAL archive {} has no segments but records a last durable transaction",
            manifest_path.display()
        )));
    }
    if !manifest.record_timestamps.is_empty()
        && manifest.record_timestamps.len() != manifest.checkpoint.durable_record_count
    {
        return Err(EngineError::Durability(format!(
            "WAL archive {} expected {} timestamp records but manifest contains {}",
            manifest_path.display(),
            manifest.checkpoint.durable_record_count,
            manifest.record_timestamps.len()
        )));
    }
    Ok(())
}

fn validate_archive_ingest_continuity(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
    segment_records: &[WalRecord],
) -> Result<(), EngineError> {
    let Some(first_appended_txn) = segment_records.first().map(|record| record.txn_id) else {
        return Ok(());
    };
    if let Some(last_durable_txn) = manifest.checkpoint.last_durable_txn_id {
        if first_appended_txn <= last_durable_txn {
            return Err(EngineError::Durability(format!(
                "WAL archive {} ingest segment starts at transaction {} not after durable transaction {}",
                manifest_path.display(),
                first_appended_txn,
                last_durable_txn
            )));
        }
    }
    for window in segment_records.windows(2) {
        if window[0].txn_id >= window[1].txn_id {
            return Err(EngineError::Durability(format!(
                "WAL archive {} ingest segment has non-increasing transaction order at {} then {}",
                manifest_path.display(),
                window[0].txn_id,
                window[1].txn_id
            )));
        }
    }
    Ok(())
}

fn validate_archive_ingest_timestamps(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
    segment_records: &[WalRecord],
    record_timestamps: &[WalArchiveRecordTimestamp],
) -> Result<(), EngineError> {
    match (
        manifest.record_timestamps.is_empty(),
        manifest.checkpoint.durable_record_count,
        record_timestamps.is_empty(),
    ) {
        (true, 0, _) => {}
        (true, _, true) => {}
        (true, _, false) => {
            return Err(EngineError::Durability(format!(
                "WAL archive {} cannot add timestamp metadata to an existing archive without timestamps",
                manifest_path.display()
            )));
        }
        (false, _, true) => {
            return Err(EngineError::Durability(format!(
                "WAL archive {} requires timestamp metadata for ingested segment",
                manifest_path.display()
            )));
        }
        (false, _, false) => {}
    }
    validate_timestamp_metadata(manifest_path, segment_records, record_timestamps)?;
    if let (Some(existing_last), Some(appended_first)) = (
        manifest
            .record_timestamps
            .last()
            .map(|timestamp| timestamp.timestamp_micros),
        record_timestamps
            .first()
            .map(|timestamp| timestamp.timestamp_micros),
    ) {
        if appended_first < existing_last {
            return Err(EngineError::Durability(format!(
                "WAL archive {} ingest segment timestamp {} is before last archived timestamp {}",
                manifest_path.display(),
                appended_first,
                existing_last
            )));
        }
    }
    Ok(())
}

fn validate_timeline_identity(
    timeline_path: &Path,
    timeline_id: &str,
    parent_timeline_id: Option<&str>,
    source_manifest_path: &Path,
    branch_manifest_path: &Path,
    fork_txn_id: TxnId,
    fork_timestamp_micros: Option<u64>,
) -> Result<WalArchiveTimeline, EngineError> {
    validate_timeline_value(timeline_path, "timeline_id", timeline_id)?;
    if let Some(parent) = parent_timeline_id {
        validate_timeline_value(timeline_path, "parent_timeline_id", parent)?;
        if parent == timeline_id {
            return Err(EngineError::Durability(format!(
                "WAL archive timeline {timeline_id} cannot be its own parent"
            )));
        }
    }
    if source_manifest_path == branch_manifest_path {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline {timeline_id} cannot fork into the source manifest {}",
            source_manifest_path.display()
        )));
    }
    validate_timeline_path(timeline_path, "source_manifest_path", source_manifest_path)?;
    validate_timeline_path(timeline_path, "branch_manifest_path", branch_manifest_path)?;
    Ok(WalArchiveTimeline {
        timeline_id: timeline_id.to_string(),
        parent_timeline_id: parent_timeline_id.map(ToOwned::to_owned),
        fork_txn_id,
        fork_timestamp_micros,
        source_manifest_path: source_manifest_path.to_path_buf(),
        branch_manifest_path: branch_manifest_path.to_path_buf(),
    })
}

fn validate_timeline_value(path: &Path, field: &str, value: &str) -> Result<(), EngineError> {
    if value.is_empty() {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline {field} must not be empty in {}",
            path.display()
        )));
    }
    if value == "none" || value.contains('\n') || value.contains('\r') {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline {field} contains unsupported value in {}",
            path.display()
        )));
    }
    Ok(())
}

fn validate_timeline_path(path: &Path, field: &str, value: &Path) -> Result<(), EngineError> {
    let rendered = value.to_string_lossy();
    if rendered.is_empty()
        || rendered.contains('\n')
        || rendered.contains('\r')
        || rendered.contains('|')
    {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline {field} contains unsupported path in {}",
            path.display()
        )));
    }
    Ok(())
}

fn validate_timeline_registry_shape(
    path: &Path,
    registry: &WalArchiveTimelineRegistry,
) -> Result<(), EngineError> {
    let mut seen = HashSet::new();
    for entry in &registry.timelines {
        validate_timeline_value(path, "timeline_id", &entry.timeline_id)?;
        if !seen.insert(entry.timeline_id.clone()) {
            return Err(EngineError::Durability(format!(
                "WAL archive timeline registry {} contains duplicate timeline {}",
                path.display(),
                entry.timeline_id
            )));
        }
        if let Some(parent) = entry.parent_timeline_id.as_ref() {
            validate_timeline_value(path, "parent_timeline_id", parent)?;
            if parent == &entry.timeline_id {
                return Err(EngineError::Durability(format!(
                    "WAL archive timeline {} cannot be its own parent",
                    entry.timeline_id
                )));
            }
            if !seen.contains(parent) {
                return Err(EngineError::Durability(format!(
                    "WAL archive timeline registry {} lists child {} before parent {}",
                    path.display(),
                    entry.timeline_id,
                    parent
                )));
            }
        }
        validate_timeline_path(path, "timeline_path", &entry.timeline_path)?;
        validate_timeline_path(path, "branch_manifest_path", &entry.branch_manifest_path)?;
    }
    Ok(())
}

fn validate_registered_timeline_entry(
    registry_path: &Path,
    entry: &WalArchiveTimelineRegistryEntry,
) -> Result<(), EngineError> {
    let timeline = read_wal_archive_timeline(&entry.timeline_path)?;
    if timeline.timeline_id != entry.timeline_id
        || timeline.parent_timeline_id != entry.parent_timeline_id
        || timeline.fork_txn_id != entry.fork_txn_id
        || timeline.fork_timestamp_micros != entry.fork_timestamp_micros
        || timeline.branch_manifest_path != entry.branch_manifest_path
    {
        return Err(EngineError::Durability(format!(
            "WAL archive timeline registry {} entry {} does not match sidecar {}",
            registry_path.display(),
            entry.timeline_id,
            entry.timeline_path.display()
        )));
    }
    let (_manifest, _records) = read_wal_archive(&entry.branch_manifest_path)?;
    Ok(())
}

fn remove_wal_archive_timeline_artifact(kind: &str, path: &Path) -> Result<(), EngineError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(EngineError::Durability(format!(
            "failed to remove obsolete WAL archive timeline {kind} {}: {err}",
            path.display()
        ))),
    }
}

fn validate_archive_segment(
    manifest_path: &Path,
    segment: &WalArchiveSegment,
    records: &[WalRecord],
) -> Result<(), EngineError> {
    if records.len() != segment.record_count {
        return Err(EngineError::Durability(format!(
            "WAL archive {} segment {} expected {} records but contains {}",
            manifest_path.display(),
            segment.segment_path.display(),
            segment.record_count,
            records.len()
        )));
    }
    let actual_first = records.first().map(|record| record.txn_id);
    let actual_last = records.last().map(|record| record.txn_id);
    if actual_first != segment.first_txn_id || actual_last != segment.last_txn_id {
        return Err(EngineError::Durability(format!(
            "WAL archive {} segment {} expected txn range {:?}..{:?} but contains {:?}..{:?}",
            manifest_path.display(),
            segment.segment_path.display(),
            segment.first_txn_id,
            segment.last_txn_id,
            actual_first,
            actual_last
        )));
    }
    Ok(())
}

fn validate_archive_records(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
    records: &[WalRecord],
) -> Result<(), EngineError> {
    if records.len() != manifest.checkpoint.durable_record_count {
        return Err(EngineError::Durability(format!(
            "WAL archive {} expected {} durable records but read {}",
            manifest_path.display(),
            manifest.checkpoint.durable_record_count,
            records.len()
        )));
    }
    let actual_last = records.last().map(|record| record.txn_id);
    if actual_last != manifest.checkpoint.last_durable_txn_id {
        return Err(EngineError::Durability(format!(
            "WAL archive {} expected last durable txn {:?} but read {:?}",
            manifest_path.display(),
            manifest.checkpoint.last_durable_txn_id,
            actual_last
        )));
    }
    for window in records.windows(2) {
        if window[0].txn_id >= window[1].txn_id {
            return Err(EngineError::Durability(format!(
                "WAL archive {} has non-increasing transaction order at {} then {}",
                manifest_path.display(),
                window[0].txn_id,
                window[1].txn_id
            )));
        }
    }
    Ok(())
}

fn validate_timestamp_metadata(
    manifest_path: &Path,
    records: &[WalRecord],
    record_timestamps: &[WalArchiveRecordTimestamp],
) -> Result<(), EngineError> {
    if record_timestamps.is_empty() {
        return Ok(());
    }
    if record_timestamps.len() != records.len() {
        return Err(EngineError::Durability(format!(
            "WAL archive {} expected {} timestamp records but received {}",
            manifest_path.display(),
            records.len(),
            record_timestamps.len()
        )));
    }
    for (record, timestamp) in records.iter().zip(record_timestamps) {
        if record.txn_id != timestamp.txn_id {
            return Err(EngineError::Durability(format!(
                "WAL archive {} timestamp metadata transaction {} does not match record transaction {}",
                manifest_path.display(),
                timestamp.txn_id,
                record.txn_id
            )));
        }
    }
    Ok(())
}

fn validate_archive_timestamps(
    manifest_path: &Path,
    manifest: &WalArchiveManifest,
    records: &[WalRecord],
) -> Result<(), EngineError> {
    validate_timestamp_metadata(manifest_path, records, &manifest.record_timestamps)?;
    for window in manifest.record_timestamps.windows(2) {
        if window[0].timestamp_micros > window[1].timestamp_micros {
            return Err(EngineError::Durability(format!(
                "WAL archive {} has decreasing timestamp order at {} then {}",
                manifest_path.display(),
                window[0].timestamp_micros,
                window[1].timestamp_micros
            )));
        }
    }
    Ok(())
}

fn resolve_manifest_path(manifest_path: &Path, data_path: &Path) -> PathBuf {
    if data_path.is_absolute() {
        data_path.to_path_buf()
    } else {
        manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(data_path)
    }
}

fn format_optional_txn(txn_id: Option<TxnId>) -> String {
    txn_id
        .map(|txn_id| txn_id.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn format_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn parse_optional_txn(raw: &str, field: &str, path: &Path) -> Result<Option<TxnId>, EngineError> {
    match raw {
        "none" => Ok(None),
        raw => raw.parse().map(Some).map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive {field} {}: {err}",
                path.display()
            ))
        }),
    }
}

fn parse_optional_u64(raw: &str, field: &str, path: &Path) -> Result<Option<u64>, EngineError> {
    match raw {
        "none" => Ok(None),
        raw => raw.parse().map(Some).map_err(|err| {
            EngineError::Durability(format!(
                "invalid WAL archive {field} {}: {err}",
                path.display()
            ))
        }),
    }
}

fn write_record(file: &mut File, record: &WalRecord) -> Result<(), EngineError> {
    let mut bytes = Vec::with_capacity(WAL_RECORD_HEADER_LEN + record.payload.len());
    encode_record_into(&mut bytes, record)?;
    file.write_all(&bytes)
        .map_err(|err| EngineError::Durability(format!("failed to write WAL record: {err}")))
}

/// Serialize one record (header + payload) onto `buf` in the on-disk segment format.
/// Encode one record's envelope+payload into `buf` from raw parts — the fused
/// single-pass form for lane pumps (payload bytes are warm from the row-id
/// patch that immediately precedes this in the caller's loop).
pub fn encode_wal_record_parts_into(buf: &mut Vec<u8>, txn_id: TxnId, payload: &[u8]) {
    let payload_len = payload.len() as u64;
    let checksum = wal_record_checksum(txn_id, payload_len, payload);
    buf.extend_from_slice(&txn_id.to_le_bytes());
    buf.extend_from_slice(&payload_len.to_le_bytes());
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf.extend_from_slice(payload);
}

fn encode_record_into(buf: &mut Vec<u8>, record: &WalRecord) -> Result<(), EngineError> {
    let payload_len = u64::try_from(record.payload.len()).map_err(|_| {
        EngineError::Durability("WAL record payload length exceeds u64".to_string())
    })?;
    let checksum = wal_record_checksum(record.txn_id, payload_len, &record.payload);
    buf.extend_from_slice(&record.txn_id.to_le_bytes());
    buf.extend_from_slice(&payload_len.to_le_bytes());
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf.extend_from_slice(&record.payload);
    Ok(())
}

/// On-disk byte length of one serialized record.
fn encoded_record_len(record: &WalRecord) -> u64 {
    WAL_RECORD_HEADER_LEN as u64 + record.payload.len() as u64
}

/// Strictly decode a run of [`encode_record_into`]-encoded records that must consume EXACTLY
/// `bytes` (no torn tail — the caller has already validated the container's integrity, e.g. a FUA
/// frame's payload CRC). This is the FUA backend's replay decode: a frame payload is the byte-for-
/// byte record run the serial group flush would have written, so it decodes to the identical
/// `WalRecord`s. Each record's own checksum is re-verified as defense in depth, and any leftover
/// or truncated bytes are a hard error (an intact frame can never contain a partial record).
#[cfg(unix)]
pub(crate) fn decode_wal_record_run(bytes: &[u8]) -> Result<Vec<WalRecord>, EngineError> {
    let mut records = Vec::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        if bytes.len() - offset < WAL_RECORD_HEADER_LEN {
            return Err(EngineError::Durability(
                "FUA WAL frame payload ends with a truncated record header".to_string(),
            ));
        }
        let header = &bytes[offset..offset + WAL_RECORD_HEADER_LEN];
        let txn_id = u64::from_le_bytes(header[0..8].try_into().expect("txn id bytes"));
        let payload_len = u64::from_le_bytes(header[8..16].try_into().expect("payload len bytes"));
        let expected_checksum =
            u64::from_le_bytes(header[16..24].try_into().expect("checksum bytes"));
        let payload_start = offset + WAL_RECORD_HEADER_LEN;
        let payload_end = usize::try_from(payload_len)
            .ok()
            .and_then(|len| payload_start.checked_add(len))
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| {
                EngineError::Durability(
                    "FUA WAL frame payload record length overruns the frame".to_string(),
                )
            })?;
        let payload = &bytes[payload_start..payload_end];
        if wal_record_checksum(txn_id, payload_len, payload) != expected_checksum {
            return Err(EngineError::Durability(format!(
                "FUA WAL frame payload record checksum mismatch for txn {txn_id}"
            )));
        }
        records.push(WalRecord {
            txn_id,
            payload: payload.to_vec().into(),
        });
        offset = payload_end;
    }
    Ok(records)
}

fn temporary_segment_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    path.with_file_name(format!(".{file_name}.tmp.{}", std::process::id()))
}

fn temporary_control_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.control");
    path.with_file_name(format!(".{file_name}.tmp.{}", std::process::id()))
}

fn parse_control_value<'a>(
    line: Option<&'a str>,
    key: &str,
    path: &Path,
) -> Result<&'a str, EngineError> {
    let line = line.ok_or_else(|| {
        EngineError::Durability(format!(
            "missing WAL control field {key} in {}",
            path.display()
        ))
    })?;
    line.strip_prefix(&format!("{key}=")).ok_or_else(|| {
        EngineError::Durability(format!(
            "invalid WAL control field {key} in {}",
            path.display()
        ))
    })
}

fn wal_record_checksum(txn_id: TxnId, payload_len: u64, payload: &[u8]) -> u64 {
    // FNV-1a, BYTE-IDENTICAL to the original chained-iterator form (same
    // format, same values) but in tight slice loops: the chained iterator
    // defeated optimization and was measured at ~1.5us per ~100B record on
    // the lane pump's encode stage (1.5ms of a 1000-record wave).
    #[inline]
    fn fnv_step(mut hash: u64, bytes: &[u8]) -> u64 {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    hash = fnv_step(hash, &txn_id.to_le_bytes());
    hash = fnv_step(hash, &payload_len.to_le_bytes());
    fnv_step(hash, payload)
}

fn wal_object_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_PATH_ID: AtomicU64 = AtomicU64::new(1);

    fn test_wal_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "gpu-db-wal-{name}-{}-{}.segment",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn flush_commits_all_appended_records() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        });

        wal.flush_all().unwrap();

        assert_eq!(wal.flushed_count(), 2);
    }

    #[test]
    fn fail_next_flush_is_one_shot() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        });

        wal.fail_next_flush();
        let err = wal.flush_all().unwrap_err();
        assert!(matches!(err, EngineError::Durability(_)));
        assert_eq!(wal.flushed_count(), 0);

        wal.flush_all().unwrap();
        assert_eq!(wal.flushed_count(), 1);
    }

    #[test]
    fn truncate_shrinks_records_and_adjusts_flushed_count() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        });

        wal.flush_all().unwrap();
        assert_eq!(wal.flushed_count(), 2);

        wal.truncate(1);
        assert_eq!(wal.len(), 1);
        assert_eq!(wal.flushed_count(), 1);
    }

    #[test]
    fn unflushed_count_tracks_unpersisted_tail() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        });

        assert_eq!(wal.unflushed_count(), 2);

        wal.flush_all().unwrap();
        assert_eq!(wal.unflushed_count(), 0);

        wal.append(WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        });
        assert_eq!(wal.unflushed_count(), 1);
    }

    #[test]
    fn flushed_records_expose_only_durable_prefix() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        });

        assert!(wal.flushed_records().is_empty());

        wal.flush_all().unwrap();
        wal.append(WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec().into(),
        });

        let durable = wal.flushed_records();
        assert_eq!(durable.len(), 2);
        assert_eq!(durable[0].txn_id, 1);
        assert_eq!(durable[1].txn_id, 2);
    }

    #[test]
    fn flush_failure_does_not_advance_flushed_records() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        });

        wal.fail_next_flush();
        let _ = wal.flush_all();

        assert!(wal.flushed_records().is_empty());
        assert_eq!(wal.unflushed_count(), 1);
    }

    #[test]
    fn checkpoint_meta_tracks_durable_prefix_and_last_txn_id() {
        let mut wal = WalBuffer::default();
        assert_eq!(
            wal.checkpoint_meta(),
            WalCheckpointMeta {
                durable_record_count: 0,
                last_durable_txn_id: None,
            }
        );

        wal.append(WalRecord {
            txn_id: 7,
            payload: b"SET a=1".to_vec().into(),
        });
        wal.append(WalRecord {
            txn_id: 8,
            payload: b"SET b=2".to_vec().into(),
        });

        assert_eq!(
            wal.checkpoint_meta(),
            WalCheckpointMeta {
                durable_record_count: 0,
                last_durable_txn_id: None,
            }
        );

        wal.flush_all().unwrap();
        assert_eq!(
            wal.checkpoint_meta(),
            WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(8),
            }
        );

        wal.append(WalRecord {
            txn_id: 9,
            payload: b"SET c=3".to_vec().into(),
        });
        assert_eq!(
            wal.checkpoint_meta(),
            WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(8),
            }
        );
    }

    #[test]
    fn checkpoint_meta_does_not_advance_on_failed_flush() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 11,
            payload: b"SET a=1".to_vec().into(),
        });
        wal.flush_all().unwrap();

        wal.append(WalRecord {
            txn_id: 12,
            payload: b"SET b=2".to_vec().into(),
        });
        wal.fail_next_flush();
        assert!(wal.flush_all().is_err());

        assert_eq!(
            wal.checkpoint_meta(),
            WalCheckpointMeta {
                durable_record_count: 1,
                last_durable_txn_id: Some(11),
            }
        );
    }

    #[test]
    fn durable_flush_persists_records_to_real_segment() {
        let path = test_wal_path("durable-flush");
        let mut wal = WalBuffer::with_durable_segment(&path);
        assert!(wal.is_durable());
        assert_eq!(wal.durable_segment_path(), Some(path.as_path()));

        wal.append(WalRecord {
            txn_id: 1,
            payload: b"CREATE TABLE t (id INT)".to_vec().into(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"INSERT INTO t (id) VALUES (1)".to_vec().into(),
        });
        // Nothing on disk until the flush.
        assert!(!path.exists());

        wal.flush_all().unwrap();
        assert_eq!(wal.flushed_count(), 2);

        // The flushed records are now a real, CRC-checked, fsynced segment.
        let recovered = read_wal_segment(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].txn_id, 1);
        assert_eq!(
            &recovered[1].payload[..],
            &b"INSERT INTO t (id) VALUES (1)"[..]
        );
    }

    #[test]
    fn durable_flush_preserves_history_across_flushes() {
        let path = test_wal_path("durable-history");
        let mut wal = WalBuffer::with_durable_segment(&path);

        wal.append(WalRecord {
            txn_id: 1,
            payload: b"one".to_vec().into(),
        });
        wal.flush_all().unwrap();
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"two".to_vec().into(),
        });
        wal.flush_all().unwrap();

        // The segment must contain BOTH records after the second flush (appended, not clobbered).
        let recovered = read_wal_segment(&path).unwrap();
        remove_segment_files(&path);
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].txn_id, 1);
        assert_eq!(recovered[1].txn_id, 2);
    }

    fn remove_segment_files(path: &Path) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(wal_tail_offset_path(path));
    }

    #[test]
    fn durable_flush_appends_only_the_unflushed_tail() {
        // D1: the k-th commit costs O(its own bytes), not O(all bytes ever written). The already-
        // durable prefix must be byte-identical after later flushes, and each flush must grow the
        // file by exactly the new records' serialized size.
        let path = test_wal_path("durable-append-only");
        let mut wal = WalBuffer::with_durable_segment(&path);

        let first = WalRecord {
            txn_id: 1,
            payload: b"a large first record payload".to_vec().into(),
        };
        wal.append(first.clone());
        wal.flush_all().unwrap();
        // W4a: the physical file is PREALLOCATED (zero tail); the logical watermark is the
        // durable length, and the logical prefix must stay byte-identical across flushes.
        let first_logical = WAL_SEGMENT_MAGIC.len() as u64 + encoded_record_len(&first);
        assert_eq!(wal.durable_segment_bytes(), first_logical);
        let after_first = fs::read(&path).unwrap()[..first_logical as usize].to_vec();

        let second = WalRecord {
            txn_id: 2,
            payload: b"b".to_vec().into(),
        };
        wal.append(second.clone());
        wal.flush_all().unwrap();
        let second_logical = first_logical + encoded_record_len(&second);
        assert_eq!(wal.durable_segment_bytes(), second_logical);
        let after_second = fs::read(&path).unwrap()[..second_logical as usize].to_vec();
        remove_segment_files(&path);
        assert_eq!(&after_second[..after_first.len()], &after_first[..]);
        // The zero tail past the watermark reads back as clean end-of-log.
    }

    #[test]
    fn recover_truncates_torn_tail_beyond_recorded_offset_and_appends_continue() {
        // A crash mid-append leaves a torn record BEYOND the recorded durable tail: recovery
        // truncates it (that commit was never acknowledged) and the segment keeps accepting
        // appends at the valid boundary.
        let path = test_wal_path("recover-torn-tail");
        {
            let mut wal = WalBuffer::with_durable_segment(&path);
            for txn_id in 1..=2 {
                wal.append(WalRecord {
                    txn_id,
                    payload: format!("record {txn_id}").into_bytes().into(),
                });
                wal.flush_all().unwrap();
            }
            // Drop records the durable tail offset (clean shutdown).
        }
        // W4a: the physical file carries the preallocated zero tail; the LOGICAL valid length
        // is what recovery must report. Plant the torn garbage AT the logical tail (a real torn
        // append writes positionally there), overwriting the first zero bytes.
        let valid_bytes = recover_wal_segment(&path).unwrap().valid_bytes;
        {
            use std::os::unix::fs::FileExt;
            let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.write_all_at(&[0xAB; 17], valid_bytes).unwrap();
        }

        let recovery = recover_wal_segment(&path).unwrap();
        assert_eq!(recovery.records.len(), 2);
        assert_eq!(recovery.valid_bytes, valid_bytes);
        // Discarded = the torn garbage plus the preallocated zero tail behind it.
        assert!(recovery.discarded_torn_bytes >= 17);

        let mut wal =
            WalBuffer::with_recovered_durable_segment(&path, recovery.records.clone(), &recovery)
                .unwrap();
        assert_eq!(wal.flushed_count(), 2);
        wal.append(WalRecord {
            txn_id: 3,
            payload: b"post-recovery".to_vec().into(),
        });
        wal.flush_all().unwrap();
        drop(wal);

        // The torn bytes are gone and the post-recovery append reads back strictly.
        let records = read_wal_segment(&path).unwrap();
        remove_segment_files(&path);
        assert_eq!(
            records.iter().map(|r| r.txn_id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn recover_rejects_corruption_below_recorded_tail_offset() {
        // Damage to a record BELOW the recorded durable tail is bit rot of acknowledged data —
        // recovery must fail loudly, never silently truncate it away.
        let path = test_wal_path("recover-bit-rot");
        {
            let mut wal = WalBuffer::with_durable_segment(&path);
            for txn_id in 1..=2 {
                wal.append(WalRecord {
                    txn_id,
                    payload: format!("record {txn_id}").into_bytes().into(),
                });
                wal.flush_all().unwrap();
            }
        }
        // Corrupt the FIRST record's payload (well below the recorded tail).
        let mut bytes = fs::read(&path).unwrap();
        bytes[WAL_SEGMENT_MAGIC.len() + WAL_RECORD_HEADER_LEN] ^= 0x01;
        fs::write(&path, bytes).unwrap();

        let err = recover_wal_segment(&path).unwrap_err();
        remove_segment_files(&path);
        assert!(
            err.to_string().contains("checksum mismatch"),
            "expected loud CRC failure for acknowledged-durable corruption, got {err}"
        );
    }

    #[test]
    fn recover_rejects_segment_ending_short_of_recorded_tail() {
        // A segment that ends CLEANLY before the recorded durable tail (external truncation, a
        // lost file) has lost acknowledged records — loud failure, not a silent fresh database.
        let path = test_wal_path("recover-short");
        {
            let mut wal = WalBuffer::with_durable_segment(&path);
            wal.append(WalRecord {
                txn_id: 1,
                payload: b"acknowledged".to_vec().into(),
            });
            wal.flush_all().unwrap();
        }
        fs::remove_file(&path).unwrap();
        let err = recover_wal_segment(&path).unwrap_err();
        let _ = fs::remove_file(wal_tail_offset_path(&path));
        assert!(
            err.to_string().contains("before the recorded durable tail"),
            "expected loud short-segment failure, got {err}"
        );
    }

    #[test]
    fn recover_without_sidecar_tolerates_any_trailing_invalid_region() {
        // With no recorded tail offset (sidecar lost), the whole trailing invalid region is
        // treated as torn — the safe direction for an advisory lower bound.
        let path = test_wal_path("recover-no-sidecar");
        {
            let mut wal = WalBuffer::with_durable_segment(&path);
            wal.append(WalRecord {
                txn_id: 1,
                payload: b"kept".to_vec().into(),
            });
            wal.flush_all().unwrap();
        }
        // W4a: plant the garbage at the LOGICAL tail (inside the preallocated region).
        let valid_bytes = recover_wal_segment(&path).unwrap().valid_bytes;
        {
            use std::os::unix::fs::FileExt;
            let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.write_all_at(&[0xCD; 5], valid_bytes).unwrap();
        }
        fs::remove_file(wal_tail_offset_path(&path)).unwrap();

        let recovery = recover_wal_segment(&path).unwrap();
        remove_segment_files(&path);
        assert_eq!(recovery.records.len(), 1);
        assert_eq!(recovery.valid_bytes, valid_bytes);
        assert!(recovery.discarded_torn_bytes >= 5);
    }

    #[test]
    fn w4a_rotation_then_frontier_crossing_group_survives_reopen() {
        // AUDIT 9d6e9f96 BLOCKER regression: after a checkpoint prefix-truncation (rotation),
        // the reopened handle must write POSITIONALLY (not O_APPEND, whose pwrite ignores the
        // offset on Linux) and the preallocation frontier must be re-established — a
        // frontier-crossing group after rotation previously stranded acknowledged records
        // behind a zero hole (loud startup rejection after clean shutdown; silent loss after
        // a crash).
        let path = test_wal_path("w4a-rotation-crossing");
        {
            let mut wal = WalBuffer::with_durable_segment(&path);
            for txn_id in 1..=3 {
                wal.append(WalRecord {
                    txn_id,
                    payload: format!("pre-rotation {txn_id}").into_bytes().into(),
                });
                wal.flush_all().unwrap();
            }
            // Rotate: records [0,2) move to a (simulated) checkpoint; the live file keeps [2,3).
            wal.truncate_durable_segment_prefix(2).unwrap();
            // A group LARGER than the preallocation chunk forces the extension arm on the
            // post-rotation handle — the exact interaction the blocker corrupted.
            let big = vec![0xBB_u8; (wal_prealloc_chunk_bytes() + 256 * 1024) as usize];
            wal.append(WalRecord {
                txn_id: 4,
                payload: big.into(),
            });
            wal.flush_all().unwrap();
            wal.append(WalRecord {
                txn_id: 5,
                payload: b"after crossing".to_vec().into(),
            });
            wal.flush_all().unwrap();
            // Drop = clean shutdown (records the tail-offset sidecar).
        }
        // Reopen: every post-rotation record must be present and the segment clean.
        let recovery = recover_wal_segment(&path).unwrap();
        remove_segment_files(&path);
        assert_eq!(recovery.discarded_torn_bytes, 0);
        let txns: Vec<u64> = recovery.records.iter().map(|r| r.txn_id).collect();
        assert_eq!(txns, vec![3, 4, 5]);
    }

    #[test]
    fn w4a_zero_header_is_never_a_valid_record() {
        // The preallocated zero tail is end-of-log to BOTH readers only because an all-zero
        // 24-byte header is unrepresentable: a real record with txn_id=0 and payload_len=0
        // would carry the FNV checksum of the zero header, which must never itself be 0.
        assert_ne!(
            wal_record_checksum(0, 0, &[]),
            0,
            "FNV checksum of a zero header must be nonzero or zero-tail detection is unsound"
        );
    }

    #[test]
    fn w4a_preallocation_keeps_physical_size_stable_across_flushes() {
        // The whole point of W4a: per-flush fdatasync must not grow the file (size-change
        // journaling is what cost 3x on fdatasync latency).
        let path = test_wal_path("w4a-stable-size");
        let mut wal = WalBuffer::with_durable_segment(&path);
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"first".to_vec().into(),
        });
        wal.flush_all().unwrap();
        let physical_after_first = fs::metadata(&path).unwrap().len();
        for txn_id in 2..=50 {
            wal.append(WalRecord {
                txn_id,
                payload: format!("record {txn_id}").into_bytes().into(),
            });
            wal.flush_all().unwrap();
        }
        let physical_after_fifty = fs::metadata(&path).unwrap().len();
        assert_eq!(
            physical_after_first, physical_after_fifty,
            "flushes inside the preallocated window must not change the physical size"
        );
        drop(wal);
        // Clean recovery: the zero tail reads as a clean end (no torn bytes).
        let recovery = recover_wal_segment(&path).unwrap();
        remove_segment_files(&path);
        assert_eq!(recovery.records.len(), 50);
        assert_eq!(recovery.discarded_torn_bytes, 0);
    }

    #[test]
    fn w4a_growth_crosses_the_prealloc_chunk_inline_flush() {
        // A record larger than the preallocation chunk forces the INLINE flush's extension arm;
        // everything must stay readable and recoverable across the boundary.
        let path = test_wal_path("w4a-growth-inline");
        let mut wal = WalBuffer::with_durable_segment(&path);
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"small before growth".to_vec().into(),
        });
        wal.flush_all().unwrap();
        let big = vec![0xEE_u8; (wal_prealloc_chunk_bytes() + 1024 * 1024) as usize];
        wal.append(WalRecord {
            txn_id: 2,
            payload: big.clone().into(),
        });
        wal.flush_all().unwrap();
        wal.append(WalRecord {
            txn_id: 3,
            payload: b"small after growth".to_vec().into(),
        });
        wal.flush_all().unwrap();
        drop(wal);
        let recovery = recover_wal_segment(&path).unwrap();
        remove_segment_files(&path);
        assert_eq!(recovery.records.len(), 3);
        assert_eq!(recovery.discarded_torn_bytes, 0);
        assert_eq!(&recovery.records[1].payload[..], &big[..]);
    }

    #[test]
    fn w4a_growth_crosses_the_prealloc_chunk_group_job() {
        // Same boundary crossing through the GROUP flush job (the concurrent path's flusher).
        let path = test_wal_path("w4a-growth-job");
        let mut wal = WalBuffer::with_durable_segment(&path);
        let big = vec![0xDD_u8; (wal_prealloc_chunk_bytes() + 512 * 1024) as usize];
        wal.append(WalRecord {
            txn_id: 1,
            payload: big.clone().into(),
        });
        let begun = wal.begin_group_flush().unwrap();
        let flushed = match begun {
            WalGroupFlushBegin::Job(job) => job.commit().unwrap(),
            WalGroupFlushBegin::Clean { flushed_records } => flushed_records,
        };
        assert_eq!(flushed, 1);
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"after job growth".to_vec().into(),
        });
        wal.flush_all().unwrap();
        drop(wal);
        let recovery = recover_wal_segment(&path).unwrap();
        remove_segment_files(&path);
        assert_eq!(recovery.records.len(), 2);
        assert_eq!(recovery.discarded_torn_bytes, 0);
        assert_eq!(&recovery.records[0].payload[..], &big[..]);
    }

    #[test]
    fn truncate_durable_segment_prefix_bounds_live_file_and_keeps_appending() {
        // D2: after a checkpoint, the live segment drops the checkpointed prefix; logical
        // counters are unchanged and appends continue against the trimmed file.
        let path = test_wal_path("truncate-prefix");
        let mut wal = WalBuffer::with_durable_segment(&path);
        for txn_id in 1..=3 {
            wal.append(WalRecord {
                txn_id,
                payload: format!("record {txn_id}").into_bytes().into(),
            });
            wal.flush_all().unwrap();
        }
        let full_bytes = wal.durable_segment_bytes();

        wal.truncate_durable_segment_prefix(3).unwrap();
        assert_eq!(wal.durable_segment_bytes(), WAL_SEGMENT_MAGIC.len() as u64);
        assert!(wal.durable_segment_bytes() < full_bytes);
        assert_eq!(wal.durable_segment_base_records(), 3);
        // Logical counters are untouched — only the FILE was trimmed.
        assert_eq!(wal.len(), 3);
        assert_eq!(wal.flushed_count(), 3);
        assert_eq!(wal.flushed_records().len(), 3);

        wal.append(WalRecord {
            txn_id: 4,
            payload: b"post-checkpoint".to_vec().into(),
        });
        wal.flush_all().unwrap();
        drop(wal);

        let live = read_wal_segment(&path).unwrap();
        remove_segment_files(&path);
        assert_eq!(
            live.iter().map(|r| r.txn_id).collect::<Vec<_>>(),
            vec![4],
            "the live segment holds only post-checkpoint records"
        );
    }

    #[test]
    fn durable_group_commit_stats_count_one_group_per_fsync() {
        let path = test_wal_path("durable-groups");
        let mut wal = WalBuffer::with_durable_segment(&path);

        // Two records, then ONE flush => a single group of size 2.
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"a".to_vec().into(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"b".to_vec().into(),
        });
        wal.flush_all().unwrap();
        // One more record, separate flush => a second group of size 1.
        wal.append(WalRecord {
            txn_id: 3,
            payload: b"c".to_vec().into(),
        });
        wal.flush_all().unwrap();
        // A flush with nothing new must NOT count as a group (no fsync performed).
        wal.flush_all().unwrap();

        let _ = fs::remove_file(&path);
        let stats = wal.group_commit_stats();
        assert_eq!(stats.flush_groups, 2);
        assert_eq!(stats.durable_records, 3);
        assert_eq!(stats.max_group_size, 2);
        assert!((stats.mean_group_size() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn durable_flush_failure_leaves_no_durable_advance() {
        let path = test_wal_path("durable-fail");
        let mut wal = WalBuffer::with_durable_segment(&path);
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"a".to_vec().into(),
        });

        wal.fail_next_flush();
        let err = wal.flush_all().unwrap_err();
        assert!(matches!(err, EngineError::Durability(_)));
        // The watermark did not advance and (because the simulated failure short-circuits before
        // any I/O) the segment was never created — nothing partially durable.
        assert_eq!(wal.flushed_count(), 0);
        assert!(!path.exists());

        // A subsequent successful flush makes the record durable.
        wal.flush_all().unwrap();
        let recovered = read_wal_segment(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(recovered.len(), 1);
        assert_eq!(wal.flushed_count(), 1);
    }

    #[test]
    fn in_memory_flush_writes_no_segment() {
        // The default buffer is in-memory only: flush advances the watermark but touches no disk.
        let mut wal = WalBuffer::new();
        assert!(!wal.is_durable());
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"a".to_vec().into(),
        });
        wal.flush_all().unwrap();
        assert_eq!(wal.flushed_count(), 1);
        assert_eq!(wal.group_commit_stats(), WalGroupCommitStats::default());
    }

    #[test]
    fn reinstate_durable_records_seeds_flushed_prefix() {
        let mut wal = WalBuffer::new();
        wal.reinstate_durable_records(vec![
            WalRecord {
                txn_id: 1,
                payload: b"a".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"b".to_vec().into(),
            },
        ]);
        assert_eq!(wal.len(), 2);
        assert_eq!(wal.flushed_count(), 2);
        assert_eq!(wal.unflushed_count(), 0);
    }

    #[test]
    fn wal_segment_round_trips_durable_records() {
        let path = test_wal_path("roundtrip");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                    .to_vec()
                    .into(),
            },
        ];

        write_wal_segment(&path, &records).unwrap();
        let recovered = read_wal_segment(&path).unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].txn_id, 1);
        assert_eq!(recovered[0].payload, records[0].payload);
        assert_eq!(recovered[1].txn_id, 2);
        assert_eq!(recovered[1].payload, records[1].payload);
    }

    #[test]
    fn wal_segment_rejects_checksum_mismatch() {
        let path = test_wal_path("checksum");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        }];

        write_wal_segment(&path, &records).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.last_mut().unwrap();
        *last ^= 0x01;
        fs::write(&path, bytes).unwrap();

        let err = read_wal_segment(&path).unwrap_err();
        let _ = fs::remove_file(path);

        assert!(err.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn wal_segment_rejects_truncated_record_header() {
        let path = test_wal_path("truncated");
        fs::write(
            &path,
            [WAL_SEGMENT_MAGIC.as_slice(), &[1_u8, 2, 3]].concat(),
        )
        .unwrap();

        let err = read_wal_segment(&path).unwrap_err();
        let _ = fs::remove_file(path);

        assert!(err.to_string().contains("record header"));
    }

    #[test]
    fn wal_control_file_round_trips_checkpoint_metadata() {
        let control_path = test_wal_path("control").with_extension("control");
        let control = WalControlFile {
            segment_path: PathBuf::from("segment-0001.wal"),
            checkpoint: WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(42),
            },
        };

        write_wal_control_file(&control_path, &control).unwrap();
        let recovered = read_wal_control_file(&control_path).unwrap();
        let _ = fs::remove_file(control_path);

        assert_eq!(recovered, control);
    }

    #[test]
    fn wal_checkpoint_reads_segment_named_by_control_file() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-checkpoint-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let control_path = dir.join("CONTROL");
        let segment_path = dir.join("segment-0001.wal");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
        ];
        let control = WalControlFile {
            segment_path: PathBuf::from("segment-0001.wal"),
            checkpoint: WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(2),
            },
        };

        write_wal_segment(&segment_path, &records).unwrap();
        write_wal_control_file(&control_path, &control).unwrap();
        let (recovered_control, recovered_records) = read_wal_checkpoint(&control_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(recovered_control, control);
        assert_eq!(recovered_records.len(), 2);
        assert_eq!(&recovered_records[1].payload[..], &b"SET b=2"[..]);
    }

    #[test]
    fn wal_checkpoint_rejects_control_record_count_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-checkpoint-mismatch-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let control_path = dir.join("CONTROL");
        let segment_path = dir.join("segment-0001.wal");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        }];
        let control = WalControlFile {
            segment_path: PathBuf::from("segment-0001.wal"),
            checkpoint: WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(1),
            },
        };

        write_wal_segment(&segment_path, &records).unwrap();
        write_wal_control_file(&control_path, &control).unwrap();
        let err = read_wal_checkpoint(&control_path).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("expected 2 durable records"));
    }

    #[test]
    fn wal_archive_round_trips_ordered_segments() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
        ];

        let manifest = write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
        let (recovered_manifest, recovered_records) = read_wal_archive(&manifest_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(recovered_manifest, manifest);
        assert_eq!(recovered_manifest.segments.len(), 2);
        assert_eq!(
            recovered_manifest.checkpoint,
            WalCheckpointMeta {
                durable_record_count: 3,
                last_durable_txn_id: Some(3),
            }
        );
        assert_eq!(recovered_records.len(), 3);
        assert_eq!(&recovered_records[2].payload[..], &b"SET c=3"[..]);
    }

    #[test]
    fn wal_archive_object_backup_exports_and_restores_archive() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-object-backup-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("source").join("MANIFEST");
        let segment_dir = dir.join("source").join("segments");
        let backup_path = dir.join("backup").join("BACKUP");
        let object_dir = dir.join("backup").join("objects");
        let restored_manifest_path = dir.join("restored").join("MANIFEST");
        let restored_segment_dir = dir.join("restored").join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
        ];

        let source_manifest = write_wal_archive_with_timestamps(
            &manifest_path,
            &segment_dir,
            &records,
            2,
            &timestamps,
        )
        .unwrap();
        let backup =
            export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
        let restored_manifest = restore_wal_archive_object_backup(
            &backup_path,
            &restored_manifest_path,
            &restored_segment_dir,
        )
        .unwrap();
        let (_validated_manifest, restored_records) =
            read_wal_archive(&restored_manifest_path).unwrap();
        let (_timestamp_manifest, target, timestamp_records) =
            read_wal_archive_to_timestamp_micros(&restored_manifest_path, 2_000).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(backup.archive_manifest, source_manifest);
        assert_eq!(backup.objects.len(), 3);
        assert_eq!(restored_manifest.checkpoint, source_manifest.checkpoint);
        assert_eq!(restored_manifest.record_timestamps, timestamps);
        assert_eq!(restored_records, records);
        assert_eq!(target.target_txn_id, 2);
        assert_eq!(timestamp_records.len(), 2);
    }

    #[test]
    fn wal_archive_object_backup_rejects_corrupt_object_before_manifest_install() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-object-backup-corrupt-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("source").join("MANIFEST");
        let segment_dir = dir.join("source").join("segments");
        let backup_path = dir.join("backup").join("BACKUP");
        let object_dir = dir.join("backup").join("objects");
        let restored_manifest_path = dir.join("restored").join("MANIFEST");
        let restored_segment_dir = dir.join("restored").join("segments");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        }];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let backup =
            export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
        let segment_object = backup
            .objects
            .iter()
            .find(|object| object.source_path != Path::new("MANIFEST"))
            .unwrap();
        let object_path = resolve_manifest_path(&backup_path, &segment_object.object_path);
        fs::write(&object_path, b"corrupt wal object").unwrap();

        let err = restore_wal_archive_object_backup(
            &backup_path,
            &restored_manifest_path,
            &restored_segment_dir,
        )
        .unwrap_err();
        let manifest_installed = restored_manifest_path.exists();
        let _ = fs::remove_dir_all(dir);

        let err = err.to_string();
        assert!(
            err.contains("checksum mismatch")
                || (err.contains("expected") && err.contains("bytes"))
        );
        assert!(!manifest_installed);
    }

    #[test]
    fn wal_archive_object_backup_rejects_late_corrupt_object_before_segment_install() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-object-backup-late-corrupt-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("source").join("MANIFEST");
        let segment_dir = dir.join("source").join("segments");
        let backup_path = dir.join("backup").join("BACKUP");
        let object_dir = dir.join("backup").join("objects");
        let restored_manifest_path = dir.join("restored").join("MANIFEST");
        let restored_segment_dir = dir.join("restored").join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let backup =
            export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
        let second_segment_object = backup
            .objects
            .iter()
            .filter(|object| object.source_path != Path::new("MANIFEST"))
            .nth(1)
            .unwrap();
        let object_path = resolve_manifest_path(&backup_path, &second_segment_object.object_path);
        fs::write(&object_path, b"late corrupt wal object").unwrap();

        let err = restore_wal_archive_object_backup(
            &backup_path,
            &restored_manifest_path,
            &restored_segment_dir,
        )
        .unwrap_err();
        let manifest_installed = restored_manifest_path.exists();
        let segment_dir_installed = restored_segment_dir.exists();
        let staging_segment_dir_installed = restored_segment_dir
            .parent()
            .unwrap()
            .join(format!(
                ".{}.restore-{}",
                restored_segment_dir.file_name().unwrap().to_string_lossy(),
                std::process::id()
            ))
            .exists();
        let _ = fs::remove_dir_all(dir);

        let err = err.to_string();
        assert!(
            err.contains("checksum mismatch")
                || (err.contains("expected") && err.contains("bytes"))
        );
        assert!(!manifest_installed);
        assert!(!segment_dir_installed);
        assert!(!staging_segment_dir_installed);
    }

    #[test]
    fn wal_archive_object_backup_rejects_manifest_metadata_drift_before_install() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-object-backup-manifest-drift-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("source").join("MANIFEST");
        let segment_dir = dir.join("source").join("segments");
        let backup_path = dir.join("backup").join("BACKUP");
        let object_dir = dir.join("backup").join("objects");
        let restored_manifest_path = dir.join("restored").join("MANIFEST");
        let restored_segment_dir = dir.join("restored").join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        export_wal_archive_object_backup(&manifest_path, &backup_path, &object_dir).unwrap();
        let backup_body = fs::read_to_string(&backup_path).unwrap();
        fs::write(
            &backup_path,
            backup_body.replace("record_timestamp=2|2000", "record_timestamp=2|2500"),
        )
        .unwrap();

        let err = restore_wal_archive_object_backup(
            &backup_path,
            &restored_manifest_path,
            &restored_segment_dir,
        )
        .unwrap_err();
        let manifest_installed = restored_manifest_path.exists();
        let segment_dir_installed = restored_segment_dir.exists();
        let _ = fs::remove_dir_all(dir);

        let err = err.to_string();
        assert!(err.contains("manifest object does not match backup manifest metadata"));
        assert!(!manifest_installed);
        assert!(!segment_dir_installed);
    }

    #[test]
    fn wal_archive_reads_prefix_to_transaction_target() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-target-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
        let (_manifest, target, recovered_records) =
            read_wal_archive_to_txn(&manifest_path, 2).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(
            target,
            WalArchiveRecoveryTarget {
                target_txn_id: 2,
                recovered_record_count: 2,
                last_recovered_txn_id: 2,
            }
        );
        assert_eq!(recovered_records.len(), 2);
        assert_eq!(&recovered_records[1].payload[..], &b"SET b=2"[..]);
    }

    #[test]
    fn wal_archive_target_rejects_before_first_transaction() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-target-before-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![WalRecord {
            txn_id: 10,
            payload: b"SET a=1".to_vec().into(),
        }];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let err = read_wal_archive_to_txn(&manifest_path, 9).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err
            .to_string()
            .contains("before first archived transaction"));
    }

    #[test]
    fn wal_archive_target_rejects_beyond_durable_archive() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-target-beyond-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        }];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let err = read_wal_archive_to_txn(&manifest_path, 2).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("beyond last durable transaction"));
    }

    #[test]
    fn wal_archive_target_rejects_missing_transaction_boundary() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-target-missing-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let err = read_wal_archive_to_txn(&manifest_path, 2).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err
            .to_string()
            .contains("does not contain target transaction"));
    }

    #[test]
    fn wal_archive_reads_prefix_to_timestamp_target() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 2, &timestamps)
            .unwrap();
        let (_manifest, target, recovered_records) =
            read_wal_archive_to_timestamp_micros(&manifest_path, 2_000).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(
            target,
            WalArchiveTimestampRecoveryTarget {
                target_timestamp_micros: 2_000,
                target_txn_id: 2,
                recovered_record_count: 2,
                last_recovered_txn_id: 2,
            }
        );
        assert_eq!(recovered_records.len(), 2);
        assert_eq!(&recovered_records[1].payload[..], &b"SET b=2"[..]);
    }

    #[test]
    fn wal_archive_timestamp_target_rejects_missing_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-missing-meta-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        }];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let err = read_wal_archive_to_timestamp_micros(&manifest_path, 1_000).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("no timestamp metadata"));
    }

    #[test]
    fn wal_archive_timestamp_target_rejects_unavailable_boundaries() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-unavailable-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 10,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 20,
                payload: b"SET b=2".to_vec().into(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 10,
                timestamp_micros: 10_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 20,
                timestamp_micros: 20_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let before = read_wal_archive_to_timestamp_micros(&manifest_path, 9_999).unwrap_err();
        let between = read_wal_archive_to_timestamp_micros(&manifest_path, 15_000).unwrap_err();
        let beyond = read_wal_archive_to_timestamp_micros(&manifest_path, 20_001).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(before
            .to_string()
            .contains("before first archived timestamp"));
        assert!(between
            .to_string()
            .contains("falls between archived transaction boundaries"));
        assert!(beyond.to_string().contains("beyond last durable timestamp"));
    }

    #[test]
    fn wal_archive_timestamp_target_rejects_ambiguous_boundary() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-ambiguous-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 1_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let err = read_wal_archive_to_timestamp_micros(&manifest_path, 1_000).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("ambiguous"));
    }

    #[test]
    fn wal_archive_ingests_next_segment_and_preserves_timestamps() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-ingest-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let ingest_segment = segment_dir.join("segment-0002.wal");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
        ];
        let ingest_records = vec![
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec().into(),
            },
        ];
        let ingest_timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 4,
                timestamp_micros: 4_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 2, &timestamps)
            .unwrap();
        write_wal_segment(&ingest_segment, &ingest_records).unwrap();
        let manifest = append_wal_archive_segment_with_timestamps(
            &manifest_path,
            &ingest_segment,
            &ingest_timestamps,
        )
        .unwrap();
        let (_read_manifest, read_records) = read_wal_archive(&manifest_path).unwrap();
        let (_manifest, target, target_records) =
            read_wal_archive_to_timestamp_micros(&manifest_path, 4_000).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(manifest.segments.len(), 2);
        assert_eq!(manifest.checkpoint.durable_record_count, 4);
        assert_eq!(manifest.checkpoint.last_durable_txn_id, Some(4));
        assert_eq!(manifest.record_timestamps.len(), 4);
        assert_eq!(read_records.len(), 4);
        assert_eq!(&read_records[3].payload[..], &b"SET d=4"[..]);
        assert_eq!(target.target_txn_id, 4);
        assert_eq!(target_records.len(), 4);
    }

    #[test]
    fn wal_archive_ingest_rejects_non_increasing_segment_without_manifest_mutation() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-ingest-reject-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let ingest_segment = segment_dir.join("segment-0002.wal");
        let records = vec![WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        }];
        let ingest_records = vec![WalRecord {
            txn_id: 2,
            payload: b"SET duplicate=2".to_vec().into(),
        }];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let before_manifest = fs::read_to_string(&manifest_path).unwrap();
        write_wal_segment(&ingest_segment, &ingest_records).unwrap();
        let err = append_wal_archive_segment(&manifest_path, &ingest_segment).unwrap_err();
        let after_manifest = fs::read_to_string(&manifest_path).unwrap();
        let (_manifest, read_records) = read_wal_archive(&manifest_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("not after durable transaction 2"));
        assert_eq!(after_manifest, before_manifest);
        assert_eq!(read_records.len(), 1);
        assert_eq!(&read_records[0].payload[..], &b"SET b=2"[..]);
    }

    #[test]
    fn wal_archive_ingest_requires_timestamp_metadata_when_archive_has_timestamps() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-ingest-timestamp-required-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let ingest_segment = segment_dir.join("segment-0002.wal");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        }];
        let timestamps = vec![WalArchiveRecordTimestamp {
            txn_id: 1,
            timestamp_micros: 1_000,
        }];
        let ingest_records = vec![WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec().into(),
        }];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let before_manifest = fs::read_to_string(&manifest_path).unwrap();
        write_wal_segment(&ingest_segment, &ingest_records).unwrap();
        let err = append_wal_archive_segment(&manifest_path, &ingest_segment).unwrap_err();
        let after_manifest = fs::read_to_string(&manifest_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert!(err
            .to_string()
            .contains("requires timestamp metadata for ingested segment"));
        assert_eq!(after_manifest, before_manifest);
    }

    #[test]
    fn wal_archive_forks_transaction_timeline_with_ancestry() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-txn-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let branch_manifest = dir.join("branch").join("MANIFEST");
        let branch_segments = dir.join("branch").join("segments");
        let timeline_path = dir.join("branch").join("TIMELINE");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                    .to_vec()
                    .into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                    .to_vec()
                    .into(),
            },
        ];
        write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();

        let branch = fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &timeline_path,
            "timeline-0002",
            Some("timeline-0001"),
            2,
        )
        .unwrap();
        let timeline = read_wal_archive_timeline(&timeline_path).unwrap();
        let (_manifest, branch_records) = read_wal_archive(&branch_manifest).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(branch.timeline, timeline);
        assert_eq!(timeline.timeline_id, "timeline-0002");
        assert_eq!(
            timeline.parent_timeline_id.as_deref(),
            Some("timeline-0001")
        );
        assert_eq!(timeline.fork_txn_id, 2);
        assert_eq!(timeline.fork_timestamp_micros, None);
        assert_eq!(branch.manifest.checkpoint.durable_record_count, 2);
        assert_eq!(branch.manifest.checkpoint.last_durable_txn_id, Some(2));
        assert_eq!(branch_records.len(), 2);
        assert_eq!(branch_records[1].txn_id, 2);
    }

    #[test]
    fn wal_archive_forks_timestamp_timeline_and_rejects_self_parent_without_mutation() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-timestamp-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let branch_manifest = dir.join("branch").join("MANIFEST");
        let branch_segments = dir.join("branch").join("segments");
        let timeline_path = dir.join("branch").join("TIMELINE");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                    .to_vec()
                    .into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                    .to_vec()
                    .into(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
        ];
        write_wal_archive_with_timestamps(
            &source_manifest,
            &source_segments,
            &records,
            2,
            &timestamps,
        )
        .unwrap();

        let self_parent_err = fork_wal_archive_timeline_to_timestamp_micros(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &timeline_path,
            "timeline-0002",
            Some("timeline-0002"),
            2_000,
        )
        .unwrap_err();
        assert!(!branch_manifest.exists());
        assert!(self_parent_err
            .to_string()
            .contains("cannot be its own parent"));

        let branch = fork_wal_archive_timeline_to_timestamp_micros(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &timeline_path,
            "timeline-0002",
            Some("timeline-0001"),
            2_000,
        )
        .unwrap();
        let timeline = read_wal_archive_timeline(&timeline_path).unwrap();
        let (_manifest, branch_records) = read_wal_archive(&branch_manifest).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(timeline.fork_txn_id, 2);
        assert_eq!(timeline.fork_timestamp_micros, Some(2_000));
        assert_eq!(branch.manifest.record_timestamps.len(), 2);
        assert_eq!(branch_records.len(), 2);
        assert_eq!(branch.timeline, timeline);
    }

    #[test]
    fn wal_archive_timeline_registry_requires_parent_before_child_and_unique_ids() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-registry-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let root_timeline_path = dir.join("source").join("TIMELINE");
        let branch_manifest = dir.join("branch").join("MANIFEST");
        let branch_segments = dir.join("branch").join("segments");
        let branch_timeline_path = dir.join("branch").join("TIMELINE");
        let missing_parent_branch_manifest = dir.join("missing-parent").join("MANIFEST");
        let missing_parent_branch_segments = dir.join("missing-parent").join("segments");
        let missing_parent_timeline_path = dir.join("missing-parent").join("TIMELINE");
        let registry_path = dir.join("TIMELINE_REGISTRY");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                    .to_vec()
                    .into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                    .to_vec()
                    .into(),
            },
        ];
        write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
        write_wal_archive_timeline(
            &root_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-0001".to_string(),
                parent_timeline_id: None,
                fork_txn_id: 0,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: source_manifest.clone(),
            },
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &branch_timeline_path,
            "timeline-0002",
            Some("timeline-0001"),
            2,
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &missing_parent_branch_manifest,
            &missing_parent_branch_segments,
            &missing_parent_timeline_path,
            "timeline-0003",
            Some("timeline-missing"),
            2,
        )
        .unwrap();

        let missing_parent_err =
            register_wal_archive_timeline(&registry_path, &missing_parent_timeline_path)
                .unwrap_err();
        assert!(missing_parent_err
            .to_string()
            .contains("missing parent timeline timeline-missing"));
        assert!(!registry_path.exists());

        let root_registry =
            register_wal_archive_timeline(&registry_path, &root_timeline_path).unwrap();
        assert_eq!(root_registry.timelines.len(), 1);
        assert_eq!(root_registry.timelines[0].timeline_id, "timeline-0001");

        let registry =
            register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap();
        assert_eq!(registry.timelines.len(), 2);
        assert_eq!(registry.timelines[1].timeline_id, "timeline-0002");
        assert_eq!(
            registry.timelines[1].parent_timeline_id.as_deref(),
            Some("timeline-0001")
        );
        assert_eq!(registry.timelines[1].fork_txn_id, 2);

        let before_registry = fs::read_to_string(&registry_path).unwrap();
        let duplicate_err =
            register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap_err();
        let after_registry = fs::read_to_string(&registry_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert!(duplicate_err
            .to_string()
            .contains("already contains timeline timeline-0002"));
        assert_eq!(after_registry, before_registry);
    }

    #[test]
    fn wal_archive_timeline_registry_selects_validated_branch_and_rejects_stale_sidecar() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-select-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let root_timeline_path = dir.join("source").join("TIMELINE");
        let branch_manifest = dir.join("branch").join("MANIFEST");
        let branch_segments = dir.join("branch").join("segments");
        let branch_timeline_path = dir.join("branch").join("TIMELINE");
        let registry_path = dir.join("TIMELINE_REGISTRY");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                    .to_vec()
                    .into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                    .to_vec()
                    .into(),
            },
        ];
        write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
        write_wal_archive_timeline(
            &root_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-0001".to_string(),
                parent_timeline_id: None,
                fork_txn_id: 0,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: source_manifest.clone(),
            },
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &branch_timeline_path,
            "timeline-0002",
            Some("timeline-0001"),
            2,
        )
        .unwrap();
        register_wal_archive_timeline(&registry_path, &root_timeline_path).unwrap();
        register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap();

        let selection = select_wal_archive_timeline(&registry_path, "timeline-0002").unwrap();
        assert_eq!(selection.entry.timeline_id, "timeline-0002");
        assert_eq!(selection.timeline.fork_txn_id, 2);
        assert_eq!(selection.manifest.checkpoint.durable_record_count, 2);

        let missing_err =
            select_wal_archive_timeline(&registry_path, "timeline-missing").unwrap_err();
        assert!(missing_err
            .to_string()
            .contains("has no timeline timeline-missing"));

        write_wal_archive_timeline(
            &branch_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-0002".to_string(),
                parent_timeline_id: Some("timeline-0001".to_string()),
                fork_txn_id: 3,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: branch_manifest.clone(),
            },
        )
        .unwrap();
        let stale_err = select_wal_archive_timeline(&registry_path, "timeline-0002").unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(stale_err.to_string().contains("does not match sidecar"));
    }

    #[test]
    fn wal_archive_timeline_prune_keeps_target_ancestry_and_removes_unreferenced_artifacts() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-prune-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let source_timeline_path = dir.join("source").join("TIMELINE");
        let keep_manifest = dir.join("keep").join("MANIFEST");
        let keep_segments = dir.join("keep").join("segments");
        let keep_timeline_path = dir.join("keep").join("TIMELINE");
        let prune_manifest = dir.join("prune").join("MANIFEST");
        let prune_segments = dir.join("prune").join("segments");
        let prune_timeline_path = dir.join("prune").join("TIMELINE");
        let registry_path = dir.join("TIMELINE_REGISTRY");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                    .to_vec()
                    .into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"INSERT INTO people (id, name) VALUES (2, 'Grace')"
                    .to_vec()
                    .into(),
            },
        ];
        write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
        write_wal_archive_timeline(
            &source_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-main-0001".to_string(),
                parent_timeline_id: None,
                fork_txn_id: 0,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: source_manifest.clone(),
            },
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &keep_manifest,
            &keep_segments,
            &keep_timeline_path,
            "timeline-keep-0002",
            Some("timeline-main-0001"),
            3,
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &prune_manifest,
            &prune_segments,
            &prune_timeline_path,
            "timeline-prune-0003",
            Some("timeline-main-0001"),
            2,
        )
        .unwrap();
        register_wal_archive_timeline(&registry_path, &source_timeline_path).unwrap();
        register_wal_archive_timeline(&registry_path, &keep_timeline_path).unwrap();
        register_wal_archive_timeline(&registry_path, &prune_timeline_path).unwrap();

        let plan = plan_wal_archive_timeline_prune(&registry_path, "timeline-keep-0002").unwrap();
        let applied =
            apply_wal_archive_timeline_prune(&registry_path, "timeline-keep-0002").unwrap();
        let registry = read_wal_archive_timeline_registry(&registry_path).unwrap();
        let selection = select_wal_archive_timeline(&registry_path, "timeline-keep-0002").unwrap();
        let pruned_err =
            select_wal_archive_timeline(&registry_path, "timeline-prune-0003").unwrap_err();

        assert_eq!(plan.retained_timeline_id, "timeline-keep-0002");
        assert_eq!(
            plan.retained_timeline_ids,
            vec![
                "timeline-main-0001".to_string(),
                "timeline-keep-0002".to_string()
            ]
        );
        assert_eq!(
            plan.removed_timeline_ids,
            vec!["timeline-prune-0003".to_string()]
        );
        assert_eq!(applied, plan);
        assert_eq!(registry.timelines.len(), 2);
        assert_eq!(selection.manifest.checkpoint.durable_record_count, 3);
        assert!(!prune_timeline_path.exists());
        assert!(!prune_manifest.exists());
        assert!(!prune_segments.join("segment-0001.wal").exists());
        assert!(keep_timeline_path.exists());
        assert!(keep_manifest.exists());
        assert!(keep_segments.join("segment-0001.wal").exists());
        assert!(pruned_err
            .to_string()
            .contains("has no timeline timeline-prune-0003"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn wal_archive_timeline_prune_rejects_stale_sidecar_without_registry_mutation() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timeline-prune-stale-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let source_manifest = dir.join("source").join("MANIFEST");
        let source_segments = dir.join("source").join("segments");
        let source_timeline_path = dir.join("source").join("TIMELINE");
        let branch_manifest = dir.join("branch").join("MANIFEST");
        let branch_segments = dir.join("branch").join("segments");
        let branch_timeline_path = dir.join("branch").join("TIMELINE");
        let registry_path = dir.join("TIMELINE_REGISTRY");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"CREATE TABLE people (id INT, name TEXT)".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"INSERT INTO people (id, name) VALUES (1, 'Ada')"
                    .to_vec()
                    .into(),
            },
        ];
        write_wal_archive(&source_manifest, &source_segments, &records, 1).unwrap();
        write_wal_archive_timeline(
            &source_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-main-0001".to_string(),
                parent_timeline_id: None,
                fork_txn_id: 0,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: source_manifest.clone(),
            },
        )
        .unwrap();
        fork_wal_archive_timeline_to_txn(
            &source_manifest,
            &branch_manifest,
            &branch_segments,
            &branch_timeline_path,
            "timeline-branch-0002",
            Some("timeline-main-0001"),
            2,
        )
        .unwrap();
        register_wal_archive_timeline(&registry_path, &source_timeline_path).unwrap();
        register_wal_archive_timeline(&registry_path, &branch_timeline_path).unwrap();
        let before_registry = fs::read_to_string(&registry_path).unwrap();
        write_wal_archive_timeline(
            &branch_timeline_path,
            &WalArchiveTimeline {
                timeline_id: "timeline-branch-0002".to_string(),
                parent_timeline_id: Some("timeline-main-0001".to_string()),
                fork_txn_id: 1,
                fork_timestamp_micros: None,
                source_manifest_path: source_manifest.clone(),
                branch_manifest_path: branch_manifest.clone(),
            },
        )
        .unwrap();

        let err =
            apply_wal_archive_timeline_prune(&registry_path, "timeline-branch-0002").unwrap_err();
        let after_registry = fs::read_to_string(&registry_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("does not match sidecar"));
        assert_eq!(after_registry, before_registry);
    }

    #[test]
    fn wal_archive_retention_plan_keeps_exact_transaction_prefix() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-retention-plan-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec().into(),
            },
            WalRecord {
                txn_id: 5,
                payload: b"SET e=5".to_vec().into(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
        let plan = plan_wal_archive_retention_to_txn(&manifest_path, 3).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(plan.target_txn_id, 3);
        assert_eq!(plan.retained_record_count, 3);
        assert_eq!(plan.removed_record_count, 2);
        assert_eq!(plan.retained_manifest.segments.len(), 2);
        assert_eq!(
            plan.retained_manifest.checkpoint,
            WalCheckpointMeta {
                durable_record_count: 3,
                last_durable_txn_id: Some(3),
            }
        );
        assert_eq!(plan.retained_manifest.segments[1].record_count, 1);
        assert_eq!(plan.retained_manifest.segments[1].last_txn_id, Some(3));
        assert_eq!(
            plan.removed_segments,
            vec![segment_dir.join("segment-0003.wal")]
        );
    }

    #[test]
    fn wal_archive_retention_apply_rewrites_manifest_and_removes_tail_segments() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-retention-apply-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec().into(),
            },
            WalRecord {
                txn_id: 5,
                payload: b"SET e=5".to_vec().into(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 2).unwrap();
        let removed_tail = segment_dir.join("segment-0003.wal");
        assert!(removed_tail.exists());

        let plan = apply_wal_archive_retention_to_txn(&manifest_path, 3).unwrap();
        let (retained_manifest, retained_records) = read_wal_archive(&manifest_path).unwrap();
        let target_err = read_wal_archive_to_txn(&manifest_path, 4).unwrap_err();

        assert_eq!(plan.retained_record_count, 3);
        assert_eq!(plan.removed_record_count, 2);
        assert!(!removed_tail.exists());
        assert_eq!(retained_manifest.checkpoint.durable_record_count, 3);
        assert_eq!(retained_manifest.checkpoint.last_durable_txn_id, Some(3));
        assert_eq!(retained_records.len(), 3);
        assert_eq!(&retained_records[2].payload[..], &b"SET c=3"[..]);
        assert!(target_err
            .to_string()
            .contains("beyond last durable transaction"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn wal_archive_timestamp_retention_rewrites_manifest_and_preserves_timestamp_prefix() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-retention-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec().into(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 4,
                timestamp_micros: 4_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let removed_tail = segment_dir.join("segment-0004.wal");
        assert!(removed_tail.exists());

        let plan = apply_wal_archive_retention_to_timestamp_micros(&manifest_path, 3_000).unwrap();
        let (retained_manifest, retained_records) = read_wal_archive(&manifest_path).unwrap();
        let target_err = read_wal_archive_to_timestamp_micros(&manifest_path, 4_000).unwrap_err();

        assert_eq!(plan.target_txn_id, 3);
        assert_eq!(plan.retained_record_count, 3);
        assert_eq!(plan.removed_record_count, 1);
        assert!(!removed_tail.exists());
        assert_eq!(retained_manifest.checkpoint.durable_record_count, 3);
        assert_eq!(retained_manifest.checkpoint.last_durable_txn_id, Some(3));
        assert_eq!(
            retained_manifest.record_timestamps,
            vec![
                WalArchiveRecordTimestamp {
                    txn_id: 1,
                    timestamp_micros: 1_000,
                },
                WalArchiveRecordTimestamp {
                    txn_id: 2,
                    timestamp_micros: 2_000,
                },
                WalArchiveRecordTimestamp {
                    txn_id: 3,
                    timestamp_micros: 3_000,
                },
            ]
        );
        assert_eq!(retained_records.len(), 3);
        assert_eq!(&retained_records[2].payload[..], &b"SET c=3"[..]);
        assert!(target_err
            .to_string()
            .contains("beyond last durable timestamp"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn wal_archive_timestamp_retention_rejects_between_boundary_without_mutation() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-timestamp-retention-missing-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 10,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 20,
                payload: b"SET b=2".to_vec().into(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 10,
                timestamp_micros: 10_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 20,
                timestamp_micros: 20_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let before_manifest = fs::read_to_string(&manifest_path).unwrap();
        let err =
            apply_wal_archive_retention_to_timestamp_micros(&manifest_path, 15_000).unwrap_err();
        let after_manifest = fs::read_to_string(&manifest_path).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert!(err
            .to_string()
            .contains("falls between archived transaction boundaries"));
        assert_eq!(after_manifest, before_manifest);
    }

    #[test]
    fn wal_archive_base_retention_plan_keeps_base_boundary_suffix() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-base-retention-plan-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec().into(),
            },
        ];
        let timestamps = vec![
            WalArchiveRecordTimestamp {
                txn_id: 1,
                timestamp_micros: 1_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 2,
                timestamp_micros: 2_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 3,
                timestamp_micros: 3_000,
            },
            WalArchiveRecordTimestamp {
                txn_id: 4,
                timestamp_micros: 4_000,
            },
        ];

        write_wal_archive_with_timestamps(&manifest_path, &segment_dir, &records, 1, &timestamps)
            .unwrap();
        let plan = plan_wal_archive_retention_from_txn(&manifest_path, 2).unwrap();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(plan.target_txn_id, 2);
        assert_eq!(plan.retained_record_count, 3);
        assert_eq!(plan.removed_record_count, 1);
        assert_eq!(plan.retained_manifest.checkpoint.durable_record_count, 3);
        assert_eq!(
            plan.retained_manifest.checkpoint.last_durable_txn_id,
            Some(4)
        );
        assert_eq!(plan.retained_manifest.segments[0].first_txn_id, Some(2));
        assert_eq!(plan.retained_manifest.record_timestamps[0].txn_id, 2);
        assert_eq!(
            plan.retained_manifest.record_timestamps[0].timestamp_micros,
            2_000
        );
    }

    #[test]
    fn wal_archive_base_retention_apply_rewrites_to_base_suffix() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-base-retention-apply-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
            WalRecord {
                txn_id: 3,
                payload: b"SET c=3".to_vec().into(),
            },
            WalRecord {
                txn_id: 4,
                payload: b"SET d=4".to_vec().into(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let plan = apply_wal_archive_retention_from_txn(&manifest_path, 2).unwrap();
        let (retained_manifest, retained_records) = read_wal_archive(&manifest_path).unwrap();
        let target_err = read_wal_archive_to_txn(&manifest_path, 1).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert_eq!(plan.retained_record_count, 3);
        assert_eq!(plan.removed_record_count, 1);
        assert_eq!(retained_manifest.checkpoint.durable_record_count, 3);
        assert_eq!(retained_manifest.checkpoint.last_durable_txn_id, Some(4));
        assert_eq!(
            retained_records
                .iter()
                .map(|record| record.txn_id)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
        assert!(target_err
            .to_string()
            .contains("before first archived transaction"));
    }

    #[test]
    fn wal_archive_retention_rejects_malformed_archive_before_cleanup() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-retention-malformed-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![
            WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            },
            WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            },
        ];

        write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        let err = apply_wal_archive_retention_to_txn(&manifest_path, 1).unwrap_err();
        assert!(segment_dir.join("segment-0002.wal").exists());
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("non-increasing transaction order"));
    }

    #[test]
    fn wal_archive_rejects_missing_segment() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-missing-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_dir = dir.join("segments");
        let records = vec![WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec().into(),
        }];

        let manifest = write_wal_archive(&manifest_path, &segment_dir, &records, 1).unwrap();
        fs::remove_file(resolve_manifest_path(
            &manifest_path,
            &manifest.segments[0].segment_path,
        ))
        .unwrap();
        let err = read_wal_archive(&manifest_path).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("failed to open WAL segment"));
    }

    #[test]
    fn wal_archive_rejects_manifest_record_count_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-count-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_path = dir.join("segment-0001.wal");
        write_wal_segment(
            &segment_path,
            &[WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            }],
        )
        .unwrap();
        let manifest = WalArchiveManifest {
            segments: vec![WalArchiveSegment {
                segment_path: PathBuf::from("segment-0001.wal"),
                record_count: 2,
                first_txn_id: Some(1),
                last_txn_id: Some(1),
            }],
            checkpoint: WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(1),
            },
            record_timestamps: Vec::new(),
        };
        write_wal_archive_manifest(&manifest_path, &manifest).unwrap();

        let err = read_wal_archive(&manifest_path).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("expected 2 records"));
    }

    #[test]
    fn wal_archive_rejects_non_increasing_transaction_order() {
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-archive-order-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let manifest_path = dir.join("MANIFEST");
        let segment_a = dir.join("segment-0001.wal");
        let segment_b = dir.join("segment-0002.wal");
        write_wal_segment(
            &segment_a,
            &[WalRecord {
                txn_id: 2,
                payload: b"SET b=2".to_vec().into(),
            }],
        )
        .unwrap();
        write_wal_segment(
            &segment_b,
            &[WalRecord {
                txn_id: 1,
                payload: b"SET a=1".to_vec().into(),
            }],
        )
        .unwrap();
        let manifest = WalArchiveManifest {
            segments: vec![
                WalArchiveSegment {
                    segment_path: PathBuf::from("segment-0001.wal"),
                    record_count: 1,
                    first_txn_id: Some(2),
                    last_txn_id: Some(2),
                },
                WalArchiveSegment {
                    segment_path: PathBuf::from("segment-0002.wal"),
                    record_count: 1,
                    first_txn_id: Some(1),
                    last_txn_id: Some(1),
                },
            ],
            checkpoint: WalCheckpointMeta {
                durable_record_count: 2,
                last_durable_txn_id: Some(1),
            },
            record_timestamps: Vec::new(),
        };
        write_wal_archive_manifest(&manifest_path, &manifest).unwrap();

        let err = read_wal_archive(&manifest_path).unwrap_err();
        let _ = fs::remove_dir_all(dir);

        assert!(err.to_string().contains("non-increasing transaction order"));
    }

    // ---- E1 step 1: FUA fence-pool durability backend -----------------------------------------

    #[cfg(unix)]
    fn fua_test_base(name: &str) -> PathBuf {
        // A per-test DIRECTORY so the `<base>.fua.<id>` segment files don't collide, and cleanup
        // can drop the whole dir.
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-wal-fua-{name}-{}-{}",
            std::process::id(),
            NEXT_TEST_PATH_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create fua test dir");
        dir.join("wal.segment")
    }

    #[cfg(unix)]
    fn rec(txn_id: TxnId, payload: &[u8]) -> WalRecord {
        WalRecord {
            txn_id,
            payload: payload.to_vec().into(),
        }
    }

    /// (a) Roundtrip + recovery parity: the same logical records recover BYTE-IDENTICALLY through
    /// the FUA backend's frame log and through the serial segment reader.
    #[cfg(unix)]
    #[test]
    fn fua_roundtrip_recovers_identically_to_serial_path() {
        let base = fua_test_base("roundtrip");
        let records = vec![
            rec(1, b"CREATE TABLE t (id INT)"),
            rec(2, b"INSERT INTO t (id) VALUES (1)"),
            rec(3, &vec![0xABu8; 9000]), // multi-4KiB payload, exercises frame padding
            rec(4, b"UPDATE t SET id = 2 WHERE id = 1"),
        ];

        // FUA path: append + group-flush all records as one group, then recover from disk.
        {
            let mut wal =
                WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("fua create");
            assert!(wal.is_durable());
            assert_eq!(wal.durable_segment_path(), Some(base.as_path()));
            for record in &records {
                wal.append(record.clone());
            }
            wal.flush_all().expect("fua flush");
            assert_eq!(wal.flushed_count(), records.len());
            assert_eq!(wal.unflushed_count(), 0);
        }
        let fua_recovered = recover_fua_wal_records(&base).expect("fua recover");

        // Serial path: the same records written to a plain segment and read back.
        let serial_path = base.with_file_name("serial.segment");
        write_wal_segment(&serial_path, &records).expect("serial write");
        let serial_recovered = read_wal_segment(&serial_path).expect("serial read");

        assert_eq!(fua_recovered, records, "fua recovery must match input");
        assert_eq!(
            fua_recovered, serial_recovered,
            "fua and serial recovery must be identical"
        );

        let _ = std::fs::remove_dir_all(base.parent().unwrap());
    }

    /// (b) The durable watermark advances ONLY over the contiguous durable cut and is monotonic,
    /// even with MULTIPLE flush jobs in flight completing out of order across the fence pool.
    #[cfg(unix)]
    #[test]
    fn fua_watermark_is_monotonic_under_concurrent_in_flight_jobs() {
        use std::sync::Arc as StdArc;
        use std::sync::Mutex as StdMutex;

        let base = fua_test_base("monotonic");
        let wal = StdArc::new(StdMutex::new(
            WalBuffer::with_fua_durable_segment(&base, 32, 4 << 20).expect("fua create"),
        ));

        // Producer: append records and start group flushes concurrently. Each begin snapshots a
        // disjoint record range under the outer lock; commits run lock-free and may finish out of
        // order, so a watching thread must never see the watermark go backwards or exceed the
        // published count.
        let groups = 40usize;
        let per_group = 5usize;
        let total = groups * per_group;
        // A watcher samples the durable watermark under the outer lock throughout the run and
        // asserts it NEVER decreases (advances only over the contiguous cut) and never overshoots
        // the records appended so far. This is the time-domain monotonicity property; the
        // per-commit return values below are the value-domain property.
        let done = StdArc::new(std::sync::atomic::AtomicBool::new(false));
        let watcher = {
            let wal = StdArc::clone(&wal);
            let done = StdArc::clone(&done);
            std::thread::spawn(move || {
                let mut last = 0usize;
                while !done.load(std::sync::atomic::Ordering::Acquire) {
                    let watermark = wal.lock().unwrap().flushed_count();
                    assert!(
                        watermark >= last,
                        "watermark regressed: {watermark} < {last}"
                    );
                    assert!(
                        watermark <= total,
                        "watermark {watermark} exceeds published {total}"
                    );
                    last = watermark;
                    std::thread::yield_now();
                }
            })
        };

        // (job_target, join handle): each commit must return a watermark that already covers its
        // own group (the durable cut reached at least its target) and never exceeds the total.
        let mut handles = Vec::new();
        let mut next_txn = 1u64;
        for group in 0..groups {
            let target = (group + 1) * per_group;
            let begun = {
                let mut guard = wal.lock().unwrap();
                for _ in 0..per_group {
                    guard.append(rec(next_txn, format!("op-{next_txn}").as_bytes()));
                    next_txn += 1;
                }
                guard.begin_group_flush().expect("begin")
            };
            match begun {
                WalGroupFlushBegin::Clean { .. } => {}
                WalGroupFlushBegin::Job(job) => {
                    handles.push((
                        target,
                        std::thread::spawn(move || job.commit().expect("commit")),
                    ));
                }
            }
        }

        for (target, handle) in handles {
            let watermark = handle.join().expect("join");
            assert!(
                watermark >= target,
                "commit returned watermark {watermark} below its own group target {target}"
            );
            assert!(
                watermark <= total,
                "watermark {watermark} exceeds published {total}"
            );
        }
        done.store(true, std::sync::atomic::Ordering::Release);
        watcher.join().expect("watcher");

        {
            let mut guard = wal.lock().unwrap();
            guard.flush_all().expect("final flush");
            assert_eq!(guard.flushed_count(), total);
        }
        // Everything recovers, in order.
        let recovered = recover_fua_wal_records(&base).expect("recover");
        assert_eq!(recovered.len(), total);
        for (index, record) in recovered.iter().enumerate() {
            assert_eq!(record.txn_id, index as u64 + 1);
        }
        let _ = std::fs::remove_dir_all(base.parent().unwrap());
    }

    /// (c) Segment roll mid-stream: a small per-segment capacity forces several rolls; the
    /// totally-ordered record history recovers contiguously ACROSS the rolled segment files.
    #[cfg(unix)]
    #[test]
    fn fua_segment_roll_recovers_across_segments() {
        let base = fua_test_base("roll");
        let payload = vec![0x5Au8; 2000]; // ~3 frames fit a 12KiB-ish segment before StorageFull
        let total = 60usize;
        {
            // 16KiB data capacity per segment: each ~2KB record pads to 4KiB, so ~4 frames/segment
            // -> many rolls over 60 records.
            let mut wal =
                WalBuffer::with_fua_durable_segment(&base, 8, 16 * 1024).expect("fua create");
            for txn in 1..=total as u64 {
                wal.append(rec(txn, &payload));
                // Flush each record individually so every group is its own frame — maximizes rolls.
                wal.flush_all().expect("flush");
                assert_eq!(wal.flushed_count(), txn as usize);
            }
        }
        // More than one segment file must exist (a roll happened).
        let dir = base.parent().unwrap();
        let segment_count = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n.starts_with("wal.segment.fua."))
                    .unwrap_or(false)
            })
            .count();
        assert!(
            segment_count > 1,
            "expected multiple rolled segments, found {segment_count}"
        );

        let recovered = recover_fua_wal_records(&base).expect("recover across segments");
        assert_eq!(recovered.len(), total);
        for (index, record) in recovered.iter().enumerate() {
            assert_eq!(record.txn_id, index as u64 + 1);
            assert_eq!(&record.payload[..], &payload[..]);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The FUA backend surfaces its config errors and the step-1 unsupported ops fail-closed.
    #[cfg(unix)]
    #[test]
    fn fua_config_and_unsupported_ops_error() {
        let base = fua_test_base("config");
        assert!(WalBuffer::with_fua_durable_segment(&base, 8, 0).is_err());

        let mut wal = WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("create");
        wal.append(rec(1, b"x"));
        wal.flush_all().expect("flush");
        // Prefix truncation is a step-2 capability; it must error rather than silently no-op.
        assert!(wal.truncate_durable_segment_prefix(0).is_err());
        assert_eq!(wal.durable_segment_base_records(), 0);
        let _ = std::fs::remove_dir_all(base.parent().unwrap());
    }

    /// E1 step 3: REOPEN an existing FUA log — write + drop, reopen and verify the recovered
    /// history + that appends CONTINUE the contiguous log (new segment above the old id), then
    /// drop + recover a THIRD time to prove both segments chain end-to-end.
    #[cfg(unix)]
    #[test]
    fn fua_reopen_continues_the_log_and_recovers_across_lives() {
        let base = fua_test_base("reopen");
        let first = vec![
            rec(1, b"CREATE TABLE t (id INT)"),
            rec(2, b"INSERT INTO t (id) VALUES (1)"),
            rec(3, b"INSERT INTO t (id) VALUES (2)"),
        ];
        // Life 1: write + flush + clean drop (drain).
        {
            let mut wal = WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("create");
            for record in &first {
                wal.append(record.clone());
            }
            wal.flush_all().expect("flush life 1");
        }
        let recovered_1 = recover_fua_wal_records(&base).expect("recover life 1");
        assert_eq!(recovered_1, first, "life-1 recovery matches input");

        // Life 2: reopen seeded with the recovered history, verify watermark, then APPEND more.
        let second = vec![
            rec(4, b"UPDATE t SET id = 3 WHERE id = 1"),
            rec(5, b"DELETE FROM t WHERE id = 2"),
        ];
        {
            let mut wal = WalBuffer::with_recovered_fua_durable_segment(
                &base,
                recovered_1.clone(),
                8,
                1 << 20,
            )
            .expect("reopen");
            assert!(wal.is_durable());
            assert_eq!(
                wal.flushed_count(),
                first.len(),
                "reopen reports the recovered records as already durable"
            );
            assert_eq!(wal.unflushed_count(), 0, "nothing unflushed on reopen");
            assert_eq!(wal.durable_segment_path(), Some(base.as_path()));
            for record in &second {
                wal.append(record.clone());
            }
            assert_eq!(wal.unflushed_count(), second.len());
            wal.flush_all().expect("flush life 2");
            assert_eq!(wal.flushed_count(), first.len() + second.len());
        }

        // Life 3: recover across BOTH segments — the chain must be first ++ second, in order.
        let mut expected = first.clone();
        expected.extend(second.clone());
        let recovered_2 = recover_fua_wal_records(&base).expect("recover life 2");
        assert_eq!(
            recovered_2, expected,
            "recovery chains the old and new segments contiguously"
        );
        assert!(
            fua_wal_segments_exist(&base),
            "segments are retained for recovery"
        );
        let _ = std::fs::remove_dir_all(base.parent().unwrap());
    }

    /// E1 torn-tail crash safety: a crash that leaves the LAST frame's payload corrupt must recover
    /// to the durable cut BEFORE the torn frame (the torn commit was never acknowledged), and a
    /// reopen must chain new appends contiguously above that cut WITHOUT ever resurrecting the torn
    /// frame.
    #[cfg(unix)]
    #[test]
    fn fua_torn_tail_recovers_to_cut_and_reopen_chains_contiguously() {
        // On-disk frame-log layout (crate `gpu_db_write_conveyor::fua_frame_log`): a 4096B file
        // header, then each frame is a 64B header + payload padded up to a 4096B block. With tiny,
        // individually-flushed records every frame is exactly one 4096B block, so frame `i`'s header
        // sits at `4096 * (i + 1)` and its payload starts 64B later.
        const FRAME_LOG_HEADER_BYTES: u64 = 4096;
        const FRAME_ALIGN: u64 = 4096;
        const FRAME_HEADER_BYTES: u64 = 64;

        let base = fua_test_base("torn_tail");
        // txn 5 is the record whose frame we tear; 1..=4 form the durable prefix.
        let pre_tear = vec![
            rec(1, b"CREATE TABLE t (id INT)"),
            rec(2, b"INSERT INTO t (id) VALUES (1)"),
            rec(3, b"INSERT INTO t (id) VALUES (2)"),
            rec(4, b"INSERT INTO t (id) VALUES (3)"),
        ];
        let torn = rec(5, b"INSERT INTO t (id) VALUES (99)");

        // Life 1: append + group-flush each record as its OWN frame (one 4096B block each) into a
        // single large segment (no roll), then clean-drop so the file is fully written and closed.
        {
            let mut wal = WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20).expect("create");
            for record in pre_tear.iter().chain(std::iter::once(&torn)) {
                wal.append(record.clone());
                wal.flush_all().expect("flush frame");
            }
            assert_eq!(wal.flushed_count(), pre_tear.len() + 1);
        }
        // All five frames live in segment id 1 (no roll at 1 MiB); before tearing, all recover.
        let seg1 = base.with_file_name("wal.segment.fua.1");
        assert!(seg1.is_file(), "single un-rolled segment at id 1");
        assert_eq!(
            recover_fua_wal_records(&base)
                .expect("pre-tear recover")
                .len(),
            pre_tear.len() + 1,
            "all frames recover before the tear"
        );

        // Corrupt the LAST frame's first payload byte so its payload CRC fails on scan.
        let last_frame_index = pre_tear.len() as u64; // 0-based index of txn 5's frame
        let payload_offset =
            FRAME_LOG_HEADER_BYTES + last_frame_index * FRAME_ALIGN + FRAME_HEADER_BYTES;
        {
            use std::io::{Read, Seek, SeekFrom, Write};
            let mut file = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&seg1)
                .expect("open segment for corruption");
            file.seek(SeekFrom::Start(payload_offset)).expect("seek");
            let mut byte = [0u8; 1];
            file.read_exact(&mut byte).expect("read payload byte");
            byte[0] ^= 0xFF; // flip -> payload CRC no longer matches the frame header
            file.seek(SeekFrom::Start(payload_offset))
                .expect("seek back");
            file.write_all(&byte).expect("write corrupted byte");
            file.sync_all().expect("persist corruption");
        }

        // Recovery stops at the durable cut BEFORE the torn frame: only the pre-tear prefix.
        let recovered_1 = recover_fua_wal_records(&base).expect("torn recover");
        assert_eq!(
            recovered_1, pre_tear,
            "recovery stops at the durable cut before the torn frame"
        );

        // Reopen above the recovered cut, append more records, flush, clean-drop.
        let post = vec![
            rec(6, b"UPDATE t SET id = 4 WHERE id = 1"),
            rec(7, b"DELETE FROM t WHERE id = 2"),
        ];
        {
            let mut wal = WalBuffer::with_recovered_fua_durable_segment(
                &base,
                recovered_1.clone(),
                8,
                1 << 20,
            )
            .expect("reopen after tear");
            assert_eq!(
                wal.flushed_count(),
                pre_tear.len(),
                "reopen reports the durable cut as already durable"
            );
            for record in &post {
                wal.append(record.clone());
            }
            wal.flush_all().expect("flush after reopen");
            assert_eq!(wal.flushed_count(), pre_tear.len() + post.len());
        }

        // Re-recover across both segments: the history chains contiguously (pre-tear ++ new), with
        // no gap and the torn frame (txn 5) NEVER resurrected.
        let mut expected = pre_tear.clone();
        expected.extend(post.clone());
        let recovered_2 = recover_fua_wal_records(&base).expect("recover after reopen");
        assert_eq!(
            recovered_2, expected,
            "post-reopen history chains pre-tear ++ new records with no gap"
        );
        assert!(
            !recovered_2.iter().any(|r| r.txn_id == 5),
            "the torn frame (txn 5) is never resurrected"
        );
        let _ = std::fs::remove_dir_all(base.parent().unwrap());
    }

    /// A fresh FUA create must REFUSE to clobber a plain serial WAL file sitting at `<base>` — it
    /// might be a durable serial log, and it would later trip the mixed-backend reopen refusal.
    #[cfg(unix)]
    #[test]
    fn fua_create_refuses_to_clobber_a_plain_serial_file() {
        let base = fua_test_base("serial_guard");
        // A leftover plain serial WAL file exactly at `<base>`.
        fs::write(&base, b"pretend durable serial WAL").expect("write serial file");

        let err = WalBuffer::with_fua_durable_segment(&base, 8, 1 << 20)
            .expect_err("fresh FUA create must fail when a plain serial file is present");
        let msg = err.to_string();
        assert!(
            msg.contains("plain serial WAL file") && msg.contains("remove it"),
            "error must tell the operator to remove the serial file: {msg}"
        );
        // Fail-closed: the serial file is NOT deleted, and no FUA segment was created.
        assert!(base.is_file(), "serial file must be left intact");
        assert!(
            !fua_wal_segments_exist(&base),
            "no FUA segment should be created on the refused path"
        );
        let _ = std::fs::remove_dir_all(base.parent().unwrap());
    }
}
