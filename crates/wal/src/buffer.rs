//! WAL buffering, group flush, and durable segment ownership.

use super::*;

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
pub(super) fn wal_prealloc_chunk_bytes() -> u64 {
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
    static ZEROS: [u8; 1024 * 1024] = [0; 1024 * 1024];
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
        // E2.5c-3 DEFAULT FLIP: the FUA fence-pool backend is the durable default on unix
        // (pre-written extents + pipelined FUA write-through; the serial fdatasync path was
        // measured ~20x slower on the reference NVMe). Opt out with GPU_DB_WAL_DURABILITY=serial.
        let selected = std::env::var("GPU_DB_WAL_DURABILITY")
            .map(|v| v.eq_ignore_ascii_case("fua"))
            .unwrap_or(cfg!(unix));
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
