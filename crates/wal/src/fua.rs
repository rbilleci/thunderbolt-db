//! FUA fence-pool durability backend for [`WalBuffer`] (E1 step 1).
//!
//! The serial `WalDurableCore` makes each group durable with ONE `write_all` + `fdatasync` and a
//! single `io_in_flight` slot — at most one durable IO in flight, ~0.86ms/job. This backend
//! replaces that single slot with a PIPELINED FUA fence pool
//! (`gpu_db_write_conveyor::FuaFrameLog`): every group's encoded record run is published as one
//! 4KiB-aligned frame into a bounded ring, `lanes` fence lanes FUA-write frames concurrently, and
//! the contiguous durable cut (`durable_seq`) is the record watermark. Multiple groups can be
//! durable in flight at once, and the device's FUA coalescing rewards the queue depth (measured
//! fast-mode flip qd 16-48 on the reference NVMe).
//!
//! Correctness contract preserved from the serial path:
//!
//! * **Frame payload == the serial on-disk bytes.** A frame payload is exactly the
//!   [`super::encode_record_into`] run the serial group flush would have written for that group,
//!   so both backends recover to identical `WalRecord`s. Recovery = [`recover_fua_wal_records`]
//!   (a `recover_frame_log_by_scan` over each segment) then [`super::decode_wal_record_run`].
//! * **Watermark advances over the contiguous durable cut ONLY.** The frame log's `first_seq` /
//!   `seq_count` are the group's GLOBAL record-count range; `durable_seq()` reports the record
//!   count of the last contiguously-fenced frame — precisely the WAL-before-visibility gate.
//! * **Fail-closed wedge.** A fence-lane IO failure (`FuaFrameLog::fence_failed`), a segment-roll
//!   failure, a pool-join error, or an abandoned-mid-flight job POISONS the backend so no later
//!   flush proceeds — the same fail-closed shape as the serial backend's `poisoned` state.
//!
//! Deferred to step 2 (documented in the crate handover): segment RECYCLE (this step create-only
//! rolls to fresh files), external checkpoint/prefix truncation, and recovery-reopen (append to an
//! existing FUA log). Charter note: this is host control-plane work (WAL/durability I/O); no
//! data-plane logic lives here.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use gpu_db_types::EngineError;
use gpu_db_write_conveyor::{
    recover_frame_log_by_scan, FuaFrameLog, FuaFrameLogAppender, FuaFrameLogConfig,
    FuaFrameLogFencePool,
};

use crate::{decode_wal_record_run, WalGroupCommitStats, WalRecord};

/// Busy-spins before falling back to `yield_now` in the durable-cut wait. Pure spin at low
/// contention keeps the p50 ack near one fence latency; the yield fallback avoids burning a core
/// when the pool is genuinely backed up. NO futex/condvar per-commit wakeups — measured law: futex
/// wakes are unpayable at high ack rates.
const SPIN_BEFORE_YIELD: u32 = 256;

/// The currently-open FUA segment: its frame log, the single appender, and its fence pool. On a
/// roll this whole record is replaced; `appender`/`pool` are `Option` so `Drop` (and the roll) can
/// take ownership to `finish` + `join` the lanes.
struct ActiveSegment {
    log: Arc<FuaFrameLog>,
    appender: Option<FuaFrameLogAppender>,
    pool: Option<FuaFrameLogFencePool>,
    segment_id: u64,
}

/// The FUA fence-pool durability backend behind an optional field of [`WalBuffer`].
pub(crate) struct FuaWalBackend {
    /// Base path; per-segment files are `<base>.fua.<segment_id>`.
    base_path: PathBuf,
    lanes: usize,
    segment_bytes: usize,
    /// Guards the appender (publish is single-threaded and totally ordered) and segment rolling.
    active: Mutex<ActiveSegment>,
    /// Records handed to frames so far (advanced in `begin_group_flush` under the buffer's OUTER
    /// lock, so `first_seq` values are assigned in total order). Distinct from the durable cut.
    published: AtomicUsize,
    /// Monotonic per-Job ticket allocated in `begin_group_flush` (outer lock held). The commit
    /// PUBLISH must happen in ticket order (`publish_cursor`) so frames enter the log in `first_seq`
    /// order — the frame log's contiguous durable cut assumes `end_seq` is monotonic in frame id.
    /// Only the fast staging memcpy is serialized this way; the fence-pool durability still
    /// pipelines across all published frames.
    next_ticket: AtomicU64,
    publish_cursor: AtomicU64,
    /// Durable record count carried over from fully-drained (rolled-away) segments; the active
    /// segment's `durable_seq` continues above this. Monotonic.
    rolled_baseline: AtomicU64,
    /// Segment id to assign to the NEXT rolled segment (low 32 bits are the recycle epoch; must be
    /// non-zero — starts at 1).
    next_segment_id: AtomicU64,
    /// Fail-closed wedge reason; once set, every flush errors until restart recovery.
    poison: Mutex<Option<String>>,
    stats: Mutex<WalGroupCommitStats>,
}

impl std::fmt::Debug for FuaWalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuaWalBackend")
            .field("base_path", &self.base_path)
            .field("lanes", &self.lanes)
            .field("segment_bytes", &self.segment_bytes)
            .field("published", &self.published.load(Ordering::Relaxed))
            .field(
                "rolled_baseline",
                &self.rolled_baseline.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl FuaWalBackend {
    /// Create a FRESH FUA-durable database at `base_path` (any stale `<base>.fua.*` segments are
    /// removed first — the same clobber-on-create semantic as the serial fresh constructor).
    pub(crate) fn create(
        base_path: PathBuf,
        lanes: usize,
        segment_bytes: usize,
    ) -> Result<Self, EngineError> {
        let lanes = lanes.max(1);
        if segment_bytes == 0 {
            return Err(EngineError::Durability(
                "FUA WAL segment_bytes must be non-zero".to_string(),
            ));
        }
        if let Some(parent) = base_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|err| {
                EngineError::Durability(format!(
                    "failed to create FUA WAL directory {}: {err}",
                    parent.display()
                ))
            })?;
        }
        remove_stale_segments(&base_path);
        let segment_id = 1;
        let log = open_segment(&base_path, segment_id, segment_bytes)?;
        let pool = log.spawn_fence_pool(lanes);
        let appender = log.appender();
        Ok(Self {
            base_path,
            lanes,
            segment_bytes,
            active: Mutex::new(ActiveSegment {
                log,
                appender: Some(appender),
                pool: Some(pool),
                segment_id,
            }),
            published: AtomicUsize::new(0),
            next_ticket: AtomicU64::new(0),
            publish_cursor: AtomicU64::new(0),
            rolled_baseline: AtomicU64::new(0),
            next_segment_id: AtomicU64::new(segment_id + 1),
            poison: Mutex::new(None),
            stats: Mutex::new(WalGroupCommitStats::default()),
        })
    }

    /// REOPEN an existing FUA-durable database at `base_path` and continue appending ABOVE the
    /// recovered history. `recovered_records` is the count of records
    /// [`recover_fua_wal_records`] read back from the retained `<base>.fua.*` segments (the caller
    /// replayed them). We do NOT append into any recovered segment — a fresh segment is opened
    /// above the highest existing id, which keeps torn-tail recovery semantics trivial (a crash can
    /// only ever tear the tail of the newest segment; older segments stay byte-frozen). The old
    /// segments are RETAINED for recovery until prefix-truncation lands (E1 step 3); a future
    /// reopen re-recovers them in ascending id order and the new segment chains on top (its first
    /// frame's `first_seq` == `recovered_records`, exactly the contiguous cut the old segments end
    /// at). The published/durable watermarks start at `recovered_records` so `flushed_count()` and
    /// the record→frame `first_seq` mapping are continuous with the recovered log.
    ///
    /// Torn-tail caveat (documented, untested — reopen is only exercised after a CLEAN drain): if a
    /// prior crash left a GAP inside a retained old segment, recovery stops at that gap and this
    /// new segment's higher-id frames are never reached on the next recovery. Truncation (step 3)
    /// will retire fully-drained old segments and remove this window.
    pub(crate) fn reopen(
        base_path: PathBuf,
        lanes: usize,
        segment_bytes: usize,
        recovered_records: usize,
    ) -> Result<Self, EngineError> {
        let lanes = lanes.max(1);
        if segment_bytes == 0 {
            return Err(EngineError::Durability(
                "FUA WAL segment_bytes must be non-zero".to_string(),
            ));
        }
        if let Some(parent) = base_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|err| {
                EngineError::Durability(format!(
                    "failed to create FUA WAL directory {}: {err}",
                    parent.display()
                ))
            })?;
        }
        // Open a FRESH segment ABOVE every existing id (never recycle a recovered file's epoch).
        let segment_id = highest_existing_segment_id(&base_path) + 1;
        let log = open_segment(&base_path, segment_id, segment_bytes)?;
        let pool = log.spawn_fence_pool(lanes);
        let appender = log.appender();
        let recovered = recovered_records as u64;
        Ok(Self {
            base_path,
            lanes,
            segment_bytes,
            active: Mutex::new(ActiveSegment {
                log,
                appender: Some(appender),
                pool: Some(pool),
                segment_id,
            }),
            published: AtomicUsize::new(recovered_records),
            next_ticket: AtomicU64::new(0),
            publish_cursor: AtomicU64::new(0),
            rolled_baseline: AtomicU64::new(recovered),
            next_segment_id: AtomicU64::new(segment_id + 1),
            poison: Mutex::new(None),
            stats: Mutex::new(WalGroupCommitStats::default()),
        })
    }

    pub(crate) fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Free fence lanes in the active segment's pool (the engine-seam PACING signal — a committer
    /// may begin its own group flush only while a lane is free; see the engine's concurrent
    /// durability wait).
    pub(crate) fn free_fence_slots(&self) -> usize {
        let active = self.lock_active();
        active.log.free_fence_slots(self.lanes)
    }

    /// Records already handed to frames (the `begin_group_flush` cursor; buffer outer lock held).
    pub(crate) fn published_records(&self) -> usize {
        self.published.load(Ordering::Acquire)
    }

    /// Advance the published cursor (called under the buffer's outer lock so frame order is total).
    pub(crate) fn set_published(&self, target: usize) {
        self.published.store(target, Ordering::Release);
    }

    /// Allocate the next publish ticket (called under the buffer's outer lock, once per Job, so
    /// tickets are contiguous and ordered by `first_seq`).
    pub(crate) fn next_ticket(&self) -> u64 {
        self.next_ticket.fetch_add(1, Ordering::AcqRel)
    }

    /// The durable record watermark: the contiguous durable cut of the active segment, never below
    /// the count carried from fully-drained rolled-away segments.
    pub(crate) fn durable_records(&self) -> usize {
        let active = self.lock_active();
        active
            .log
            .durable_seq()
            .max(self.rolled_baseline.load(Ordering::Acquire)) as usize
    }

    pub(crate) fn group_commit_stats(&self) -> WalGroupCommitStats {
        *self.stats.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn poison_reason(&self) -> Option<String> {
        self.poison
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub(crate) fn set_poison(&self, reason: &str) {
        let mut guard = self.poison.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_none() {
            *guard = Some(reason.to_string());
        }
    }

    pub(crate) fn poison_error(&self, reason: &str) -> EngineError {
        EngineError::Durability(format!(
            "FUA WAL backend {} is wedged by an earlier failure ({reason}); restart to recover \
             from the durable cut",
            self.base_path.display()
        ))
    }

    fn lock_active(&self) -> std::sync::MutexGuard<'_, ActiveSegment> {
        self.active.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Publish one group's frame into the active segment (staging memcpy only — the fence pool does
    /// the IO), rolling to a fresh segment on `StorageFull`. Returns the specific frame log the
    /// frame landed in so the caller can poll THAT log's durable cut across concurrent rolls.
    ///
    /// Fence-pool PACING (the measured anti-convoy law): publish only while a fence slot is free
    /// (`free_fence_slots > 0`), accumulating otherwise, so backlog spreads across the pool instead
    /// of shipping tiny frames that collapse the pool to serial-fence latency.
    fn publish(
        &self,
        ticket: u64,
        payload: &[u8],
        first_seq: u64,
        seq_count: u32,
    ) -> Result<Arc<FuaFrameLog>, EngineError> {
        // Wait our turn: publish frames in ticket (== `first_seq`) order. This is the ONLY ordering
        // point and it only gates a staging memcpy; the fence pool then pipelines the durability of
        // every published frame concurrently. A prior ticket that wedged the backend without
        // advancing the cursor is surfaced as poison so later tickets abort rather than hang.
        let mut spins = 0u32;
        while self.publish_cursor.load(Ordering::Acquire) != ticket {
            if let Some(reason) = self.poison_reason() {
                return Err(self.poison_error(&reason));
            }
            if spins < SPIN_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        let mut active = self.lock_active();
        loop {
            while active.log.free_fence_slots(self.lanes) == 0 {
                if active.log.fence_failed() {
                    self.set_poison("FUA fence lane failed");
                    return Err(self.poison_error("FUA fence lane failed"));
                }
                std::thread::yield_now();
            }
            let result = active
                .appender
                .as_mut()
                .expect("FUA appender present")
                .publish_frame(payload, first_seq, seq_count);
            match result {
                Ok(_) => {
                    let log = Arc::clone(&active.log);
                    // Release the turn so the next ticket can publish. Store while holding the
                    // active lock so the memcpy is fully visible before the successor publishes.
                    self.publish_cursor.store(ticket + 1, Ordering::Release);
                    return Ok(log);
                }
                Err(err) if err.kind() == std::io::ErrorKind::StorageFull => {
                    self.roll(&mut active)?;
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    // Slot ring momentarily full despite pacing (shouldn't happen with lanes <<
                    // ring); back off and retry.
                    std::thread::yield_now();
                }
                Err(err) => {
                    let reason = format!("FUA frame publish failed: {err}");
                    self.set_poison(&reason);
                    return Err(self.poison_error(&reason));
                }
            }
        }
    }

    /// Roll the full active segment to a fresh one. The current segment is DRAINED (appender
    /// finished + pool joined) so it is 100% durable BEFORE any record lands in the next segment —
    /// otherwise a crash with the new segment partly durable and the old segment's tail not durable
    /// would leave a GAP in the totally-ordered log. Caller holds the active lock.
    fn roll(&self, active: &mut ActiveSegment) -> Result<(), EngineError> {
        if let Some(appender) = active.appender.take() {
            appender.finish();
        }
        if let Some(pool) = active.pool.take() {
            pool.join().map_err(|err| {
                let reason = format!("FUA segment drain (fence-pool join) failed: {err}");
                self.set_poison(&reason);
                self.poison_error(&reason)
            })?;
        }
        // The drained segment is now fully durable up to its published count.
        let baseline = active.log.durable_seq();
        let mut current = self.rolled_baseline.load(Ordering::Acquire);
        while baseline > current {
            match self.rolled_baseline.compare_exchange(
                current,
                baseline,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
        let new_id = self.next_segment_id.fetch_add(1, Ordering::AcqRel);
        let new_log = open_segment(&self.base_path, new_id, self.segment_bytes)
            .inspect_err(|_| self.set_poison("FUA segment roll (create) failed"))?;
        let new_pool = new_log.spawn_fence_pool(self.lanes);
        let new_appender = new_log.appender();
        active.log = new_log;
        active.appender = Some(new_appender);
        active.pool = Some(new_pool);
        active.segment_id = new_id;
        Ok(())
    }

    /// Make a group durable: publish its frame, then WAIT for the contiguous durable cut to cover
    /// it by spin-then-yield polling (no per-commit thread wakeups). Returns the new watermark.
    fn commit_group(
        &self,
        ticket: u64,
        payload: &[u8],
        first_seq: u64,
        seq_count: u32,
        target: usize,
    ) -> Result<usize, EngineError> {
        if let Some(reason) = self.poison_reason() {
            return Err(self.poison_error(&reason));
        }
        let log = self.publish(ticket, payload, first_seq, seq_count)?;
        let mut spins = 0u32;
        loop {
            if log.durable_seq() >= target as u64 {
                break;
            }
            if log.fence_failed() {
                self.set_poison("FUA fence lane failed");
                return Err(self.poison_error("FUA fence lane failed"));
            }
            if let Some(reason) = self.poison_reason() {
                return Err(self.poison_error(&reason));
            }
            if spins < SPIN_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        {
            let mut stats = self.stats.lock().unwrap_or_else(|p| p.into_inner());
            stats.flush_groups += 1;
            stats.durable_records += seq_count as u64;
            stats.max_group_size = stats.max_group_size.max(seq_count as usize);
        }
        Ok(self.durable_records())
    }

    /// Wait (spin-then-yield) until every record up to `target` is durable — the inline
    /// `flush_all` completion for the FUA backend after its own group (if any) committed.
    pub(crate) fn wait_durable(&self, target: usize) -> Result<(), EngineError> {
        let mut spins = 0u32;
        loop {
            if let Some(reason) = self.poison_reason() {
                return Err(self.poison_error(&reason));
            }
            let (durable, failed) = {
                let active = self.lock_active();
                (
                    active
                        .log
                        .durable_seq()
                        .max(self.rolled_baseline.load(Ordering::Acquire)),
                    active.log.fence_failed(),
                )
            };
            if durable >= target as u64 {
                return Ok(());
            }
            if failed {
                self.set_poison("FUA fence lane failed");
                return Err(self.poison_error("FUA fence lane failed"));
            }
            if spins < SPIN_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

impl Drop for FuaWalBackend {
    fn drop(&mut self) {
        // Clean-shutdown flush: finish the appender so the fence lanes drain and exit, then join
        // them. Errors are unreportable from Drop (and recovery reads the on-disk cut regardless).
        let active = self.active.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some(appender) = active.appender.take() {
            appender.finish();
        }
        if let Some(pool) = active.pool.take() {
            let _ = pool.join();
        }
    }
}

/// One snapshotted FUA group flush handed back inside a [`crate::WalGroupFlushJob`]. Its `commit`
/// publishes the frame and waits for the durable cut; a drop without commit (caller panicked
/// between begin and commit) wedges the backend — the group's records were already accounted
/// `published` but never framed, which would be a permanent gap in the totally-ordered log.
pub(crate) struct FuaFlushJob {
    backend: Arc<FuaWalBackend>,
    ticket: u64,
    payload: Vec<u8>,
    first_seq: u64,
    seq_count: u32,
    target: usize,
}

impl FuaFlushJob {
    pub(crate) fn new(
        backend: Arc<FuaWalBackend>,
        ticket: u64,
        payload: Vec<u8>,
        first_seq: u64,
        seq_count: u32,
        target: usize,
    ) -> Self {
        Self {
            backend,
            ticket,
            payload,
            first_seq,
            seq_count,
            target,
        }
    }

    pub(crate) fn commit(self) -> Result<usize, EngineError> {
        self.backend.commit_group(
            self.ticket,
            &self.payload,
            self.first_seq,
            self.seq_count,
            self.target,
        )
    }

    pub(crate) fn abandon(self) {
        self.backend
            .set_poison("FUA group flush abandoned between begin and commit");
    }
}

/// `<base>.fua.<segment_id>` — one physical FUA segment file.
fn segment_file_path(base: &Path, segment_id: u64) -> PathBuf {
    let name = base
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("wal.segment");
    base.with_file_name(format!("{name}.fua.{segment_id}"))
}

/// Parse the `segment_id` out of a `<stem>.fua.<id>` file name.
fn parse_segment_id(name: &str, stem: &str) -> Option<u64> {
    let prefix = format!("{stem}.fua.");
    name.strip_prefix(&prefix).and_then(|id| id.parse().ok())
}

/// The highest `<stem>.fua.<id>` segment id present beside `base` (0 when none exist). A reopen
/// opens `highest + 1` so a fresh segment never collides with (or appends into) a recovered file.
fn highest_existing_segment_id(base: &Path) -> u64 {
    let mut highest = 0u64;
    let Some(parent) = base.parent().filter(|p| !p.as_os_str().is_empty()) else {
        return highest;
    };
    let Some(stem) = base.file_name().and_then(|n| n.to_str()) else {
        return highest;
    };
    if let Ok(entries) = std::fs::read_dir(parent) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(id) = parse_segment_id(name, stem) {
                    highest = highest.max(id);
                }
            }
        }
    }
    highest
}

/// Whether ANY `<stem>.fua.<id>` segment exists beside `base` — the guard the engine uses to
/// refuse a SERIAL reopen of what is physically a FUA log (and vice-versa), rather than silently
/// starting a fresh database that shadows the durable FUA data.
pub fn fua_wal_segments_exist(base: impl AsRef<Path>) -> bool {
    highest_existing_segment_id(base.as_ref()) > 0
}

fn open_segment(
    base: &Path,
    segment_id: u64,
    segment_bytes: usize,
) -> Result<Arc<FuaFrameLog>, EngineError> {
    let path = segment_file_path(base, segment_id);
    let config = FuaFrameLogConfig {
        path: path.clone(),
        segment_id,
        capacity_bytes: segment_bytes,
    };
    // Safety: this backend owns the segment path exclusively for the log's lifetime (fresh
    // create-only in step 1; the WalBuffer holds the sole appender/pool).
    unsafe { FuaFrameLog::create(config) }.map_err(|err| {
        EngineError::Durability(format!(
            "failed to create FUA WAL segment {}: {err}",
            path.display()
        ))
    })
}

/// Remove any `<base>.fua.*` segment files left by a previous database at this path (fresh-create
/// clobber). Best-effort — a create of the fresh segment 1 would fail loudly if a stale file with
/// that exact id survived.
fn remove_stale_segments(base: &Path) {
    let Some(parent) = base.parent().filter(|p| !p.as_os_str().is_empty()) else {
        return;
    };
    let Some(stem) = base.file_name().and_then(|n| n.to_str()) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            if parse_segment_id(name, stem).is_some() {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

/// Recover the totally-ordered `WalRecord` history from a FUA-durable database at `base_path`.
///
/// Reads every `<base>.fua.<segment_id>` segment in ascending id order, scan-recovers each to its
/// contiguous valid frame prefix ([`recover_frame_log_by_scan`]), verifies each frame's global
/// `first_seq` chains contiguously (a gap ends the durable prefix — a torn tail), and decodes each
/// frame payload (the serial-encoded record run) back into `WalRecord`s. The result is
/// byte/semantic-identical to what the serial reader would recover from the same logical records.
pub fn recover_fua_wal_records(base_path: impl AsRef<Path>) -> Result<Vec<WalRecord>, EngineError> {
    let base = base_path.as_ref();
    let Some(parent) = base.parent().filter(|p| !p.as_os_str().is_empty()) else {
        return Ok(Vec::new());
    };
    let Some(stem) = base.file_name().and_then(|n| n.to_str()) else {
        return Ok(Vec::new());
    };
    let mut segments: Vec<(u64, PathBuf)> = Vec::new();
    match std::fs::read_dir(parent) {
        Ok(entries) => {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(id) = parse_segment_id(name, stem) {
                        segments.push((id, entry.path()));
                    }
                }
            }
        }
        // No directory / no segments = a fresh (never-flushed) database.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(EngineError::Durability(format!(
                "failed to enumerate FUA WAL segments in {}: {err}",
                parent.display()
            )));
        }
    }
    segments.sort_by_key(|(id, _)| *id);

    let mut records = Vec::new();
    let mut expected_first = 0u64;
    for (_id, path) in segments {
        let frames = recover_frame_log_by_scan(&path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to scan-recover FUA WAL segment {}: {err}",
                path.display()
            ))
        })?;
        for frame in frames {
            if frame.first_seq != expected_first {
                // A gap in the totally-ordered log: the durable prefix ends here (an earlier
                // segment's tail was torn, or a segment is missing). Stop at the contiguous cut.
                return Ok(records);
            }
            let decoded = decode_wal_record_run(&frame.payload)?;
            if decoded.len() as u64 != frame.seq_count as u64 {
                return Err(EngineError::Durability(format!(
                    "FUA WAL segment {} frame {} declares {} records but its payload decodes to {}",
                    path.display(),
                    frame.frame_id,
                    frame.seq_count,
                    decoded.len()
                )));
            }
            expected_first += frame.seq_count as u64;
            records.extend(decoded);
        }
    }
    Ok(records)
}
