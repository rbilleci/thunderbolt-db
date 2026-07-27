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
use std::sync::{Arc, Condvar, Mutex};

use gpu_db_types::EngineError;
use gpu_db_write_conveyor::{
    fua_frame_padded_bytes, recover_frame_log_by_scan, FuaControllerDecision,
    FuaControllerEligibility, FuaControllerPhase, FuaFrameLog, FuaFrameLogAppender,
    FuaFrameLogConfig, FuaFrameLogFencePool, FuaFrameLogTelemetry, FuaPhysicalController,
    FUA_CONTROLLER_QD16_FRAGMENTS,
};

use crate::{decode_wal_record_run, FuaDurabilityTelemetry, WalGroupCommitStats, WalRecord};

/// Busy-spins before falling back to `yield_now` in the durable-cut wait. Pure spin at low
/// contention keeps the p50 ack near one fence latency; the yield fallback avoids burning a core
/// when the pool is genuinely backed up. NO futex/condvar per-commit wakeups — measured law: futex
/// wakes are unpayable at high ack rates.
const SPIN_BEFORE_YIELD: u32 = 256;

fn saturating_add(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}

fn split_physical_chunks(payload: &[u8], fragments: usize) -> Option<Vec<&[u8]>> {
    if fragments < 2 || payload.len() < fragments {
        return None;
    }
    let base = payload.len() / fragments;
    let remainder = payload.len() % fragments;
    let mut offset = 0usize;
    let mut chunks = Vec::with_capacity(fragments);
    for index in 0..fragments {
        let bytes = base + usize::from(index < remainder);
        debug_assert!(bytes > 0);
        chunks.push(&payload[offset..offset + bytes]);
        offset += bytes;
    }
    Some(chunks)
}

fn padded_chunks_bytes(chunks: &[&[u8]]) -> Option<usize> {
    chunks.iter().try_fold(0usize, |total, chunk| {
        total.checked_add(fua_frame_padded_bytes(chunk.len()))
    })
}

/// The currently-open FUA segment: its frame log, the single appender, and its fence pool. On a
/// roll this whole record is replaced; `appender`/`pool` are `Option` so `Drop` (and the roll) can
/// take ownership to `finish` + `join` the lanes.
struct ActiveSegment {
    log: Arc<FuaFrameLog>,
    appender: Option<FuaFrameLogAppender>,
    pool: Option<FuaFrameLogFencePool>,
    segment_id: u64,
    /// Segment-chain aggregation. The existing active lock serializes a roll with snapshots, so
    /// retired segments are counted exactly once without retaining their staging buffers or adding
    /// an observability lock.
    retired_telemetry: FuaFrameLogTelemetry,
}

/// A successfully visible physical representation of one logical WAL group. The terminal frame
/// is the only frame whose `seq_count` advances the WAL cut, and is therefore also the correct
/// direct-service sample after the group becomes durable.
struct PublishedGroup {
    log: Arc<FuaFrameLog>,
    terminal_frame_id: u64,
    controller_sample: Option<ControllerSampleSettlement>,
}

/// Owns a published QD1 token until it is observed.  A post-publication error must not strand a
/// pending Sparse/Verify gate or leave a Fast result unaccounted: dropping this guard settles the
/// token as abandoned and conservatively returns the shared controller to sustained QD16.
struct ControllerSampleSettlement {
    controller: Arc<Mutex<FuaPhysicalController>>,
    token: Option<gpu_db_write_conveyor::FuaControllerSampleToken>,
}

impl ControllerSampleSettlement {
    fn observe(&mut self, direct_nanos: u64) {
        let Some(token) = self.token.take() else {
            return;
        };
        self.controller
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .observe_qd1_sample(token, direct_nanos);
    }

    fn unavailable(&mut self) {
        let Some(token) = self.token.take() else {
            return;
        };
        self.controller
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .record_qd1_sample_unavailable(token);
    }
}

impl Drop for ControllerSampleSettlement {
    fn drop(&mut self) {
        let Some(token) = self.token.take() else {
            return;
        };
        self.controller
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .abandon_qd1_sample(token);
    }
}

/// The pre-staged NEXT segment (Chronicle's pre-toucher). A background thread prewrites the
/// next segment file at a temp path while the active segment fills, so a roll only drains,
/// renames and swaps. Prewriting inline under the active lock (~100ms for a 64MiB segment)
/// was a visibility stall: a rolling lane blocks the CROSS-LANE contiguous cut, so every
/// lane's acks stall behind one lane's extent prewrite.
enum PrestageSlot {
    /// Nothing staged (transient: between a take and the follow-up kick, and before the
    /// constructor's first kick).
    Empty,
    /// The background thread is prewriting segment `id` at the temp path.
    Pending(u64),
    /// Segment `id` is prewritten at the temp path, ready to rename + swap in; the bool
    /// records whether it was RECYCLED from a retired file (telemetry).
    Ready(u64, Arc<FuaFrameLog>, bool),
    /// Pre-create failed; the next roll surfaces this and wedges the backend.
    Failed(String),
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
    /// Lock-free mirror of `poison.is_some()` — pumps poll poison state at iteration rate.
    poisoned: std::sync::atomic::AtomicBool,
    stats: Mutex<WalGroupCommitStats>,
    /// See [`PrestageSlot`]: the next segment, prewritten off the roll path.
    prestaged: Arc<(Mutex<PrestageSlot>, Condvar)>,
    /// RECYCLE pool (E2.5c-2): retired segment files offered back by checkpoint truncation.
    /// The pre-stager consumes one instead of prewriting a fresh file — the retired file's
    /// extents are already WRITTEN, so the ~100ms prewrite (whose fsync is a device-wide NVMe
    /// FLUSH landing during live FUA fencing) is skipped entirely. Epoch safety: the file gets
    /// a fresh monotonic segment id, so its previous life's frames are scan-rejected
    /// (`WAL_SEGMENT_FLAG_EPOCH_STAMPED` law, frame-log epoch field).
    recycle_pool: Arc<Mutex<Vec<PathBuf>>>,
    /// Segments actually recycled into service (non-vacuity telemetry).
    stat_recycled: AtomicU64,
    /// Group-level attribution not owned by an individual frame-log segment.
    stat_logical_groups: AtomicU64,
    stat_logical_payload_bytes: AtomicU64,
    stat_single_frame_padded_baseline_bytes: AtomicU64,
    stat_publish_turn_wait_ns: AtomicU64,
    stat_publish_turn_wait_groups: AtomicU64,
    stat_waiter_cut_to_observe_ns: AtomicU64,
    stat_waiter_cut_to_observe_count: AtomicU64,
    /// One physical cadence controller for this FUA backend/device, never per session/query.
    /// Its lock is outside the appender and durable-cut authorities; decisions are committed only
    /// after the frame-log's atomic batch publication succeeds.
    controller: Arc<Mutex<FuaPhysicalController>>,
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
            crate::create_wal_dir_all(parent).map_err(|err| {
                EngineError::Durability(format!(
                    "failed to create FUA WAL directory {}: {err}",
                    parent.display()
                ))
            })?;
        }
        // A plain serial WAL file at `<base>` is NOT ours to clobber — it may be a durable serial
        // log, and it would later trip the mixed-backend reopen refusal. Fail closed and make the
        // operator remove it deliberately (we only ever manage `<base>.fua.*` segments here).
        if base_path.is_file() {
            return Err(EngineError::Durability(format!(
                "cannot create a fresh FUA WAL at {}: a plain serial WAL file already exists there; \
                 remove it (it may be a durable serial log) before creating a FUA log",
                base_path.display()
            )));
        }
        remove_stale_segments(&base_path);
        let segment_id = 1;
        let log = open_segment(&base_path, segment_id, segment_bytes)?;
        let pool = log.spawn_fence_pool(lanes);
        let appender = log.appender();
        let backend = Self {
            base_path,
            lanes,
            segment_bytes,
            active: Mutex::new(ActiveSegment {
                log,
                appender: Some(appender),
                pool: Some(pool),
                segment_id,
                retired_telemetry: FuaFrameLogTelemetry::default(),
            }),
            published: AtomicUsize::new(0),
            next_ticket: AtomicU64::new(0),
            publish_cursor: AtomicU64::new(0),
            rolled_baseline: AtomicU64::new(0),
            next_segment_id: AtomicU64::new(segment_id + 1),
            poison: Mutex::new(None),
            poisoned: std::sync::atomic::AtomicBool::new(false),
            stats: Mutex::new(WalGroupCommitStats::default()),
            prestaged: Arc::new((Mutex::new(PrestageSlot::Empty), Condvar::new())),
            recycle_pool: Arc::new(Mutex::new(Vec::new())),
            stat_recycled: AtomicU64::new(0),
            stat_logical_groups: AtomicU64::new(0),
            stat_logical_payload_bytes: AtomicU64::new(0),
            stat_single_frame_padded_baseline_bytes: AtomicU64::new(0),
            stat_publish_turn_wait_ns: AtomicU64::new(0),
            stat_publish_turn_wait_groups: AtomicU64::new(0),
            stat_waiter_cut_to_observe_ns: AtomicU64::new(0),
            stat_waiter_cut_to_observe_count: AtomicU64::new(0),
            controller: Arc::new(Mutex::new(FuaPhysicalController::default())),
        };
        backend.kick_prestage();
        Ok(backend)
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
            crate::create_wal_dir_all(parent).map_err(|err| {
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
        let backend = Self {
            base_path,
            lanes,
            segment_bytes,
            active: Mutex::new(ActiveSegment {
                log,
                appender: Some(appender),
                pool: Some(pool),
                segment_id,
                retired_telemetry: FuaFrameLogTelemetry::default(),
            }),
            published: AtomicUsize::new(recovered_records),
            next_ticket: AtomicU64::new(0),
            publish_cursor: AtomicU64::new(0),
            rolled_baseline: AtomicU64::new(recovered),
            next_segment_id: AtomicU64::new(segment_id + 1),
            poison: Mutex::new(None),
            poisoned: std::sync::atomic::AtomicBool::new(false),
            stats: Mutex::new(WalGroupCommitStats::default()),
            prestaged: Arc::new((Mutex::new(PrestageSlot::Empty), Condvar::new())),
            recycle_pool: Arc::new(Mutex::new(Vec::new())),
            stat_recycled: AtomicU64::new(0),
            stat_logical_groups: AtomicU64::new(0),
            stat_logical_payload_bytes: AtomicU64::new(0),
            stat_single_frame_padded_baseline_bytes: AtomicU64::new(0),
            stat_publish_turn_wait_ns: AtomicU64::new(0),
            stat_publish_turn_wait_groups: AtomicU64::new(0),
            stat_waiter_cut_to_observe_ns: AtomicU64::new(0),
            stat_waiter_cut_to_observe_count: AtomicU64::new(0),
            controller: Arc::new(Mutex::new(FuaPhysicalController::default())),
        };
        backend.kick_prestage();
        Ok(backend)
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

    /// Aggregate publish->fence-done latency of the complete live segment chain: (ns, frames).
    pub(crate) fn fence_latency_stats(&self) -> (u64, u64) {
        let telemetry = self.telemetry();
        (
            telemetry
                .publish_to_claim_nanos
                .saturating_add(telemetry.claim_to_write_done_nanos),
            telemetry.fenced_frames,
        )
    }

    /// Return physical frame and waiter attribution across active and rolled FUA segments.
    /// Locking `active` also makes the snapshot roll-safe: a segment is either still active or
    /// already folded into `retired_telemetry`, never both.
    pub(crate) fn telemetry(&self) -> FuaDurabilityTelemetry {
        let frame = {
            let active = self.lock_active();
            let mut frame = active.retired_telemetry;
            frame.saturating_add_assign(active.log.telemetry());
            frame
        };
        let controller_guard = self.controller.lock().unwrap_or_else(|p| p.into_inner());
        let controller = controller_guard.telemetry();
        // The controller has no authority until the synchronous canonical group route publishes
        // through it.  In particular, legacy lane appends use the same physical writer but must
        // not leak a dormant controller's default phase/counters into their telemetry.
        let controller_active = controller.published_actions != 0;
        FuaDurabilityTelemetry {
            configured_fence_lanes: self.lanes as u64,
            logical_groups: self.stat_logical_groups.load(Ordering::Relaxed),
            logical_payload_bytes: self.stat_logical_payload_bytes.load(Ordering::Relaxed),
            single_frame_padded_baseline_bytes: self
                .stat_single_frame_padded_baseline_bytes
                .load(Ordering::Relaxed),
            publish_turn_wait_nanos: self.stat_publish_turn_wait_ns.load(Ordering::Relaxed),
            publish_turn_wait_groups: self.stat_publish_turn_wait_groups.load(Ordering::Relaxed),
            published_frames: frame.published_frames,
            fenced_frames: frame.fenced_frames,
            fence_failures: frame.fence_failures,
            payload_bytes: frame.payload_bytes,
            padded_bytes: frame.padded_bytes,
            stage_copy_nanos: frame.stage_copy_nanos,
            stage_copy_frames: frame.stage_copy_frames,
            publish_to_claim_nanos: frame.publish_to_claim_nanos,
            publish_to_claim_frames: frame.publish_to_claim_frames,
            claim_to_write_done_nanos: frame.claim_to_write_done_nanos,
            claim_to_write_done_frames: frame.claim_to_write_done_frames,
            write_done_to_contiguous_cut_nanos: frame.write_done_to_contiguous_cut_nanos,
            write_done_to_contiguous_cut_frames: frame.write_done_to_contiguous_cut_frames,
            contiguous_cut_events: frame.contiguous_cut_events,
            contiguous_cut_advanced_frames: frame.contiguous_cut_advanced_frames,
            contiguous_cut_advance_max_frames: frame.contiguous_cut_advance_max_frames,
            waiter_cut_to_observe_nanos: self.stat_waiter_cut_to_observe_ns.load(Ordering::Relaxed),
            waiter_cut_to_observe_count: self
                .stat_waiter_cut_to_observe_count
                .load(Ordering::Relaxed),
            in_flight_depth_max: frame.in_flight_depth_max,
            in_flight_depth_histogram: frame.in_flight_depth_histogram,
            controller_sustained_actions: controller.sustained_qd16_actions,
            controller_pending_probe_cover_actions: controller.pending_probe_cover_actions,
            controller_qd1_samples: controller.qd1_probe_actions,
            controller_qd1_sparse_actions: controller.qd1_sparse_actions,
            controller_qd1_verify_actions: controller.qd1_verify_actions,
            controller_qd1_fast_actions: controller.qd1_fast_actions,
            controller_unfragmented_actions: controller.unfragmented_actions,
            controller_pool_too_narrow: controller.pool_too_narrow,
            controller_empty_chunk: controller.empty_chunk,
            controller_insufficient_free_slots: controller.insufficient_free_slots,
            controller_natural_depth: controller.natural_depth,
            controller_segment_boundary: controller.segment_boundary,
            controller_amplification_cap: controller.amplification_cap,
            controller_fast_samples: controller.fast_probes,
            controller_nonfast_samples: controller.gray_or_slow_probes,
            controller_transitions_to_verify: controller.transitions_to_verify,
            controller_transitions_to_fast: controller.transitions_to_fast,
            controller_transitions_to_sustained: controller.transitions_to_sustained,
            controller_stale_qd1_samples: controller.stale_qd1_samples,
            controller_unavailable_qd1_samples: controller.unavailable_qd1_samples,
            controller_abandoned_qd1_samples: controller.abandoned_qd1_samples,
            controller_protocol_faults: controller.protocol_faults,
            controller_protocol_fallback_actions: controller.protocol_fallback_actions,
            controller_phase: if controller_active {
                match controller.phase {
                    FuaControllerPhase::SustainedSlow => 1,
                    FuaControllerPhase::Verify => 2,
                    FuaControllerPhase::Fast => 3,
                }
            } else {
                0
            },
            controller_verify_fast_streak: controller.verify_fast_streak as u64,
            controller_sustained_remaining: if controller_active {
                controller.sustained_qd16_remaining as u64
            } else {
                0
            },
            controller_generation: controller.generation,
            controller_pending_qd1_samples: controller.pending_qd1_samples,
            controller_fast_in_flight: controller.fast_in_flight,
            controller_fast_in_flight_max: controller.fast_in_flight_max,
            controller_generation_exhausted: controller.generation_exhausted,
            controller_ordinal_exhausted: controller.ordinal_exhausted,
            controller_action_reconciliation: u64::from(
                controller_active && controller_guard.action_reconciliation_ok(),
            ),
            controller_sample_reconciliation: u64::from(
                controller_active && controller_guard.sample_reconciliation_ok(),
            ),
        }
    }

    /// Segments recycled into service by the pre-stager (non-vacuity telemetry).
    pub(crate) fn recycled_segments(&self) -> u64 {
        self.stat_recycled.load(Ordering::Relaxed)
    }

    /// E2.5c-2 CHECKPOINT TRUNCATION: retire every NON-ACTIVE segment whose frames all lie
    /// below `cut_end` (its records are checkpoint-covered and never needed by recovery again —
    /// the caller made the checkpoint + its baseline sidecar durable FIRST). One retired file is
    /// kept in the recycle pool for the pre-stager; the rest are deleted. Returns the number of
    /// segments retired. Concurrent-safe with live appends: the active segment (and anything
    /// newer) is never touched, and retired segments are by definition rolled-away (drained,
    /// byte-frozen).
    pub(crate) fn retire_segments_below(&self, cut_end: u64) -> Result<usize, EngineError> {
        let active_id = self.lock_active().segment_id;
        let Some(stem) = self.base_path.file_name().and_then(|n| n.to_str()) else {
            return Ok(0);
        };
        let mut retired = 0usize;
        for path in fua_segment_paths_sorted(&self.base_path)? {
            let Some(id) = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|name| parse_segment_id(name, stem))
            else {
                continue;
            };
            if id >= active_id {
                continue; // never touch the active (or a newer) segment
            }
            {
                // Already offered to the recycle pool by an earlier pass — leave it for the
                // pre-stager (deleting it here would strand a dangling pool entry).
                let pool = self.recycle_pool.lock().unwrap_or_else(|p| p.into_inner());
                if pool.iter().any(|pooled| pooled == &path) {
                    continue;
                }
            }
            let frames = recover_frame_log_by_scan(&path).map_err(|err| {
                EngineError::Durability(format!(
                    "failed to scan FUA WAL segment {} for retirement: {err}",
                    path.display()
                ))
            })?;
            let max_end = frames
                .iter()
                .map(|frame| frame.first_seq + frame.seq_count as u64)
                .max()
                .unwrap_or(0);
            if max_end > cut_end {
                continue; // still holds records above the checkpoint baseline
            }
            let mut pool = self.recycle_pool.lock().unwrap_or_else(|p| p.into_inner());
            if pool.is_empty() {
                pool.push(path);
            } else {
                drop(pool);
                std::fs::remove_file(&path).map_err(|err| {
                    EngineError::Durability(format!(
                        "failed to delete retired FUA WAL segment {}: {err}",
                        path.display()
                    ))
                })?;
                sync_parent_dir(&path).map_err(|err| {
                    EngineError::Durability(format!(
                        "failed to fsync FUA WAL directory after retiring {}: {err}",
                        path.display()
                    ))
                })?;
            }
            retired += 1;
        }
        Ok(retired)
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
        // Flag AFTER the reason is stored (under the lock) so a reader that
        // observes the flag always finds a reason.
        self.poisoned
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Lock-free poison probe (see the lane-set settle path).
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned.load(std::sync::atomic::Ordering::Acquire)
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

    /// Atomically publish one ordered logical group into the canonical FUA log. The shared
    /// controller may choose a bounded physical subframe batch, but records, ticket order,
    /// segment ownership, WAL cursor, and acknowledgement remain this backend's sole authority.
    fn publish(
        &self,
        ticket: u64,
        payload: &[u8],
        first_seq: u64,
        seq_count: u32,
        controller_enabled: bool,
    ) -> Result<PublishedGroup, EngineError> {
        // Wait our turn: publish frames in ticket (== `first_seq`) order. This is the ONLY ordering
        // point and it only gates a staging memcpy; the fence pool then pipelines the durability of
        // every published frame concurrently. A prior ticket that wedged the backend without
        // advancing the cursor is surfaced as poison so later tickets abort rather than hang.
        let publish_turn_started = std::time::Instant::now();
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
        saturating_add(
            &self.stat_publish_turn_wait_ns,
            publish_turn_started
                .elapsed()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
        );
        saturating_add(&self.stat_publish_turn_wait_groups, 1);
        let mut active = self.lock_active();
        loop {
            while active.log.free_fence_slots(self.lanes) == 0 {
                if active.log.fence_failed() {
                    self.set_poison("FUA fence lane failed");
                    return Err(self.poison_error("FUA fence lane failed"));
                }
                std::thread::yield_now();
            }
            let free_slots = active.log.free_fence_slots(self.lanes);
            let natural_depth = self.lanes.saturating_sub(free_slots);
            let fragment_count = FUA_CONTROLLER_QD16_FRAGMENTS
                .saturating_sub(natural_depth.min(FUA_CONTROLLER_QD16_FRAGMENTS));
            let chunks = split_physical_chunks(payload, fragment_count);
            let fragmented_padded_bytes = chunks
                .as_deref()
                .and_then(padded_chunks_bytes)
                .unwrap_or(usize::MAX);
            let single_frame_padded_bytes = fua_frame_padded_bytes(payload.len());
            let batch_fit = chunks.as_deref().map(|chunks| {
                active
                    .appender
                    .as_ref()
                    .expect("FUA appender present")
                    .can_publish_batch(chunks)
            });
            let one_segment = matches!(batch_fit, Some(Ok(())));
            let eligibility = FuaControllerEligibility {
                pool_lanes: self.lanes,
                natural_depth,
                free_slots,
                fragment_count,
                chunks_nonempty: chunks.is_some(),
                one_segment,
                single_frame_padded_bytes,
                fragmented_padded_bytes,
            };
            // If the controller wants a fragmented group and this live segment has only a suffix
            // left, drain/roll BEFORE touching any batch cursor. A fresh segment may admit the
            // complete group; a segment whose geometry is too small remains an ineligible
            // unfragmented group without an infinite roll loop.
            let wants_fragmented = controller_enabled
                && self
                    .controller
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .select(FuaControllerEligibility {
                        one_segment: true,
                        ..eligibility
                    })
                    .fragments()
                    >= 2;
            if wants_fragmented
                && !one_segment
                && fragmented_padded_bytes <= self.segment_bytes
                && matches!(batch_fit, Some(Err(ref err)) if err.kind() == std::io::ErrorKind::StorageFull)
            {
                self.roll(&mut active)?;
                continue;
            }
            // The final choice, atomic frame-group visibility publication, and token issue are
            // one controller critical section.  A fence completion can therefore never advance
            // generation between selection and token creation.
            let mut controller = controller_enabled
                .then(|| self.controller.lock().unwrap_or_else(|p| p.into_inner()));
            let decision = controller.as_deref().map_or(
                FuaControllerDecision::Unfragmented(
                    gpu_db_write_conveyor::FuaControllerReason::NaturalDepth,
                ),
                |controller| controller.select(eligibility),
            );
            let result = match decision {
                FuaControllerDecision::SustainedEpoch { .. }
                | FuaControllerDecision::PendingProbeCover { .. } => active
                    .appender
                    .as_mut()
                    .expect("FUA appender present")
                    .publish_batch(
                        chunks.as_deref().expect("eligible fragmented chunks"),
                        first_seq,
                        seq_count,
                    )
                    .map(|handle| handle.terminal_frame_id),
                FuaControllerDecision::Qd1Sample { .. }
                | FuaControllerDecision::Unfragmented(_) => active
                    .appender
                    .as_mut()
                    .expect("FUA appender present")
                    .publish_frame(payload, first_seq, seq_count)
                    .map(|handle| handle.frame_id),
            };
            match result {
                Ok(terminal_frame_id) => {
                    let log = Arc::clone(&active.log);
                    let controller_sample = controller
                        .as_deref_mut()
                        .and_then(|controller| controller.record_published(decision))
                        .map(|token| ControllerSampleSettlement {
                            controller: Arc::clone(&self.controller),
                            token: Some(token),
                        });
                    saturating_add(&self.stat_logical_groups, 1);
                    saturating_add(&self.stat_logical_payload_bytes, payload.len() as u64);
                    saturating_add(
                        &self.stat_single_frame_padded_baseline_bytes,
                        fua_frame_padded_bytes(payload.len()) as u64,
                    );
                    // Release the turn so the next ticket can publish. Store while holding the
                    // active lock so the memcpy is fully visible before the successor publishes.
                    self.publish_cursor.store(ticket + 1, Ordering::Release);
                    return Ok(PublishedGroup {
                        log,
                        terminal_frame_id,
                        controller_sample,
                    });
                }
                Err(err) if err.kind() == std::io::ErrorKind::StorageFull => {
                    // This may only be the unfragmented fallback (fragmented batches were
                    // preflighted above). Nothing was visible, so a roll preserves all-or-nothing
                    // group ownership and lets the same ticket retry on one fresh segment.
                    self.roll(&mut active)?;
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    // Slot ring momentarily full despite pacing (shouldn't happen with lanes <<
                    // ring); back off and retry. Re-check the fail-closed state EVERY iteration so a
                    // poisoned/fence-failed backend aborts here instead of spinning forever.
                    if active.log.fence_failed() {
                        self.set_poison("FUA fence lane failed");
                        return Err(self.poison_error("FUA fence lane failed"));
                    }
                    if let Some(reason) = self.poison_reason() {
                        return Err(self.poison_error(&reason));
                    }
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

    /// Kick the background pre-create of the NEXT segment: claim its id now (ids must ascend in
    /// roll order for recovery) and prewrite the file at the TEMP path off-thread. The temp name
    /// keeps half-prewritten files invisible to recovery/`highest_existing_segment_id` (both parse
    /// only `<base>.fua.<id>` names); the roll renames it into place.
    fn kick_prestage(&self) {
        let id = self.next_segment_id.fetch_add(1, Ordering::AcqRel);
        {
            let (lock, _) = &*self.prestaged;
            *lock.lock().unwrap_or_else(|p| p.into_inner()) = PrestageSlot::Pending(id);
        }
        let slot = Arc::clone(&self.prestaged);
        let temp = prestage_file_path(&self.base_path);
        let segment_bytes = self.segment_bytes;
        let recycle_pool = Arc::clone(&self.recycle_pool);
        std::thread::spawn(move || {
            // AUDIT (minor): the slot MUST leave Pending even if this body
            // panics — Drop and take_prestaged wait on the Ready/Failed
            // transition with no timeout; a panicking pre-stager wedges
            // FAIL-CLOSED (Failed) instead of hanging shutdown/roll.
            let body = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || -> std::io::Result<(Arc<FuaFrameLog>, bool)> {
                    // A stale temp from a crashed prior life (or an unrolled leftover) is ours
                    // to clobber.
                    match std::fs::remove_file(&temp) {
                        Ok(()) => sync_parent_dir(&temp)?,
                        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                        Err(err) => return Err(err),
                    }
                    // RECYCLE arm (E2.5c-2): reuse a retired segment file when one is offered —
                    // its extents are already written, so the whole prewrite (fallocate +
                    // zero-fill + the fsync that lands a device-wide NVMe FLUSH during live
                    // fencing) is skipped. The fresh monotonic id epoch-stamps the header so the
                    // previous life's frames are scan-rejected. Any recycle failure (geometry
                    // drift, rename error) falls back to the fresh-create arm — recycle is an
                    // optimization, never a correctness gate.
                    let retired = recycle_pool.lock().unwrap_or_else(|p| p.into_inner()).pop();
                    let recycled = if let Some(retired) = retired {
                        std::fs::rename(&retired, &temp)?;
                        sync_parent_dir(&temp)?;
                        let config = FuaFrameLogConfig {
                            path: temp.clone(),
                            segment_id: id,
                            capacity_bytes: segment_bytes,
                        };
                        // Safety: the temp path is owned exclusively by this backend (one
                        // prestage in flight; renamed to its final segment name before any
                        // other opener).
                        match unsafe { FuaFrameLog::recycle(config) } {
                            Ok(log) => Some(log),
                            Err(_) => {
                                std::fs::remove_file(&temp)?;
                                sync_parent_dir(&temp)?;
                                None
                            }
                        }
                    } else {
                        None
                    };
                    match recycled {
                        Some(log) => Ok((log, true)),
                        None => {
                            // Fresh create (a failed recycle above may have left a stale temp —
                            // clobber).
                            match std::fs::remove_file(&temp) {
                                Ok(()) => sync_parent_dir(&temp)?,
                                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                                Err(err) => return Err(err),
                            }
                            let config = FuaFrameLogConfig {
                                path: temp.clone(),
                                segment_id: id,
                                capacity_bytes: segment_bytes,
                            };
                            // Safety: as above — exclusive temp-path ownership.
                            unsafe { FuaFrameLog::create(config) }.map(|log| (log, false))
                        }
                    }
                },
            ));
            let (lock, cvar) = &*slot;
            let mut guard = lock.lock().unwrap_or_else(|p| p.into_inner());
            *guard = match body {
                Ok(Ok((log, was_recycled))) => PrestageSlot::Ready(id, log, was_recycled),
                Ok(Err(err)) => PrestageSlot::Failed(format!("{err}")),
                Err(_) => PrestageSlot::Failed("pre-stager thread panicked".to_string()),
            };
            cvar.notify_all();
        });
    }

    /// Take the pre-staged next segment, waiting if the prewrite is still in flight (a roll
    /// arriving before ~100ms of prewrite finishes — only under tiny test segments), and rename
    /// it to its final `<base>.fua.<id>` name. The rename (+ parent dir fsync) must be durable
    /// BEFORE any frame lands in the segment: acked commits would otherwise sit in a file
    /// recovery ignores.
    fn take_prestaged(&self) -> Result<(u64, Arc<FuaFrameLog>), EngineError> {
        let (lock, cvar) = &*self.prestaged;
        let mut guard = lock.lock().unwrap_or_else(|p| p.into_inner());
        let (id, log, was_recycled) = loop {
            match std::mem::replace(&mut *guard, PrestageSlot::Empty) {
                PrestageSlot::Ready(id, log, was_recycled) => break (id, log, was_recycled),
                PrestageSlot::Failed(reason) => {
                    return Err(EngineError::Durability(format!(
                        "FUA segment pre-create failed: {reason}"
                    )));
                }
                PrestageSlot::Pending(id) => {
                    *guard = PrestageSlot::Pending(id);
                    guard = cvar.wait(guard).unwrap_or_else(|p| p.into_inner());
                }
                PrestageSlot::Empty => {
                    // No prestage in flight (constructor always kicks one; defensive): create
                    // inline exactly like the pre-prestager roll did.
                    drop(guard);
                    let id = self.next_segment_id.fetch_add(1, Ordering::AcqRel);
                    let log = open_segment(&self.base_path, id, self.segment_bytes)?;
                    return Ok((id, log));
                }
            }
        };
        drop(guard);
        let temp = prestage_file_path(&self.base_path);
        let final_path = segment_file_path(&self.base_path, id);
        std::fs::rename(&temp, &final_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to rename pre-staged FUA segment {} -> {}: {err}",
                temp.display(),
                final_path.display()
            ))
        })?;
        sync_parent_dir(&final_path).map_err(|err| {
            EngineError::Durability(format!(
                "failed to fsync FUA WAL directory after segment rename: {err}"
            ))
        })?;
        if was_recycled {
            self.stat_recycled.fetch_add(1, Ordering::Relaxed);
        }
        Ok((id, log))
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
        let (new_id, new_log) = self
            .take_prestaged()
            .inspect_err(|err| self.set_poison(&format!("FUA segment roll failed: {err}")))?;
        // Start prewriting the FOLLOWING segment while this one fills (the whole point: the
        // ~100ms extent prewrite runs concurrent with normal appends, never under this lock).
        self.kick_prestage();
        let new_pool = new_log.spawn_fence_pool(self.lanes);
        let new_appender = new_log.appender();
        active
            .retired_telemetry
            .saturating_add_assign(active.log.telemetry());
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
        let mut published = self.publish(ticket, payload, first_seq, seq_count, true)?;
        let mut spins = 0u32;
        loop {
            if published.log.durable_seq() >= target as u64 {
                self.record_waiter_cut_observation(&published.log);
                if let Some(sample) = published.controller_sample.as_mut() {
                    if let Some(direct_nanos) = published
                        .log
                        .frame_direct_write_service_nanos(published.terminal_frame_id)
                    {
                        sample.observe(direct_nanos);
                    } else {
                        sample.unavailable();
                    }
                }
                break;
            }
            if published.log.fence_failed() {
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

    /// Append ONE frame carrying an EXPLICIT global-seq range and return WITHOUT waiting for
    /// durability (the fence pool pipelines it). `first_seq` is the GLOBAL commit sequence of the
    /// frame's first record and `seq_count` the record count (== the number of contiguous global
    /// seqs the frame covers); the durable cut (`durable_records`) reports the largest global end
    /// of this lane's contiguous-durable frame prefix. This is the per-lane primitive behind
    /// [`crate::FuaWalLaneSet`] — unlike [`Self::commit_group`] it does not block on the cut, so a
    /// caller drives N lanes independently and polls the cross-lane merged cut. This legacy/test
    /// lane primitive deliberately preserves the canonical appender, roll, fence-pool, and
    /// recovery representation while remaining controller-disabled: it has no synchronous QD1
    /// direct-service observation owner. Single-writer per lane is assumed (tickets serialize the
    /// staging memcpy; the fence pool then pipelines).
    pub(crate) fn append_frame(
        &self,
        payload: &[u8],
        first_seq: u64,
        seq_count: u32,
    ) -> Result<(), EngineError> {
        if let Some(reason) = self.poison_reason() {
            return Err(self.poison_error(&reason));
        }
        let ticket = self.next_ticket();
        self.publish(ticket, payload, first_seq, seq_count, false)?;
        let mut stats = self.stats.lock().unwrap_or_else(|p| p.into_inner());
        stats.flush_groups += 1;
        stats.durable_records += seq_count as u64;
        stats.max_group_size = stats.max_group_size.max(seq_count as usize);
        Ok(())
    }

    /// Wait (spin-then-yield) until every record up to `target` is durable — the inline
    /// `flush_all` completion for the FUA backend after its own group (if any) committed.
    pub(crate) fn wait_durable(&self, target: usize) -> Result<(), EngineError> {
        let mut spins = 0u32;
        loop {
            if let Some(reason) = self.poison_reason() {
                return Err(self.poison_error(&reason));
            }
            let (durable, failed, cut_to_observe) = {
                let active = self.lock_active();
                let active_durable = active.log.durable_seq();
                (
                    active_durable.max(self.rolled_baseline.load(Ordering::Acquire)),
                    active.log.fence_failed(),
                    (active_durable >= target as u64)
                        .then(|| active.log.durable_cut_to_observe_nanos())
                        .flatten(),
                )
            };
            if durable >= target as u64 {
                if let Some(nanos) = cut_to_observe {
                    saturating_add(&self.stat_waiter_cut_to_observe_ns, nanos);
                    saturating_add(&self.stat_waiter_cut_to_observe_count, 1);
                }
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

    fn record_waiter_cut_observation(&self, log: &FuaFrameLog) {
        if let Some(nanos) = log.durable_cut_to_observe_nanos() {
            saturating_add(&self.stat_waiter_cut_to_observe_ns, nanos);
            saturating_add(&self.stat_waiter_cut_to_observe_count, 1);
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
        // Drain the pre-stager: a Pending thread still owns the temp path, and letting it outlive
        // this backend could race a same-path successor's kick (its remove_file) in a rapid
        // drop+reopen. The thread's slot fill is its last temp-path-relevant action, so waiting
        // for Pending to clear is a full ownership handoff.
        let (lock, cvar) = &*self.prestaged;
        let mut guard = lock.lock().unwrap_or_else(|p| p.into_inner());
        while matches!(*guard, PrestageSlot::Pending(_)) {
            guard = cvar.wait(guard).unwrap_or_else(|p| p.into_inner());
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

/// `<base>.fua.prestage` — the temp path pre-created segments are prewritten at. The non-numeric
/// suffix keeps the file invisible to [`parse_segment_id`] (recovery, reopen id scan, stale-file
/// clobber) until the roll renames it to its final `<base>.fua.<id>` name.
fn prestage_file_path(base: &Path) -> PathBuf {
    let name = base
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("wal.segment");
    base.with_file_name(format!("{name}.fua.prestage"))
}

/// fsync the parent directory so a just-renamed segment file's directory entry is durable.
fn sync_parent_dir(path: &Path) -> std::io::Result<()> {
    let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) else {
        return Ok(());
    };
    std::fs::File::open(parent)?.sync_all()
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

/// Every `<base>.fua.<id>` segment file beside `base`, ascending by id (empty when none exist).
/// Shared by [`recover_fua_wal_records`] and the lane-set merge recovery so both walk segments in
/// the same total order.
pub(crate) fn fua_segment_paths_sorted(base: &Path) -> Result<Vec<PathBuf>, EngineError> {
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
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(EngineError::Durability(format!(
                "failed to enumerate FUA WAL segments in {}: {err}",
                parent.display()
            )));
        }
    }
    segments.sort_by_key(|(id, _)| *id);
    Ok(segments.into_iter().map(|(_, path)| path).collect())
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
    let log = unsafe { FuaFrameLog::create(config) }.map_err(|err| {
        EngineError::Durability(format!(
            "failed to create FUA WAL segment {}: {err}",
            path.display()
        ))
    })?;
    sync_parent_dir(&path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to fsync FUA WAL directory after creating {}: {err}",
            path.display()
        ))
    })?;
    Ok(log)
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
    let mut removed = false;
    for entry in entries.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            if parse_segment_id(name, stem).is_some() && std::fs::remove_file(entry.path()).is_ok()
            {
                removed = true;
            }
        }
    }
    if removed {
        let _ = sync_parent_dir(base);
    }
}

/// One complete logical WAL run recovered from the physical frames of one segment.  This is the
/// sole physical-fragment coalescer: single-log and lane-set recovery both decode these runs.
pub(crate) struct RecoveredFuaWalRun {
    pub first_seq: u64,
    pub seq_count: u32,
    pub terminal_frame_id: u64,
    pub payload: Vec<u8>,
}

pub(crate) fn recover_fua_wal_runs(path: &Path) -> Result<Vec<RecoveredFuaWalRun>, EngineError> {
    let frames = recover_frame_log_by_scan(path).map_err(|err| {
        EngineError::Durability(format!(
            "failed to scan-recover FUA WAL segment {}: {err}",
            path.display()
        ))
    })?;
    let mut runs = Vec::new();
    let mut pending: Option<(gpu_db_write_conveyor::FrameGroupMetadata, u64, Vec<u8>)> = None;
    for frame in frames {
        match frame.group {
            None => {
                if pending.is_some() {
                    return Err(EngineError::Durability(format!(
                        "FUA WAL segment {} interleaves a legacy frame into a fragmented group",
                        path.display()
                    )));
                }
                runs.push(RecoveredFuaWalRun {
                    first_seq: frame.first_seq,
                    seq_count: frame.seq_count,
                    terminal_frame_id: frame.frame_id,
                    payload: frame.payload,
                });
            }
            Some(metadata) if metadata.fragment_index == 0 => {
                if pending.is_some() {
                    return Err(EngineError::Durability(format!(
                        "FUA WAL segment {} starts a fragmented group before the prior group ends",
                        path.display()
                    )));
                }
                pending = Some((metadata, frame.first_seq, frame.payload));
            }
            Some(metadata) => {
                let Some((expected, first_seq, payload)) = pending.as_mut() else {
                    return Err(EngineError::Durability(format!(
                        "FUA WAL segment {} frame {} has a fragmented continuation without a prefix",
                        path.display(), frame.frame_id
                    )));
                };
                if metadata.total_payload_bytes != expected.total_payload_bytes
                    || metadata.fragment_count != expected.fragment_count
                    || metadata.group_crc32c != expected.group_crc32c
                    || metadata.fragment_index == 0
                    || frame.first_seq != *first_seq
                {
                    return Err(EngineError::Durability(format!(
                        "FUA WAL segment {} frame {} has inconsistent fragmented metadata",
                        path.display(),
                        frame.frame_id
                    )));
                }
                payload.extend_from_slice(&frame.payload);
                if metadata.fragment_index + 1 == metadata.fragment_count {
                    let (expected, first_seq, payload) = pending.take().expect("group is pending");
                    if payload.len() as u64 != expected.total_payload_bytes {
                        return Err(EngineError::Durability(format!(
                            "FUA WAL segment {} fragmented group ending at frame {} fails length validation",
                            path.display(), frame.frame_id
                        )));
                    }
                    runs.push(RecoveredFuaWalRun {
                        first_seq,
                        seq_count: frame.seq_count,
                        terminal_frame_id: frame.frame_id,
                        payload,
                    });
                }
            }
        }
    }
    if pending.is_some() {
        return Err(EngineError::Durability(format!(
            "FUA WAL segment {} exposes an incomplete fragmented group after scan validation",
            path.display()
        )));
    }
    Ok(runs)
}

/// Recover the totally-ordered `WalRecord` history from a FUA-durable database at `base_path`.
pub fn recover_fua_wal_records(base_path: impl AsRef<Path>) -> Result<Vec<WalRecord>, EngineError> {
    let mut records = Vec::new();
    let mut expected_first = 0u64;
    for path in fua_segment_paths_sorted(base_path.as_ref())? {
        for run in recover_fua_wal_runs(&path)? {
            if run.first_seq != expected_first {
                return Ok(records);
            }
            let decoded = decode_wal_record_run(&run.payload)?;
            if decoded.len() as u64 != u64::from(run.seq_count) {
                return Err(EngineError::Durability(format!(
                    "FUA WAL segment {} terminal frame {} declares {} records but its run decodes to {}",
                    path.display(), run.terminal_frame_id, run.seq_count, decoded.len()
                )));
            }
            expected_first = run
                .first_seq
                .checked_add(u64::from(run.seq_count))
                .ok_or_else(|| {
                    EngineError::Durability("FUA WAL recovery sequence range overflow".to_string())
                })?;
            records.extend(decoded);
        }
    }
    Ok(records)
}
