//! WAL buffering, group flush, and durable segment ownership.
//!
//! Callers must exclusively own a durable WAL base/path for each buffer or artifact lifetime;
//! under that precondition, initial identity install and later require-only handoffs fail closed.

use super::*;
use crate::identity::{check_or_install_durable_identity, require_durable_identity};
#[cfg(all(test, unix))]
use gpu_db_types::DurabilityStage;
use gpu_db_types::{DurabilityFault, DurabilityPoison};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) mod group;
mod typed_exact;
pub use typed_exact::WalTypedExactAppendReservation;

static NEXT_CANONICAL_APPEND_RESERVATION_OWNER_ID: AtomicU64 = AtomicU64::new(1);

fn next_canonical_append_reservation_owner_id() -> u64 {
    let owner_id = NEXT_CANONICAL_APPEND_RESERVATION_OWNER_ID.fetch_add(1, Ordering::Relaxed);
    assert_ne!(
        owner_id, 0,
        "WAL canonical append reservation owner identifiers exhausted"
    );
    owner_id
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
    /// First fixed post-handoff exact fault. Once set, it is immutable and dominates every later
    /// dynamic compatibility/admin poison diagnostic for this live backing.
    fixed_poison: DurabilityPoison,
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
            fixed_poison: DurabilityPoison::new(),
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
        if let Some(fault) = self.fixed_fault() {
            return EngineError::DurabilityFault(fault);
        }
        EngineError::Durability(format!(
            "WAL segment {} is poisoned by an earlier flush failure ({reason}); restart to \
             recover from the durable prefix",
            self.segment_path.display()
        ))
    }

    pub(super) fn install_fixed_fault(&self, fault: DurabilityFault) -> DurabilityFault {
        self.fixed_poison.install(fault)
    }

    pub(super) fn fixed_fault(&self) -> Option<DurabilityFault> {
        self.fixed_poison.snapshot()
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
            crate::create_wal_dir_all(parent).map_err(|err| {
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

    /// Establish a zero-filled positional-I/O extent before a typed record is proposed or before
    /// a legacy compatibility group hands ownership off.  A returned serial job never creates,
    /// reserves, or extends the file.
    fn ensure_preallocated_through(
        &self,
        state: &mut WalDurableState,
        write_end: u64,
    ) -> Result<(), EngineError> {
        if write_end <= state.prealloc_bytes {
            return Ok(());
        }
        let new_prealloc = write_end
            .max(
                state
                    .prealloc_bytes
                    .saturating_add(wal_prealloc_chunk_bytes()),
            )
            .max(wal_prealloc_chunk_bytes());
        let file = state.file.as_ref().ok_or_else(|| {
            EngineError::Durability("WAL preallocation requires an open segment handle".to_string())
        })?;
        zero_fill_extend(file, state.prealloc_bytes, new_prealloc).map_err(|err| {
            EngineError::Durability(format!(
                "failed to extend WAL segment preallocation {}: {err}",
                self.segment_path.display()
            ))
        })?;
        state.prealloc_bytes = new_prealloc;
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
// Keep jobs inline: `begin_group_flush` runs after logical claim, where boxing a job would create
// a fallible post-WAL allocation and an alternate failure path.
#[allow(clippy::large_enum_variant)]
pub enum WalGroupFlushBegin {
    /// Unflushed records were snapshotted; run [`WalGroupFlushJob::commit`] to make them durable.
    Job(WalGroupFlushJob),
    /// Nothing to flush — every appended record is already durable up to `flushed_records`.
    Clean { flushed_records: usize },
    /// Every bounded permanent descriptor is owned by an in-flight/forming group.  This is
    /// explicit backpressure: callers wait or retry; a claimed exact record is never re-encoded,
    /// copied into a fallback buffer, or failed for post-claim capacity.
    Busy,
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

// FUA handoff ownership is inline for the same post-claim no-allocation invariant.
#[allow(clippy::large_enum_variant)]
enum WalGroupFlushJobKind {
    /// The serial-fdatasync backend: one positional `write_all` + `fdatasync` on the live segment.
    Serial(group::SerialFlushJob),
    /// The FUA fence-pool backend: publish the presealed frame run and wait for its contiguous
    /// durable cut. Fence-lane parallelism is physical implementation detail, not a second
    /// logical durability lifecycle.
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

/// Per-buffer proof that the in-memory WAL history has one immutable canonical lineage.
///
/// The public artifact writers deliberately validate their complete supplied slice. A live
/// append-only buffer instead verifies only its newly appended suffix, installing the small
/// durable sidecar only when it first binds and requiring it on every later durability handoff.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct DurableIdentityBinding {
    identity: Option<CanonicalIdentity>,
    verified_records: usize,
    #[cfg(test)]
    decoded_records_for_test: usize,
}

impl DurableIdentityBinding {
    fn bound(identity: CanonicalIdentity, verified_records: usize) -> Self {
        Self {
            identity: Some(identity),
            verified_records,
            #[cfg(test)]
            decoded_records_for_test: 0,
        }
    }

    fn canonical_identity_in_records(
        records: &[WalRecord],
    ) -> Result<Option<CanonicalIdentity>, EngineError> {
        let mut identity = None;
        for record in records {
            let Some(envelope) = decode_canonical_record_payload(&record.payload)? else {
                continue;
            };
            match identity {
                None => identity = Some(envelope.header.identity),
                Some(expected) if expected == envelope.header.identity => {}
                Some(_) => {
                    return Err(EngineError::Durability(
                        "canonical WAL records span multiple durable identities".to_string(),
                    ));
                }
            }
        }
        Ok(identity)
    }

    /// Raw recovery derives its canonical lineage before it touches any serial segment or FUA
    /// backend state. Canonical history is already durable, so its sidecar must already exist;
    /// only all-legacy recovery stays unbound and may install on its first later canonical write.
    fn for_recovered_history(base: &Path, records: &[WalRecord]) -> Result<Self, EngineError> {
        let Some(identity) = Self::canonical_identity_in_records(records)? else {
            return Ok(Self {
                verified_records: records.len(),
                ..Self::default()
            });
        };
        require_durable_identity(base, identity)?;
        Ok(Self::bound(identity, records.len()))
    }

    /// An explicit recovered constructor may skip the live hot-path scan only after it proves
    /// every canonical record in the supplied recovery history agrees with its checked anchor.
    /// Legacy records are intentionally neutral: they neither establish nor alter lineage.
    fn validate_recovered_records(
        records: &[WalRecord],
        identity: CanonicalIdentity,
    ) -> Result<(), EngineError> {
        match Self::canonical_identity_in_records(records)? {
            Some(found) if found != identity => Err(EngineError::Durability(
                "recovered canonical WAL records do not match the durable identity anchor"
                    .to_string(),
            )),
            _ => Ok(()),
        }
    }

    /// Decode only the unverified tail, then install the tiny identity sidecar on the initial
    /// bind or require the installed sidecar thereafter. The binding cursor advances only after
    /// that operation succeeds, so a failed anchor check cannot hide an unverified record.
    fn verify_through(
        &mut self,
        base: &Path,
        records: &[WalRecord],
        target: usize,
    ) -> Result<(), EngineError> {
        debug_assert!(target <= records.len());
        debug_assert!(self.verified_records <= target);
        if self.verified_records == target {
            return Ok(());
        }
        let was_bound = self.identity.is_some();
        let mut identity = self.identity;
        for record in &records[self.verified_records..target] {
            #[cfg(test)]
            {
                self.decoded_records_for_test += 1;
            }
            let Some(envelope) = decode_canonical_record_payload(&record.payload)? else {
                continue;
            };
            match identity {
                None => identity = Some(envelope.header.identity),
                Some(expected) if expected == envelope.header.identity => {}
                Some(_) => {
                    return Err(EngineError::Durability(
                        "canonical WAL records span multiple durable identities".to_string(),
                    ));
                }
            }
        }
        if let Some(identity) = identity {
            if was_bound {
                require_durable_identity(base, identity)?;
            } else {
                check_or_install_durable_identity(base, identity)?;
            }
        }
        self.identity = identity;
        self.verified_records = target;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CanonicalCatalogTailCache {
    /// The cache describes exactly `record_count` logical records. `None` is the known
    /// legacy/empty boundary; its genesis digest remains an engine concern because it depends on
    /// the current engine identity.
    Known {
        record_count: usize,
        tail: Option<CanonicalCatalogTail>,
    },
    /// A generic append/reinstatement/recovery install changed the logical history without a
    /// sealed canonical tail. Decode the actual last record on the next boundary request.
    Dirty,
}

impl Default for CanonicalCatalogTailCache {
    fn default() -> Self {
        Self::Known {
            record_count: 0,
            tail: None,
        }
    }
}

/// Aggregate physical FUA attribution surfaced to the engine probe.
///
/// A zero snapshot means that this `WalBuffer` is not using the FUA backend. The counters are
/// observability only; the durable/published frontiers remain the acknowledgement authority.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FuaDurabilityTelemetry {
    pub configured_fence_lanes: u64,
    pub logical_groups: u64,
    pub logical_payload_bytes: u64,
    pub single_frame_padded_baseline_bytes: u64,
    pub publish_turn_wait_nanos: u64,
    pub publish_turn_wait_groups: u64,
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
    pub waiter_cut_to_observe_nanos: u64,
    pub waiter_cut_to_observe_count: u64,
    pub in_flight_depth_max: u64,
    pub in_flight_depth_histogram: [u64; 7],
    /// Persistent physical controller evidence. All fields remain zero until the controller-owned
    /// canonical FUA group route publishes an action (and for a non-FUA backend).
    pub controller_sustained_actions: u64,
    pub controller_pending_probe_cover_actions: u64,
    pub controller_qd1_samples: u64,
    pub controller_qd1_sparse_actions: u64,
    pub controller_qd1_verify_actions: u64,
    pub controller_qd1_fast_actions: u64,
    pub controller_unfragmented_actions: u64,
    pub controller_pool_too_narrow: u64,
    pub controller_empty_chunk: u64,
    pub controller_insufficient_free_slots: u64,
    pub controller_natural_depth: u64,
    pub controller_segment_boundary: u64,
    pub controller_amplification_cap: u64,
    pub controller_fast_samples: u64,
    pub controller_nonfast_samples: u64,
    pub controller_transitions_to_verify: u64,
    pub controller_transitions_to_fast: u64,
    pub controller_transitions_to_sustained: u64,
    pub controller_stale_qd1_samples: u64,
    pub controller_unavailable_qd1_samples: u64,
    pub controller_abandoned_qd1_samples: u64,
    pub controller_protocol_faults: u64,
    pub controller_protocol_fallback_actions: u64,
    pub controller_phase: u64,
    pub controller_verify_fast_streak: u64,
    pub controller_sustained_remaining: u64,
    pub controller_generation: u64,
    pub controller_pending_qd1_samples: u64,
    pub controller_fast_in_flight: u64,
    pub controller_fast_in_flight_max: u64,
    pub controller_generation_exhausted: u64,
    pub controller_ordinal_exhausted: u64,
    pub controller_action_reconciliation: u64,
    pub controller_sample_reconciliation: u64,
}

#[derive(Debug)]
pub struct WalBuffer {
    records: Vec<WalRecord>,
    /// Sparse exact outer-byte authorities for only unflushed typed records. Recovered and
    /// already-durable history allocates no parallel sidecar; entries retire at the frontier.
    exact_typed_wire_records: Vec<typed_exact::ExactTypedWireRecord>,
    /// At most one typed record can be tentative under the commit owner.  This O(1) frontier is
    /// the group-flush gate; never scan the per-record wire sidecar on the 48M fixture path.
    typed_exact_tentative_count: usize,
    /// Compact non-truncatable boundary. Exact serialized Arcs retire at durability, but a
    /// claimed typed record remains part of immutable history after that retirement.
    highest_claimed_typed_exact_record: Option<usize>,
    /// Last exact owner range handed to a formed scatter group.  Exact sidecars leave the live
    /// tail at this monotonic cursor and can never be restored after handoff.
    typed_exact_handoff_cursor: usize,
    /// Two permanent scatter descriptors plus typed preproposal byte/record credits.
    prepared_group_arena: group::WalPreparedGroupArena,
    /// Changes whenever the logical record vector changes.  A typed canonical append reservation
    /// captures this generation with the exact record frontier so an old Vec slot cannot be
    /// replayed after truncation, recovery reinstatement, or an intervening append.
    canonical_append_reservation_generation: u64,
    /// Stable process-local identity, preventing an equal-length independent WAL buffer from
    /// consuming this buffer's spare-capacity reservation.
    canonical_append_reservation_owner_id: u64,
    /// Logical canonical catalog boundary, including records not yet flushed to durable media.
    /// It is independent of physical segment-prefix truncation because that operation preserves
    /// the complete logical record vector.
    canonical_catalog_tail: CanonicalCatalogTailCache,
    /// The live-buffer equivalent of the public full-slice identity bind. It remains private so
    /// only constructors that proved the sidecar may seed a recovered/fresh history.
    durable_identity_binding: DurableIdentityBinding,
    /// Durable watermark for the IN-MEMORY mode only (`durable: None`). The durable mode's
    /// watermark lives in [`WalDurableState::flushed_records`] so a group flush can advance it
    /// under the core's own lock, without the buffer's outer lock (the engine commit_mutex).
    flushed_memory: usize,
    fail_next_flush: bool,
    durable: Option<Arc<WalDurableCore>>,
    /// Optional FUA frame-log durability backend. It replaces `durable`; the fence pool may
    /// overlap physical writes while the buffer preserves one logical lifecycle and watermark.
    /// Gated behind `#[cfg(unix)]` because `FuaFrameLog` uses unix `O_DIRECT|O_DSYNC` operations.
    #[cfg(unix)]
    fua: Option<Arc<fua::FuaWalBackend>>,
    #[cfg(test)]
    canonical_catalog_tail_decodes_for_test: usize,
}

impl Default for WalBuffer {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            exact_typed_wire_records: Vec::new(),
            typed_exact_tentative_count: 0,
            highest_claimed_typed_exact_record: None,
            typed_exact_handoff_cursor: 0,
            prepared_group_arena: group::WalPreparedGroupArena::default(),
            canonical_append_reservation_generation: 0,
            canonical_append_reservation_owner_id: next_canonical_append_reservation_owner_id(),
            canonical_catalog_tail: CanonicalCatalogTailCache::default(),
            durable_identity_binding: DurableIdentityBinding::default(),
            flushed_memory: 0,
            fail_next_flush: false,
            durable: None,
            #[cfg(unix)]
            fua: None,
            #[cfg(test)]
            canonical_catalog_tail_decodes_for_test: 0,
        }
    }
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
    /// durable cut advances the one logical record watermark.
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
    fn invalidate_canonical_append_reservations(&mut self) {
        self.canonical_append_reservation_generation = self
            .canonical_append_reservation_generation
            .checked_add(1)
            .expect("WAL canonical append reservation generation exhausted");
    }

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

    /// Construct a fresh serial durable buffer after the caller has durably installed exactly
    /// `identity` beside `segment_path`. The constructor rechecks that proof rather than exposing
    /// a mutable identity setter; every flush checks it again before durable I/O.
    pub fn with_durable_segment_bound_to_identity(
        segment_path: impl Into<PathBuf>,
        identity: CanonicalIdentity,
    ) -> Result<Self, EngineError> {
        let segment_path = segment_path.into();
        require_durable_identity(&segment_path, identity)?;
        let mut buffer = Self::with_durable_segment(segment_path);
        buffer.durable_identity_binding = DurableIdentityBinding::bound(identity, 0);
        Ok(buffer)
    }

    /// A WAL buffer backed by the FUA fence-pool durability backend (E1 step 1) — a FRESH durable
    /// database. `flush_all` / `begin_group_flush` publish each group's encoded record run into
    /// the FUA frame log; the contiguous durable cut is the one logical record watermark. Segment files are created next to
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

    /// Construct a fresh FUA durable buffer after the caller has durably installed exactly
    /// `identity` beside `segment_path`. This is the checked constructor used by the engine's
    /// fresh durable path; it cannot be used to smuggle an unchecked mutable lineage into a WAL.
    #[cfg(unix)]
    pub fn with_fua_durable_segment_bound_to_identity(
        segment_path: impl Into<PathBuf>,
        lanes: usize,
        segment_bytes: usize,
        identity: CanonicalIdentity,
    ) -> Result<Self, EngineError> {
        let segment_path = segment_path.into();
        require_durable_identity(&segment_path, identity)?;
        let mut buffer = Self::with_fua_durable_segment(segment_path, lanes, segment_bytes)?;
        buffer.durable_identity_binding = DurableIdentityBinding::bound(identity, 0);
        Ok(buffer)
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
        Self::validate_recovered_segment_history(&records, recovery)?;
        let durable_identity_binding =
            DurableIdentityBinding::for_recovered_history(&segment_path, &records)?;
        Self::with_recovered_durable_segment_with_binding(
            segment_path,
            records,
            recovery,
            durable_identity_binding,
        )
    }

    fn validate_recovered_segment_history(
        records: &[WalRecord],
        recovery: &WalSegmentRecovery,
    ) -> Result<(), EngineError> {
        if records.len() < recovery.records.len() || !records.ends_with(&recovery.records) {
            return Err(EngineError::Durability(
                "recovered logical WAL history must end with the exact recovered segment suffix"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn with_recovered_durable_segment_with_binding(
        segment_path: PathBuf,
        records: Vec<WalRecord>,
        recovery: &WalSegmentRecovery,
        durable_identity_binding: DurableIdentityBinding,
    ) -> Result<Self, EngineError> {
        debug_assert!(records.len() >= recovery.records.len());
        debug_assert!(records.ends_with(&recovery.records));
        let record_count = records.len();
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
            exact_typed_wire_records: Vec::new(),
            typed_exact_tentative_count: 0,
            highest_claimed_typed_exact_record: record_count.checked_sub(1),
            typed_exact_handoff_cursor: record_count,
            prepared_group_arena: group::WalPreparedGroupArena::default(),
            records,
            canonical_append_reservation_generation: 0,
            canonical_append_reservation_owner_id: next_canonical_append_reservation_owner_id(),
            canonical_catalog_tail: CanonicalCatalogTailCache::Dirty,
            durable_identity_binding,
            flushed_memory: 0,
            fail_next_flush: false,
            durable: Some(Arc::new(core)),
            #[cfg(unix)]
            fua: None,
            #[cfg(test)]
            canonical_catalog_tail_decodes_for_test: 0,
        })
    }

    /// Reopen a serial durable buffer after recovery already proved the complete history and its
    /// checked sidecar anchor. New flushes decode only appended records, but still re-read the
    /// anchor before every durable handoff.
    pub fn with_recovered_durable_segment_bound_to_identity(
        segment_path: impl Into<PathBuf>,
        records: Vec<WalRecord>,
        recovery: &WalSegmentRecovery,
        identity: CanonicalIdentity,
    ) -> Result<Self, EngineError> {
        let segment_path = segment_path.into();
        Self::validate_recovered_segment_history(&records, recovery)?;
        require_durable_identity(&segment_path, identity)?;
        DurableIdentityBinding::validate_recovered_records(&records, identity)?;
        let verified_records = records.len();
        Self::with_recovered_durable_segment_with_binding(
            segment_path,
            records,
            recovery,
            DurableIdentityBinding::bound(identity, verified_records),
        )
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
        let segment_path = segment_path.into();
        let durable_identity_binding =
            DurableIdentityBinding::for_recovered_history(&segment_path, &records)?;
        Self::with_recovered_fua_durable_segment_with_binding(
            segment_path,
            records,
            lanes,
            segment_bytes,
            durable_identity_binding,
        )
    }

    #[cfg(unix)]
    fn with_recovered_fua_durable_segment_with_binding(
        segment_path: PathBuf,
        records: Vec<WalRecord>,
        lanes: usize,
        segment_bytes: usize,
        durable_identity_binding: DurableIdentityBinding,
    ) -> Result<Self, EngineError> {
        let recovered = records.len();
        let backend = fua::FuaWalBackend::reopen(segment_path, lanes, segment_bytes, recovered)?;
        Ok(Self {
            exact_typed_wire_records: Vec::new(),
            typed_exact_tentative_count: 0,
            highest_claimed_typed_exact_record: recovered.checked_sub(1),
            typed_exact_handoff_cursor: recovered,
            prepared_group_arena: group::WalPreparedGroupArena::default(),
            records,
            canonical_append_reservation_generation: 0,
            canonical_append_reservation_owner_id: next_canonical_append_reservation_owner_id(),
            canonical_catalog_tail: CanonicalCatalogTailCache::Dirty,
            durable_identity_binding,
            flushed_memory: 0,
            fail_next_flush: false,
            durable: None,
            fua: Some(Arc::new(backend)),
            #[cfg(test)]
            canonical_catalog_tail_decodes_for_test: 0,
        })
    }

    /// FUA counterpart of [`Self::with_recovered_durable_segment_bound_to_identity`].
    #[cfg(unix)]
    pub fn with_recovered_fua_durable_segment_bound_to_identity(
        segment_path: impl Into<PathBuf>,
        records: Vec<WalRecord>,
        lanes: usize,
        segment_bytes: usize,
        identity: CanonicalIdentity,
    ) -> Result<Self, EngineError> {
        let segment_path = segment_path.into();
        require_durable_identity(&segment_path, identity)?;
        DurableIdentityBinding::validate_recovered_records(&records, identity)?;
        let verified_records = records.len();
        Self::with_recovered_fua_durable_segment_with_binding(
            segment_path,
            records,
            lanes,
            segment_bytes,
            DurableIdentityBinding::bound(identity, verified_records),
        )
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

    /// Whether this buffer's durable backing is the FUA frame-log implementation.
    ///
    /// This is probe identity only. It does not grant callers a logical-concurrency exception:
    /// typed exact FUA ownership serializes one presealed logical group through its lifecycle.
    pub fn is_fua_durable(&self) -> bool {
        #[cfg(unix)]
        {
            self.fua.is_some()
        }
        #[cfg(not(unix))]
        {
            false
        }
    }

    /// Permanent physical FUA aggregates for the build-only engine probe. In-memory and serial
    /// WAL paths intentionally report all zeros.
    pub fn fua_durability_telemetry(&self) -> FuaDurabilityTelemetry {
        #[cfg(unix)]
        if let Some(fua) = self.fua.as_ref().cloned() {
            return fua.telemetry();
        }
        FuaDurabilityTelemetry::default()
    }

    pub fn append(&mut self, rec: WalRecord) {
        self.prune_durable_typed_exact_wire_records();
        self.records.push(rec);
        self.canonical_catalog_tail = CanonicalCatalogTailCache::Dirty;
        self.invalidate_canonical_append_reservations();
    }

    /// Append one sealed canonical envelope and advance the logical catalog boundary in the same
    /// mutation. The cache includes unflushed records because the next canonical proposal must
    /// chain from the logical append order, not only the durable prefix.
    pub fn append_canonical(&mut self, prepared: PreparedCanonicalWalRecord) {
        self.prune_durable_typed_exact_wire_records();
        assert!(
            prepared.exact_authority().is_none(),
            "commit-path invariant violation: exact typed canonical records must use the move-only tentative/claim lifecycle"
        );
        let (record, tail, exact) = prepared.into_parts();
        debug_assert!(exact.is_none());
        self.records.push(record);
        self.canonical_catalog_tail = CanonicalCatalogTailCache::Known {
            record_count: self.records.len(),
            tail: Some(tail),
        };
        self.invalidate_canonical_append_reservations();
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
        self.exact_typed_wire_records.clear();
        self.typed_exact_tentative_count = 0;
        self.highest_claimed_typed_exact_record = self.records.len().checked_sub(1);
        self.typed_exact_handoff_cursor = self.records.len();
        self.canonical_catalog_tail = CanonicalCatalogTailCache::Dirty;
        self.invalidate_canonical_append_reservations();
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Last logical record, including an unflushed member of the current group. Canonical WAL
    /// construction uses its catalog-after binding as the next record's catalog-before binding.
    pub fn last_record(&self) -> Option<&WalRecord> {
        self.records.last()
    }

    /// Return the last canonical record's catalog-after boundary, lazily decoding only when a
    /// generic logical-history mutation made the cache dirty. A legacy final record returns
    /// `None`; a canonical-magic payload must decode completely or the request fails closed.
    pub fn canonical_catalog_tail(&mut self) -> Result<Option<CanonicalCatalogTail>, EngineError> {
        if let CanonicalCatalogTailCache::Known { record_count, tail } = self.canonical_catalog_tail
        {
            if record_count == self.records.len() {
                return Ok(tail);
            }
        }

        if self.records.is_empty() {
            self.canonical_catalog_tail = CanonicalCatalogTailCache::Known {
                record_count: 0,
                tail: None,
            };
            return Ok(None);
        }

        self.canonical_catalog_tail = CanonicalCatalogTailCache::Dirty;
        #[cfg(test)]
        {
            self.canonical_catalog_tail_decodes_for_test = self
                .canonical_catalog_tail_decodes_for_test
                .saturating_add(1);
        }
        let tail = decode_canonical_record_payload(
            &self
                .records
                .last()
                .expect("non-empty WAL tail was checked above")
                .payload,
        )?
        .map(|envelope| CanonicalCatalogTail {
            identity: envelope.header.identity,
            catalog_after_epoch: envelope.header.catalog_after_epoch,
            catalog_after_digest: envelope.header.catalog_after_digest,
        });
        self.canonical_catalog_tail = CanonicalCatalogTailCache::Known {
            record_count: self.records.len(),
            tail,
        };
        Ok(tail)
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn truncate(&mut self, len: usize) {
        if len >= self.records.len() {
            return;
        }
        if self
            .highest_claimed_typed_exact_record
            .is_some_and(|claimed| len <= claimed)
        {
            panic!(
                "commit-path invariant violation: claimed typed exact WAL position is non-truncatable"
            );
        }
        if self.has_typed_exact_wire_at_or_after(len) {
            panic!(
                "commit-path invariant violation: typed exact WAL positions are non-truncatable; use their move-only rollback before claim"
            );
        }
        // A rollback can discard already-verified records, but not the immutable lineage they
        // established. Clamping makes any surviving/new tail decode again before it can flush.
        self.durable_identity_binding.verified_records =
            self.durable_identity_binding.verified_records.min(len);
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
            self.canonical_catalog_tail = if self.records.is_empty() {
                CanonicalCatalogTailCache::Known {
                    record_count: 0,
                    tail: None,
                }
            } else {
                CanonicalCatalogTailCache::Dirty
            };
            self.invalidate_canonical_append_reservations();
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
        self.canonical_catalog_tail = if self.records.is_empty() {
            CanonicalCatalogTailCache::Known {
                record_count: 0,
                tail: None,
            }
        } else {
            CanonicalCatalogTailCache::Dirty
        };
        self.invalidate_canonical_append_reservations();
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
        self.prune_durable_typed_exact_wire_records();
        self.require_no_unclaimed_typed_exact_tail()?;
        // FUA backend: the inline serial-path flush is just a group flush that also WAITS for the
        // durable cut. Delegating keeps one publish/wait path (and one `fail_next_flush`
        // consumption, handled by `begin_group_flush`).
        #[cfg(unix)]
        if self.fua.is_some() {
            let target = self.records.len();
            let fua = Arc::clone(self.fua.as_ref().expect("checked FUA backend"));
            let mut previous_durable = fua.durable_records();
            let mut previous_published = fua.published_records();
            loop {
                if let Some(fault) = fua.exact_fault() {
                    return Err(EngineError::DurabilityFault(fault));
                }
                match self.begin_group_flush()? {
                    WalGroupFlushBegin::Clean { .. } => {}
                    WalGroupFlushBegin::Job(job) => {
                        job.commit()?;
                    }
                    WalGroupFlushBegin::Busy => {
                        fua.wait_durable(fua.published_records())?;
                    }
                }
                let durable = fua.durable_records();
                if durable >= target {
                    self.prune_durable_typed_exact_wire_records();
                    return Ok(());
                }
                let published = fua.published_records();
                if published == target {
                    fua.wait_durable(target)?;
                    self.prune_durable_typed_exact_wire_records();
                    return Ok(());
                }
                if durable <= previous_durable && published <= previous_published {
                    return Err(EngineError::Durability(
                        "FUA WAL flush_all made no bounded prefix progress".to_string(),
                    ));
                }
                previous_durable = durable;
                previous_published = published;
            }
        }
        let target = self.records.len();
        loop {
            match self.begin_group_flush()? {
                WalGroupFlushBegin::Clean { flushed_records } if flushed_records >= target => {
                    self.prune_durable_typed_exact_wire_records();
                    return Ok(());
                }
                WalGroupFlushBegin::Clean { .. } => {}
                WalGroupFlushBegin::Job(job) => {
                    job.commit()?;
                }
                WalGroupFlushBegin::Busy => {
                    // The serial core owns the in-flight descriptor/job and signals completion
                    // without needing the buffer's outer lock.  Wait, then form the next prefix.
                    if let Some(core) = self.durable.as_ref() {
                        drop(core.lock_state_idle());
                    } else {
                        return Err(EngineError::Durability(
                            "in-memory WAL group descriptor remained busy".to_string(),
                        ));
                    }
                }
            }
            if self.flushed_count() >= target {
                self.prune_durable_typed_exact_wire_records();
                return Ok(());
            }
        }
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
    /// INLINE [`WalBuffer::flush_all`] path in the meantime. The FUA backend has no serial-core
    /// slot, but typed exact ownership still permits exactly one presealed logical group through
    /// handoff and settlement at a time.
    pub fn begin_group_flush(&mut self) -> Result<WalGroupFlushBegin, EngineError> {
        self.prune_durable_typed_exact_wire_records();
        self.require_no_unclaimed_typed_exact_tail()?;
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
        if let Some(fua) = self.fua.as_ref().cloned() {
            if let Some(fault) = fua.exact_fault() {
                return Err(EngineError::DurabilityFault(fault));
            }
            self.durable_identity_binding.verify_through(
                fua.base_path(),
                &self.records,
                self.records.len(),
            )?;
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
            if self
                .exact_wire_for_group(
                    published,
                    self.records
                        .get(published)
                        .expect("nonempty FUA group has a logical head"),
                )?
                .is_some()
            {
                let prefix = self
                    .select_prepared_group_prefix(published, None)?
                    .expect("nonempty FUA exact head retains a group prefix");
                let Some(group) = self.handoff_prepared_group(prefix)? else {
                    return Ok(WalGroupFlushBegin::Busy);
                };
                let handoff = match fua.handoff_exact_group(&group) {
                    Ok(handoff) => handoff,
                    Err(error) => {
                        if let EngineError::DurabilityFault(fault) = error {
                            group.poison(fault);
                            return Err(EngineError::DurabilityFault(fault));
                        }
                        return Err(error);
                    }
                };
                return Ok(WalGroupFlushBegin::Job(WalGroupFlushJob {
                    kind: Some(WalGroupFlushJobKind::Fua(fua::FuaFlushJob::new_exact(
                        Arc::clone(&fua),
                        handoff,
                        group,
                    ))),
                }));
            }
            fua.reject_legacy_while_exact_active()?;
            let seq_count = u32::try_from(group_size).map_err(|_| {
                EngineError::Durability(
                    "FUA WAL group exceeds u32 records; split the commit batch".to_string(),
                )
            })?;
            let mut payload = Vec::new();
            for index in published..target {
                self.encode_fua_legacy_wire_record_into(&mut payload, index)?;
            }
            // Allocate the ticket and advance the published cursor together under this outer lock
            // so tickets and `first_seq` ranges are assigned in one total order.
            let ticket = fua.next_ticket();
            fua.set_published(target);
            return Ok(WalGroupFlushBegin::Job(WalGroupFlushJob {
                kind: Some(WalGroupFlushJobKind::Fua(
                    fua::FuaFlushJob::new_legacy_compatibility(
                        Arc::clone(&fua),
                        ticket,
                        payload,
                        published as u64,
                        seq_count,
                        target,
                    ),
                )),
            }));
        }
        let Some(core) = self.durable.clone() else {
            let Some(prefix) = self.select_prepared_group_prefix(self.flushed_memory, None)? else {
                return Ok(WalGroupFlushBegin::Clean {
                    flushed_records: self.flushed_memory,
                });
            };
            let Some(group) = self.handoff_prepared_group(prefix)? else {
                return Ok(WalGroupFlushBegin::Busy);
            };
            self.flushed_memory = prefix.target_records;
            group.finish_success();
            return Ok(WalGroupFlushBegin::Clean {
                flushed_records: self.flushed_memory,
            });
        };
        let mut state = core.lock_state();
        if let Some(fault) = core.fixed_fault() {
            return Err(EngineError::DurabilityFault(fault));
        }
        if state.io_in_flight {
            return Ok(WalGroupFlushBegin::Busy);
        }
        if let Some(reason) = state.poisoned.clone() {
            return Err(core.poisoned_error(&reason));
        }
        let Some(natural_prefix) =
            self.select_prepared_group_prefix(state.flushed_records, None)?
        else {
            return Ok(WalGroupFlushBegin::Clean {
                flushed_records: state.flushed_records,
            });
        };
        let mut prefix = natural_prefix;
        if natural_prefix.exact_records != 0 {
            let free = state
                .prealloc_bytes
                .checked_sub(state.durable_bytes)
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or_else(|| {
                    EngineError::Durability(
                        "serial WAL preallocation frontier is behind its durable tail".to_string(),
                    )
                })?;
            prefix = self
                .select_prepared_group_prefix(state.flushed_records, Some(free))?
                .expect("nonempty serial WAL prefix remains nonempty after extent cap");
            if prefix.exact_records != 0
                && (state.file.is_none()
                    || state
                        .durable_bytes
                        .checked_add(prefix.wire_bytes as u64)
                        .is_none_or(|end| end > state.prealloc_bytes))
            {
                return Err(EngineError::Durability(
                    "claimed typed exact WAL group escaped its pre-proposal serial extent"
                        .to_string(),
                ));
            }
        }
        if self.exact_typed_wire_records.is_empty() {
            self.durable_identity_binding.verify_through(
                &core.segment_path,
                &self.records,
                prefix.target_records,
            )?;
        }
        if prefix.exact_records == 0 && self.exact_typed_wire_records.is_empty() {
            // Legacy-only compatibility groups retain the established create/preallocate route.
            core.ensure_created(&mut state)?;
            let write_end = state
                .durable_bytes
                .checked_add(prefix.wire_bytes as u64)
                .ok_or_else(|| {
                    EngineError::Durability("WAL group preallocation overflow".to_string())
                })?;
            core.ensure_preallocated_through(&mut state, write_end)?;
        } else if prefix.exact_records == 0 {
            // A legacy head may precede an admitted exact tail.  That reservation already
            // created and sized the serial extent; mutating it here would make this a post-claim
            // create/preallocation route.
            let write_end = state
                .durable_bytes
                .checked_add(prefix.wire_bytes as u64)
                .ok_or_else(|| {
                    EngineError::Durability("WAL group extent validation overflow".to_string())
                })?;
            if state.file.is_none() || write_end > state.prealloc_bytes {
                return Err(EngineError::Durability(
                    "legacy prefix before a claimed typed exact WAL owner escaped its pre-proposal serial extent"
                        .to_string(),
                ));
            }
        }
        let file = state.file.clone().expect("write handle present");
        let Some(group) = self.handoff_prepared_group(prefix)? else {
            return Ok(WalGroupFlushBegin::Busy);
        };
        state.io_in_flight = true;
        Ok(WalGroupFlushBegin::Job(WalGroupFlushJob {
            kind: Some(WalGroupFlushJobKind::Serial(group::SerialFlushJob::new(
                Arc::clone(&core),
                file,
                state.durable_bytes,
                prefix.target_records,
                prefix.group_size(),
                group,
            ))),
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
            if let Some(fault) = self.fua.as_ref().and_then(|backend| backend.exact_fault()) {
                return Err(EngineError::DurabilityFault(fault));
            }
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
        if let Some(fault) = core.fixed_fault() {
            return Err(EngineError::DurabilityFault(fault));
        }
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
        let bound_identity = self.durable_identity_binding.identity;
        if let Some(identity) = bound_identity {
            // A live rotation never establishes lineage: even an empty retained suffix must
            // retain its already-bound anchor before the rewrite can touch the segment.
            require_durable_identity(&core.segment_path, identity)?;
        }
        // Close the old handle first: the rename below unlinks the inode it points at.
        // W1b audit fix 3: any failure past this point leaves `file = None`, and the next
        // flush's ensure_created would CLOBBER the live segment with fresh-database semantics —
        // poison the backing instead so the half-rotated state is fail-closed until restart
        // recovery (which reads the on-disk files, not this handle).
        state.file = None;
        let retained = &self.records[base..flushed];
        let rewrite = match bound_identity {
            Some(identity) => crate::rewrite_wal_segment_requiring_durable_identity(
                &core.segment_path,
                retained,
                identity,
            ),
            None => write_wal_segment(&core.segment_path, retained),
        };
        if let Err(err) = rewrite {
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

    #[cfg(test)]
    pub(crate) fn durable_identity_decoded_records_for_test(&self) -> usize {
        self.durable_identity_binding.decoded_records_for_test
    }

    #[cfg(test)]
    pub(crate) fn durable_identity_verified_records_for_test(&self) -> usize {
        self.durable_identity_binding.verified_records
    }

    #[cfg(test)]
    pub(crate) fn canonical_catalog_tail_decodes_for_test(&self) -> usize {
        self.canonical_catalog_tail_decodes_for_test
    }

    #[cfg(all(test, unix))]
    pub(crate) fn fua_published_records_for_test(&self) -> Option<usize> {
        self.fua.as_ref().map(|fua| fua.published_records())
    }

    #[cfg(all(test, unix))]
    pub(crate) fn fail_next_fua_exact_commit_for_test(
        &self,
        stage: DurabilityStage,
        raw_os_error: Option<i32>,
    ) {
        self.fua
            .as_ref()
            .expect("FUA exact fault seam requires an FUA buffer")
            .fail_next_exact_commit_for_test(stage, raw_os_error);
    }
}

#[cfg(test)]
mod reservation_tests {
    use super::*;

    fn prepared_record(txn_id: TxnId) -> PreparedCanonicalWalRecord {
        let identity = CanonicalIdentity {
            database_id: [1; 16],
            cluster_id: [2; 16],
            timeline_id: [3; 16],
            format_epoch: 1,
        };
        let request_digest = [4; 32];
        let operation = CanonicalFragment {
            kind: CanonicalFragmentKind::RowMutation,
            body: vec![5],
        };
        let status = CanonicalFragment {
            kind: CanonicalFragmentKind::TransactionClaimStatus,
            body: vec![6],
        };
        let header = CanonicalPreApplyHeader {
            identity,
            leader_epoch: 1,
            commit_seq: 1,
            stable_transaction_id: txn_id,
            request_digest,
            isolation: CanonicalIsolation::ReadCommitted,
            flags: u32::from(CanonicalFragmentKind::RowMutation as u16),
            catalog_before_epoch: 0,
            catalog_after_epoch: 0,
            catalog_before_digest: [7; 32],
            catalog_after_digest: [7; 32],
            operation_count: 2,
            table_block_count: 0,
            allocator_high_water: 0,
        };
        let outcome = CanonicalOutcome {
            kind: CanonicalOutcomeKind::CommitSuccess,
            affected_rows: 1,
            sqlstate: None,
            constraint_id: 0,
            target_digest: [8; 32],
            returning_digest: [0; 32],
        };
        prepare_exact_canonical_wal_record(
            txn_id,
            CanonicalPhysicalRange {
                log_epoch: 1,
                lane_id: 0,
                segment_id: 1,
                first_frame_ordinal: 0,
            },
            header,
            &[operation, status],
            outcome,
        )
        .expect("prepare exact test canonical record")
    }

    #[test]
    fn typed_exact_tentative_tail_is_unflushable_and_rolls_back_without_reencoding() {
        let mut wal = WalBuffer::new();
        let prepared = prepared_record(1);
        let expected_payload = prepared.exact_packed_payload().unwrap().clone();
        let expected_serialized = prepared
            .exact_authority()
            .unwrap()
            .serialized_record()
            .clone();
        let mut reservation = wal
            .reserve_typed_exact_append(prepared)
            .expect("reserve exact owner before proposal");
        let proposal = reservation.replication_payload().expect("proposal payload");
        assert!(std::sync::Arc::ptr_eq(&proposal, &expected_payload));
        wal.append_typed_exact_tentative(&mut reservation)
            .expect("append tentative exact owner");
        assert_eq!(wal.len(), 1);
        let group_error = match wal.begin_group_flush() {
            Ok(_) => panic!("tentative tail must not group flush"),
            Err(error) => error,
        };
        assert!(group_error
            .to_string()
            .contains("cannot flush an unclaimed tentative typed exact WAL tail"));
        assert!(
            wal.flush_all().is_err(),
            "tentative tail must not inline flush"
        );
        assert_eq!(
            wal.typed_exact_wire_payload_ptr_for_test(0),
            Some(expected_payload.as_ptr())
        );
        assert_eq!(
            wal.typed_exact_serialized_bytes_for_test(0).as_deref(),
            Some(expected_serialized.as_ref())
        );
        wal.rollback_typed_exact_append(&mut reservation)
            .expect("rollback restores exact owner");
        assert_eq!(wal.len(), 0);
        assert!(wal.exact_typed_wire_records.is_empty());
        assert!(wal.canonical_catalog_tail().unwrap().is_none());
        let retried = reservation
            .replication_payload()
            .expect("same exact retry payload");
        assert!(std::sync::Arc::ptr_eq(&proposal, &retried));
    }

    #[test]
    fn typed_exact_reservation_rejects_equal_bytes_from_a_distinct_payload_owner() {
        let prepared = prepared_record(1);
        let (mut record, tail, exact) = prepared.into_parts();
        let original = Arc::clone(&record.payload);
        record.payload = Arc::from(record.payload.as_ref());
        assert_eq!(record.payload.as_ref(), original.as_ref());
        assert!(!Arc::ptr_eq(&record.payload, &original));

        let mut wal = WalBuffer::new();
        let error = match wal
            .reserve_typed_exact_append(PreparedCanonicalWalRecord::from_parts(record, tail, exact))
        {
            Ok(_) => panic!("equal bytes from a different owner must not inherit exact authority"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("canonical record binding diverged"));
    }

    #[test]
    fn typed_exact_reservation_rejects_state_owner_and_generic_append_bypass() {
        let mut wal = WalBuffer::new();
        let mut reservation = wal
            .reserve_typed_exact_append(prepared_record(1))
            .expect("reserve exact owner");
        wal.append(WalRecord {
            txn_id: 9,
            payload: std::sync::Arc::from(&b"legacy"[..]),
        });
        let error = wal
            .append_typed_exact_tentative(&mut reservation)
            .expect_err("intervening append must reject the token");
        assert!(error.to_string().contains("state drifted"));
        assert_eq!(wal.len(), 1);

        let mut first = WalBuffer::new();
        let mut reservation = first
            .reserve_typed_exact_append(prepared_record(1))
            .expect("reserve first owner");
        first
            .append_typed_exact_tentative(&mut reservation)
            .expect("tentative append on first owner");
        let mut second = WalBuffer::new();
        let error = second
            .rollback_typed_exact_append(&mut reservation)
            .expect_err("another WAL owner must reject this token");
        assert!(error.to_string().contains("state drifted"));
        assert_eq!(second.len(), 0);

        let bypass = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WalBuffer::new().append_canonical(prepared_record(1));
        }));
        assert!(
            bypass.is_err(),
            "generic append must reject exact typed authority"
        );
    }

    #[test]
    fn claimed_typed_record_remains_nontruncatable_after_wire_arc_retires() {
        let mut wal = WalBuffer::new();
        let mut reservation = wal
            .reserve_typed_exact_append(prepared_record(1))
            .expect("reserve exact owner");
        wal.append_typed_exact_tentative(&mut reservation)
            .expect("append tentative exact owner");
        wal.claim_typed_exact_append(reservation)
            .expect("claim exact owner");
        wal.flush_all().expect("in-memory durable claim");
        assert!(
            wal.exact_typed_wire_records.is_empty(),
            "durability retires sparse wire Arc"
        );
        let truncate = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| wal.truncate(0)));
        assert!(
            truncate.is_err(),
            "claimed history remains non-truncatable after pruning"
        );

        wal.append(WalRecord {
            txn_id: 2,
            payload: std::sync::Arc::from(&b"generic tail"[..]),
        });
        wal.truncate(1);
        assert_eq!(wal.len(), 1, "a newer generic tail remains rollbackable");
    }
}
