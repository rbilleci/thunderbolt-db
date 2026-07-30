//! Final-name, fully-owned successor preparation for FUA WAL segment rolls.
//!
//! A [`PreparedSuccessor`] is deliberately move-only: when the active segment exhausts its
//! pre-admitted extent, the roll consumes this owner and swaps its already-open appender and fixed
//! fence pool into the active slot.  The background pre-stager builds under a unique ignored name,
//! then installs and directory-syncs the final numeric recovery name before Ready.  A take must
//! never need to rename a file, synchronise a directory, allocate a staging arena, or spawn fence
//! lanes at that boundary.

use super::{segment_file_path, sync_parent_dir};
use gpu_db_types::EngineError;
use gpu_db_write_conveyor::{
    FuaFrameLog, FuaFrameLogAppender, FuaFrameLogConfig, FuaFrameLogFixedFencePool,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

/// A side-effect-free successful preflight for a final-named successor.  Future FUA group-credit
/// admission can retain this fact before it claims a typed WAL envelope; it intentionally carries
/// no handle that could be used to publish ahead of the owning roll.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Retained for the next typed FUA group-credit admission slice.
pub(crate) struct FuaSuccessorPreflight {
    pub(crate) segment_id: u64,
    pub(crate) capacity_bytes: usize,
    pub(crate) free_fence_slots: usize,
    pub(crate) recycled: bool,
}

/// Observable, side-effect-free successor state.  `Ready` is current only when its predecessor
/// matches the active segment observed by the caller; a former ready owner cannot be reused after
/// a later roll.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Retained for the next typed FUA group-credit admission slice.
pub(crate) enum FuaSuccessorReadiness {
    Empty,
    Pending {
        segment_id: u64,
        predecessor_id: u64,
    },
    Ready {
        segment_id: u64,
        predecessor_id: u64,
        current: bool,
        recycled: bool,
    },
    Failed,
}

/// The complete successor ownership record.  The final recovery-visible filename, frame-log
/// staging allocation, one appender, and fixed fence pool are all present before this value can
/// enter [`PrestageSlot::Ready`].  A mid-prestage crash leaves only the ignored temp name, never a
/// partially initialized numeric segment that recovery would scan.
pub(super) struct PreparedSuccessor {
    segment_id: u64,
    predecessor_id: u64,
    final_path: PathBuf,
    log: Option<Arc<FuaFrameLog>>,
    appender: Option<FuaFrameLogAppender>,
    pool: Option<FuaFrameLogFixedFencePool>,
    recycled: bool,
}

impl PreparedSuccessor {
    pub(super) fn was_recycled(&self) -> bool {
        self.recycled
    }

    pub(super) fn into_parts(
        mut self,
    ) -> (
        u64,
        Arc<FuaFrameLog>,
        FuaFrameLogAppender,
        FuaFrameLogFixedFencePool,
    ) {
        (
            self.segment_id,
            self.log
                .take()
                .expect("ready successor must own its frame log"),
            self.appender
                .take()
                .expect("ready successor must own its appender"),
            self.pool
                .take()
                .expect("ready successor must own its fixed fence pool"),
        )
    }

    /// Release an unused ready successor without leaking its parked fixed fence lanes.  The
    /// appender must finish before the pool can drain and join.  [`Drop`] owns that ordering.
    pub(super) fn drain(self) {}

    fn is_current_for(&self, active_segment_id: u64) -> bool {
        self.predecessor_id == active_segment_id
            && self.segment_id > active_segment_id
            && self.final_path.is_file()
            && self
                .log
                .as_ref()
                .is_some_and(|log| log.published_frames() == 0 && !log.fence_failed())
    }
}

impl Drop for PreparedSuccessor {
    fn drop(&mut self) {
        if let Some(appender) = self.appender.take() {
            appender.finish();
        }
        if let Some(pool) = self.pool.take() {
            let _ = pool.join_fixed();
        }
    }
}

// Keep the prepared owner inline: boxing `Ready` would add a fallible allocation to successor
// publication and weaken the fixed rollover ownership proof.
#[allow(clippy::large_enum_variant)]
enum PrestageSlot {
    Empty,
    Pending {
        segment_id: u64,
        predecessor_id: u64,
    },
    Ready(PreparedSuccessor),
    Failed(String),
}

/// One background pre-stager and its single successor slot.  The state is intentionally separate
/// from the active segment mutex: preparing a next extent may take a filesystem prewrite, but it
/// never blocks the appender or current fence lanes.
pub(super) struct PrestageState {
    slot: Mutex<PrestageSlot>,
    wake: Condvar,
}

impl PrestageState {
    pub(super) fn new() -> Self {
        Self {
            slot: Mutex::new(PrestageSlot::Empty),
            wake: Condvar::new(),
        }
    }

    /// Start exactly one background preparation.  This is only called when the slot is empty;
    /// violating that law is a lifecycle bug, not an excuse to replace an extant prepared owner.
    pub(super) fn kick(
        self: &Arc<Self>,
        base_path: &Path,
        segment_id: u64,
        predecessor_id: u64,
        segment_bytes: usize,
        lanes: usize,
        recycle_pool: Arc<Mutex<Vec<PathBuf>>>,
    ) {
        {
            let mut slot = self
                .slot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert!(
                matches!(&*slot, PrestageSlot::Empty),
                "FUA pre-stager may have only one pending or ready successor"
            );
            *slot = PrestageSlot::Pending {
                segment_id,
                predecessor_id,
            };
        }

        let state = Arc::clone(self);
        let base_path = base_path.to_path_buf();
        let spawn = std::thread::Builder::new()
            .name("gpu-db-fua-prestage".to_string())
            .spawn(move || {
                // A panic must not strand a rolling thread behind Pending.  We retain no join
                // handle because slot completion is the complete filesystem ownership handoff.
                let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    prepare_successor(
                        &base_path,
                        segment_id,
                        predecessor_id,
                        segment_bytes,
                        lanes,
                        recycle_pool,
                    )
                }));
                state.complete(segment_id, predecessor_id, prepared);
            });
        if let Err(error) = spawn {
            self.complete_failure(
                segment_id,
                predecessor_id,
                format!("failed to spawn FUA successor pre-stager: {error}"),
            );
        }
    }

    /// Read-only readiness/currentness observation for the active segment.  It neither creates a
    /// successor nor waits for one; admission uses this as a pre-claim proof rather than relying
    /// on a post-claim fallback.
    #[allow(dead_code)] // Exposed through FuaWalBackend for future typed group-credit admission.
    pub(super) fn readiness(&self, active_segment_id: u64) -> FuaSuccessorReadiness {
        let slot = self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &*slot {
            PrestageSlot::Empty => FuaSuccessorReadiness::Empty,
            PrestageSlot::Pending {
                segment_id,
                predecessor_id,
            } => FuaSuccessorReadiness::Pending {
                segment_id: *segment_id,
                predecessor_id: *predecessor_id,
            },
            PrestageSlot::Ready(successor) => FuaSuccessorReadiness::Ready {
                segment_id: successor.segment_id,
                predecessor_id: successor.predecessor_id,
                current: successor.is_current_for(active_segment_id),
                recycled: successor.recycled,
            },
            PrestageSlot::Failed(_) => FuaSuccessorReadiness::Failed,
        }
    }

    /// Read-only capacity and fence-slot proof for a possible roll.  `None` is deliberately a
    /// refusal, not a kick/wait/retry: a typed caller must arrange successor preparation before
    /// it owns an exact WAL claim.
    #[allow(dead_code)] // Exposed through FuaWalBackend for future typed group-credit admission.
    pub(super) fn preflight(
        &self,
        active_segment_id: u64,
        required_padded_bytes: usize,
        required_fence_slots: usize,
        lanes: usize,
    ) -> Option<FuaSuccessorPreflight> {
        let slot = self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let PrestageSlot::Ready(successor) = &*slot else {
            return None;
        };
        let log = successor.log.as_ref()?;
        let capacity_bytes = successor.log_capacity_bytes();
        let free_fence_slots = log.free_fence_slots(lanes);
        (successor.is_current_for(active_segment_id)
            && required_padded_bytes <= capacity_bytes
            && required_fence_slots <= free_fence_slots)
            .then_some(FuaSuccessorPreflight {
                segment_id: successor.segment_id,
                capacity_bytes,
                free_fence_slots,
                recycled: successor.recycled,
            })
    }

    /// Wait only for the already-started one successor, then move it out.  There is no inline
    /// creation, rename, directory sync, appender creation, fence-pool spawn, or retry here.
    pub(super) fn take_current(
        &self,
        active_segment_id: u64,
    ) -> Result<PreparedSuccessor, EngineError> {
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            match std::mem::replace(&mut *slot, PrestageSlot::Empty) {
                PrestageSlot::Ready(successor) => {
                    if successor.is_current_for(active_segment_id) {
                        return Ok(successor);
                    }
                    let segment_id = successor.segment_id;
                    let predecessor_id = successor.predecessor_id;
                    *slot = PrestageSlot::Ready(successor);
                    return Err(EngineError::Durability(format!(
                        "FUA prepared successor {segment_id} is not current for active segment {active_segment_id} (prepared for {predecessor_id})"
                    )));
                }
                PrestageSlot::Failed(reason) => {
                    return Err(EngineError::Durability(format!(
                        "FUA segment pre-create failed: {reason}"
                    )));
                }
                PrestageSlot::Pending {
                    segment_id,
                    predecessor_id,
                } => {
                    *slot = PrestageSlot::Pending {
                        segment_id,
                        predecessor_id,
                    };
                    slot = self
                        .wake
                        .wait(slot)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
                PrestageSlot::Empty => {
                    return Err(EngineError::Durability(
                        "FUA segment roll has no prebuilt final-named successor".to_string(),
                    ));
                }
            }
        }
    }

    /// Drain the final ready owner during backend shutdown.  A pending pre-stager is first
    /// awaited so its final-path operation cannot race a rapid same-path reopen.
    pub(super) fn drain_ready_on_drop(&self) {
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while matches!(&*slot, PrestageSlot::Pending { .. }) {
            slot = self
                .wake
                .wait(slot)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        if let PrestageSlot::Ready(successor) = std::mem::replace(&mut *slot, PrestageSlot::Empty) {
            drop(slot);
            successor.drain();
        }
    }

    fn complete(
        &self,
        segment_id: u64,
        predecessor_id: u64,
        prepared: Result<Result<PreparedSuccessor, std::io::Error>, Box<dyn std::any::Any + Send>>,
    ) {
        let next = match prepared {
            Ok(Ok(successor)) => PrestageSlot::Ready(successor),
            Ok(Err(error)) => PrestageSlot::Failed(format!("{error}")),
            Err(_) => PrestageSlot::Failed("FUA successor pre-stager panicked".to_string()),
        };
        let mut slot = self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        debug_assert!(matches!(
            &*slot,
            PrestageSlot::Pending {
                segment_id: pending_id,
                predecessor_id: pending_predecessor,
            } if *pending_id == segment_id && *pending_predecessor == predecessor_id
        ));
        *slot = next;
        self.wake.notify_all();
    }

    fn complete_failure(&self, segment_id: u64, predecessor_id: u64, reason: String) {
        self.complete(
            segment_id,
            predecessor_id,
            Ok(Err(std::io::Error::other(reason))),
        );
    }
}

impl PreparedSuccessor {
    #[allow(dead_code)] // Used by the retained preflight API above.
    fn log_capacity_bytes(&self) -> usize {
        self.appender
            .as_ref()
            .expect("ready successor must retain its appender")
            .remaining_capacity_bytes()
    }
}

fn prepare_successor(
    base_path: &Path,
    segment_id: u64,
    predecessor_id: u64,
    segment_bytes: usize,
    lanes: usize,
    recycle_pool: Arc<Mutex<Vec<PathBuf>>>,
) -> std::io::Result<PreparedSuccessor> {
    let final_path = segment_file_path(base_path, segment_id);
    let temp_path = prestage_temp_path(base_path, segment_id);
    let (log, recycled) = prepare_temp_log(&temp_path, segment_id, segment_bytes, recycle_pool)?;
    let appender = log.appender();
    let pool = match log.spawn_fixed_fence_pool(lanes) {
        Ok(pool) => pool,
        Err(fault) => {
            appender.finish();
            drop(log);
            cleanup_failed_path(&temp_path)?;
            return Err(std::io::Error::other(fault.to_string()));
        }
    };
    if let Err(error) =
        std::fs::rename(&temp_path, &final_path).and_then(|()| sync_parent_dir(&final_path))
    {
        appender.finish();
        let _ = pool.join_fixed();
        drop(log);
        let cleanup_path = if final_path.exists() {
            &final_path
        } else {
            &temp_path
        };
        let _ = cleanup_failed_path(cleanup_path);
        return Err(error);
    }
    Ok(PreparedSuccessor {
        segment_id,
        predecessor_id,
        final_path,
        log: Some(log),
        appender: Some(appender),
        pool: Some(pool),
        recycled,
    })
}

fn prepare_temp_log(
    temp_path: &Path,
    segment_id: u64,
    segment_bytes: usize,
    recycle_pool: Arc<Mutex<Vec<PathBuf>>>,
) -> std::io::Result<(Arc<FuaFrameLog>, bool)> {
    let config = FuaFrameLogConfig {
        path: temp_path.to_path_buf(),
        segment_id,
        capacity_bytes: segment_bytes,
    };
    // A crash before Ready can leave this ignored temp path.  It has no recovery authority and
    // is owned by the next preparation for this exact monotonic segment id.
    cleanup_failed_path(temp_path)?;
    let retired = recycle_pool
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .pop();
    if let Some(retired) = retired {
        match std::fs::rename(&retired, temp_path)
            .and_then(|()| sync_parent_dir(temp_path))
            .and_then(|()| unsafe { FuaFrameLog::recycle(config.clone()) })
        {
            Ok(log) => return Ok((log, true)),
            Err(_) => {
                cleanup_failed_path(temp_path)?;
            }
        }
    }

    // The final recovery-visible name is installed only after the log has its fully prepared
    // appender and fixed pool.  Until then, recovery ignores this unique nonnumeric suffix.
    match unsafe { FuaFrameLog::create(config) } {
        Ok(log) => Ok((log, false)),
        Err(error) => {
            let _ = cleanup_failed_path(temp_path);
            Err(error)
        }
    }
}

/// `<base>.fua.prestage.<segment_id>` is unique per prepared successor but is intentionally
/// ignored by `parse_segment_id`; only a post-setup rename grants a numeric recovery authority.
fn prestage_temp_path(base: &Path, segment_id: u64) -> PathBuf {
    let name = base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("wal.segment");
    base.with_file_name(format!("{name}.fua.prestage.{segment_id}"))
}

fn cleanup_failed_path(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => sync_parent_dir(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fua::{recover_fua_wal_runs, FuaWalBackend};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PATH: AtomicU64 = AtomicU64::new(1);

    fn test_base(name: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "gpu-db-fua-prestage-{name}-{}-{}",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).expect("create prestage test directory");
        directory.join("wal.segment")
    }

    fn wait_ready(backend: &FuaWalBackend, active_segment_id: u64) -> FuaSuccessorReadiness {
        for _ in 0..100_000 {
            let readiness = backend.prestaged_readiness();
            if matches!(
                readiness,
                FuaSuccessorReadiness::Ready { current: true, .. }
            ) {
                return readiness;
            }
            assert!(
                !matches!(readiness, FuaSuccessorReadiness::Failed),
                "pre-stager failed: {readiness:?}"
            );
            std::thread::yield_now();
        }
        panic!("pre-stager did not become ready for active {active_segment_id}");
    }

    #[test]
    fn final_named_ready_owner_is_scan_visible_empty_and_fully_owned_before_take() {
        let base = test_base("final-ready");
        let backend = FuaWalBackend::create(base.clone(), 2, 64 * 1024).expect("backend");
        let readiness = wait_ready(&backend, 1);
        let FuaSuccessorReadiness::Ready {
            segment_id,
            predecessor_id,
            current,
            recycled,
        } = readiness
        else {
            unreachable!("wait_ready returned Ready")
        };
        assert_eq!(predecessor_id, 1);
        assert!(current);
        assert!(!recycled);
        let final_path = segment_file_path(&base, segment_id);
        assert!(final_path.is_file(), "ready successor has its final name");
        assert!(
            recover_fua_wal_runs(&final_path)
                .expect("scan final successor")
                .is_empty(),
            "a ready successor is recovery-visible but contributes no records"
        );
        assert!(
            !std::fs::read_dir(base.parent().expect("test parent"))
                .expect("list test parent")
                .flatten()
                .any(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .is_some_and(|name| name.starts_with("wal.segment.fua.prestage."))
                }),
            "a Ready successor leaves no stale ignored temp path"
        );
        assert_eq!(
            backend
                .preflight_prestaged_successor(64 * 1024, 2)
                .expect("side-effect-free successor preflight")
                .segment_id,
            segment_id,
            "preflight proves the same current ready owner before take"
        );
        let successor = backend.take_prestaged(1).expect("take ready successor");
        assert_eq!(successor.segment_id, segment_id);
        assert!(
            successor
                .appender
                .as_ref()
                .expect("ready owner appender")
                .remaining_capacity_bytes()
                >= 64 * 1024,
            "the moved owner retains its already-created appender"
        );
        successor.drain();
        drop(backend);
        let _ = std::fs::remove_dir_all(base.parent().expect("test parent"));
    }

    #[test]
    fn recycled_final_named_successor_keeps_epoch_safe_owner_and_no_temp_dependency() {
        let base = test_base("recycled");
        let backend = FuaWalBackend::create(base.clone(), 2, 64 * 1024).expect("backend");
        wait_ready(&backend, 1);
        backend.take_prestaged(1).expect("take initial").drain();

        let retired_path = segment_file_path(&base, 91);
        let retired = super::super::open_segment(&base, 91, 64 * 1024).expect("retired segment");
        drop(retired);
        backend
            .recycle_pool
            .lock()
            .expect("recycle pool")
            .push(retired_path.clone());
        backend.kick_prestage(1);
        let readiness = wait_ready(&backend, 1);
        let FuaSuccessorReadiness::Ready {
            segment_id,
            recycled,
            ..
        } = readiness
        else {
            unreachable!("wait_ready returned Ready")
        };
        assert!(
            recycled,
            "retired extent becomes a prebuilt recycled successor"
        );
        assert!(
            !retired_path.exists(),
            "retired name is not left as a second authority"
        );
        let final_path = segment_file_path(&base, segment_id);
        assert!(final_path.is_file());
        assert!(recover_fua_wal_runs(&final_path).unwrap().is_empty());
        backend.take_prestaged(1).expect("take recycled").drain();
        drop(backend);
        let _ = std::fs::remove_dir_all(base.parent().expect("test parent"));
    }

    #[test]
    fn panic_and_failure_completion_are_fail_closed_and_never_leave_pending() {
        let state = PrestageState::new();
        *state.slot.lock().expect("slot") = PrestageSlot::Pending {
            segment_id: 8,
            predecessor_id: 7,
        };
        let panic_result = std::panic::catch_unwind(|| panic!("test prestager panic"));
        state.complete(8, 7, panic_result.map(|_| unreachable!()));
        assert_eq!(state.readiness(7), FuaSuccessorReadiness::Failed);
        assert!(
            state.take_current(7).is_err(),
            "panic becomes a terminal refusal"
        );

        let state = PrestageState::new();
        *state.slot.lock().expect("slot") = PrestageSlot::Pending {
            segment_id: 8,
            predecessor_id: 7,
        };
        state.complete_failure(8, 7, "injected prepare failure".to_string());
        assert_eq!(state.readiness(7), FuaSuccessorReadiness::Failed);
        assert!(
            state.take_current(7).is_err(),
            "failure becomes a terminal refusal"
        );
    }

    #[test]
    fn take_and_roll_boundaries_have_no_namespace_or_pool_setup_operations() {
        let source = include_str!("../fua.rs");
        let take_start = source
            .find("fn take_prestaged")
            .expect("take_prestaged source marker");
        let roll_start = source[take_start..]
            .find("/// Roll the full active segment")
            .map(|offset| take_start + offset)
            .expect("roll source marker");
        let roll_end = source[roll_start..]
            .find("/// Make a group durable")
            .map(|offset| roll_start + offset)
            .expect("commit source marker");
        let take = &source[take_start..roll_start];
        let roll = &source[roll_start..roll_end];
        for forbidden in [
            "rename",
            "sync_parent_dir",
            "open_segment",
            ".appender(",
            "spawn_fence_pool",
            "spawn_fixed_fence_pool",
            "kick_prestage",
            "Vec::",
        ] {
            assert!(
                !take.contains(forbidden) && !roll.contains(forbidden),
                "successor take/roll must not perform forbidden post-claim operation {forbidden}"
            );
        }
    }
}
