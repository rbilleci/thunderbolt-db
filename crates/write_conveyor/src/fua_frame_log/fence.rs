//! Fixed-returning FUA fence execution and bounded fence-pool ownership.

use super::{
    saturating_add, FuaFrameFault, FuaFrameFaultStage, FuaFrameLog, FuaFrameLogFencePool,
    FRAME_HEADER_BYTES, FRAME_LOG_HEADER_BYTES, FRAME_SLOTS,
};
use std::os::fd::AsRawFd;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::JoinHandle;

/// Maximum lane count accepted by the fixed-returning fence pool.
pub const FUA_FIXED_FENCE_POOL_MAX_LANES: usize = 32;

/// Fixed-capacity owner of FUA fence threads.
pub struct FuaFrameLogFixedFencePool {
    log: Arc<FuaFrameLog>,
    handles: [Option<JoinHandle<Result<u64, FuaFrameFault>>>; FUA_FIXED_FENCE_POOL_MAX_LANES],
}

impl FuaFrameLog {
    /// Spawn a bounded fence pool that returns only fixed frame-log faults.
    ///
    /// A failed partial spawn poisons the frame log before returning and drains every already
    /// launched lane. The caller can therefore never accidentally publish into a partially
    /// provisioned fixed pool.
    pub fn spawn_fixed_fence_pool(
        self: &Arc<Self>,
        lanes: usize,
    ) -> Result<FuaFrameLogFixedFencePool, FuaFrameFault> {
        if !(1..=FUA_FIXED_FENCE_POOL_MAX_LANES).contains(&lanes) {
            return Err(FuaFrameFault::new(
                FuaFrameFaultStage::Spawn,
                None,
                self.segment_id,
                super::FUA_FRAME_FAULT_NO_FRAME,
            ));
        }
        let mut handles = std::array::from_fn(|_| None);
        for index in 0..lanes {
            let log = Arc::clone(self);
            match std::thread::Builder::new().spawn(move || log.fence_lane_loop_fixed()) {
                Ok(handle) => handles[index] = Some(handle),
                Err(error) => {
                    let fault = self.install_fixed_fault(FuaFrameFault::new(
                        FuaFrameFaultStage::Spawn,
                        error.raw_os_error(),
                        self.segment_id,
                        super::FUA_FRAME_FAULT_NO_FRAME,
                    ));
                    for handle in &mut handles[..index] {
                        if let Some(handle) = handle.take() {
                            let _ = handle.join();
                        }
                    }
                    return Err(fault);
                }
            }
        }
        Ok(FuaFrameLogFixedFencePool {
            log: Arc::clone(self),
            handles,
        })
    }

    /// Spawn the historical dynamically-returning pool as a compatibility control.
    ///
    /// The threads execute the same fixed fence state machine. Formatting an I/O compatibility
    /// diagnostic happens only after that protected state machine returned its fixed fault.
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
        self.fence_lane_loop_fixed().map_err(fault_to_io_error)
    }

    fn fence_lane_loop_fixed(&self) -> Result<u64, FuaFrameFault> {
        // Spin briefly before parking: at high rates the next frame lands within the window and
        // the mutex is never touched.
        const SPINS_BEFORE_PARK: u32 = 2_000;
        let mut fenced = 0_u64;
        loop {
            if let Some(fault) = self.fixed_fault() {
                return Err(fault);
            }
            let (frame, claim_ns) = loop {
                if let Some(fault) = self.fixed_fault() {
                    return Err(fault);
                }
                let claimed = self.fence_cursor.load_acquire();
                let published = self.published_frames.load_acquire();
                if published > claimed {
                    if self.fence_cursor.compare_exchange(claimed, claimed + 1) {
                        let claim_ns = self.stat_now_nanos();
                        self.claim_ns[(claimed as usize) % FRAME_SLOTS]
                            .store(claim_ns, Ordering::Relaxed);
                        let published_ns = self.publish_ns[(claimed as usize) % FRAME_SLOTS]
                            .load(Ordering::Relaxed);
                        saturating_add(
                            &self.stat_publish_to_claim_ns,
                            claim_ns.saturating_sub(published_ns),
                        );
                        saturating_add(&self.stat_publish_to_claim_frames, 1);
                        break (claimed, claim_ns);
                    }
                    continue;
                }
                if self.publishing_finished.load(Ordering::Acquire) {
                    return Ok(fenced);
                }
                let mut spins = 0_u32;
                let should_park = loop {
                    if self.fixed_fault().is_some()
                        || self.published_frames.load_acquire() > self.fence_cursor.load_acquire()
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
                    let mut parked = self
                        .park
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if self.published_frames.load_acquire() <= self.fence_cursor.load_acquire()
                        && !self.publishing_finished.load(Ordering::Acquire)
                        && self.fixed_fault().is_none()
                    {
                        *parked += 1;
                        parked = self
                            .park_wake
                            .wait(parked)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        *parked = parked.saturating_sub(1);
                    }
                }
            };
            if let Err(error) = self.fence_frame_fixed(frame, claim_ns) {
                saturating_add(&self.stat_fence_failures, 1);
                return Err(self.install_fixed_fault(error));
            }
            fenced += 1;
        }
    }

    /// Wake fence lanes after state they wait on changed. A batch exposes several independent
    /// fenceable frames at one release point, so wake up to that many parked lanes; all is for
    /// finish/failure draining.
    pub(super) fn wake_fence_lanes(&self, frames: usize, all: bool) {
        let parked = self
            .park
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if *parked > 0 {
            if all {
                self.park_wake.notify_all();
            } else {
                for _ in 0..frames.min(*parked) {
                    self.park_wake.notify_one();
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn fence_frame(&self, frame_id: u64, claim_ns: u64) -> std::io::Result<()> {
        self.fence_frame_fixed(frame_id, claim_ns)
            .map_err(fault_to_io_error)
    }

    // BEGIN FUA_FIXED_FENCE_NO_ALLOC
    fn fence_frame_fixed(&self, frame_id: u64, claim_ns: u64) -> Result<(), FuaFrameFault> {
        #[cfg(test)]
        if let Some(fault) = self.take_fixed_fence_fault_for_test() {
            return Err(fault);
        }
        let fault = |stage, raw_os_error| {
            FuaFrameFault::new(stage, raw_os_error, self.segment_id, frame_id)
        };
        let slot = &self.slots[(frame_id as usize) % FRAME_SLOTS];
        if slot.frame_id.load(Ordering::Acquire) != frame_id {
            return Err(fault(FuaFrameFaultStage::FrameSlot, None));
        }
        let offset = slot.offset.load(Ordering::Acquire) as usize;
        let padded_len = slot.padded_len.load(Ordering::Acquire) as usize;
        if padded_len < FRAME_HEADER_BYTES
            || offset
                .checked_add(padded_len)
                .is_none_or(|end| end > self.capacity)
        {
            return Err(fault(FuaFrameFaultStage::FrameSlot, None));
        }
        let mut written = 0_usize;
        while written < padded_len {
            let file_offset = match FRAME_LOG_HEADER_BYTES
                .checked_add(offset)
                .and_then(|base| base.checked_add(written))
                .and_then(|value| libc::off_t::try_from(value).ok())
            {
                Some(value) => value,
                None => return Err(fault(FuaFrameFaultStage::FenceWriteOverflow, None)),
            };
            let remaining = padded_len - written;
            let result = unsafe {
                libc::pwrite(
                    self.file.as_raw_fd(),
                    self.staging.ptr().add(offset + written).cast(),
                    remaining,
                    file_offset,
                )
            };
            if result < 0 {
                let raw_os_error = std::io::Error::last_os_error().raw_os_error();
                if raw_os_error == Some(libc::EINTR) {
                    continue;
                }
                return Err(fault(FuaFrameFaultStage::FenceIo, raw_os_error));
            }
            if result == 0 {
                return Err(fault(FuaFrameFaultStage::FenceWriteZero, None));
            }
            let advanced = result as usize;
            if advanced > remaining {
                return Err(fault(FuaFrameFaultStage::FenceWriteOverflow, None));
            }
            written += advanced;
        }
        let write_done_ns = self.stat_now_nanos();
        self.write_done_ns[(frame_id as usize) % FRAME_SLOTS]
            .store(write_done_ns, Ordering::Relaxed);
        saturating_add(
            &self.stat_claim_to_write_done_ns,
            write_done_ns.saturating_sub(claim_ns),
        );
        saturating_add(&self.stat_claim_to_write_done_frames, 1);
        slot.fenced.store(true, Ordering::Release);
        let published = self.publish_ns[(frame_id as usize) % FRAME_SLOTS].load(Ordering::Relaxed);
        saturating_add(&self.stat_fence_ns, write_done_ns.saturating_sub(published));
        saturating_add(&self.stat_fenced_frames, 1);
        self.fences_completed.fetch_add(1);
        let mut frontier = match self.durable_frontier.lock() {
            Ok(frontier) => frontier,
            Err(_) => return Err(fault(FuaFrameFaultStage::Frontier, None)),
        };
        let mut advanced = *frontier;
        while advanced < self.published_frames.load_acquire() {
            let slot = &self.slots[(advanced as usize) % FRAME_SLOTS];
            if !slot.fenced.load(Ordering::Acquire) {
                break;
            }
            advanced += 1;
        }
        if advanced != *frontier {
            let previous = *frontier;
            let cut_ns = self.stat_now_nanos();
            for cut_frame in previous..advanced {
                let write_done_ns =
                    self.write_done_ns[(cut_frame as usize) % FRAME_SLOTS].load(Ordering::Relaxed);
                saturating_add(
                    &self.stat_write_done_to_cut_ns,
                    cut_ns.saturating_sub(write_done_ns),
                );
            }
            let advanced_frames = advanced.saturating_sub(previous);
            saturating_add(&self.stat_write_done_to_cut_frames, advanced_frames);
            saturating_add(&self.stat_cut_events, 1);
            saturating_add(&self.stat_cut_advanced_frames, advanced_frames);
            self.stat_cut_advance_max_frames
                .fetch_max(advanced_frames, Ordering::Relaxed);
            *frontier = advanced;
            self.durable_cut_ns
                .store(cut_ns.saturating_add(1), Ordering::Relaxed);
            self.durable_frames.store_release(advanced);
            let end_seq = self.slots[((advanced - 1) as usize) % FRAME_SLOTS]
                .end_seq
                .load(Ordering::Relaxed);
            let prior_end = self.durable_seq.load_relaxed();
            self.durable_seq.store_release(prior_end.max(end_seq));
        }
        Ok(())
    }
    // END FUA_FIXED_FENCE_NO_ALLOC

    fn install_fixed_fault(&self, fault: FuaFrameFault) -> FuaFrameFault {
        let fault = self.fixed_poison.install(fault);
        self.fence_failed.store(true, Ordering::Release);
        self.publishing_finished.store(true, Ordering::Release);
        self.wake_fence_lanes(0, true);
        fault
    }

    /// First fixed fault from a fence lane or a fixed-pool lifecycle operation.
    pub fn fixed_fault(&self) -> Option<FuaFrameFault> {
        self.fixed_poison.snapshot()
    }

    #[cfg(test)]
    pub(super) fn fail_next_fixed_fence_for_test(&self, fault: FuaFrameFault) {
        let mut pending = self
            .fixed_fence_test_fault
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            pending.replace(fault).is_none(),
            "only one test fence fault is pending"
        );
    }

    #[cfg(test)]
    fn take_fixed_fence_fault_for_test(&self) -> Option<FuaFrameFault> {
        self.fixed_fence_test_fault
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}

impl FuaFrameLogFixedFencePool {
    /// Join every launched lane before returning the first stable fault, if any.
    pub fn join_fixed(mut self) -> Result<u64, FuaFrameFault> {
        let mut fences = 0_u64;
        let mut first = self.log.fixed_fault();
        for handle in &mut self.handles {
            let Some(handle) = handle.take() else {
                continue;
            };
            match handle.join() {
                Ok(Ok(count)) => fences = fences.saturating_add(count),
                Ok(Err(fault)) => {
                    first.get_or_insert_with(|| self.log.install_fixed_fault(fault));
                }
                Err(_) => {
                    first.get_or_insert_with(|| {
                        self.log.install_fixed_fault(FuaFrameFault::new(
                            FuaFrameFaultStage::Join,
                            None,
                            self.log.segment_id,
                            super::FUA_FRAME_FAULT_NO_FRAME,
                        ))
                    });
                }
            }
        }
        match first {
            Some(fault) => Err(fault),
            None => Ok(fences),
        }
    }
}

fn fault_to_io_error(fault: FuaFrameFault) -> std::io::Error {
    std::io::Error::other(fault.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_alloc::assert_no_allocations;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_PATH: AtomicU64 = AtomicU64::new(1);

    fn test_path(label: &str) -> PathBuf {
        let directory = std::env::temp_dir().join("fua-frame-log-fence-tests");
        std::fs::create_dir_all(&directory).expect("create test directory");
        directory.join(format!(
            "{label}-{}-{}.dat",
            std::process::id(),
            NEXT_TEST_PATH.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn config(path: &std::path::Path) -> super::super::FuaFrameLogConfig {
        super::super::FuaFrameLogConfig {
            path: path.to_path_buf(),
            segment_id: 57,
            capacity_bytes: 1 << 20,
        }
    }

    fn publish_one(log: &Arc<FuaFrameLog>) -> super::super::FuaFrameLogAppender {
        let mut appender = log.appender();
        appender
            .publish_frame(&[0x55; 512], 0, 1)
            .expect("publish frame");
        appender
    }

    #[test]
    fn fixed_fence_io_fault_is_stable_and_protected_path_allocates_nothing() {
        let path = test_path("io-fault");
        let log = unsafe { FuaFrameLog::create(config(&path)).expect("create frame log") };
        let appender = publish_one(&log);
        let expected = FuaFrameFault::new(FuaFrameFaultStage::FenceIo, Some(libc::EIO), 57, 0);
        log.fail_next_fixed_fence_for_test(expected);
        assert_eq!(
            assert_no_allocations(|| {
                log.fence_frame_fixed(0, log.stat_now_nanos())
                    .expect_err("injected fixed I/O fault")
            }),
            expected
        );
        drop(appender);
        let _ = std::fs::remove_file(path);

        let path = test_path("io-fault-pool");
        let log = unsafe { FuaFrameLog::create(config(&path)).expect("create frame log") };
        let appender = publish_one(&log);
        log.fail_next_fixed_fence_for_test(expected);
        let pool = log.spawn_fixed_fence_pool(1).expect("fixed pool");
        appender.finish();
        assert_eq!(pool.join_fixed().expect_err("fixed I/O failure"), expected);
        assert_eq!(log.fixed_fault(), Some(expected));
        assert!(log.fence_failed());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn fixed_frontier_fault_is_stable_and_first_wins() {
        let path = test_path("frontier-fault");
        let log = unsafe { FuaFrameLog::create(config(&path)).expect("create frame log") };
        let appender = publish_one(&log);
        let poison_log = Arc::clone(&log);
        let poisoner = std::thread::spawn(move || {
            let _frontier = poison_log.durable_frontier.lock().expect("lock frontier");
            panic!("poison fixed frontier");
        });
        assert!(poisoner.join().is_err(), "frontier poisoning must run");
        let expected = FuaFrameFault::new(FuaFrameFaultStage::Frontier, None, 57, 0);
        let pool = log.spawn_fixed_fence_pool(1).expect("fixed pool");
        appender.finish();
        assert_eq!(pool.join_fixed().expect_err("frontier failure"), expected);
        assert_eq!(log.fixed_fault(), Some(expected));
        assert_eq!(
            log.install_fixed_fault(FuaFrameFault::new(
                FuaFrameFaultStage::FenceWriteZero,
                None,
                57,
                0,
            )),
            expected,
            "first fixed frame fault remains immutable"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn fixed_join_and_spawn_faults_are_exact() {
        let path = test_path("join-fault");
        let log = unsafe { FuaFrameLog::create(config(&path)).expect("create frame log") };
        let mut handles = std::array::from_fn(|_| None);
        handles[0] = Some(std::thread::spawn(|| -> Result<u64, FuaFrameFault> {
            panic!("deterministic fixed join fault")
        }));
        let pool = FuaFrameLogFixedFencePool {
            log: Arc::clone(&log),
            handles,
        };
        let expected = FuaFrameFault::new(
            FuaFrameFaultStage::Join,
            None,
            57,
            super::super::FUA_FRAME_FAULT_NO_FRAME,
        );
        assert_eq!(pool.join_fixed().expect_err("join fault"), expected);
        assert_eq!(log.fixed_fault(), Some(expected));
        let zero_lanes = match log.spawn_fixed_fence_pool(0) {
            Ok(_) => panic!("zero fixed lanes are rejected"),
            Err(fault) => fault,
        };
        assert_eq!(zero_lanes.stage, FuaFrameFaultStage::Spawn);
        let over_capacity = match log.spawn_fixed_fence_pool(FUA_FIXED_FENCE_POOL_MAX_LANES + 1) {
            Ok(_) => panic!("over-capacity fixed pool is rejected"),
            Err(fault) => fault,
        };
        assert_eq!(over_capacity.stage, FuaFrameFaultStage::Spawn);
        let _ = std::fs::remove_file(path);

        let path = test_path("maximum-fixed-pool");
        let log = unsafe { FuaFrameLog::create(config(&path)).expect("create frame log") };
        let pool = log
            .spawn_fixed_fence_pool(FUA_FIXED_FENCE_POOL_MAX_LANES)
            .expect("the established 32-lane WAL default is accepted");
        log.appender().finish();
        assert_eq!(pool.join_fixed().expect("join maximum fixed pool"), 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn fixed_fence_leaf_forbids_dynamic_allocation_escape_hatches() {
        let source = include_str!("fence.rs");
        let (_, marked) = source
            .split_once("// BEGIN FUA_FIXED_FENCE_NO_ALLOC")
            .expect("fence begin marker");
        let (protected, _) = marked
            .split_once("// END FUA_FIXED_FENCE_NO_ALLOC")
            .expect("fence end marker");
        for forbidden in [
            "format!",
            ".to_string()",
            "String",
            "Vec",
            "io::Error::new",
            "io::Error::other",
            "try_reserve",
        ] {
            assert!(
                !protected.contains(forbidden),
                "fixed fence path must not contain {forbidden}"
            );
        }
    }
}
