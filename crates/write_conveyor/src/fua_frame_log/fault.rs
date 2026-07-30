//! Fixed, allocation-free failure identity for the FUA frame-log fence path.
//!
//! The frame log is intentionally independent from the canonical WAL crate. Its fence lanes
//! still need a stable post-publication failure value: building a string while a write or join is
//! already failing would make the fail-stop path depend on allocator progress.

use std::fmt;
use std::sync::OnceLock;

/// No physical frame is associated with the fault (for example a pool-spawn failure).
pub const FUA_FRAME_FAULT_NO_FRAME: u64 = u64::MAX;

/// Exact stage of a fixed FUA frame-log failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FuaFrameFaultStage {
    FenceIo,
    FenceWriteZero,
    FenceWriteOverflow,
    FrameSlot,
    Frontier,
    FencePoison,
    Join,
    Spawn,
    ScatterPlan,
    ScatterSource,
    ScatterCapacity,
    ScatterSlots,
    ScatterSequence,
    ScatterReservationDrift,
}

impl fmt::Display for FuaFrameFaultStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::FenceIo => "fence-io",
            Self::FenceWriteZero => "fence-write-zero",
            Self::FenceWriteOverflow => "fence-write-overflow",
            Self::FrameSlot => "frame-slot",
            Self::Frontier => "frontier",
            Self::FencePoison => "fence-poison",
            Self::Join => "join",
            Self::Spawn => "spawn",
            Self::ScatterPlan => "scatter-plan",
            Self::ScatterSource => "scatter-source",
            Self::ScatterCapacity => "scatter-capacity",
            Self::ScatterSlots => "scatter-slots",
            Self::ScatterSequence => "scatter-sequence",
            Self::ScatterReservationDrift => "scatter-reservation-drift",
        })
    }
}

/// Fixed identity of a terminal frame-log failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FuaFrameFault {
    pub stage: FuaFrameFaultStage,
    /// Platform error value captured directly from the failed syscall, when one exists.
    pub raw_os_error: Option<i32>,
    pub segment_id: u64,
    pub frame_id: u64,
}

impl FuaFrameFault {
    pub const fn new(
        stage: FuaFrameFaultStage,
        raw_os_error: Option<i32>,
        segment_id: u64,
        frame_id: u64,
    ) -> Self {
        Self {
            stage,
            raw_os_error,
            segment_id,
            frame_id,
        }
    }
}

impl fmt::Display for FuaFrameFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "stage={} raw_os_error={:?} segment_id={} frame_id={}",
            self.stage, self.raw_os_error, self.segment_id, self.frame_id
        )
    }
}

/// First-wins poison for fixed post-publication frame-log work.
#[derive(Debug, Default)]
pub struct FuaFramePoison {
    first: OnceLock<FuaFrameFault>,
}

impl FuaFramePoison {
    pub const fn new() -> Self {
        Self {
            first: OnceLock::new(),
        }
    }

    /// Publish a fault unless another lane already published the terminal failure.
    pub fn install(&self, fault: FuaFrameFault) -> FuaFrameFault {
        let _ = self.first.set(fault);
        *self
            .first
            .get()
            .expect("FuaFramePoison must hold the installed or winning fault")
    }

    pub fn snapshot(&self) -> Option<FuaFrameFault> {
        self.first.get().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_alloc::assert_no_allocations;
    use std::sync::{Arc, Barrier};

    fn fault(stage: FuaFrameFaultStage) -> FuaFrameFault {
        FuaFrameFault::new(stage, Some(libc::EIO), 91, 17)
    }

    #[test]
    fn fixed_poison_is_allocation_free_and_first_wins_under_race() {
        let poison = FuaFramePoison::new();
        let first = fault(FuaFrameFaultStage::FenceIo);
        assert_eq!(
            assert_no_allocations(|| poison.install(first)),
            first,
            "the post-publication poison install has no dynamic diagnostic"
        );
        assert_eq!(
            poison.install(fault(FuaFrameFaultStage::Frontier)),
            first,
            "a later fault cannot replace the immutable first fault"
        );

        let poison = Arc::new(FuaFramePoison::new());
        let barrier = Arc::new(Barrier::new(3));
        let left_fault = fault(FuaFrameFaultStage::FenceWriteZero);
        let right_fault = fault(FuaFrameFaultStage::Join);
        let left_poison = Arc::clone(&poison);
        let left_barrier = Arc::clone(&barrier);
        let left = std::thread::spawn(move || {
            left_barrier.wait();
            left_poison.install(left_fault)
        });
        let right_poison = Arc::clone(&poison);
        let right_barrier = Arc::clone(&barrier);
        let right = std::thread::spawn(move || {
            right_barrier.wait();
            right_poison.install(right_fault)
        });
        barrier.wait();
        let winner = poison.snapshot().expect("one racing fault publishes");
        assert!(winner == left_fault || winner == right_fault);
        assert_eq!(left.join().expect("left join"), winner);
        assert_eq!(right.join().expect("right join"), winner);
    }
}
