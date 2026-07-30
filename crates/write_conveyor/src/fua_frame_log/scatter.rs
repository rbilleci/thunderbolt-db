//! Fixed-shape source scattering into the frame-log staging arena.
//!
//! This is the production ingress for already-encoded logical bytes. It deliberately owns no
//! WAL interpretation: a source is a contiguous byte stream, and physical fragments may cut
//! through a logical record. All payload bytes are staged before frame headers, slots, and the
//! sole release publication become visible to fence lanes.

use super::{
    header_crc, padded_frame_bytes, saturating_add, FrameBatchHandle, FrameGroupMetadata,
    FrameHeader, FuaFrameFault, FuaFrameFaultStage, FuaFrameLogAppender, FRAME_HEADER_BYTES,
    FRAME_MAGIC, FRAME_SLOTS,
};
use crate::wal_segment::bytes_of;
use std::sync::atomic::Ordering;

/// Maximum physical fragments in one source-scattered logical group.
pub const FUA_SCATTER_MAX_FRAGMENTS: usize = 16;

/// Immutable production byte source for a source-scattered FUA group.
///
/// Implementors write directly into the supplied staging slice. A failure carries no partially
/// published state: the appender has not changed its cursors or release frontier before this
/// method returns.
pub trait FuaScatterSource {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn copy_into(
        &mut self,
        source_offset: usize,
        destination: &mut [u8],
    ) -> Result<(), FuaFrameFault>;
}

impl FuaScatterSource for [u8] {
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }

    fn copy_into(
        &mut self,
        source_offset: usize,
        destination: &mut [u8],
    ) -> Result<(), FuaFrameFault> {
        let end = source_offset
            .checked_add(destination.len())
            .ok_or_else(|| {
                FuaFrameFault::new(
                    FuaFrameFaultStage::ScatterSource,
                    None,
                    0,
                    super::FUA_FRAME_FAULT_NO_FRAME,
                )
            })?;
        let source = self.get(source_offset..end).ok_or_else(|| {
            FuaFrameFault::new(
                FuaFrameFaultStage::ScatterSource,
                None,
                0,
                super::FUA_FRAME_FAULT_NO_FRAME,
            )
        })?;
        destination.copy_from_slice(source);
        Ok(())
    }
}

impl FuaScatterSource for Vec<u8> {
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn copy_into(
        &mut self,
        source_offset: usize,
        destination: &mut [u8],
    ) -> Result<(), FuaFrameFault> {
        <[u8] as FuaScatterSource>::copy_into(self.as_mut_slice(), source_offset, destination)
    }
}

/// Fixed physical split of one contiguous logical byte stream.
///
/// The first remainder fragments receive one byte each, making the split deterministic. The
/// stored padded total is exactly the sum of align_up(64 + chunk_i, 4096) for active fragments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FuaScatterPlan {
    chunks: [u32; FUA_SCATTER_MAX_FRAGMENTS],
    fragment_count: u8,
    payload_bytes: usize,
    padded_bytes: usize,
}

impl FuaScatterPlan {
    /// Build a deterministic 1..=16-fragment plan. Every fragment is nonempty.
    pub fn new(payload_bytes: usize, fragment_count: usize) -> Option<Self> {
        if payload_bytes == 0
            || !(1..=FUA_SCATTER_MAX_FRAGMENTS).contains(&fragment_count)
            || payload_bytes < fragment_count
        {
            return None;
        }
        let base = payload_bytes / fragment_count;
        let remainder = payload_bytes % fragment_count;
        if base > u32::MAX as usize {
            return None;
        }
        let mut chunks = [0_u32; FUA_SCATTER_MAX_FRAGMENTS];
        let mut padded_bytes = 0_usize;
        for (index, chunk) in chunks.iter_mut().take(fragment_count).enumerate() {
            let bytes = base.checked_add(usize::from(index < remainder))?;
            let bytes = u32::try_from(bytes).ok()?;
            *chunk = bytes;
            padded_bytes = padded_bytes.checked_add(padded_frame_bytes(bytes as usize))?;
        }
        Some(Self {
            chunks,
            fragment_count: fragment_count as u8,
            payload_bytes,
            padded_bytes,
        })
    }

    pub const fn fragment_count(self) -> usize {
        self.fragment_count as usize
    }

    pub const fn payload_bytes(self) -> usize {
        self.payload_bytes
    }

    pub const fn padded_bytes(self) -> usize {
        self.padded_bytes
    }

    pub const fn fragment_payload_bytes(self, index: usize) -> Option<usize> {
        if index < self.fragment_count as usize {
            Some(self.chunks[index] as usize)
        } else {
            None
        }
    }
}

/// Move-only proof that one exact scatter geometry was admitted at one exact appender cursor.
///
/// The reservation is intentionally opaque and non-Copy. It never owns payload bytes, does not
/// publish visibility, and cannot be retargeted to another segment, cursor, or fragment shape.
#[must_use = "a reserved scatter must be published once or dropped before WAL admission"]
#[derive(Debug)]
pub struct FuaScatterReservation {
    segment_id: u64,
    first_frame: u64,
    first_offset: usize,
    plan: FuaScatterPlan,
    sealed_chunks: [u32; FUA_SCATTER_MAX_FRAGMENTS],
    sealed_fragment_count: u8,
    sealed_payload_bytes: usize,
    sealed_padded_bytes: usize,
}

impl FuaScatterReservation {
    fn new(segment_id: u64, first_frame: u64, first_offset: usize, plan: FuaScatterPlan) -> Self {
        Self {
            segment_id,
            first_frame,
            first_offset,
            sealed_chunks: plan.chunks,
            sealed_fragment_count: plan.fragment_count,
            sealed_payload_bytes: plan.payload_bytes,
            sealed_padded_bytes: plan.padded_bytes,
            plan,
        }
    }

    fn geometry_is_sealed(&self) -> bool {
        self.plan.chunks == self.sealed_chunks
            && self.plan.fragment_count == self.sealed_fragment_count
            && self.plan.payload_bytes == self.sealed_payload_bytes
            && self.plan.padded_bytes == self.sealed_padded_bytes
    }
}

impl FuaFrameLogAppender {
    /// Remaining physical bytes in this segment before a complete group must roll.
    pub fn remaining_capacity_bytes(&self) -> usize {
        self.log.capacity.saturating_sub(self.next_offset)
    }

    /// Verify that the entire fixed scatter plan can become visible as one physical group.
    pub fn can_publish_scatter(&self, plan: FuaScatterPlan) -> Result<(), FuaFrameFault> {
        self.preflight_scatter(plan).map(|_| ())
    }

    /// Seal the current segment, cursor, and fixed physical geometry without staging or
    /// publishing any frame.
    pub fn reserve_scatter(
        &self,
        plan: FuaScatterPlan,
    ) -> Result<FuaScatterReservation, FuaFrameFault> {
        self.preflight_scatter(plan)?;
        Ok(FuaScatterReservation::new(
            self.log.segment_id,
            self.next_frame,
            self.next_offset,
            plan,
        ))
    }

    /// Stage a fixed source scatter and make every fragment visible with one release store.
    ///
    /// A one-fragment plan is intentionally encoded as the legacy metadata-zero frame. Two or
    /// more fragments use the existing grouped wire metadata, so scan recovery validates exact
    /// concatenation without any format extension.
    pub fn publish_scatter<S: FuaScatterSource + ?Sized>(
        &mut self,
        source: &mut S,
        plan: FuaScatterPlan,
        first_seq: u64,
        seq_count: u32,
    ) -> Result<FrameBatchHandle, FuaFrameFault> {
        let last_seq = self.validate_scatter_input(source, plan, first_seq, seq_count)?;
        let reservation = self.reserve_scatter(plan)?;
        self.verify_scatter_reservation(&reservation)?;
        self.stage_sealed_scatter(reservation, source, first_seq, seq_count, last_seq)
    }

    /// Consume one exact cursor seal and publish only its preflighted physical geometry.
    pub fn publish_reserved_scatter<S: FuaScatterSource + ?Sized>(
        &mut self,
        reservation: FuaScatterReservation,
        source: &mut S,
        first_seq: u64,
        seq_count: u32,
    ) -> Result<FrameBatchHandle, FuaFrameFault> {
        self.verify_scatter_reservation(&reservation)?;
        let last_seq =
            self.validate_scatter_input(source, reservation.plan, first_seq, seq_count)?;
        self.stage_sealed_scatter(reservation, source, first_seq, seq_count, last_seq)
    }

    fn validate_scatter_input<S: FuaScatterSource + ?Sized>(
        &self,
        source: &S,
        plan: FuaScatterPlan,
        first_seq: u64,
        seq_count: u32,
    ) -> Result<u64, FuaFrameFault> {
        if seq_count == 0 {
            return Err(self.scatter_fault(FuaFrameFaultStage::ScatterSequence));
        }
        if source.len() != plan.payload_bytes() {
            return Err(self.scatter_fault(FuaFrameFaultStage::ScatterSource));
        }
        first_seq
            .checked_add(u64::from(seq_count))
            .ok_or_else(|| self.scatter_fault(FuaFrameFaultStage::ScatterSequence))
    }

    fn verify_scatter_reservation(
        &self,
        reservation: &FuaScatterReservation,
    ) -> Result<(), FuaFrameFault> {
        if self.log.fence_failed() {
            return Err(self
                .log
                .fixed_fault()
                .unwrap_or_else(|| self.scatter_fault(FuaFrameFaultStage::FencePoison)));
        }
        if reservation.segment_id != self.log.segment_id
            || reservation.first_frame != self.next_frame
            || reservation.first_offset != self.next_offset
            || !reservation.geometry_is_sealed()
        {
            return Err(FuaFrameFault::new(
                FuaFrameFaultStage::ScatterReservationDrift,
                None,
                self.log.segment_id,
                reservation.first_frame,
            ));
        }
        Ok(())
    }

    // BEGIN FUA_SCATTER_NO_ALLOC
    fn stage_sealed_scatter<S: FuaScatterSource + ?Sized>(
        &mut self,
        seal: FuaScatterReservation,
        source: &mut S,
        first_seq: u64,
        seq_count: u32,
        last_seq: u64,
    ) -> Result<FrameBatchHandle, FuaFrameFault> {
        let plan = seal.plan;
        let total_payload = seal.sealed_payload_bytes;
        let total_padded = seal.sealed_padded_bytes;
        let log = &*self.log;
        let stage_copy_started = log.stat_now_nanos();
        let first_frame_id = seal.first_frame;
        let first_offset = seal.first_offset;
        let mut payload_crc = [0_u32; FUA_SCATTER_MAX_FRAGMENTS];
        let mut group_crc = 0_u32;
        let mut source_offset = 0_usize;
        let mut offset = first_offset;

        // Source copies are the only fallible staging work. No header, slot, cursor, statistic,
        // or release frontier is updated until each fragment succeeds.
        for (index, payload_crc_slot) in payload_crc
            .iter_mut()
            .enumerate()
            .take(plan.fragment_count())
        {
            let chunk_bytes = plan.chunks[index] as usize;
            let padded = padded_frame_bytes(chunk_bytes);
            let destination = unsafe {
                std::slice::from_raw_parts_mut(
                    log.staging.ptr().add(offset + FRAME_HEADER_BYTES),
                    chunk_bytes,
                )
            };
            if source.copy_into(source_offset, destination).is_err() {
                return Err(FuaFrameFault::new(
                    FuaFrameFaultStage::ScatterSource,
                    None,
                    log.segment_id,
                    first_frame_id,
                ));
            }
            *payload_crc_slot = crc32c::crc32c(destination);
            group_crc = crc32c::crc32c_append(group_crc, destination);
            let used = FRAME_HEADER_BYTES + chunk_bytes;
            if padded > used {
                unsafe {
                    std::ptr::write_bytes(log.staging.ptr().add(offset + used), 0, padded - used);
                }
            }
            source_offset += chunk_bytes;
            offset += padded;
        }
        debug_assert_eq!(source_offset, total_payload);
        debug_assert_eq!(offset - first_offset, total_padded);

        let group = (plan.fragment_count() > 1).then_some(FrameGroupMetadata {
            total_payload_bytes: total_payload as u64,
            fragment_index: 0,
            fragment_count: plan.fragment_count() as u32,
            group_crc32c: group_crc,
        });
        offset = first_offset;
        for (index, payload_crc) in payload_crc
            .iter()
            .copied()
            .enumerate()
            .take(plan.fragment_count())
        {
            let frame_id = first_frame_id + index as u64;
            let chunk_bytes = plan.chunks[index] as usize;
            let padded = padded_frame_bytes(chunk_bytes);
            let terminal = index + 1 == plan.fragment_count();
            let mut header = FrameHeader {
                magic: FRAME_MAGIC,
                frame_id,
                first_seq,
                reserved0: group.map_or(0, |metadata| metadata.total_payload_bytes),
                epoch: log.epoch,
                payload_bytes: chunk_bytes as u32,
                seq_count: u32::from(terminal).saturating_mul(seq_count),
                payload_crc,
                header_crc: 0,
                reserved1: group.map_or(0, |_| index as u32),
                reserved2: group.map_or(0, |metadata| metadata.fragment_count),
                reserved3: group.map_or(0, |metadata| metadata.group_crc32c),
            };
            header.header_crc = header_crc(&header);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes_of(&header).as_ptr(),
                    log.staging.ptr().add(offset),
                    FRAME_HEADER_BYTES,
                );
            }
            let slot = &log.slots[(frame_id as usize) % FRAME_SLOTS];
            let publish_ns = log.stat_now_nanos();
            log.publish_ns[(frame_id as usize) % FRAME_SLOTS].store(publish_ns, Ordering::Relaxed);
            slot.fenced.store(false, Ordering::Relaxed);
            slot.frame_id.store(frame_id, Ordering::Release);
            slot.offset.store(offset as u64, Ordering::Relaxed);
            slot.padded_len.store(padded as u64, Ordering::Relaxed);
            slot.end_seq.store(
                if terminal { last_seq } else { first_seq },
                Ordering::Relaxed,
            );
            offset += padded;
        }
        self.next_offset = offset;
        self.next_frame = first_frame_id + plan.fragment_count() as u64;
        let completed = log.fences_completed.load_acquire();
        for frame_id in first_frame_id..self.next_frame {
            log.record_in_flight_depth(frame_id.saturating_add(1).saturating_sub(completed));
        }
        saturating_add(&log.stat_payload_bytes, total_payload as u64);
        saturating_add(&log.stat_padded_bytes, total_padded as u64);
        saturating_add(
            &log.stat_stage_copy_ns,
            log.stat_now_nanos().saturating_sub(stage_copy_started),
        );
        saturating_add(&log.stat_stage_copy_frames, plan.fragment_count() as u64);
        log.published_frames.store_release(self.next_frame);
        log.wake_fence_lanes(plan.fragment_count(), false);
        Ok(FrameBatchHandle {
            first_frame_id,
            terminal_frame_id: self.next_frame - 1,
            last_seq,
        })
    }
    // END FUA_SCATTER_NO_ALLOC

    fn preflight_scatter(&self, plan: FuaScatterPlan) -> Result<(usize, usize), FuaFrameFault> {
        if self.log.fence_failed() {
            return Err(self
                .log
                .fixed_fault()
                .unwrap_or_else(|| self.scatter_fault(FuaFrameFaultStage::FencePoison)));
        }
        if plan.fragment_count() == 0
            || plan.fragment_count() > FUA_SCATTER_MAX_FRAGMENTS
            || plan.payload_bytes() == 0
        {
            return Err(self.scatter_fault(FuaFrameFaultStage::ScatterPlan));
        }
        if self
            .next_offset
            .checked_add(plan.padded_bytes())
            .is_none_or(|end| end > self.log.capacity)
        {
            return Err(self.scatter_fault(FuaFrameFaultStage::ScatterCapacity));
        }
        let terminal = self
            .next_frame
            .checked_add(plan.fragment_count() as u64)
            .ok_or_else(|| self.scatter_fault(FuaFrameFaultStage::ScatterPlan))?;
        if terminal > self.log.durable_frames.load_acquire() + FRAME_SLOTS as u64 {
            return Err(self.scatter_fault(FuaFrameFaultStage::ScatterSlots));
        }
        Ok((plan.payload_bytes(), plan.padded_bytes()))
    }

    fn scatter_fault(&self, stage: FuaFrameFaultStage) -> FuaFrameFault {
        FuaFrameFault::new(stage, None, self.log.segment_id, self.next_frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_alloc::assert_no_allocations;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_PATH: AtomicU64 = AtomicU64::new(1);

    fn test_path(label: &str) -> PathBuf {
        let directory = std::env::temp_dir().join("fua-frame-log-scatter-tests");
        std::fs::create_dir_all(&directory).expect("create test directory");
        directory.join(format!(
            "{label}-{}-{}.dat",
            std::process::id(),
            NEXT_TEST_PATH.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn config(path: &std::path::Path, capacity_bytes: usize) -> super::super::FuaFrameLogConfig {
        super::super::FuaFrameLogConfig {
            path: path.to_path_buf(),
            segment_id: 41,
            capacity_bytes,
        }
    }

    struct CrossSpanSource {
        bytes: Vec<u8>,
        record_ends: [usize; 5],
        crossed_record_boundaries: usize,
    }

    impl FuaScatterSource for CrossSpanSource {
        fn len(&self) -> usize {
            self.bytes.len()
        }

        fn copy_into(
            &mut self,
            source_offset: usize,
            destination: &mut [u8],
        ) -> Result<(), FuaFrameFault> {
            let end = source_offset + destination.len();
            destination.copy_from_slice(&self.bytes[source_offset..end]);
            self.crossed_record_boundaries += self
                .record_ends
                .iter()
                .filter(|boundary| source_offset < **boundary && **boundary < end)
                .count();
            Ok(())
        }
    }

    struct FailAfterOneFragment {
        bytes: Vec<u8>,
        copies: usize,
    }

    impl FuaScatterSource for FailAfterOneFragment {
        fn len(&self) -> usize {
            self.bytes.len()
        }

        fn copy_into(
            &mut self,
            source_offset: usize,
            destination: &mut [u8],
        ) -> Result<(), FuaFrameFault> {
            if self.copies != 0 {
                return Err(FuaFrameFault::new(
                    FuaFrameFaultStage::ScatterSource,
                    None,
                    41,
                    super::super::FUA_FRAME_FAULT_NO_FRAME,
                ));
            }
            self.copies += 1;
            let end = source_offset + destination.len();
            destination.copy_from_slice(&self.bytes[source_offset..end]);
            Ok(())
        }
    }

    #[test]
    fn one_fragment_scatter_is_legacy_metadata_zero_with_exact_bytes() {
        let path = test_path("one-fragment");
        let logical = (0..1_025).map(|value| value as u8).collect::<Vec<_>>();
        let plan = FuaScatterPlan::new(logical.len(), 1).expect("one fragment plan");
        assert_eq!(plan.fragment_payload_bytes(0), Some(logical.len()));
        assert_eq!(
            plan.padded_bytes(),
            super::super::fua_frame_padded_bytes(logical.len())
        );
        let log = unsafe {
            super::super::FuaFrameLog::create(config(&path, 1 << 20)).expect("create frame log")
        };
        let pool = log.spawn_fixed_fence_pool(1).expect("fixed pool");
        let mut appender = log.appender();
        assert_eq!(appender.remaining_capacity_bytes(), 1 << 20);
        appender
            .can_publish_scatter(plan)
            .expect("one fragment preflight");
        let mut source = logical.clone();
        let batch = appender
            .publish_scatter(&mut source, plan, 3, 2)
            .expect("scatter publish");
        assert_eq!(batch.first_frame_id, 0);
        assert_eq!(batch.terminal_frame_id, 0);
        assert_eq!(
            appender.remaining_capacity_bytes(),
            (1 << 20) - plan.padded_bytes()
        );
        appender.finish();
        assert_eq!(pool.join_fixed().expect("fixed join"), 1);
        drop(log);
        let recovered = super::super::recover_frame_log_by_scan(&path).expect("scan");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].group, None);
        assert_eq!(recovered[0].seq_count, 2);
        assert_eq!(recovered[0].payload, logical);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sixteen_fragment_scatter_crosses_record_spans_and_recovers_exact_concatenation() {
        let path = test_path("sixteen-fragment");
        let bytes = (0_u32..24_007)
            .map(|value| value.wrapping_mul(17) as u8)
            .collect::<Vec<_>>();
        let expected = bytes.clone();
        let plan = FuaScatterPlan::new(bytes.len(), 16).expect("sixteen fragment plan");
        assert_eq!(plan.fragment_count(), 16);
        assert_eq!(
            plan.padded_bytes(),
            (0..16)
                .map(|index| {
                    super::super::fua_frame_padded_bytes(
                        plan.fragment_payload_bytes(index).expect("fragment bytes"),
                    )
                })
                .sum()
        );
        let mut source = CrossSpanSource {
            bytes,
            record_ends: [301, 1_901, 4_003, 9_017, 17_503],
            crossed_record_boundaries: 0,
        };
        let log = unsafe {
            super::super::FuaFrameLog::create(config(&path, 1 << 20)).expect("create frame log")
        };
        let pool = log.spawn_fixed_fence_pool(4).expect("fixed pool");
        let mut appender = log.appender();
        let batch = appender
            .publish_scatter(&mut source, plan, 7, 3)
            .expect("sixteen fragment scatter");
        assert_eq!(batch.first_frame_id, 0);
        assert_eq!(batch.terminal_frame_id, 15);
        assert!(
            source.crossed_record_boundaries > 0,
            "physical fragments must be permitted to cross logical record spans"
        );
        appender.finish();
        assert_eq!(pool.join_fixed().expect("fixed join"), 16);
        drop(log);
        let recovered = super::super::recover_frame_log_by_scan(&path).expect("scan");
        assert_eq!(recovered.len(), 16);
        assert!(recovered.iter().all(|frame| frame.group.is_some()));
        assert!(recovered[..15].iter().all(|frame| frame.seq_count == 0));
        assert_eq!(recovered[15].seq_count, 3);
        let concatenated = recovered
            .iter()
            .flat_map(|frame| frame.payload.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(concatenated, expected);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn source_failure_after_staging_a_fragment_publishes_zero_frames() {
        let path = test_path("source-failure");
        let plan = FuaScatterPlan::new(4_096, 4).expect("four fragment plan");
        let log = unsafe {
            super::super::FuaFrameLog::create(config(&path, 1 << 20)).expect("create frame log")
        };
        let mut appender = log.appender();
        let capacity_before = appender.remaining_capacity_bytes();
        let mut source = FailAfterOneFragment {
            bytes: vec![0xA5; plan.payload_bytes()],
            copies: 0,
        };
        assert_eq!(
            appender
                .publish_scatter(&mut source, plan, 0, 1)
                .expect_err("source must fail after first staged fragment")
                .stage,
            FuaFrameFaultStage::ScatterSource
        );
        assert_eq!(log.published_frames(), 0);
        assert_eq!(appender.remaining_capacity_bytes(), capacity_before);
        let one = FuaScatterPlan::new(64, 1).expect("one fragment plan");
        let mut good_source = vec![0x5A; 64];
        let recovered_cursor = appender
            .publish_scatter(&mut good_source, one, 0, 1)
            .expect("cursor remains reusable");
        assert_eq!(recovered_cursor.first_frame_id, 0);
        assert_eq!(log.published_frames(), 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scatter_capacity_preflight_has_no_visible_prefix() {
        let path = test_path("capacity");
        let plan = FuaScatterPlan::new(300, 3).expect("three fragment plan");
        let log = unsafe {
            super::super::FuaFrameLog::create(config(&path, 2 * super::super::FRAME_ALIGN))
                .expect("create frame log")
        };
        let mut appender = log.appender();
        assert_eq!(
            appender
                .can_publish_scatter(plan)
                .expect_err("three frames cannot fit")
                .stage,
            FuaFrameFaultStage::ScatterCapacity
        );
        assert_eq!(log.published_frames(), 0);
        let one = FuaScatterPlan::new(300, 1).expect("one fragment plan");
        let mut source = vec![0xD3; 300];
        appender
            .publish_scatter(&mut source, one, 0, 1)
            .expect("preflight failure does not advance the cursor");
        assert_eq!(log.published_frames(), 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scatter_publish_allocates_nothing_after_plan_and_source_exist() {
        let path = test_path("no-allocation");
        let plan = FuaScatterPlan::new(8_192, 8).expect("eight fragment plan");
        let log = unsafe {
            super::super::FuaFrameLog::create(config(&path, 1 << 20)).expect("create frame log")
        };
        let mut appender = log.appender();
        let mut source = vec![0x7C; plan.payload_bytes()];
        let batch = assert_no_allocations(|| {
            appender
                .publish_scatter(&mut source, plan, 0, 1)
                .expect("allocation-free scatter publication")
        });
        assert_eq!(batch.terminal_frame_id, 7);
        assert_eq!(log.published_frames(), 8);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reserved_scatter_roundtrips_exactly_for_one_and_sixteen_fragments() {
        for (label, fragment_count, payload_bytes) in [
            ("reserved-one", 1_usize, 1_337_usize),
            ("reserved-sixteen", 16, 24_007),
        ] {
            let path = test_path(label);
            let expected = (0..payload_bytes)
                .map(|value| value.wrapping_mul(29) as u8)
                .collect::<Vec<_>>();
            let plan = FuaScatterPlan::new(expected.len(), fragment_count).expect("reserved plan");
            let log = unsafe {
                super::super::FuaFrameLog::create(config(&path, 1 << 20)).expect("create frame log")
            };
            let pool = log.spawn_fixed_fence_pool(4).expect("fixed pool");
            let mut appender = log.appender();
            let reservation = appender.reserve_scatter(plan).expect("reserve scatter");
            let mut source = expected.clone();
            let batch = appender
                .publish_reserved_scatter(reservation, &mut source, 11, 3)
                .expect("publish reserved scatter");
            assert_eq!(batch.first_frame_id, 0);
            assert_eq!(batch.terminal_frame_id, fragment_count as u64 - 1);
            appender.finish();
            assert_eq!(
                pool.join_fixed().expect("fixed join"),
                fragment_count as u64
            );
            drop(log);
            let recovered = super::super::recover_frame_log_by_scan(&path).expect("scan");
            assert_eq!(recovered.len(), fragment_count);
            assert_eq!(recovered[0].group.is_some(), fragment_count > 1);
            let concatenated = recovered
                .iter()
                .flat_map(|frame| frame.payload.iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(concatenated, expected);
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn reserved_scatter_rejects_intervening_cursor_drift_without_a_second_prefix() {
        let path = test_path("reserved-cursor-drift");
        let plan = FuaScatterPlan::new(1_024, 4).expect("reserved plan");
        let log = unsafe {
            super::super::FuaFrameLog::create(config(&path, 1 << 20)).expect("create frame log")
        };
        let mut appender = log.appender();
        let reservation = appender.reserve_scatter(plan).expect("reserve scatter");
        let one = FuaScatterPlan::new(64, 1).expect("intervening plan");
        let mut intervening_source = vec![0x44; 64];
        appender
            .publish_scatter(&mut intervening_source, one, 0, 1)
            .expect("advance cursor");
        let mut reserved_source = vec![0xA4; plan.payload_bytes()];
        let error = appender
            .publish_reserved_scatter(reservation, &mut reserved_source, 1, 1)
            .expect_err("reserved cursor cannot be retargeted after drift");
        assert_eq!(error.stage, FuaFrameFaultStage::ScatterReservationDrift);
        assert_eq!(error.segment_id, 41);
        assert_eq!(error.frame_id, 0);
        assert_eq!(
            log.published_frames(),
            1,
            "the drifted reservation publishes zero additional frames"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reservation_capacity_slot_and_source_failures_leave_no_reserved_prefix() {
        let path = test_path("reserved-capacity");
        let log = unsafe {
            super::super::FuaFrameLog::create(config(&path, 2 * super::super::FRAME_ALIGN))
                .expect("create frame log")
        };
        let appender = log.appender();
        let too_large = FuaScatterPlan::new(300, 3).expect("three fragments");
        assert_eq!(
            appender
                .reserve_scatter(too_large)
                .expect_err("capacity reservation fails before staging")
                .stage,
            FuaFrameFaultStage::ScatterCapacity
        );
        assert_eq!(log.published_frames(), 0);
        drop(appender);
        let _ = std::fs::remove_file(path);

        let path = test_path("reserved-slots");
        let log = unsafe {
            super::super::FuaFrameLog::create(config(&path, 1 << 20)).expect("create frame log")
        };
        let mut appender = log.appender();
        appender.next_frame = super::super::FRAME_SLOTS as u64;
        let one = FuaScatterPlan::new(64, 1).expect("one fragment");
        assert_eq!(
            appender
                .reserve_scatter(one)
                .expect_err("slot reservation fails before staging")
                .stage,
            FuaFrameFaultStage::ScatterSlots
        );
        assert_eq!(log.published_frames(), 0);
        drop(appender);
        let _ = std::fs::remove_file(path);

        let path = test_path("reserved-source");
        let plan = FuaScatterPlan::new(4_096, 4).expect("four fragments");
        let log = unsafe {
            super::super::FuaFrameLog::create(config(&path, 1 << 20)).expect("create frame log")
        };
        let mut appender = log.appender();
        let reservation = appender.reserve_scatter(plan).expect("reserve scatter");
        let mut source = FailAfterOneFragment {
            bytes: vec![0x9E; plan.payload_bytes()],
            copies: 0,
        };
        assert_eq!(
            appender
                .publish_reserved_scatter(reservation, &mut source, 0, 1)
                .expect_err("source fails after a staged fragment")
                .stage,
            FuaFrameFaultStage::ScatterSource
        );
        assert_eq!(log.published_frames(), 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reserved_scatter_publish_allocates_nothing_after_reservation() {
        let path = test_path("reserved-no-allocation");
        let plan = FuaScatterPlan::new(8_192, 8).expect("eight fragment plan");
        let log = unsafe {
            super::super::FuaFrameLog::create(config(&path, 1 << 20)).expect("create frame log")
        };
        let mut appender = log.appender();
        let reservation = appender.reserve_scatter(plan).expect("reserve scatter");
        let mut source = vec![0xC7; plan.payload_bytes()];
        let batch = assert_no_allocations(|| {
            appender
                .publish_reserved_scatter(reservation, &mut source, 0, 1)
                .expect("allocation-free reserved publication")
        });
        assert_eq!(batch.terminal_frame_id, 7);
        assert_eq!(log.published_frames(), 8);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn protected_scatter_path_forbids_dynamic_allocation_escape_hatches() {
        let source = include_str!("scatter.rs");
        let (_, marked) = source
            .split_once("// BEGIN FUA_SCATTER_NO_ALLOC")
            .expect("scatter begin marker");
        let (protected, _) = marked
            .split_once("// END FUA_SCATTER_NO_ALLOC")
            .expect("scatter end marker");
        for forbidden in [
            "format!",
            ".to_string()",
            "String",
            "Vec",
            "io::Error",
            ".reserve(",
            "try_reserve",
            "retry",
            "fallback",
        ] {
            assert!(
                !protected.contains(forbidden),
                "protected scatter path must not contain {forbidden}"
            );
        }
    }

    #[test]
    fn reserved_publish_cannot_replan_retry_or_select_an_alternate_path() {
        let source = include_str!("scatter.rs");
        let (_, reserved) = source
            .split_once("pub fn publish_reserved_scatter")
            .expect("reserved publish method");
        let (reserved, _) = reserved
            .split_once("fn validate_scatter_input")
            .expect("reserved publish method end");
        for forbidden in [
            "reserve_scatter(",
            "retry",
            "fallback",
            "Vec",
            "format!",
            "String",
        ] {
            assert!(
                !reserved.contains(forbidden),
                "reserved publication must not contain {forbidden}"
            );
        }
    }
}
