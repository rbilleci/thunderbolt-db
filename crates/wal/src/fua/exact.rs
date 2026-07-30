//! Exact typed-WAL ownership for the FUA frame-log backend.
//!
//! This leaf owns the one logical exact group permitted by FUA at a time.  The mutable ledger is
//! deliberately backend-local: the WAL buffer owns logical record order, while this owner seals
//! the physical segment/cursor/controller geometry before proposal and carries it unchanged into
//! the one-release scatter publication.

use super::{
    saturating_add, ControllerSampleSettlement, FuaWalBackend, PublishedGroup, SPIN_BEFORE_YIELD,
};
use crate::buffer::group::{PreparedWalGroup, MAX_WAL_GROUP_RECORDS, MAX_WAL_GROUP_WIRE_BYTES};
use gpu_db_types::{DurabilityBackend, DurabilityFault, DurabilityStage, EngineError};
use gpu_db_write_conveyor::{
    fua_frame_padded_bytes, FuaControllerDecision, FuaControllerEligibility, FuaFrameFault,
    FuaFrameFaultStage, FuaScatterPlan, FuaScatterReservation, FUA_CONTROLLER_QD16_FRAGMENTS,
};
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct FuaExactLedger {
    generation: u64,
    state: ExactLedgerState,
}

impl Default for FuaExactLedger {
    fn default() -> Self {
        Self {
            generation: 0,
            state: ExactLedgerState::Idle,
        }
    }
}

// Forming state carries the complete fixed cursor/geometry seal inline. Indirection would add an
// allocation to exact pre-WAL admission and make that allocation absent from the sealed ledger.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum ExactLedgerState {
    Idle,
    Forming(ExactFormingGroup),
    InFlight(ExactInFlightGroup),
    Poisoned(DurabilityFault),
}

#[derive(Debug)]
struct ExactFormingGroup {
    generation: u64,
    first_record: usize,
    target_records: usize,
    wire_bytes: usize,
    segment_id: u64,
    plan: FuaScatterPlan,
    decision: FuaControllerDecision,
    eligibility: FuaControllerEligibility,
    reservation: FuaScatterReservation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExactInFlightGroup {
    generation: u64,
    first_record: usize,
    target_records: usize,
    wire_bytes: usize,
    segment_id: u64,
    fragment_count: usize,
    padded_bytes: usize,
}

/// Move-only extension guard for the rollbackable pre-proposal portion of one exact FUA group.
///
/// Each added exact record replaces the forming group's immutable cursor seal with a seal for its
/// complete new extent.  Dropping an unclaimed guard restores the prior tail exactly; consuming it
/// at logical claim makes that extension non-rollbackable without allocating another carrier.
#[must_use = "a FUA exact reservation must be claimed or dropped before proposal completes"]
#[derive(Debug)]
pub(crate) struct FuaExactReservation {
    backend: Arc<FuaWalBackend>,
    generation: u64,
    first_record: usize,
    prior: Option<ExactLedgerState>,
}

/// Pre-proposal exact admission is deliberately two-valued: a fully sealed current group must be
/// drained before a valid next record can extend it, but that condition is internal backpressure,
/// not a user-visible unsupported-shape error.
// The reserved arm is deliberately inline: exact admission may return only its preallocated
// move-only proof, never allocate a wrapper around it.
#[allow(clippy::large_enum_variant)]
pub(crate) enum FuaExactReservationAttempt {
    Reserved(FuaExactReservation),
    SealPrior,
}

enum ExactGeometryRefusal {
    SealPrior,
    Fatal(EngineError),
}

struct ExactGeometry {
    eligibility: FuaControllerEligibility,
    plan: FuaScatterPlan,
    decision: FuaControllerDecision,
    fresh_decision: FuaControllerDecision,
    /// The current suffix forced `SegmentBoundary`, but the same controller state would choose
    /// fragmentation on a fresh segment.  Exact admission must activate the successor before
    /// proposal instead of silently degrading that physical intent to QD1.
    requires_fresh_segment: bool,
}

impl FuaExactReservation {
    pub(crate) fn claim(mut self) -> Result<(), EngineError> {
        let ledger = self
            .backend
            .exact_ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let valid = matches!(
            &ledger.state,
            ExactLedgerState::Forming(forming)
                if forming.generation == self.generation && forming.first_record == self.first_record
        );
        drop(ledger);
        if !valid {
            let fault = self.backend.fail_exact(DurabilityFault::new(
                DurabilityBackend::FuaWal,
                DurabilityStage::FuaDescriptor,
                None,
                0,
                self.first_record as u64,
            ));
            return Err(EngineError::DurabilityFault(fault));
        }
        self.prior.take();
        Ok(())
    }
}

impl Drop for FuaExactReservation {
    fn drop(&mut self) {
        let Some(prior) = self.prior.take() else {
            return;
        };
        let mut ledger = self
            .backend
            .exact_ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let restore = matches!(
            &ledger.state,
            ExactLedgerState::Forming(forming)
                if forming.generation == self.generation && forming.first_record == self.first_record
        );
        if restore {
            ledger.state = prior;
            return;
        }
        drop(ledger);
        let _ = self.backend.fail_exact(DurabilityFault::new(
            DurabilityBackend::FuaWal,
            DurabilityStage::FuaDescriptor,
            None,
            0,
            self.first_record as u64,
        ));
    }
}

/// Exact physical ownership after the descriptor leaves the forming ledger.  The only path that
/// consumes the reserved cursor is [`FuaWalBackend::commit_exact_group`].
#[derive(Debug)]
pub(crate) struct FuaExactHandoff {
    generation: u64,
    ticket: u64,
    first_record: usize,
    target_records: usize,
    wire_bytes: usize,
    segment_id: u64,
    fragment_count: usize,
    padded_bytes: usize,
    decision: FuaControllerDecision,
    eligibility: FuaControllerEligibility,
    reservation: Option<FuaScatterReservation>,
}

impl FuaWalBackend {
    pub(crate) fn exact_fault(&self) -> Option<DurabilityFault> {
        self.fixed_poison.snapshot()
    }

    pub(crate) fn exact_forming_or_in_flight(&self) -> bool {
        let ledger = self
            .exact_ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        matches!(
            ledger.state,
            ExactLedgerState::Forming(_) | ExactLedgerState::InFlight(_)
        )
    }

    /// A durable cut is not itself settlement: the owning job must still consume its controller
    /// token, return the permanent descriptor, and transition `InFlight -> Idle`.  Preproposal
    /// admission waits for that owner rather than mistaking the short handoff seam for failed
    /// capacity progress.
    pub(crate) fn wait_exact_settlement(&self) -> Result<(), EngineError> {
        let mut spins = 0u32;
        loop {
            if let Some(fault) = self.exact_fault() {
                return Err(EngineError::DurabilityFault(fault));
            }
            let in_flight = {
                let ledger = self
                    .exact_ledger
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                matches!(ledger.state, ExactLedgerState::InFlight(_))
            };
            if !in_flight {
                return Ok(());
            }
            if spins < SPIN_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Retain the complete currently-forming exact byte extent before replication proposal.  The
    /// caller has already drained any legacy tail and proved every preceding member is exact.
    pub(crate) fn reserve_exact_forming(
        self: &Arc<Self>,
        first_record: usize,
        target_records: usize,
        wire_bytes: usize,
    ) -> Result<FuaExactReservationAttempt, EngineError> {
        if let Some(fault) = self.exact_fault() {
            return Err(EngineError::DurabilityFault(fault));
        }
        let record_count = target_records.checked_sub(first_record).ok_or_else(|| {
            EngineError::ProposalFailed("typed exact FUA group record cursor underflow".to_string())
        })?;
        let mut ledger = self
            .exact_ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let prior = std::mem::replace(&mut ledger.state, ExactLedgerState::Idle);
        if record_count == 0 || wire_bytes == 0 {
            ledger.state = prior;
            return Err(EngineError::ProposalFailed(
                "typed exact FUA group has an invalid zero record or wire envelope".to_string(),
            ));
        }
        // A bounded next record can make the *aggregate* forming group too large.  That is
        // ordinary internal sealing pressure: publish the exact prefix and retry this record as
        // the first member of a fresh group.  Only an idle single-record request is a real
        // client-visible resource refusal.
        if record_count > MAX_WAL_GROUP_RECORDS || wire_bytes > MAX_WAL_GROUP_WIRE_BYTES {
            match prior {
                ExactLedgerState::Idle => {
                    ledger.state = ExactLedgerState::Idle;
                    return Err(EngineError::ProposalFailed(
                        "typed exact FUA record exceeds its bounded record or wire envelope"
                            .to_string(),
                    ));
                }
                ExactLedgerState::Forming(forming) => {
                    ledger.state = ExactLedgerState::Forming(forming);
                    return Ok(FuaExactReservationAttempt::SealPrior);
                }
                ExactLedgerState::InFlight(in_flight) => {
                    ledger.state = ExactLedgerState::InFlight(in_flight);
                    return Ok(FuaExactReservationAttempt::SealPrior);
                }
                ExactLedgerState::Poisoned(fault) => {
                    ledger.state = ExactLedgerState::Poisoned(fault);
                    return Err(EngineError::DurabilityFault(fault));
                }
            }
        }
        let allow_activate_successor = match &prior {
            ExactLedgerState::Idle => true,
            ExactLedgerState::Forming(forming)
                if forming.first_record == first_record
                    && forming.target_records.checked_add(1) == Some(target_records)
                    && forming.wire_bytes < wire_bytes =>
            {
                false
            }
            ExactLedgerState::Forming(_) => {
                ledger.state = prior;
                return Err(EngineError::ProposalFailed(
                    "typed exact FUA extension no longer matches the sealed forming group"
                        .to_string(),
                ));
            }
            ExactLedgerState::InFlight(in_flight) => {
                ledger.state = ExactLedgerState::InFlight(*in_flight);
                return Ok(FuaExactReservationAttempt::SealPrior);
            }
            ExactLedgerState::Poisoned(fault) => {
                ledger.state = ExactLedgerState::Poisoned(*fault);
                return Err(EngineError::DurabilityFault(*fault));
            }
        };

        let reservation = self.reserve_exact_geometry(wire_bytes, allow_activate_successor);
        let (segment_id, plan, decision, eligibility, reservation) = match reservation {
            Ok(reservation) => reservation,
            Err(ExactGeometryRefusal::SealPrior) => {
                ledger.state = prior;
                return Ok(FuaExactReservationAttempt::SealPrior);
            }
            Err(ExactGeometryRefusal::Fatal(error)) => {
                ledger.state = prior;
                return Err(error);
            }
        };
        let generation = ledger
            .generation
            .checked_add(1)
            .expect("FUA exact group generation exhausted");
        ledger.generation = generation;
        ledger.state = ExactLedgerState::Forming(ExactFormingGroup {
            generation,
            first_record,
            target_records,
            wire_bytes,
            segment_id,
            plan,
            decision,
            eligibility,
            reservation,
        });
        Ok(FuaExactReservationAttempt::Reserved(FuaExactReservation {
            backend: Arc::clone(self),
            generation,
            first_record,
            prior: Some(prior),
        }))
    }

    /// Move the sole forming exact cursor seal into a group job.  This occurs only after every
    /// group member has crossed the logical claim boundary, so a later failure is fail-stop.
    pub(crate) fn handoff_exact_group(
        &self,
        group: &PreparedWalGroup,
    ) -> Result<FuaExactHandoff, EngineError> {
        if !group.contains_only_exact() {
            return self.exact_handoff_fault(group.first_record(), DurabilityStage::FuaDescriptor);
        }
        let mut ledger = self
            .exact_ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let state = std::mem::replace(&mut ledger.state, ExactLedgerState::Idle);
        let ExactLedgerState::Forming(forming) = state else {
            ledger.state = state;
            drop(ledger);
            return self.exact_handoff_fault(group.first_record(), DurabilityStage::FuaDescriptor);
        };
        if forming.first_record != group.first_record()
            || forming.target_records != group.target_records()
            || forming.wire_bytes != group.wire_bytes()
            || forming.plan.payload_bytes() != group.wire_bytes()
        {
            ledger.state = ExactLedgerState::Forming(forming);
            drop(ledger);
            return self.exact_handoff_fault(group.first_record(), DurabilityStage::FuaDescriptor);
        }
        if self.published.load(Ordering::Acquire) != forming.first_record {
            ledger.state = ExactLedgerState::Forming(forming);
            drop(ledger);
            return self.exact_handoff_fault(group.first_record(), DurabilityStage::FuaFrontier);
        }
        let ticket = self.next_ticket();
        self.published
            .store(forming.target_records, Ordering::Release);
        ledger.state = ExactLedgerState::InFlight(ExactInFlightGroup {
            generation: forming.generation,
            first_record: forming.first_record,
            target_records: forming.target_records,
            wire_bytes: forming.wire_bytes,
            segment_id: forming.segment_id,
            fragment_count: forming.plan.fragment_count(),
            padded_bytes: forming.plan.padded_bytes(),
        });
        Ok(FuaExactHandoff {
            generation: forming.generation,
            ticket,
            first_record: forming.first_record,
            target_records: forming.target_records,
            wire_bytes: forming.wire_bytes,
            segment_id: forming.segment_id,
            fragment_count: forming.plan.fragment_count(),
            padded_bytes: forming.plan.padded_bytes(),
            decision: forming.decision,
            eligibility: forming.eligibility,
            reservation: Some(forming.reservation),
        })
    }

    /// Publish one pre-reserved exact group and wait for its durable cut.  No retry, segment roll,
    /// materializer, allocation, or controller re-selection is permitted after this point.
    // BEGIN FUA_EXACT_POST_HANDOFF_NO_ALLOC
    pub(crate) fn commit_exact_group(
        &self,
        mut handoff: FuaExactHandoff,
        mut group: PreparedWalGroup,
    ) -> Result<usize, EngineError> {
        if let Some(fault) = self.exact_fault() {
            group.poison(fault);
            return Err(EngineError::DurabilityFault(fault));
        }
        if let Err(fault) = self.wait_exact_publish_turn(&handoff) {
            return Err(self.fail_exact_group(group, fault));
        }
        #[cfg(test)]
        if let Some((stage, raw_os_error)) = self.take_exact_commit_test_fault() {
            let fault = self.exact_durability_fault(&handoff, stage, raw_os_error);
            return Err(self.fail_exact_group(group, fault));
        }
        let published_result = {
            let mut active = self.lock_active();
            if active.segment_id != handoff.segment_id {
                Err(self.exact_durability_fault(&handoff, DurabilityStage::FuaReservation, None))
            } else {
                let mut controller = self
                    .controller
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if controller.select(handoff.eligibility) != handoff.decision {
                    Err(self.exact_durability_fault(&handoff, DurabilityStage::FuaController, None))
                } else {
                    let publish = active
                        .appender
                        .as_mut()
                        .expect("active FUA exact appender must be present")
                        .publish_reserved_scatter(
                            handoff.reservation.take().expect(
                                "exact FUA handoff may consume its sealed scatter only once",
                            ),
                            &mut group,
                            handoff.first_record as u64,
                            u32::try_from(handoff.target_records - handoff.first_record)
                                .expect("bounded FUA exact record count"),
                        );
                    match publish {
                        Ok(handle) => {
                            let controller_sample = controller
                                .record_published(handoff.decision)
                                .map(|token| ControllerSampleSettlement {
                                    controller: Arc::clone(&self.controller),
                                    token: Some(token),
                                });
                            self.publish_cursor
                                .store(handoff.ticket + 1, Ordering::Release);
                            // Match legacy FUA attribution at the physical publication boundary:
                            // a later fence failure is still a published logical group whose
                            // durable cut never arrived, not an unaccounted alternate route.
                            saturating_add(&self.stat_logical_groups, 1);
                            saturating_add(
                                &self.stat_logical_payload_bytes,
                                handoff.wire_bytes as u64,
                            );
                            saturating_add(
                                &self.stat_single_frame_padded_baseline_bytes,
                                fua_frame_padded_bytes(handoff.wire_bytes) as u64,
                            );
                            Ok(PublishedGroup {
                                log: Arc::clone(&active.log),
                                terminal_frame_id: handle.terminal_frame_id,
                                controller_sample,
                            })
                        }
                        Err(frame_fault) => Err(self.frame_fault(&handoff, frame_fault)),
                    }
                }
            }
        };
        let mut published = match published_result {
            Ok(published) => published,
            Err(fault) => return Err(self.fail_exact_group(group, fault)),
        };
        let mut spins = 0u32;
        loop {
            if published.log.durable_seq() >= handoff.target_records as u64 {
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
            if let Some(frame_fault) = published.log.fixed_fault() {
                return Err(self.fail_exact_group(group, self.frame_fault(&handoff, frame_fault)));
            }
            if spins < SPIN_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        if let Err(fault) = self.settle_exact(&handoff) {
            return Err(self.fail_exact_group(group, fault));
        }
        {
            let mut stats = self
                .stats
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.flush_groups += 1;
            stats.durable_records += (handoff.target_records - handoff.first_record) as u64;
            stats.max_group_size = stats
                .max_group_size
                .max(handoff.target_records - handoff.first_record);
        }
        group.finish_success();
        Ok(self.durable_records())
    }
    // END FUA_EXACT_POST_HANDOFF_NO_ALLOC

    pub(crate) fn abandon_exact_group(&self, handoff: FuaExactHandoff, group: PreparedWalGroup) {
        let fault = self.exact_durability_fault(&handoff, DurabilityStage::FuaAbandoned, None);
        let _ = self.fail_exact_group(group, fault);
    }

    pub(crate) fn reject_legacy_while_exact_active(&self) -> Result<(), EngineError> {
        if let Some(fault) = self.exact_fault() {
            return Err(EngineError::DurabilityFault(fault));
        }
        if self.exact_forming_or_in_flight() {
            return Err(EngineError::ProposalFailed(
                "legacy FUA flush cannot consume the exact group owner".to_string(),
            ));
        }
        Ok(())
    }

    fn reserve_exact_geometry(
        &self,
        wire_bytes: usize,
        allow_activate_successor: bool,
    ) -> Result<
        (
            u64,
            FuaScatterPlan,
            FuaControllerDecision,
            FuaControllerEligibility,
            FuaScatterReservation,
        ),
        ExactGeometryRefusal,
    > {
        let active = self.lock_active();
        let selected = self
            .exact_geometry_for_active(&active, wire_bytes)
            .map_err(ExactGeometryRefusal::Fatal)?;
        if selected.requires_fresh_segment {
            if !allow_activate_successor {
                return Err(ExactGeometryRefusal::SealPrior);
            }
            return self.activate_exact_successor_and_reserve(active, wire_bytes);
        }
        let reserve = active
            .appender
            .as_ref()
            .expect("active FUA exact appender must be present")
            .reserve_scatter(selected.plan);
        if let Ok(reservation) = reserve {
            return Ok((
                active.segment_id,
                selected.plan,
                selected.decision,
                selected.eligibility,
                reservation,
            ));
        }
        if !allow_activate_successor {
            return Err(ExactGeometryRefusal::SealPrior);
        }
        self.activate_exact_successor_and_reserve(active, wire_bytes)
    }

    fn activate_exact_successor_and_reserve(
        &self,
        mut active: std::sync::MutexGuard<'_, super::ActiveSegment>,
        wire_bytes: usize,
    ) -> Result<
        (
            u64,
            FuaScatterPlan,
            FuaControllerDecision,
            FuaControllerEligibility,
            FuaScatterReservation,
        ),
        ExactGeometryRefusal,
    > {
        let selected = self
            .exact_geometry_for_active(&active, wire_bytes)
            .map_err(ExactGeometryRefusal::Fatal)?;
        let fresh_plan = FuaScatterPlan::new(wire_bytes, selected.fresh_decision.fragments())
            .expect("controller decisions always select a nonempty bounded scatter plan");
        if fresh_plan.padded_bytes() > self.segment_bytes {
            return Err(ExactGeometryRefusal::Fatal(EngineError::ProposalFailed(
                "typed exact FUA record exceeds the configured segment capacity before WAL"
                    .to_string(),
            )));
        }
        if self
            .prestaged
            .preflight(
                active.segment_id,
                fresh_plan.padded_bytes(),
                fresh_plan.fragment_count(),
                self.lanes,
            )
            .is_none()
        {
            return Err(ExactGeometryRefusal::Fatal(EngineError::ProposalFailed(
                "typed exact FUA record needs a current fully prepared successor before WAL"
                    .to_string(),
            )));
        }
        self.roll(&mut active)
            .map_err(ExactGeometryRefusal::Fatal)?;
        self.kick_prestage(active.segment_id);
        let selected = self
            .exact_geometry_for_active(&active, wire_bytes)
            .map_err(ExactGeometryRefusal::Fatal)?;
        if selected.requires_fresh_segment {
            return Err(ExactGeometryRefusal::Fatal(EngineError::ProposalFailed(
                "activated FUA successor cannot retain the sealed exact scatter geometry"
                    .to_string(),
            )));
        }
        let reservation = active
            .appender
            .as_ref()
            .expect("activated FUA exact appender must be present")
            .reserve_scatter(selected.plan)
            .map_err(|fault| {
                ExactGeometryRefusal::Fatal(EngineError::ProposalFailed(format!(
                    "typed exact FUA successor cursor refused its pre-WAL reservation: {fault}"
                )))
            })?;
        Ok((
            active.segment_id,
            selected.plan,
            selected.decision,
            selected.eligibility,
            reservation,
        ))
    }

    fn exact_geometry_for_active(
        &self,
        active: &super::ActiveSegment,
        wire_bytes: usize,
    ) -> Result<ExactGeometry, EngineError> {
        let free_slots = active.log.free_fence_slots(self.lanes);
        let natural_depth = self.lanes.saturating_sub(free_slots);
        let fragment_count = FUA_CONTROLLER_QD16_FRAGMENTS
            .saturating_sub(natural_depth.min(FUA_CONTROLLER_QD16_FRAGMENTS));
        let fragmented = FuaScatterPlan::new(wire_bytes, fragment_count);
        let single = FuaScatterPlan::new(wire_bytes, 1).ok_or_else(|| {
            EngineError::ProposalFailed(
                "typed exact FUA record has invalid zero/overflow geometry".to_string(),
            )
        })?;
        let one_segment = fragmented.is_some_and(|plan| {
            active
                .appender
                .as_ref()
                .expect("active FUA exact appender must be present")
                .can_publish_scatter(plan)
                .is_ok()
        });
        let eligibility = FuaControllerEligibility {
            pool_lanes: self.lanes,
            natural_depth,
            free_slots,
            fragment_count,
            chunks_nonempty: fragmented.is_some(),
            one_segment,
            single_frame_padded_bytes: single.padded_bytes(),
            fragmented_padded_bytes: fragmented.map_or(usize::MAX, FuaScatterPlan::padded_bytes),
        };
        let controller = self
            .controller
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let decision = controller.select(eligibility);
        let fresh_decision = controller.select(FuaControllerEligibility {
            one_segment: true,
            ..eligibility
        });
        drop(controller);
        let plan = FuaScatterPlan::new(wire_bytes, decision.fragments()).ok_or_else(|| {
            EngineError::ProposalFailed(
                "typed exact FUA controller selected invalid scatter geometry".to_string(),
            )
        })?;
        Ok(ExactGeometry {
            eligibility,
            plan,
            decision,
            fresh_decision,
            requires_fresh_segment: !one_segment
                && fresh_decision.fragments() >= 2
                && fragmented
                    .is_some_and(|candidate| candidate.padded_bytes() <= self.segment_bytes),
        })
    }

    fn wait_exact_publish_turn(&self, handoff: &FuaExactHandoff) -> Result<(), DurabilityFault> {
        let started = std::time::Instant::now();
        let mut spins = 0u32;
        while self.publish_cursor.load(Ordering::Acquire) != handoff.ticket {
            if let Some(fault) = self.exact_fault() {
                return Err(fault);
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
            started.elapsed().as_nanos().try_into().unwrap_or(u64::MAX),
        );
        saturating_add(&self.stat_publish_turn_wait_groups, 1);
        Ok(())
    }

    fn settle_exact(&self, handoff: &FuaExactHandoff) -> Result<(), DurabilityFault> {
        let mut ledger = self
            .exact_ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let expected = ExactInFlightGroup {
            generation: handoff.generation,
            first_record: handoff.first_record,
            target_records: handoff.target_records,
            wire_bytes: handoff.wire_bytes,
            segment_id: handoff.segment_id,
            fragment_count: handoff.fragment_count,
            padded_bytes: handoff.padded_bytes,
        };
        if matches!(&ledger.state, ExactLedgerState::InFlight(in_flight) if *in_flight == expected)
        {
            ledger.state = ExactLedgerState::Idle;
            return Ok(());
        }
        drop(ledger);
        Err(self.exact_durability_fault(handoff, DurabilityStage::FuaDescriptor, None))
    }

    fn fail_exact_group(&self, group: PreparedWalGroup, fault: DurabilityFault) -> EngineError {
        let first = self.fail_exact(fault);
        group.poison(first);
        EngineError::DurabilityFault(first)
    }

    fn fail_exact(&self, fault: DurabilityFault) -> DurabilityFault {
        let first = self.fixed_poison.install(fault);
        let mut ledger = self
            .exact_ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ledger.state = ExactLedgerState::Poisoned(first);
        drop(ledger);
        self.poisoned.store(true, Ordering::Release);
        first
    }

    fn exact_handoff_fault<T>(
        &self,
        first_record: usize,
        stage: DurabilityStage,
    ) -> Result<T, EngineError> {
        let fault = self.fail_exact(DurabilityFault::new(
            DurabilityBackend::FuaWal,
            stage,
            None,
            0,
            first_record as u64,
        ));
        Err(EngineError::DurabilityFault(fault))
    }

    fn exact_durability_fault(
        &self,
        handoff: &FuaExactHandoff,
        stage: DurabilityStage,
        raw_os_error: Option<i32>,
    ) -> DurabilityFault {
        DurabilityFault::new(
            DurabilityBackend::FuaWal,
            stage,
            raw_os_error,
            handoff.segment_id,
            handoff.first_record as u64,
        )
    }

    fn frame_fault(&self, handoff: &FuaExactHandoff, fault: FuaFrameFault) -> DurabilityFault {
        let stage = match fault.stage {
            FuaFrameFaultStage::ScatterSource
            | FuaFrameFaultStage::ScatterPlan
            | FuaFrameFaultStage::ScatterCapacity
            | FuaFrameFaultStage::ScatterSlots
            | FuaFrameFaultStage::ScatterSequence => DurabilityStage::FuaScatter,
            FuaFrameFaultStage::ScatterReservationDrift => DurabilityStage::FuaReservation,
            FuaFrameFaultStage::FenceIo
            | FuaFrameFaultStage::FenceWriteZero
            | FuaFrameFaultStage::FenceWriteOverflow
            | FuaFrameFaultStage::FencePoison
            | FuaFrameFaultStage::FrameSlot
            | FuaFrameFaultStage::Join
            | FuaFrameFaultStage::Spawn => DurabilityStage::FuaFence,
            FuaFrameFaultStage::Frontier => DurabilityStage::FuaFrontier,
        };
        self.exact_durability_fault(handoff, stage, fault.raw_os_error)
    }
}
