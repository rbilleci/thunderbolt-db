//! Reusable physical FUA cadence controller.
//!
//! The controller owns only the physical QD policy.  Its caller owns WAL ordering and holds the
//! controller mutex from `select` through successful frame publication and `record_published`.
//! In particular, a token is not observable until the corresponding physical group is visible.

use std::collections::BTreeMap;

/// Physical frame count selected for a sustained FUA group.
pub const FUA_CONTROLLER_QD16_FRAGMENTS: usize = 16;
/// Number of eligible QD16 groups between sparse QD1 probes.
pub const FUA_CONTROLLER_SUSTAINED_GROUPS: u16 = 512;
/// A QD1 probe at or below this latency begins or continues verification.
pub const FUA_CONTROLLER_FAST_NANOS: u64 = 900_000;

/// Persistent physical-backend controller phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FuaControllerPhase {
    #[default]
    SustainedSlow,
    Verify,
    Fast,
}

impl FuaControllerPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SustainedSlow => "sustained_slow",
            Self::Verify => "verify",
            Self::Fast => "fast",
        }
    }
}

/// Why a group remains physically unfragmented.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FuaControllerReason {
    PoolTooNarrow,
    EmptyChunk,
    InsufficientFreeSlots,
    NaturalDepth,
    SegmentBoundary,
    AmplificationCap,
}

impl FuaControllerReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PoolTooNarrow => "pool_too_narrow",
            Self::EmptyChunk => "empty_chunk",
            Self::InsufficientFreeSlots => "insufficient_free_slots",
            Self::NaturalDepth => "natural_depth",
            Self::SegmentBoundary => "segment_boundary",
            Self::AmplificationCap => "amplification_cap",
        }
    }
}

/// The policy state that issued a QD1 token.  It is identity, not a latency classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FuaControllerSampleKind {
    Sparse,
    Verify,
    Fast,
}

/// One chosen physical layout.  The explicit pending-cover action is deliberately distinct from
/// an epoch QD16 action: it keeps publication moving while the one non-Fast sample is outstanding
/// and must not consume the next 512-group epoch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FuaControllerDecision {
    SustainedEpoch { fragments: usize },
    PendingProbeCover { fragments: usize },
    Qd1Sample { kind: FuaControllerSampleKind },
    Unfragmented(FuaControllerReason),
}

impl FuaControllerDecision {
    pub const fn fragments(self) -> usize {
        match self {
            Self::SustainedEpoch { fragments } | Self::PendingProbeCover { fragments } => fragments,
            Self::Qd1Sample { .. } | Self::Unfragmented(_) => 1,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SustainedEpoch { .. } => "sustained_epoch",
            Self::PendingProbeCover { .. } => "pending_probe_cover",
            Self::Qd1Sample { .. } => "qd1_sample",
            Self::Unfragmented(_) => "unfragmented",
        }
    }
}

/// Move-only identity of one published controller-owned QD1 sample.  A completion consumes the
/// token, so safe callers cannot apply a direct-service result twice.
#[must_use = "a published controller sample must be observed or abandoned"]
#[derive(Debug)]
pub struct FuaControllerSampleToken {
    generation: u64,
    publication_ordinal: u64,
    kind: FuaControllerSampleKind,
}

impl FuaControllerSampleToken {
    pub const fn kind(&self) -> FuaControllerSampleKind {
        self.kind
    }

    pub const fn publication_ordinal(&self) -> u64 {
        self.publication_ordinal
    }
}

/// All physical conditions required before the controller may fragment one logical group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FuaControllerEligibility {
    pub pool_lanes: usize,
    pub natural_depth: usize,
    pub free_slots: usize,
    pub fragment_count: usize,
    pub chunks_nonempty: bool,
    pub one_segment: bool,
    pub single_frame_padded_bytes: usize,
    pub fragmented_padded_bytes: usize,
}

impl FuaControllerEligibility {
    pub const fn ineligible_reason(self) -> Option<FuaControllerReason> {
        if self.pool_lanes < FUA_CONTROLLER_QD16_FRAGMENTS {
            Some(FuaControllerReason::PoolTooNarrow)
        } else if !self.chunks_nonempty {
            Some(FuaControllerReason::EmptyChunk)
        } else if self.fragment_count < 2 || self.natural_depth >= FUA_CONTROLLER_QD16_FRAGMENTS {
            Some(FuaControllerReason::NaturalDepth)
        } else if self.free_slots < self.fragment_count {
            Some(FuaControllerReason::InsufficientFreeSlots)
        } else if !self.one_segment {
            Some(FuaControllerReason::SegmentBoundary)
        } else if self.fragmented_padded_bytes > self.single_frame_padded_bytes.saturating_mul(2) {
            Some(FuaControllerReason::AmplificationCap)
        } else {
            None
        }
    }
}

/// Snapshot of independent physical-policy evidence.  The two reconciliation laws are exposed
/// as methods on the controller so tests and consumers can require a quiescent clean state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FuaControllerTelemetry {
    pub published_actions: u64,
    pub sustained_qd16_actions: u64,
    pub pending_probe_cover_actions: u64,
    pub qd1_probe_actions: u64,
    pub qd1_sparse_actions: u64,
    pub qd1_verify_actions: u64,
    pub qd1_fast_actions: u64,
    pub unfragmented_actions: u64,
    pub pool_too_narrow: u64,
    pub empty_chunk: u64,
    pub insufficient_free_slots: u64,
    pub natural_depth: u64,
    pub segment_boundary: u64,
    pub amplification_cap: u64,
    pub fast_probes: u64,
    pub gray_or_slow_probes: u64,
    pub unavailable_qd1_samples: u64,
    pub abandoned_qd1_samples: u64,
    pub stale_qd1_samples: u64,
    pub protocol_faults: u64,
    pub protocol_fallback_actions: u64,
    pub transitions_to_verify: u64,
    pub transitions_to_fast: u64,
    pub transitions_to_sustained: u64,
    pub phase: FuaControllerPhase,
    pub verify_fast_streak: u8,
    pub sustained_qd16_remaining: u16,
    pub generation: u64,
    pub generation_exhausted: u64,
    pub ordinal_exhausted: u64,
    pub pending_qd1_samples: u64,
    pub current_generation_serial_pending_samples: u64,
    pub fast_in_flight: u64,
    pub fast_in_flight_max: u64,
}

/// Per-device/backend sustained-first controller.  It contains no lock or durability authority;
/// the backend holds the mutex over selection, visibility publication, and token issuance.
#[derive(Debug)]
pub struct FuaPhysicalController {
    phase: FuaControllerPhase,
    verify_fast_streak: u8,
    sustained_qd16_remaining: u16,
    generation: u64,
    generation_exhausted: bool,
    current_generation_ordinal_floor: u64,
    next_publication_ordinal: u64,
    ordinal_exhausted: bool,
    outstanding: BTreeMap<u64, OutstandingSample>,
    telemetry: FuaControllerTelemetry,
}

#[derive(Clone, Copy, Debug)]
struct OutstandingSample {
    generation: u64,
    kind: FuaControllerSampleKind,
}

impl Default for FuaPhysicalController {
    fn default() -> Self {
        let mut controller = Self {
            phase: FuaControllerPhase::SustainedSlow,
            verify_fast_streak: 0,
            sustained_qd16_remaining: FUA_CONTROLLER_SUSTAINED_GROUPS,
            generation: 0,
            generation_exhausted: false,
            current_generation_ordinal_floor: 0,
            next_publication_ordinal: 0,
            ordinal_exhausted: false,
            outstanding: BTreeMap::new(),
            telemetry: FuaControllerTelemetry::default(),
        };
        controller.refresh_snapshot();
        controller
    }
}

impl FuaPhysicalController {
    /// Select without advancing state.  Callers must keep this controller exclusively held until
    /// `record_published`, otherwise an earlier sample could change the generation at the seam.
    pub fn select(&self, eligibility: FuaControllerEligibility) -> FuaControllerDecision {
        if let Some(reason) = eligibility.ineligible_reason() {
            return FuaControllerDecision::Unfragmented(reason);
        }
        let fragments = eligibility.fragment_count;
        match self.phase {
            FuaControllerPhase::SustainedSlow if self.sustained_qd16_remaining > 0 => {
                FuaControllerDecision::SustainedEpoch { fragments }
            }
            FuaControllerPhase::SustainedSlow if self.has_nonfast_outstanding() => {
                FuaControllerDecision::PendingProbeCover { fragments }
            }
            FuaControllerPhase::SustainedSlow if self.no_more_sample_identities() => {
                FuaControllerDecision::SustainedEpoch { fragments }
            }
            FuaControllerPhase::SustainedSlow => FuaControllerDecision::Qd1Sample {
                kind: FuaControllerSampleKind::Sparse,
            },
            FuaControllerPhase::Verify if self.has_nonfast_outstanding() => {
                FuaControllerDecision::PendingProbeCover { fragments }
            }
            FuaControllerPhase::Verify if self.no_more_sample_identities() => {
                FuaControllerDecision::SustainedEpoch { fragments }
            }
            FuaControllerPhase::Verify => FuaControllerDecision::Qd1Sample {
                kind: FuaControllerSampleKind::Verify,
            },
            FuaControllerPhase::Fast if self.no_more_sample_identities() => {
                FuaControllerDecision::SustainedEpoch { fragments }
            }
            FuaControllerPhase::Fast => FuaControllerDecision::Qd1Sample {
                kind: FuaControllerSampleKind::Fast,
            },
        }
    }

    /// Commit a selected decision after the corresponding physical frames are atomically visible.
    /// The ordinal is allocated here, never at `select`, so failed staging cannot create a token.
    pub fn record_published(
        &mut self,
        decision: FuaControllerDecision,
    ) -> Option<FuaControllerSampleToken> {
        self.telemetry.published_actions = self.telemetry.published_actions.saturating_add(1);
        let token = match decision {
            FuaControllerDecision::SustainedEpoch { .. } => {
                self.telemetry.sustained_qd16_actions =
                    self.telemetry.sustained_qd16_actions.saturating_add(1);
                self.sustained_qd16_remaining = self.sustained_qd16_remaining.saturating_sub(1);
                None
            }
            FuaControllerDecision::PendingProbeCover { .. } => {
                self.telemetry.pending_probe_cover_actions =
                    self.telemetry.pending_probe_cover_actions.saturating_add(1);
                None
            }
            FuaControllerDecision::Qd1Sample { kind } => {
                if self.no_more_sample_identities()
                    || !self.kind_is_current(kind)
                    || (kind != FuaControllerSampleKind::Fast && self.has_nonfast_outstanding())
                {
                    self.telemetry.protocol_faults =
                        self.telemetry.protocol_faults.saturating_add(1);
                    self.telemetry.protocol_fallback_actions =
                        self.telemetry.protocol_fallback_actions.saturating_add(1);
                    self.restart_sustained();
                    self.refresh_snapshot();
                    return None;
                }
                self.telemetry.qd1_probe_actions =
                    self.telemetry.qd1_probe_actions.saturating_add(1);
                match kind {
                    FuaControllerSampleKind::Sparse => {
                        self.telemetry.qd1_sparse_actions =
                            self.telemetry.qd1_sparse_actions.saturating_add(1)
                    }
                    FuaControllerSampleKind::Verify => {
                        self.telemetry.qd1_verify_actions =
                            self.telemetry.qd1_verify_actions.saturating_add(1)
                    }
                    FuaControllerSampleKind::Fast => {
                        self.telemetry.qd1_fast_actions =
                            self.telemetry.qd1_fast_actions.saturating_add(1)
                    }
                }
                let publication_ordinal = self.next_publication_ordinal;
                if publication_ordinal == u64::MAX {
                    self.ordinal_exhausted = true;
                } else {
                    self.next_publication_ordinal += 1;
                }
                let previous = self.outstanding.insert(
                    publication_ordinal,
                    OutstandingSample {
                        generation: self.generation,
                        kind,
                    },
                );
                debug_assert!(previous.is_none());
                Some(FuaControllerSampleToken {
                    generation: self.generation,
                    publication_ordinal,
                    kind,
                })
            }
            FuaControllerDecision::Unfragmented(reason) => {
                self.telemetry.unfragmented_actions =
                    self.telemetry.unfragmented_actions.saturating_add(1);
                self.increment_reason(reason);
                None
            }
        };
        self.refresh_snapshot();
        token
    }

    /// Settle one direct-write measurement.  The move-only token means a safe caller cannot
    /// replay a result.  A stale generation is evidence-only; an unknown *current* identity is a
    /// protocol fault and fails safely back to a fresh sustained epoch.
    pub fn observe_qd1_sample(&mut self, token: FuaControllerSampleToken, latency_nanos: u64) {
        let Some(kind) = self.take_current_token(token, Settlement::Observed) else {
            return;
        };
        if latency_nanos <= FUA_CONTROLLER_FAST_NANOS {
            self.telemetry.fast_probes = self.telemetry.fast_probes.saturating_add(1);
            match kind {
                FuaControllerSampleKind::Sparse => self.enter_verify(1),
                FuaControllerSampleKind::Verify => self.enter_verify(self.verify_fast_streak + 1),
                FuaControllerSampleKind::Fast => {}
            }
        } else {
            self.telemetry.gray_or_slow_probes =
                self.telemetry.gray_or_slow_probes.saturating_add(1);
            self.restart_sustained();
        }
        self.refresh_snapshot();
    }

    /// Settle a timestamp that cannot safely be read (slot reuse, I/O failure, or shutdown).  A
    /// current Fast sample is intentionally no exception: it returns to sustained QD16 just like
    /// a measured non-fast result.
    pub fn record_qd1_sample_unavailable(&mut self, token: FuaControllerSampleToken) {
        if self
            .take_current_token(token, Settlement::Unavailable)
            .is_none()
        {
            return;
        }
        self.telemetry.unavailable_qd1_samples =
            self.telemetry.unavailable_qd1_samples.saturating_add(1);
        self.restart_sustained();
        self.refresh_snapshot();
    }

    /// RAII users call this when a published token cannot reach normal observation.  It is a
    /// distinct reconciliation outcome, but carries the same conservative sustained fallback.
    pub fn abandon_qd1_sample(&mut self, token: FuaControllerSampleToken) {
        if self
            .take_current_token(token, Settlement::Abandoned)
            .is_none()
        {
            return;
        }
        self.telemetry.abandoned_qd1_samples =
            self.telemetry.abandoned_qd1_samples.saturating_add(1);
        self.restart_sustained();
        self.refresh_snapshot();
    }

    pub fn telemetry(&self) -> FuaControllerTelemetry {
        self.telemetry
    }

    /// Every visible group has exactly one decision accounting entry.
    pub fn action_reconciliation_ok(&self) -> bool {
        self.telemetry.published_actions
            == self
                .telemetry
                .sustained_qd16_actions
                .saturating_add(self.telemetry.pending_probe_cover_actions)
                .saturating_add(self.telemetry.qd1_probe_actions)
                .saturating_add(self.telemetry.unfragmented_actions)
                .saturating_add(self.telemetry.protocol_fallback_actions)
    }

    /// Every issued token is settled exactly once or remains outstanding. Generation changes do
    /// not erase identities: an old completion is counted as stale exactly once.
    pub fn sample_reconciliation_ok(&self) -> bool {
        self.telemetry.qd1_probe_actions
            == self
                .telemetry
                .fast_probes
                .saturating_add(self.telemetry.gray_or_slow_probes)
                .saturating_add(self.telemetry.unavailable_qd1_samples)
                .saturating_add(self.telemetry.abandoned_qd1_samples)
                .saturating_add(self.telemetry.stale_qd1_samples)
                .saturating_add(self.outstanding.len() as u64)
    }

    pub fn is_quiescent(&self) -> bool {
        self.outstanding.is_empty()
    }

    fn take_current_token(
        &mut self,
        token: FuaControllerSampleToken,
        settlement: Settlement,
    ) -> Option<FuaControllerSampleKind> {
        let Some(issued) = self.outstanding.get(&token.publication_ordinal).copied() else {
            self.telemetry.protocol_faults = self.telemetry.protocol_faults.saturating_add(1);
            self.restart_sustained();
            self.refresh_snapshot();
            return None;
        };
        if issued.generation != token.generation || issued.kind != token.kind {
            self.telemetry.protocol_faults = self.telemetry.protocol_faults.saturating_add(1);
            self.restart_sustained();
            self.refresh_snapshot();
            return None;
        }
        self.outstanding.remove(&token.publication_ordinal);
        if token.generation != self.generation
            || token.publication_ordinal < self.current_generation_ordinal_floor
            || self.generation_exhausted
        {
            match settlement {
                Settlement::Observed => {
                    self.telemetry.stale_qd1_samples =
                        self.telemetry.stale_qd1_samples.saturating_add(1)
                }
                Settlement::Unavailable => {
                    self.telemetry.unavailable_qd1_samples =
                        self.telemetry.unavailable_qd1_samples.saturating_add(1)
                }
                Settlement::Abandoned => {
                    self.telemetry.abandoned_qd1_samples =
                        self.telemetry.abandoned_qd1_samples.saturating_add(1)
                }
            }
            self.refresh_snapshot();
            return None;
        }
        Some(issued.kind)
    }

    fn enter_verify(&mut self, streak: u8) {
        self.verify_fast_streak = streak;
        if streak >= 4 {
            self.phase = FuaControllerPhase::Fast;
            self.verify_fast_streak = 0;
            self.telemetry.transitions_to_fast =
                self.telemetry.transitions_to_fast.saturating_add(1);
        } else {
            if self.phase != FuaControllerPhase::Verify {
                self.telemetry.transitions_to_verify =
                    self.telemetry.transitions_to_verify.saturating_add(1);
                self.bump_generation();
            }
            self.phase = FuaControllerPhase::Verify;
            return;
        }
        self.bump_generation();
    }

    fn restart_sustained(&mut self) {
        self.phase = FuaControllerPhase::SustainedSlow;
        self.verify_fast_streak = 0;
        self.sustained_qd16_remaining = FUA_CONTROLLER_SUSTAINED_GROUPS;
        self.telemetry.transitions_to_sustained =
            self.telemetry.transitions_to_sustained.saturating_add(1);
        self.bump_generation();
    }

    fn bump_generation(&mut self) {
        if self.generation == u64::MAX {
            // Never saturate an identity generation: a later Fast token could then look current
            // to an old completion.  Retire all current tokens and permanently use the safe QD16
            // arm until the backend is recreated.
            self.generation_exhausted = true;
            self.phase = FuaControllerPhase::SustainedSlow;
            self.verify_fast_streak = 0;
            self.sustained_qd16_remaining = FUA_CONTROLLER_SUSTAINED_GROUPS;
            self.current_generation_ordinal_floor = self.next_publication_ordinal;
            self.telemetry.generation_exhausted = 1;
            self.telemetry.protocol_faults = self.telemetry.protocol_faults.saturating_add(1);
        } else {
            self.generation += 1;
            self.current_generation_ordinal_floor = 0;
        }
    }

    fn has_nonfast_outstanding(&self) -> bool {
        self.outstanding.values().any(|sample| {
            sample.generation == self.generation && sample.kind != FuaControllerSampleKind::Fast
        })
    }

    fn kind_is_current(&self, kind: FuaControllerSampleKind) -> bool {
        matches!(
            (self.phase, kind),
            (
                FuaControllerPhase::SustainedSlow,
                FuaControllerSampleKind::Sparse
            ) | (FuaControllerPhase::Verify, FuaControllerSampleKind::Verify)
                | (FuaControllerPhase::Fast, FuaControllerSampleKind::Fast)
        )
    }

    fn no_more_sample_identities(&self) -> bool {
        self.generation_exhausted || self.ordinal_exhausted
    }

    fn increment_reason(&mut self, reason: FuaControllerReason) {
        let counter = match reason {
            FuaControllerReason::PoolTooNarrow => &mut self.telemetry.pool_too_narrow,
            FuaControllerReason::EmptyChunk => &mut self.telemetry.empty_chunk,
            FuaControllerReason::InsufficientFreeSlots => {
                &mut self.telemetry.insufficient_free_slots
            }
            FuaControllerReason::NaturalDepth => &mut self.telemetry.natural_depth,
            FuaControllerReason::SegmentBoundary => &mut self.telemetry.segment_boundary,
            FuaControllerReason::AmplificationCap => &mut self.telemetry.amplification_cap,
        };
        *counter = counter.saturating_add(1);
    }

    fn refresh_snapshot(&mut self) {
        self.telemetry.phase = self.phase;
        self.telemetry.verify_fast_streak = self.verify_fast_streak;
        self.telemetry.sustained_qd16_remaining = self.sustained_qd16_remaining;
        self.telemetry.generation = self.generation;
        self.telemetry.generation_exhausted = u64::from(self.generation_exhausted);
        self.telemetry.ordinal_exhausted = u64::from(self.ordinal_exhausted);
        self.telemetry.pending_qd1_samples = self.outstanding.len() as u64;
        self.telemetry.current_generation_serial_pending_samples = self
            .outstanding
            .values()
            .filter(|sample| {
                sample.generation == self.generation && sample.kind != FuaControllerSampleKind::Fast
            })
            .count() as u64;
        self.telemetry.fast_in_flight = self
            .outstanding
            .values()
            .filter(|sample| {
                sample.generation == self.generation && sample.kind == FuaControllerSampleKind::Fast
            })
            .count() as u64;
        self.telemetry.fast_in_flight_max = self
            .telemetry
            .fast_in_flight_max
            .max(self.telemetry.fast_in_flight);
    }
}

#[derive(Clone, Copy)]
enum Settlement {
    Observed,
    Unavailable,
    Abandoned,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eligible() -> FuaControllerEligibility {
        FuaControllerEligibility {
            pool_lanes: 32,
            natural_depth: 0,
            free_slots: 32,
            fragment_count: 16,
            chunks_nonempty: true,
            one_segment: true,
            single_frame_padded_bytes: 32 * 1024,
            fragmented_padded_bytes: 64 * 1024,
        }
    }

    fn publish(controller: &mut FuaPhysicalController) -> Option<FuaControllerSampleToken> {
        let decision = controller.select(eligible());
        controller.record_published(decision)
    }

    fn reach_sparse(controller: &mut FuaPhysicalController) -> FuaControllerSampleToken {
        for _ in 0..FUA_CONTROLLER_SUSTAINED_GROUPS {
            assert!(publish(controller).is_none());
        }
        publish(controller).expect("sparse token")
    }

    fn reach_fast(controller: &mut FuaPhysicalController) {
        let sparse = reach_sparse(controller);
        controller.observe_qd1_sample(sparse, FUA_CONTROLLER_FAST_NANOS);
        for _ in 0..3 {
            let verify = publish(controller).expect("verify token");
            controller.observe_qd1_sample(verify, FUA_CONTROLLER_FAST_NANOS);
        }
        assert_eq!(controller.telemetry().phase, FuaControllerPhase::Fast);
    }

    #[test]
    fn epoch_and_pending_cover_are_separate_and_reconcile() {
        let mut controller = FuaPhysicalController::default();
        let sparse = reach_sparse(&mut controller);
        assert_eq!(
            controller.select(eligible()),
            FuaControllerDecision::PendingProbeCover { fragments: 16 }
        );
        assert!(publish(&mut controller).is_none());
        let telemetry = controller.telemetry();
        assert_eq!(telemetry.sustained_qd16_remaining, 0);
        assert_eq!(telemetry.pending_probe_cover_actions, 1);
        controller.observe_qd1_sample(sparse, FUA_CONTROLLER_FAST_NANOS);
        assert!(controller.action_reconciliation_ok());
        assert!(controller.sample_reconciliation_ok());
        assert!(controller.is_quiescent());
    }

    #[test]
    fn exact_verify_streak_enters_fast_and_bumps_generations() {
        let mut controller = FuaPhysicalController::default();
        let sparse = reach_sparse(&mut controller);
        controller.observe_qd1_sample(sparse, FUA_CONTROLLER_FAST_NANOS);
        let first_generation = controller.telemetry().generation;
        for expected_generation in [first_generation, first_generation, first_generation + 1] {
            let verify = publish(&mut controller).expect("verify token");
            controller.observe_qd1_sample(verify, FUA_CONTROLLER_FAST_NANOS);
            assert_eq!(controller.telemetry().generation, expected_generation);
        }
        assert_eq!(controller.telemetry().phase, FuaControllerPhase::Fast);
        assert!(controller.is_quiescent());
    }

    #[test]
    fn fast_pipeline_tracks_in_flight_and_slow_result_restarts_all() {
        let mut controller = FuaPhysicalController::default();
        reach_fast(&mut controller);
        let first = publish(&mut controller).expect("fast token");
        let second = publish(&mut controller).expect("fast token");
        assert_eq!(controller.telemetry().fast_in_flight, 2);
        controller.observe_qd1_sample(first, FUA_CONTROLLER_FAST_NANOS + 1);
        assert_eq!(
            controller.telemetry().phase,
            FuaControllerPhase::SustainedSlow
        );
        assert_eq!(controller.telemetry().pending_qd1_samples, 1);
        controller.observe_qd1_sample(second, FUA_CONTROLLER_FAST_NANOS);
        assert_eq!(controller.telemetry().stale_qd1_samples, 1);
        assert!(controller.sample_reconciliation_ok());
        assert!(controller.is_quiescent());
    }

    #[test]
    fn current_fast_unavailable_restarts_sustained() {
        let mut controller = FuaPhysicalController::default();
        reach_fast(&mut controller);
        let token = publish(&mut controller).expect("fast token");
        controller.record_qd1_sample_unavailable(token);
        assert_eq!(
            controller.telemetry().phase,
            FuaControllerPhase::SustainedSlow
        );
        assert_eq!(
            controller.telemetry().sustained_qd16_remaining,
            FUA_CONTROLLER_SUSTAINED_GROUPS
        );
        assert!(controller.is_quiescent());
    }

    #[test]
    fn abandon_is_a_distinct_quiescent_settlement() {
        let mut controller = FuaPhysicalController::default();
        let token = reach_sparse(&mut controller);
        controller.abandon_qd1_sample(token);
        assert_eq!(controller.telemetry().abandoned_qd1_samples, 1);
        assert!(controller.is_quiescent());
        assert!(controller.sample_reconciliation_ok());
    }

    #[test]
    fn stale_token_is_evidence_only_after_generation_transition() {
        let mut controller = FuaPhysicalController::default();
        reach_fast(&mut controller);
        let stale = publish(&mut controller).expect("first fast token");
        let transition = publish(&mut controller).expect("second fast token");
        controller.observe_qd1_sample(transition, FUA_CONTROLLER_FAST_NANOS + 1);
        controller.observe_qd1_sample(stale, FUA_CONTROLLER_FAST_NANOS);
        assert_eq!(controller.telemetry().stale_qd1_samples, 1);
        assert!(controller.sample_reconciliation_ok());
    }

    #[test]
    fn unknown_current_token_is_protocol_fault_and_falls_back() {
        let mut controller = FuaPhysicalController::default();
        let token = reach_sparse(&mut controller);
        let forged = FuaControllerSampleToken {
            generation: token.generation,
            publication_ordinal: token.publication_ordinal.saturating_add(1),
            kind: token.kind,
        };
        controller.observe_qd1_sample(forged, FUA_CONTROLLER_FAST_NANOS);
        assert_eq!(controller.telemetry().protocol_faults, 1);
        assert_eq!(
            controller.telemetry().phase,
            FuaControllerPhase::SustainedSlow
        );
        assert!(controller.sample_reconciliation_ok());
    }

    #[test]
    fn post_visibility_protocol_fallback_keeps_action_accounting_exact() {
        let mut controller = FuaPhysicalController::default();
        assert!(controller
            .record_published(FuaControllerDecision::Qd1Sample {
                kind: FuaControllerSampleKind::Fast,
            })
            .is_none());
        assert_eq!(controller.telemetry().protocol_fallback_actions, 1);
        assert!(controller.action_reconciliation_ok());
        assert!(controller.is_quiescent());
    }

    #[test]
    fn eligibility_is_fail_closed_without_consuming_epoch() {
        let mut controller = FuaPhysicalController::default();
        let mut input = eligible();
        input.natural_depth = 16;
        let decision = controller.select(input);
        assert_eq!(
            decision,
            FuaControllerDecision::Unfragmented(FuaControllerReason::NaturalDepth)
        );
        assert!(controller.record_published(decision).is_none());
        assert_eq!(
            controller.telemetry().sustained_qd16_remaining,
            FUA_CONTROLLER_SUSTAINED_GROUPS
        );
        assert!(controller.action_reconciliation_ok());
    }

    #[test]
    fn depth_eight_needs_only_eight_fragments() {
        let controller = FuaPhysicalController::default();
        let mut input = eligible();
        input.natural_depth = 8;
        input.free_slots = 8;
        input.fragment_count = 8;
        input.fragmented_padded_bytes = 32 * 1024;
        assert_eq!(
            controller.select(input),
            FuaControllerDecision::SustainedEpoch { fragments: 8 }
        );
        input.free_slots = 7;
        assert_eq!(
            controller.select(input),
            FuaControllerDecision::Unfragmented(FuaControllerReason::InsufficientFreeSlots)
        );
    }

    #[test]
    fn mutex_owner_keeps_selection_and_token_issue_atomic_against_observation() {
        use std::sync::{Arc, Barrier, Mutex};

        let mut initial = FuaPhysicalController::default();
        for _ in 0..FUA_CONTROLLER_SUSTAINED_GROUPS {
            assert!(publish(&mut initial).is_none());
        }
        let controller = Arc::new(Mutex::new(initial));
        let barrier = Arc::new(Barrier::new(2));
        let observer_controller = Arc::clone(&controller);
        let observer_barrier = Arc::clone(&barrier);
        let observer = std::thread::spawn(move || {
            observer_barrier.wait();
            observer_barrier.wait();
            // This lock must be acquired only after the publisher issued its post-visibility
            // token.  A split select/record lock would expose qd1_probe_actions == 0 here.
            assert_eq!(
                observer_controller
                    .lock()
                    .unwrap()
                    .telemetry()
                    .qd1_probe_actions,
                1
            );
        });
        let mut guard = controller.lock().unwrap();
        let decision = guard.select(eligible());
        assert_eq!(
            decision,
            FuaControllerDecision::Qd1Sample {
                kind: FuaControllerSampleKind::Sparse
            }
        );
        barrier.wait();
        barrier.wait();
        let token = guard
            .record_published(decision)
            .expect("published sparse token");
        drop(guard);
        observer.join().expect("observer join");
        controller
            .lock()
            .unwrap()
            .record_qd1_sample_unavailable(token);
    }

    #[test]
    fn generation_exhaustion_refuses_identity_reuse() {
        let mut controller = FuaPhysicalController {
            generation: u64::MAX,
            ..Default::default()
        };
        controller.restart_sustained();
        assert_eq!(controller.telemetry().generation_exhausted, 1);
        for _ in 0..=FUA_CONTROLLER_SUSTAINED_GROUPS {
            assert!(matches!(
                controller.select(eligible()),
                FuaControllerDecision::SustainedEpoch { .. }
            ));
        }
        assert!(controller.is_quiescent());
    }
}
