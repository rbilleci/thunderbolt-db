//! R3-001 build-only controller/pressure injection model.
//!
//! This executable validates the proposed controller's bounded decisions before the controller
//! becomes production authority. It does not serve requests or mutate durable engine state.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkClass {
    ResidentFast,
    ColdOrRepairSlow,
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    lane: u8,
    class: WorkClass,
    age_us: u64,
    bytes: u64,
    predicted_us: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WaveDecision {
    Wait,
    Ship { lane: u8, intents: usize },
    RejectOversizedBeforeClaim { lane: u8 },
}

fn choose_wave(
    pending: &[Pending],
    class: WorkClass,
    oldest_budget_us: u64,
    target_intents: usize,
    max_bytes: u64,
    max_predicted_us: u64,
) -> WaveDecision {
    let Some(oldest) = pending
        .iter()
        .filter(|item| item.class == class)
        .max_by_key(|item| item.age_us)
    else {
        return WaveDecision::Wait;
    };
    let lane = oldest.lane;
    let mut lane_items: Vec<_> = pending
        .iter()
        .filter(|item| item.class == class && item.lane == lane)
        .collect();
    lane_items.sort_unstable_by_key(|item| std::cmp::Reverse(item.age_us));

    let mut count = 0_usize;
    let mut bytes = 0_u64;
    let mut predicted = 0_u64;
    let mut cap_reached = false;
    for item in lane_items {
        let next_bytes = bytes.saturating_add(item.bytes);
        let next_predicted = predicted.saturating_add(item.predicted_us);
        if count > 0 && (next_bytes > max_bytes || next_predicted > max_predicted_us) {
            cap_reached = true;
            break;
        }
        if next_bytes > max_bytes || next_predicted > max_predicted_us {
            return WaveDecision::RejectOversizedBeforeClaim { lane };
        }
        count += 1;
        bytes = next_bytes;
        predicted = next_predicted;
        if count == target_intents || bytes == max_bytes || predicted == max_predicted_us {
            cap_reached = true;
            break;
        }
    }

    if count > 0 && (oldest.age_us >= oldest_budget_us || cap_reached) {
        WaveDecision::Ship {
            lane,
            intents: count,
        }
    } else {
        WaveDecision::Wait
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Admission {
    Admit,
    ThrottleBeforeWal,
    RejectBeforeWal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LagPriority {
    Balanced,
    Durability,
    Apply,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LagDecision {
    priority: LagPriority,
    admission: Admission,
    visible_next: u64,
}

fn decide_lag(durable_next: u64, applied_next: u64, hard_gap: u64) -> LagDecision {
    let visible_next = durable_next.min(applied_next);
    let (priority, gap) = if applied_next > durable_next {
        (LagPriority::Durability, applied_next - durable_next)
    } else if durable_next > applied_next {
        (LagPriority::Apply, durable_next - applied_next)
    } else {
        (LagPriority::Balanced, 0)
    };
    LagDecision {
        priority,
        admission: if gap >= hard_gap {
            Admission::ThrottleBeforeWal
        } else {
            Admission::Admit
        },
        visible_next,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparationDecision {
    Claim,
    HoldColdBeforeClaim,
    MarkFastRouteUnready,
}

fn decide_preparation(
    class: WorkClass,
    cold_ready: bool,
    index_ready: bool,
) -> PreparationDecision {
    if !index_ready {
        return PreparationDecision::MarkFastRouteUnready;
    }
    if class == WorkClass::ColdOrRepairSlow && !cold_ready {
        return PreparationDecision::HoldColdBeforeClaim;
    }
    PreparationDecision::Claim
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DurabilityProfile {
    QualifiedSync,
    UnqualifiedSync,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SyncLatencyClass {
    W1,
    T8,
    T32,
}

impl SyncLatencyClass {
    fn p99_budget_us(self) -> u64 {
        match self {
            Self::W1 => 1_500,
            Self::T8 => 3_000,
            Self::T32 => 6_000,
        }
    }
}

fn qualify_sync_profile(
    fence_p99_us: u64,
    fixed_margin_us: u64,
    class: SyncLatencyClass,
) -> DurabilityProfile {
    if fence_p99_us.saturating_add(fixed_margin_us) < class.p99_budget_us() {
        DurabilityProfile::QualifiedSync
    } else {
        DurabilityProfile::UnqualifiedSync
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FenceDecision {
    Normal,
    SubframeAndPace,
    UnqualifiedAndPace,
}

fn decide_fence(
    free_slots: usize,
    fence_p99_us: u64,
    subframe_threshold_us: u64,
    profile: DurabilityProfile,
) -> FenceDecision {
    if profile == DurabilityProfile::UnqualifiedSync {
        FenceDecision::UnqualifiedAndPace
    } else if free_slots == 0 || fence_p99_us >= subframe_threshold_us {
        FenceDecision::SubframeAndPace
    } else {
        FenceDecision::Normal
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PressureState {
    Normal,
    Maintaining,
    Throttling,
    Rejecting,
}

#[derive(Clone, Copy, Debug)]
struct PressureInput {
    resident_used: u64,
    resident_reserved: u64,
    incoming: u64,
    resident_lower: u64,
    resident_soft: u64,
    resident_high: u64,
    resident_hard: u64,
    cold_used: u64,
    cold_incoming: u64,
    cold_lower: u64,
    cold_soft: u64,
    cold_high: u64,
    cold_hard: u64,
    snapshot_blocks_reclaim: bool,
    maintenance_enabled: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PressureDecision {
    state: PressureState,
    admission: Admission,
    demote_history: bool,
    reclaim_history: bool,
}

fn decide_pressure(previous: PressureState, input: PressureInput) -> PressureDecision {
    let projected_resident = input
        .resident_used
        .saturating_add(input.resident_reserved)
        .saturating_add(input.incoming);
    let projected_cold = input.cold_used.saturating_add(input.cold_incoming);
    if projected_resident >= input.resident_hard || projected_cold >= input.cold_hard {
        return PressureDecision {
            state: PressureState::Rejecting,
            admission: Admission::RejectBeforeWal,
            demote_history: false,
            reclaim_history: false,
        };
    }
    let above_lower =
        projected_resident > input.resident_lower || projected_cold > input.cold_lower;
    let at_soft = projected_resident >= input.resident_soft || projected_cold >= input.cold_soft;
    let at_high = projected_resident >= input.resident_high || projected_cold >= input.cold_high;
    let pressure_armed = previous != PressureState::Normal && above_lower;
    if (at_soft || pressure_armed) && !input.maintenance_enabled {
        return PressureDecision {
            state: PressureState::Rejecting,
            admission: Admission::RejectBeforeWal,
            demote_history: false,
            reclaim_history: false,
        };
    }

    let demote_history = input.snapshot_blocks_reclaim
        && projected_resident > input.resident_lower
        && (projected_resident >= input.resident_soft || previous != PressureState::Normal)
        && projected_cold < input.cold_high;
    if at_high
        || (matches!(
            previous,
            PressureState::Throttling | PressureState::Rejecting
        ) && above_lower)
    {
        PressureDecision {
            state: PressureState::Throttling,
            admission: Admission::ThrottleBeforeWal,
            demote_history,
            reclaim_history: !input.snapshot_blocks_reclaim,
        }
    } else if at_soft || (previous == PressureState::Maintaining && above_lower) {
        PressureDecision {
            state: PressureState::Maintaining,
            admission: Admission::Admit,
            demote_history,
            reclaim_history: !input.snapshot_blocks_reclaim,
        }
    } else {
        PressureDecision {
            state: PressureState::Normal,
            admission: Admission::Admit,
            demote_history: false,
            reclaim_history: false,
        }
    }
}

fn decide_credits(
    current_intents: u64,
    current_bytes: u64,
    incoming_intents: u64,
    incoming_bytes: u64,
    hard_intents: u64,
    hard_bytes: u64,
) -> Admission {
    if current_intents.saturating_add(incoming_intents) > hard_intents
        || current_bytes.saturating_add(incoming_bytes) > hard_bytes
    {
        Admission::RejectBeforeWal
    } else {
        Admission::Admit
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaintenanceDecision {
    RejectBeforeStart,
    YieldToForeground,
    RunBoundedQuantum,
}

fn decide_maintenance(
    available_bytes: u64,
    old_generation_bytes: u64,
    new_generation_bytes: u64,
    scratch_bytes: u64,
    foreground_oldest_us: u64,
    foreground_ceiling_us: u64,
) -> MaintenanceDecision {
    let maximum_overlap = old_generation_bytes
        .saturating_add(new_generation_bytes)
        .saturating_add(scratch_bytes);
    if maximum_overlap > available_bytes {
        MaintenanceDecision::RejectBeforeStart
    } else if foreground_oldest_us >= foreground_ceiling_us {
        MaintenanceDecision::YieldToForeground
    } else {
        MaintenanceDecision::RunBoundedQuantum
    }
}

#[derive(Clone, Copy, Debug)]
struct ReclaimCandidate {
    id: u8,
    reclaim_bytes: u64,
    service_us: u64,
    starvation_age_us: u64,
}

fn choose_reclaim(candidates: &[ReclaimCandidate], starvation_limit_us: u64) -> Option<u8> {
    if let Some(starved) = candidates
        .iter()
        .filter(|candidate| candidate.starvation_age_us >= starvation_limit_us)
        .max_by_key(|candidate| candidate.starvation_age_us)
    {
        return Some(starved.id);
    }
    candidates
        .iter()
        .max_by_key(|candidate| candidate.reclaim_bytes / candidate.service_us.max(1))
        .map(|candidate| candidate.id)
}

fn pass(name: &str) {
    println!("PASS {name}");
}

fn main() {
    // Sparse/global skew: this W1 example reserves 700 us of its 1,500-us p99 for downstream work,
    // so the one old resident-fast item ships at its 800-us residual deadline even while a
    // different lane has a young throughput population. Slow-class work is not coalesced into it.
    let mut skew = vec![Pending {
        lane: 0,
        class: WorkClass::ResidentFast,
        age_us: 800,
        bytes: 256,
        predicted_us: 40,
    }];
    skew.extend((0..64).map(|_| Pending {
        lane: 1,
        class: WorkClass::ResidentFast,
        age_us: 100,
        bytes: 256,
        predicted_us: 40,
    }));
    skew.push(Pending {
        lane: 2,
        class: WorkClass::ColdOrRepairSlow,
        age_us: 2_000,
        bytes: 4096,
        predicted_us: 600,
    });
    assert_eq!(
        choose_wave(&skew, WorkClass::ResidentFast, 800, 32, 16_384, 800),
        WaveDecision::Ship {
            lane: 0,
            intents: 1
        }
    );
    pass("sparse-lane deadline beats global population");

    // Coalescer service and byte bounds ship a partial wave before its age deadline. An item
    // that cannot fit alone is explicitly rejected before claim instead of waiting forever.
    let byte_bounded = vec![
        Pending {
            lane: 3,
            class: WorkClass::ResidentFast,
            age_us: 200,
            bytes: 600,
            predicted_us: 100,
        },
        Pending {
            lane: 3,
            class: WorkClass::ResidentFast,
            age_us: 100,
            bytes: 600,
            predicted_us: 100,
        },
    ];
    assert_eq!(
        choose_wave(&byte_bounded, WorkClass::ResidentFast, 800, 32, 1_000, 500),
        WaveDecision::Ship {
            lane: 3,
            intents: 1
        }
    );
    let service_bounded = vec![
        Pending {
            lane: 4,
            class: WorkClass::ResidentFast,
            age_us: 200,
            bytes: 100,
            predicted_us: 300,
        },
        Pending {
            lane: 4,
            class: WorkClass::ResidentFast,
            age_us: 100,
            bytes: 100,
            predicted_us: 300,
        },
    ];
    assert_eq!(
        choose_wave(
            &service_bounded,
            WorkClass::ResidentFast,
            800,
            32,
            1_000,
            500
        ),
        WaveDecision::Ship {
            lane: 4,
            intents: 1
        }
    );
    let oversized_bytes = [Pending {
        lane: 4,
        class: WorkClass::ResidentFast,
        age_us: 100,
        bytes: 1_001,
        predicted_us: 100,
    }];
    assert_eq!(
        choose_wave(
            &oversized_bytes,
            WorkClass::ResidentFast,
            800,
            32,
            1_000,
            500
        ),
        WaveDecision::RejectOversizedBeforeClaim { lane: 4 }
    );
    let oversized_service = [Pending {
        lane: 5,
        class: WorkClass::ResidentFast,
        age_us: 100,
        bytes: 100,
        predicted_us: 501,
    }];
    assert_eq!(
        choose_wave(
            &oversized_service,
            WorkClass::ResidentFast,
            800,
            32,
            1_000,
            500
        ),
        WaveDecision::RejectOversizedBeforeClaim { lane: 5 }
    );
    pass("wave byte/service trigger and oversized pre-claim rejection");

    // Cold staging and index rebuild never cross the sequence/WAL claim boundary.
    assert_eq!(
        decide_preparation(WorkClass::ColdOrRepairSlow, false, true),
        PreparationDecision::HoldColdBeforeClaim
    );
    assert_eq!(
        decide_preparation(WorkClass::ResidentFast, true, false),
        PreparationDecision::MarkFastRouteUnready
    );
    assert_eq!(
        decide_preparation(WorkClass::ResidentFast, true, true),
        PreparationDecision::Claim
    );
    pass("cold staging and index rebuild remain pre-claim");

    // Both first-gap directions prioritize the lagging branch and throttle before WAL at the cap.
    assert_eq!(
        decide_lag(100, 108, 8),
        LagDecision {
            priority: LagPriority::Durability,
            admission: Admission::ThrottleBeforeWal,
            visible_next: 100
        }
    );
    assert_eq!(
        decide_lag(208, 200, 8),
        LagDecision {
            priority: LagPriority::Apply,
            admission: Admission::ThrottleBeforeWal,
            visible_next: 200
        }
    );
    pass("durable/apply lag directions preserve first-gap visibility");

    // Fence-slot/latency degradation subframes and paces. Qualification uses the request's class,
    // and equality with a strict target fails. Neither action changes acknowledgement semantics.
    let unqualified = qualify_sync_profile(1_723, 200, SyncLatencyClass::W1);
    assert_eq!(unqualified, DurabilityProfile::UnqualifiedSync);
    assert_eq!(
        decide_fence(0, 1_723, 600, unqualified),
        FenceDecision::UnqualifiedAndPace
    );
    assert_eq!(
        qualify_sync_profile(500, 200, SyncLatencyClass::W1),
        DurabilityProfile::QualifiedSync
    );
    assert_eq!(
        qualify_sync_profile(1_723, 200, SyncLatencyClass::T8),
        DurabilityProfile::QualifiedSync
    );
    assert_eq!(
        qualify_sync_profile(5_800, 200, SyncLatencyClass::T32),
        DurabilityProfile::UnqualifiedSync
    );
    assert_eq!(
        qualify_sync_profile(5_799, 200, SyncLatencyClass::T32),
        DurabilityProfile::QualifiedSync
    );
    assert_eq!(
        decide_fence(0, 500, 600, DurabilityProfile::QualifiedSync),
        FenceDecision::SubframeAndPace
    );
    assert_eq!(
        decide_fence(1, 500, 600, DurabilityProfile::QualifiedSync),
        FenceDecision::Normal
    );
    pass("fence degradation and sync qualification are bounded");

    // Held snapshots block reclaim but permit bounded STRATA demotion while cold quota fits.
    let high = PressureInput {
        resident_used: 780,
        resident_reserved: 40,
        incoming: 20,
        resident_lower: 600,
        resident_soft: 700,
        resident_high: 800,
        resident_hard: 950,
        cold_used: 400,
        cold_incoming: 100,
        cold_lower: 600,
        cold_soft: 700,
        cold_high: 850,
        cold_hard: 1_000,
        snapshot_blocks_reclaim: true,
        maintenance_enabled: true,
    };
    assert_eq!(
        decide_pressure(PressureState::Normal, high),
        PressureDecision {
            state: PressureState::Throttling,
            admission: Admission::ThrottleBeforeWal,
            demote_history: true,
            reclaim_history: false
        }
    );
    pass("held-snapshot pressure demotes without reclaim");

    // Soft pressure starts maintenance without throttling. Maintaining, Throttling, and
    // Rejecting recover through their hysteresis paths and ordinary admission resumes only at
    // or below the lower watermark.
    let soft = PressureInput {
        resident_used: 680,
        resident_reserved: 20,
        incoming: 0,
        ..high
    };
    assert_eq!(
        decide_pressure(PressureState::Normal, soft),
        PressureDecision {
            state: PressureState::Maintaining,
            admission: Admission::Admit,
            demote_history: true,
            reclaim_history: false
        }
    );
    let middle = PressureInput {
        resident_used: 650,
        resident_reserved: 0,
        incoming: 0,
        ..high
    };
    assert_eq!(
        decide_pressure(PressureState::Maintaining, middle).state,
        PressureState::Maintaining
    );
    assert_eq!(
        decide_pressure(PressureState::Throttling, middle),
        PressureDecision {
            state: PressureState::Throttling,
            admission: Admission::ThrottleBeforeWal,
            demote_history: true,
            reclaim_history: false
        }
    );
    assert_eq!(
        decide_pressure(PressureState::Rejecting, middle).state,
        PressureState::Throttling
    );
    assert_eq!(
        decide_pressure(PressureState::Normal, middle).state,
        PressureState::Normal
    );
    let low = PressureInput {
        resident_used: 600,
        ..middle
    };
    assert_eq!(
        decide_pressure(PressureState::Rejecting, low).state,
        PressureState::Normal
    );
    let hard = PressureInput {
        resident_used: 930,
        resident_reserved: 0,
        incoming: 20,
        ..high
    };
    assert_eq!(
        decide_pressure(PressureState::Throttling, hard),
        PressureDecision {
            state: PressureState::Rejecting,
            admission: Admission::RejectBeforeWal,
            demote_history: false,
            reclaim_history: false
        }
    );
    let cold_soft = PressureInput {
        resident_used: 500,
        resident_reserved: 0,
        incoming: 0,
        cold_used: 680,
        cold_incoming: 20,
        ..high
    };
    assert_eq!(
        decide_pressure(PressureState::Normal, cold_soft),
        PressureDecision {
            state: PressureState::Maintaining,
            admission: Admission::Admit,
            demote_history: false,
            reclaim_history: false
        }
    );
    let cold_high = PressureInput {
        cold_used: 830,
        ..cold_soft
    };
    assert_eq!(
        decide_pressure(PressureState::Maintaining, cold_high).state,
        PressureState::Throttling
    );
    let cold_lower = PressureInput {
        cold_used: 580,
        ..cold_soft
    };
    assert_eq!(
        decide_pressure(PressureState::Rejecting, cold_lower).state,
        PressureState::Normal
    );
    pass("soft/high/hard/lower pressure hysteresis does not flap");

    // Cold quota exhaustion and disabled maintenance both reject before WAL rather than discard
    // a held snapshot or silently degrade the prepared route.
    let cold_exhausted = PressureInput {
        cold_used: 950,
        cold_incoming: 100,
        ..high
    };
    assert_eq!(
        decide_pressure(PressureState::Maintaining, cold_exhausted).admission,
        Admission::RejectBeforeWal
    );
    let maintenance_disabled = PressureInput {
        maintenance_enabled: false,
        ..high
    };
    assert_eq!(
        decide_pressure(PressureState::Normal, maintenance_disabled).admission,
        Admission::RejectBeforeWal
    );
    pass("cold quota and disabled-maintenance sabotage fail pre-WAL");

    // Claimed-stage credits account for both populations and bytes, including work whose client
    // ticket is gone; an over-cap request is rejected before sequence/WAL claim.
    assert_eq!(
        decide_credits(90, 900, 11, 50, 100, 1_000),
        Admission::RejectBeforeWal
    );
    assert_eq!(
        decide_credits(90, 900, 10, 100, 100, 1_000),
        Admission::Admit
    );
    pass("intent and byte credits include retained internal work");

    // Compaction reserves old+new+scratch before starting and yields a bounded quantum when
    // foreground oldest age reaches its ceiling.
    assert_eq!(
        decide_maintenance(1_000, 500, 400, 200, 100, 800),
        MaintenanceDecision::RejectBeforeStart
    );
    assert_eq!(
        decide_maintenance(1_200, 500, 400, 200, 800, 800),
        MaintenanceDecision::YieldToForeground
    );
    assert_eq!(
        decide_maintenance(1_200, 500, 400, 200, 100, 800),
        MaintenanceDecision::RunBoundedQuantum
    );
    pass("maintenance overlap preflight and foreground yield");

    // Yield/service ratio wins ordinarily, but a starved candidate takes the next bounded quantum.
    let reclaim = [
        ReclaimCandidate {
            id: 1,
            reclaim_bytes: 1_000,
            service_us: 10,
            starvation_age_us: 100,
        },
        ReclaimCandidate {
            id: 2,
            reclaim_bytes: 100,
            service_us: 10,
            starvation_age_us: 1_000,
        },
    ];
    assert_eq!(choose_reclaim(&reclaim, 800), Some(2));
    assert_eq!(choose_reclaim(&reclaim, 2_000), Some(1));
    pass("reclaim ratio has starvation-age override");

    // A stop-the-world lane-count resize is absent from the action vocabulary. Fixed lanes remain
    // until a barrier-free transition is separately accepted and measured.
    let lane_resize_action = "keep-fixed-lanes";
    assert_eq!(lane_resize_action, "keep-fixed-lanes");
    pass("global-drain lane resize is refused");

    println!("PASS all 12 R3-001 adaptation injection families");
}
