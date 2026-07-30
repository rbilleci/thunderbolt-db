use super::*;
use crate::engine_insert_plan::pre_wal_footprint::{
    AggregateTypedInsertShape, GpuGenerationWitness, GpuTargetKey, LogicalStatementContribution,
    PreWalFinalPhysicalFootprintInput, PreWalFootprintInput, PreWalGpuFootprint,
    PreWalLogicalFootprintInput, PreWalOverlayLink, PreWalPublicationTarget,
    PreWalStatementIdentity, PreWalTransactionOwnerIdentity, RequiresAggregateFormatDecision,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Barrier};

fn target() -> GpuTargetKey {
    GpuTargetKey {
        gpu_id: 0,
        table_oid: 7,
        schema_digest: [7; 32],
    }
}

fn current() -> CurrentTwoFragmentShape {
    CurrentTwoFragmentShape::new(
        PreWalFootprintInput {
            logical: PreWalLogicalFootprintInput {
                host_plan_retained_bytes: 17,
                host_allocation_slots: 3,
                typed_shadow_retained_bytes: 19,
                host_generation_pin_slots: 1,
                host_scratch_peak_bytes: 39,
                statement_identity: PreWalStatementIdentity {
                    owner: PreWalTransactionOwnerIdentity {
                        txn_id: 5,
                        registration_nonce: 7,
                    },
                    overlay_link: PreWalOverlayLink {
                        before_generation: 9,
                        after_generation: 10,
                        before_root_digest: [0x31; 32],
                        after_root_digest: [0x32; 32],
                    },
                    statement_ordinal: 0,
                },
                row_id_bytes: 16,
                row_id_slots: 2,
                sequence_effect_bytes: 24,
                sequence_effect_slots: 3,
                terminal_response_bytes: 41,
                terminal_response_slots: 1,
            },
            final_physical: PreWalFinalPhysicalFootprintInput {
                gpus: vec![PreWalGpuFootprint {
                    target: target(),
                    generation: GpuGenerationWitness {
                        catalog_seq: 7,
                        predecessor_boundary: 11,
                        open_shard_id: 3,
                        row_start: 13,
                        row_count: 17,
                        capacity: 23,
                        index_mutation_epoch_even: 28,
                    },
                    old_generation_pinned_bytes: 23,
                    new_persistent_bytes: 29,
                    plan_retained_transient_bytes: 31,
                    result_retained_device_bytes: 43,
                    allocation_slots: 5,
                    generation_pin_slots: 7,
                    scratch_peak_bytes: 37,
                    max_host_readback_bytes: 0,
                }]
                .into(),
                publication_targets: vec![PreWalPublicationTarget {
                    target: target(),
                    bytes: 32,
                    slots: 1,
                }]
                .into(),
            },
        },
        101,
    )
    .expect("fixture current shape is valid")
}

fn request() -> PreWalCapacityRequest {
    current().into_capacity_request()
}

fn aggregate_shape() -> AggregateTypedInsertShape {
    use crate::typed_insert_aggregate::{
        AggregateSectionMeasure, TypedInsertAggregateMeasure, AGGREGATE_FLAG_EXPLICIT,
        AGGREGATE_SECTION_COUNT, OUTER_CONTENT_ROW, OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
    };
    let first = current().into_logical_statement();
    let second = PreWalLogicalFootprintInput {
        host_plan_retained_bytes: 17,
        host_allocation_slots: 3,
        typed_shadow_retained_bytes: 19,
        host_generation_pin_slots: 1,
        host_scratch_peak_bytes: 39,
        statement_identity: PreWalStatementIdentity {
            owner: PreWalTransactionOwnerIdentity {
                txn_id: 5,
                registration_nonce: 7,
            },
            overlay_link: PreWalOverlayLink {
                before_generation: 10,
                after_generation: 11,
                before_root_digest: [0x32; 32],
                after_root_digest: [0x33; 32],
            },
            statement_ordinal: 1,
        },
        row_id_bytes: 16,
        row_id_slots: 2,
        sequence_effect_bytes: 24,
        sequence_effect_slots: 3,
        terminal_response_bytes: 41,
        terminal_response_slots: 1,
    };
    let second = LogicalStatementContribution::new(second, 103).unwrap();
    let unresolved = RequiresAggregateFormatDecision::from_two(first, second)
        .unwrap()
        .bind_final_overlay(PreWalFinalPhysicalFootprintInput {
            gpus: vec![PreWalGpuFootprint {
                target: target(),
                generation: GpuGenerationWitness {
                    catalog_seq: 7,
                    predecessor_boundary: 11,
                    open_shard_id: 3,
                    row_start: 13,
                    row_count: 17,
                    capacity: 23,
                    index_mutation_epoch_even: 28,
                },
                old_generation_pinned_bytes: 23,
                new_persistent_bytes: 29,
                plan_retained_transient_bytes: 31,
                result_retained_device_bytes: 43,
                allocation_slots: 5,
                generation_pin_slots: 7,
                scratch_peak_bytes: 37,
                max_host_readback_bytes: 0,
            }]
            .into(),
            publication_targets: vec![PreWalPublicationTarget {
                target: target(),
                bytes: 32,
                slots: 1,
            }]
            .into(),
        })
        .unwrap();
    let mut sections = [AggregateSectionMeasure {
        entry_count: 0,
        payload_bytes: 0,
    }; AGGREGATE_SECTION_COUNT];
    sections[0] = AggregateSectionMeasure {
        entry_count: 2,
        payload_bytes: 224,
    };
    sections[1] = AggregateSectionMeasure {
        entry_count: 2,
        payload_bytes: 96,
    };
    sections[3] = AggregateSectionMeasure {
        entry_count: 4,
        payload_bytes: 256,
    };
    sections[5] = AggregateSectionMeasure {
        entry_count: 2,
        payload_bytes: 272,
    };
    sections[6] = AggregateSectionMeasure {
        entry_count: 1,
        payload_bytes: 160,
    };
    let layout = TypedInsertAggregateMeasure {
        flags: AGGREGATE_FLAG_EXPLICIT,
        outer_flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
        stable_transaction_id: 5,
        statement_count: 2,
        insert_statement_count: 2,
        original_inserted_row_count: 4,
        final_row_transition_count: 4,
        allocator_before: 100,
        allocator_high_water: 104,
        table_block_count: 1,
        sections,
    }
    .measure()
    .unwrap();
    unresolved
        .bind_typed_insert_aggregate_layout(layout)
        .unwrap()
}

fn larger_aggregate_request() -> PreWalCapacityRequest {
    let mut request = request();
    request.footprint.host_plan_retained_bytes += 11;
    request.demand.global.host_retained_bytes += 11;
    request.footprint.host_allocation_slots += 2;
    request.demand.global.host_allocation_slots += 2;
    request.footprint.gpus[0].new_persistent_bytes += 13;
    request.footprint.gpus[0].allocation_slots += 1;
    request
}

#[test]
fn aggregate_entry_point_admits_only_the_resolved_codec_five_shape() {
    let request = aggregate_shape().into_capacity_request();
    let (limits, authoritative) = exact_limits(&request);
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let lease = pool
        .try_acquire_aggregate(aggregate_shape())
        .expect("exact codec-5 aggregate capacity is admitted");
    assert_eq!(lease.footprint().wal_record_slots, 1);
    assert_eq!(lease.footprint().wal_frame_slots, 3);
    drop(lease);
    assert!(empty_used(&pool));
}

fn exact_pool(request: &PreWalCapacityRequest) -> PreWalGpuPoolFootprint {
    let mut pools = request.footprint.gpu_pool_cursor();
    let pool = pools
        .try_next()
        .expect("fixture pool arithmetic fits")
        .expect("fixture names one GPU");
    assert!(pools
        .try_next()
        .expect("fixture pool arithmetic fits")
        .is_none());
    pool
}

fn exact_limits(
    request: &PreWalCapacityRequest,
) -> (PreWalCapacityLimits, Box<[PreWalAuthoritativeGpuResidency]>) {
    let global = &request.demand.global;
    let gpu = exact_pool(request);
    let envelope = gpu
        .old_generation_pinned_bytes
        .checked_add(gpu.new_persistent_bytes)
        .expect("fixture generation evidence fits");
    (
        PreWalCapacityLimits {
            host_retained_bytes: global.host_retained_bytes,
            host_allocation_slots: global.host_allocation_slots,
            host_scratch_peak_bytes: global.host_scratch_peak_bytes,
            host_generation_pin_slots: global.host_generation_pin_slots,
            wal_packed_record_bytes: global.wal_packed_record_bytes,
            wal_serialized_record_bytes: global.wal_serialized_record_bytes,
            wal_record_slots: global.wal_record_slots,
            wal_frame_slots: global.wal_frame_slots,
            row_id_slots: global.row_id_slots,
            sequence_effect_slots: global.sequence_effect_slots,
            status_index_bytes: global.status_index_bytes,
            status_index_slots: global.status_index_slots,
            terminal_response_bytes: global.terminal_response_bytes,
            terminal_response_slots: global.terminal_response_slots,
            completion_bytes: global.completion_bytes,
            completion_slots: global.completion_slots,
            publication_bytes: global.publication_bytes,
            publication_target_slots: global.publication_target_slots,
            publication_join_entry_bytes: global.publication_join_entry_bytes,
            publication_join_entry_slots: global.publication_join_entry_slots,
            gpus: vec![PreWalGpuCapacityLimit {
                gpu_id: gpu.gpu_id,
                hard_device_bytes: gpu.old_generation_pinned_bytes
                    + gpu.incremental_device_peak_bytes().unwrap(),
                plan_retained_transient_bytes: gpu.plan_retained_transient_bytes,
                result_retained_device_bytes: gpu.result_retained_device_bytes,
                scratch_peak_bytes: gpu.scratch_peak_bytes,
                allocation_slots: gpu.allocation_slots,
                generation_pin_slots: gpu.generation_pin_slots,
                generation_envelope_bytes: Some(envelope),
            }]
            .into(),
        },
        vec![PreWalAuthoritativeGpuResidency {
            gpu_id: gpu.gpu_id,
            bytes: gpu.old_generation_pinned_bytes,
        }]
        .into(),
    )
}

fn empty_used(pool: &PreWalCapacityPool) -> bool {
    let state = lock_recover(&pool.state);
    state.global == PreWalGlobalUsed::default()
        && state
            .gpus
            .iter()
            .all(|gpu| gpu.used == PreWalGpuUsed::default())
}

fn double_additive_limits(limits: &mut PreWalCapacityLimits) {
    limits.host_retained_bytes *= 2;
    limits.host_allocation_slots *= 2;
    limits.host_scratch_peak_bytes *= 2;
    limits.host_generation_pin_slots *= 2;
    limits.wal_packed_record_bytes *= 2;
    limits.wal_serialized_record_bytes *= 2;
    limits.wal_record_slots *= 2;
    limits.wal_frame_slots *= 2;
    limits.row_id_slots *= 2;
    limits.sequence_effect_slots *= 2;
    limits.status_index_bytes *= 2;
    limits.status_index_slots *= 2;
    limits.terminal_response_bytes *= 2;
    limits.terminal_response_slots *= 2;
    limits.completion_bytes *= 2;
    limits.completion_slots *= 2;
    limits.publication_bytes *= 2;
    limits.publication_target_slots *= 2;
    limits.publication_join_entry_bytes *= 2;
    limits.publication_join_entry_slots *= 2;
    limits.gpus[0].hard_device_bytes *= 2;
    limits.gpus[0].plan_retained_transient_bytes *= 2;
    limits.gpus[0].result_retained_device_bytes *= 2;
    limits.gpus[0].scratch_peak_bytes *= 2;
    limits.gpus[0].allocation_slots *= 2;
    limits.gpus[0].generation_pin_slots *= 2;
}

fn unbound_global_limits(limits: &mut PreWalCapacityLimits) {
    for limit in [
        &mut limits.host_retained_bytes,
        &mut limits.host_allocation_slots,
        &mut limits.host_scratch_peak_bytes,
        &mut limits.host_generation_pin_slots,
        &mut limits.wal_packed_record_bytes,
        &mut limits.wal_serialized_record_bytes,
        &mut limits.wal_record_slots,
        &mut limits.wal_frame_slots,
        &mut limits.row_id_slots,
        &mut limits.sequence_effect_slots,
        &mut limits.status_index_bytes,
        &mut limits.status_index_slots,
        &mut limits.terminal_response_bytes,
        &mut limits.terminal_response_slots,
        &mut limits.completion_bytes,
        &mut limits.completion_slots,
        &mut limits.publication_bytes,
        &mut limits.publication_target_slots,
        &mut limits.publication_join_entry_bytes,
        &mut limits.publication_join_entry_slots,
    ] {
        *limit = u64::MAX;
    }
}

macro_rules! exact_and_one_short_global {
    ($field:ident, $domain:literal) => {{
        let exact_request = request();
        let (limits, authoritative) = exact_limits(&exact_request);
        let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
        drop(pool.try_acquire_request(exact_request).unwrap());

        let short_request = request();
        let (mut limits, authoritative) = exact_limits(&short_request);
        limits.$field -= 1;
        let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
        let error = pool.try_acquire_request(short_request).unwrap_err();
        assert_eq!(
            error.cause(),
            &PreWalCapacityError::ResourceExhausted($domain)
        );
        assert!(empty_used(&pool));
    }};
}

macro_rules! exact_and_one_short_gpu {
    ($field:ident, $domain:literal) => {{
        let exact_request = request();
        let (limits, authoritative) = exact_limits(&exact_request);
        let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
        drop(pool.try_acquire_request(exact_request).unwrap());

        let short_request = request();
        let (mut limits, authoritative) = exact_limits(&short_request);
        limits.gpus[0].$field -= 1;
        let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
        let error = pool.try_acquire_request(short_request).unwrap_err();
        assert_eq!(
            error.cause(),
            &PreWalCapacityError::ResourceExhausted($domain)
        );
        assert!(empty_used(&pool));
    }};
}

#[test]
fn every_global_domain_admits_exactly_and_rejects_one_short() {
    exact_and_one_short_global!(host_retained_bytes, "host retained bytes");
    exact_and_one_short_global!(host_allocation_slots, "host allocation slots");
    exact_and_one_short_global!(host_scratch_peak_bytes, "host scratch peak bytes");
    exact_and_one_short_global!(host_generation_pin_slots, "host generation-pin slots");
    exact_and_one_short_global!(wal_packed_record_bytes, "WAL packed-record bytes");
    exact_and_one_short_global!(wal_serialized_record_bytes, "WAL serialized-record bytes");
    exact_and_one_short_global!(wal_record_slots, "WAL record slots");
    exact_and_one_short_global!(wal_frame_slots, "WAL frame slots");
    exact_and_one_short_global!(row_id_slots, "row-id slots");
    exact_and_one_short_global!(sequence_effect_slots, "sequence-effect slots");
    exact_and_one_short_global!(status_index_bytes, "status-index bytes");
    exact_and_one_short_global!(status_index_slots, "status-index slots");
    exact_and_one_short_global!(terminal_response_bytes, "terminal-response bytes");
    exact_and_one_short_global!(terminal_response_slots, "terminal-response slots");
    exact_and_one_short_global!(completion_bytes, "completion bytes");
    exact_and_one_short_global!(completion_slots, "completion slots");
    exact_and_one_short_global!(publication_bytes, "publication bytes");
    exact_and_one_short_global!(publication_target_slots, "publication target slots");
    exact_and_one_short_global!(publication_join_entry_bytes, "publication-join entry bytes");
    exact_and_one_short_global!(publication_join_entry_slots, "publication-join entry slots");
}

#[test]
fn every_gpu_domain_admits_exactly_and_rejects_one_short() {
    exact_and_one_short_gpu!(hard_device_bytes, "GPU hard device bytes");
    exact_and_one_short_gpu!(
        plan_retained_transient_bytes,
        "GPU plan-retained transient bytes"
    );
    exact_and_one_short_gpu!(
        result_retained_device_bytes,
        "GPU result-retained device bytes"
    );
    exact_and_one_short_gpu!(scratch_peak_bytes, "GPU scratch peak bytes");
    exact_and_one_short_gpu!(allocation_slots, "GPU allocation slots");
    exact_and_one_short_gpu!(generation_pin_slots, "GPU generation-pin slots");

    let exact_request = request();
    let (limits, authoritative) = exact_limits(&exact_request);
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    drop(pool.try_acquire_request(exact_request).unwrap());

    let short_request = request();
    let (mut limits, authoritative) = exact_limits(&short_request);
    *limits.gpus[0].generation_envelope_bytes.as_mut().unwrap() -= 1;
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let error = pool.try_acquire_request(short_request).unwrap_err();
    assert!(matches!(
        error.cause(),
        PreWalCapacityError::GenerationEnvelopeExceeded { gpu_id: 0, .. }
    ));
    assert!(empty_used(&pool));
}

#[test]
fn candidate_overflow_and_late_rejection_leave_no_partial_ledger() {
    let first = request();
    let (mut limits, authoritative) = exact_limits(&first);
    limits.host_retained_bytes = u64::MAX;
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let lease = pool.try_acquire_request(first).unwrap();

    let mut overflow = request();
    overflow.demand.global.host_retained_bytes = u64::MAX;
    let error = pool.try_acquire_request(overflow).unwrap_err();
    assert_eq!(
        error.cause(),
        &PreWalCapacityError::Overflow("host retained bytes")
    );
    assert_eq!(lock_recover(&pool.state).global.host_retained_bytes, 76);
    drop(lease);
    assert!(empty_used(&pool));

    let rejected = request();
    let (mut limits, authoritative) = exact_limits(&rejected);
    limits.publication_join_entry_slots -= 1;
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let error = pool.try_acquire_request(rejected).unwrap_err();
    assert_eq!(
        error.cause(),
        &PreWalCapacityError::ResourceExhausted("publication-join entry slots")
    );
    assert!(empty_used(&pool));
}

#[test]
fn late_gpu_failure_and_overflow_leave_global_and_hard_candidates_uninstalled() {
    let rejected = request();
    let (mut limits, authoritative) = exact_limits(&rejected);
    limits.gpus[0].allocation_slots -= 1;
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let error = pool.try_acquire_request(rejected).unwrap_err();
    assert_eq!(
        error.cause(),
        &PreWalCapacityError::ResourceExhausted("GPU allocation slots")
    );
    assert!(empty_used(&pool));

    let overflow = request();
    let (mut limits, authoritative) = exact_limits(&overflow);
    limits.host_retained_bytes = u64::MAX;
    limits.gpus[0].hard_device_bytes = u64::MAX;
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let incremental = exact_pool(&overflow)
        .incremental_device_peak_bytes()
        .unwrap();
    let prior = u64::MAX - incremental + 1;
    lock_recover(&pool.state).gpus[0]
        .used
        .incremental_device_peak_bytes = prior;
    let error = pool.try_acquire_request(overflow).unwrap_err();
    assert_eq!(
        error.cause(),
        &PreWalCapacityError::Overflow("GPU hard device bytes")
    );
    let state = lock_recover(&pool.state);
    assert_eq!(state.global, PreWalGlobalUsed::default());
    assert_eq!(state.gpus[0].used.incremental_device_peak_bytes, prior);
    drop(state);
}

#[test]
fn rejected_request_retries_after_the_incumbent_lease_drops() {
    let first = request();
    let (limits, authoritative) = exact_limits(&first);
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let incumbent = pool.try_acquire_request(first).unwrap();
    let rejected = pool.try_acquire(current()).unwrap_err();
    assert_eq!(
        rejected.cause(),
        &PreWalCapacityError::ResourceExhausted("host retained bytes")
    );
    let retry = rejected.into_request();
    drop(incumbent);
    drop(pool.try_acquire_request(retry).unwrap());
    assert!(empty_used(&pool));
}

#[test]
fn explicit_aggregate_replacement_reserves_net_candidate_not_old_plus_candidate() {
    let candidate = larger_aggregate_request();
    let (limits, authoritative) = exact_limits(&candidate);
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let incumbent = pool.try_acquire_request(request()).unwrap();

    let replacement = incumbent
        .try_replace(candidate)
        .expect("exact candidate replaces the old aggregate atomically");
    assert_eq!(replacement.footprint().host_plan_retained_bytes, 17 + 11);
    let blocked = pool.try_acquire_request(request()).unwrap_err();
    assert!(matches!(
        blocked.cause(),
        PreWalCapacityError::ResourceExhausted(_)
    ));
    drop(blocked);
    drop(replacement);
    assert!(empty_used(&pool));
    drop(
        pool.try_acquire_request(larger_aggregate_request())
            .expect("replacement drop releases exactly the candidate aggregate"),
    );
}

#[test]
fn rejected_replacement_returns_incumbent_lease_and_candidate_unchanged() {
    let candidate = larger_aggregate_request();
    let (mut limits, authoritative) = exact_limits(&candidate);
    limits.host_retained_bytes -= 1;
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let incumbent = pool.try_acquire_request(request()).unwrap();

    let error = incumbent.try_replace(candidate).unwrap_err();
    assert_eq!(
        error.cause(),
        &PreWalCapacityError::ResourceExhausted("host retained bytes")
    );
    let (incumbent, candidate) = error.into_parts();
    assert_eq!(incumbent.footprint().host_plan_retained_bytes, 17);
    assert_eq!(candidate.footprint.host_plan_retained_bytes, 17 + 11);
    assert!(pool.try_acquire_request(request()).is_err());
    drop(incumbent);
    assert!(empty_used(&pool));

    let (limits, authoritative) = exact_limits(&candidate);
    let exact = PreWalCapacityPool::new(limits, authoritative).unwrap();
    drop(
        exact
            .try_acquire_request(candidate)
            .expect("returned candidate remains the exact move-only request"),
    );
}

#[test]
fn replacement_guard_abort_preserves_old_lease_and_releases_only_headroom() {
    let candidate = larger_aggregate_request();
    let (limits, authoritative) = exact_limits(&candidate);
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let incumbent = pool.try_acquire_request(request()).unwrap();
    let guard = incumbent
        .begin_replace(candidate)
        .expect("candidate headroom fits exactly");

    let (incumbent, candidate) = guard.abort();
    assert_eq!(incumbent.footprint().host_plan_retained_bytes, 17);
    assert_eq!(candidate.footprint.host_plan_retained_bytes, 28);
    drop(incumbent);
    assert!(empty_used(&pool));
    drop(
        pool.try_acquire_request(candidate)
            .expect("aborted candidate remains retryable after incumbent release"),
    );
}

#[test]
fn replacement_guard_unwind_releases_headroom_and_incumbent_exactly_once() {
    let candidate = larger_aggregate_request();
    let (limits, authoritative) = exact_limits(&candidate);
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let incumbent = pool.try_acquire_request(request()).unwrap();
    let guard = incumbent.begin_replace(candidate).unwrap();

    let unwound = catch_unwind(AssertUnwindSafe(move || {
        let _guard = guard;
        panic!("test unwind while replacement headroom is armed");
    }));
    assert!(unwound.is_err());
    assert!(empty_used(&pool));
    drop(
        pool.try_acquire_request(larger_aggregate_request())
            .expect("unwind released both headroom and incumbent demand"),
    );
}

#[test]
fn additive_leases_share_limits_without_charging_the_generation_envelope_twice() {
    let first = request();
    let (mut limits, authoritative) = exact_limits(&first);
    let envelope = limits.gpus[0].generation_envelope_bytes;
    double_additive_limits(&mut limits);
    unbound_global_limits(&mut limits);
    limits.gpus[0].hard_device_bytes =
        authoritative[0].bytes + exact_pool(&first).incremental_device_peak_bytes().unwrap() * 2;
    assert_eq!(limits.gpus[0].generation_envelope_bytes, envelope);
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let first = pool.try_acquire_request(first).unwrap();
    let second = pool.try_acquire(current()).unwrap();
    assert_eq!(
        lock_recover(&pool.state).gpus[0]
            .used
            .incremental_device_peak_bytes,
        280
    );
    let rejected = pool.try_acquire(current()).unwrap_err();
    assert_eq!(
        rejected.cause(),
        &PreWalCapacityError::ResourceExhausted("GPU hard device bytes")
    );
    assert_eq!(
        lock_recover(&pool.state).gpus[0]
            .used
            .incremental_device_peak_bytes,
        280
    );
    drop(first);
    drop(second);
    assert!(empty_used(&pool));
}

#[test]
fn concurrent_exact_budget_allows_one_lease_and_drop_releases_it() {
    let initial = request();
    let (limits, authoritative) = exact_limits(&initial);
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let start = Arc::new(Barrier::new(3));
    let (sender, receiver) = mpsc::channel();
    let mut threads = Vec::new();
    for _ in 0..2 {
        let pool = pool.clone();
        let start = Arc::clone(&start);
        let sender = sender.clone();
        threads.push(std::thread::spawn(move || {
            start.wait();
            sender.send(pool.try_acquire(current())).unwrap();
        }));
    }
    drop(sender);
    start.wait();
    let outcomes: Vec<_> = receiver.into_iter().collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(outcomes.iter().filter(|outcome| outcome.is_ok()).count(), 1);
    let lease = outcomes
        .into_iter()
        .find_map(Result::ok)
        .expect("exact budget admits one concurrent request");
    assert_eq!(lease.footprint().wal_frame_slots, 3);
    drop(lease);
    assert!(empty_used(&pool));
}

#[test]
fn panic_unwind_drops_the_only_lease_and_restores_exact_capacity() {
    let initial = request();
    let (limits, authoritative) = exact_limits(&initial);
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let unwind = catch_unwind(AssertUnwindSafe(|| {
        let _lease = pool.try_acquire(current()).unwrap();
        panic!("test unwind after admission");
    }));
    assert!(unwind.is_err());
    assert!(empty_used(&pool));
    drop(pool.try_acquire(current()).unwrap());
    assert!(empty_used(&pool));
}

#[test]
fn fixed_authoritative_residency_is_validated_and_never_recharged_as_demand() {
    let initial = request();
    let (mut limits, authoritative) = exact_limits(&initial);
    limits.gpus[0].hard_device_bytes = authoritative[0].bytes;
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let error = pool.try_acquire_request(initial).unwrap_err();
    assert_eq!(
        error.cause(),
        &PreWalCapacityError::ResourceExhausted("GPU hard device bytes")
    );
    assert!(empty_used(&pool));

    let rejected = request();
    let (limits, mut authoritative) = exact_limits(&rejected);
    authoritative[0].bytes = limits.gpus[0].hard_device_bytes + 1;
    assert!(matches!(
        PreWalCapacityPool::new(limits, authoritative),
        Err(PreWalCapacityError::AuthoritativeResidencyExceedsHard { gpu_id: 0, .. })
    ));
}

#[test]
fn configuration_and_unknown_gpu_sabotage_are_rejected_before_mutation() {
    let seed = request();
    let (mut limits, _) = exact_limits(&seed);
    limits.gpus = Box::default();
    assert!(matches!(
        PreWalCapacityPool::new(limits, Box::default()),
        Err(PreWalCapacityError::EmptyGpuLimits)
    ));

    let seed = request();
    let (mut limits, authoritative) = exact_limits(&seed);
    let mut next = limits.gpus[0];
    next.gpu_id = 1;
    limits.gpus = vec![next, limits.gpus[0]].into();
    assert!(matches!(
        PreWalCapacityPool::new(limits, authoritative),
        Err(PreWalCapacityError::GpuLimitOrderDrift(0))
    ));

    let seed = request();
    let (mut limits, authoritative) = exact_limits(&seed);
    limits.gpus = vec![limits.gpus[0], limits.gpus[0]].into();
    assert!(matches!(
        PreWalCapacityPool::new(limits, authoritative),
        Err(PreWalCapacityError::DuplicateGpuLimit(0))
    ));

    let seed = request();
    let (limits, _) = exact_limits(&seed);
    assert!(matches!(
        PreWalCapacityPool::new(limits, Box::default()),
        Err(PreWalCapacityError::MissingAuthoritativeResidency(0))
    ));

    let seed = request();
    let (limits, authoritative) = exact_limits(&seed);
    let authoritative = vec![
        PreWalAuthoritativeGpuResidency {
            gpu_id: 1,
            bytes: 0,
        },
        authoritative[0].clone(),
    ]
    .into();
    assert!(matches!(
        PreWalCapacityPool::new(limits, authoritative),
        Err(PreWalCapacityError::AuthoritativeResidencyOrderDrift(0))
    ));

    let seed = request();
    let (limits, authoritative) = exact_limits(&seed);
    let authoritative = vec![authoritative[0].clone(), authoritative[0].clone()].into();
    assert!(matches!(
        PreWalCapacityPool::new(limits, authoritative),
        Err(PreWalCapacityError::DuplicateAuthoritativeResidency(0))
    ));

    let seed = request();
    let (limits, authoritative) = exact_limits(&seed);
    let authoritative = vec![
        authoritative[0].clone(),
        PreWalAuthoritativeGpuResidency {
            gpu_id: 1,
            bytes: 0,
        },
    ]
    .into();
    assert!(matches!(
        PreWalCapacityPool::new(limits, authoritative),
        Err(PreWalCapacityError::UnexpectedAuthoritativeResidency(1))
    ));

    let unknown = request();
    let (limits, authoritative) = exact_limits(&unknown);
    let pool = PreWalCapacityPool::new(limits, authoritative).unwrap();
    let mut unknown = unknown;
    unknown.footprint.gpus[0].target.gpu_id = 1;
    let error = pool.try_acquire_request(unknown).unwrap_err();
    assert_eq!(error.cause(), &PreWalCapacityError::UnknownGpuDemand(1));
    assert!(empty_used(&pool));
}

#[test]
fn pool_leaf_has_no_live_engine_or_publication_authority() {
    let source = include_str!("pre_wal_capacity.rs")
        .split("\n#[cfg(test)]\n#[path")
        .next()
        .expect("production leaf precedes tests");
    for forbidden in [
        "Engine",
        "WalBuffer",
        "BTreeMap",
        "BTreeSet",
        "clone_used",
        "PreWalGpuDemand",
        "pub(super) fn release",
        "pub fn release",
        "resize",
        "finalize",
        "rollback",
        "transfer",
        "forget",
    ] {
        assert!(
            !source.contains(forbidden),
            "capacity leaf must not gain live authority: {forbidden}"
        );
    }
}
