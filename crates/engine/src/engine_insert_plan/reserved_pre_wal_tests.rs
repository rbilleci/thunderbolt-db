use super::*;
use crate::engine_insert_plan::pre_wal_capacity::{
    PreWalAuthoritativeGpuResidency, PreWalCapacityLimits, PreWalGpuCapacityLimit,
};
use crate::engine_insert_plan::pre_wal_footprint::{
    AggregateTypedInsertShape, GpuGenerationWitness, GpuTargetKey, IndexedPreWalPhysicalForecast,
    LogicalStatementContribution, PreWalLogicalFootprintInput, PreWalOverlayLink,
    PreWalPublicationGeometry, PreWalStatementIdentity, PreWalTransactionOwnerIdentity,
    RequiresAggregateFormatDecision,
};
use crate::typed_insert_aggregate::{
    typed_insert_aggregate_status_roots, TypedInsertAggregateSectionView, TypedInsertAggregateView,
    TypedInsertStatusV2, AGGREGATE_FLAG_AUTOCOMMIT, AGGREGATE_FLAG_EXPLICIT,
    AGGREGATE_SECTION_COUNT, OUTER_CONTENT_ROW, OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier};

const TXN_ID: u64 = 41;
const REQUEST_DIGEST: [u8; 32] = [4; 32];
const SECTION_COUNTS: [u32; AGGREGATE_SECTION_COUNT] = [1, 1, 0, 1, 0, 1, 1, 0];

fn section_payloads() -> [Vec<u8>; AGGREGATE_SECTION_COUNT] {
    std::array::from_fn(|index| vec![index as u8 + 1; index * 7 + 1])
}

fn autocommit_view<'a>(
    payloads: &'a [Vec<u8>; AGGREGATE_SECTION_COUNT],
) -> TypedInsertAggregateView<'a> {
    TypedInsertAggregateView {
        flags: AGGREGATE_FLAG_AUTOCOMMIT,
        outer_flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
        stable_transaction_id: TXN_ID,
        statement_count: 1,
        insert_statement_count: 1,
        original_inserted_row_count: 1,
        final_row_transition_count: 1,
        allocator_before: 99,
        allocator_high_water: 100,
        table_block_count: 1,
        sections: std::array::from_fn(|index| TypedInsertAggregateSectionView {
            entry_count: SECTION_COUNTS[index],
            payload: &payloads[index],
        }),
    }
}

fn explicit_view<'a>(
    payloads: &'a [Vec<u8>; AGGREGATE_SECTION_COUNT],
) -> TypedInsertAggregateView<'a> {
    let mut view = autocommit_view(payloads);
    view.flags = AGGREGATE_FLAG_EXPLICIT;
    view.statement_count = 2;
    view.insert_statement_count = 2;
    view.original_inserted_row_count = 2;
    view.final_row_transition_count = 2;
    view.allocator_high_water = 101;
    view.sections[0].entry_count = 2;
    view.sections[1].entry_count = 2;
    view.sections[3].entry_count = 2;
    view.sections[5].entry_count = 2;
    view
}

fn forecast() -> IndexedPreWalPhysicalForecast {
    IndexedPreWalPhysicalForecast {
        target: GpuTargetKey {
            gpu_id: 0,
            table_oid: 7,
            schema_digest: [7; 32],
        },
        generation: GpuGenerationWitness {
            catalog_seq: 7,
            predecessor_boundary: 11,
            open_shard_id: 3,
            row_start: 13,
            row_count: 17,
            capacity: 23,
            index_mutation_epoch_even: 28,
        },
        final_host_retained_bytes: 101,
        final_host_allocation_slots: 11,
        final_host_generation_pin_slots: 7,
        peak_host_retained_bytes: 131,
        peak_host_allocation_slots: 13,
        peak_host_generation_pin_slots: 9,
        old_generation_pinned_bytes: 151,
        new_persistent_bytes: 181,
        retained_device_transient_bytes: 191,
        retained_device_result_bytes: 0,
        incremental_allocation_slots: 17,
        generation_pin_slots: 19,
        maximum_concurrent_device_scratch_bytes: 211,
        maximum_host_readback_bytes: 23,
    }
}

fn fused_forecast() -> IndexedPreWalPhysicalForecast {
    let mut forecast = forecast();
    // One final fused owner array, staging image, and uniform stamp box remain retained across
    // WAL; its pooled device lease is simultaneous with the prepared index lease.
    forecast.final_host_retained_bytes += 128;
    forecast.final_host_allocation_slots += 3;
    forecast.peak_host_retained_bytes += 192;
    forecast.peak_host_allocation_slots += 5;
    forecast.retained_device_transient_bytes += 256;
    forecast.maximum_concurrent_device_scratch_bytes += 256;
    forecast.incremental_allocation_slots += 1;
    forecast.maximum_host_readback_bytes = forecast.maximum_host_readback_bytes.max(4);
    forecast
}

fn logical(
    ordinal: u32,
    before_generation: u64,
    before_root: u8,
    after_root: u8,
) -> PreWalLogicalFootprintInput {
    PreWalLogicalFootprintInput {
        host_plan_retained_bytes: 29,
        host_allocation_slots: 3,
        typed_shadow_retained_bytes: 31,
        host_generation_pin_slots: 5,
        host_scratch_peak_bytes: 37,
        statement_identity: PreWalStatementIdentity {
            owner: PreWalTransactionOwnerIdentity {
                txn_id: TXN_ID,
                registration_nonce: 43,
            },
            overlay_link: PreWalOverlayLink {
                before_generation,
                after_generation: before_generation + 1,
                before_root_digest: [before_root; 32],
                after_root_digest: [after_root; 32],
            },
            statement_ordinal: ordinal,
        },
        row_id_bytes: 8,
        row_id_slots: 1,
        sequence_effect_bytes: 0,
        sequence_effect_slots: 0,
        terminal_response_bytes: 0,
        terminal_response_slots: 0,
    }
}

fn autocommit_shape(view: &TypedInsertAggregateView<'_>) -> AggregateTypedInsertShape {
    LogicalStatementContribution::new(logical(0, 10, 0x31, 0x32), 97)
        .unwrap()
        .bind_single_indexed_final_overlay(
            forecast(),
            PreWalPublicationGeometry {
                bytes: 47,
                slots: 1,
            },
        )
        .unwrap()
        .bind_typed_insert_aggregate_layout(view.measure().unwrap())
        .unwrap()
}

fn fused_autocommit_shape(view: &TypedInsertAggregateView<'_>) -> AggregateTypedInsertShape {
    LogicalStatementContribution::new(logical(0, 10, 0x31, 0x32), 97)
        .unwrap()
        .bind_single_indexed_final_overlay(
            fused_forecast(),
            PreWalPublicationGeometry {
                bytes: 47,
                slots: 1,
            },
        )
        .unwrap()
        .bind_typed_insert_aggregate_layout(view.measure().unwrap())
        .unwrap()
}

fn explicit_shape(view: &TypedInsertAggregateView<'_>) -> AggregateTypedInsertShape {
    RequiresAggregateFormatDecision::from_two(
        LogicalStatementContribution::new(logical(0, 10, 0x31, 0x32), 97).unwrap(),
        LogicalStatementContribution::new(logical(1, 11, 0x32, 0x33), 101).unwrap(),
    )
    .unwrap()
    .bind_indexed_final_overlay(
        forecast(),
        PreWalPublicationGeometry {
            bytes: 47,
            slots: 1,
        },
    )
    .unwrap()
    .bind_typed_insert_aggregate_layout(view.measure().unwrap())
    .unwrap()
}

fn status(view: &TypedInsertAggregateView<'_>) -> TypedInsertStatusV2 {
    let layout = view.measure().unwrap();
    let roots = typed_insert_aggregate_status_roots(view, &layout).unwrap();
    TypedInsertStatusV2 {
        database_id: [1; 16],
        timeline_id: [3; 16],
        txn_id: TXN_ID,
        request_digest: REQUEST_DIGEST,
        isolation: 1,
        flags: 0,
        retention_deadline: 0,
        statement_count: view.statement_count,
        response_artifact_count: 0,
        statement_outcome_root: roots.statement_outcome_root,
        response_root: roots.response_root,
        aggregate_root: roots.aggregate_root,
    }
}

fn physical_range() -> gpu_db_wal::CanonicalPhysicalRange {
    gpu_db_wal::CanonicalPhysicalRange {
        log_epoch: 11,
        lane_id: 0,
        segment_id: 17,
        first_frame_ordinal: 23,
    }
}

fn header(view: &TypedInsertAggregateView<'_>) -> gpu_db_wal::CanonicalPreApplyHeader {
    let layout = view.measure().unwrap();
    gpu_db_wal::CanonicalPreApplyHeader {
        identity: gpu_db_wal::CanonicalIdentity {
            database_id: [1; 16],
            cluster_id: [2; 16],
            timeline_id: [3; 16],
            format_epoch: 1,
        },
        leader_epoch: 11,
        commit_seq: 37,
        stable_transaction_id: TXN_ID,
        request_digest: REQUEST_DIGEST,
        isolation: gpu_db_wal::CanonicalIsolation::ReadCommitted,
        flags: view.outer_flags,
        catalog_before_epoch: 5,
        catalog_after_epoch: 5,
        catalog_before_digest: [6; 32],
        catalog_after_digest: [6; 32],
        operation_count: layout.fragment_count,
        table_block_count: view.table_block_count,
        allocator_high_water: view.allocator_high_water,
    }
}

fn outcome(view: &TypedInsertAggregateView<'_>) -> gpu_db_wal::CanonicalOutcome {
    let roots = typed_insert_aggregate_status_roots(view, &view.measure().unwrap()).unwrap();
    gpu_db_wal::CanonicalOutcome {
        kind: gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
        affected_rows: if view.flags & AGGREGATE_FLAG_AUTOCOMMIT != 0 {
            view.original_inserted_row_count
        } else {
            view.final_row_transition_count
        },
        sqlstate: None,
        constraint_id: 0,
        target_digest: roots.aggregate_root,
        returning_digest: roots.response_root,
    }
}

fn exact_pool(shape: &AggregateTypedInsertShape) -> PreWalCapacityPool {
    pool_with_host_retained_delta(shape, 0)
}

fn pool_with_host_retained_delta(
    shape: &AggregateTypedInsertShape,
    host_retained_delta: i64,
) -> PreWalCapacityPool {
    let footprint = shape.footprint();
    let mut pools = footprint.gpu_pool_cursor();
    let gpu = pools.try_next().unwrap().unwrap();
    assert!(pools.try_next().unwrap().is_none());
    let exact_host = footprint.global_host_retained_bytes().unwrap();
    let host_retained_bytes = if host_retained_delta < 0 {
        exact_host - host_retained_delta.unsigned_abs()
    } else {
        exact_host + host_retained_delta as u64
    };
    let generation_envelope = gpu
        .old_generation_pinned_bytes
        .checked_add(gpu.new_persistent_bytes)
        .unwrap();
    PreWalCapacityPool::new(
        PreWalCapacityLimits {
            host_retained_bytes,
            host_allocation_slots: footprint.host_allocation_slots,
            host_scratch_peak_bytes: footprint.host_scratch_peak_bytes,
            host_generation_pin_slots: footprint.host_generation_pin_slots,
            wal_packed_record_bytes: footprint.wal_packed_record_bytes,
            wal_serialized_record_bytes: footprint.wal_serialized_record_bytes,
            wal_record_slots: footprint.wal_record_slots,
            wal_frame_slots: footprint.wal_frame_slots,
            row_id_slots: footprint.row_id_slots,
            sequence_effect_slots: footprint.sequence_effect_slots,
            status_index_bytes: footprint.status_index_bytes,
            status_index_slots: footprint.status_index_slots,
            terminal_response_bytes: footprint.terminal_response_bytes,
            terminal_response_slots: footprint.terminal_response_slots,
            completion_bytes: footprint.completion_bytes,
            completion_slots: footprint.completion_slots,
            publication_bytes: footprint.publication_bytes,
            publication_target_slots: footprint.publication_target_slots,
            publication_join_entry_bytes: footprint.publication_join_entry_bytes,
            publication_join_entry_slots: footprint.publication_join_entry_slots,
            gpus: vec![PreWalGpuCapacityLimit {
                gpu_id: gpu.gpu_id,
                hard_device_bytes: gpu.old_generation_pinned_bytes
                    + gpu.incremental_device_peak_bytes().unwrap(),
                plan_retained_transient_bytes: gpu.plan_retained_transient_bytes,
                result_retained_device_bytes: gpu.result_retained_device_bytes,
                scratch_peak_bytes: gpu.scratch_peak_bytes,
                allocation_slots: gpu.allocation_slots,
                generation_pin_slots: gpu.generation_pin_slots,
                generation_envelope_bytes: Some(generation_envelope),
            }]
            .into(),
        },
        vec![PreWalAuthoritativeGpuResidency {
            gpu_id: gpu.gpu_id,
            bytes: gpu.old_generation_pinned_bytes,
        }]
        .into(),
    )
    .unwrap()
}

struct PhysicalDropProbe {
    pool: PreWalCapacityPool,
    dropped: Arc<AtomicBool>,
}

impl Drop for PhysicalDropProbe {
    fn drop(&mut self) {
        assert!(
            !self.pool.is_empty_for_test(),
            "capacity lease released before private physical reservation"
        );
        self.dropped.store(true, Ordering::SeqCst);
    }
}

/// Keeps a test-only resource-drop window fail-safe: if an assertion unwinds while the worker is
/// parked in its Drop probe, this controller's Drop releases it instead of stranding the scoped
/// worker thread.
struct ResourceDropWindowRelease(Option<mpsc::Sender<()>>);

impl ResourceDropWindowRelease {
    fn release(&mut self) {
        if let Some(sender) = self.0.take() {
            sender
                .send(())
                .expect("resource-drop worker remains parked until explicitly released");
        }
    }
}

impl Drop for ResourceDropWindowRelease {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}

fn exact_budget_waiter_is_blocked_during_resource_drop_window(pool: PreWalCapacityPool) -> bool {
    let payloads = section_payloads();
    let view = autocommit_view(&payloads);
    match pool.try_acquire_aggregate(autocommit_shape(&view)) {
        Ok(admitted) => {
            drop(admitted);
            true
        }
        Err(_) => false,
    }
}

#[test]
fn terminal_owner_retains_exact_envelope_and_drops_physical_before_capacity() {
    let payloads = section_payloads();
    let view = autocommit_view(&payloads);
    let shape = autocommit_shape(&view);
    let expected = shape.footprint();
    assert_eq!(
        expected.host_plan_retained_bytes,
        29 + forecast().final_host_retained_bytes + view.measure().unwrap().wal.fragment_body_bytes
    );
    assert_eq!(
        expected.host_scratch_peak_bytes,
        37.max(forecast().peak_host_retained_bytes - forecast().final_host_retained_bytes)
    );
    assert_eq!(
        expected.gpus[0].scratch_peak_bytes,
        forecast().maximum_concurrent_device_scratch_bytes
    );
    let expected_wal = (
        expected.wal_packed_record_bytes,
        expected.wal_serialized_record_bytes,
    );
    let expected_terminal = (
        expected.status_index_bytes,
        expected.publication_bytes,
        expected.publication_join_entry_bytes,
        expected.completion_bytes,
    );
    let pool = exact_pool(&shape);
    let dropped = Arc::new(AtomicBool::new(false));
    let plan = reserve_indexed_insert_pre_wal_plan(
        &pool,
        shape,
        &view,
        &status(&view),
        physical_range(),
        header(&view),
        outcome(&view),
        &forecast(),
        |_| {
            Ok(PhysicalDropProbe {
                pool: pool.clone(),
                dropped: Arc::clone(&dropped),
            })
        },
    )
    .unwrap();
    assert!(!pool.is_empty_for_test());
    let report = plan
        .inspect(|physical, report| {
            drop(physical);
            Ok(report)
        })
        .unwrap();
    assert!(dropped.load(Ordering::SeqCst));
    assert!(pool.is_empty_for_test());
    assert_eq!(
        (report.packed_wal_bytes, report.serialized_wal_bytes),
        expected_wal
    );
    assert_eq!(
        (
            report.status_index_bytes,
            report.publication_bytes,
            report.publication_join_bytes,
            report.completion_bytes,
        ),
        expected_terminal
    );
    assert_eq!(report.owner.txn_id, TXN_ID);
    assert_eq!(report.final_overlay.before_generation, 10);
    assert_eq!(report.final_overlay.after_generation, 11);
    assert_eq!(
        (
            report.first_statement_ordinal,
            report.last_statement_ordinal
        ),
        (0, 0)
    );
}

#[test]
fn every_post_lease_failure_releases_capacity_without_materializing_physical_work() {
    for fault in [
        ReservationFault::AfterLease,
        ReservationFault::AfterBodies,
        ReservationFault::AfterEnvelope,
        ReservationFault::BeforePhysical,
    ] {
        let payloads = section_payloads();
        let view = autocommit_view(&payloads);
        let shape = autocommit_shape(&view);
        let pool = exact_pool(&shape);
        let materialized = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&materialized);
        let result = reserve_indexed_insert_pre_wal_plan_inner(
            &pool,
            shape,
            &view,
            &status(&view),
            physical_range(),
            header(&view),
            outcome(&view),
            &forecast(),
            fault,
            move |_| {
                observed.store(true, Ordering::SeqCst);
                Ok(())
            },
        );
        assert!(matches!(
            result,
            Err(ReserveInsertPreWalPlanError::Identity(
                "injected inert pre-WAL reservation failure"
            ))
        ));
        assert!(!materialized.load(Ordering::SeqCst));
        assert!(pool.is_empty_for_test());
    }
}

#[test]
fn physical_failure_releases_every_reserved_domain_and_capacity_credit() {
    let payloads = section_payloads();
    let view = autocommit_view(&payloads);
    let shape = autocommit_shape(&view);
    let pool = exact_pool(&shape);
    let called = Arc::new(AtomicBool::new(false));
    let observed = Arc::clone(&called);
    let result = reserve_indexed_insert_pre_wal_plan(
        &pool,
        shape,
        &view,
        &status(&view),
        physical_range(),
        header(&view),
        outcome(&view),
        &forecast(),
        move |_| {
            observed.store(true, Ordering::SeqCst);
            Err::<(), _>(ExecuteError::Serialization(
                "injected private materialization failure".to_string(),
            ))
        },
    );
    assert!(matches!(
        result,
        Err(ReserveInsertPreWalPlanError::Physical(
            ExecuteError::Serialization(_)
        ))
    ));
    assert!(called.load(Ordering::SeqCst));
    assert!(pool.is_empty_for_test());
}

#[test]
fn materialize_err_drops_envelope_and_terminal_resources_before_capacity_release() {
    let payloads = section_payloads();
    let view = autocommit_view(&payloads);
    let shape = autocommit_shape(&view);
    let pool = exact_pool(&shape);
    let (resources_dropped_sender, resources_dropped) = mpsc::channel();
    let (outcome_sender, outcome_receiver) = mpsc::channel();

    std::thread::scope(|scope| {
        let (release_sender, release_receiver) = mpsc::channel();
        let mut release = ResourceDropWindowRelease(Some(release_sender));
        let worker_pool = pool.clone();
        scope.spawn(move || {
            let payloads = section_payloads();
            let view = autocommit_view(&payloads);
            arm_resource_drop_window_probe(resources_dropped_sender, release_receiver);
            let result = reserve_indexed_insert_pre_wal_plan(
                &worker_pool,
                autocommit_shape(&view),
                &view,
                &status(&view),
                physical_range(),
                header(&view),
                outcome(&view),
                &forecast(),
                |_| {
                    Err::<(), _>(ExecuteError::Serialization(
                        "injected materialization error after exact resource reservation"
                            .to_string(),
                    ))
                },
            );
            outcome_sender
                .send((
                    matches!(
                        result,
                        Err(ReserveInsertPreWalPlanError::Physical(
                            ExecuteError::Serialization(_)
                        ))
                    ),
                    worker_pool.is_empty_for_test(),
                ))
                .unwrap();
        });

        // The probe is declared before the envelope and terminal locals, so this receive proves
        // they drained. Its worker remains parked before the still-owned aggregate lease drops.
        resources_dropped.recv().unwrap();
        let waiter_pool = pool.clone();
        let (waiter_sender, waiter) = mpsc::channel();
        scope.spawn(move || {
            waiter_sender
                .send(exact_budget_waiter_is_blocked_during_resource_drop_window(
                    waiter_pool,
                ))
                .unwrap();
        });
        assert!(
            !waiter.recv().unwrap(),
            "an exact-budget waiter acquired while resource drops were still fenced"
        );
        assert!(
            !pool.is_empty_for_test(),
            "the aggregate lease must remain held through the resource-drop window"
        );
        release.release();
        assert_eq!(outcome_receiver.recv().unwrap(), (true, true));
    });

    assert!(pool.is_empty_for_test());
}

#[test]
fn materialize_panic_drops_envelope_and_terminal_resources_before_capacity_release() {
    let payloads = section_payloads();
    let view = autocommit_view(&payloads);
    let shape = autocommit_shape(&view);
    let pool = exact_pool(&shape);
    let (resources_dropped_sender, resources_dropped) = mpsc::channel();
    let (outcome_sender, outcome_receiver) = mpsc::channel();

    std::thread::scope(|scope| {
        let (release_sender, release_receiver) = mpsc::channel();
        let mut release = ResourceDropWindowRelease(Some(release_sender));
        let worker_pool = pool.clone();
        scope.spawn(move || {
            let payloads = section_payloads();
            let view = autocommit_view(&payloads);
            arm_resource_drop_window_probe(resources_dropped_sender, release_receiver);
            let unwound = catch_unwind(AssertUnwindSafe(|| {
                let _ = reserve_indexed_insert_pre_wal_plan(
                    &worker_pool,
                    autocommit_shape(&view),
                    &view,
                    &status(&view),
                    physical_range(),
                    header(&view),
                    outcome(&view),
                    &forecast(),
                    |_| -> Result<(), ExecuteError> {
                        panic!("injected materialization panic after exact resource reservation")
                    },
                );
            }));
            outcome_sender
                .send((unwound.is_err(), worker_pool.is_empty_for_test()))
                .unwrap();
        });

        resources_dropped.recv().unwrap();
        let waiter_pool = pool.clone();
        let (waiter_sender, waiter) = mpsc::channel();
        scope.spawn(move || {
            waiter_sender
                .send(exact_budget_waiter_is_blocked_during_resource_drop_window(
                    waiter_pool,
                ))
                .unwrap();
        });
        assert!(
            !waiter.recv().unwrap(),
            "an exact-budget waiter acquired during panic cleanup before the lease could drop"
        );
        assert!(!pool.is_empty_for_test());
        release.release();
        assert_eq!(outcome_receiver.recv().unwrap(), (true, true));
    });

    assert!(pool.is_empty_for_test());
}

#[test]
fn one_byte_short_rejects_before_materialization_and_preserves_exact_retry_identity() {
    let payloads = section_payloads();
    let view = autocommit_view(&payloads);
    let shape = autocommit_shape(&view);
    let short = pool_with_host_retained_delta(&shape, -1);
    let exact_retry = exact_pool(&shape);
    let called = Arc::new(AtomicBool::new(false));
    let observed = Arc::clone(&called);
    let result = reserve_indexed_insert_pre_wal_plan(
        &short,
        shape,
        &view,
        &status(&view),
        physical_range(),
        header(&view),
        outcome(&view),
        &forecast(),
        move |_| {
            observed.store(true, Ordering::SeqCst);
            Ok(())
        },
    );
    let error = match result {
        Err(ReserveInsertPreWalPlanError::Capacity(error)) => error,
        Err(other) => panic!("unexpected one-byte-short failure: {other}"),
        Ok(_) => panic!("one-byte-short capacity must reject at admission"),
    };
    assert!(matches!(
        error.cause(),
        super::super::pre_wal_capacity::PreWalCapacityError::ResourceExhausted(
            "host retained bytes"
        )
    ));
    assert!(!called.load(Ordering::SeqCst));
    assert!(short.is_empty_for_test());
    let admitted = error
        .retry(&exact_retry)
        .expect("exact rejected aggregate retries without rebuilding");
    assert_eq!(admitted.binding().owner().txn_id, TXN_ID);
    assert_eq!(admitted.binding().indexed_physical(), Some(&forecast()));
    drop(admitted);
    assert!(exact_retry.is_empty_for_test());
}

#[test]
fn fused_pre_wal_binding_is_admitted_as_one_exact_outer_capacity_demand() {
    let payloads = section_payloads();
    let view = autocommit_view(&payloads);
    let physical = fused_forecast();
    let shape = fused_autocommit_shape(&view);
    let footprint = shape.footprint();
    assert_eq!(
        footprint.gpus[0].plan_retained_transient_bytes,
        physical.retained_device_transient_bytes
    );
    assert_eq!(
        footprint.gpus[0].scratch_peak_bytes,
        physical.maximum_concurrent_device_scratch_bytes
    );
    assert_eq!(
        footprint.gpus[0].allocation_slots,
        physical.incremental_allocation_slots
    );
    let short = pool_with_host_retained_delta(&shape, -1);
    let materialized = Arc::new(AtomicBool::new(false));
    let observed = Arc::clone(&materialized);
    let result = reserve_indexed_insert_pre_wal_plan(
        &short,
        shape,
        &view,
        &status(&view),
        physical_range(),
        header(&view),
        outcome(&view),
        &physical,
        move |_| {
            observed.store(true, Ordering::SeqCst);
            Ok(())
        },
    );
    assert!(matches!(
        result,
        Err(ReserveInsertPreWalPlanError::Capacity(_))
    ));
    assert!(
        !materialized.load(Ordering::SeqCst),
        "one-short fused aggregate capacity must reject before materialization"
    );
    assert!(short.is_empty_for_test());
}

#[test]
fn explicit_statements_aggregate_one_overlay_one_physical_forecast_and_one_lease() {
    let payloads = section_payloads();
    let view = explicit_view(&payloads);
    let shape = explicit_shape(&view);
    let footprint = shape.footprint();
    assert_eq!(footprint.gpus.len(), 1);
    assert_eq!(footprint.publication_target_slots, 1);
    assert_eq!(footprint.row_id_slots, 2);
    // The final physical owner is charged once; logical statement bytes remain additive.
    assert_eq!(
        footprint.host_plan_retained_bytes,
        29 * 2
            + forecast().final_host_retained_bytes
            + view.measure().unwrap().wal.fragment_body_bytes
    );
    let pool = exact_pool(&shape);
    let plan = reserve_indexed_insert_pre_wal_plan(
        &pool,
        shape,
        &view,
        &status(&view),
        physical_range(),
        header(&view),
        outcome(&view),
        &forecast(),
        |_| Ok(()),
    )
    .unwrap();
    let report = plan.inspect(|(), report| Ok(report)).unwrap();
    assert_eq!(
        (
            report.first_statement_ordinal,
            report.last_statement_ordinal
        ),
        (0, 1)
    );
    assert_eq!(report.final_overlay.before_generation, 10);
    assert_eq!(report.final_overlay.after_generation, 12);
    assert_eq!(report.final_overlay.before_root_digest, [0x31; 32]);
    assert_eq!(report.final_overlay.after_root_digest, [0x33; 32]);
    assert!(pool.is_empty_for_test());
}

#[test]
fn concurrent_terminal_admission_has_one_materializer_and_exact_loser_retry() {
    let payloads = section_payloads();
    let view = autocommit_view(&payloads);
    let shape = autocommit_shape(&view);
    let pool = exact_pool(&shape);
    let admitted = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let materializations = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|scope| {
        let first_pool = pool.clone();
        let first_admitted = Arc::clone(&admitted);
        let first_release = Arc::clone(&release);
        let first_materializations = Arc::clone(&materializations);
        scope.spawn(move || {
            let payloads = section_payloads();
            let view = autocommit_view(&payloads);
            let plan = reserve_indexed_insert_pre_wal_plan(
                &first_pool,
                autocommit_shape(&view),
                &view,
                &status(&view),
                physical_range(),
                header(&view),
                outcome(&view),
                &forecast(),
                move |_| {
                    first_materializations.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            )
            .unwrap();
            first_admitted.wait();
            first_release.wait();
            drop(plan);
        });

        admitted.wait();
        let loser_materialized = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&loser_materialized);
        let result = reserve_indexed_insert_pre_wal_plan(
            &pool,
            autocommit_shape(&view),
            &view,
            &status(&view),
            physical_range(),
            header(&view),
            outcome(&view),
            &forecast(),
            move |_| {
                observed.store(true, Ordering::SeqCst);
                Ok(())
            },
        );
        let error = match result {
            Err(ReserveInsertPreWalPlanError::Capacity(error)) => error,
            Err(other) => panic!("unexpected concurrent loser failure: {other}"),
            Ok(_) => panic!("one-request pool must reject the concurrent loser"),
        };
        assert!(!loser_materialized.load(Ordering::SeqCst));
        release.wait();
        // Scoped thread has not necessarily returned yet; retry until its deterministic barrier
        // release has completed by joining at scope exit, then exercise exact retry below.
        let retry_pool = pool.clone();
        scope.spawn(move || {
            while !retry_pool.is_empty_for_test() {
                std::thread::yield_now();
            }
            let admitted = error.retry(&retry_pool).unwrap();
            drop(admitted);
        });
    });

    assert_eq!(materializations.load(Ordering::SeqCst), 1);
    assert!(pool.is_empty_for_test());
}

#[test]
fn identity_and_root_sabotage_fail_before_capacity_or_materialization() {
    for sabotage in 0..3 {
        let payloads = section_payloads();
        let view = autocommit_view(&payloads);
        let mut physical = forecast();
        let mut status = status(&view);
        let shape = autocommit_shape(&view);
        let pool = exact_pool(&shape);
        match sabotage {
            0 => physical.target.schema_digest[0] ^= 1,
            1 => physical.generation.index_mutation_epoch_even += 2,
            2 => status.aggregate_root[0] ^= 1,
            _ => unreachable!(),
        }
        let called = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&called);
        let result = reserve_indexed_insert_pre_wal_plan(
            &pool,
            shape,
            &view,
            &status,
            physical_range(),
            header(&view),
            outcome(&view),
            &physical,
            move |_| {
                observed.store(true, Ordering::SeqCst);
                Ok(())
            },
        );
        assert!(matches!(
            result,
            Err(ReserveInsertPreWalPlanError::Identity(_))
        ));
        assert!(!called.load(Ordering::SeqCst));
        assert!(pool.is_empty_for_test());
    }
}

#[test]
fn production_terminal_has_no_wal_apply_publication_or_fallback_exit() {
    let source = include_str!("reserved_pre_wal.rs")
        .split("\n#[cfg(test)]\nimpl<P>")
        .next()
        .expect("production terminal precedes test inspection");
    assert!(source.contains("struct ReservedInsertPreWalPlan<P>"));
    assert!(source.contains("physical: P"));
    assert!(source.contains("lease: PreWalCapacityLease"));
    assert!(source.contains("IndexedPhysicalMaterializationPermit"));
    for forbidden in [
        "append_canonical",
        "WalBuffer",
        "apply_resident",
        "publish_relational",
        "status_index.insert",
        "fallback(",
        "reprepare(",
        "pub(crate) fn into_parts",
        "serialized_record()",
    ] {
        assert!(
            !source.contains(forbidden),
            "inert terminal exposed forbidden authority: {forbidden}"
        );
    }
}
