use super::*;
const OPERATION_BODY_ONE: u64 = 121;
const OPERATION_BODY_TWO: u64 = 123;
const OPERATION_BODY_THREE: u64 = 127;

fn target(gpu_id: u16, table_oid: u32) -> GpuTargetKey {
    GpuTargetKey {
        gpu_id,
        table_oid,
        schema_digest: [u8::try_from(table_oid).unwrap(); 32],
    }
}

fn witness() -> GpuGenerationWitness {
    GpuGenerationWitness {
        catalog_seq: 7,
        predecessor_boundary: 11,
        open_shard_id: 3,
        row_start: 13,
        row_count: 17,
        capacity: 23,
        index_mutation_epoch_even: 28,
    }
}

fn gpu(gpu_id: u16, table_oid: u32) -> PreWalGpuFootprint {
    PreWalGpuFootprint {
        target: target(gpu_id, table_oid),
        generation: witness(),
        old_generation_pinned_bytes: 23,
        new_persistent_bytes: 29,
        plan_retained_transient_bytes: 31,
        result_retained_device_bytes: 0,
        allocation_slots: 5,
        generation_pin_slots: 7,
        scratch_peak_bytes: 37,
        max_host_readback_bytes: 0,
    }
}

fn logical_input() -> PreWalLogicalFootprintInput {
    logical_input_at(0, 9, 0x31, 0x32)
}

fn logical_input_at(
    statement_ordinal: u32,
    before_generation: u64,
    before_root: u8,
    after_root: u8,
) -> PreWalLogicalFootprintInput {
    PreWalLogicalFootprintInput {
        host_plan_retained_bytes: 17,
        host_allocation_slots: 3,
        typed_shadow_retained_bytes: 4_096,
        host_generation_pin_slots: 1,
        host_scratch_peak_bytes: 39,
        statement_identity: PreWalStatementIdentity {
            owner: PreWalTransactionOwnerIdentity {
                txn_id: 5,
                registration_nonce: 7,
            },
            overlay_link: PreWalOverlayLink {
                before_generation,
                after_generation: before_generation + 1,
                before_root_digest: [before_root; 32],
                after_root_digest: [after_root; 32],
            },
            statement_ordinal,
        },
        row_id_bytes: 16,
        row_id_slots: 2,
        sequence_effect_bytes: 24,
        sequence_effect_slots: 3,
        terminal_response_bytes: 0,
        terminal_response_slots: 0,
    }
}

fn input(gpu_id: u16, table_oid: u32) -> PreWalFootprintInput {
    PreWalFootprintInput {
        logical: logical_input(),
        final_physical: PreWalFinalPhysicalFootprintInput {
            gpus: vec![gpu(gpu_id, table_oid)].into(),
            publication_targets: vec![PreWalPublicationTarget {
                target: target(gpu_id, table_oid),
                bytes: 32,
                slots: 1,
            }]
            .into(),
        },
    }
}

fn two_statement_aggregate_layout() -> crate::typed_insert_aggregate::TypedInsertAggregateLayout {
    use crate::typed_insert_aggregate::{
        AggregateSectionMeasure, TypedInsertAggregateMeasure, AGGREGATE_FLAG_EXPLICIT,
        AGGREGATE_SECTION_COUNT, OUTER_CONTENT_ROW, OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
    };
    let mut sections = [AggregateSectionMeasure {
        entry_count: 0,
        payload_bytes: 0,
    }; AGGREGATE_SECTION_COUNT];
    sections[0] = AggregateSectionMeasure {
        entry_count: 2,
        payload_bytes: 2 * 112,
    };
    sections[1] = AggregateSectionMeasure {
        entry_count: 2,
        payload_bytes: 2 * 48,
    };
    sections[3] = AggregateSectionMeasure {
        entry_count: 4,
        payload_bytes: 4 * 64,
    };
    sections[5] = AggregateSectionMeasure {
        entry_count: 2,
        payload_bytes: 2 * 136,
    };
    sections[6] = AggregateSectionMeasure {
        entry_count: 1,
        payload_bytes: 160,
    };
    TypedInsertAggregateMeasure {
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
    .unwrap()
}

#[test]
fn current_shape_uses_encoder_owned_sizes_and_exact_embed_limit() {
    let current =
        CurrentTwoFragmentShape::new(input(0, 7), OPERATION_BODY_ONE).expect("current shape fits");
    let footprint = current.footprint();
    assert_eq!(
        footprint.wal_packed_record_bytes,
        OPERATION_BODY_ONE + 1_292
    );
    assert_eq!(
        footprint.wal_serialized_record_bytes,
        OPERATION_BODY_ONE + 1_316
    );
    assert_eq!(footprint.wal_record_slots, 1);
    assert_eq!(footprint.wal_frame_slots, 3);
    assert_eq!(current.operation_fragment_body_bytes(), OPERATION_BODY_ONE);
    assert_eq!(footprint.typed_shadow_retained_bytes, 4_096);
    assert_eq!(footprint.host_allocation_slots, 3);
    assert_eq!(footprint.host_scratch_peak_bytes, 39);
    assert_eq!(footprint.status_index_slots, 1);
    assert_eq!(
        footprint.status_index_bytes,
        crate::durable_transaction_status_index_entry_bytes() as u64
    );
    assert_eq!(footprint.publication_join_entry_slots, 1);
    assert_eq!(
        footprint.publication_join_entry_bytes,
        crate::engine_commit_coordinator::commit_publication_join_entry_payload_bytes() as u64
    );
    assert_eq!(footprint.completion_slots, 1);
    assert_eq!(
        footprint.completion_bytes,
        crate::engine_dml_concurrent::commit_wave_done_payload_bytes() as u64
    );
    assert_eq!(footprint.terminal_response_bytes, 0);
    assert_eq!(footprint.gpus[0].result_retained_device_bytes, 0);
    assert_eq!(footprint.gpus[0].max_host_readback_bytes, 0);
    assert!(CurrentTwoFragmentShape::new(input(0, 7), 0).is_err());
    assert!(CurrentTwoFragmentShape::new(
        input(0, 7),
        gpu_db_wal::canonical_fragment_body_limit() + 1
    )
    .is_err());
    assert!(
        CurrentTwoFragmentShape::new(input(0, 7), gpu_db_wal::canonical_fragment_body_limit())
            .is_ok()
    );
}

#[test]
fn aggregate_layout_resolves_exact_wal_capacity_only_after_final_overlay() {
    let first = LogicalStatementContribution::new(logical_input(), OPERATION_BODY_ONE).unwrap();
    let second =
        LogicalStatementContribution::new(logical_input_at(1, 10, 0x32, 0x33), OPERATION_BODY_TWO)
            .unwrap();
    let unresolved = RequiresAggregateFormatDecision::from_two(first, second)
        .unwrap()
        .bind_final_overlay(PreWalFinalPhysicalFootprintInput {
            gpus: vec![gpu(0, 7)].into(),
            publication_targets: vec![PreWalPublicationTarget {
                target: target(0, 7),
                bytes: 32,
                slots: 1,
            }]
            .into(),
        })
        .unwrap();
    let layout = two_statement_aggregate_layout();
    let expected_wal = layout.wal;
    let shape = unresolved
        .bind_typed_insert_aggregate_layout(layout)
        .expect("codec-5 layout resolves aggregate capacity");
    assert_eq!(shape.layout().chunk_count, 1);
    assert_eq!(shape.layout().fragment_count, 2);
    assert_eq!(
        shape.layout().live_fragment_body_bytes()[1],
        crate::typed_insert_aggregate::AGGREGATE_STATUS_V2_BYTES
    );
    assert_eq!(
        shape.footprint().wal_packed_record_bytes,
        expected_wal.packed_record_bytes
    );
    assert_eq!(
        shape.footprint().wal_serialized_record_bytes,
        expected_wal.serialized_record_bytes
    );
    assert_eq!(
        shape.footprint().wal_frame_slots,
        u64::from(expected_wal.frame_count)
    );
}

#[test]
fn aggregate_layout_identity_statement_and_allocator_sabotage_fail_closed() {
    fn unresolved() -> AggregateFinalOverlayFormatDecision {
        RequiresAggregateFormatDecision::from_two(
            LogicalStatementContribution::new(logical_input(), OPERATION_BODY_ONE).unwrap(),
            LogicalStatementContribution::new(
                logical_input_at(1, 10, 0x32, 0x33),
                OPERATION_BODY_TWO,
            )
            .unwrap(),
        )
        .unwrap()
        .bind_final_overlay(PreWalFinalPhysicalFootprintInput {
            gpus: vec![gpu(0, 7)].into(),
            publication_targets: vec![PreWalPublicationTarget {
                target: target(0, 7),
                bytes: 32,
                slots: 1,
            }]
            .into(),
        })
        .unwrap()
    }

    for sabotage in [
        |layout: &mut crate::typed_insert_aggregate::TypedInsertAggregateLayout| {
            layout.measure.stable_transaction_id += 1;
        },
        |layout: &mut crate::typed_insert_aggregate::TypedInsertAggregateLayout| {
            layout.measure.statement_count += 1;
        },
        |layout: &mut crate::typed_insert_aggregate::TypedInsertAggregateLayout| {
            layout.measure.insert_statement_count -= 1;
        },
        |layout: &mut crate::typed_insert_aggregate::TypedInsertAggregateLayout| {
            layout.measure.original_inserted_row_count += 1;
        },
        |layout: &mut crate::typed_insert_aggregate::TypedInsertAggregateLayout| {
            layout.measure.allocator_high_water += 1;
        },
    ] {
        let mut layout = two_statement_aggregate_layout();
        sabotage(&mut layout);
        assert!(matches!(
            unresolved().bind_typed_insert_aggregate_layout(layout),
            Err(PreWalFootprintError::AggregateLayoutDrift)
        ));
    }
}

#[test]
fn aggregate_merges_only_logical_statement_lifetimes_then_validates_one_final_overlay() {
    let first = LogicalStatementContribution::new(logical_input(), OPERATION_BODY_ONE).unwrap();
    let mut second_input = logical_input_at(1, 10, 0x32, 0x33);
    second_input.host_plan_retained_bytes = 43;
    second_input.host_allocation_slots = 5;
    second_input.typed_shadow_retained_bytes = 8_192;
    second_input.host_generation_pin_slots = 2;
    second_input.host_scratch_peak_bytes = 47;
    second_input.row_id_bytes = 88;
    second_input.row_id_slots = 11;
    second_input.sequence_effect_bytes = 104;
    second_input.sequence_effect_slots = 13;
    second_input.terminal_response_bytes = 59;
    second_input.terminal_response_slots = 1;
    let second = LogicalStatementContribution::new(second_input, OPERATION_BODY_TWO).unwrap();
    let aggregate = RequiresAggregateFormatDecision::from_two(first, second)
        .expect("aggregate remains a format decision");
    assert_eq!(aggregate.statement_count(), 2);

    let mut final_gpu = gpu(0, 7);
    final_gpu.new_persistent_bytes = 82;
    final_gpu.plan_retained_transient_bytes = 90;
    final_gpu.scratch_peak_bytes = 61;
    final_gpu.result_retained_device_bytes = 71;
    final_gpu.max_host_readback_bytes = 67;
    let bound = aggregate
        .bind_final_overlay(PreWalFinalPhysicalFootprintInput {
            gpus: vec![final_gpu].into(),
            publication_targets: vec![PreWalPublicationTarget {
                target: target(0, 7),
                bytes: 32,
                slots: 1,
            }]
            .into(),
        })
        .expect("one final overlay target is attached once");
    assert_eq!(bound.statement_count(), 2);
    assert_eq!(bound.logical.host_plan_retained_bytes, 60);
    assert_eq!(bound.logical.host_allocation_slots, 8);
    assert_eq!(bound.logical.typed_shadow_retained_bytes, 12_288);
    assert_eq!(bound.logical.host_generation_pin_slots, 3);
    assert_eq!(bound.logical.host_scratch_peak_bytes, 47);
    assert_eq!(bound.logical.row_id_bytes, 104);
    assert_eq!(bound.logical.row_id_slots, 13);
    assert_eq!(bound.logical.sequence_effect_bytes, 128);
    assert_eq!(bound.logical.sequence_effect_slots, 16);
    assert_eq!(bound.logical.terminal_response_bytes, 59);
    assert_eq!(bound.logical.terminal_response_slots, 1);
    assert_eq!(bound.final_physical.gpus.len(), 1);
    assert_eq!(
        bound.final_physical.gpus[0].result_retained_device_bytes,
        71
    );
    assert_eq!(bound.final_physical.gpus[0].max_host_readback_bytes, 67);
    assert_eq!(bound.final_physical.publication_targets.len(), 1);
    assert_eq!(bound.final_physical.publication_targets[0].bytes, 32);
}

#[test]
fn aggregate_extension_remains_format_unresolved_after_final_overlay_validation() {
    let first = LogicalStatementContribution::new(logical_input(), OPERATION_BODY_ONE).unwrap();
    let second =
        LogicalStatementContribution::new(logical_input_at(1, 10, 0x32, 0x33), OPERATION_BODY_TWO)
            .unwrap();
    let third = LogicalStatementContribution::new(
        logical_input_at(2, 11, 0x33, 0x34),
        OPERATION_BODY_THREE,
    )
    .unwrap();
    let aggregate = RequiresAggregateFormatDecision::from_two(first, second)
        .unwrap()
        .extend(third)
        .unwrap();
    assert_eq!(aggregate.statement_count(), 3);
    let bound = aggregate
        .bind_final_overlay(PreWalFinalPhysicalFootprintInput {
            gpus: vec![gpu(0, 7)].into(),
            publication_targets: vec![PreWalPublicationTarget {
                target: target(0, 7),
                bytes: 32,
                slots: 1,
            }]
            .into(),
        })
        .unwrap();
    assert_eq!(bound.statement_count(), 3);
    assert_eq!(bound.logical.row_id_slots, 6);
    assert_eq!(bound.final_physical.gpus.len(), 1);
}

#[test]
fn final_overlay_rejects_duplicate_targets_generation_drift_and_noncanonical_order() {
    let mut duplicate = input(0, 7);
    duplicate.final_physical.gpus = vec![gpu(0, 7), gpu(0, 7)].into();
    assert!(matches!(
        CurrentTwoFragmentShape::new(duplicate, 101),
        Err(PreWalFootprintError::DuplicateGpuTarget(_))
    ));

    let mut drift = input(0, 7);
    let mut changed = gpu(0, 7);
    changed.generation.row_count += 1;
    drift.final_physical.gpus = vec![gpu(0, 7), changed].into();
    assert!(matches!(
        CurrentTwoFragmentShape::new(drift, 101),
        Err(PreWalFootprintError::GpuGenerationWitnessDrift(_))
    ));

    let mut gpu_schema_drift = input(0, 7);
    let mut changed_schema_gpu = gpu(0, 7);
    changed_schema_gpu.target.schema_digest = [8; 32];
    gpu_schema_drift.final_physical.gpus = vec![gpu(0, 7), changed_schema_gpu.clone()].into();
    gpu_schema_drift.final_physical.publication_targets = vec![
        PreWalPublicationTarget {
            target: target(0, 7),
            bytes: 32,
            slots: 1,
        },
        PreWalPublicationTarget {
            target: changed_schema_gpu.target.clone(),
            bytes: 32,
            slots: 1,
        },
    ]
    .into();
    assert!(matches!(
        CurrentTwoFragmentShape::new(gpu_schema_drift, 101),
        Err(PreWalFootprintError::GpuTargetSchemaDrift(_))
    ));

    let mut unordered = input(0, 8);
    unordered.final_physical.gpus = vec![gpu(0, 8), gpu(0, 7)].into();
    assert!(matches!(
        CurrentTwoFragmentShape::new(unordered, 101),
        Err(PreWalFootprintError::GpuTargetOrderDrift(_))
    ));

    let mut missing_publication = input(0, 7);
    missing_publication.final_physical.publication_targets = Box::default();
    assert!(matches!(
        CurrentTwoFragmentShape::new(missing_publication, 101),
        Err(PreWalFootprintError::EmptyPublicationTargetSet)
    ));
    let mut missing_gpu = input(0, 7);
    missing_gpu.final_physical.gpus = Box::default();
    assert!(matches!(
        CurrentTwoFragmentShape::new(missing_gpu, 101),
        Err(PreWalFootprintError::EmptyGpuTargetSet)
    ));

    let mut absent_gpu_target = input(0, 7);
    absent_gpu_target.final_physical.publication_targets[0].target = target(0, 8);
    assert!(matches!(
        CurrentTwoFragmentShape::new(absent_gpu_target, 101),
        Err(PreWalFootprintError::GpuTargetMissingPublication(_))
            | Err(PreWalFootprintError::PublicationTargetMissingGpu(_))
    ));

    let mut invalid_witness = input(0, 7);
    invalid_witness.final_physical.gpus[0]
        .generation
        .index_mutation_epoch_even += 1;
    assert!(matches!(
        CurrentTwoFragmentShape::new(invalid_witness, 101),
        Err(PreWalFootprintError::InvalidGpuGenerationWitness(_))
    ));
    let mut invalid_target = input(0, 7);
    invalid_target.final_physical.gpus[0].target.table_oid = 0;
    assert!(matches!(
        CurrentTwoFragmentShape::new(invalid_target, 101),
        Err(PreWalFootprintError::InvalidGpuTarget(_))
    ));

    let mut row_count_over_capacity = input(0, 7);
    row_count_over_capacity.final_physical.gpus[0]
        .generation
        .row_count = 24;
    assert!(matches!(
        CurrentTwoFragmentShape::new(row_count_over_capacity, 101),
        Err(PreWalFootprintError::InvalidGpuGenerationWitness(_))
    ));
    let mut row_end_overflow = input(0, 7);
    row_end_overflow.final_physical.gpus[0].generation.row_start = u64::MAX;
    row_end_overflow.final_physical.gpus[0].generation.row_count = 1;
    row_end_overflow.final_physical.gpus[0].generation.capacity = 1;
    assert!(matches!(
        CurrentTwoFragmentShape::new(row_end_overflow, 101),
        Err(PreWalFootprintError::InvalidGpuGenerationWitness(_))
    ));

    let mut zero_publication = input(0, 7);
    zero_publication.final_physical.publication_targets[0].slots = 0;
    assert!(matches!(
        CurrentTwoFragmentShape::new(zero_publication, 101),
        Err(PreWalFootprintError::EmptyPublicationTarget(_))
    ));
    let mut duplicate_publication = input(0, 7);
    duplicate_publication.final_physical.publication_targets = vec![
        PreWalPublicationTarget {
            target: target(0, 7),
            bytes: 32,
            slots: 1,
        },
        PreWalPublicationTarget {
            target: target(0, 7),
            bytes: 32,
            slots: 1,
        },
    ]
    .into();
    assert!(matches!(
        CurrentTwoFragmentShape::new(duplicate_publication, 101),
        Err(PreWalFootprintError::DuplicatePublicationTarget(_))
    ));
    let mut publication_schema_drift = input(0, 7);
    let mut changed_schema_publication = target(0, 7);
    changed_schema_publication.schema_digest = [8; 32];
    publication_schema_drift.final_physical.publication_targets = vec![
        PreWalPublicationTarget {
            target: target(0, 7),
            bytes: 32,
            slots: 1,
        },
        PreWalPublicationTarget {
            target: changed_schema_publication,
            bytes: 32,
            slots: 1,
        },
    ]
    .into();
    assert!(matches!(
        CurrentTwoFragmentShape::new(publication_schema_drift, 101),
        Err(PreWalFootprintError::PublicationTargetSchemaDrift(_))
    ));
    let mut unordered_publication = input(0, 7);
    unordered_publication.final_physical.gpus = vec![gpu(0, 7), gpu(0, 8)].into();
    unordered_publication.final_physical.publication_targets = vec![
        PreWalPublicationTarget {
            target: target(0, 8),
            bytes: 32,
            slots: 1,
        },
        PreWalPublicationTarget {
            target: target(0, 7),
            bytes: 32,
            slots: 1,
        },
    ]
    .into();
    assert!(matches!(
        CurrentTwoFragmentShape::new(unordered_publication, 101),
        Err(PreWalFootprintError::PublicationTargetOrderDrift(_))
    ));
}

#[test]
fn footprint_checked_arithmetic_rejects_every_owned_aggregate_overflow() {
    let mut host = logical_input();
    host.host_plan_retained_bytes = u64::MAX;
    let host = LogicalStatementContribution::new(host, 101).unwrap();
    assert!(matches!(
        RequiresAggregateFormatDecision::from_two(
            host,
            LogicalStatementContribution::new(logical_input_at(1, 10, 0x32, 0x33), 103).unwrap()
        ),
        Err(PreWalFootprintError::Overflow("host-plan retained bytes"))
    ));

    let mut typed_shadow = logical_input();
    typed_shadow.typed_shadow_retained_bytes = u64::MAX;
    let typed_shadow = LogicalStatementContribution::new(typed_shadow, 101).unwrap();
    assert!(matches!(
        RequiresAggregateFormatDecision::from_two(
            typed_shadow,
            LogicalStatementContribution::new(logical_input_at(1, 10, 0x32, 0x33), 103).unwrap()
        ),
        Err(PreWalFootprintError::Overflow(
            "typed shadow retained bytes"
        ))
    ));

    let mut host_allocation_slots = logical_input();
    host_allocation_slots.host_allocation_slots = u64::MAX;
    let host_allocation_slots =
        LogicalStatementContribution::new(host_allocation_slots, 101).unwrap();
    assert!(matches!(
        RequiresAggregateFormatDecision::from_two(
            host_allocation_slots,
            LogicalStatementContribution::new(logical_input_at(1, 10, 0x32, 0x33), 103).unwrap(),
        ),
        Err(PreWalFootprintError::Overflow("host allocation slots"))
    ));

    let mut generation_pins = logical_input();
    generation_pins.host_generation_pin_slots = u64::MAX;
    let generation_pins = LogicalStatementContribution::new(generation_pins, 101).unwrap();
    assert!(matches!(
        RequiresAggregateFormatDecision::from_two(
            generation_pins,
            LogicalStatementContribution::new(logical_input_at(1, 10, 0x32, 0x33), 103).unwrap()
        ),
        Err(PreWalFootprintError::Overflow("host generation-pin slots"))
    ));

    let mut rows = logical_input();
    rows.row_id_slots = u64::MAX / 8;
    rows.row_id_bytes = rows.row_id_slots * 8;
    let rows = LogicalStatementContribution::new(rows, 101).unwrap();
    assert!(matches!(
        RequiresAggregateFormatDecision::from_two(
            rows,
            LogicalStatementContribution::new(logical_input_at(1, 10, 0x32, 0x33), 103).unwrap()
        ),
        Err(PreWalFootprintError::Overflow("row-id bytes"))
    ));

    let mut sequences = logical_input();
    sequences.sequence_effect_bytes = u64::MAX;
    sequences.sequence_effect_slots = u64::MAX;
    let sequences = LogicalStatementContribution::new(sequences, 101).unwrap();
    assert!(matches!(
        RequiresAggregateFormatDecision::from_two(
            sequences,
            LogicalStatementContribution::new(logical_input_at(1, 10, 0x32, 0x33), 103).unwrap()
        ),
        Err(PreWalFootprintError::Overflow("sequence-effect bytes"))
    ));

    let mut sequence_slots = logical_input();
    sequence_slots.sequence_effect_bytes = 1;
    sequence_slots.sequence_effect_slots = u64::MAX;
    let sequence_slots = LogicalStatementContribution::new(sequence_slots, 101).unwrap();
    assert!(matches!(
        RequiresAggregateFormatDecision::from_two(
            sequence_slots,
            LogicalStatementContribution::new(logical_input_at(1, 10, 0x32, 0x33), 103).unwrap()
        ),
        Err(PreWalFootprintError::Overflow("sequence-effect slots"))
    ));

    let mut publication = input(0, 7);
    publication.final_physical.gpus = vec![gpu(0, 7), gpu(0, 8)].into();
    publication.final_physical.publication_targets = vec![
        PreWalPublicationTarget {
            target: target(0, 7),
            bytes: 32,
            slots: u64::MAX,
        },
        PreWalPublicationTarget {
            target: target(0, 8),
            bytes: 32,
            slots: 1,
        },
    ]
    .into();
    assert!(matches!(
        CurrentTwoFragmentShape::new(publication, 101),
        Err(PreWalFootprintError::Overflow("publication slots"))
    ));

    let mut publication_bytes_overflow = input(0, 7);
    publication_bytes_overflow.final_physical.gpus = vec![gpu(0, 7), gpu(0, 8)].into();
    publication_bytes_overflow
        .final_physical
        .publication_targets = vec![
        PreWalPublicationTarget {
            target: target(0, 7),
            bytes: u64::MAX,
            slots: 1,
        },
        PreWalPublicationTarget {
            target: target(0, 8),
            bytes: 1,
            slots: 1,
        },
    ]
    .into();
    assert!(matches!(
        CurrentTwoFragmentShape::new(publication_bytes_overflow, 101),
        Err(PreWalFootprintError::Overflow("publication bytes"))
    ));

    let mut target_peak = input(0, 7);
    target_peak.final_physical.gpus[0].old_generation_pinned_bytes = u64::MAX;
    target_peak.final_physical.gpus[0].new_persistent_bytes = 1;
    assert!(matches!(
        CurrentTwoFragmentShape::new(target_peak, 101),
        Err(PreWalFootprintError::Overflow(
            "GPU old and new generation bytes"
        ))
    ));

    let mut gpu_total = input(0, 7);
    let mut second = gpu(0, 8);
    gpu_total.final_physical.gpus[0].old_generation_pinned_bytes = u64::MAX;
    gpu_total.final_physical.gpus[0].new_persistent_bytes = 0;
    gpu_total.final_physical.gpus[0].plan_retained_transient_bytes = 0;
    gpu_total.final_physical.gpus[0].scratch_peak_bytes = 0;
    second.old_generation_pinned_bytes = 1;
    second.new_persistent_bytes = 0;
    second.plan_retained_transient_bytes = 0;
    gpu_total.final_physical.gpus = vec![gpu_total.final_physical.gpus[0].clone(), second].into();
    gpu_total.final_physical.publication_targets = vec![
        PreWalPublicationTarget {
            target: target(0, 7),
            bytes: 32,
            slots: 1,
        },
        PreWalPublicationTarget {
            target: target(0, 8),
            bytes: 32,
            slots: 1,
        },
    ]
    .into();
    assert!(matches!(
        CurrentTwoFragmentShape::new(gpu_total, 101),
        Err(PreWalFootprintError::Overflow(
            "GPU pool old-generation bytes"
        ))
    ));
}

#[test]
fn owner_link_order_and_byte_slot_domains_are_required_before_aggregation() {
    let first = LogicalStatementContribution::new(logical_input(), 101).unwrap();
    let mut changed = logical_input_at(1, 10, 0x32, 0x33);
    changed.statement_identity.owner.registration_nonce += 1;
    let changed = LogicalStatementContribution::new(changed, 103).unwrap();
    assert!(matches!(
        RequiresAggregateFormatDecision::from_two(first, changed),
        Err(PreWalFootprintError::OwnerIdentityDrift)
    ));

    let first = LogicalStatementContribution::new(logical_input(), 101).unwrap();
    let changed =
        LogicalStatementContribution::new(logical_input_at(1, 10, 0x44, 0x45), 103).unwrap();
    assert!(matches!(
        RequiresAggregateFormatDecision::from_two(first, changed),
        Err(PreWalFootprintError::OverlayLinkDrift)
    ));

    let first = LogicalStatementContribution::new(logical_input(), 101).unwrap();
    let changed =
        LogicalStatementContribution::new(logical_input_at(2, 10, 0x32, 0x33), 103).unwrap();
    assert!(matches!(
        RequiresAggregateFormatDecision::from_two(first, changed),
        Err(PreWalFootprintError::StatementOrderDrift)
    ));

    let mut result = logical_input();
    result.terminal_response_bytes = 1;
    assert!(matches!(
        LogicalStatementContribution::new(result, 101),
        Err(PreWalFootprintError::ByteSlotDomainMismatch(
            "terminal response"
        ))
    ));
}

#[test]
fn statement_identity_rejects_zero_owner_nonce_and_forged_link() {
    let mut zero_owner = logical_input();
    zero_owner.statement_identity.owner.txn_id = 0;
    assert!(matches!(
        LogicalStatementContribution::new(zero_owner, 101),
        Err(PreWalFootprintError::ZeroTransactionOwner)
    ));

    let mut zero_nonce = logical_input();
    zero_nonce.statement_identity.owner.registration_nonce = 0;
    assert!(matches!(
        LogicalStatementContribution::new(zero_nonce, 101),
        Err(PreWalFootprintError::ZeroTransactionRegistrationNonce)
    ));

    for sabotage in [
        |link: &mut PreWalOverlayLink| link.after_generation += 1,
        |link: &mut PreWalOverlayLink| link.before_root_digest = [0; 32],
        |link: &mut PreWalOverlayLink| link.after_root_digest = link.before_root_digest,
    ] {
        let mut invalid = logical_input();
        sabotage(&mut invalid.statement_identity.overlay_link);
        assert!(matches!(
            LogicalStatementContribution::new(invalid, 101),
            Err(PreWalFootprintError::InvalidOverlayLink)
        ));
    }
}

#[test]
fn row_id_geometry_is_exact_and_checked() {
    let mut mismatch = logical_input();
    mismatch.row_id_bytes = 8;
    mismatch.row_id_slots = 2;
    assert!(matches!(
        LogicalStatementContribution::new(mismatch, 101),
        Err(PreWalFootprintError::RowIdBytesMismatch { bytes: 8, slots: 2 })
    ));

    let mut overflow = logical_input();
    overflow.row_id_slots = u64::MAX;
    overflow.row_id_bytes = 0;
    assert!(matches!(
        LogicalStatementContribution::new(overflow, 101),
        Err(PreWalFootprintError::Overflow("row-id bytes from slots"))
    ));
}

#[test]
fn pool_totals_sum_distinct_target_storage_per_gpu_and_max_peak_scratch() {
    let mut input = input(0, 7);
    let mut second = gpu(0, 8);
    second.old_generation_pinned_bytes = 47;
    second.new_persistent_bytes = 53;
    second.plan_retained_transient_bytes = 59;
    second.allocation_slots = 11;
    second.generation_pin_slots = 13;
    second.scratch_peak_bytes = 61;
    second.max_host_readback_bytes = 67;
    input.final_physical.gpus = vec![gpu(0, 7), second].into();
    input.final_physical.publication_targets = vec![
        PreWalPublicationTarget {
            target: target(0, 7),
            bytes: 32,
            slots: 1,
        },
        PreWalPublicationTarget {
            target: target(0, 8),
            bytes: 32,
            slots: 1,
        },
    ]
    .into();
    let current = CurrentTwoFragmentShape::new(input, 101).unwrap();
    let mut pools = current.footprint().gpu_pool_cursor();
    let total = pools.try_next().unwrap().expect("one GPU pool");
    assert!(pools.try_next().unwrap().is_none());
    assert_eq!(total.old_generation_pinned_bytes, 23 + 47);
    assert_eq!(total.new_persistent_bytes, 29 + 53);
    assert_eq!(total.plan_retained_transient_bytes, 31 + 59);
    assert_eq!(total.result_retained_device_bytes, 0);
    assert_eq!(total.allocation_slots, 5 + 11);
    assert_eq!(total.generation_pin_slots, 7 + 13);
    assert_eq!(total.scratch_peak_bytes, 61);
    assert_eq!(total.max_host_readback_bytes, 67);
    assert_eq!(total.incremental_device_bytes().unwrap(), 172);
    assert_eq!(total.incremental_device_peak_bytes().unwrap(), 233);
    assert_eq!(current.footprint().publication_target_slots, 2);
}

#[test]
fn gpu_pool_incremental_admission_excludes_old_authoritative_generation() {
    let mut input = input(0, 7);
    let gpu = &mut input.final_physical.gpus[0];
    gpu.old_generation_pinned_bytes = u64::MAX;
    gpu.new_persistent_bytes = 0;
    gpu.plan_retained_transient_bytes = 0;
    gpu.result_retained_device_bytes = 0;
    gpu.scratch_peak_bytes = 0;

    let current = CurrentTwoFragmentShape::new(input, 101).unwrap();
    let mut pools = current.footprint().gpu_pool_cursor();
    let total = pools.try_next().unwrap().expect("one GPU pool");
    assert!(pools.try_next().unwrap().is_none());
    assert_eq!(total.old_generation_pinned_bytes, u64::MAX);
    assert_eq!(total.incremental_device_bytes().unwrap(), 0);
    assert_eq!(total.incremental_device_peak_bytes().unwrap(), 0);
}

#[test]
fn footprint_leaf_stays_scalar_and_has_no_live_authority_words() {
    let source = include_str!("pre_wal_footprint.rs")
        .split("\n#[cfg(test)]\nmod tests")
        .next()
        .expect("production leaf precedes tests");
    for forbidden in [
        "WalBuffer",
        "CanonicalFragmentKind",
        "opcode",
        "publish_",
        "acknowledge",
        "eligibility",
        "Engine::",
    ] {
        assert!(
            !source.contains(forbidden),
            "footprint leaf must not gain live authority: {forbidden}"
        );
    }
}
