use super::*;

fn target(gpu_id: u16, table_oid: u32, schema: u8) -> GpuTargetKey {
    GpuTargetKey {
        gpu_id,
        table_oid,
        schema_digest: [schema; 32],
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

fn gpu(gpu_id: u16, table_oid: u32, schema: u8) -> PreWalGpuFootprint {
    PreWalGpuFootprint {
        target: target(gpu_id, table_oid, schema),
        generation: witness(),
        old_generation_pinned_bytes: 10,
        new_persistent_bytes: 20,
        plan_retained_transient_bytes: 30,
        result_retained_device_bytes: 40,
        allocation_slots: 2,
        generation_pin_slots: 3,
        scratch_peak_bytes: 50,
        max_host_readback_bytes: 4,
    }
}

#[test]
fn final_gpu_set_requires_one_catalog_and_predecessor_cut() {
    let first = gpu(0, 7, 7);
    let mut catalog_drift = gpu(1, 8, 8);
    catalog_drift.generation.catalog_seq += 1;
    assert!(matches!(
        checked_sorted_gpus(&[first.clone(), catalog_drift]),
        Err(PreWalFootprintError::GlobalGenerationCutDrift(_))
    ));

    let mut predecessor_drift = gpu(1, 8, 8);
    predecessor_drift.generation.predecessor_boundary += 1;
    assert!(matches!(
        checked_sorted_gpus(&[first, predecessor_drift]),
        Err(PreWalFootprintError::GlobalGenerationCutDrift(_))
    ));
}

#[test]
fn catalog_cut_cannot_follow_its_predecessor_boundary() {
    let mut invalid = gpu(0, 7, 7);
    invalid.generation.catalog_seq = invalid.generation.predecessor_boundary + 1;
    assert!(matches!(
        checked_sorted_gpus(&[invalid]),
        Err(PreWalFootprintError::InvalidGpuGenerationWitness(_))
    ));
}

#[test]
fn table_schema_is_global_across_gpu_and_publication_targets() {
    let first = gpu(0, 7, 7);
    let second = gpu(1, 7, 8);
    assert!(matches!(
        checked_sorted_gpus(&[first, second]),
        Err(PreWalFootprintError::GpuTargetSchemaDrift(_))
    ));

    let publications = [
        PreWalPublicationTarget {
            target: target(0, 7, 7),
            bytes: 1,
            slots: 1,
        },
        PreWalPublicationTarget {
            target: target(1, 7, 8),
            bytes: 1,
            slots: 1,
        },
    ];
    assert!(matches!(
        checked_sorted_publications(&publications),
        Err(PreWalFootprintError::PublicationTargetSchemaDrift(_))
    ));
}

#[test]
fn zero_schema_digest_is_not_a_stable_target_identity() {
    let invalid = gpu(0, 7, 0);
    assert!(matches!(
        checked_sorted_gpus(&[invalid]),
        Err(PreWalFootprintError::InvalidGpuTarget(_))
    ));
    assert!(matches!(
        checked_sorted_publications(&[PreWalPublicationTarget {
            target: target(0, 7, 0),
            bytes: 1,
            slots: 1,
        }]),
        Err(PreWalFootprintError::InvalidGpuTarget(_))
    ));
}

#[test]
fn one_table_has_one_even_mutation_epoch_across_devices() {
    let first = gpu(0, 7, 7);
    let mut second = gpu(1, 7, 7);
    second.generation.index_mutation_epoch_even += 2;
    assert!(matches!(
        checked_sorted_gpus(&[first, second]),
        Err(PreWalFootprintError::TableMutationEpochDrift(_))
    ));
}

#[test]
fn witness_checks_the_addressable_capacity_extent_not_only_live_rows() {
    let mut invalid = gpu(0, 7, 7);
    invalid.generation.row_start = u64::MAX - invalid.generation.row_count;
    assert_eq!(
        invalid
            .generation
            .row_start
            .checked_add(invalid.generation.row_count),
        Some(u64::MAX)
    );
    assert!(invalid
        .generation
        .row_start
        .checked_add(invalid.generation.capacity)
        .is_none());
    assert!(matches!(
        checked_sorted_gpus(&[invalid]),
        Err(PreWalFootprintError::InvalidGpuGenerationWitness(_))
    ));
}
