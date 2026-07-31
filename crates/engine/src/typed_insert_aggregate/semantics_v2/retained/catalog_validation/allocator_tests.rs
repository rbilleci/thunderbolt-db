//! Bounded hostile cases for the concrete immutable durable-index proof.
//!
//! The cfg(test) adapter supplies an authenticated immutable index, opaque pin, and explicit
//! selected witnesses to the production validator. No test relies on an inferred active epoch.

use super::{
    immutable_index_root, proof_from_immutable_checked_records_for_test,
    validate_allocator_closure, validate_durable_allocator_index_identity,
    SemanticsV2DurableAllocatorLeaseRecord, SemanticsV2ImmutableDurableAllocatorIndex,
    SemanticsV2PinnedAllocatorIndexGeneration, SemanticsV2SelectedAllocatorLeaseWitness,
};
use crate::typed_insert_aggregate::semantics_v2::{
    pass_zero::SemanticsV2S7HeaderIdentity,
    retained::{
        graph::{ReservedSemanticsV2Graph, RetainedTable},
        SemanticsV2BoundIdentity,
    },
};

const DATABASE: [u8; 16] = [1; 16];
const CLUSTER: [u8; 16] = [2; 16];
const TIMELINE: [u8; 16] = [3; 16];
const FORMAT_EPOCH: u64 = 5;
const LEADER_EPOCH: u64 = 6;
const CATALOG_EPOCH: u64 = 7;
const INDEX_GENERATION: u64 = 11;
const USER_TRANSACTION: u64 = 77;
const USER_COMMIT: u64 = 50;

fn identity() -> SemanticsV2BoundIdentity {
    SemanticsV2BoundIdentity {
        database_id: DATABASE,
        cluster_id: CLUSTER,
        timeline_id: TIMELINE,
        format_epoch: FORMAT_EPOCH,
        leader_epoch: LEADER_EPOCH,
        catalog_epoch: CATALOG_EPOCH,
        catalog_digest: [4; 32],
        stable_transaction_id: USER_TRANSACTION,
        request_digest: [11; 32],
        autocommit: true,
        commit_sequence: USER_COMMIT,
        initial_database_root: [5; 32],
    }
}

fn record(
    stable_allocator_id: u64,
    lease_epoch: u64,
    start: u64,
    end: u64,
    marker_commit_sequence: u64,
) -> SemanticsV2DurableAllocatorLeaseRecord {
    SemanticsV2DurableAllocatorLeaseRecord {
        database_id: DATABASE,
        allocator_kind: 1,
        stable_allocator_id,
        lease_epoch,
        lease_start: start,
        lease_end: end,
        prior_high_water: start,
        new_high_water: end,
        marker_system_transaction_id: 9,
        marker_commit_sequence,
    }
}

fn selected(
    record_ordinal: u32,
    record: SemanticsV2DurableAllocatorLeaseRecord,
) -> SemanticsV2SelectedAllocatorLeaseWitness {
    SemanticsV2SelectedAllocatorLeaseWitness {
        record_ordinal,
        database_id: record.database_id,
        allocator_kind: record.allocator_kind,
        stable_allocator_id: record.stable_allocator_id,
        lease_epoch: record.lease_epoch,
        lease_start: record.lease_start,
        lease_end: record.lease_end,
        prior_high_water: record.prior_high_water,
        new_high_water: record.new_high_water,
        marker_system_transaction_id: record.marker_system_transaction_id,
        marker_commit_sequence: record.marker_commit_sequence,
    }
}

fn snapshot<'a>(
    records: &'a [SemanticsV2DurableAllocatorLeaseRecord],
    complete_next: u64,
    durable_next: u64,
    published_next: u64,
) -> SemanticsV2ImmutableDurableAllocatorIndex<'a> {
    let mut snapshot = SemanticsV2ImmutableDurableAllocatorIndex {
        database_id: DATABASE,
        index_generation: INDEX_GENERATION,
        index_root: [0; 32],
        complete_next_commit_sequence: complete_next,
        durable_next_commit_sequence: durable_next,
        published_next_commit_sequence: published_next,
        records,
    };
    snapshot.index_root = immutable_index_root(&snapshot);
    snapshot
}

fn pin(
    snapshot: &SemanticsV2ImmutableDurableAllocatorIndex<'_>,
    retained_through: u64,
) -> SemanticsV2PinnedAllocatorIndexGeneration {
    SemanticsV2PinnedAllocatorIndexGeneration {
        database_id: DATABASE,
        cluster_id: CLUSTER,
        timeline_id: TIMELINE,
        format_epoch: FORMAT_EPOCH,
        leader_epoch: LEADER_EPOCH,
        index_generation: snapshot.index_generation,
        index_root: snapshot.index_root,
        retained_through_commit_sequence: retained_through,
    }
}

fn validates_identity(
    snapshot: &SemanticsV2ImmutableDurableAllocatorIndex<'_>,
    pin: &SemanticsV2PinnedAllocatorIndexGeneration,
    selected: &[SemanticsV2SelectedAllocatorLeaseWitness],
) -> bool {
    validate_durable_allocator_index_identity(
        identity(),
        &proof_from_immutable_checked_records_for_test(snapshot, pin, selected),
    )
    .is_ok()
}

fn table(table_ref: u32, stable_table_id: u64, before: u64, high_water: u64) -> RetainedTable {
    RetainedTable {
        table_ref,
        stable_table_id,
        display_oid: table_ref + 1,
        target_dependency_ref: u32::MAX,
        catalog_epoch: CATALOG_EPOCH,
        data_generation_before: 1,
        data_generation_after: 1,
        row_allocator_before: before,
        row_allocator_high_water: high_water,
        initial_logical_row_count: 0,
        final_logical_row_count: 0,
        disposition_start: 0,
        disposition_count: 0,
        transition_start: 0,
        transition_count: 0,
        key_effect_start: 0,
        key_effect_count: 0,
        owned_index_start: 0,
        owned_index_count: 0,
        image_ref: table_ref,
        catalog_column_count: 0,
        schema_digest: [0; 32],
        initial_table_root: [0; 32],
        final_table_root: [0; 32],
        transition_root: [0; 32],
        index_effect_root: [0; 32],
        image_layout_digest: [0; 32],
        image_content_digest: [0; 32],
        image_arena_offset: 0,
        image_encoded_bytes: 0,
        image_descriptor_digest: [0; 32],
        manifest_digest: [0; 32],
    }
}

fn graph(tables: Vec<RetainedTable>) -> ReservedSemanticsV2Graph {
    ReservedSemanticsV2Graph {
        header: SemanticsV2S7HeaderIdentity {
            total_bytes: 1,
            root_descriptor_version: 1,
            catalog_before_epoch: CATALOG_EPOCH,
            catalog_after_epoch: CATALOG_EPOCH,
            catalog_before_digest: [4; 32],
            catalog_after_digest: [4; 32],
            initial_database_root: [5; 32],
            final_database_root: [6; 32],
            initial_overlay_root: [7; 32],
            final_overlay_root: [8; 32],
            root_descriptor: [9; 32],
            payload_digest: [10; 32],
        },
        statements: Vec::new(),
        records: Vec::new(),
        dispositions: Vec::new(),
        sequence_effects: Vec::new(),
        outcomes: Vec::new(),
        tables,
        table_dispositions: Vec::new(),
        resolutions: Vec::new(),
        dependencies: Vec::new(),
        dependency_uses: Vec::new(),
        indexes: Vec::new(),
        index_key_columns: Vec::new(),
        transitions: Vec::new(),
        key_effects: Vec::new(),
        key_components: Vec::new(),
        projections: Vec::new(),
        images: Vec::new(),
        response:
            crate::typed_insert_aggregate::semantics_v2::retained::graph::empty_response_for_test(),
    }
}

fn closes(
    snapshot: &SemanticsV2ImmutableDurableAllocatorIndex<'_>,
    pin: &SemanticsV2PinnedAllocatorIndexGeneration,
    selected: &[SemanticsV2SelectedAllocatorLeaseWitness],
    tables: Vec<RetainedTable>,
) -> bool {
    validate_allocator_closure(
        identity(),
        &graph(tables),
        &proof_from_immutable_checked_records_for_test(snapshot, pin, selected),
    )
    .is_ok()
}

#[test]
fn durable_index_proof_accepts_mixed_selected_epochs_and_unrelated_complete_records() {
    let records = [
        record(102, 19, 30, 40, 49),
        record(100, 12, 1, 3, 70),
        record(101, 12, 10, 20, 49),
    ];
    let snapshot = snapshot(&records, u64::MAX, u64::MAX, u64::MAX);
    let pin = pin(&snapshot, u64::MAX);
    let selected = [selected(2, records[2]), selected(0, records[0])];
    assert!(closes(
        &snapshot,
        &pin,
        &selected,
        vec![table(0, 101, 12, 18), table(1, 102, 31, 39)],
    ));
}

#[test]
fn durable_index_proof_rejects_missing_duplicate_reordered_or_nonmember_selected_witnesses() {
    let records = [record(101, 12, 10, 20, 49), record(102, 19, 30, 40, 49)];
    let snapshot = snapshot(&records, 80, 80, 80);
    let pin = pin(&snapshot, 49);
    let first = selected(0, records[0]);
    let second = selected(1, records[1]);
    assert!(!closes(
        &snapshot,
        &pin,
        &[],
        vec![table(0, 101, 12, 18), table(1, 102, 31, 39)],
    ));
    assert!(!validates_identity(&snapshot, &pin, &[first, first]));
    assert!(!closes(
        &snapshot,
        &pin,
        &[second, first],
        vec![table(0, 101, 12, 18), table(1, 102, 31, 39)],
    ));
    let mut nonmember = first;
    nonmember.record_ordinal = 99;
    assert!(!validates_identity(&snapshot, &pin, &[nonmember]));
}

#[test]
fn durable_index_proof_rejects_refund_shaped_selected_lease_against_unchanged_index() {
    let records = [record(101, 12, 10, 20, 49)];
    let snapshot = snapshot(&records, 80, 80, 80);
    let pin = pin(&snapshot, 49);
    let full = selected(0, records[0]);
    assert!(closes(
        &snapshot,
        &pin,
        &[full],
        vec![table(0, 101, 12, 18)],
    ));
    let mut narrowed = full;
    narrowed.lease_start = 12;
    narrowed.lease_end = 18;
    narrowed.prior_high_water = 12;
    narrowed.new_high_water = 18;
    assert!(
        !closes(&snapshot, &pin, &[narrowed], vec![table(0, 101, 12, 18)],),
        "the selected witness must equal the unchanged authoritative full lease"
    );
}

#[test]
fn durable_index_proof_rejects_narrowed_authoritative_replacement_under_unchanged_pin() {
    let full_records = [record(101, 12, 10, 20, 49)];
    let full_snapshot = snapshot(&full_records, 80, 80, 80);
    let pin = pin(&full_snapshot, 49);
    let narrow_records = [record(101, 12, 12, 18, 49)];
    let mut unchanged_root = snapshot(&narrow_records, 80, 80, 80);
    unchanged_root.index_root = full_snapshot.index_root;
    assert!(
        !closes(
            &unchanged_root,
            &pin,
            &[selected(0, narrow_records[0])],
            vec![table(0, 101, 12, 18)],
        ),
        "an unchanged authoritative root must not authenticate narrowed record bytes"
    );
    let narrow_snapshot = snapshot(&narrow_records, 80, 80, 80);
    assert!(
        !closes(
            &narrow_snapshot,
            &pin,
            &[selected(0, narrow_records[0])],
            vec![table(0, 101, 12, 18)],
        ),
        "the unchanged opaque pin must not authenticate a replaced index root"
    );
}

#[test]
fn durable_index_proof_rejects_selected_incomplete_nondurable_or_unpublished_marker() {
    let records = [record(101, 12, 10, 20, 49)];
    for (complete, durable, published) in [(49, 80, 80), (80, 49, 80), (80, 80, 49)] {
        let snapshot = snapshot(&records, complete, durable, published);
        let pin = pin(&snapshot, 49);
        assert!(!validates_identity(
            &snapshot,
            &pin,
            &[selected(0, records[0])],
        ));
    }
}

#[test]
fn durable_index_proof_rejects_selected_wrong_lineage_post_user_or_unretained_marker() {
    let records = [record(101, 12, 10, 20, 49)];
    let selected_snapshot = snapshot(&records, 80, 80, 80);
    let mut wrong_lineage = pin(&selected_snapshot, 49);
    wrong_lineage.cluster_id = [9; 16];
    assert!(!validates_identity(
        &selected_snapshot,
        &wrong_lineage,
        &[selected(0, records[0])],
    ));
    let unretained = pin(&selected_snapshot, 48);
    assert!(!validates_identity(
        &selected_snapshot,
        &unretained,
        &[selected(0, records[0])],
    ));

    let later_records = [record(101, 12, 10, 20, USER_COMMIT)];
    let later_snapshot = snapshot(&later_records, 80, 80, 80);
    let later_pin = pin(&later_snapshot, USER_COMMIT);
    assert!(!validates_identity(
        &later_snapshot,
        &later_pin,
        &[selected(0, later_records[0])],
    ));
}

#[test]
fn durable_index_proof_rejects_coherent_lineage_replacement_away_from_outer_identity() {
    let mut records = [record(101, 12, 10, 20, 49)];
    records[0].database_id = [9; 16];
    let mut replaced = snapshot(&records, 80, 80, 80);
    replaced.database_id = [9; 16];
    replaced.index_root = immutable_index_root(&replaced);
    let mut replaced_pin = pin(&replaced, 49);
    replaced_pin.database_id = [9; 16];
    replaced_pin.cluster_id = [9; 16];
    replaced_pin.timeline_id = [9; 16];
    replaced_pin.format_epoch = FORMAT_EPOCH + 1;
    replaced_pin.leader_epoch = LEADER_EPOCH + 1;
    assert!(
        !validates_identity(&replaced, &replaced_pin, &[selected(0, records[0])]),
        "an internally coherent replacement must still bind the exact outer publication identity"
    );

    let original_records = [record(101, 12, 10, 20, 49)];
    let original_snapshot = snapshot(&original_records, 80, 80, 80);
    let mut leader_only_drift = pin(&original_snapshot, 49);
    leader_only_drift.leader_epoch = LEADER_EPOCH + 1;
    assert!(
        !validates_identity(
            &original_snapshot,
            &leader_only_drift,
            &[selected(0, original_records[0])],
        ),
        "leader/log epoch is part of the exact outer publication identity"
    );
}

#[test]
fn durable_index_proof_rejects_global_same_allocator_epoch_overlap() {
    let records = [record(101, 12, 10, 20, 49), record(101, 12, 15, 30, 49)];
    let snapshot = snapshot(&records, 80, 80, 80);
    let pin = pin(&snapshot, 49);
    assert!(!validates_identity(&snapshot, &pin, &[]));
}

#[test]
fn durable_index_proof_rejects_malformed_complete_record_without_self_described_lifecycle_bits() {
    let mut malformed = record(101, 12, 10, 20, 49);
    malformed.new_high_water = 19;
    let records = [malformed];
    let snapshot = snapshot(&records, 80, 80, 80);
    let pin = pin(&snapshot, 49);
    assert!(!validates_identity(
        &snapshot,
        &pin,
        &[selected(0, records[0])],
    ));
}
