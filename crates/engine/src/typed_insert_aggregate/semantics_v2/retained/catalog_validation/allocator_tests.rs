//! Bounded hostile cases for the concrete immutable durable-index proof.
//!
//! The cfg(test) adapter supplies an authenticated immutable index, opaque pin, and explicit
//! selected witnesses to the production validator. No test relies on an inferred active epoch.

use super::super::{
    ValidatedAllocatorAssignment, ValidatedAllocatorReplayRowBinding,
    ValidatedAllocatorReplayRowBindings,
};
use super::{
    immutable_index_root, proof_from_immutable_checked_records_for_test,
    validate_durable_allocator_index_identity, AllocatorAssignmentSeal,
    SemanticsV2DurableAllocatorAssignmentRecord, SemanticsV2DurableAllocatorLeaseRecord,
    SemanticsV2ImmutableDurableAllocatorIndex, SemanticsV2PinnedAllocatorIndexGeneration,
    SemanticsV2SelectedAllocatorAssignmentWitness, SemanticsV2SelectedAllocatorLeaseWitness,
};
use crate::typed_insert_aggregate::semantics_v2::retained::{
    graph::RetainedTable, SemanticsV2BoundIdentity,
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
        marker_system_transaction_id: marker_commit_sequence,
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

fn selected_assignment(
    assignment_ordinal: u32,
    stable_table_id: u64,
    statement_ordinal: u32,
    source_row_ordinal: u32,
    stable_row_id: u64,
) -> SemanticsV2SelectedAllocatorAssignmentWitness {
    SemanticsV2SelectedAllocatorAssignmentWitness {
        assignment_ordinal,
        database_id: DATABASE,
        cluster_id: CLUSTER,
        timeline_id: TIMELINE,
        format_epoch: FORMAT_EPOCH,
        leader_epoch: LEADER_EPOCH,
        parent_stable_transaction_id: USER_TRANSACTION,
        parent_request_digest: [11; 32],
        parent_commit_sequence: USER_COMMIT,
        parent_autocommit: true,
        parent_statement_ordinal: statement_ordinal,
        parent_statement_request_digest: [12; 32],
        parent_typed_statement_digest: [13; 32],
        lease_record_ordinal: assignment_ordinal,
        allocator_kind: 1,
        stable_allocator_id: stable_table_id,
        lease_epoch: 12,
        lease_start: stable_row_id,
        lease_end: stable_row_id + 1,
        mapping_version: 1,
        source_order: u64::from(source_row_ordinal),
        source_row_ordinal,
        assignment_start: stable_row_id,
        assignment_end: stable_row_id + 1,
        marker_system_transaction_id: 49,
        marker_commit_sequence: 49,
    }
}

fn assert_replay_binding(
    binding: ValidatedAllocatorReplayRowBinding,
    stable_table_id: u64,
    statement_ordinal: u32,
    source_row_ordinal: u32,
    stable_row_id: u64,
) {
    assert_eq!(binding.stable_table_id(), stable_table_id);
    assert_eq!(binding.statement_ordinal(), statement_ordinal);
    assert_eq!(binding.source_row_ordinal(), source_row_ordinal);
    assert_eq!(binding.stable_row_id(), stable_row_id);
}

fn replay_binding_count(rows: ValidatedAllocatorReplayRowBindings<'_>) -> usize {
    rows.len()
}

fn snapshot<'a>(
    records: &'a [SemanticsV2DurableAllocatorLeaseRecord],
    complete_next: u64,
    durable_next: u64,
    published_next: u64,
) -> SemanticsV2ImmutableDurableAllocatorIndex<'a> {
    snapshot_with_assignments(records, &[], complete_next, durable_next, published_next)
}

fn snapshot_with_assignments<'a>(
    records: &'a [SemanticsV2DurableAllocatorLeaseRecord],
    assignments: &'a [SemanticsV2DurableAllocatorAssignmentRecord],
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
        assignments,
    };
    snapshot.index_root = immutable_index_root(&snapshot);
    snapshot
}

fn immutable_assignment(
    lease_record_ordinal: u32,
    lease: SemanticsV2DurableAllocatorLeaseRecord,
    stable_row_id: u64,
) -> SemanticsV2DurableAllocatorAssignmentRecord {
    SemanticsV2DurableAllocatorAssignmentRecord {
        database_id: DATABASE,
        cluster_id: CLUSTER,
        timeline_id: TIMELINE,
        format_epoch: FORMAT_EPOCH,
        leader_epoch: LEADER_EPOCH,
        parent_stable_transaction_id: USER_TRANSACTION,
        parent_request_digest: [11; 32],
        parent_commit_sequence: USER_COMMIT,
        parent_autocommit: true,
        parent_statement_ordinal: 0,
        parent_statement_request_digest: [12; 32],
        parent_typed_statement_digest: [13; 32],
        lease_record_ordinal,
        allocator_kind: lease.allocator_kind,
        stable_allocator_id: lease.stable_allocator_id,
        lease_epoch: lease.lease_epoch,
        lease_start: lease.lease_start,
        lease_end: lease.lease_end,
        mapping_version: 1,
        source_order: 0,
        source_row_ordinal: 0,
        assignment_start: stable_row_id,
        assignment_end: stable_row_id + 1,
        marker_system_transaction_id: lease.marker_system_transaction_id,
        marker_commit_sequence: lease.marker_commit_sequence,
    }
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
        &proof_from_immutable_checked_records_for_test(snapshot, pin, selected, &[]),
    )
    .is_ok()
}

fn table(table_ref: u32, stable_table_id: u64, before: u64, high_water: u64) -> RetainedTable {
    RetainedTable {
        table_ref,
        resets_existing_rows: false,
        initial_table_absent: false,
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

fn closes(
    snapshot: &SemanticsV2ImmutableDurableAllocatorIndex<'_>,
    pin: &SemanticsV2PinnedAllocatorIndexGeneration,
    selected: &[SemanticsV2SelectedAllocatorLeaseWitness],
    tables: Vec<RetainedTable>,
) -> bool {
    let proof = proof_from_immutable_checked_records_for_test(snapshot, pin, selected, &[]);
    validate_durable_allocator_index_identity(identity(), &proof).is_ok()
        && selected.len() == tables.len()
        && tables
            .iter()
            .zip(selected)
            .enumerate()
            .all(|(ordinal, (table, lease))| {
                table.table_ref == ordinal as u32
                    && table.stable_table_id == lease.stable_allocator_id
                    && lease.lease_start <= table.row_allocator_before
                    && table.row_allocator_high_water <= lease.lease_end
            })
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
fn durable_index_proof_rejects_frontier_before_parent_with_unselected_same_parent_assignment() {
    let records = [record(101, 12, 10, 20, 1), record(102, 12, 30, 40, 49)];
    let assignments = [immutable_assignment(1, records[1], 30)];
    // Marker one is a valid selected predecessor, while marker 49 is a later durable assignment
    // for this same parent. A next frontier of two would let a partial root omit that row unless
    // complete/durable/published frontiers are all bound beyond the parent commit (50).
    let snapshot = snapshot_with_assignments(&records, &assignments, 2, 2, 2);
    let pin = pin(&snapshot, 49);
    let error = validate_durable_allocator_index_identity(
        identity(),
        &proof_from_immutable_checked_records_for_test(
            &snapshot,
            &pin,
            &[selected(0, records[0])],
            &[],
        ),
    )
    .expect_err("a complete allocator root must cover the parent commit");
    assert!(
        error.to_string().contains("lineage/frontier"),
        "the incomplete frontier must reject before an omitted same-parent assignment can pose: {error}"
    );
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
fn durable_index_proof_rejects_system_transaction_reused_at_a_second_commit() {
    let mut records = [record(101, 12, 10, 20, 49), record(102, 12, 30, 40, 70)];
    records[1].marker_system_transaction_id = records[0].marker_system_transaction_id;
    let snapshot = snapshot(&records, 80, 80, 80);
    let pin = pin(&snapshot, 70);
    assert!(
        !validates_identity(&snapshot, &pin, &[]),
        "one stable system transaction must not authenticate allocator records at two commits"
    );
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

#[test]
fn replay_row_bindings_project_prevalidated_rows_in_canonical_table_statement_source_order() {
    // These slots model canonical S1/S2 source rows whose later retained dispositions are
    // canceled and suppressed.  The capability intentionally does not inspect those later
    // relations: each input source row remains assigned before generation.
    let selected = [
        selected_assignment(0, 101, 0, 0, 11),
        selected_assignment(1, 101, 0, 1, 12),
        selected_assignment(2, 102, 1, 0, 30),
    ];
    let assignment = ValidatedAllocatorAssignment {
        selected: &selected,
        seal: AllocatorAssignmentSeal,
    };

    let mut rows = assignment.replay_row_bindings();
    assert_eq!(
        replay_binding_count(assignment.replay_row_bindings()),
        3,
        "the re-exported iterator remains usable by a sibling retained compiler module"
    );
    assert_eq!(
        rows.len(),
        3,
        "the exact-size iterator retains all source rows"
    );
    assert_eq!(rows.size_hint(), (3, Some(3)));

    let first = rows.next().expect("first canonical binding");
    assert_replay_binding(first, 101, 0, 0, 11);

    let second = rows.next().expect("canceled source-row binding");
    assert_replay_binding(second, 101, 0, 1, 12);

    let third = rows.next().expect("suppressed source-row binding");
    assert_replay_binding(third, 102, 1, 0, 30);
    assert_eq!(rows.len(), 0);
    assert!(rows.next().is_none());
}

#[test]
fn production_replay_row_binding_iteration_stays_narrow_and_allocation_free() {
    let source = include_str!("allocator.rs");
    let binding_fields = source
        .split("struct ValidatedAllocatorReplayRowBinding")
        .nth(1)
        .and_then(|tail| tail.split("struct AllocatorReplayRowBindingSeal").next())
        .expect("allocator source has a sealed replay-row binding");
    for required in [
        "stable_table_id: u64",
        "statement_ordinal: u32",
        "source_row_ordinal: u32",
        "stable_row_id: u64",
    ] {
        assert!(
            binding_fields.contains(required),
            "replay-row binding must retain scalar compiler input {required}"
        );
    }
    for forbidden in [
        "lease_",
        "marker_",
        "root",
        "assignment_",
        "selected",
        "range",
    ] {
        assert!(
            !binding_fields.contains(forbidden),
            "replay-row binding must not expose allocator authority through {forbidden}"
        );
    }
    let projection = source
        .split("impl Iterator for ValidatedAllocatorReplayRowBindings")
        .nth(1)
        .and_then(|tail| tail.split("/// Test-only adapter").next())
        .expect("allocator source has a replay-row iterator boundary");
    for required in [
        "ExactSizeIterator for ValidatedAllocatorReplayRowBindings",
        "stable_table_id: assignment.stable_allocator_id",
        "statement_ordinal: assignment.parent_statement_ordinal",
        "source_row_ordinal: assignment.source_row_ordinal",
        "stable_row_id: assignment.assignment_start",
    ] {
        assert!(
            projection.contains(required),
            "production replay-row iterator must retain {required}"
        );
    }
    for forbidden in [
        "HashMap",
        "Vec<",
        "Box<",
        ".collect()",
        ".clone()",
        "assignment_end",
        "lease_",
        "marker_",
        "index_root",
        "graph.",
        "S4",
        "S7",
        "disposition",
        "resolution",
        "row_allocator",
    ] {
        assert!(
            !projection.contains(forbidden),
            "production replay-row iterator must not infer compiler inputs through {forbidden}"
        );
    }
}

#[test]
fn production_allocator_assignment_closure_stays_root_bound_and_allocation_free() {
    let source = include_str!("allocator.rs");
    let production = source
        .split("fn validate_exact_assignment_closure")
        .nth(1)
        .and_then(|tail| tail.split("pub(super) fn validate_table_order").next())
        .expect("allocator source has an assignment-validation boundary");
    for required in [
        "SemanticsV2DurableAllocatorAssignmentRecord",
        "SemanticsV2SelectedAllocatorAssignmentWitness",
        "ValidatedAllocatorAssignment",
        "validate_exact_assignment_closure",
        "immutable_index_root",
    ] {
        assert!(
            source.contains(required),
            "production durable allocator proof must retain {required}"
        );
    }
    for forbidden in ["HashMap", "Vec<", "Box<", ".collect()", ".clone()"] {
        assert!(
            !production.contains(forbidden),
            "production durable allocator assignment validation must not allocate through {forbidden}"
        );
    }
    for forbidden in [
        "graph.dispositions",
        "graph.resolutions",
        "graph.tables",
        "row_allocator",
        "S4",
        "S6",
        "S7",
        "outcome",
        "RetentionAuthority",
    ] {
        assert!(
            !production.contains(forbidden),
            "production assignment validation must not select compiler inputs through {forbidden}"
        );
    }
}
