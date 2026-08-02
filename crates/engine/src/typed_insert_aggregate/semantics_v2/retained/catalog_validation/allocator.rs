//! Durable allocator leases and exact table closure for pinned catalog evidence.

use super::super::{
    graph::{ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedTable},
    SemanticsV2BoundIdentity, SemanticsV2CatalogColumnWitness, SemanticsV2CatalogTableWitness,
    SemanticsV2CatalogWitness,
};
use crate::typed_insert_batch::DecodedDependencyFacts;
use sha2::{Digest, Sha256};

/// Concrete borrowed authority from the durable allocator index.  It carries both the complete
/// immutable index and the transaction's explicit frozen selection; the latter is never inferred
/// from an epoch.  The non-`Copy` proof borrows an opaque pin, so its lifetime keeps the pinned
/// checkpoint/index generation alive until generation validation consumes the pending owner.
pub(in crate::typed_insert_aggregate::semantics_v2::retained) struct SemanticsV2DurableAllocatorIndexProof<
    'a,
> {
    snapshot: &'a SemanticsV2ImmutableDurableAllocatorIndex<'a>,
    checkpoint_pin: &'a SemanticsV2PinnedAllocatorIndexGeneration,
    selected: &'a [SemanticsV2SelectedAllocatorLeaseWitness],
    selected_assignments: &'a [SemanticsV2SelectedAllocatorAssignmentWitness],
}

/// The immutable complete index carries authenticated marker frontiers and a root recomputed
/// over every row.  Its fields are private to the durable-index adapter; codec bytes cannot
/// manufacture either a root or a record selection from this owner.
#[allow(dead_code)]
pub(super) struct SemanticsV2ImmutableDurableAllocatorIndex<'a> {
    database_id: [u8; 16],
    index_generation: u64,
    index_root: [u8; 32],
    complete_next_commit_sequence: u64,
    durable_next_commit_sequence: u64,
    published_next_commit_sequence: u64,
    records: &'a [SemanticsV2DurableAllocatorLeaseRecord],
    assignments: &'a [SemanticsV2DurableAllocatorAssignmentRecord],
}

/// Opaque active checkpoint/index-generation ownership.  Its non-`Copy` shape is intentional:
/// a proof can borrow this guard but cannot duplicate, retire, or recreate its retention claim.
#[allow(dead_code)]
pub(super) struct SemanticsV2PinnedAllocatorIndexGeneration {
    database_id: [u8; 16],
    cluster_id: [u8; 16],
    timeline_id: [u8; 16],
    format_epoch: u64,
    leader_epoch: u64,
    index_generation: u64,
    index_root: [u8; 32],
    retained_through_commit_sequence: u64,
}

/// One record in the authoritative complete index.  It stores only the frozen allocator-marker
/// tuple; complete/durable/published state comes from authenticated snapshot frontiers instead
/// of self-described copies on every row.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2DurableAllocatorLeaseRecord {
    database_id: [u8; 16],
    allocator_kind: u8,
    stable_allocator_id: u64,
    lease_epoch: u64,
    lease_start: u64,
    lease_end: u64,
    prior_high_water: u64,
    new_high_water: u64,
    marker_system_transaction_id: u64,
    marker_commit_sequence: u64,
}

/// Exact transaction-selected witness.  The index ordinal is a bounded handle, while the full
/// frozen tuple prevents a narrowed/reconstructed lease from posing as the authoritative row.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2SelectedAllocatorLeaseWitness {
    record_ordinal: u32,
    database_id: [u8; 16],
    allocator_kind: u8,
    stable_allocator_id: u64,
    lease_epoch: u64,
    lease_start: u64,
    lease_end: u64,
    prior_high_water: u64,
    new_high_water: u64,
    marker_system_transaction_id: u64,
    marker_commit_sequence: u64,
}

/// One root-authenticated allocation of a single typed source row.  It is intentionally an
/// allocator-index record rather than a derived S4/S7 map: the selected durable lease remains a
/// containing interval only, while this exact parent-bound row is the authority for its ID.  Its
/// marker is the owning lease marker, so the lease and every exact assignment are one durable
/// system-transaction event.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2DurableAllocatorAssignmentRecord {
    database_id: [u8; 16],
    cluster_id: [u8; 16],
    timeline_id: [u8; 16],
    format_epoch: u64,
    leader_epoch: u64,
    parent_stable_transaction_id: u64,
    parent_request_digest: [u8; 32],
    parent_commit_sequence: u64,
    parent_autocommit: bool,
    parent_statement_ordinal: u32,
    parent_statement_request_digest: [u8; 32],
    parent_typed_statement_digest: [u8; 32],
    lease_record_ordinal: u32,
    allocator_kind: u8,
    stable_allocator_id: u64,
    lease_epoch: u64,
    lease_start: u64,
    lease_end: u64,
    mapping_version: u32,
    source_order: u64,
    source_row_ordinal: u32,
    assignment_start: u64,
    assignment_end: u64,
    marker_system_transaction_id: u64,
    marker_commit_sequence: u64,
}

/// Exact selected copy of an immutable allocation row.  As with a lease witness, an ordinal
/// alone is never sufficient: the full tuple prevents a narrowed, reparented, or re-ordered
/// allocation from posing as the root-authenticated record.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2SelectedAllocatorAssignmentWitness {
    assignment_ordinal: u32,
    database_id: [u8; 16],
    cluster_id: [u8; 16],
    timeline_id: [u8; 16],
    format_epoch: u64,
    leader_epoch: u64,
    parent_stable_transaction_id: u64,
    parent_request_digest: [u8; 32],
    parent_commit_sequence: u64,
    parent_autocommit: bool,
    parent_statement_ordinal: u32,
    parent_statement_request_digest: [u8; 32],
    parent_typed_statement_digest: [u8; 32],
    lease_record_ordinal: u32,
    allocator_kind: u8,
    stable_allocator_id: u64,
    lease_epoch: u64,
    lease_start: u64,
    lease_end: u64,
    mapping_version: u32,
    source_order: u64,
    source_row_ordinal: u32,
    assignment_start: u64,
    assignment_end: u64,
    marker_system_transaction_id: u64,
    marker_commit_sequence: u64,
}

/// The only validated exact-row assignment capability.  It is produced while validating the
/// existing allocator proof and merely borrows its selected root-authenticated records; no map,
/// constructor, S4/S7 inference, or extraction API exists outside this authority.
pub(in crate::typed_insert_aggregate::semantics_v2::retained) struct ValidatedAllocatorAssignment<
    'a,
> {
    selected: &'a [SemanticsV2SelectedAllocatorAssignmentWitness],
    seal: AllocatorAssignmentSeal,
}

struct AllocatorAssignmentSeal;

/// One immutable, root-authenticated row binding narrowed to the only four scalar compiler
/// inputs.  Its fields remain private: it is not a replay authority, cannot be constructed by
/// callers, and deliberately carries none of the lease, marker, root, or allocation-range
/// evidence from which validation derived this capability.
#[allow(dead_code)]
pub(in crate::typed_insert_aggregate::semantics_v2::retained) struct ValidatedAllocatorReplayRowBinding
{
    stable_table_id: u64,
    statement_ordinal: u32,
    source_row_ordinal: u32,
    stable_row_id: u64,
    seal: AllocatorReplayRowBindingSeal,
}

struct AllocatorReplayRowBindingSeal;

/// Allocation-free traversal over the selected exact allocator assignments.  This iterator is
/// intentionally non-constructible outside the validated capability and retains the canonical
/// selected-witness order established by `validate_exact_assignment_closure`.
#[allow(dead_code)]
pub(in crate::typed_insert_aggregate::semantics_v2::retained) struct ValidatedAllocatorReplayRowBindings<
    'a,
> {
    selected: std::slice::Iter<'a, SemanticsV2SelectedAllocatorAssignmentWitness>,
}

impl<'a> ValidatedAllocatorAssignment<'a> {
    /// Return the exact stable row bindings for the future replay compiler.  Construction of
    /// `ValidatedAllocatorAssignment` is private to the root-authenticated closure, so
    /// this is a read-only capability projection rather than an inference from a lease interval
    /// or retained S4/S7 state.
    #[must_use]
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) fn replay_row_bindings(
        &self,
    ) -> ValidatedAllocatorReplayRowBindings<'a> {
        ValidatedAllocatorReplayRowBindings {
            selected: self.selected.iter(),
        }
    }
}

impl ValidatedAllocatorReplayRowBinding {
    #[must_use]
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) fn stable_table_id(&self) -> u64 {
        self.stable_table_id
    }

    #[must_use]
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) fn statement_ordinal(
        &self,
    ) -> u32 {
        self.statement_ordinal
    }

    #[must_use]
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) fn source_row_ordinal(
        &self,
    ) -> u32 {
        self.source_row_ordinal
    }

    #[must_use]
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) fn stable_row_id(&self) -> u64 {
        self.stable_row_id
    }
}

impl Iterator for ValidatedAllocatorReplayRowBindings<'_> {
    type Item = ValidatedAllocatorReplayRowBinding;

    fn next(&mut self) -> Option<Self::Item> {
        self.selected
            .next()
            .map(|assignment| ValidatedAllocatorReplayRowBinding {
                stable_table_id: assignment.stable_allocator_id,
                statement_ordinal: assignment.parent_statement_ordinal,
                source_row_ordinal: assignment.source_row_ordinal,
                stable_row_id: assignment.assignment_start,
                seal: AllocatorReplayRowBindingSeal,
            })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.selected.size_hint()
    }
}

impl ExactSizeIterator for ValidatedAllocatorReplayRowBindings<'_> {
    fn len(&self) -> usize {
        self.selected.len()
    }
}

/// Test-only adapter for immutable authoritative records and explicit selected witnesses.  It
/// is the sole local construction seam; production has no callback or codec-visible constructor.
#[cfg(test)]
pub(super) fn proof_from_immutable_checked_records_for_test<'a>(
    snapshot: &'a SemanticsV2ImmutableDurableAllocatorIndex<'a>,
    checkpoint_pin: &'a SemanticsV2PinnedAllocatorIndexGeneration,
    selected: &'a [SemanticsV2SelectedAllocatorLeaseWitness],
    selected_assignments: &'a [SemanticsV2SelectedAllocatorAssignmentWitness],
) -> SemanticsV2DurableAllocatorIndexProof<'a> {
    SemanticsV2DurableAllocatorIndexProof {
        snapshot,
        checkpoint_pin,
        selected,
        selected_assignments,
    }
}

/// Closure-scoped input for a retained-boundary test proof.  It carries only the table allocator
/// interval; the durable snapshot, authenticated root, active pin, selected rows, and all marker
/// lifecycle fields are constructed locally below and cannot escape this adapter.
#[cfg(test)]
#[derive(Clone, Copy)]
pub(in crate::typed_insert_aggregate::semantics_v2::retained) struct AllocatorLeaseSpecForTest {
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) stable_allocator_id: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) lease_start: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) lease_end: u64,
}

/// Hostile variants for the closure-scoped exact-assignment fixture adapter.  Production cannot
/// name this type or construct assignment backing; every variant still travels through the one
/// immutable allocator index and its authenticated root.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AllocatorAssignmentProofSabotageForTest {
    Absent,
    Duplicate,
    WrongParent,
    HiddenConflictingParent,
    HiddenDifferentParentOverlap,
    StatementChain,
    Table,
    Order,
    Range,
    Lineage,
    Lifecycle,
    MarkerIdentity,
}

/// Build an authoritative allocator proof only for the dynamic extent of `operation`.  The proof
/// borrows local records/snapshot/pin, so neither it nor its fixture backing can escape; callers
/// must consume it immediately through the ordinary catalog-and-allocator validation path.
#[cfg(test)]
pub(super) fn with_allocator_proof_for_test<T>(
    identity: SemanticsV2BoundIdentity,
    graph: ReservedSemanticsV2Graph,
    leases: &[AllocatorLeaseSpecForTest],
    sabotage: Option<AllocatorAssignmentProofSabotageForTest>,
    operation: impl FnOnce(SemanticsV2DurableAllocatorIndexProof<'_>, ReservedSemanticsV2Graph) -> T,
) -> T {
    assert!(
        identity.commit_sequence > 2,
        "test allocator proof needs a prior durable marker"
    );
    let marker_transaction = if identity.stable_transaction_id == 1 {
        2
    } else {
        1
    };
    let records: Vec<_> = leases
        .iter()
        .map(|lease| {
            assert!(
                lease.stable_allocator_id != 0
                    && lease.lease_start != 0
                    && lease.lease_start < lease.lease_end,
                "test allocator lease has an invalid exact interval"
            );
            SemanticsV2DurableAllocatorLeaseRecord {
                database_id: identity.database_id,
                allocator_kind: 1,
                stable_allocator_id: lease.stable_allocator_id,
                lease_epoch: 1,
                lease_start: lease.lease_start,
                lease_end: lease.lease_end,
                prior_high_water: lease.lease_start,
                new_high_water: lease.lease_end,
                marker_system_transaction_id: marker_transaction,
                marker_commit_sequence: 1,
            }
        })
        .collect();
    let mut assignments = assignment_records_for_test(identity, &graph, &records);
    let hidden_assignment_ordinal =
        apply_assignment_sabotage_for_test(identity, &mut assignments, sabotage);
    let selected: Vec<_> = records
        .iter()
        .enumerate()
        .map(
            |(ordinal, record)| SemanticsV2SelectedAllocatorLeaseWitness {
                record_ordinal: u32::try_from(ordinal).expect("test allocator record ordinal fits"),
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
            },
        )
        .collect();
    let selected_assignments: Vec<_> = assignments
        .iter()
        .enumerate()
        .filter(|(ordinal, _)| hidden_assignment_ordinal != Some(*ordinal))
        .map(
            |(ordinal, assignment)| SemanticsV2SelectedAllocatorAssignmentWitness {
                assignment_ordinal: u32::try_from(ordinal)
                    .expect("test allocator assignment ordinal fits"),
                database_id: assignment.database_id,
                cluster_id: assignment.cluster_id,
                timeline_id: assignment.timeline_id,
                format_epoch: assignment.format_epoch,
                leader_epoch: assignment.leader_epoch,
                parent_stable_transaction_id: assignment.parent_stable_transaction_id,
                parent_request_digest: assignment.parent_request_digest,
                parent_commit_sequence: assignment.parent_commit_sequence,
                parent_autocommit: assignment.parent_autocommit,
                parent_statement_ordinal: assignment.parent_statement_ordinal,
                parent_statement_request_digest: assignment.parent_statement_request_digest,
                parent_typed_statement_digest: assignment.parent_typed_statement_digest,
                lease_record_ordinal: assignment.lease_record_ordinal,
                allocator_kind: assignment.allocator_kind,
                stable_allocator_id: assignment.stable_allocator_id,
                lease_epoch: assignment.lease_epoch,
                lease_start: assignment.lease_start,
                lease_end: assignment.lease_end,
                mapping_version: assignment.mapping_version,
                source_order: assignment.source_order,
                source_row_ordinal: assignment.source_row_ordinal,
                assignment_start: assignment.assignment_start,
                assignment_end: assignment.assignment_end,
                marker_system_transaction_id: assignment.marker_system_transaction_id,
                marker_commit_sequence: assignment.marker_commit_sequence,
            },
        )
        .collect();
    let mut snapshot = SemanticsV2ImmutableDurableAllocatorIndex {
        database_id: identity.database_id,
        index_generation: 1,
        index_root: [0; 32],
        complete_next_commit_sequence: identity.commit_sequence + 1,
        durable_next_commit_sequence: identity.commit_sequence + 1,
        published_next_commit_sequence: identity.commit_sequence + 1,
        records: &records,
        assignments: &assignments,
    };
    snapshot.index_root = immutable_index_root(&snapshot);
    let pin = SemanticsV2PinnedAllocatorIndexGeneration {
        database_id: identity.database_id,
        cluster_id: identity.cluster_id,
        timeline_id: identity.timeline_id,
        format_epoch: identity.format_epoch,
        leader_epoch: identity.leader_epoch,
        index_generation: snapshot.index_generation,
        index_root: snapshot.index_root,
        retained_through_commit_sequence: identity.commit_sequence,
    };
    operation(
        SemanticsV2DurableAllocatorIndexProof {
            snapshot: &snapshot,
            checkpoint_pin: &pin,
            selected: &selected,
            selected_assignments: &selected_assignments,
        },
        graph,
    )
}

#[cfg(test)]
fn assignment_records_for_test(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    records: &[SemanticsV2DurableAllocatorLeaseRecord],
) -> Vec<SemanticsV2DurableAllocatorAssignmentRecord> {
    let mut assignments = Vec::new();
    for (table_ordinal, table) in graph.tables.iter().enumerate() {
        let lease = records
            .get(table_ordinal)
            .expect("test allocator lease has one selected record per S7 table");
        let mut source_order = 0_u64;
        for (statement_ordinal, resolution) in graph.resolutions.iter().enumerate() {
            if resolution.table_ref != table.table_ref {
                continue;
            }
            let statement = graph
                .statements
                .get(statement_ordinal)
                .expect("test assignment resolution has an S1 statement");
            let start = usize::try_from(resolution.s4_start)
                .expect("test S4 start fits host addressability");
            let count = usize::try_from(resolution.s4_count)
                .expect("test S4 count fits host addressability");
            let end = start
                .checked_add(count)
                .expect("test S4 range does not overflow");
            for disposition in &graph.dispositions[start..end] {
                let assignment_end = disposition
                    .stable_row_id
                    .checked_add(1)
                    .expect("test source row ID is not u64::MAX");
                assignments.push(SemanticsV2DurableAllocatorAssignmentRecord {
                    database_id: identity.database_id,
                    cluster_id: identity.cluster_id,
                    timeline_id: identity.timeline_id,
                    format_epoch: identity.format_epoch,
                    leader_epoch: identity.leader_epoch,
                    parent_stable_transaction_id: identity.stable_transaction_id,
                    parent_request_digest: identity.request_digest,
                    parent_commit_sequence: identity.commit_sequence,
                    parent_autocommit: identity.autocommit,
                    parent_statement_ordinal: u32::try_from(statement_ordinal)
                        .expect("test statement ordinal fits u32"),
                    parent_statement_request_digest: statement.request_digest,
                    parent_typed_statement_digest: statement.typed_statement_digest,
                    lease_record_ordinal: u32::try_from(table_ordinal)
                        .expect("test lease record ordinal fits u32"),
                    allocator_kind: lease.allocator_kind,
                    stable_allocator_id: table.stable_table_id,
                    lease_epoch: lease.lease_epoch,
                    lease_start: lease.lease_start,
                    lease_end: lease.lease_end,
                    mapping_version: 1,
                    source_order,
                    source_row_ordinal: disposition.source_row_ordinal,
                    assignment_start: disposition.stable_row_id,
                    assignment_end,
                    marker_system_transaction_id: lease.marker_system_transaction_id,
                    marker_commit_sequence: lease.marker_commit_sequence,
                });
                source_order = source_order
                    .checked_add(1)
                    .expect("test source order does not overflow");
            }
        }
    }
    assignments
}

#[cfg(test)]
fn apply_assignment_sabotage_for_test(
    identity: SemanticsV2BoundIdentity,
    assignments: &mut Vec<SemanticsV2DurableAllocatorAssignmentRecord>,
    sabotage: Option<AllocatorAssignmentProofSabotageForTest>,
) -> Option<usize> {
    let sabotage = sabotage?;
    match sabotage {
        AllocatorAssignmentProofSabotageForTest::Absent => {
            assignments
                .pop()
                .expect("assignment sabotage needs a source row");
        }
        AllocatorAssignmentProofSabotageForTest::Duplicate => {
            assignments.push(
                *assignments
                    .first()
                    .expect("assignment sabotage needs a source row"),
            );
        }
        AllocatorAssignmentProofSabotageForTest::WrongParent => {
            assignments[0].parent_stable_transaction_id = identity
                .stable_transaction_id
                .checked_add(1)
                .filter(|value| *value != u64::MAX)
                .unwrap_or(1);
        }
        AllocatorAssignmentProofSabotageForTest::HiddenConflictingParent => {
            let mut hidden = *assignments
                .first()
                .expect("assignment sabotage needs a source row");
            hidden.parent_request_digest = if hidden.parent_request_digest != [0xa5; 32] {
                [0xa5; 32]
            } else {
                [0x5a; 32]
            };
            assignments.push(hidden);
            return Some(assignments.len() - 1);
        }
        AllocatorAssignmentProofSabotageForTest::HiddenDifferentParentOverlap => {
            let mut hidden = *assignments
                .first()
                .expect("assignment sabotage needs a source row");
            hidden.parent_stable_transaction_id = identity
                .stable_transaction_id
                .checked_add(1)
                .filter(|value| *value != u64::MAX)
                .unwrap_or(1);
            hidden.parent_request_digest = if hidden.parent_request_digest != [0x4b; 32] {
                [0x4b; 32]
            } else {
                [0xb4; 32]
            };
            assignments.push(hidden);
            return Some(assignments.len() - 1);
        }
        AllocatorAssignmentProofSabotageForTest::StatementChain => {
            assignments[0].parent_typed_statement_digest[0] ^= 0x80;
        }
        AllocatorAssignmentProofSabotageForTest::Table => {
            assignments[0].stable_allocator_id ^= 1;
        }
        AllocatorAssignmentProofSabotageForTest::Order => {
            assignments[0].source_order = assignments[0]
                .source_order
                .checked_add(1)
                .expect("assignment source order sabotage does not overflow");
        }
        AllocatorAssignmentProofSabotageForTest::Range => {
            assignments[0].assignment_start = assignments[0]
                .assignment_start
                .checked_add(1)
                .expect("assignment range sabotage does not overflow");
        }
        AllocatorAssignmentProofSabotageForTest::Lineage => {
            assignments[0].cluster_id[0] ^= 0x80;
        }
        AllocatorAssignmentProofSabotageForTest::Lifecycle => {
            assignments[0].marker_commit_sequence = identity.commit_sequence;
        }
        AllocatorAssignmentProofSabotageForTest::MarkerIdentity => {
            assignments[0].marker_system_transaction_id = identity.stable_transaction_id;
        }
    }
    None
}

pub(super) fn validate_durable_allocator_index_identity(
    identity: SemanticsV2BoundIdentity,
    proof: &SemanticsV2DurableAllocatorIndexProof<'_>,
) -> Result<(), crate::EngineError> {
    validate_complete_index(identity, proof)?;
    validate_selected_marker_lifecycle(identity, proof)
}

fn validate_complete_index(
    identity: SemanticsV2BoundIdentity,
    proof: &SemanticsV2DurableAllocatorIndexProof<'_>,
) -> Result<(), crate::EngineError> {
    let snapshot = proof.snapshot;
    let pin = proof.checkpoint_pin;
    super::require(
        snapshot.database_id == identity.database_id
            && snapshot.index_generation != 0
            && snapshot.index_generation != u64::MAX
            && snapshot.index_root != [0; 32]
            && snapshot.complete_next_commit_sequence > identity.commit_sequence
            && snapshot.durable_next_commit_sequence > identity.commit_sequence
            && snapshot.published_next_commit_sequence > identity.commit_sequence
            && pin.database_id == identity.database_id
            && pin.cluster_id == identity.cluster_id
            && pin.timeline_id == identity.timeline_id
            && pin.format_epoch == identity.format_epoch
            && pin.leader_epoch == identity.leader_epoch
            && pin.leader_epoch != 0
            && pin.index_generation == snapshot.index_generation
            && pin.index_root == snapshot.index_root
            && pin.retained_through_commit_sequence != 0
            && immutable_index_root(snapshot) == snapshot.index_root,
        "durable allocator pinned index lacks one authenticated lineage/frontier",
    )?;

    for (record_ordinal, record) in snapshot.records.iter().enumerate() {
        validate_complete_record(identity, record)?;
        for earlier in &snapshot.records[..record_ordinal] {
            if earlier.marker_system_transaction_id == record.marker_system_transaction_id {
                super::require(
                    earlier.marker_commit_sequence == record.marker_commit_sequence,
                    "complete durable allocator index maps one system transaction to multiple commits",
                )?;
            }
            if earlier.stable_allocator_id == record.stable_allocator_id
                && earlier.lease_epoch == record.lease_epoch
            {
                super::require(
                    earlier.lease_end <= record.lease_start
                        || record.lease_end <= earlier.lease_start,
                    "complete durable allocator index has hidden allocator lease overlap",
                )?;
            }
        }
    }
    for (assignment_ordinal, assignment) in snapshot.assignments.iter().enumerate() {
        validate_complete_assignment_record(identity, snapshot, pin, assignment)?;
        if assignment_has_same_immutable_parent(identity, assignment) {
            super::require(
                assignment.parent_request_digest == identity.request_digest
                    && assignment.parent_commit_sequence == identity.commit_sequence
                    && assignment.parent_autocommit == identity.autocommit,
                "complete durable allocator index has a stable-parent root allocator assignment with a conflicting mutable parent tuple",
            )?;
            let assignment_ordinal = u32::try_from(assignment_ordinal).map_err(|_| {
                super::validation_error("durable allocator assignment ordinal exceeds u32")
            })?;
            super::require(
                proof
                    .selected_assignments
                    .iter()
                    .any(|selected| selected.assignment_ordinal == assignment_ordinal),
                "complete durable allocator index leaves a stable-parent root assignment unselected",
            )?;
        }
        for earlier in &snapshot.assignments[..assignment_ordinal] {
            if earlier.stable_allocator_id == assignment.stable_allocator_id
                && earlier.lease_epoch == assignment.lease_epoch
            {
                super::require(
                    earlier.assignment_end <= assignment.assignment_start
                        || assignment.assignment_end <= earlier.assignment_start,
                    "complete durable allocator index has overlapping root allocator assignment stable row IDs",
                )?;
            }
        }
    }
    Ok(())
}

fn assignment_has_same_immutable_parent(
    identity: SemanticsV2BoundIdentity,
    assignment: &SemanticsV2DurableAllocatorAssignmentRecord,
) -> bool {
    assignment.database_id == identity.database_id
        && assignment.cluster_id == identity.cluster_id
        && assignment.timeline_id == identity.timeline_id
        && assignment.format_epoch == identity.format_epoch
        && assignment.leader_epoch == identity.leader_epoch
        && assignment.parent_stable_transaction_id == identity.stable_transaction_id
}

fn validate_complete_record(
    identity: SemanticsV2BoundIdentity,
    record: &SemanticsV2DurableAllocatorLeaseRecord,
) -> Result<(), crate::EngineError> {
    super::require(
        record.database_id == identity.database_id
            && record.allocator_kind == 1
            && record.stable_allocator_id != 0
            && record.stable_allocator_id != u64::MAX
            && record.lease_epoch != 0
            && record.lease_epoch != u64::MAX
            && record.lease_start != 0
            && record.lease_start != u64::MAX
            && record.lease_end != 0
            && record.lease_end != u64::MAX
            && record.prior_high_water <= record.lease_start
            && record.lease_start < record.lease_end
            && record.new_high_water == record.lease_end
            && record.marker_system_transaction_id != 0
            && record.marker_system_transaction_id != u64::MAX
            && record.marker_commit_sequence != 0
            && record.marker_commit_sequence != u64::MAX,
        "complete durable allocator index record has an invalid immutable shape",
    )
}

fn validate_complete_assignment_record(
    identity: SemanticsV2BoundIdentity,
    snapshot: &SemanticsV2ImmutableDurableAllocatorIndex<'_>,
    pin: &SemanticsV2PinnedAllocatorIndexGeneration,
    assignment: &SemanticsV2DurableAllocatorAssignmentRecord,
) -> Result<(), crate::EngineError> {
    let lease = snapshot
        .records
        .get(
            usize::try_from(assignment.lease_record_ordinal).map_err(|_| {
                super::validation_error(
                    "durable allocator assignment lease ordinal is unaddressable",
                )
            })?,
        )
        .ok_or_else(|| super::validation_error("durable allocator assignment lease is absent"))?;
    let exact_assignment_end = assignment.assignment_start.checked_add(1).ok_or_else(|| {
        super::validation_error("complete durable allocator assignment reaches u64::MAX")
    })?;
    super::require(
        assignment.database_id == identity.database_id
            && assignment.cluster_id == pin.cluster_id
            && assignment.timeline_id == pin.timeline_id
            && assignment.format_epoch == pin.format_epoch
            && assignment.leader_epoch == pin.leader_epoch
            && assignment.parent_stable_transaction_id != 0
            && assignment.parent_stable_transaction_id != u64::MAX
            && assignment.parent_request_digest != [0; 32]
            && assignment.parent_commit_sequence != 0
            && assignment.parent_commit_sequence != u64::MAX
            && assignment.parent_statement_ordinal != u32::MAX
            && assignment.parent_statement_request_digest != [0; 32]
            && assignment.parent_typed_statement_digest != [0; 32]
            && assignment.allocator_kind == lease.allocator_kind
            && assignment.stable_allocator_id == lease.stable_allocator_id
            && assignment.lease_epoch == lease.lease_epoch
            && assignment.lease_start == lease.lease_start
            && assignment.lease_end == lease.lease_end
            && assignment.mapping_version == 1
            && assignment.assignment_start != 0
            && assignment.assignment_start != u64::MAX
            && assignment.assignment_end != 0
            && assignment.assignment_end != u64::MAX
            && assignment.assignment_end == exact_assignment_end
            && assignment.assignment_start >= lease.lease_start
            && assignment.assignment_end <= lease.lease_end
            && assignment.marker_system_transaction_id != 0
            && assignment.marker_system_transaction_id != u64::MAX
            && assignment.marker_commit_sequence != 0
            && assignment.marker_commit_sequence != u64::MAX,
        "complete durable allocator assignment has an invalid immutable shape",
    )?;
    super::require(
        assignment.marker_system_transaction_id == lease.marker_system_transaction_id
            && assignment.marker_commit_sequence == lease.marker_commit_sequence,
        "complete durable allocator assignment does not share its durable lease marker identity",
    )
}

fn validate_selected_marker_lifecycle(
    identity: SemanticsV2BoundIdentity,
    proof: &SemanticsV2DurableAllocatorIndexProof<'_>,
) -> Result<(), crate::EngineError> {
    for (selected_ordinal, selected) in proof.selected.iter().enumerate() {
        let record = proof
            .snapshot
            .records
            .get(usize::try_from(selected.record_ordinal).map_err(|_| {
                super::validation_error(
                    "selected durable allocator record ordinal is unaddressable",
                )
            })?)
            .ok_or_else(|| {
                super::validation_error("selected durable allocator record is absent")
            })?;
        super::require(
            selected_matches_record(selected, record)
                && selected.marker_system_transaction_id != identity.stable_transaction_id
                && selected.marker_commit_sequence < identity.commit_sequence
                && selected.marker_commit_sequence < proof.snapshot.complete_next_commit_sequence
                && selected.marker_commit_sequence < proof.snapshot.durable_next_commit_sequence
                && selected.marker_commit_sequence < proof.snapshot.published_next_commit_sequence
                && selected.marker_commit_sequence
                    <= proof.checkpoint_pin.retained_through_commit_sequence,
            "selected durable allocator witness lacks exact member, lifecycle, or checkpoint evidence",
        )?;
        for earlier in &proof.selected[..selected_ordinal] {
            super::require(
                earlier.record_ordinal != selected.record_ordinal,
                "selected durable allocator witness duplicates an authoritative record",
            )?;
        }
    }
    validate_selected_assignment_lifecycle(identity, proof)
}

fn validate_selected_assignment_lifecycle(
    identity: SemanticsV2BoundIdentity,
    proof: &SemanticsV2DurableAllocatorIndexProof<'_>,
) -> Result<(), crate::EngineError> {
    for (selected_ordinal, selected) in proof.selected_assignments.iter().enumerate() {
        let assignment = proof
            .snapshot
            .assignments
            .get(usize::try_from(selected.assignment_ordinal).map_err(|_| {
                super::validation_error("selected allocator assignment ordinal is unaddressable")
            })?)
            .ok_or_else(|| super::validation_error("selected allocator assignment is absent"))?;
        let lease = proof
            .snapshot
            .records
            .get(
                usize::try_from(assignment.lease_record_ordinal).map_err(|_| {
                    super::validation_error(
                        "selected allocator assignment lease ordinal is unaddressable",
                    )
                })?,
            )
            .ok_or_else(|| {
                super::validation_error("selected allocator assignment lease is absent")
            })?;
        let selected_lease_count = proof
            .selected
            .iter()
            .filter(|candidate| candidate.record_ordinal == assignment.lease_record_ordinal)
            .count();
        super::require(
            selected_assignment_matches_record(selected, assignment)
                && selected_lease_count == 1
                && selected_matches_record(
                    proof
                        .selected
                        .iter()
                        .find(|candidate| {
                            candidate.record_ordinal == assignment.lease_record_ordinal
                        })
                        .expect("selected lease count checked"),
                    lease,
                )
                && assignment.parent_stable_transaction_id == identity.stable_transaction_id
                && assignment.parent_request_digest == identity.request_digest
                && assignment.parent_commit_sequence == identity.commit_sequence
                && assignment.parent_autocommit == identity.autocommit
                && assignment.marker_system_transaction_id == lease.marker_system_transaction_id
                && assignment.marker_commit_sequence == lease.marker_commit_sequence
                && assignment.marker_system_transaction_id != identity.stable_transaction_id
                && assignment.marker_commit_sequence < identity.commit_sequence
                && assignment.marker_commit_sequence < proof.snapshot.complete_next_commit_sequence
                && assignment.marker_commit_sequence < proof.snapshot.durable_next_commit_sequence
                && assignment.marker_commit_sequence
                    < proof.snapshot.published_next_commit_sequence
                && assignment.marker_commit_sequence
                    <= proof.checkpoint_pin.retained_through_commit_sequence,
            "selected durable allocator assignment lacks exact parent, lease, lifecycle, or checkpoint evidence",
        )?;
        for earlier in &proof.selected_assignments[..selected_ordinal] {
            super::require(
                earlier.assignment_ordinal != selected.assignment_ordinal,
                "selected durable allocator assignment duplicates an authoritative record",
            )?;
        }
    }
    Ok(())
}

fn selected_matches_record(
    selected: &SemanticsV2SelectedAllocatorLeaseWitness,
    record: &SemanticsV2DurableAllocatorLeaseRecord,
) -> bool {
    selected.database_id == record.database_id
        && selected.allocator_kind == record.allocator_kind
        && selected.stable_allocator_id == record.stable_allocator_id
        && selected.lease_epoch == record.lease_epoch
        && selected.lease_start == record.lease_start
        && selected.lease_end == record.lease_end
        && selected.prior_high_water == record.prior_high_water
        && selected.new_high_water == record.new_high_water
        && selected.marker_system_transaction_id == record.marker_system_transaction_id
        && selected.marker_commit_sequence == record.marker_commit_sequence
}

fn selected_assignment_matches_record(
    selected: &SemanticsV2SelectedAllocatorAssignmentWitness,
    assignment: &SemanticsV2DurableAllocatorAssignmentRecord,
) -> bool {
    selected.database_id == assignment.database_id
        && selected.cluster_id == assignment.cluster_id
        && selected.timeline_id == assignment.timeline_id
        && selected.format_epoch == assignment.format_epoch
        && selected.leader_epoch == assignment.leader_epoch
        && selected.parent_stable_transaction_id == assignment.parent_stable_transaction_id
        && selected.parent_request_digest == assignment.parent_request_digest
        && selected.parent_commit_sequence == assignment.parent_commit_sequence
        && selected.parent_autocommit == assignment.parent_autocommit
        && selected.parent_statement_ordinal == assignment.parent_statement_ordinal
        && selected.parent_statement_request_digest == assignment.parent_statement_request_digest
        && selected.parent_typed_statement_digest == assignment.parent_typed_statement_digest
        && selected.lease_record_ordinal == assignment.lease_record_ordinal
        && selected.allocator_kind == assignment.allocator_kind
        && selected.stable_allocator_id == assignment.stable_allocator_id
        && selected.lease_epoch == assignment.lease_epoch
        && selected.lease_start == assignment.lease_start
        && selected.lease_end == assignment.lease_end
        && selected.mapping_version == assignment.mapping_version
        && selected.source_order == assignment.source_order
        && selected.source_row_ordinal == assignment.source_row_ordinal
        && selected.assignment_start == assignment.assignment_start
        && selected.assignment_end == assignment.assignment_end
        && selected.marker_system_transaction_id == assignment.marker_system_transaction_id
        && selected.marker_commit_sequence == assignment.marker_commit_sequence
}

fn immutable_index_root(snapshot: &SemanticsV2ImmutableDurableAllocatorIndex<'_>) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"gpu-db/write001/durable-allocator-index/v1");
    digest.update(snapshot.database_id);
    digest.update(snapshot.index_generation.to_le_bytes());
    digest.update(snapshot.complete_next_commit_sequence.to_le_bytes());
    digest.update(snapshot.durable_next_commit_sequence.to_le_bytes());
    digest.update(snapshot.published_next_commit_sequence.to_le_bytes());
    digest.update(
        u64::try_from(snapshot.records.len())
            .expect("durable allocator index record count fits u64")
            .to_le_bytes(),
    );
    for record in snapshot.records {
        digest.update(record.database_id);
        digest.update([record.allocator_kind]);
        digest.update(record.stable_allocator_id.to_le_bytes());
        digest.update(record.lease_epoch.to_le_bytes());
        digest.update(record.lease_start.to_le_bytes());
        digest.update(record.lease_end.to_le_bytes());
        digest.update(record.prior_high_water.to_le_bytes());
        digest.update(record.new_high_water.to_le_bytes());
        digest.update(record.marker_system_transaction_id.to_le_bytes());
        digest.update(record.marker_commit_sequence.to_le_bytes());
    }
    digest.update(
        u64::try_from(snapshot.assignments.len())
            .expect("durable allocator index assignment count fits u64")
            .to_le_bytes(),
    );
    for assignment in snapshot.assignments {
        digest.update(assignment.database_id);
        digest.update(assignment.cluster_id);
        digest.update(assignment.timeline_id);
        digest.update(assignment.format_epoch.to_le_bytes());
        digest.update(assignment.leader_epoch.to_le_bytes());
        digest.update(assignment.parent_stable_transaction_id.to_le_bytes());
        digest.update(assignment.parent_request_digest);
        digest.update(assignment.parent_commit_sequence.to_le_bytes());
        digest.update([u8::from(assignment.parent_autocommit)]);
        digest.update(assignment.parent_statement_ordinal.to_le_bytes());
        digest.update(assignment.parent_statement_request_digest);
        digest.update(assignment.parent_typed_statement_digest);
        digest.update(assignment.lease_record_ordinal.to_le_bytes());
        digest.update([assignment.allocator_kind]);
        digest.update(assignment.stable_allocator_id.to_le_bytes());
        digest.update(assignment.lease_epoch.to_le_bytes());
        digest.update(assignment.lease_start.to_le_bytes());
        digest.update(assignment.lease_end.to_le_bytes());
        digest.update(assignment.mapping_version.to_le_bytes());
        digest.update(assignment.source_order.to_le_bytes());
        digest.update(assignment.source_row_ordinal.to_le_bytes());
        digest.update(assignment.assignment_start.to_le_bytes());
        digest.update(assignment.assignment_end.to_le_bytes());
        digest.update(assignment.marker_system_transaction_id.to_le_bytes());
        digest.update(assignment.marker_commit_sequence.to_le_bytes());
    }
    digest.finalize().into()
}

/// Match the explicit selected witnesses, in stable allocator/table order, to the complete
/// immutable index.  Unrelated index records at any epoch remain valid evidence and are never
/// selected by a heuristic scan.
pub(super) fn validate_allocator_closure<'a>(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
    proof: &SemanticsV2DurableAllocatorIndexProof<'a>,
) -> Result<ValidatedAllocatorAssignment<'a>, crate::EngineError> {
    validate_durable_allocator_index_identity(identity, proof)?;
    let mut prior_table_id = None;
    for selected in proof.selected {
        super::require(
            prior_table_id.is_none_or(|prior| prior < selected.stable_allocator_id)
                && selected.lease_start < selected.lease_end,
            "selected durable allocator lease order is invalid",
        )?;
        prior_table_id = Some(selected.stable_allocator_id);
    }
    validate_exact_assignment_closure(graph, catalog, proof)
}

fn validate_exact_assignment_closure<'a>(
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
    proof: &SemanticsV2DurableAllocatorIndexProof<'a>,
) -> Result<ValidatedAllocatorAssignment<'a>, crate::EngineError> {
    super::require(
        graph.statements.len() == graph.records.len(),
        "durable allocator assignment has unequal S1/S2 statement cardinality",
    )?;
    for (statement_ordinal, record) in graph.records.iter().enumerate() {
        let selected_count = proof
            .selected
            .iter()
            .try_fold(0_u32, |count, selected_lease| {
                if selected_catalog_target_matches(record, catalog, selected_lease)? {
                    count.checked_add(1).ok_or_else(|| {
                        super::validation_error(
                            "allocator assignment selected-table match count overflows",
                        )
                    })
                } else {
                    Ok(count)
                }
            })?;
        super::require(
            selected_count == 1
                && record.facts().statement_ordinal.as_u32()
                    == u32::try_from(statement_ordinal).map_err(|_| {
                        super::validation_error(
                            "allocator assignment statement ordinal exceeds u32",
                        )
                    })?,
            "durable allocator assignment has an unselected or ambiguous S1/S2 target table",
        )?;
    }
    let mut selected_assignment_ordinal = 0_usize;
    for selected_lease in proof.selected {
        let mut source_order = 0_u64;
        let mut previous_assignment_end = None;
        let mut has_selected_source = false;
        for (statement_ordinal, statement) in graph.statements.iter().enumerate() {
            let record = graph.records.get(statement_ordinal).ok_or_else(|| {
                super::validation_error("allocator assignment S2 record is absent")
            })?;
            let source_statement_ordinal = u32::try_from(statement_ordinal).map_err(|_| {
                super::validation_error("allocator assignment statement ordinal exceeds u32")
            })?;
            super::require(
                statement.statement_ordinal == source_statement_ordinal
                    && record.facts().statement_ordinal.as_u32() == source_statement_ordinal
                    && record.facts().row_count == statement.input_row_count
                    && record.facts().typed_statement_digest == statement.typed_statement_digest,
                "durable allocator assignment cannot close its exact S1/S2 source rows",
            )?;
            if !selected_catalog_target_matches(record, catalog, selected_lease)? {
                continue;
            }
            has_selected_source = true;
            for source_row_ordinal in 0..statement.input_row_count {
                let assignment = proof
                    .selected_assignments
                    .get(selected_assignment_ordinal)
                    .ok_or_else(|| {
                        super::validation_error(
                            "durable allocator assignment is missing an S1/S2 source row",
                        )
                    })?;
                let expected_assignment_end =
                    assignment.assignment_start.checked_add(1).ok_or_else(|| {
                        super::validation_error("allocator assignment source row reaches u64::MAX")
                    })?;
                super::require(
                    assignment.lease_record_ordinal == selected_lease.record_ordinal
                        && assignment.stable_allocator_id == selected_lease.stable_allocator_id
                        && assignment.parent_statement_ordinal == source_statement_ordinal
                        && assignment.parent_statement_request_digest == statement.request_digest
                        && assignment.parent_typed_statement_digest
                            == statement.typed_statement_digest
                        && assignment.source_order == source_order
                        && assignment.source_row_ordinal == source_row_ordinal
                        && assignment.assignment_end == expected_assignment_end
                        && assignment.assignment_start >= selected_lease.lease_start
                        && assignment.assignment_end <= selected_lease.lease_end
                        && previous_assignment_end
                            .is_none_or(|previous| previous == assignment.assignment_start),
                    "durable allocator assignment does not exactly bind stable-table source order to its row ID",
                )?;
                previous_assignment_end = Some(assignment.assignment_end);
                source_order = source_order.checked_add(1).ok_or_else(|| {
                    super::validation_error("allocator assignment source order overflows")
                })?;
                selected_assignment_ordinal =
                    selected_assignment_ordinal.checked_add(1).ok_or_else(|| {
                        super::validation_error("allocator assignment selection ordinal overflows")
                    })?;
            }
        }
        super::require(
            has_selected_source && source_order != 0,
            "durable allocator assignment selected lease has no exact S1/S2 source rows",
        )?;
    }
    super::require(
        selected_assignment_ordinal == proof.selected_assignments.len(),
        "durable allocator assignment selection has duplicate or unordered source rows",
    )?;
    Ok(ValidatedAllocatorAssignment {
        selected: proof.selected_assignments,
        seal: AllocatorAssignmentSeal,
    })
}

fn selected_catalog_target_matches(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    catalog: &SemanticsV2CatalogWitness<'_>,
    selected_lease: &SemanticsV2SelectedAllocatorLeaseWitness,
) -> Result<bool, crate::EngineError> {
    let mut matched = None;
    let mut count = 0_u32;
    for table in catalog.tables {
        if table.stable_table_id == selected_lease.stable_allocator_id {
            count = count.checked_add(1).ok_or_else(|| {
                super::validation_error("allocator assignment pinned-table match count overflows")
            })?;
            matched = Some(table);
        }
    }
    super::require(
        count == 1,
        "durable allocator assignment lease does not select one pinned target table",
    )?;
    let table = matched.expect("pinned table count checked");
    let target = record.target_identity();
    Ok(target.oid == table.display_oid
        && target.schema == table.schema
        && target.name == table.name
        && target.schema_digest == table.schema_digest)
}

pub(super) fn validate_table_order(
    rows: &[SemanticsV2CatalogTableWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    for row in rows {
        super::require_stable_identity(row.stable_table_id, row.display_oid, "catalog table")?;
        super::require(
            super::valid_identifier(row.schema) && super::valid_identifier(row.name),
            "catalog table name is not a resolved identifier",
        )?;
        super::require(
            previous.is_none_or(|prior| prior < row.stable_table_id),
            "catalog tables are not in strict stable-ID order",
        )?;
        previous = Some(row.stable_table_id);
        validate_column_order(row.catalog_columns)?;
        super::guards::validate_guard_list_order(
            row.not_null_guards,
            super::NOT_NULL_GUARD,
            "NOT NULL",
        )?;
        super::guards::validate_guard_list_order(row.check_guards, super::CHECK_GUARD, "CHECK")?;
        validate_foreign_key_order(row.foreign_keys)?;
    }
    Ok(())
}

pub(super) fn validate_table_closure(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for table in &graph.tables {
        let mut matched = None;
        let mut count = 0_u32;
        for candidate in catalog.tables {
            if candidate.stable_table_id == table.stable_table_id
                && candidate.display_oid == table.display_oid
            {
                count = count.checked_add(1).ok_or_else(|| {
                    super::validation_error("catalog target-table match count overflows")
                })?;
                matched = Some(candidate);
            }
        }
        let Some(candidate) = matched else {
            return Err(super::validation_error(
                "S7 target table is absent from the pinned catalog",
            ));
        };
        super::require(
            count == 1,
            "S7 target table is ambiguous in the pinned catalog",
        )?;
        validate_target_table(identity, graph, catalog, table, candidate)?;
    }

    for dependency in &graph.dependencies {
        if matches!(
            dependency.kind,
            super::TARGET_TABLE | super::FOREIGN_PARENT_TABLE
        ) {
            let matches = catalog
                .tables
                .iter()
                .filter(|table| {
                    table.stable_table_id == dependency.stable_object_id
                        && table.display_oid == dependency.display_oid
                })
                .count();
            super::require(
                matches == 1,
                "table dependency does not select exactly one pinned catalog table",
            )?;
            if dependency.kind == super::FOREIGN_PARENT_TABLE {
                let table = catalog
                    .tables
                    .iter()
                    .find(|table| {
                        table.stable_table_id == dependency.stable_object_id
                            && table.display_oid == dependency.display_oid
                    })
                    .expect("count checked");
                validate_foreign_parent_table(identity, graph, dependency, table)?;
            }
        }
    }

    for table in catalog.tables {
        let is_target = graph.tables.iter().any(|retained| {
            retained.stable_table_id == table.stable_table_id
                && retained.display_oid == table.display_oid
        });
        let is_parent = graph.dependencies.iter().any(|dependency| {
            dependency.kind == super::FOREIGN_PARENT_TABLE
                && dependency.stable_object_id == table.stable_table_id
                && dependency.display_oid == table.display_oid
        });
        super::require(
            is_target || is_parent,
            "pinned catalog contains a table outside the exact S2 dependency closure",
        )?;
        if is_parent {
            validate_parent_table_columns(graph, table)?;
        }
    }
    Ok(())
}

fn validate_target_table(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    all_catalog: &SemanticsV2CatalogWitness<'_>,
    retained: &RetainedTable,
    catalog: &SemanticsV2CatalogTableWitness<'_>,
) -> Result<(), crate::EngineError> {
    super::require(
        catalog.data_generation == retained.data_generation_before
            && catalog.data_root == retained.initial_table_root
            && catalog.schema_digest == retained.schema_digest
            && retained.catalog_epoch == identity.catalog_epoch
            && retained.data_generation_before != 0
            && retained.initial_table_root != [0; 32]
            && usize::try_from(retained.catalog_column_count)
                .ok()
                .is_some_and(|count| count == catalog.catalog_columns.len()),
        "pinned target table does not match its S7 initial identity",
    )?;

    let mut resolution_count = 0_u32;
    for resolution in graph
        .resolutions
        .iter()
        .filter(|resolution| resolution.table_ref == retained.table_ref)
    {
        resolution_count = resolution_count
            .checked_add(1)
            .ok_or_else(|| super::validation_error("target-table resolution count overflows"))?;
        let record = graph
            .records
            .get(usize::try_from(resolution.record_ref).map_err(|_| {
                super::validation_error("S7 record reference exceeds host addressability")
            })?)
            .ok_or_else(|| super::validation_error("S7 target-table record is absent"))?;
        let target = record.target_identity();
        super::require(
            target.oid == catalog.display_oid
                && target.schema == catalog.schema
                && target.name == catalog.name
                && target.schema_digest == catalog.schema_digest,
            "S2 target identity does not match its pinned catalog table",
        )?;
        validate_target_columns(record, catalog.catalog_columns)?;
        super::guards::validate_target_foreign_keys(graph, all_catalog, retained, record, catalog)?;
    }
    super::require(
        resolution_count != 0,
        "S7 target table has no S2 statement resolution",
    )
}

fn validate_target_columns(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    columns: &[SemanticsV2CatalogColumnWitness<'_>],
) -> Result<(), crate::EngineError> {
    super::require(
        record.catalog_columns().len() == columns.len(),
        "S2 target columns do not exactly close the pinned catalog columns",
    )?;
    for (source, catalog) in record.catalog_columns().zip(columns) {
        super::require(
            super::catalog_column_matches_source(catalog, source),
            "S2 target column differs from its pinned catalog column",
        )?;
    }
    Ok(())
}

fn validate_foreign_parent_table(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    dependency: &RetainedDependencyToken,
    catalog: &SemanticsV2CatalogTableWitness<'_>,
) -> Result<(), crate::EngineError> {
    super::require(
        dependency.catalog_epoch == identity.catalog_epoch
            && dependency.base_generation == catalog.data_generation
            && dependency.base_root == catalog.data_root
            && dependency.schema_digest == catalog.schema_digest
            && dependency.name_digest
                == super::qualified_name_digest(catalog.schema, catalog.name)?,
        "foreign-parent table token does not match the pinned catalog identity",
    )?;
    let mut source_count = 0_u32;
    for usage in graph.dependency_uses.iter().filter(|usage| {
        usage.dependency_ref == dependency.dependency_ref
            && usage.role == super::FOREIGN_PARENT_TABLE_ROLE
    }) {
        let source =
            source_dependency_for_use(graph, usage.statement_ordinal, usage.source_ordinal)?;
        source_count = source_count
            .checked_add(1)
            .ok_or_else(|| super::validation_error("foreign-parent source count overflows"))?;
        super::require(
            source.oid == catalog.display_oid
                && source.schema == catalog.schema
                && source.name == catalog.name
                && source.schema_digest == catalog.schema_digest,
            "S2 foreign-parent dependency does not match its pinned catalog table",
        )?;
    }
    super::require(
        source_count != 0,
        "foreign-parent table token has no exact S2 dependency use",
    )
}

fn validate_parent_table_columns(
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogTableWitness<'_>,
) -> Result<(), crate::EngineError> {
    for record in &graph.records {
        for foreign_key in record.foreign_keys() {
            let dependency = record
                .dependencies()
                .nth(
                    usize::try_from(foreign_key.parent_dependency_ordinal).map_err(|_| {
                        super::validation_error(
                            "S2 foreign-parent dependency ordinal exceeds addressability",
                        )
                    })?,
                )
                .ok_or_else(|| super::validation_error("S2 foreign-parent dependency is absent"))?;
            if dependency.oid == catalog.display_oid
                && dependency.schema == catalog.schema
                && dependency.name == catalog.name
            {
                super::require(
                    catalog.catalog_columns.iter().any(|column| {
                        super::catalog_column_matches_binding(column, foreign_key.parent_column)
                    }),
                    "S2 FK parent column is absent from its pinned catalog table",
                )?;
                for key in record.foreign_key_supporting_index_keys(foreign_key.raw_ordinal)? {
                    super::require(
                        catalog
                            .catalog_columns
                            .iter()
                            .any(|column| super::catalog_column_matches_binding(column, key)),
                        "S2 FK supporting-index key is absent from its parent catalog table",
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn source_dependency_for_use(
    graph: &ReservedSemanticsV2Graph,
    statement_ordinal: u32,
    source_ordinal: u32,
) -> Result<DecodedDependencyFacts<'_>, crate::EngineError> {
    let resolution = graph
        .resolutions
        .iter()
        .find(|resolution| resolution.statement_ordinal == statement_ordinal)
        .ok_or_else(|| super::validation_error("dependency use has no statement resolution"))?;
    let record = graph
        .records
        .get(usize::try_from(resolution.record_ref).map_err(|_| {
            super::validation_error("dependency-use record reference exceeds addressability")
        })?)
        .ok_or_else(|| super::validation_error("dependency-use record is absent"))?;
    record
        .dependencies()
        .nth(usize::try_from(source_ordinal).map_err(|_| {
            super::validation_error("dependency-use source ordinal exceeds addressability")
        })?)
        .ok_or_else(|| super::validation_error("dependency-use source dependency is absent"))
}

fn validate_column_order(
    rows: &[SemanticsV2CatalogColumnWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut expected = 0_u32;
    for row in rows {
        super::require(
            row.catalog_column_ordinal == expected
                && row.stable_column_id != 0
                && row.attnum != 0
                && super::valid_identifier(row.name)
                && row.column_shape_digest != [0; 32]
                && row.column_root != [0; 32],
            "catalog columns are not dense, resolved, and complete",
        )?;
        expected = expected
            .checked_add(1)
            .ok_or_else(|| super::validation_error("catalog column ordinal overflows"))?;
    }
    Ok(())
}

fn validate_foreign_key_order(
    rows: &[super::super::SemanticsV2CatalogForeignKeyWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut expected = 0_u32;
    for row in rows {
        super::require(
            row.raw_foreign_key_ordinal == expected
                && row.stable_constraint_id != 0
                && row.stable_constraint_id != u64::MAX
                && row.display_oid != 0
                && row.display_oid <= 0x7fff_ffff
                && super::valid_identifier(row.schema)
                && super::valid_identifier(row.name)
                && row.child_stable_column_id != 0
                && row.parent_stable_table_id != 0
                && row.parent_stable_column_id != 0
                && row.supporting_stable_index_id != 0,
            "catalog foreign keys are not dense and complete",
        )?;
        expected = expected
            .checked_add(1)
            .ok_or_else(|| super::validation_error("catalog FK ordinal overflows"))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "allocator_tests.rs"]
mod allocator_tests;
