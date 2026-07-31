//! Exact retained-owner reservation for the private semantics-v2 graph.
//!
//! This is deliberately a capacity reservation, not a semantic decoder.  It consumes the
//! allocation-free pass-zero measure, reserves every graph directory before an allocating source
//! decode can begin, and carries the exact expected lengths for the later strict fill.  Nested S2
//! records and images retain their own typed vectors/names/text through their existing measured
//! decoder seams; their measured ABI terms are included here but their construction remains with
//! those move-only owners.

use super::super::pass_zero::SemanticsV2StructuralMeasure;
#[cfg(test)]
use super::super::pass_zero::{empty_s8_measure_for_test, nonempty_s8_measure_for_test};
use super::graph::{
    ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedDisposition,
    RetainedIndexDescriptor, RetainedIndexKeyColumn, RetainedKeyComponent, RetainedKeyEffect,
    RetainedProjectionBinding, RetainedResponseArtifact, RetainedResponseEnvelope,
    RetainedResponseEnvelopeIdentity, RetainedResponseGraph, RetainedResponseSelection,
    RetainedSequenceEffect, RetainedStatement, RetainedStatementDependencyUse,
    RetainedStatementOutcome, RetainedStatementResolution, RetainedTable, RetainedTableDisposition,
    RetainedTransition,
};
use crate::typed_insert_batch::{DecodedTypedImage, DecodedTypedInsertRecord};
use crate::EngineError;

/// The exact owner budget established before any retained source/image decode.  It separates
/// persistent graph ownership from sequential raw-copy/decode scratch, whose maximum is already
/// proven by pass zero and never contributes to published retained state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RetainedSemanticsV2ReservationBudget {
    persistent_bytes: u64,
    persistent_allocation_slots: u64,
    maximum_scratch_bytes: u64,
    maximum_scratch_allocation_slots: u64,
}

/// Non-forgeable (outside this module) exact capacities for every direct retained graph owner.
/// It consumes `SemanticsV2StructuralMeasure`, preventing a later phase from sizing itself from
/// hostile wire counts or raw arena lengths.
pub(super) struct RetainedSemanticsV2Reservation {
    expected: RetainedSemanticsV2ExpectedGraph,
    header: super::super::pass_zero::SemanticsV2S7HeaderIdentity,
    source_measure: RetainedSemanticsV2SourceMeasure,
    budget: RetainedSemanticsV2ReservationBudget,
    direct_persistent_bytes: u64,
    direct_persistent_allocation_slots: u64,
}

/// The source/image ABI established by pass zero. Strict fill recomputes these terms while it
/// decodes into the already-reserved graph, refusing any shape drift before release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RetainedSemanticsV2SourceMeasure {
    pub(super) s2_persistent_bytes: u64,
    pub(super) s2_persistent_slots: u64,
    pub(super) image_persistent_bytes: u64,
    pub(super) image_persistent_slots: u64,
    pub(super) response_image_persistent_bytes: u64,
    pub(super) response_image_persistent_slots: u64,
    pub(super) maximum_scratch_bytes: u64,
    pub(super) maximum_scratch_slots: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RetainedSemanticsV2ExpectedGraph {
    statements: usize,
    records: usize,
    dispositions: usize,
    sequence_effects: usize,
    outcomes: usize,
    tables: usize,
    table_dispositions: usize,
    resolutions: usize,
    dependencies: usize,
    dependency_uses: usize,
    indexes: usize,
    index_key_columns: usize,
    transitions: usize,
    key_effects: usize,
    key_components: usize,
    projections: usize,
    images: usize,
    response_artifacts: usize,
    response_selections: usize,
    response_images: usize,
    response_identity: RetainedResponseEnvelopeIdentity,
}

/// Reserved direct graph capacity plus its consumed exact measure.  No field is exposed outside
/// this private retained module, so a future strict decoder must consume this owner to fill all
/// directories and cannot substitute a separately-sized graph.
pub(super) struct ReservedSemanticsV2GraphOwner {
    graph: ReservedSemanticsV2Graph,
    expected: RetainedSemanticsV2ExpectedGraph,
    source_measure: RetainedSemanticsV2SourceMeasure,
    budget: RetainedSemanticsV2ReservationBudget,
}

impl RetainedSemanticsV2Reservation {
    pub(super) fn from_structural(
        structural: SemanticsV2StructuralMeasure,
    ) -> Result<Self, EngineError> {
        let expected = RetainedSemanticsV2ExpectedGraph {
            statements: exact_count(structural.s1_statement_count, "S1 statement count")?,
            records: exact_count(structural.s2_record_count, "S2 record count")?,
            dispositions: exact_count(structural.s4_disposition_count, "S4 disposition count")?,
            sequence_effects: exact_count(structural.s5_effect_count, "S5 effect count")?,
            outcomes: exact_count(structural.s6_outcome_count, "S6 outcome count")?,
            tables: exact_count(structural.s7_directory_counts[0], "S7 table count")?,
            table_dispositions: exact_count(
                structural.s7_directory_counts[2],
                "S7 table-disposition count",
            )?,
            resolutions: exact_count(structural.s7_directory_counts[1], "S7 resolution count")?,
            dependencies: exact_count(structural.s7_directory_counts[3], "S7 dependency count")?,
            dependency_uses: exact_count(
                structural.s7_directory_counts[4],
                "S7 dependency-use count",
            )?,
            indexes: exact_count(structural.s7_directory_counts[5], "S7 index count")?,
            index_key_columns: exact_count(
                structural.s7_directory_counts[6],
                "S7 index-key-column count",
            )?,
            transitions: exact_count(structural.s7_directory_counts[7], "S7 transition count")?,
            key_effects: exact_count(structural.s7_directory_counts[8], "S7 key-effect count")?,
            key_components: exact_count(
                structural.s7_directory_counts[9],
                "S7 key-component count",
            )?,
            projections: exact_count(structural.s7_directory_counts[10], "S7 projection count")?,
            images: exact_count(structural.s7_directory_counts[11], "S7 image count")?,
            response_artifacts: exact_count(
                structural.s8.identity.artifact_count,
                "S8 artifact count",
            )?,
            response_selections: exact_count(
                structural.s8.identity.selection_count,
                "S8 selection count",
            )?,
            response_images: exact_count(structural.s8.identity.artifact_count, "S8 image count")?,
            response_identity: RetainedResponseEnvelopeIdentity {
                present: structural.s8.identity.present,
                aggregate_flags: structural.s8.identity.aggregate_flags,
                stable_transaction_id: structural.s8.identity.stable_transaction_id,
                request_digest: structural.s8.identity.request_digest,
                s6_section_root: structural.s8.identity.s6_section_root,
                s7_section_root: structural.s8.identity.s7_section_root,
                s8_section_root: structural.s8.identity.s8_section_root,
                response_root: structural.s8.identity.response_root,
                status_artifact_count: structural.s8.identity.status_artifact_count,
                retention_deadline: structural.s8.identity.retention_deadline,
                total_bytes: structural.s8.identity.total_bytes,
                artifact_count: structural.s8.identity.artifact_count,
                selection_count: structural.s8.identity.selection_count,
                image_arena_bytes: structural.s8.identity.image_arena_bytes,
                payload_digest: structural.s8.identity.payload_digest,
            },
        };
        let direct_persistent_bytes = expected.direct_persistent_bytes()?;
        let direct_persistent_allocation_slots = expected.direct_persistent_allocation_slots()?;
        let persistent_bytes = direct_persistent_bytes
            .checked_add(structural.s2_decoded_persistent_bytes)
            .and_then(|value| value.checked_add(structural.image_decoded_persistent_bytes))
            .and_then(|value| value.checked_add(structural.s8.image_persistent_bytes))
            .ok_or_else(|| reservation_error("retained persistent byte budget overflows"))?;
        let persistent_allocation_slots = direct_persistent_allocation_slots
            .checked_add(structural.s2_decoded_persistent_slots)
            .and_then(|value| value.checked_add(structural.image_decoded_persistent_slots))
            .and_then(|value| value.checked_add(structural.s8.image_persistent_slots))
            .ok_or_else(|| {
                reservation_error("retained persistent allocation-slot budget overflows")
            })?;
        Ok(Self {
            expected,
            header: structural.s7_header,
            source_measure: RetainedSemanticsV2SourceMeasure {
                s2_persistent_bytes: structural.s2_decoded_persistent_bytes,
                s2_persistent_slots: structural.s2_decoded_persistent_slots,
                image_persistent_bytes: structural.image_decoded_persistent_bytes,
                image_persistent_slots: structural.image_decoded_persistent_slots,
                response_image_persistent_bytes: structural.s8.image_persistent_bytes,
                response_image_persistent_slots: structural.s8.image_persistent_slots,
                maximum_scratch_bytes: structural
                    .raw_maximum_scratch_bytes
                    .max(structural.s8.maximum_scratch_bytes),
                maximum_scratch_slots: structural
                    .raw_maximum_scratch_slots
                    .max(structural.s8.maximum_scratch_slots),
            },
            budget: RetainedSemanticsV2ReservationBudget {
                persistent_bytes,
                persistent_allocation_slots,
                maximum_scratch_bytes: structural
                    .raw_maximum_scratch_bytes
                    .max(structural.s8.maximum_scratch_bytes),
                maximum_scratch_allocation_slots: structural
                    .raw_maximum_scratch_slots
                    .max(structural.s8.maximum_scratch_slots),
            },
            direct_persistent_bytes,
            direct_persistent_allocation_slots,
        })
    }

    pub(super) fn reserve(self) -> Result<ReservedSemanticsV2GraphOwner, EngineError> {
        let budget = self.budget;
        self.reserve_with_budget(budget)
    }

    /// Reject an insufficient caller admission budget before the first owner allocation.  The
    /// caller can provide a larger pool reservation, but no lower byte/slot/scratch limit can
    /// drain into a partial retained graph.
    pub(super) fn reserve_with_budget(
        self,
        available: RetainedSemanticsV2ReservationBudget,
    ) -> Result<ReservedSemanticsV2GraphOwner, EngineError> {
        ensure_budget(available, self.budget)?;
        let expected = self.expected;
        Ok(ReservedSemanticsV2GraphOwner {
            graph: ReservedSemanticsV2Graph {
                header: self.header,
                statements: reserve_exact(expected.statements, "S1 statement directory")?,
                records: reserve_exact(expected.records, "S2 decoded typed-record directory")?,
                dispositions: reserve_exact(expected.dispositions, "S4 disposition directory")?,
                sequence_effects: reserve_exact(
                    expected.sequence_effects,
                    "S5 published sequence-effect directory",
                )?,
                outcomes: reserve_exact(expected.outcomes, "S6 outcome directory")?,
                tables: reserve_exact(expected.tables, "S7 table directory")?,
                table_dispositions: reserve_exact(
                    expected.table_dispositions,
                    "S7 table-disposition directory",
                )?,
                resolutions: reserve_exact(expected.resolutions, "S7 resolution directory")?,
                dependencies: reserve_exact(expected.dependencies, "S7 dependency directory")?,
                dependency_uses: reserve_exact(
                    expected.dependency_uses,
                    "S7 statement-dependency-use directory",
                )?,
                indexes: reserve_exact(expected.indexes, "S7 index directory")?,
                index_key_columns: reserve_exact(
                    expected.index_key_columns,
                    "S7 index-key-column directory",
                )?,
                transitions: reserve_exact(expected.transitions, "S7 transition directory")?,
                key_effects: reserve_exact(expected.key_effects, "S7 key-effect directory")?,
                key_components: reserve_exact(
                    expected.key_components,
                    "S7 key-component directory",
                )?,
                projections: reserve_exact(expected.projections, "S7 projection directory")?,
                images: reserve_exact(expected.images, "S7 decoded image directory")?,
                response: if !expected.response_identity.present {
                    RetainedResponseEnvelope::Empty(expected.response_identity)
                } else {
                    RetainedResponseEnvelope::Present(RetainedResponseGraph {
                        identity: expected.response_identity,
                        artifacts: reserve_exact(
                            expected.response_artifacts,
                            "S8 response-artifact directory",
                        )?,
                        selections: reserve_exact(
                            expected.response_selections,
                            "S8 response-selection directory",
                        )?,
                        images: reserve_exact(
                            expected.response_images,
                            "S8 decoded response-image directory",
                        )?,
                    })
                },
            },
            expected,
            source_measure: self.source_measure,
            budget: self.budget,
        })
    }

    #[cfg(test)]
    fn required_budget(&self) -> RetainedSemanticsV2ReservationBudget {
        self.budget
    }

    #[cfg(test)]
    fn direct_persistent_bytes(&self) -> u64 {
        self.direct_persistent_bytes
    }

    #[cfg(test)]
    fn direct_persistent_allocation_slots(&self) -> u64 {
        self.direct_persistent_allocation_slots
    }
}

impl RetainedSemanticsV2ExpectedGraph {
    fn direct_persistent_bytes(self) -> Result<u64, EngineError> {
        let mut bytes = 0_u64;
        macro_rules! add_owner {
            ($count:expr, $ty:ty) => {
                bytes = bytes
                    .checked_add(owner_bytes::<$ty>($count)?)
                    .ok_or_else(|| reservation_error("retained graph byte budget overflows"))?;
            };
        }
        add_owner!(self.statements, RetainedStatement);
        add_owner!(self.records, DecodedTypedInsertRecord);
        add_owner!(self.dispositions, RetainedDisposition);
        add_owner!(self.sequence_effects, RetainedSequenceEffect);
        add_owner!(self.outcomes, RetainedStatementOutcome);
        add_owner!(self.tables, RetainedTable);
        add_owner!(self.table_dispositions, RetainedTableDisposition);
        add_owner!(self.resolutions, RetainedStatementResolution);
        add_owner!(self.dependencies, RetainedDependencyToken);
        add_owner!(self.dependency_uses, RetainedStatementDependencyUse);
        add_owner!(self.indexes, RetainedIndexDescriptor);
        add_owner!(self.index_key_columns, RetainedIndexKeyColumn);
        add_owner!(self.transitions, RetainedTransition);
        add_owner!(self.key_effects, RetainedKeyEffect);
        add_owner!(self.key_components, RetainedKeyComponent);
        add_owner!(self.projections, RetainedProjectionBinding);
        add_owner!(self.images, DecodedTypedImage);
        add_owner!(self.response_artifacts, RetainedResponseArtifact);
        add_owner!(self.response_selections, RetainedResponseSelection);
        add_owner!(self.response_images, DecodedTypedImage);
        Ok(bytes)
    }

    fn direct_persistent_allocation_slots(self) -> Result<u64, EngineError> {
        let mut slots = 0_u64;
        for count in [
            self.statements,
            self.records,
            self.dispositions,
            self.sequence_effects,
            self.outcomes,
            self.tables,
            self.table_dispositions,
            self.resolutions,
            self.dependencies,
            self.dependency_uses,
            self.indexes,
            self.index_key_columns,
            self.transitions,
            self.key_effects,
            self.key_components,
            self.projections,
            self.images,
            self.response_artifacts,
            self.response_selections,
            self.response_images,
        ] {
            slots = slots.checked_add(u64::from(count != 0)).ok_or_else(|| {
                reservation_error("retained graph allocation-slot budget overflows")
            })?;
        }
        Ok(slots)
    }
}

impl ReservedSemanticsV2GraphOwner {
    /// The complete strict decoder is the only future caller allowed to receive this private
    /// owner.  It must prove every length before `RetainedSemanticsV2Graph` can be constructed.
    pub(super) fn into_exact_graph(self) -> Result<ReservedSemanticsV2Graph, EngineError> {
        self.expected.require_exact_lengths(&self.graph)?;
        Ok(self.graph)
    }

    pub(super) fn graph_mut(&mut self) -> &mut ReservedSemanticsV2Graph {
        &mut self.graph
    }

    pub(super) fn graph(&self) -> &ReservedSemanticsV2Graph {
        &self.graph
    }

    pub(super) fn source_measure(&self) -> RetainedSemanticsV2SourceMeasure {
        self.source_measure
    }

    #[cfg(test)]
    fn capacities_match_expected(&self) -> bool {
        self.expected.capacities_match(&self.graph)
    }

    #[cfg(test)]
    fn budget(&self) -> RetainedSemanticsV2ReservationBudget {
        self.budget
    }
}

impl RetainedSemanticsV2ExpectedGraph {
    fn require_exact_lengths(&self, graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
        let exact = self.capacities_match(graph);
        let filled = graph.statements.len() == self.statements
            && graph.records.len() == self.records
            && graph.dispositions.len() == self.dispositions
            && graph.sequence_effects.len() == self.sequence_effects
            && graph.outcomes.len() == self.outcomes
            && graph.tables.len() == self.tables
            && graph.table_dispositions.len() == self.table_dispositions
            && graph.resolutions.len() == self.resolutions
            && graph.dependencies.len() == self.dependencies
            && graph.dependency_uses.len() == self.dependency_uses
            && graph.indexes.len() == self.indexes
            && graph.index_key_columns.len() == self.index_key_columns
            && graph.transitions.len() == self.transitions
            && graph.key_effects.len() == self.key_effects
            && graph.key_components.len() == self.key_components
            && graph.projections.len() == self.projections
            && graph.images.len() == self.images
            && response_lengths_match(graph, self);
        if !(exact && filled) {
            return Err(reservation_error(
                "strict retained decoder did not fill every exact graph owner",
            ));
        }
        Ok(())
    }

    fn capacities_match(&self, graph: &ReservedSemanticsV2Graph) -> bool {
        graph.statements.capacity() == self.statements
            && graph.records.capacity() == self.records
            && graph.dispositions.capacity() == self.dispositions
            && graph.sequence_effects.capacity() == self.sequence_effects
            && graph.outcomes.capacity() == self.outcomes
            && graph.tables.capacity() == self.tables
            && graph.table_dispositions.capacity() == self.table_dispositions
            && graph.resolutions.capacity() == self.resolutions
            && graph.dependencies.capacity() == self.dependencies
            && graph.dependency_uses.capacity() == self.dependency_uses
            && graph.indexes.capacity() == self.indexes
            && graph.index_key_columns.capacity() == self.index_key_columns
            && graph.transitions.capacity() == self.transitions
            && graph.key_effects.capacity() == self.key_effects
            && graph.key_components.capacity() == self.key_components
            && graph.projections.capacity() == self.projections
            && graph.images.capacity() == self.images
            && response_capacities_match(graph, self)
    }
}

fn response_lengths_match(
    graph: &ReservedSemanticsV2Graph,
    expected: &RetainedSemanticsV2ExpectedGraph,
) -> bool {
    match &graph.response {
        RetainedResponseEnvelope::Empty(identity) => {
            !expected.response_identity.present
                && expected.response_artifacts == 0
                && expected.response_selections == 0
                && expected.response_images == 0
                && identity == &expected.response_identity
        }
        RetainedResponseEnvelope::Present(response) => {
            expected.response_identity.present
                && response.identity == expected.response_identity
                && response.artifacts.len() == expected.response_artifacts
                && response.selections.len() == expected.response_selections
                && response.images.len() == expected.response_images
        }
    }
}

fn response_capacities_match(
    graph: &ReservedSemanticsV2Graph,
    expected: &RetainedSemanticsV2ExpectedGraph,
) -> bool {
    match &graph.response {
        RetainedResponseEnvelope::Empty(identity) => {
            !identity.present
                && expected.response_artifacts == 0
                && expected.response_selections == 0
                && expected.response_images == 0
        }
        RetainedResponseEnvelope::Present(response) => {
            response.identity.present
                && response.artifacts.capacity() == expected.response_artifacts
                && response.selections.capacity() == expected.response_selections
                && response.images.capacity() == expected.response_images
        }
    }
}

fn exact_count(value: impl TryInto<usize>, owner: &str) -> Result<usize, EngineError> {
    value
        .try_into()
        .map_err(|_| reservation_error(&format!("{owner} exceeds host addressability")))
}

fn owner_bytes<T>(count: usize) -> Result<u64, EngineError> {
    u64::try_from(count)
        .ok()
        .and_then(|count| {
            count.checked_mul(u64::try_from(std::mem::size_of::<T>()).expect("type size fits u64"))
        })
        .ok_or_else(|| reservation_error("retained graph owner byte count overflows"))
}

fn ensure_budget(
    available: RetainedSemanticsV2ReservationBudget,
    required: RetainedSemanticsV2ReservationBudget,
) -> Result<(), EngineError> {
    if available.persistent_bytes < required.persistent_bytes
        || available.persistent_allocation_slots < required.persistent_allocation_slots
        || available.maximum_scratch_bytes < required.maximum_scratch_bytes
        || available.maximum_scratch_allocation_slots < required.maximum_scratch_allocation_slots
    {
        return Err(reservation_error(
            "retained owner admission budget is below the exact raw-pass requirement",
        ));
    }
    Ok(())
}

fn reserve_exact<T>(count: usize, owner: &'static str) -> Result<Vec<T>, EngineError> {
    #[cfg(test)]
    note_reservation(owner, owner_bytes::<T>(count)?)?;
    #[cfg(not(test))]
    let _ = owner;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| reservation_error("retained graph owner reservation failed"))?;
    if values.capacity() != count {
        return Err(reservation_error(
            "retained graph owner capacity is not the exact raw-pass count",
        ));
    }
    Ok(values)
}

fn reservation_error(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 retained reservation: {message}"
    ))
}

#[cfg(test)]
thread_local! {
    static OBSERVATION: std::cell::Cell<Option<TestReservationState>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct TestReservationState {
    attempts: u64,
    fail_at: Option<u64>,
    direct_persistent_bytes: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TestReservationStats {
    attempts: u64,
    direct_persistent_bytes: u64,
}

#[cfg(test)]
fn note_reservation(owner: &str, persistent_bytes: u64) -> Result<(), EngineError> {
    OBSERVATION.with(|observation| {
        if let Some(mut state) = observation.get() {
            state.attempts = state
                .attempts
                .checked_add(1)
                .expect("retained reservation test counter overflow");
            state.direct_persistent_bytes = state
                .direct_persistent_bytes
                .checked_add(persistent_bytes)
                .expect("retained reservation test byte counter overflow");
            observation.set(Some(state));
            if state.fail_at == Some(state.attempts) {
                return Err(reservation_error(&format!(
                    "injected retained graph owner reservation failure at {owner}"
                )));
            }
        }
        Ok(())
    })
}

#[cfg(test)]
fn observe_reservations<T>(operation: impl FnOnce() -> T) -> (T, TestReservationStats) {
    OBSERVATION.with(|observation| {
        assert!(
            observation
                .replace(Some(TestReservationState {
                    attempts: 0,
                    fail_at: None,
                    direct_persistent_bytes: 0,
                }))
                .is_none(),
            "retained reservation observation cannot nest"
        );
        let result = operation();
        let state = observation
            .replace(None)
            .expect("retained reservation observation remains armed");
        (
            result,
            TestReservationStats {
                attempts: state.attempts,
                direct_persistent_bytes: state.direct_persistent_bytes,
            },
        )
    })
}

#[cfg(test)]
fn fail_reservation_at<T>(attempt: u64, operation: impl FnOnce() -> T) -> T {
    assert_ne!(attempt, 0, "retained reservation injection is one-based");
    OBSERVATION.with(|observation| {
        assert!(
            observation
                .replace(Some(TestReservationState {
                    attempts: 0,
                    fail_at: Some(attempt),
                    direct_persistent_bytes: 0,
                }))
                .is_none(),
            "retained reservation injection cannot nest"
        );
        let result = operation();
        observation
            .replace(None)
            .expect("retained reservation injection remains armed");
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn structural_measure() -> SemanticsV2StructuralMeasure {
        SemanticsV2StructuralMeasure {
            s1_statement_count: 2,
            s2_record_count: 2,
            s4_disposition_count: 3,
            s5_effect_count: 1,
            s6_outcome_count: 2,
            s7_directory_counts: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1],
            s7_header: super::super::super::pass_zero::SemanticsV2S7HeaderIdentity {
                total_bytes: 640,
                root_descriptor_version: 1,
                catalog_before_epoch: 7,
                catalog_after_epoch: 7,
                catalog_before_digest: [1; 32],
                catalog_after_digest: [1; 32],
                initial_database_root: [2; 32],
                final_database_root: [3; 32],
                initial_overlay_root: [4; 32],
                final_overlay_root: [5; 32],
                root_descriptor: [6; 32],
                payload_digest: [7; 32],
            },
            s8: empty_s8_measure_for_test(),
            // These stand for the complete nested decoded-record/image ABI, including their
            // names, text/vector values, and allocation slots. The outer graph only reserves
            // the move-only owner directories themselves.
            s2_decoded_persistent_bytes: 4096,
            s2_decoded_persistent_slots: 37,
            image_decoded_persistent_bytes: 8192,
            image_decoded_persistent_slots: 19,
            raw_maximum_scratch_bytes: 2048,
            raw_maximum_scratch_slots: 3,
        }
    }

    fn present_s8_structural_measure() -> SemanticsV2StructuralMeasure {
        let mut measure = structural_measure();
        measure.s8 = nonempty_s8_measure_for_test();
        measure
    }

    #[test]
    fn reserves_every_direct_owner_at_its_exact_pass_zero_count() {
        let measure = RetainedSemanticsV2Reservation::from_structural(structural_measure())
            .expect("test structural measure is addressable");
        let expected_bytes = measure.direct_persistent_bytes();
        let expected_slots = measure.direct_persistent_allocation_slots();
        let (owner, stats) = observe_reservations(|| measure.reserve());
        let owner = owner.expect("exact direct graph reservation succeeds");
        assert!(owner.capacities_match_expected());
        assert_eq!(owner.graph.header.root_descriptor, [6; 32]);
        assert_eq!(owner.graph.header.payload_digest, [7; 32]);
        assert_eq!(
            stats.attempts, 17,
            "one reservation per real direct graph owner"
        );
        assert_eq!(stats.direct_persistent_bytes, expected_bytes);
        assert_eq!(expected_slots, 17, "all test owners are nonempty");
        assert_eq!(
            owner.budget().persistent_bytes,
            expected_bytes + 4096 + 8192
        );
        assert_eq!(
            owner.budget().persistent_allocation_slots,
            expected_slots + 37 + 19
        );
    }

    #[test]
    fn rejects_one_byte_below_persistent_or_scratch_before_any_owner_reservation() {
        let measure =
            RetainedSemanticsV2Reservation::from_structural(present_s8_structural_measure())
                .expect("test structural measure is addressable");
        let exact = measure.required_budget();
        let persistent_short = RetainedSemanticsV2ReservationBudget {
            persistent_bytes: exact.persistent_bytes - 1,
            ..exact
        };
        let (result, stats) =
            observe_reservations(|| measure.reserve_with_budget(persistent_short));
        assert!(result.is_err());
        assert_eq!(stats.attempts, 0);

        let measure =
            RetainedSemanticsV2Reservation::from_structural(present_s8_structural_measure())
                .expect("test structural measure is addressable");
        let exact = measure.required_budget();
        let scratch_short = RetainedSemanticsV2ReservationBudget {
            maximum_scratch_bytes: exact.maximum_scratch_bytes - 1,
            ..exact
        };
        let (result, stats) = observe_reservations(|| measure.reserve_with_budget(scratch_short));
        assert!(result.is_err());
        assert_eq!(stats.attempts, 0);

        let measure =
            RetainedSemanticsV2Reservation::from_structural(present_s8_structural_measure())
                .expect("test structural measure is addressable");
        let exact = measure.required_budget();
        let persistent_slots_short = RetainedSemanticsV2ReservationBudget {
            persistent_allocation_slots: exact.persistent_allocation_slots - 1,
            ..exact
        };
        let (result, stats) =
            observe_reservations(|| measure.reserve_with_budget(persistent_slots_short));
        assert!(result.is_err());
        assert_eq!(stats.attempts, 0);

        let measure =
            RetainedSemanticsV2Reservation::from_structural(present_s8_structural_measure())
                .expect("test structural measure is addressable");
        let exact = measure.required_budget();
        let scratch_slots_short = RetainedSemanticsV2ReservationBudget {
            maximum_scratch_allocation_slots: exact.maximum_scratch_allocation_slots - 1,
            ..exact
        };
        let (result, stats) =
            observe_reservations(|| measure.reserve_with_budget(scratch_slots_short));
        assert!(result.is_err());
        assert_eq!(stats.attempts, 0);
    }

    #[test]
    fn every_direct_owner_failure_drains_without_a_partial_graph() {
        let (_, stats) = observe_reservations(|| {
            RetainedSemanticsV2Reservation::from_structural(structural_measure())
                .and_then(RetainedSemanticsV2Reservation::reserve)
        });
        for attempt in 1..=stats.attempts {
            let result = fail_reservation_at(attempt, || {
                RetainedSemanticsV2Reservation::from_structural(structural_measure())
                    .and_then(RetainedSemanticsV2Reservation::reserve)
            });
            assert!(result.is_err(), "injected owner {attempt} must fail");

            let retry = RetainedSemanticsV2Reservation::from_structural(structural_measure())
                .and_then(RetainedSemanticsV2Reservation::reserve);
            assert!(
                retry.is_ok(),
                "injected owner {attempt} leaves no retained state"
            );
        }
    }

    #[test]
    fn present_s8_reservation_owns_response_directories_and_retries_every_failure() {
        let measure =
            RetainedSemanticsV2Reservation::from_structural(present_s8_structural_measure())
                .expect("present S8 structural measure is addressable");
        let expected_bytes = measure.direct_persistent_bytes();
        let expected_slots = measure.direct_persistent_allocation_slots();
        let (owner, stats) = observe_reservations(|| measure.reserve());
        let owner = owner.expect("present S8 graph reserves exactly");
        assert!(owner.capacities_match_expected());
        assert_eq!(
            stats.attempts, 20,
            "S8 adds artifact, selection, and image owners"
        );
        assert_eq!(expected_slots, 20, "all direct owners are nonempty");
        assert_eq!(stats.direct_persistent_bytes, expected_bytes);
        assert_eq!(
            owner.budget().persistent_bytes,
            expected_bytes + 4096 + 8192 + 3072,
            "nested S2, S7-image, and S8-image ownership is budgeted"
        );
        assert_eq!(
            owner.budget().persistent_allocation_slots,
            expected_slots + 37 + 19 + 11
        );
        assert_eq!(owner.budget().maximum_scratch_bytes, 2048);
        assert_eq!(owner.budget().maximum_scratch_allocation_slots, 5);

        for attempt in 1..=stats.attempts {
            let result = fail_reservation_at(attempt, || {
                RetainedSemanticsV2Reservation::from_structural(present_s8_structural_measure())
                    .and_then(RetainedSemanticsV2Reservation::reserve)
            });
            assert!(
                result.is_err(),
                "present S8 owner {attempt} must fail injected reservation"
            );
            let retry =
                RetainedSemanticsV2Reservation::from_structural(present_s8_structural_measure())
                    .and_then(RetainedSemanticsV2Reservation::reserve);
            assert!(
                retry.is_ok(),
                "present S8 owner {attempt} leaves no partial owner"
            );
        }
    }
}
