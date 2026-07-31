//! Q2's deliberately tiny, test-only generation authority.
//!
//! This is a fixed witness producer, not a retained-graph adapter.  Construction is keyed only
//! by the closed checked-in case tag; it cannot receive Q1 bytes, sections, a graph, a pending
//! owner, or decoded S7 outputs.  The launched work observes only sealed neutral facts and the
//! pre-reserved output slots supplied by the lifecycle.

use super::{
    input, sealed, validate_reserved_builder_result, DrainOutcome, LaunchedAttempt,
    SemanticsV2ReservedGenerationBuilder, SemanticsV2ReservedGenerationWork,
};
use crate::typed_insert_aggregate::semantics_v2::retained::{
    FullyWitnessValidatedSemanticsV2, GenerationPendingSemanticsV2,
};
use crate::EngineError;
use std::{cell::Cell, rc::Rc};

type CanonicalDigest = gpu_db_wal::CanonicalDigest;
type DigestObserver = Rc<Cell<Option<CanonicalDigest>>>;

const ROOT_DESCRIPTOR_VERSION: u16 = 1;

/// Closed Q2 evidence cases.  This is the builder's only construction input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum GoldenCase {
    MinimalAbort,
    ExplicitAbort,
    SuccessfulInterleaved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum GoldenBuilderSabotage {
    None,
    FinalRoot,
}

/// The only Q2 builder.  It owns no borrowed witness and has no retained-model input.
pub(super) struct GoldenBuilder {
    case: GoldenCase,
    sabotage: GoldenBuilderSabotage,
    // Test evidence only: allocated before total launch and never read by the builder.
    observer: Option<DigestObserver>,
}

/// Opaque, non-`Clone` candidate retained by the fully validated owner.
pub(super) struct GoldenCandidate {
    _case: GoldenCase,
}

/// Immediate-quiesced test work.  Output writes happen during total launch, so drain is only the
/// proof that every builder-visible owner is safe to hand back to the independent validator.
pub(super) struct GoldenWork;

impl GoldenBuilder {
    pub(super) fn new(case: GoldenCase) -> Self {
        Self {
            case,
            sabotage: GoldenBuilderSabotage::None,
            observer: None,
        }
    }

    pub(super) fn with_sabotage(case: GoldenCase, sabotage: GoldenBuilderSabotage) -> Self {
        Self {
            case,
            sabotage,
            observer: None,
        }
    }

    fn observe_generation_input(mut self, observer: DigestObserver) -> Self {
        self.observer = Some(observer);
        self
    }
}

/// Retained-boundary-only end-to-end evidence seam.  The opaque candidate never leaves this
/// module: post-validation byte reconstruction still borrows only `FullyWitnessValidated`.
pub(in super::super) fn validate_and_reencode_case(
    pending: GenerationPendingSemanticsV2<'_>,
    case: GoldenCase,
) -> Result<([Vec<u8>; 7], CanonicalDigest), EngineError> {
    // Allocate the observer before launch.  It only captures the sealed neutral digest copied to
    // the output header, and no expected literal participates in output construction.
    let observer = Rc::new(Cell::new(None));
    let registry = input::GenerationQuarantineRegistry::<GoldenCandidate, GoldenWork>::new();
    let owner = validate_reserved_builder_result(
        pending,
        GoldenBuilder::new(case).observe_generation_input(Rc::clone(&observer)),
        &registry,
    )?;
    let builder_input_digest = observer.take().ok_or_else(|| {
        generation_error("golden builder did not observe its sealed input digest")
    })?;
    Ok((owner.reencode_s1_s7_for_test()?, builder_input_digest))
}

#[cfg(test)]
fn validate_case_with_sabotage(
    pending: GenerationPendingSemanticsV2<'_>,
    case: GoldenCase,
    sabotage: GoldenBuilderSabotage,
) -> Result<FullyWitnessValidatedSemanticsV2<GoldenCandidate>, EngineError> {
    let registry = input::GenerationQuarantineRegistry::<GoldenCandidate, GoldenWork>::new();
    validate_reserved_builder_result(
        pending,
        GoldenBuilder::with_sabotage(case, sabotage),
        &registry,
    )
}

/// Sabotage evidence deliberately drops a successful fully-owned result immediately.  Its
/// public result cannot expose the candidate or turn this builder into a reusable authority.
pub(in super::super) fn validate_case_sabotage_result(
    pending: GenerationPendingSemanticsV2<'_>,
    case: GoldenCase,
    sabotage: GoldenBuilderSabotage,
) -> Result<(), EngineError> {
    validate_case_with_sabotage(pending, case, sabotage).map(|_| ())
}

impl sealed::Builder for GoldenBuilder {}
impl sealed::Candidate for GoldenCandidate {}
impl sealed::Work for GoldenWork {}

impl SemanticsV2ReservedGenerationWork for GoldenWork {
    fn drain_once(&mut self) -> DrainOutcome {
        DrainOutcome::Quiesced { execution: Ok(()) }
    }
}

impl SemanticsV2ReservedGenerationBuilder for GoldenBuilder {
    type Candidate = GoldenCandidate;
    type Work = GoldenWork;

    fn try_reserve_candidate(
        &self,
        reservation: input::GenerationBuilderReservation,
    ) -> Result<Self::Candidate, EngineError> {
        let expected = match self.case {
            GoldenCase::MinimalAbort => (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
            GoldenCase::ExplicitAbort => (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
            GoldenCase::SuccessfulInterleaved => (2, 3, 13, 46, 4, 6, 6, 9, 28, 2, 4),
        };
        let actual = (
            reservation.tables(),
            reservation.rows(),
            reservation.cells(),
            reservation.value_bytes(),
            reservation.indexes(),
            reservation.keys(),
            reservation.effects(),
            reservation.effect_values(),
            reservation.effect_value_bytes(),
            reservation.table_outputs(),
            reservation.index_outputs(),
        );
        if actual != expected {
            return Err(generation_error(&format!(
                "golden case reservation facts are not pinned: expected {expected:?}, got {actual:?}"
            )));
        }
        Ok(GoldenCandidate { _case: self.case })
    }

    fn launch(
        self,
        mut launch: input::ReservedGenerationLaunch<Self::Candidate, Self::Work>,
    ) -> LaunchedAttempt<Self::Candidate, Self::Work> {
        // The observer is installed before total launch and merely records the sealed neutral
        // digest that every final header carries, including abort echoes.
        if let Some(observer) = &self.observer {
            observer.set(Some(
                launch
                    .input()
                    .builder_input_digest()
                    .expect("sealed Q2 neutral input is self-consistent"),
            ));
        }
        match self.case {
            GoldenCase::MinimalAbort | GoldenCase::ExplicitAbort => {
                // Abort witnesses are intentionally echoed from the sealed neutral base state;
                // no final identity is a builder-side input for an abort.
                launch.write_abort_echo();
            }
            GoldenCase::SuccessfulInterleaved => write_success_outputs(&mut launch),
        }
        if self.sabotage == GoldenBuilderSabotage::FinalRoot {
            launch.outputs_mut().header.final_database_root[0] ^= 1;
        }
        LaunchedAttempt::launched(launch, GoldenWork)
    }
}

fn write_success_outputs(
    launch: &mut input::ReservedGenerationLaunch<GoldenCandidate, GoldenWork>,
) {
    let input = launch.input();
    let neutral = input.neutral_view();
    let identity = neutral.identity();
    assert_eq!(
        neutral.tables().count(),
        2,
        "Q2 success neutral table count"
    );
    assert_eq!(
        neutral.indexes().count(),
        4,
        "Q2 success neutral index count"
    );
    let digest = input
        .builder_input_digest()
        .expect("sealed Q2 success neutral input is self-consistent");
    let outputs = launch.outputs_mut();
    outputs.set_header(input::GenerationOutputHeader {
        filled: true,
        duplicate: false,
        root_descriptor_version: ROOT_DESCRIPTOR_VERSION,
        database_id: identity.database_id,
        catalog_epoch: identity.catalog_epoch,
        catalog_digest: identity.catalog_digest,
        stable_transaction_id: identity.stable_transaction_id,
        commit_sequence: identity.commit_sequence,
        initial_database_root: identity.initial_database_root,
        generation_input_digest: digest,
        final_database_root: [0x45; 32],
    });
    outputs.set_table(0, table_output(1_101, 12, [0x23; 32], 5, 0, 2));
    outputs.set_table(1, table_output(1_102, 22, [0x34; 32], 8, 2, 2));
    outputs.set_index(0, index_output(2_101, 2, [0x80; 32]));
    outputs.set_index(1, index_output(2_102, 2, [0x81; 32]));
    outputs.set_index(2, index_output(3_101, 2, [0x82; 32]));
    outputs.set_index(3, index_output(3_102, 2, [0x83; 32]));
}

fn table_output(
    stable_table_id: u64,
    final_data_generation: u64,
    final_table_root: CanonicalDigest,
    final_logical_row_count: u64,
    index_start: u32,
    index_count: u32,
) -> input::GenerationOutputTable {
    input::GenerationOutputTable {
        filled: true,
        duplicate: false,
        stable_table_id,
        final_data_generation,
        final_table_root,
        final_logical_row_count,
        index_start,
        index_count,
    }
}

fn index_output(
    stable_index_id: u64,
    final_index_generation: u64,
    final_index_root: CanonicalDigest,
) -> input::GenerationOutputIndex {
    input::GenerationOutputIndex {
        filled: true,
        duplicate: false,
        stable_index_id,
        final_index_generation,
        final_index_root,
    }
}

fn generation_error(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 golden generation: {message}"
    ))
}
