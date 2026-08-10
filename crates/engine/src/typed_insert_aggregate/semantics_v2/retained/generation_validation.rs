//! Sealed, owned generation lifecycle for the second retained typestate transition.
//!
//! The builder sees one flat neutral input plus fixed output slots.  It never receives a pending
//! owner, retained graph, catalog/allocator proof, expected digest, or S7 final roots.  Launch is
//! total; after that point the attempt owns every builder-visible allocation until a proven drain
//! permits independent validation or an unproven drain parks all of it in an infallible ticket.

#[path = "generation_validation/input.rs"]
mod input;

pub(super) use input::GenerationQuarantineRegistry;

#[cfg(test)]
use super::{
    CatalogAndAllocatorValidated, FullyWitnessValidatedSemanticsV2, GenerationPendingSemanticsV2,
};
use super::{RetainedSemanticsV2Graph, SemanticsV2CatalogWitness};
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedTable,
};
use crate::EngineError;

type CanonicalDigest = gpu_db_wal::CanonicalDigest;

const ROOT_DESCRIPTOR_VERSION: u16 = 1;

mod sealed {
    pub trait Builder {}
    pub trait Candidate {}
    pub trait Work {}
}

#[path = "generation_validation/runtime_builder.rs"]
mod runtime_builder;
// Validation children consume these private concrete owners through this module boundary.
#[allow(unused_imports)]
pub(super) use runtime_builder::{
    LiveTypedInsertGenerationBuilder, LiveTypedInsertGenerationCandidate,
    LiveTypedInsertGenerationWork,
};

/// A builder work owner reports whether the device/runtime is definitely quiescent.  The
/// lifecycle, not the work implementation, enforces exact-once invocation.
#[allow(dead_code)]
pub(super) trait SemanticsV2ReservedGenerationWork: sealed::Work + 'static {
    fn drain_once(&mut self) -> DrainOutcome;
}

/// This is deliberately not `Result`: an immediate or partial launch failure remains an owned
/// launched attempt and must pass through the same drain/quarantine transition as normal work.
#[allow(dead_code)]
pub(super) enum DrainOutcome {
    Quiesced { execution: Result<(), EngineError> },
    QuiescenceUnproven(EngineError),
}

/// Sealed builder boundary.  `try_reserve_candidate` is the last fallible builder operation and
/// receives count-only capacity facts.  `launch` is total and receives the already-owned flat
/// input, fixed output slots, opaque candidate/private backing, and pre-created quarantine
/// reservation bundled in `ReservedGenerationLaunch`.
#[allow(dead_code)]
pub(super) trait SemanticsV2ReservedGenerationBuilder: sealed::Builder {
    type Candidate: sealed::Candidate + 'static;
    type Work: SemanticsV2ReservedGenerationWork + 'static;

    fn try_reserve_candidate(
        &self,
        reservation: input::GenerationBuilderReservation,
    ) -> Result<Self::Candidate, EngineError>;

    fn launch(
        self,
        launch: input::ReservedGenerationLaunch<Self::Candidate, Self::Work>,
    ) -> LaunchedAttempt<Self::Candidate, Self::Work>;

    /// After the work fence proves quiescence, transfer the device-produced fixed commitments
    /// from the candidate into the already-reserved validation slots.  This operation may not
    /// launch, allocate, hash, or consult the retained graph.
    fn finalize_quiesced(
        _candidate: &mut Self::Candidate,
        _outputs: &mut input::ReservedGenerationOutputs,
    ) -> Result<(), EngineError> {
        Ok(())
    }
}

/// Opaque, armed post-launch owner.  It intentionally has no getters for input, output slots,
/// candidate, pending, or work.  The only release is `drain_once`, followed internally by an
/// owned `DrainedAttempt` on proven quiescence.
#[allow(dead_code)]
pub(super) struct LaunchedAttempt<C, W: SemanticsV2ReservedGenerationWork> {
    input: Option<input::SealedGenerationInput>,
    outputs: Option<input::ReservedGenerationOutputs>,
    candidate: Option<C>,
    work: Option<W>,
    ticket: Option<input::PreReservedQuarantine<C, W>>,
    drain_attempted: bool,
}

struct DrainedAttempt<C> {
    outputs: input::ReservedGenerationOutputs,
    candidate: C,
}

impl<C, W: SemanticsV2ReservedGenerationWork> LaunchedAttempt<C, W> {
    fn launched(launch: input::ReservedGenerationLaunch<C, W>, work: W) -> Self {
        let (input, outputs, candidate, ticket) = launch.into_parts();
        Self {
            input: Some(input),
            outputs: Some(outputs),
            candidate: Some(candidate),
            work: Some(work),
            ticket: Some(ticket),
            drain_attempted: false,
        }
    }

    /// Marks the attempt before invoking work.  A subsequent call cannot retry the work and
    /// therefore cannot accidentally release ownership after an earlier uncertain completion.
    pub(super) fn drain_once(&mut self) -> DrainOutcome {
        if self.drain_attempted {
            return DrainOutcome::QuiescenceUnproven(generation_error(
                "generation attempt drain was already attempted",
            ));
        }
        self.drain_attempted = true;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.work
                .as_mut()
                .expect("armed generation attempt retains work")
                .drain_once()
        }));
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(panic) => {
                // `drain_attempted` is already armed.  Retain every generic owner before the
                // panic resumes so Drop cannot retry work or free backing still visible to an
                // unknown device/runtime completion.
                self.park_unproven(input::QuarantineFailure::DrainPanicked);
                std::panic::resume_unwind(panic);
            }
        };
        match outcome {
            DrainOutcome::Quiesced { execution } => {
                // Quiescence proves that the work owner may be dropped before outputs move to
                // the validator.  Execution failure is still a proven completion, not a leak.
                let _ = self.work.take();
                DrainOutcome::Quiesced { execution }
            }
            DrainOutcome::QuiescenceUnproven(error) => {
                self.park_unproven(input::QuarantineFailure::QuiescenceUnproven);
                DrainOutcome::QuiescenceUnproven(error)
            }
        }
    }

    fn park_unproven(&mut self, failure: input::QuarantineFailure) {
        self.ticket
            .take()
            .expect("unproven generation attempt retains a precreated ticket")
            .park(
                self.work
                    .take()
                    .expect("unproven generation attempt retains work"),
                self.input
                    .take()
                    .expect("unproven generation attempt retains neutral input"),
                self.outputs
                    .take()
                    .expect("unproven generation attempt retains output backing"),
                self.candidate
                    .take()
                    .expect("unproven generation attempt retains candidate backing"),
                failure,
            );
    }

    fn into_drained(mut self) -> DrainedAttempt<C> {
        debug_assert!(self.drain_attempted && self.work.is_none());
        let _ = self.input.take();
        DrainedAttempt {
            outputs: self
                .outputs
                .take()
                .expect("quiesced generation attempt retains output backing"),
            candidate: self
                .candidate
                .take()
                .expect("quiesced generation attempt retains candidate backing"),
        }
    }

    fn finalize_quiesced<B>(&mut self) -> Result<(), EngineError>
    where
        B: SemanticsV2ReservedGenerationBuilder<Candidate = C, Work = W>,
    {
        debug_assert!(self.drain_attempted && self.work.is_none());
        B::finalize_quiesced(
            self.candidate
                .as_mut()
                .expect("quiesced generation retains its candidate"),
            self.outputs
                .as_mut()
                .expect("quiesced generation retains its output slots"),
        )
    }
}

impl<C, W: SemanticsV2ReservedGenerationWork> Drop for LaunchedAttempt<C, W> {
    fn drop(&mut self) {
        if self.work.is_some() && !self.drain_attempted {
            // This is the exact same one-shot transition as the explicit path.  A failure to
            // prove quiescence parks the complete backing; an execution error after proven
            // quiescence merely drops safe owners.  Neither outcome is ignored or retried.
            let _ = self.drain_once();
        }
    }
}

/// Final owned generation output.  There is no constructor or extractor outside this sealed
/// validation leaf, and it has no lifetime parameter: pinned catalog/allocator borrows ended
/// before this value was created.
#[allow(dead_code)]
pub(super) struct OwnedGenerationResult<C> {
    header: input::GenerationOutputHeader,
    tables: Box<[input::GenerationOutputTable]>,
    indexes: Box<[input::GenerationOutputIndex]>,
    candidate: C,
}

/// Consume catalog/allocator-validated ownership through the only reserved builder lifecycle.
/// All fallible preparation is complete before `launch`; every post-launch path drains or parks
/// without exposing partial output.
#[cfg(test)]
pub(super) fn validate_reserved_builder_result<'a, B>(
    pending: GenerationPendingSemanticsV2<'a>,
    builder: B,
    quarantine_registry: &input::GenerationQuarantineRegistry<B::Candidate, B::Work>,
) -> Result<FullyWitnessValidatedSemanticsV2<B::Candidate>, EngineError>
where
    B: SemanticsV2ReservedGenerationBuilder,
{
    let generation = validate_reserved_graph(
        &pending.graph,
        &pending.catalog_and_allocator.catalog,
        builder,
        quarantine_registry,
    )?;
    let CatalogAndAllocatorValidated {
        catalog: _,
        allocator_index: _,
        allocator_assignment: _,
    } = pending.catalog_and_allocator;
    Ok(FullyWitnessValidatedSemanticsV2 {
        graph: pending.graph,
        generation,
    })
}

/// Run the sole reserved builder lifecycle over one already-closed retained graph.  Both live
/// and replay call this exact function; only their source of the retained graph and borrowed
/// witnesses differs.
pub(super) fn validate_reserved_graph<B>(
    retained: &RetainedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
    builder: B,
    quarantine_registry: &input::GenerationQuarantineRegistry<B::Candidate, B::Work>,
) -> Result<OwnedGenerationResult<B::Candidate>, EngineError>
where
    B: SemanticsV2ReservedGenerationBuilder,
{
    let identity = retained.identity;
    let graph = &retained.graph;
    let measure = input::measure(graph, catalog, identity)?;
    let candidate = builder.try_reserve_candidate(measure.builder_reservation())?;
    let launch = input::reserve_and_fill(
        measure,
        graph,
        catalog,
        identity,
        candidate,
        quarantine_registry,
    )?;
    let mut attempt = builder.launch(launch);
    match attempt.drain_once() {
        DrainOutcome::Quiesced { execution: Ok(()) } => {
            attempt.finalize_quiesced::<B>()?;
            validate_drained(retained, catalog, attempt.into_drained())
        }
        DrainOutcome::Quiesced {
            execution: Err(error),
        }
        | DrainOutcome::QuiescenceUnproven(error) => Err(error),
    }
}

fn validate_drained<C>(
    retained: &RetainedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
    drained: DrainedAttempt<C>,
) -> Result<OwnedGenerationResult<C>, EngineError> {
    let DrainedAttempt { outputs, candidate } = drained;
    if outputs.extra_write
        || !outputs.header.filled
        || outputs.header.duplicate
        || outputs
            .tables
            .iter()
            .any(|table| !table.filled || table.duplicate)
        || outputs
            .indexes
            .iter()
            .any(|index| !index.filled || index.duplicate)
    {
        return Err(generation_error(
            "generation builder left missing, duplicate, or extra fixed outputs",
        ));
    }
    validate_header(retained, catalog, &outputs.header)?;
    validate_table_outputs(retained, &outputs.tables, &outputs.indexes)?;
    Ok(OwnedGenerationResult {
        header: outputs.header,
        tables: outputs.tables.into_boxed_slice(),
        indexes: outputs.indexes.into_boxed_slice(),
        candidate,
    })
}

fn validate_header(
    retained: &RetainedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
    header: &input::GenerationOutputHeader,
) -> Result<(), EngineError> {
    let identity = retained.identity;
    let graph = &retained.graph;
    if header.root_descriptor_version != ROOT_DESCRIPTOR_VERSION
        || header.root_descriptor_version != graph.header.root_descriptor_version
        || header.database_id != identity.database_id
        || header.catalog_epoch != identity.catalog_epoch
        || header.catalog_digest != identity.catalog_digest
        || header.stable_transaction_id != identity.stable_transaction_id
        || header.commit_sequence != identity.commit_sequence
        || header.initial_database_root != identity.initial_database_root
        || header.final_database_root != graph.header.final_database_root
        || graph.header.catalog_before_epoch != identity.catalog_epoch
        || graph.header.catalog_before_digest != identity.catalog_digest
        || graph.header.catalog_after_epoch < identity.catalog_epoch
        || graph.header.catalog_after_digest == [0; 32]
        || graph.header.initial_database_root != identity.initial_database_root
        || catalog.database_id != identity.database_id
        || catalog.catalog_epoch != identity.catalog_epoch
        || catalog.catalog_digest != identity.catalog_digest
    {
        return Err(generation_error(
            "generation output header does not match retained envelope",
        ));
    }
    // Deliberately independent of the builder's flat-input traversal.
    let expected = input::generation_input_digest(graph, catalog, identity)?;
    if header.generation_input_digest != expected {
        return Err(generation_error(
            "generation output input digest does not match retained neutral facts",
        ));
    }
    Ok(())
}

fn validate_table_outputs(
    retained: &RetainedSemanticsV2Graph,
    outputs: &[input::GenerationOutputTable],
    index_outputs: &[input::GenerationOutputIndex],
) -> Result<(), EngineError> {
    let graph = &retained.graph;
    let initial_database_root = retained.identity.initial_database_root;
    if outputs.len() != graph.tables.len() {
        return Err(generation_error(
            "generation output table count does not equal retained tables",
        ));
    }
    let abort = graph.outcomes.last().is_some_and(|outcome| {
        outcome.outcome.kind == gpu_db_wal::CanonicalOutcomeKind::AbortError
    });
    if abort
        && (!graph.transitions.is_empty()
            || !graph.key_effects.is_empty()
            || graph.header.final_database_root != initial_database_root)
    {
        return Err(generation_error(
            "abort generation form has transitions, effects, or changed database root",
        ));
    }
    let mut previous_table = None;
    let mut packed_index_start = 0_u32;
    for (ordinal, (table, output)) in graph.tables.iter().zip(outputs.iter()).enumerate() {
        if previous_table.is_some_and(|previous| table.stable_table_id <= previous)
            || table.table_ref != ordinal as u32
            || table.image_ref != ordinal as u32
            || table.stable_table_id != output.stable_table_id
            || table.data_generation_after != output.final_data_generation
            || table.final_table_root != output.final_table_root
            || table.final_logical_row_count != output.final_logical_row_count
            || validate_packed_output_index_range(
                packed_index_start,
                table.owned_index_count,
                output,
            )
            .is_err()
        {
            return Err(generation_error(
                "generation table output identity/order differs from S7",
            ));
        }
        validate_index_outputs(graph, table, output, index_outputs, abort)?;
        if abort
            && (table.data_generation_after != table.data_generation_before
                || table.final_table_root != table.initial_table_root
                || table.final_logical_row_count != table.initial_logical_row_count)
        {
            return Err(generation_error(
                "abort table output differs from its initial retained state",
            ));
        }
        previous_table = Some(table.stable_table_id);
        packed_index_start = packed_index_start
            .checked_add(table.owned_index_count)
            .ok_or_else(|| generation_error("packed generation output index range overflows"))?;
    }
    validate_packed_index_output_exhaustion(packed_index_start, index_outputs.len())?;
    Ok(())
}

fn validate_packed_output_index_range(
    packed_start: u32,
    owned_count: u32,
    output: &input::GenerationOutputTable,
) -> Result<(), EngineError> {
    if output.index_start != packed_start || output.index_count != owned_count {
        return Err(generation_error(
            "generation output table uses a non-packed index range",
        ));
    }
    packed_start
        .checked_add(owned_count)
        .ok_or_else(|| generation_error("packed generation output index range overflows"))?;
    Ok(())
}

fn validate_packed_index_output_exhaustion(
    packed_index_end: u32,
    output_len: usize,
) -> Result<(), EngineError> {
    let output_end = u32::try_from(output_len)
        .map_err(|_| generation_error("flat generation output length is unaddressable"))?;
    if packed_index_end != output_end {
        return Err(generation_error(
            "packed generation output index ranges do not exactly exhaust the output owner",
        ));
    }
    Ok(())
}

fn validate_index_outputs(
    graph: &ReservedSemanticsV2Graph,
    table: &RetainedTable,
    output: &input::GenerationOutputTable,
    index_outputs: &[input::GenerationOutputIndex],
    abort: bool,
) -> Result<(), EngineError> {
    let retained = input::range(
        &graph.indexes,
        table.owned_index_start,
        table.owned_index_count,
        "table index",
    )?;
    let outputs = flat_output_range(index_outputs, output.index_start, output.index_count)?;
    validate_flat_index_outputs(retained, outputs, abort)
}

fn validate_flat_index_outputs(
    retained: &[super::graph::RetainedIndexDescriptor],
    outputs: &[input::GenerationOutputIndex],
    abort: bool,
) -> Result<(), EngineError> {
    if retained.len() != outputs.len() {
        return Err(generation_error(
            "generation flat index output count is not exact",
        ));
    }
    let mut previous = None;
    for (index, output) in retained.iter().zip(outputs.iter()) {
        if previous.is_some_and(|prior| index.stable_index_id <= prior)
            || index.stable_index_id != output.stable_index_id
            || index.final_index_generation != output.final_index_generation
            || index.final_index_root != output.final_index_root
        {
            return Err(generation_error(
                "generation flat index output identity/order differs from S7",
            ));
        }
        if abort
            && (index.final_index_generation != index.base_index_generation
                || index.final_index_root != index.base_index_root)
        {
            return Err(generation_error(
                "abort index output differs from base index state",
            ));
        }
        previous = Some(index.stable_index_id);
    }
    Ok(())
}

fn flat_output_range<T>(values: &[T], start: u32, count: u32) -> Result<&[T], EngineError> {
    let start = usize::try_from(start)
        .map_err(|_| generation_error("flat generation output start is unaddressable"))?;
    let count = usize::try_from(count)
        .map_err(|_| generation_error("flat generation output count is unaddressable"))?;
    let end = start
        .checked_add(count)
        .ok_or_else(|| generation_error("flat generation output range overflows"))?;
    values
        .get(start..end)
        .ok_or_else(|| generation_error("flat generation output range is outside reservation"))
}

fn generation_error(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 generation: {message}"
    ))
}

/// The sole deterministic builder implementation.  It is test-only and derives the minimal
/// abort fixture's outputs from neutral base state; no production builder exists.
#[cfg(test)]
pub(super) struct DeterministicAbortBuilder {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    sabotage: DeterministicAbortSabotage,
    observed_input_digest: Option<std::sync::Arc<std::sync::Mutex<Option<CanonicalDigest>>>>,
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) enum DeterministicAbortSabotage {
    None,
    CandidateReservationFailure,
    MissingOutput,
    DuplicateOutput,
    ExtraOutput,
    HeaderIdentity,
    HeaderInputDigest,
    HeaderFinalRoot,
    TableIdentity,
    TableGeneration,
    TableRoot,
    TableCount,
    TableIndexRange,
    NeutralInput,
    ImmediateLaunchFailure,
    PartialLaunchFailure,
}

#[cfg(test)]
impl DeterministicAbortBuilder {
    pub(super) fn new(
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        Self {
            calls,
            drops,
            sabotage: DeterministicAbortSabotage::None,
            observed_input_digest: None,
        }
    }

    pub(super) fn with_sabotage(
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        sabotage: DeterministicAbortSabotage,
    ) -> Self {
        Self {
            calls,
            drops,
            sabotage,
            observed_input_digest: None,
        }
    }

    pub(super) fn with_observed_input_digest(
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        drops: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        sabotage: DeterministicAbortSabotage,
        observed_input_digest: std::sync::Arc<std::sync::Mutex<Option<CanonicalDigest>>>,
    ) -> Self {
        Self {
            calls,
            drops,
            sabotage,
            observed_input_digest: Some(observed_input_digest),
        }
    }
}

#[cfg(test)]
pub(super) struct AbortCandidate(std::sync::Arc<std::sync::atomic::AtomicUsize>);

#[cfg(test)]
impl Drop for AbortCandidate {
    fn drop(&mut self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
pub(super) struct AbortWork {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    outcome: AbortWorkOutcome,
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum AbortWorkOutcome {
    Complete,
    ImmediateFailure,
    PartialFailure,
}

#[cfg(test)]
impl sealed::Builder for DeterministicAbortBuilder {}
#[cfg(test)]
impl sealed::Candidate for AbortCandidate {}
#[cfg(test)]
impl sealed::Work for AbortWork {}
#[cfg(test)]
impl SemanticsV2ReservedGenerationWork for AbortWork {
    fn drain_once(&mut self) -> DrainOutcome {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match self.outcome {
            AbortWorkOutcome::Complete => DrainOutcome::Quiesced { execution: Ok(()) },
            AbortWorkOutcome::ImmediateFailure => DrainOutcome::Quiesced {
                execution: Err(generation_error(
                    "injected immediate generation launch failure",
                )),
            },
            AbortWorkOutcome::PartialFailure => DrainOutcome::QuiescenceUnproven(generation_error(
                "injected partial generation launch failure",
            )),
        }
    }
}
#[cfg(test)]
impl SemanticsV2ReservedGenerationBuilder for DeterministicAbortBuilder {
    type Candidate = AbortCandidate;
    type Work = AbortWork;

    fn try_reserve_candidate(
        &self,
        reservation: input::GenerationBuilderReservation,
    ) -> Result<Self::Candidate, EngineError> {
        if matches!(
            self.sabotage,
            DeterministicAbortSabotage::CandidateReservationFailure
        ) {
            return Err(generation_error("injected candidate reservation failure"));
        }
        assert_eq!(reservation.tables(), 1);
        assert_eq!(reservation.rows(), 0);
        assert_eq!(reservation.cells(), 0);
        assert_eq!(reservation.value_bytes(), 0);
        assert_eq!(reservation.indexes(), 0);
        assert_eq!(reservation.keys(), 0);
        assert_eq!(reservation.effects(), 0);
        assert_eq!(reservation.effect_values(), 0);
        assert_eq!(reservation.effect_value_bytes(), 0);
        assert_eq!(reservation.table_outputs(), 1);
        assert_eq!(reservation.index_outputs(), 0);
        Ok(AbortCandidate(std::sync::Arc::clone(&self.drops)))
    }

    fn launch(
        self,
        mut launch: input::ReservedGenerationLaunch<Self::Candidate, Self::Work>,
    ) -> LaunchedAttempt<Self::Candidate, Self::Work> {
        let neutral = launch.input().neutral_view();
        assert_eq!(neutral.tables().count(), 1);
        assert_eq!(neutral.rows().count(), 0);
        assert_eq!(neutral.cells().count(), 0);
        assert_eq!(neutral.indexes().count(), 0);
        assert_eq!(neutral.effects().count(), 0);
        assert_ne!(neutral.identity().catalog_epoch, 0);
        if let Some(observed) = &self.observed_input_digest {
            *observed.lock().expect("test input-digest observation lock") = Some(
                launch
                    .input()
                    .builder_input_digest()
                    .expect("sealed input remains addressable before launch"),
            );
        }
        if matches!(self.sabotage, DeterministicAbortSabotage::NeutralInput) {
            launch.sabotage_neutral_input();
        }
        if !matches!(self.sabotage, DeterministicAbortSabotage::MissingOutput) {
            launch.write_abort_echo();
        }
        match self.sabotage {
            DeterministicAbortSabotage::None
            | DeterministicAbortSabotage::CandidateReservationFailure
            | DeterministicAbortSabotage::MissingOutput
            | DeterministicAbortSabotage::ImmediateLaunchFailure
            | DeterministicAbortSabotage::PartialLaunchFailure => {}
            DeterministicAbortSabotage::DuplicateOutput => launch.write_abort_echo(),
            DeterministicAbortSabotage::ExtraOutput => launch.outputs_mut().extra_write = true,
            DeterministicAbortSabotage::HeaderIdentity => {
                launch.outputs_mut().header.database_id[0] ^= 1;
            }
            DeterministicAbortSabotage::HeaderInputDigest => {
                launch.outputs_mut().header.generation_input_digest[0] ^= 1;
            }
            DeterministicAbortSabotage::HeaderFinalRoot => {
                launch.outputs_mut().header.final_database_root[0] ^= 1;
            }
            DeterministicAbortSabotage::TableIdentity => {
                launch.outputs_mut().tables[0].stable_table_id ^= 1;
            }
            DeterministicAbortSabotage::TableGeneration => {
                launch.outputs_mut().tables[0].final_data_generation ^= 1;
            }
            DeterministicAbortSabotage::TableRoot => {
                launch.outputs_mut().tables[0].final_table_root[0] ^= 1;
            }
            DeterministicAbortSabotage::TableCount => {
                launch.outputs_mut().tables[0].final_logical_row_count ^= 1;
            }
            DeterministicAbortSabotage::TableIndexRange => {
                launch.outputs_mut().tables[0].index_count ^= 1;
            }
            DeterministicAbortSabotage::NeutralInput => {}
        }
        let outcome = match self.sabotage {
            DeterministicAbortSabotage::ImmediateLaunchFailure => {
                AbortWorkOutcome::ImmediateFailure
            }
            DeterministicAbortSabotage::PartialLaunchFailure => AbortWorkOutcome::PartialFailure,
            _ => AbortWorkOutcome::Complete,
        };
        LaunchedAttempt::launched(
            launch,
            AbortWork {
                calls: self.calls,
                outcome,
            },
        )
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Clone, Copy)]
    enum TestDrain {
        Ok,
        ExecutionError,
        Unproven,
        Panic,
    }

    struct TestCandidate(Arc<AtomicUsize>);

    impl Drop for TestCandidate {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct TestWork {
        calls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        outcome: TestDrain,
    }

    impl Drop for TestWork {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl sealed::Candidate for TestCandidate {}
    impl sealed::Work for TestWork {}

    impl SemanticsV2ReservedGenerationWork for TestWork {
        fn drain_once(&mut self) -> DrainOutcome {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.outcome {
                TestDrain::Ok => DrainOutcome::Quiesced { execution: Ok(()) },
                TestDrain::ExecutionError => DrainOutcome::Quiesced {
                    execution: Err(generation_error("injected immediate execution failure")),
                },
                TestDrain::Unproven => DrainOutcome::QuiescenceUnproven(generation_error(
                    "injected unknown quiescence",
                )),
                TestDrain::Panic => panic!("injected generation drain panic"),
            }
        }
    }

    fn test_attempt(
        registry: &input::GenerationQuarantineRegistry<TestCandidate, TestWork>,
        drain: TestDrain,
    ) -> (
        LaunchedAttempt<TestCandidate, TestWork>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let calls = Arc::new(AtomicUsize::new(0));
        let candidate_drops = Arc::new(AtomicUsize::new(0));
        let work_drops = Arc::new(AtomicUsize::new(0));
        let launch = input::test_launch::<TestCandidate, TestWork>(
            TestCandidate(Arc::clone(&candidate_drops)),
            registry,
        )
        .expect("test neutral launch reserves every fixed owner");
        (
            LaunchedAttempt::launched(
                launch,
                TestWork {
                    calls: Arc::clone(&calls),
                    drops: Arc::clone(&work_drops),
                    outcome: drain,
                },
            ),
            calls,
            candidate_drops,
            work_drops,
        )
    }

    fn park_unknown_with_retention_sentinels(
        registry: &input::GenerationQuarantineRegistry<TestCandidate, TestWork>,
        candidate_drops: Arc<AtomicUsize>,
        work_drops: Arc<AtomicUsize>,
        input_drops: Arc<AtomicUsize>,
        output_drops: Arc<AtomicUsize>,
    ) {
        let launch = input::test_launch_with_retention_sentinels(
            TestCandidate(candidate_drops),
            registry,
            input_drops,
            output_drops,
        )
        .expect("test launch reserves and registers its exact ticket");
        let mut attempt = LaunchedAttempt::launched(
            launch,
            TestWork {
                calls: Arc::new(AtomicUsize::new(0)),
                drops: work_drops,
                outcome: TestDrain::Unproven,
            },
        );
        assert!(matches!(
            attempt.drain_once(),
            DrainOutcome::QuiescenceUnproven(_)
        ));
        drop(attempt);
    }

    #[test]
    fn reservation_failures_cover_every_neutral_and_output_owner_before_launch() {
        // `test_launch` executes the same thirteen direct production fill reservations: all
        // neutral flats, exact table/index output owners, and the real one-slot
        // quarantine ticket plus its separately fallible registry entry.
        for attempt in 1..=13 {
            let drops = Arc::new(AtomicUsize::new(0));
            let failed = input::fail_reservation_at(attempt, || {
                let registry = input::GenerationQuarantineRegistry::new();
                input::test_launch::<TestCandidate, TestWork>(
                    TestCandidate(Arc::clone(&drops)),
                    &registry,
                )
            });
            assert!(
                failed.is_err(),
                "reservation family {attempt} rejects before launch"
            );
            assert_eq!(
                drops.load(Ordering::SeqCst),
                1,
                "candidate returns on prelaunch failure"
            );
            let registry = input::GenerationQuarantineRegistry::new();
            input::test_launch::<TestCandidate, TestWork>(
                TestCandidate(Arc::new(AtomicUsize::new(0))),
                &registry,
            )
            .expect("same reservation family retries cleanly");
        }
    }

    #[test]
    fn explicit_and_drop_drain_are_exactly_once() {
        let registry = input::GenerationQuarantineRegistry::new();
        let (mut explicit, calls, drops, work_drops) = test_attempt(&registry, TestDrain::Ok);
        assert!(matches!(
            explicit.drain_once(),
            DrainOutcome::Quiesced { execution: Ok(()) }
        ));
        assert!(matches!(
            explicit.drain_once(),
            DrainOutcome::QuiescenceUnproven(_)
        ));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "a second call cannot retry work"
        );
        drop(explicit);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "drop does not retry explicit drain"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "proven completion releases candidate"
        );
        assert_eq!(work_drops.load(Ordering::SeqCst), 1);

        let (armed, calls, drops, work_drops) = test_attempt(&registry, TestDrain::Ok);
        drop(armed);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "armed drop drains once");
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "armed proven drain releases candidate"
        );
        assert_eq!(work_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn execution_failure_is_quiesced_but_unknown_quiescence_parks_every_owner() {
        let registry = input::GenerationQuarantineRegistry::new();
        let (mut failed, calls, drops, work_drops) =
            test_attempt(&registry, TestDrain::ExecutionError);
        assert!(matches!(
            failed.drain_once(),
            DrainOutcome::Quiesced { execution: Err(_) }
        ));
        drop(failed);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "execution failure drains once"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "quiesced failure releases candidate"
        );
        assert_eq!(work_drops.load(Ordering::SeqCst), 1);

        let (mut unknown, calls, drops, work_drops) = test_attempt(&registry, TestDrain::Unproven);
        assert!(matches!(
            unknown.drain_once(),
            DrainOutcome::QuiescenceUnproven(_)
        ));
        drop(unknown);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "unknown quiescence is never retried"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "unknown quiescence quarantines candidate"
        );
        assert_eq!(work_drops.load(Ordering::SeqCst), 0);
        let reaper = registry.authorized_reaper_for_test();
        assert_eq!(reaper.occupancy().occupied, 1);
        reaper.reap_after_external_quiescence();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(work_drops.load(Ordering::SeqCst), 1);
        assert_eq!(registry.authorized_reaper_for_test().occupancy().vacant, 1);

        let (armed_unknown, calls, drops, work_drops) =
            test_attempt(&registry, TestDrain::Unproven);
        drop(armed_unknown);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "armed unknown drain runs once"
        );
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "drop-path unknown keeps candidate parked"
        );
        assert_eq!(work_drops.load(Ordering::SeqCst), 0);
        let reaper = registry.authorized_reaper_for_test();
        assert_eq!(reaper.occupancy().occupied, 1);
        reaper.reap_after_external_quiescence();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(work_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn registry_retains_all_four_owners_until_authorized_reap() {
        let registry = input::GenerationQuarantineRegistry::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let candidate_drops = Arc::new(AtomicUsize::new(0));
        let work_drops = Arc::new(AtomicUsize::new(0));
        let input_drops = Arc::new(AtomicUsize::new(0));
        let output_drops = Arc::new(AtomicUsize::new(0));
        let launch = input::test_launch_with_retention_sentinels(
            TestCandidate(Arc::clone(&candidate_drops)),
            &registry,
            Arc::clone(&input_drops),
            Arc::clone(&output_drops),
        )
        .expect("test launch reserves and registers its exact ticket");
        let mut attempt = LaunchedAttempt::launched(
            launch,
            TestWork {
                calls: Arc::clone(&calls),
                drops: Arc::clone(&work_drops),
                outcome: TestDrain::Unproven,
            },
        );
        assert!(matches!(
            attempt.drain_once(),
            DrainOutcome::QuiescenceUnproven(_)
        ));
        drop(attempt);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 0);
        assert_eq!(work_drops.load(Ordering::SeqCst), 0);
        assert_eq!(input_drops.load(Ordering::SeqCst), 0);
        assert_eq!(output_drops.load(Ordering::SeqCst), 0);
        let reaper = registry.authorized_reaper_for_test();
        let occupancy = reaper.occupancy();
        assert_eq!(occupancy.armed, 0);
        assert_eq!(occupancy.occupied, 1);
        assert_eq!(occupancy.work_owner, 1);
        assert_eq!(occupancy.input_owner, 1);
        assert_eq!(occupancy.output_owner, 1);
        assert_eq!(occupancy.candidate_owner, 1);
        reaper.reap_after_external_quiescence();
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 1);
        assert_eq!(work_drops.load(Ordering::SeqCst), 1);
        assert_eq!(input_drops.load(Ordering::SeqCst), 1);
        assert_eq!(output_drops.load(Ordering::SeqCst), 1);
        assert_eq!(registry.authorized_reaper_for_test().occupancy().vacant, 1);
    }

    #[test]
    fn proof_observation_cannot_reap_an_armed_ticket_that_parks_afterward() {
        let registry = input::GenerationQuarantineRegistry::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let candidate_drops = Arc::new(AtomicUsize::new(0));
        let work_drops = Arc::new(AtomicUsize::new(0));
        let input_drops = Arc::new(AtomicUsize::new(0));
        let output_drops = Arc::new(AtomicUsize::new(0));
        let launch = input::test_launch_with_retention_sentinels(
            TestCandidate(Arc::clone(&candidate_drops)),
            &registry,
            Arc::clone(&input_drops),
            Arc::clone(&output_drops),
        )
        .expect("test launch reserves and registers its exact ticket");
        let mut armed_attempt = LaunchedAttempt::launched(
            launch,
            TestWork {
                calls: Arc::clone(&calls),
                drops: Arc::clone(&work_drops),
                outcome: TestDrain::Unproven,
            },
        );

        // The ticket exists but is Armed, so this observed proof must carry no release marker
        // for it even when the same ticket becomes Occupied afterwards.
        let stale_reaper = registry.authorized_reaper_for_test();
        assert_eq!(stale_reaper.occupancy().armed, 1);
        assert!(matches!(
            armed_attempt.drain_once(),
            DrainOutcome::QuiescenceUnproven(_)
        ));
        drop(armed_attempt);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        stale_reaper.reap_after_external_quiescence();
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 0);
        assert_eq!(work_drops.load(Ordering::SeqCst), 0);
        assert_eq!(input_drops.load(Ordering::SeqCst), 0);
        assert_eq!(output_drops.load(Ordering::SeqCst), 0);

        let fresh_reaper = registry.authorized_reaper_for_test();
        let occupancy = fresh_reaper.occupancy();
        assert_eq!(occupancy.occupied, 1);
        assert_eq!(occupancy.work_owner, 1);
        assert_eq!(occupancy.input_owner, 1);
        assert_eq!(occupancy.output_owner, 1);
        assert_eq!(occupancy.candidate_owner, 1);
        fresh_reaper.reap_after_external_quiescence();
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 1);
        assert_eq!(work_drops.load(Ordering::SeqCst), 1);
        assert_eq!(input_drops.load(Ordering::SeqCst), 1);
        assert_eq!(output_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn proof_cutoff_cannot_reap_a_generation_parked_after_its_observation() {
        let registry = input::GenerationQuarantineRegistry::new();
        let first_candidate_drops = Arc::new(AtomicUsize::new(0));
        let first_work_drops = Arc::new(AtomicUsize::new(0));
        let first_input_drops = Arc::new(AtomicUsize::new(0));
        let first_output_drops = Arc::new(AtomicUsize::new(0));
        park_unknown_with_retention_sentinels(
            &registry,
            Arc::clone(&first_candidate_drops),
            Arc::clone(&first_work_drops),
            Arc::clone(&first_input_drops),
            Arc::clone(&first_output_drops),
        );

        // This proof observes only generation one.  The later ticket gets a new exact
        // generation even though the registry remains the same typed service.
        let stale_reaper = registry.authorized_reaper_for_test();
        let second_candidate_drops = Arc::new(AtomicUsize::new(0));
        let second_work_drops = Arc::new(AtomicUsize::new(0));
        let second_input_drops = Arc::new(AtomicUsize::new(0));
        let second_output_drops = Arc::new(AtomicUsize::new(0));
        park_unknown_with_retention_sentinels(
            &registry,
            Arc::clone(&second_candidate_drops),
            Arc::clone(&second_work_drops),
            Arc::clone(&second_input_drops),
            Arc::clone(&second_output_drops),
        );
        assert_eq!(stale_reaper.occupancy().occupied, 2);

        stale_reaper.reap_after_external_quiescence();
        assert_eq!(first_candidate_drops.load(Ordering::SeqCst), 1);
        assert_eq!(first_work_drops.load(Ordering::SeqCst), 1);
        assert_eq!(first_input_drops.load(Ordering::SeqCst), 1);
        assert_eq!(first_output_drops.load(Ordering::SeqCst), 1);
        assert_eq!(second_candidate_drops.load(Ordering::SeqCst), 0);
        assert_eq!(second_work_drops.load(Ordering::SeqCst), 0);
        assert_eq!(second_input_drops.load(Ordering::SeqCst), 0);
        assert_eq!(second_output_drops.load(Ordering::SeqCst), 0);
        let remaining = registry.authorized_reaper_for_test().occupancy();
        assert_eq!(remaining.occupied, 1);
        assert_eq!(remaining.work_owner, 1);
        assert_eq!(remaining.input_owner, 1);
        assert_eq!(remaining.output_owner, 1);
        assert_eq!(remaining.candidate_owner, 1);

        registry
            .authorized_reaper_for_test()
            .reap_after_external_quiescence();
        assert_eq!(second_candidate_drops.load(Ordering::SeqCst), 1);
        assert_eq!(second_work_drops.load(Ordering::SeqCst), 1);
        assert_eq!(second_input_drops.load(Ordering::SeqCst), 1);
        assert_eq!(second_output_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn final_registry_teardown_without_proof_keeps_occupied_backing() {
        let candidate_drops = Arc::new(AtomicUsize::new(0));
        let work_drops = Arc::new(AtomicUsize::new(0));
        let input_drops = Arc::new(AtomicUsize::new(0));
        let output_drops = Arc::new(AtomicUsize::new(0));
        {
            let registry = input::GenerationQuarantineRegistry::new();
            let launch = input::test_launch_with_retention_sentinels(
                TestCandidate(Arc::clone(&candidate_drops)),
                &registry,
                Arc::clone(&input_drops),
                Arc::clone(&output_drops),
            )
            .expect("test launch reserves and registers its exact ticket");
            let mut attempt = LaunchedAttempt::launched(
                launch,
                TestWork {
                    calls: Arc::new(AtomicUsize::new(0)),
                    drops: Arc::clone(&work_drops),
                    outcome: TestDrain::Unproven,
                },
            );
            assert!(matches!(
                attempt.drain_once(),
                DrainOutcome::QuiescenceUnproven(_)
            ));
            drop(attempt);
            // `registry` is deliberately the final service handle here; no reaper/proof exists.
        }
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 0);
        assert_eq!(work_drops.load(Ordering::SeqCst), 0);
        assert_eq!(input_drops.load(Ordering::SeqCst), 0);
        assert_eq!(output_drops.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn panic_after_drain_is_parked_before_explicit_or_drop_unwind_resumes() {
        let registry = input::GenerationQuarantineRegistry::new();
        let (mut explicit, calls, candidate_drops, work_drops) =
            test_attempt(&registry, TestDrain::Panic);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            explicit.drain_once();
        }))
        .is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 0);
        assert_eq!(work_drops.load(Ordering::SeqCst), 0);
        assert_eq!(
            registry.authorized_reaper_for_test().occupancy().panicked,
            1
        );
        drop(explicit);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "panic path cannot retry work"
        );
        let reaper = registry.authorized_reaper_for_test();
        reaper.reap_after_external_quiescence();
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 1);
        assert_eq!(work_drops.load(Ordering::SeqCst), 1);

        let (armed, calls, candidate_drops, work_drops) = test_attempt(&registry, TestDrain::Panic);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(armed))).is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 0);
        assert_eq!(work_drops.load(Ordering::SeqCst), 0);
        let reaper = registry.authorized_reaper_for_test();
        assert_eq!(reaper.occupancy().panicked, 1);
        reaper.reap_after_external_quiescence();
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 1);
        assert_eq!(work_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_builder_has_only_neutral_abort_material() {
        let calls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let builder = DeterministicAbortBuilder::new(Arc::clone(&calls), Arc::clone(&drops));
        let registry = input::GenerationQuarantineRegistry::new();
        let candidate = builder
            .try_reserve_candidate(input::test_builder_reservation())
            .expect("test candidate reservation succeeds");
        let launch = input::test_launch::<AbortCandidate, AbortWork>(candidate, &registry)
            .expect("test launch reserves");
        let mut attempt = builder.launch(launch);
        assert!(matches!(
            attempt.drain_once(),
            DrainOutcome::Quiesced { execution: Ok(()) }
        ));
        drop(attempt);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn flat_index_outputs_reject_reorder_duplicate_and_substitution() {
        let retained = [
            retained_index(11, 2, [0x11; 32]),
            retained_index(12, 3, [0x12; 32]),
        ];
        let valid = [
            output_index(11, 2, [0x11; 32]),
            output_index(12, 3, [0x12; 32]),
        ];
        assert!(validate_flat_index_outputs(&retained, &valid, false).is_ok());
        for outputs in [
            [
                output_index(12, 3, [0x12; 32]),
                output_index(11, 2, [0x11; 32]),
            ],
            [
                output_index(11, 2, [0x11; 32]),
                output_index(11, 3, [0x12; 32]),
            ],
            [
                output_index(11, 9, [0x11; 32]),
                output_index(12, 3, [0x12; 32]),
            ],
            [
                output_index(11, 2, [0x44; 32]),
                output_index(12, 3, [0x12; 32]),
            ],
        ] {
            assert!(
                validate_flat_index_outputs(&retained, &outputs, false).is_err(),
                "flat index reorder/duplicate/substitution rejects"
            );
        }
        let abort_retained = [retained_index(11, 2, [0x11; 32])];
        assert!(validate_flat_index_outputs(
            &abort_retained,
            &[output_index(11, 2, [0x11; 32])],
            true,
        )
        .is_ok());
    }

    #[test]
    fn packed_output_ranges_ignore_parent_only_global_descriptor_gaps_and_reject_exhaustion() {
        let parent_only_descriptor_global_slot = 0_u32;
        let table_owned_global_start = parent_only_descriptor_global_slot + 1;
        let mut output = input::GenerationOutputTable {
            filled: true,
            duplicate: false,
            stable_table_id: 7,
            final_data_generation: 2,
            final_table_root: [7; 32],
            final_logical_row_count: 1,
            index_start: 0,
            index_count: 1,
        };
        assert_eq!(table_owned_global_start, 1);
        assert!(
            validate_packed_output_index_range(0, 1, &output).is_ok(),
            "the table's first packed output remains zero despite its global S7 descriptor gap",
        );
        output.index_start = table_owned_global_start;
        assert!(validate_packed_output_index_range(0, 1, &output).is_err());
        output.index_start = u32::MAX;
        output.index_count = 1;
        assert!(validate_packed_output_index_range(u32::MAX, 1, &output).is_err());
        assert!(validate_packed_index_output_exhaustion(2, 2).is_ok());
        assert!(
            validate_packed_index_output_exhaustion(2, 3).is_err(),
            "an extra flat output tail is not accepted",
        );
        assert!(
            validate_packed_index_output_exhaustion(2, 1).is_err(),
            "a missing flat output tail is not accepted",
        );
    }

    fn output_index(
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

    fn retained_index(
        stable_index_id: u64,
        final_index_generation: u64,
        final_index_root: CanonicalDigest,
    ) -> super::super::graph::RetainedIndexDescriptor {
        super::super::graph::RetainedIndexDescriptor {
            index_ref: 0,
            owner_table_ref: 0,
            raw_catalog_ordinal: 0,
            stable_index_id,
            display_oid: 1,
            stable_constraint_id: 0,
            constraint_display_oid: 0,
            flags: 0,
            null_equality_policy: 0,
            key_start: 0,
            key_count: 0,
            catalog_epoch: 1,
            owner_stable_table_id: 1,
            owner_display_oid: 1,
            owner_schema_digest: [1; 32],
            owner_name_digest: [2; 32],
            index_name_digest: [3; 32],
            constraint_name_digest: [0; 32],
            owner_table_base_root: [4; 32],
            base_index_root: final_index_root,
            final_index_root,
            descriptor_digest: [5; 32],
            owner_data_generation: 1,
            base_index_generation: final_index_generation,
            final_index_generation,
        }
    }

    #[test]
    fn source_boundary_forbids_borrowed_witnesses_and_live_builder_paths() {
        let retained = include_str!("../retained.rs");
        let generation = include_str!("generation_validation.rs");
        let golden_builder = include_str!("generation_validation/golden_builder.rs");
        let reencode = include_str!("reencode.rs");
        let q2_witnesses = include_str!("../goldens/q2_witnesses.rs");
        let q2_reencode = include_str!("../goldens/q2_reencode.rs");
        let typed_batch = include_str!("../../../typed_insert_batch.rs");
        assert!(generation.contains("struct LaunchedAttempt"));
        assert!(generation.contains("fn drain_once(&mut self) -> DrainOutcome"));
        assert!(generation.contains("struct OwnedGenerationResult<C>"));
        assert!(generation.contains("type Candidate: sealed::Candidate + 'static"));
        assert!(generation.contains("type Work: SemanticsV2ReservedGenerationWork + 'static"));
        let lifetime_builder = ["SemanticsV2ReservedGenerationBuilder", "<'"].concat();
        assert!(!generation.contains(&lifetime_builder));
        let borrowed_witness = ["SemanticsV2", "GenerationWitness"].concat();
        assert!(!generation.contains(&borrowed_witness));
        let consuming_build = ["fn bu", "ild("].concat();
        assert!(!generation.contains(&consuming_build));
        let pending_getter = ["fn pen", "ding(&self)"].concat();
        let witness_getter = ["fn wit", "ness(&self)"].concat();
        assert!(!generation.contains(&pending_getter));
        assert!(!generation.contains(&witness_getter));
        assert!(!retained.contains("FullyWitnessValidatedSemanticsV2<'"));
        assert!(!retained.contains("generation_witness_for_test"));
        assert!(
            retained.contains("#[cfg(test)]\n#[path = \"retained/reencode.rs\"]\nmod reencode;")
        );
        assert!(generation.contains(
            "#[cfg(test)]\n#[path = \"generation_validation/golden_builder.rs\"]\npub(super) mod golden_builder;"
        ));
        assert!(retained.contains("pub(super) fn reencode_s1_s7_for_test(&self)"));
        assert!(reencode.contains("owner: &FullyWitnessValidatedSemanticsV2<C>"));
        assert!(!reencode.contains("GenerationPendingSemanticsV2"));
        assert!(!reencode.contains("QuarantinedSemanticsV2"));
        assert!(!reencode.contains("CodecClosedSemanticsV2"));
        assert!(golden_builder.contains("enum GoldenCase"));
        assert!(golden_builder.contains("fn try_reserve_candidate"));
        assert!(golden_builder.contains("fn launch("));
        assert!(!golden_builder.contains("Q1Fixture"));
        assert!(!golden_builder.contains("ReservedSemanticsV2Graph"));
        assert!(typed_batch
            .contains("#[cfg(test)]\npub(crate) use typed_image_codec::{encode_typed_image"));

        // The two test-only typed reencoders have exactly one definition and exactly one call
        // from the retained logical reencoder.  No second generic byte bridge may appear.
        for bridge in [
            "reencode_decoded_canonical_typed_insert_record_for_test",
            "reencode_decoded_typed_image_for_test",
        ] {
            assert_eq!(
                typed_batch
                    .matches(&format!("#[cfg(test)]\npub(crate) fn {bridge}("))
                    .count(),
                1,
                "{bridge} has one cfg(test) bridge definition"
            );
            assert_eq!(
                reencode.matches(&format!("{bridge}(")).count(),
                1,
                "{bridge} has one retained reencoder call"
            );
        }

        let without_line_comments = |source: &str| {
            source
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let golden_builder_code = without_line_comments(golden_builder);
        let q2_witness_code = without_line_comments(q2_witnesses);
        for forbidden in [
            "Q1",
            "q1_vectors",
            "DecodedTyped",
            "decode_canonical",
            "MINIMAL_ABORT_SECTION",
            "SECTION_HEX",
            "q1_sabotage",
            "frozen:",
            "sections:",
        ] {
            assert!(
                !q2_witness_code.contains(forbidden),
                "Q2 catalog/lease witnesses must not receive {forbidden} input"
            );
        }
        assert!(
            golden_builder_code.contains("pub(super) fn new(case: GoldenCase) -> Self"),
            "the closed case tag is the builder's only construction input"
        );
        for forbidden in [
            "fn new(case: GoldenCase,",
            "Q1",
            "q1_vectors",
            "SUCCESS_GENERATION_INPUT_DIGEST",
            "expected_generation_input_digest",
            "expected_sections",
            "expected_s7",
            "frozen_sections",
            "SECTION_HEX",
            "SECTION_ROOT_HEX",
        ] {
            assert!(
                !golden_builder_code.contains(forbidden),
                "Q2 builder must not receive {forbidden} input"
            );
        }
        assert_eq!(
            q2_reencode
                .matches("_reencodes_from_fully_validated_owner_to_frozen_literals")
                .count(),
            3,
            "Q2 has exactly the minimal, explicit-abort, and successful positive reencodes"
        );

        let q2_retained_facade = retained
            .split("/// Consume an actual codec-closed owner through the one catalog/allocator transition while the")
            .nth(1)
            .and_then(|tail| {
                tail.split(
                    "/// Consume a real production-shaped catalog/allocator owner through the one durable sequence",
                )
                .next()
            })
            .expect("retained keeps the bounded Q2 facade");
        let q2_typed_reencode_bridges = typed_batch
            .split("/// Test-only canonical S2 reencoder for an already validated, move-only decoded record.")
            .nth(1)
            .and_then(|tail| tail.split("/// Test-only bridge to the canonical private sequence-chain fixture.").next())
            .expect("typed batch keeps the bounded Q2 reconstruction bridges");
        for (source_name, source) in [
            ("Q2 witnesses", q2_witnesses),
            ("Q2 reencode evidence", q2_reencode),
            ("Q2 golden builder", golden_builder),
            ("retained logical reencoder", reencode),
            ("retained Q2 facade", q2_retained_facade),
            ("typed Q2 reencode bridges", q2_typed_reencode_bridges),
        ] {
            for forbidden in [
                "S8_",
                "S8_BYTES",
                "S8_MAGIC",
                "AggregateReplayTxn",
                "compile_replay",
                "fn replay",
                "replay_",
                "struct Replay",
                "fn apply",
                "apply_",
                "struct Apply",
                "fn recover",
                "recovery_",
                "struct Recovery",
                "fn publish",
                "publication_",
                "publish_generation",
                "struct Publication",
                "live_builder",
                "LiveBuilder",
                "crate::wal_binary",
                "PreparedBinaryInsertTemplate",
                "WalBuffer",
            ] {
                assert!(
                    !source.contains(forbidden),
                    "{source_name} must not grow a concrete {forbidden} symbol"
                );
            }
            assert!(
                !source.contains("pub fn "),
                "{source_name} must not expose a public Q2 execution surface"
            );
        }
    }
}

#[cfg(test)]
#[path = "generation_validation/fixture_tests.rs"]
mod fixture_tests;
#[cfg(test)]
#[path = "generation_validation/golden_builder.rs"]
pub(super) mod golden_builder;
