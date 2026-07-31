//! Sealed, already-reserved generation-builder boundary for semantics v2.
//!
//! This leaf owns neither a codec/S7 reader nor a publication capability.  Its only input is the
//! move-only `GenerationPendingSemanticsV2` state produced after catalog and durable-allocator
//! validation.  A sealed builder consumes that state, and this module compares its opaque output
//! to independently recomputed, reference-neutral generation inputs before releasing the final
//! typestate.

#[path = "generation_validation/input.rs"]
mod input;

use super::{
    CatalogAndAllocatorValidated, FullyValidatedWitnesses, FullyWitnessValidatedSemanticsV2,
    GenerationPendingSemanticsV2, SemanticsV2GenerationTableWitness, SemanticsV2GenerationWitness,
};
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedTable,
};
use crate::EngineError;

type CanonicalDigest = gpu_db_wal::CanonicalDigest;

const ROOT_DESCRIPTOR_VERSION: u16 = 1;

mod sealed {
    pub trait Builder {}
    pub trait Work {}
}

/// Builder-owned work must be synchronized before a graph owner may be released.  The public
/// implementation surface is sealed: an arbitrary decoder, test, or future reencoder cannot
/// manufacture a completion or skip the drain.
#[allow(dead_code)]
pub(super) trait SemanticsV2ReservedGenerationWork: sealed::Work {
    fn drain(self) -> Result<(), EngineError>;
}

/// The only builder interface for the second retained typestate transition.  It takes the
/// pending owner by value, so a second build or a direct generation-witness comparison cannot be
/// expressed after a first attempt.
#[allow(dead_code)]
pub(super) trait SemanticsV2ReservedGenerationBuilder<'a>: sealed::Builder {
    type Work: SemanticsV2ReservedGenerationWork;

    /// If this returns `Err`, the implementation has already drained every launched operation.
    /// If it returns a completion, this module drains it on both validation success and failure.
    fn build(
        self,
        pending: GenerationPendingSemanticsV2<'a>,
    ) -> Result<SemanticsV2ReservedGenerationBuild<'a, Self::Work>, EngineError>;
}

/// Opaque builder completion.  Its only constructor is private to this module, and its `Drop`
/// path is a final best-effort drain guard for an abandoned validation attempt.
#[allow(dead_code)]
pub(super) struct SemanticsV2ReservedGenerationBuild<'a, W: SemanticsV2ReservedGenerationWork> {
    pending: Option<GenerationPendingSemanticsV2<'a>>,
    witness: Option<SemanticsV2GenerationWitness<'a>>,
    work: Option<W>,
}

impl<'a, W: SemanticsV2ReservedGenerationWork> SemanticsV2ReservedGenerationBuild<'a, W> {
    fn builder_result(
        pending: GenerationPendingSemanticsV2<'a>,
        witness: SemanticsV2GenerationWitness<'a>,
        work: W,
    ) -> Self {
        Self {
            pending: Some(pending),
            witness: Some(witness),
            work: Some(work),
        }
    }

    fn pending(&self) -> &GenerationPendingSemanticsV2<'a> {
        self.pending
            .as_ref()
            .expect("sealed generation completion retains its pending owner")
    }

    fn witness(&self) -> &SemanticsV2GenerationWitness<'a> {
        self.witness
            .as_ref()
            .expect("sealed generation completion retains its builder witness")
    }

    fn drain(&mut self) -> Result<(), EngineError> {
        self.work
            .take()
            .expect("sealed generation completion drains exactly once")
            .drain()
    }

    fn into_parts(
        mut self,
    ) -> (
        GenerationPendingSemanticsV2<'a>,
        SemanticsV2GenerationWitness<'a>,
    ) {
        assert!(
            self.work.is_none(),
            "sealed generation completion must drain before releasing owners"
        );
        (
            self.pending
                .take()
                .expect("sealed generation completion keeps a pending owner"),
            self.witness
                .take()
                .expect("sealed generation completion keeps a builder witness"),
        )
    }
}

impl<W: SemanticsV2ReservedGenerationWork> Drop for SemanticsV2ReservedGenerationBuild<'_, W> {
    fn drop(&mut self) {
        if let Some(work) = self.work.take() {
            let _ = work.drain();
        }
    }
}

/// Consume a pending graph through a builder result.  Every return path after a successful
/// launch explicitly drains work before dropping either the graph or builder witness.
pub(super) fn validate_reserved_builder_result<'a, B>(
    pending: GenerationPendingSemanticsV2<'a>,
    builder: B,
) -> Result<FullyWitnessValidatedSemanticsV2<'a>, EngineError>
where
    B: SemanticsV2ReservedGenerationBuilder<'a>,
{
    let mut completed = builder.build(pending)?;
    let validation = validate_witness(completed.pending(), completed.witness());
    let drain = completed.drain();
    match (validation, drain) {
        (Ok(()), Ok(())) => {
            let (pending, generation) = completed.into_parts();
            let CatalogAndAllocatorValidated {
                catalog,
                allocator_index: _,
            } = pending.catalog_and_allocator;
            Ok(FullyWitnessValidatedSemanticsV2 {
                graph: pending.graph,
                witnesses: FullyValidatedWitnesses {
                    catalog,
                    generation,
                },
            })
        }
        (Err(validation), Ok(())) => Err(validation),
        (Ok(()), Err(drain)) => Err(drain),
        (Err(validation), Err(drain)) => Err(generation_error(&format!(
            "generation witness validation failed ({validation}) and builder drain failed ({drain})"
        ))),
    }
}

fn validate_witness(
    pending: &GenerationPendingSemanticsV2<'_>,
    witness: &SemanticsV2GenerationWitness<'_>,
) -> Result<(), EngineError> {
    let identity = pending.graph.identity;
    let graph = &pending.graph.graph;
    let catalog = &pending.catalog_and_allocator.catalog;
    if witness.root_descriptor_version != ROOT_DESCRIPTOR_VERSION
        || witness.root_descriptor_version != graph.header.root_descriptor_version
        || witness.database_id != identity.database_id
        || witness.catalog_epoch != identity.catalog_epoch
        || witness.catalog_digest != identity.catalog_digest
        || witness.stable_transaction_id != identity.stable_transaction_id
        || witness.commit_sequence != identity.commit_sequence
        || witness.initial_database_root != identity.initial_database_root
        || graph.header.catalog_before_epoch != identity.catalog_epoch
        || graph.header.catalog_after_epoch != identity.catalog_epoch
        || graph.header.catalog_before_digest != identity.catalog_digest
        || graph.header.catalog_after_digest != identity.catalog_digest
        || graph.header.initial_database_root != identity.initial_database_root
        || witness.final_database_root != graph.header.final_database_root
        || catalog.database_id != identity.database_id
        || catalog.catalog_epoch != identity.catalog_epoch
        || catalog.catalog_digest != identity.catalog_digest
    {
        return Err(generation_error(
            "generation witness identity does not match the retained envelope",
        ));
    }
    let input_digest = input::generation_input_digest(graph, catalog, identity)?;
    if witness.generation_input_digest != input_digest {
        return Err(generation_error(
            "generation witness input digest does not match retained neutral facts",
        ));
    }
    validate_table_outputs(graph, witness, identity.initial_database_root)?;
    Ok(())
}

fn validate_table_outputs(
    graph: &ReservedSemanticsV2Graph,
    witness: &SemanticsV2GenerationWitness<'_>,
    initial_database_root: CanonicalDigest,
) -> Result<(), EngineError> {
    if witness.tables.len() != graph.tables.len() {
        return Err(generation_error(
            "generation witness table count does not exactly match retained tables",
        ));
    }
    let abort = graph.outcomes.last().is_some_and(|outcome| {
        outcome.outcome.kind == gpu_db_wal::CanonicalOutcomeKind::AbortError
    });
    if abort
        && (!graph.transitions.is_empty()
            || !graph.key_effects.is_empty()
            || graph.header.final_database_root != initial_database_root
            || witness.final_database_root != initial_database_root)
    {
        return Err(generation_error(
            "abort generation form has transitions, effects, or a changed database root",
        ));
    }
    let mut previous_table = None;
    for (table_ordinal, (table, output)) in
        graph.tables.iter().zip(witness.tables.iter()).enumerate()
    {
        if previous_table.is_some_and(|previous| table.stable_table_id <= previous)
            || table.table_ref != table_ordinal as u32
            || table.image_ref != table_ordinal as u32
            || table.stable_table_id != output.stable_table_id
            || table.data_generation_after != output.final_data_generation
            || table.final_table_root != output.final_table_root
            || table.final_logical_row_count != output.final_logical_row_count
        {
            return Err(generation_error(
                "generation witness table identity/order/output differs from S7",
            ));
        }
        validate_index_outputs(graph, table, output, abort)?;
        if abort
            && (table.data_generation_after != table.data_generation_before
                || table.final_table_root != table.initial_table_root
                || table.final_logical_row_count != table.initial_logical_row_count)
        {
            return Err(generation_error(
                "abort generation form changes a table output",
            ));
        }
        previous_table = Some(table.stable_table_id);
    }
    Ok(())
}

fn validate_index_outputs(
    graph: &ReservedSemanticsV2Graph,
    table: &RetainedTable,
    output: &SemanticsV2GenerationTableWitness<'_>,
    abort: bool,
) -> Result<(), EngineError> {
    let indexes = input::range(
        &graph.indexes,
        table.owned_index_start,
        table.owned_index_count,
        "table index",
    )?;
    if output.indexes.len() != indexes.len() {
        return Err(generation_error(
            "generation witness owned-index count differs from S7",
        ));
    }
    let mut previous_index = None;
    for (ordinal, (index, output)) in indexes.iter().zip(output.indexes.iter()).enumerate() {
        if previous_index.is_some_and(|previous| index.stable_index_id <= previous)
            || index.index_ref
                != input::directory_ref(table.owned_index_start, ordinal, "table index")?
            || index.owner_table_ref != table.table_ref
            || index.owner_stable_table_id != table.stable_table_id
            || index.stable_index_id != output.stable_index_id
            || index.final_index_generation != output.final_index_generation
            || index.final_index_root != output.final_index_root
        {
            return Err(generation_error(
                "generation witness index identity/order/output differs from S7",
            ));
        }
        if abort
            && (index.final_index_generation != index.base_index_generation
                || index.final_index_root != index.base_index_root)
        {
            return Err(generation_error(
                "abort generation form changes an index output",
            ));
        }
        previous_index = Some(index.stable_index_id);
    }
    Ok(())
}

fn generation_error(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 generation validation: {message}"
    ))
}

#[cfg(test)]
mod boundary_tests {
    #[test]
    fn builder_boundary_stays_inert_until_the_retained_semantic_closure_exists() {
        let generation = include_str!("generation_validation.rs");
        let retained = include_str!("../retained.rs");
        assert!(generation.contains("trait SemanticsV2ReservedGenerationBuilder"));
        assert!(generation.contains("struct SemanticsV2ReservedGenerationBuild"));
        let builder_impl = format!("{} {}", "impl", "sealed::Builder");
        assert!(
            !generation.contains(&builder_impl),
            "a builder implementation needs the complete retained semantic-closure gate first"
        );
        assert!(
            !retained.contains("generation_witness_for_test"),
            "test code must not manufacture a generation witness outside the sealed builder"
        );
    }
}
