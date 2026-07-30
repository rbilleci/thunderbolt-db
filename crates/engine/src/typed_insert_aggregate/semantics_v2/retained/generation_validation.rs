//! Sealed, already-reserved generation-builder boundary for semantics v2.
//!
//! This leaf owns neither a codec/S7 reader nor a publication capability.  Its only input is the
//! move-only `GenerationPendingSemanticsV2` state produced after catalog and durable-allocator
//! validation.  A sealed builder consumes that state, and this module compares its opaque output
//! to independently recomputed, reference-neutral generation inputs before releasing the final
//! typestate.

use super::{
    FullyValidatedWitnesses, FullyWitnessValidatedSemanticsV2, GenerationPendingSemanticsV2,
    SemanticsV2CatalogIndexWitness, SemanticsV2CatalogTableWitness,
    SemanticsV2GenerationTableWitness, SemanticsV2GenerationWitness,
};
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedIndexDescriptor, RetainedIndexKeyColumn,
    RetainedKeyComponent, RetainedKeyEffect, RetainedTable, RetainedTransition,
};
use crate::typed_insert_batch::{
    DecodedTypedImage, TypedImageRole, TypedInsertColumnValidity, TypedInsertColumnValues,
};
use crate::{EngineError, SqlType};
use sha2::{Digest, Sha256};

type CanonicalDigest = gpu_db_wal::CanonicalDigest;

const ROOT_DESCRIPTOR_VERSION: u16 = 1;
const PHYSICAL_MAINTENANCE_ROLE: u8 = 1;

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
            Ok(FullyWitnessValidatedSemanticsV2 {
                graph: pending.graph,
                witnesses: FullyValidatedWitnesses {
                    catalog: pending.catalog_and_allocator.catalog,
                    allocator_index: pending.catalog_and_allocator.allocator_index,
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
    let catalog = pending.catalog_and_allocator.catalog;
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
    validate_all_effect_components_against_images(graph)?;
    let input_digest = generation_input_digest(graph, catalog, identity)?;
    if witness.generation_input_digest != input_digest {
        return Err(generation_error(
            "generation witness input digest does not match retained neutral facts",
        ));
    }
    validate_table_outputs(graph, witness, identity.initial_database_root)?;
    Ok(())
}

/// Close every S7 typed-key component over the selected final-image cell.  Pass zero proves an
/// arena value hashes to its component; this traversal proves the component cannot be a coherent
/// but unrelated arena value.  Role 1/2 use target-index columns, while role 3 binds the S2 FK
/// child source to its parent supporting-index descriptor.
fn validate_all_effect_components_against_images(
    graph: &ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for (effect_ordinal, effect) in graph.key_effects.iter().enumerate() {
        if effect.effect_ref != directory_ref(0, effect_ordinal, "key effect")?
            || !matches!(effect.role, 1..=3)
            || effect.action != effect.role
        {
            return Err(generation_error(
                "retained key effect does not retain its canonical identity or role",
            ));
        }
        let transition = graph
            .transitions
            .get(effect.transition_ref as usize)
            .ok_or_else(|| generation_error("key effect transition is outside retained graph"))?;
        let table = graph
            .tables
            .get(transition.table_ref as usize)
            .ok_or_else(|| {
                generation_error("key effect transition table is outside retained graph")
            })?;
        let image = graph
            .images
            .get(transition.image_ref as usize)
            .ok_or_else(|| {
                generation_error("key effect transition image is outside retained graph")
            })?;
        if transition.transition_ref != effect.transition_ref
            || transition.table_ref != table.table_ref
            || transition.image_ref != table.image_ref
            || transition.image_row_ordinal >= image.facts().rows
        {
            return Err(generation_error(
                "key effect transition does not close over its table final image",
            ));
        }
        let index = graph
            .indexes
            .get(effect.index_ref as usize)
            .ok_or_else(|| generation_error("key effect index is outside retained graph"))?;
        let components = range(
            &graph.key_components,
            effect.new_component_start,
            effect.new_component_count,
            "key effect component",
        )?;
        let keys = range(
            &graph.index_key_columns,
            index.key_start,
            index.key_count,
            "index key column",
        )?;
        if effect.key_arity != index.key_count
            || components.len() != keys.len()
            || effect.new_component_count != effect.key_arity
        {
            return Err(generation_error(
                "key effect component arity does not match its index descriptor",
            ));
        }
        match effect.role {
            1 | 2 => validate_target_effect_components(
                image, transition, effect, index, components, keys,
            )?,
            3 => validate_foreign_key_effect_components(
                graph, image, transition, effect, index, components, keys,
            )?,
            _ => unreachable!("role is checked above"),
        }
    }
    Ok(())
}

fn validate_target_effect_components(
    image: &DecodedTypedImage,
    transition: &RetainedTransition,
    effect: &RetainedKeyEffect,
    index: &RetainedIndexDescriptor,
    components: &[RetainedKeyComponent],
    keys: &[RetainedIndexKeyColumn],
) -> Result<(), EngineError> {
    if index.owner_table_ref != transition.table_ref {
        return Err(generation_error(
            "maintenance or unique key effect references a non-target index",
        ));
    }
    for (ordinal, (component, key)) in components.iter().zip(keys.iter()).enumerate() {
        if component.effect_ref != effect.effect_ref
            || component.component_ref
                != directory_ref(effect.new_component_start, ordinal, "key effect component")?
            || component.component_ordinal != ordinal as u32
            || component.key_column_ref != key.key_column_ref
            || component.source_catalog_ordinal != key.owner_catalog_column_ordinal
            || component.storage != key.storage
            || component.declared_type_oid != key.declared_type_oid
            || component.signed_type_size != key.signed_type_size
        {
            return Err(generation_error(
                "maintenance or unique key component differs from its target index key",
            ));
        }
        let typed_value = typed_value_digest_from_image(
            image,
            transition.image_row_ordinal,
            component.source_catalog_ordinal,
            component.storage,
            component.declared_type_oid,
            component.signed_type_size,
        )?;
        if typed_value != component.typed_value_digest {
            return Err(generation_error(
                "maintenance or unique key component differs from its final-image cell",
            ));
        }
    }
    Ok(())
}

fn validate_foreign_key_effect_components(
    graph: &ReservedSemanticsV2Graph,
    image: &DecodedTypedImage,
    transition: &RetainedTransition,
    effect: &RetainedKeyEffect,
    index: &RetainedIndexDescriptor,
    components: &[RetainedKeyComponent],
    keys: &[RetainedIndexKeyColumn],
) -> Result<(), EngineError> {
    let record = graph
        .records
        .get(transition.source_statement_ordinal as usize)
        .ok_or_else(|| generation_error("FK key effect has no retained source statement"))?;
    if record.facts().statement_ordinal.as_u32() != transition.source_statement_ordinal {
        return Err(generation_error(
            "FK key effect source statement does not match its retained record",
        ));
    }
    let mut foreign_key = None;
    for candidate in record.foreign_keys() {
        if candidate.raw_ordinal == effect.source_catalog_ordinal
            && foreign_key.replace(candidate).is_some()
        {
            return Err(generation_error(
                "FK key effect selects an ambiguous S2 foreign-key source",
            ));
        }
    }
    let foreign_key = foreign_key.ok_or_else(|| {
        generation_error("FK key effect source ordinal has no S2 foreign-key source")
    })?;
    if index.raw_catalog_ordinal != foreign_key.supporting_index.raw_ordinal
        || index.display_oid != foreign_key.supporting_index.oid
        || index.key_count != foreign_key.supporting_index.key_count
        || components.len() != 1
        || keys.len() != 1
    {
        return Err(generation_error(
            "FK key effect does not match the current single-column S2 FK shape",
        ));
    }
    let mut parent_keys = record.foreign_key_supporting_index_keys(foreign_key.raw_ordinal)?;
    let parent_key = parent_keys
        .next()
        .ok_or_else(|| generation_error("S2 FK has no supporting-index key"))?;
    if parent_keys.next().is_some() {
        return Err(generation_error(
            "S2 FK has more than one supporting-index key for a single-column effect",
        ));
    }
    let component = &components[0];
    let key = &keys[0];
    if component.effect_ref != effect.effect_ref
        || component.component_ref != effect.new_component_start
        || component.component_ordinal != 0
        || component.key_column_ref != key.key_column_ref
        || key.owner_catalog_column_ordinal != parent_key.catalog_column_ordinal
        || key.stable_column_id != parent_key.column_id
        || key.attnum != parent_key.attnum
        || !storage_matches_type(parent_key.ty, key.storage)
        || key.declared_type_oid != parent_key.type_oid
        || key.signed_type_size != parent_key.type_size
        || component.source_catalog_ordinal != foreign_key.child_column.catalog_column_ordinal
        || !storage_matches_type(foreign_key.child_column.ty, component.storage)
        || component.declared_type_oid != foreign_key.child_column.type_oid
        || component.signed_type_size != foreign_key.child_column.type_size
        || component.storage != key.storage
        || component.signed_type_size != key.signed_type_size
    {
        return Err(generation_error(
            "FK key component does not bind its child column and parent supporting key",
        ));
    }
    let typed_value = typed_value_digest_from_image(
        image,
        transition.image_row_ordinal,
        component.source_catalog_ordinal,
        component.storage,
        component.declared_type_oid,
        component.signed_type_size,
    )?;
    if typed_value != component.typed_value_digest {
        return Err(generation_error(
            "FK key component differs from its final-image child cell",
        ));
    }
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
    let indexes = range(
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
            || index.index_ref != directory_ref(table.owned_index_start, ordinal, "table index")?
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

fn generation_input_digest(
    graph: &ReservedSemanticsV2Graph,
    catalog: &super::SemanticsV2CatalogWitness<'_>,
    identity: super::SemanticsV2BoundIdentity,
) -> Result<CanonicalDigest, EngineError> {
    let mut digest = begin_digest(b"gpu-db/write001/generation-input/v2");
    digest.update(identity.database_id);
    digest.update(identity.catalog_epoch.to_le_bytes());
    digest.update(identity.catalog_digest);
    digest.update(identity.stable_transaction_id.to_le_bytes());
    digest.update(identity.commit_sequence.to_le_bytes());
    digest.update(identity.initial_database_root);
    digest.update(count_u32(graph.tables.len(), "generation table count")?.to_le_bytes());
    let mut previous_table = None;
    for (table_ordinal, table) in graph.tables.iter().enumerate() {
        let catalog_table =
            catalog_target_table(catalog, table.stable_table_id, table.display_oid)?;
        if previous_table.is_some_and(|previous| table.stable_table_id <= previous)
            || table.table_ref != table_ordinal as u32
            || table.image_ref != table_ordinal as u32
            || table.stable_table_id != catalog_table.stable_table_id
            || table.display_oid != catalog_table.display_oid
            || table.catalog_epoch != identity.catalog_epoch
            || table.schema_digest != catalog_table.schema_digest
            || table.data_generation_before != catalog_table.data_generation
            || table.initial_table_root != catalog_table.data_root
            || table.catalog_column_count as usize != catalog_table.catalog_columns.len()
        {
            return Err(generation_error(
                "retained table order or catalog columns drifted during generation digest",
            ));
        }
        digest.update(table.stable_table_id.to_le_bytes());
        digest.update(table.data_generation_before.to_le_bytes());
        digest.update(table.initial_table_root);
        digest.update(table.row_allocator_before.to_le_bytes());
        digest.update(table.row_allocator_high_water.to_le_bytes());
        digest.update(table.initial_logical_row_count.to_le_bytes());
        digest.update(table.final_logical_row_count.to_le_bytes());

        let transitions = range(
            &graph.transitions,
            table.transition_start,
            table.transition_count,
            "table transition",
        )?;
        let image = graph.images.get(table.image_ref as usize).ok_or_else(|| {
            generation_error("table final image reference is outside retained graph")
        })?;
        let image_facts = image.facts();
        if image_facts.role != TypedImageRole::FinalTableImage
            || image_facts.rows != table.transition_count
            || image_facts.columns != table.catalog_column_count
            || image_facts.layout_digest != table.image_layout_digest
        {
            return Err(generation_error(
                "final image facts differ from retained table fields",
            ));
        }
        digest.update(count_u32(transitions.len(), "generation row-input count")?.to_le_bytes());
        let mut previous_row = None;
        for (transition_ordinal, transition) in transitions.iter().enumerate() {
            if previous_row.is_some_and(|previous| transition.stable_row_id <= previous)
                || transition.transition_ref
                    != directory_ref(
                        table.transition_start,
                        transition_ordinal,
                        "table transition",
                    )?
                || transition.table_ref != table.table_ref
                || transition.image_ref != table.image_ref
            {
                return Err(generation_error(
                    "generation row inputs are not stable-row ordered for their table",
                ));
            }
            digest.update(generation_row_input_digest(
                table,
                catalog_table,
                image,
                transition,
            )?);
            previous_row = Some(transition.stable_row_id);
        }
        digest.update(table.image_layout_digest);
        digest.update(table.image_content_digest);

        let indexes = range(
            &graph.indexes,
            table.owned_index_start,
            table.owned_index_count,
            "table index",
        )?;
        digest.update(count_u32(indexes.len(), "generation owned-index count")?.to_le_bytes());
        let mut previous_index = None;
        let mut maintenance_effects = 0_usize;
        for (index_ordinal, index) in indexes.iter().enumerate() {
            if previous_index.is_some_and(|previous| index.stable_index_id <= previous)
                || index.index_ref
                    != directory_ref(table.owned_index_start, index_ordinal, "table index")?
                || index.owner_table_ref != table.table_ref
                || index.owner_stable_table_id != table.stable_table_id
            {
                return Err(generation_error(
                    "generation indexes are not stable-index ordered for their table",
                ));
            }
            let catalog_index = catalog_index(catalog, index.stable_index_id)?;
            let shape = generation_index_shape_digest(graph, table, index, catalog_index)?;
            digest.update(shape);
            let effect_count = count_maintenance_effects(graph, table, index)?;
            digest.update(
                count_u32(effect_count, "generation maintenance-effect count")?.to_le_bytes(),
            );
            for transition in transitions {
                let effects = range(
                    &graph.key_effects,
                    transition.key_effect_start,
                    transition.key_effect_count,
                    "transition key effect",
                )?;
                for effect in effects {
                    if effect.role == PHYSICAL_MAINTENANCE_ROLE
                        && effect.index_ref == index.index_ref
                    {
                        digest.update(generation_index_effect_input_digest(
                            graph, table, image, transition, effect, index, shape,
                        )?);
                        maintenance_effects =
                            maintenance_effects.checked_add(1).ok_or_else(|| {
                                generation_error("generation maintenance-effect count overflows")
                            })?;
                    }
                }
            }
            previous_index = Some(index.stable_index_id);
        }
        let total_role_one = range(
            &graph.key_effects,
            table.key_effect_start,
            table.key_effect_count,
            "table key effect",
        )?
        .iter()
        .filter(|effect| effect.role == PHYSICAL_MAINTENANCE_ROLE)
        .count();
        if maintenance_effects != total_role_one {
            return Err(generation_error(
                "maintenance effect is missing from the table-owned index traversal",
            ));
        }
        previous_table = Some(table.stable_table_id);
    }
    Ok(digest.finalize().into())
}

fn generation_row_input_digest(
    table: &RetainedTable,
    catalog_table: &SemanticsV2CatalogTableWitness<'_>,
    image: &DecodedTypedImage,
    transition: &RetainedTransition,
) -> Result<CanonicalDigest, EngineError> {
    let image_facts = image.facts();
    if transition.image_row_ordinal >= image_facts.rows {
        return Err(generation_error(
            "generation transition image row is outside the final image",
        ));
    }
    let mut digest = begin_digest(b"gpu-db/write001/generation-row-input/v2");
    digest.update(table.stable_table_id.to_le_bytes());
    digest.update(transition.stable_row_id.to_le_bytes());
    digest.update(transition.source_statement_ordinal.to_le_bytes());
    digest.update(transition.source_row_ordinal.to_le_bytes());
    digest.update(table.catalog_column_count.to_le_bytes());
    let mut image_columns = image.columns();
    for (ordinal, catalog_column) in catalog_table.catalog_columns.iter().enumerate() {
        let image_column = image_columns.next().ok_or_else(|| {
            generation_error("final image has fewer columns than the retained catalog table")
        })?;
        if image_column.catalog_column_ordinal != ordinal as u32
            || image_column.catalog_column_ordinal != catalog_column.catalog_column_ordinal
            || image_column.stable_column_id != catalog_column.stable_column_id
            || image_column.attnum != catalog_column.attnum
            || image_column.type_oid != catalog_column.declared_type_oid
            || image_column.type_size != catalog_column.signed_type_size
            || !storage_matches_type(image_column.ty, catalog_column.storage)
        {
            return Err(generation_error(
                "final image column does not exactly match the pinned catalog column",
            ));
        }
        digest.update(catalog_column.catalog_column_ordinal.to_le_bytes());
        digest.update(catalog_column.stable_column_id.to_le_bytes());
        digest.update(catalog_column.attnum.to_le_bytes());
        digest.update(catalog_column.storage);
        digest.update(catalog_column.declared_type_oid.to_le_bytes());
        digest.update(catalog_column.signed_type_size.to_le_bytes());
        append_image_cell(
            &mut digest,
            image_column.validity,
            image_column.values,
            image_column.ty,
            image_facts.rows,
            transition.image_row_ordinal,
        )?;
    }
    if image_columns.next().is_some() {
        return Err(generation_error(
            "final image has more columns than the retained catalog table",
        ));
    }
    Ok(digest.finalize().into())
}

fn generation_index_shape_digest(
    graph: &ReservedSemanticsV2Graph,
    table: &RetainedTable,
    index: &RetainedIndexDescriptor,
    catalog_index: &SemanticsV2CatalogIndexWitness<'_>,
) -> Result<CanonicalDigest, EngineError> {
    if index.stable_index_id != catalog_index.stable_index_id
        || index.display_oid != catalog_index.display_oid
        || index.owner_stable_table_id != catalog_index.owner_stable_table_id
        || index.owner_display_oid != catalog_index.owner_display_oid
        || index.catalog_epoch != catalog_index.catalog_epoch
        || index.flags != catalog_index.index_flags
        || index.null_equality_policy != catalog_index.null_equality_policy
        || index.base_index_generation != catalog_index.base_generation
        || index.base_index_root != catalog_index.base_root
        || index.key_count as usize != catalog_index.key_columns.len()
    {
        return Err(generation_error(
            "retained index does not match its pinned catalog generation shape",
        ));
    }
    let keys = range(
        &graph.index_key_columns,
        index.key_start,
        index.key_count,
        "index key column",
    )?;
    let mut digest = begin_digest(b"gpu-db/write001/generation-index-shape/v2");
    digest.update(table.stable_table_id.to_le_bytes());
    digest.update(index.stable_index_id.to_le_bytes());
    digest.update(index.flags.to_le_bytes());
    digest.update([index.null_equality_policy]);
    digest.update(index.base_index_generation.to_le_bytes());
    digest.update(index.base_index_root);
    digest.update(count_u32(keys.len(), "generation index key count")?.to_le_bytes());
    for (ordinal, (key, catalog_key)) in keys
        .iter()
        .zip(catalog_index.key_columns.iter())
        .enumerate()
    {
        if key.index_ref != index.index_ref
            || key.key_column_ref != directory_ref(index.key_start, ordinal, "index key column")?
            || key.key_ordinal != ordinal as u32
            || key.key_ordinal != catalog_key.key_ordinal
            || key.owner_catalog_column_ordinal != catalog_key.owner_catalog_column_ordinal
            || key.stable_column_id != catalog_key.stable_column_id
            || key.attnum != catalog_key.attnum
            || key.storage != catalog_key.storage
            || key.declared_type_oid != catalog_key.declared_type_oid
            || key.signed_type_size != catalog_key.signed_type_size
            || key.column_name_digest != catalog_key.column_name_digest
        {
            return Err(generation_error(
                "retained index key does not match its pinned catalog key",
            ));
        }
        append_index_key_shape(&mut digest, key);
    }
    Ok(digest.finalize().into())
}

fn append_index_key_shape(digest: &mut Sha256, key: &RetainedIndexKeyColumn) {
    digest.update(key.key_ordinal.to_le_bytes());
    digest.update(key.owner_catalog_column_ordinal.to_le_bytes());
    digest.update(key.stable_column_id.to_le_bytes());
    digest.update(key.attnum.to_le_bytes());
    digest.update(key.storage);
    digest.update(key.declared_type_oid.to_le_bytes());
    digest.update(key.signed_type_size.to_le_bytes());
    digest.update(key.column_name_digest);
}

fn count_maintenance_effects(
    graph: &ReservedSemanticsV2Graph,
    table: &RetainedTable,
    index: &RetainedIndexDescriptor,
) -> Result<usize, EngineError> {
    let transitions = range(
        &graph.transitions,
        table.transition_start,
        table.transition_count,
        "table transition",
    )?;
    let mut count = 0_usize;
    for transition in transitions {
        for effect in range(
            &graph.key_effects,
            transition.key_effect_start,
            transition.key_effect_count,
            "transition key effect",
        )? {
            if effect.role == PHYSICAL_MAINTENANCE_ROLE && effect.index_ref == index.index_ref {
                count = count
                    .checked_add(1)
                    .ok_or_else(|| generation_error("maintenance-effect count overflows"))?;
            }
        }
    }
    Ok(count)
}

fn generation_index_effect_input_digest(
    graph: &ReservedSemanticsV2Graph,
    table: &RetainedTable,
    image: &DecodedTypedImage,
    transition: &RetainedTransition,
    effect: &RetainedKeyEffect,
    index: &RetainedIndexDescriptor,
    shape: CanonicalDigest,
) -> Result<CanonicalDigest, EngineError> {
    if effect.role != PHYSICAL_MAINTENANCE_ROLE
        || effect.transition_ref != transition.transition_ref
        || effect.index_ref != index.index_ref
        || effect.key_arity != index.key_count
        || effect.new_component_count != index.key_count
    {
        return Err(generation_error(
            "generation maintenance effect does not bind its transition/index shape",
        ));
    }
    let components = range(
        &graph.key_components,
        effect.new_component_start,
        effect.new_component_count,
        "maintenance key component",
    )?;
    let keys = range(
        &graph.index_key_columns,
        index.key_start,
        index.key_count,
        "index key column",
    )?;
    let mut digest = begin_digest(b"gpu-db/write001/generation-index-effect-input/v2");
    digest.update(table.stable_table_id.to_le_bytes());
    digest.update(transition.stable_row_id.to_le_bytes());
    digest.update(effect.source_catalog_ordinal.to_le_bytes());
    digest.update(shape);
    digest.update(effect.key_arity.to_le_bytes());
    for (ordinal, (component, key)) in components.iter().zip(keys.iter()).enumerate() {
        if component.effect_ref != effect.effect_ref
            || component.component_ref
                != directory_ref(
                    effect.new_component_start,
                    ordinal,
                    "maintenance key component",
                )?
            || component.component_ordinal != ordinal as u32
            || component.key_column_ref != key.key_column_ref
            || component.source_catalog_ordinal != key.owner_catalog_column_ordinal
            || component.storage != key.storage
            || component.declared_type_oid != key.declared_type_oid
            || component.signed_type_size != key.signed_type_size
        {
            return Err(generation_error(
                "maintenance key component does not match its index key shape",
            ));
        }
        let typed_value = typed_value_digest_from_image(
            image,
            transition.image_row_ordinal,
            component.source_catalog_ordinal,
            component.storage,
            component.declared_type_oid,
            component.signed_type_size,
        )?;
        if typed_value != component.typed_value_digest {
            return Err(generation_error(
                "maintenance key component differs from its decoded final-image cell",
            ));
        }
        digest.update(typed_value);
    }
    Ok(digest.finalize().into())
}

fn typed_value_digest_from_image(
    image: &DecodedTypedImage,
    row: u32,
    catalog_column_ordinal: u32,
    storage: [u8; 4],
    declared_type_oid: u32,
    signed_type_size: i16,
) -> Result<CanonicalDigest, EngineError> {
    let image_facts = image.facts();
    let mut found = None;
    for column in image.columns() {
        if column.catalog_column_ordinal == catalog_column_ordinal {
            if found.replace(()).is_some()
                || column.type_oid != declared_type_oid
                || column.type_size != signed_type_size
                || !storage_matches_type(column.ty, storage)
            {
                return Err(generation_error(
                    "decoded image column is ambiguous or disagrees with typed key metadata",
                ));
            }
            let mut digest = begin_digest(b"gpu-db/write001/s7-typed-key-value/v2");
            digest.update(storage);
            digest.update(declared_type_oid.to_le_bytes());
            digest.update(signed_type_size.to_le_bytes());
            append_image_cell(
                &mut digest,
                column.validity,
                column.values,
                column.ty,
                image_facts.rows,
                row,
            )?;
            return Ok(digest.finalize().into());
        }
    }
    Err(generation_error(
        "typed key refers to a missing decoded final-image column",
    ))
}

fn append_image_cell(
    digest: &mut Sha256,
    validity: &TypedInsertColumnValidity,
    values: &TypedInsertColumnValues,
    ty: SqlType,
    rows: u32,
    row: u32,
) -> Result<(), EngineError> {
    let row = usize::try_from(row)
        .map_err(|_| generation_error("decoded image row is not addressable"))?;
    let rows = usize::try_from(rows)
        .map_err(|_| generation_error("decoded image row count is not addressable"))?;
    if row >= rows {
        return Err(generation_error("decoded image cell row is out of range"));
    }
    let valid = match validity {
        TypedInsertColumnValidity::AllValid => true,
        TypedInsertColumnValidity::Bitmap(words) => words
            .get(row / 32)
            .is_some_and(|word| (word & (1_u32 << (row % 32))) != 0),
    };
    digest.update([u8::from(!valid)]);
    if !valid {
        digest.update(0_u32.to_le_bytes());
        return Ok(());
    }
    match (values, ty) {
        (TypedInsertColumnValues::I32(values), SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            let value = *values
                .get(row)
                .ok_or_else(|| generation_error("decoded i32 image vector is too short"))?;
            digest.update(4_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => {
            let value = *values
                .get(row)
                .ok_or_else(|| generation_error("decoded i64 image vector is too short"))?;
            digest.update(8_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        (TypedInsertColumnValues::I128(values), SqlType::Numeric { .. }) => {
            let value = *values
                .get(row)
                .ok_or_else(|| generation_error("decoded numeric image vector is too short"))?;
            digest.update(16_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => {
            let value = values
                .get(row)
                .ok_or_else(|| generation_error("decoded UUID image vector is too short"))?;
            digest.update(16_u32.to_le_bytes());
            digest.update(value);
        }
        (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => {
            let word = words
                .get(row / 32)
                .ok_or_else(|| generation_error("decoded bool image vector is too short"))?;
            digest.update(1_u32.to_le_bytes());
            digest.update([u8::from((word & (1_u32 << (row % 32))) != 0)]);
        }
        (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
            let start = *offsets
                .get(row)
                .ok_or_else(|| generation_error("decoded text image offsets are too short"))?;
            let end = *offsets
                .get(row + 1)
                .ok_or_else(|| generation_error("decoded text image offsets are too short"))?;
            let start = usize::try_from(start)
                .map_err(|_| generation_error("decoded text image offset is not addressable"))?;
            let end = usize::try_from(end)
                .map_err(|_| generation_error("decoded text image offset is not addressable"))?;
            let value = bytes
                .get(start..end)
                .ok_or_else(|| generation_error("decoded text image value range is invalid"))?;
            digest.update(count_u32(value.len(), "decoded text image value length")?.to_le_bytes());
            digest.update(value);
        }
        _ => {
            return Err(generation_error(
                "decoded image vector arm does not match its SQL storage type",
            ))
        }
    }
    Ok(())
}

fn storage_matches_type(ty: SqlType, storage: [u8; 4]) -> bool {
    match ty {
        SqlType::Int2 => storage == [1, 0, 0, 0],
        SqlType::Int4 => storage == [2, 0, 0, 0],
        SqlType::Int8 => storage == [3, 0, 0, 0],
        SqlType::Numeric { precision, scale } => storage == [4, precision, scale, 0],
        SqlType::Bool => storage == [5, 0, 0, 0],
        SqlType::Text => storage == [6, 0, 0, 0],
        SqlType::Date => storage == [7, 0, 0, 0],
        SqlType::Timestamp => storage == [8, 0, 0, 0],
        SqlType::Uuid => storage == [9, 0, 0, 0],
    }
}

fn catalog_index<'a>(
    catalog: &'a super::SemanticsV2CatalogWitness<'a>,
    stable_index_id: u64,
) -> Result<&'a SemanticsV2CatalogIndexWitness<'a>, EngineError> {
    let mut found = None;
    for index in catalog.indexes {
        if index.stable_index_id == stable_index_id && found.replace(index).is_some() {
            return Err(generation_error(
                "pinned catalog has duplicate stable index identities",
            ));
        }
    }
    found.ok_or_else(|| generation_error("retained index is absent from the pinned catalog"))
}

fn catalog_target_table<'a>(
    catalog: &'a super::SemanticsV2CatalogWitness<'a>,
    stable_table_id: u64,
    display_oid: u32,
) -> Result<&'a SemanticsV2CatalogTableWitness<'a>, EngineError> {
    let mut found = None;
    for table in catalog.tables {
        if table.stable_table_id == stable_table_id
            && table.display_oid == display_oid
            && found.replace(table).is_some()
        {
            return Err(generation_error(
                "pinned catalog has duplicate stable target-table identities",
            ));
        }
    }
    found.ok_or_else(|| generation_error("retained target table is absent from the pinned catalog"))
}

fn range<'a, T>(
    values: &'a [T],
    start: u32,
    count: u32,
    owner: &str,
) -> Result<&'a [T], EngineError> {
    let start = usize::try_from(start)
        .map_err(|_| generation_error(&format!("{owner} start is not addressable")))?;
    let count = usize::try_from(count)
        .map_err(|_| generation_error(&format!("{owner} count is not addressable")))?;
    let end = start
        .checked_add(count)
        .ok_or_else(|| generation_error(&format!("{owner} range overflows")))?;
    values
        .get(start..end)
        .ok_or_else(|| generation_error(&format!("{owner} range is outside retained graph")))
}

fn count_u32(value: usize, owner: &str) -> Result<u32, EngineError> {
    u32::try_from(value).map_err(|_| generation_error(&format!("{owner} exceeds u32")))
}

fn directory_ref(start: u32, ordinal: usize, owner: &str) -> Result<u32, EngineError> {
    let ordinal = count_u32(ordinal, owner)?;
    start
        .checked_add(ordinal)
        .ok_or_else(|| generation_error(&format!("{owner} directory reference overflows")))
}

fn begin_digest(domain: &[u8]) -> Sha256 {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest
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
