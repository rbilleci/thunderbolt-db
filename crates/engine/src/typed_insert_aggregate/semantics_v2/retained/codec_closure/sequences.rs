//! Exact published/private sequence S2/S4/S5/S7 closure without catalog witnesses.

use super::dependencies::qualified_name_digest;
use super::error;
use crate::typed_insert_aggregate::semantics_v2::retained::{
    graph::ReservedSemanticsV2Graph, SemanticsV2BoundIdentity,
};
use crate::typed_insert_batch::DecodedSequenceEffectKindFacts;
use crate::EngineError;

const SURVIVES: u8 = 1;
const PUBLISHED_SEQUENCE: u8 = 8;
const PUBLISHED_SEQUENCE_ROLE: u16 = 8;

pub(super) fn validate(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    validate_s5_sources(identity, graph)?;
    validate_terminal_restarts(identity, graph)?;
    validate_s2_sequence_inventory(graph)?;
    validate_token_and_use_bijections(graph)
}

fn validate_terminal_restarts(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    let mut seen = std::collections::BTreeSet::new();
    for (effect_index, effect) in graph.sequence_effects.iter().enumerate() {
        let Some(tail) = effect.terminal_restart else {
            continue;
        };
        crate::typed_insert_aggregate::semantics_v2::sequence_terminal::validate(
            identity.request_digest,
            tail,
        )?;
        let record = graph
            .records
            .get(effect.statement_ordinal as usize)
            .ok_or_else(|| error("terminal sequence restart has no S2 record"))?;
        let source = one_sequence_effect(record, effect.effect_ordinal)?;
        let binding = record
            .sequence_bindings()
            .find(|binding| binding.effect_ordinal == effect.effect_ordinal)
            .ok_or_else(|| error("terminal sequence restart has no S2 binding"))?;
        if tail.sequence_oid != source.request.sequence_oid
            || tail.descriptor_digest != binding.descriptor_digest
            || tail.owner_operation_ordinal <= effect.statement_ordinal
            || !seen.insert(tail.sequence_oid)
            || graph.sequence_effects[effect_index + 1..]
                .iter()
                .any(|later| sequence_oid(graph, later).ok() == Some(tail.sequence_oid))
        {
            return Err(error(
                "terminal sequence restart does not close the final S2-owned effect",
            ));
        }
    }
    Ok(())
}

fn sequence_oid(
    graph: &ReservedSemanticsV2Graph,
    effect: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedSequenceEffect,
) -> Result<u32, EngineError> {
    let record = graph
        .records
        .get(effect.statement_ordinal as usize)
        .ok_or_else(|| error("S5 sequence effect has no S2 record"))?;
    Ok(one_sequence_effect(record, effect.effect_ordinal)?
        .request
        .sequence_oid)
}

fn validate_s5_sources(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    let mut prior_transition = 0_u64;
    for effect in &graph.sequence_effects {
        let record = graph
            .records
            .get(effect.statement_ordinal as usize)
            .ok_or_else(|| error("S5 sequence effect statement has no S2 record"))?;
        let resolution = graph
            .resolutions
            .get(effect.statement_ordinal as usize)
            .ok_or_else(|| error("S5 sequence effect statement has no S7 resolution"))?;
        let source = one_sequence_effect(record, effect.effect_ordinal)?;
        let binding = record
            .sequence_bindings()
            .find(|binding| binding.effect_ordinal == effect.effect_ordinal)
            .ok_or_else(|| error("S5 sequence effect has no S2 sequence binding"))?;
        let parent = record
            .sequence_parent()
            .ok_or_else(|| error("published S2 sequence effect has no sequence parent"))?;
        let disposition = graph
            .dispositions
            .get(effect.disposition_ref as usize)
            .ok_or_else(|| error("S5 sequence effect disposition is absent"))?;
        let expected_overwritten = effect.flags & 2 != 0;
        if !matches!(effect.flags, 1 | 3)
            || disposition.statement_ordinal != effect.statement_ordinal
            || disposition.source_row_ordinal != source.request.row_ordinal
            || parent.txn_id != identity.stable_transaction_id
            || parent.autocommit != identity.autocommit
            || parent.request_digest
                != graph.statements[effect.statement_ordinal as usize].request_digest
            || parent.statement_ordinal.as_u32() != effect.statement_ordinal
            || binding.request != source.request
        {
            return Err(error(
                "S5 sequence effect does not close its S2/S4 identity",
            ));
        }
        let target = record.target_identity();
        let source_column = record
            .catalog_columns()
            .nth(source.request.catalog_column_ordinal as usize)
            .ok_or_else(|| error("S5 sequence source catalog column is absent"))?;
        if target.oid != source.request.target_table_oid
            || source_column.column_id != source.request.column_id
            || source.request.statement_ordinal.as_u32() != effect.statement_ordinal
            || resolution.table_ref != disposition.table_ref
        {
            return Err(error(
                "S5 sequence request does not close its target-column source",
            ));
        }
        let returned_value = match (source.kind, effect.reference.as_ref()) {
            (
                DecodedSequenceEffectKindFacts::Published {
                    transition_txn_id,
                    input_digest,
                    returned_value,
                },
                Some(reference),
            ) => {
                if reference.parent_txn_id != identity.stable_transaction_id
                    || reference.transition_txn_id != transition_txn_id
                    || reference.transition_txn_id <= prior_transition
                    || reference.input_digest != input_digest
                    || reference.returned_value != returned_value
                    || !reference.default_expression
                    || reference.final_value_overwritten != expected_overwritten
                    || reference.statement_ordinal != effect.statement_ordinal
                    || reference.expression_ordinal != source.request.absolute_expression_ordinal
                    || reference.sequence_oid != source.request.sequence_oid
                    || reference.table_oid != source.request.target_table_oid
                    || reference.column_id != source.request.column_id
                    || reference.row_id != disposition.stable_row_id
                    || effect.reference_body_digest == [0; 32]
                {
                    return Err(error(
                        "S5 published sequence effect does not close its durable receipt",
                    ));
                }
                prior_transition = reference.transition_txn_id;
                returned_value
            }
            (DecodedSequenceEffectKindFacts::Private { .. }, None) => {
                if effect.reference_body_digest != [0; 32]
                    || (effect.terminal_restart.is_none() && effect.body_digest != [0; 32])
                    || (effect.terminal_restart.is_some() && effect.body_digest == [0; 32])
                {
                    return Err(error("S5 private sequence effect retained an invalid body"));
                }
                source.resolved_value
            }
            _ => {
                return Err(error(
                    "S5 sequence effect kind does not match its S2 ownership",
                ));
            }
        };
        let (valid, source_value) = record.column_value_at(
            source.request.catalog_column_ordinal,
            source.request.row_ordinal,
        )?;
        if !valid || !sequence_source_value_matches_returned(source_value, returned_value) {
            return Err(error("sequence return value differs from its S2 source"));
        }
        if disposition.disposition == SURVIVES {
            let transition = graph
                .transitions
                .get(disposition.transition_ref as usize)
                .ok_or_else(|| error("surviving sequence source has no final transition"))?;
            let image = graph
                .images
                .get(transition.image_ref as usize)
                .ok_or_else(|| error("surviving sequence source has no final image"))?;
            let column = image
                .columns()
                .nth(source.request.catalog_column_ordinal as usize)
                .ok_or_else(|| error("surviving sequence source column left the final image"))?;
            let expected = i32::try_from(returned_value)
                .map_err(|_| error("sequence return value exceeds its final int4 image"))?
                .to_le_bytes();
            let final_value_matches = column
                .with_logical_cell_at(transition.image_row_ordinal as usize, |is_null, bytes| {
                    !is_null && bytes == expected
                })?;
            if final_value_matches == expected_overwritten {
                return Err(error(
                    "sequence overwrite flag differs from the retained final image",
                ));
            }
        } else if !expected_overwritten {
            return Err(error(
                "non-surviving sequence source is not marked overwritten",
            ));
        }
    }
    Ok(())
}

fn validate_s2_sequence_inventory(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    for record in &graph.records {
        let statement = record.facts().statement_ordinal.as_u32();
        for source in record.sequence_effects() {
            let count = graph
                .sequence_effects
                .iter()
                .filter(|effect| {
                    effect.statement_ordinal == statement
                        && effect.effect_ordinal == source.request.effect_ordinal
                })
                .count();
            if count != 1 {
                return Err(error(
                    "S2 sequence source does not biject exactly one S5 effect",
                ));
            }
        }
    }
    Ok(())
}

fn validate_token_and_use_bijections(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    for effect in &graph.sequence_effects {
        let Some(reference) = effect.reference.as_ref() else {
            continue;
        };
        let name_digest = effective_name_digest(graph, effect)?;
        let token_count = graph
            .dependencies
            .iter()
            .filter(|token| {
                token.kind == PUBLISHED_SEQUENCE
                    && token.display_oid == reference.sequence_oid
                    && token.base_generation == reference.transition_txn_id
                    && token.base_root == effect.reference_body_digest
                    && token.name_digest == name_digest
            })
            .count();
        if token_count != 1 {
            return Err(error(
                "published S5 sequence effect does not select one token",
            ));
        }
    }
    for token in graph
        .dependencies
        .iter()
        .filter(|token| token.kind == PUBLISHED_SEQUENCE)
    {
        let mut effect_count = 0_usize;
        let mut use_count = 0_usize;
        for effect in &graph.sequence_effects {
            let Some(reference) = effect.reference.as_ref() else {
                continue;
            };
            if reference.sequence_oid != token.display_oid
                || reference.transition_txn_id != token.base_generation
                || effect.reference_body_digest != token.base_root
                || effective_name_digest(graph, effect)? != token.name_digest
            {
                continue;
            }
            effect_count += 1;
            let disposition = graph
                .dispositions
                .get(effect.disposition_ref as usize)
                .ok_or_else(|| error("published S5 sequence effect disposition is absent"))?;
            use_count += graph
                .dependency_uses
                .iter()
                .filter(|usage| {
                    usage.dependency_ref == token.dependency_ref
                        && usage.role == PUBLISHED_SEQUENCE_ROLE
                        && usage.statement_ordinal == effect.statement_ordinal
                        && usage.source_ordinal == effect.effect_ordinal
                        && usage.transition_ref == disposition.transition_ref
                })
                .count();
        }
        if effect_count != 1 || use_count != 1 {
            return Err(error(
                "published-sequence token/use does not biject one S5 effect",
            ));
        }
    }
    Ok(())
}

fn effective_name_digest(
    graph: &ReservedSemanticsV2Graph,
    effect: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedSequenceEffect,
) -> Result<[u8; 32], EngineError> {
    let record = graph
        .records
        .get(effect.statement_ordinal as usize)
        .ok_or_else(|| error("published S5 sequence effect statement has no S2 record"))?;
    let mut bindings = record
        .sequence_bindings()
        .filter(|binding| binding.effect_ordinal == effect.effect_ordinal);
    let binding = bindings
        .next()
        .ok_or_else(|| error("published S5 sequence effect has no S2 sequence binding"))?;
    if bindings.next().is_some() {
        return Err(error(
            "published S5 sequence effect has duplicate S2 sequence bindings",
        ));
    }
    Ok(qualified_name_digest(
        record.target_identity().schema,
        binding.effective_name,
    ))
}

fn sequence_source_value_matches_returned(
    value: crate::typed_insert_batch::DecodedTypedValueFacts<'_>,
    returned: i64,
) -> bool {
    match value {
        crate::typed_insert_batch::DecodedTypedValueFacts::I32(value) => {
            i32::try_from(returned).is_ok_and(|returned| value == returned)
        }
        crate::typed_insert_batch::DecodedTypedValueFacts::I64(value) => value == returned,
        _ => false,
    }
}

fn one_sequence_effect(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    ordinal: u32,
) -> Result<crate::typed_insert_batch::DecodedSequenceEffectFacts, EngineError> {
    let mut matches = record
        .sequence_effects()
        .filter(|effect| effect.request.effect_ordinal == ordinal);
    let source = matches
        .next()
        .ok_or_else(|| error("S5 effect has no S2 sequence source"))?;
    if matches.next().is_some() {
        return Err(error("S5 effect has duplicate S2 sequence sources"));
    }
    Ok(source)
}
