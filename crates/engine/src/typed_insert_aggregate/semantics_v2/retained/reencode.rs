//! Test-only logical S1--S7 reconstruction from the fully validated retained owner.
//!
//! The source graph deliberately owns decoded S2 records and decoded final images, never raw
//! aggregate sections.  This leaf is therefore intentionally a one-way golden-evidence encoder:
//! it has no public surface, no raw input, no live caller, and no route around the fully
//! witness-validated typestate.

use super::{
    graph::{
        ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedIndexDescriptor,
        RetainedIndexKeyColumn, RetainedKeyComponent, RetainedKeyEffect, RetainedProjectionBinding,
        RetainedStatementDependencyUse, RetainedStatementOutcome, RetainedStatementResolution,
        RetainedTable, RetainedTableDisposition, RetainedTransition,
    },
    FullyWitnessValidatedSemanticsV2,
};
use crate::typed_insert_batch::{
    reencode_decoded_canonical_typed_insert_record_for_test, reencode_decoded_typed_image_for_test,
    DecodedTypedImage, TypedImageRole, TypedInsertColumnValidity, TypedInsertColumnValues,
};
use crate::{encode_sequence_value_reference_into_exact, EngineError, SqlType};
use sha2::{Digest, Sha256};

const S1_BYTES: usize = 144;
const S4_BYTES: usize = 64;
const S6_BYTES: usize = 136;
const S7_HEADER_BYTES: usize = 640;
const S7_FIXED_WIDTHS: [usize; 12] = [384, 32, 320, 224, 32, 384, 112, 192, 192, 128, 128, 160];
const S7_MAGIC: &[u8; 16] = b"GPUDBS7OVERLAY2\0";

pub(super) fn reencode_s1_s7_for_test<C>(
    owner: &FullyWitnessValidatedSemanticsV2<C>,
) -> Result<[Vec<u8>; 7], EngineError> {
    let graph = &owner.graph.graph;
    let s1 = encode_s1(graph)?;
    let s2 = encode_s2(graph)?;
    let s4 = encode_s4(graph)?;
    let s5 = encode_s5(graph)?;
    let s6 = encode_s6(graph)?;
    let s7 = encode_s7(graph)?;
    Ok([s1, s2, Vec::new(), s4, s5, s6, s7])
}

fn encode_s1(graph: &ReservedSemanticsV2Graph) -> Result<Vec<u8>, EngineError> {
    let mut out = Vec::with_capacity(
        graph
            .statements
            .len()
            .checked_mul(S1_BYTES)
            .ok_or_else(|| reencode_error("S1 size overflows"))?,
    );
    for statement in &graph.statements {
        let mut raw = [0_u8; S1_BYTES];
        put_u32(&mut raw, 0, statement.statement_ordinal);
        put_u32(&mut raw, 4, statement.family_ordinal);
        // Every retained S1 record is a typed-INSERT statement.  The pass-zero grammar fixes
        // this one-byte family tag to `1`; retaining it independently would duplicate a
        // constant rather than preserve an input fact.
        raw[8] = 1;
        put_u32(&mut raw, 12, statement.input_row_count);
        put_digest(&mut raw, 16, statement.request_digest);
        put_digest(&mut raw, 48, statement.typed_statement_digest);
        put_digest(&mut raw, 80, statement.overlay_before);
        put_digest(&mut raw, 112, statement.overlay_after);
        out.extend_from_slice(&raw);
    }
    Ok(out)
}

fn encode_s2(graph: &ReservedSemanticsV2Graph) -> Result<Vec<u8>, EngineError> {
    if graph.records.len() != graph.statements.len() {
        return Err(reencode_error("retained S1/S2 owner count differs"));
    }
    let mut out = Vec::new();
    for (statement, record) in graph.statements.iter().zip(&graph.records) {
        let bytes = reencode_decoded_canonical_typed_insert_record_for_test(record);
        let record_bytes = u32::try_from(bytes.len())
            .map_err(|_| reencode_error("reconstructed S2 record exceeds u32"))?;
        if statement.record_bytes != record_bytes
            || statement.record_digest != s2_digest(&bytes)
            || record.facts().typed_statement_digest != statement.typed_statement_digest
        {
            return Err(reencode_error(
                "reconstructed S2 record differs from retained S1 source facts",
            ));
        }
        out.extend_from_slice(&record_bytes.to_le_bytes());
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

fn encode_s4(graph: &ReservedSemanticsV2Graph) -> Result<Vec<u8>, EngineError> {
    let mut out = Vec::with_capacity(
        graph
            .dispositions
            .len()
            .checked_mul(S4_BYTES)
            .ok_or_else(|| reencode_error("S4 size overflows"))?,
    );
    for entry in &graph.dispositions {
        let mut raw = [0_u8; S4_BYTES];
        put_u32(&mut raw, 0, entry.statement_ordinal);
        put_u32(&mut raw, 4, entry.source_row_ordinal);
        put_u64(&mut raw, 8, entry.stable_row_id);
        raw[16] = entry.disposition;
        put_u32(&mut raw, 20, entry.table_ref);
        put_u32(&mut raw, 24, entry.transition_ref);
        if entry.final_writer_statement_digest == [0; 32] {
            put_digest(&mut raw, 32, entry.typed_statement_digest);
        } else {
            raw[17] = 1;
            put_u32(&mut raw, 28, entry.final_writer_statement_ordinal);
            put_digest(&mut raw, 32, entry.final_writer_statement_digest);
        }
        out.extend_from_slice(&raw);
    }
    Ok(out)
}

fn encode_s5(graph: &ReservedSemanticsV2Graph) -> Result<Vec<u8>, EngineError> {
    let mut out = Vec::new();
    for effect in &graph.sequence_effects {
        let mut prefix = [0_u8; 52];
        put_u32(&mut prefix, 0, effect.statement_ordinal);
        put_u32(&mut prefix, 4, effect.effect_ordinal);
        prefix[9] = effect.flags;
        put_u32(&mut prefix, 12, effect.disposition_ref);
        let mut encoded_body = Vec::new();
        match effect.reference.as_ref() {
            Some(reference) => {
                let mut body = [0_u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
                encode_sequence_value_reference_into_exact(reference, &mut body)?;
                encoded_body.extend_from_slice(&body);
                prefix[8] = 1;
            }
            None => {
                prefix[8] = 2;
            }
        }
        if let Some(tail) = effect.terminal_restart {
            encoded_body.extend_from_slice(
                &crate::typed_insert_aggregate::semantics_v2::sequence_terminal::encode_retained(
                    tail,
                ),
            );
        }
        let expected_digest = if encoded_body.is_empty() {
            [0; 32]
        } else {
            gpu_db_wal::canonical_request_digest(&encoded_body)
        };
        if effect.body_digest != expected_digest {
            return Err(reencode_error(
                "reconstructed S5 sequence body differs from retained digest",
            ));
        }
        put_u32(
            &mut prefix,
            16,
            u32::try_from(encoded_body.len())
                .map_err(|_| reencode_error("S5 sequence body exceeds u32"))?,
        );
        put_digest(&mut prefix, 20, effect.body_digest);
        out.extend_from_slice(&prefix);
        out.extend_from_slice(&encoded_body);
    }
    Ok(out)
}

fn encode_s6(graph: &ReservedSemanticsV2Graph) -> Result<Vec<u8>, EngineError> {
    let mut out = Vec::with_capacity(
        graph
            .outcomes
            .len()
            .checked_mul(S6_BYTES)
            .ok_or_else(|| reencode_error("S6 size overflows"))?,
    );
    for outcome in &graph.outcomes {
        out.extend_from_slice(&s6_bytes(outcome)?);
    }
    Ok(out)
}

fn s6_bytes(outcome: &RetainedStatementOutcome) -> Result<[u8; S6_BYTES], EngineError> {
    let mut raw = [0_u8; S6_BYTES];
    put_u32(&mut raw, 0, outcome.statement_ordinal);
    put_u32(&mut raw, 4, outcome.family_ordinal);
    put_u16(&mut raw, 8, outcome.semantic_class);
    put_u16(&mut raw, 10, outcome.flags);
    put_digest(&mut raw, 12, outcome.typed_statement_digest);
    let mut canonical = [0_u8; 92];
    gpu_db_wal::encode_canonical_outcome_into_exact(&outcome.outcome, &mut canonical)?;
    raw[44..136].copy_from_slice(&canonical);
    if outcome.outcome_digest != domain_digest(b"gpu-db/write001/s7-s6-entry/v2", &[&raw]) {
        return Err(reencode_error(
            "reconstructed S6 entry differs from retained outcome digest",
        ));
    }
    Ok(raw)
}

fn encode_s7(graph: &ReservedSemanticsV2Graph) -> Result<Vec<u8>, EngineError> {
    if reconstructed_root_descriptor(graph)? != graph.header.root_descriptor {
        return Err(reencode_error(
            "reconstructed root descriptor differs from retained header identity",
        ));
    }
    let images = encode_images(graph)?;
    let values = encode_key_values(graph)?;
    let counts = s7_counts(graph)?;
    let directory_bytes: [u64; 14] = std::array::from_fn(|index| {
        if index < 12 {
            u64::try_from(S7_FIXED_WIDTHS[index]).expect("fixed width")
                * u64::from(counts[[0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11][index]])
        } else if index == 12 {
            u64::try_from(values.len()).expect("host Vec length fits u64")
        } else {
            images.iter().map(|image| image.len() as u64).sum()
        }
    });
    let mut offsets = [0_u64; 14];
    let mut cursor = S7_HEADER_BYTES as u64;
    for index in 0..14 {
        offsets[index] = cursor;
        cursor = cursor
            .checked_add(directory_bytes[index])
            .ok_or_else(|| reencode_error("reconstructed S7 size overflows"))?;
    }
    if cursor != graph.header.total_bytes {
        return Err(reencode_error(
            "reconstructed S7 size differs from retained header identity",
        ));
    }
    let capacity = usize::try_from(cursor)
        .map_err(|_| reencode_error("reconstructed S7 size exceeds host addressability"))?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&s7_header(graph, counts, offsets, directory_bytes)?);
    append_tables(&mut out, &graph.tables);
    append_table_dispositions(&mut out, &graph.table_dispositions);
    append_resolutions(&mut out, &graph.resolutions);
    append_dependencies(&mut out, &graph.dependencies);
    append_dependency_uses(&mut out, &graph.dependency_uses);
    append_indexes(&mut out, &graph.indexes);
    append_index_keys(&mut out, &graph.index_key_columns);
    append_transitions(&mut out, &graph.transitions);
    append_key_effects(&mut out, &graph.key_effects);
    append_key_components(&mut out, &graph.key_components);
    append_projections(&mut out, &graph.projections);
    append_image_descriptors(&mut out, graph, &images)?;
    out.extend_from_slice(&values);
    for image in images {
        out.extend_from_slice(&image);
    }
    if out.len() != capacity {
        return Err(reencode_error(
            "reconstructed S7 directory coverage differs from header",
        ));
    }
    if reconstructed_payload_digest(&out)? != graph.header.payload_digest {
        return Err(reencode_error(
            "reconstructed S7 payload differs from retained header digest",
        ));
    }
    Ok(out)
}

fn s7_counts(graph: &ReservedSemanticsV2Graph) -> Result<[u32; 12], EngineError> {
    [
        graph.tables.len(),
        graph.resolutions.len(),
        graph.table_dispositions.len(),
        graph.dependencies.len(),
        graph.dependency_uses.len(),
        graph.indexes.len(),
        graph.index_key_columns.len(),
        graph.transitions.len(),
        graph.key_effects.len(),
        graph.key_components.len(),
        graph.projections.len(),
        graph.images.len(),
    ]
    .map(|count| u32::try_from(count).map_err(|_| reencode_error("S7 directory count exceeds u32")))
    .into_iter()
    .collect::<Result<Vec<_>, _>>()
    .and_then(|counts| {
        counts
            .try_into()
            .map_err(|_| reencode_error("S7 directory count shape changed"))
    })
}

fn s7_header(
    graph: &ReservedSemanticsV2Graph,
    counts: [u32; 12],
    offsets: [u64; 14],
    directory_bytes: [u64; 14],
) -> Result<[u8; S7_HEADER_BYTES], EngineError> {
    let mut raw = [0_u8; S7_HEADER_BYTES];
    raw[..16].copy_from_slice(S7_MAGIC);
    put_u16(&mut raw, 16, 1);
    put_u16(&mut raw, 18, 2);
    put_u32(&mut raw, 20, S7_HEADER_BYTES as u32);
    put_u16(&mut raw, 28, 14);
    put_u16(&mut raw, 30, graph.header.root_descriptor_version);
    put_u64(&mut raw, 32, graph.header.total_bytes);
    for (index, count) in counts.iter().enumerate() {
        put_u32(&mut raw, 40 + index * 4, *count);
    }
    // The variable arenas are represented twice in S7: by their directory spans and by the
    // fixed header scalars that the pass-zero grammar binds to those spans.
    put_u64(&mut raw, 88, directory_bytes[12]);
    put_u64(&mut raw, 96, directory_bytes[13]);
    for index in 0..14 {
        put_u64(&mut raw, 104 + index * 16, offsets[index]);
        put_u64(&mut raw, 112 + index * 16, directory_bytes[index]);
    }
    put_u64(&mut raw, 328, graph.header.catalog_before_epoch);
    put_u64(&mut raw, 336, graph.header.catalog_after_epoch);
    put_digest(&mut raw, 344, graph.header.catalog_before_digest);
    put_digest(&mut raw, 376, graph.header.catalog_after_digest);
    put_digest(&mut raw, 408, graph.header.initial_database_root);
    put_digest(&mut raw, 440, graph.header.final_database_root);
    put_digest(&mut raw, 472, graph.header.initial_overlay_root);
    put_digest(&mut raw, 504, graph.header.final_overlay_root);
    put_digest(&mut raw, 536, graph.header.root_descriptor);
    put_digest(&mut raw, 568, graph.header.payload_digest);
    Ok(raw)
}

fn append_tables(out: &mut Vec<u8>, values: &[RetainedTable]) {
    for value in values {
        let mut raw = [0_u8; 384];
        put_u32(&mut raw, 0, value.table_ref);
        put_u32(
            &mut raw,
            4,
            u32::from(value.resets_existing_rows) | (u32::from(value.initial_table_absent) << 1),
        );
        put_u64(&mut raw, 8, value.stable_table_id);
        put_u32(&mut raw, 16, value.display_oid);
        put_u32(&mut raw, 20, value.target_dependency_ref);
        put_u64(&mut raw, 24, value.catalog_epoch);
        put_u64(&mut raw, 32, value.data_generation_before);
        put_u64(&mut raw, 40, value.data_generation_after);
        put_u64(&mut raw, 48, value.row_allocator_before);
        put_u64(&mut raw, 56, value.row_allocator_high_water);
        put_u64(&mut raw, 64, value.initial_logical_row_count);
        put_u64(&mut raw, 72, value.final_logical_row_count);
        put_u32(&mut raw, 80, value.disposition_start);
        put_u32(&mut raw, 84, value.disposition_count);
        put_u32(&mut raw, 88, value.transition_start);
        put_u32(&mut raw, 92, value.transition_count);
        put_u32(&mut raw, 96, value.owned_index_start);
        put_u32(&mut raw, 100, value.owned_index_count);
        put_u32(&mut raw, 104, value.key_effect_start);
        put_u32(&mut raw, 108, value.key_effect_count);
        put_u32(&mut raw, 112, value.image_ref);
        put_u32(&mut raw, 116, value.catalog_column_count);
        put_digest(&mut raw, 128, value.schema_digest);
        put_digest(&mut raw, 160, value.initial_table_root);
        put_digest(&mut raw, 192, value.final_table_root);
        put_digest(&mut raw, 224, value.image_layout_digest);
        put_digest(&mut raw, 256, value.image_content_digest);
        put_digest(&mut raw, 288, value.transition_root);
        put_digest(&mut raw, 320, value.index_effect_root);
        put_digest(&mut raw, 352, value.manifest_digest);
        out.extend_from_slice(&raw);
    }
}

fn append_table_dispositions(out: &mut Vec<u8>, values: &[RetainedTableDisposition]) {
    for value in values {
        let mut raw = [0_u8; 32];
        put_u32(&mut raw, 0, value.table_ref);
        put_u32(&mut raw, 4, value.disposition_ref);
        put_u64(&mut raw, 8, value.stable_row_id);
        put_u32(&mut raw, 16, value.statement_ordinal);
        put_u32(&mut raw, 20, value.source_row_ordinal);
        raw[24] = value.disposition;
        out.extend_from_slice(&raw);
    }
}

fn append_resolutions(out: &mut Vec<u8>, values: &[RetainedStatementResolution]) {
    for value in values {
        let mut raw = [0_u8; 320];
        put_u32(&mut raw, 0, value.statement_ordinal);
        // The wire repeats the canonical statement ordinal for the record/outcome provenance
        // slot.  Strict closure already proves both references resolve to this statement, so
        // derive the repetition from the retained ordinal instead of retaining wire bytes.
        put_u32(&mut raw, 4, value.statement_ordinal);
        put_u32(&mut raw, 8, value.record_ref);
        put_u32(&mut raw, 12, value.outcome_ref);
        put_u32(&mut raw, 16, value.table_ref);
        put_u32(&mut raw, 20, value.flags);
        put_u32(&mut raw, 24, value.s4_start);
        put_u32(&mut raw, 28, value.s4_count);
        put_u32(&mut raw, 32, value.s5_start);
        put_u32(&mut raw, 36, value.s5_count);
        put_u32(&mut raw, 40, value.dependency_use_start);
        put_u32(&mut raw, 44, value.dependency_use_count);
        put_u32(&mut raw, 48, value.projection_start);
        put_u32(&mut raw, 52, value.projection_count);
        put_u32(&mut raw, 56, value.input_row_count);
        put_u32(&mut raw, 60, value.surviving_row_count);
        put_u64(&mut raw, 64, value.affected_row_count);
        put_u64(&mut raw, 72, value.dependency_validation_floor);
        put_u32(&mut raw, 80, value.record_bytes);
        put_u32(&mut raw, 84, value.terminal_dependency_ref);
        put_u32(&mut raw, 88, value.terminal_row_ordinal);
        put_u32(&mut raw, 92, value.terminal_source_ordinal);
        put_digest(&mut raw, 96, value.request_digest);
        put_digest(&mut raw, 128, value.typed_statement_digest);
        put_digest(&mut raw, 160, value.record_digest);
        put_digest(&mut raw, 192, value.returning_digest);
        put_digest(&mut raw, 224, value.overlay_before);
        put_digest(&mut raw, 256, value.overlay_after);
        put_digest(&mut raw, 288, value.outcome_digest);
        out.extend_from_slice(&raw);
    }
}

fn append_dependencies(out: &mut Vec<u8>, values: &[RetainedDependencyToken]) {
    for value in values {
        let mut raw = [0_u8; 224];
        put_u32(&mut raw, 0, value.dependency_ref);
        raw[4] = value.kind;
        raw[5] = value.access;
        put_u16(&mut raw, 6, value.flags);
        put_u64(&mut raw, 8, value.stable_object_id);
        put_u32(&mut raw, 16, value.display_oid);
        put_u32(&mut raw, 20, value.target_table_ref);
        put_u64(&mut raw, 24, value.base_generation);
        put_u64(&mut raw, 32, value.snapshot_floor);
        put_u32(&mut raw, 40, value.key_effect_ref);
        put_u32(&mut raw, 44, value.descriptor_ref);
        put_u64(&mut raw, 48, value.catalog_epoch);
        put_digest(&mut raw, 64, value.schema_digest);
        put_digest(&mut raw, 96, value.base_root);
        put_digest(&mut raw, 128, value.name_digest);
        put_digest(&mut raw, 160, value.identity_digest);
        put_digest(&mut raw, 192, value.token_digest);
        out.extend_from_slice(&raw);
    }
}

fn append_dependency_uses(out: &mut Vec<u8>, values: &[RetainedStatementDependencyUse]) {
    for value in values {
        let mut raw = [0_u8; 32];
        put_u32(&mut raw, 0, value.statement_ordinal);
        put_u32(&mut raw, 4, value.dependency_ref);
        put_u16(&mut raw, 8, value.role);
        put_u32(&mut raw, 12, value.source_ordinal);
        put_u32(&mut raw, 16, value.transition_ref);
        put_u32(&mut raw, 20, value.key_effect_ref);
        out.extend_from_slice(&raw);
    }
}

fn append_indexes(out: &mut Vec<u8>, values: &[RetainedIndexDescriptor]) {
    for value in values {
        let mut raw = [0_u8; 384];
        put_u32(&mut raw, 0, value.index_ref);
        put_u32(&mut raw, 4, value.flags);
        put_u32(&mut raw, 8, value.owner_table_ref);
        put_u32(&mut raw, 12, value.raw_catalog_ordinal);
        put_u64(&mut raw, 16, value.stable_index_id);
        put_u32(&mut raw, 24, value.display_oid);
        put_u32(&mut raw, 28, value.constraint_display_oid);
        put_u64(&mut raw, 32, value.stable_constraint_id);
        put_u32(&mut raw, 40, value.key_start);
        put_u32(&mut raw, 44, value.key_count);
        raw[48] = value.null_equality_policy;
        put_u64(&mut raw, 64, value.owner_stable_table_id);
        put_u64(&mut raw, 72, value.catalog_epoch);
        put_digest(&mut raw, 80, value.owner_schema_digest);
        put_digest(&mut raw, 112, value.owner_name_digest);
        put_digest(&mut raw, 144, value.index_name_digest);
        put_digest(&mut raw, 176, value.constraint_name_digest);
        put_digest(&mut raw, 208, value.owner_table_base_root);
        put_digest(&mut raw, 240, value.base_index_root);
        put_digest(&mut raw, 272, value.final_index_root);
        put_digest(&mut raw, 304, value.descriptor_digest);
        put_u32(&mut raw, 336, value.owner_display_oid);
        put_u64(&mut raw, 344, value.owner_data_generation);
        put_u64(&mut raw, 352, value.base_index_generation);
        put_u64(&mut raw, 360, value.final_index_generation);
        out.extend_from_slice(&raw);
    }
}

fn append_index_keys(out: &mut Vec<u8>, values: &[RetainedIndexKeyColumn]) {
    for value in values {
        let mut raw = [0_u8; 112];
        put_u32(&mut raw, 0, value.key_column_ref);
        put_u32(&mut raw, 4, value.index_ref);
        put_u32(&mut raw, 8, value.key_ordinal);
        put_u32(&mut raw, 12, value.owner_catalog_column_ordinal);
        put_u32(&mut raw, 16, value.stable_column_id);
        put_u32(&mut raw, 20, value.owner_display_table_oid);
        put_i16(&mut raw, 24, value.attnum);
        raw[28..32].copy_from_slice(&value.storage);
        put_u32(&mut raw, 32, value.declared_type_oid);
        put_i16(&mut raw, 36, value.signed_type_size);
        put_digest(&mut raw, 40, value.column_name_digest);
        put_digest(&mut raw, 72, value.key_digest);
        out.extend_from_slice(&raw);
    }
}

fn append_transitions(out: &mut Vec<u8>, values: &[RetainedTransition]) {
    for value in values {
        let mut raw = [0_u8; 192];
        put_u32(&mut raw, 0, value.transition_ref);
        put_u32(&mut raw, 4, value.table_ref);
        put_u64(&mut raw, 8, value.stable_row_id);
        // S7's transition kind is a fixed `final image row` tag for every retained entry.
        raw[16] = 1;
        put_u32(&mut raw, 20, value.source_disposition_ref);
        put_u32(&mut raw, 24, value.source_statement_ordinal);
        put_u32(&mut raw, 28, value.source_row_ordinal);
        put_u32(&mut raw, 32, value.image_ref);
        put_u32(&mut raw, 36, value.image_row_ordinal);
        put_u32(&mut raw, 40, value.key_effect_start);
        put_u32(&mut raw, 44, value.key_effect_count);
        put_u32(&mut raw, 48, value.final_writer_statement_ordinal);
        raw[17] = u8::from(value.final_writer_statement_digest != [0; 32]);
        put_digest(&mut raw, 64, value.typed_statement_digest);
        put_digest(&mut raw, 96, value.final_row_digest);
        put_digest(&mut raw, 128, value.transition_digest);
        put_digest(&mut raw, 160, value.final_writer_statement_digest);
        out.extend_from_slice(&raw);
    }
}

fn append_key_effects(out: &mut Vec<u8>, values: &[RetainedKeyEffect]) {
    for value in values {
        let mut raw = [0_u8; 192];
        put_u32(&mut raw, 0, value.effect_ref);
        raw[4] = value.role;
        raw[5] = value.action;
        put_u32(&mut raw, 8, value.transition_ref);
        put_u32(&mut raw, 12, value.index_ref);
        put_u32(&mut raw, 16, value.dependency_ref);
        // Effects only carry their canonical new key; the old-component start is the frozen
        // absent sentinel, distinct from its zero reserved neighbour at offset 24.
        put_u32(&mut raw, 20, u32::MAX);
        put_u32(&mut raw, 28, value.new_component_start);
        put_u32(&mut raw, 32, value.new_component_count);
        put_u32(&mut raw, 36, value.key_arity);
        // The fixed key-effect grammar records the canonical new-side/image action pair.
        raw[41] = 1;
        raw[42] = 1;
        raw[43] = u8::from(value.participates);
        raw[44] = u8::from(value.contains_null);
        put_u32(&mut raw, 48, value.source_catalog_ordinal);
        put_digest(&mut raw, 96, value.typed_key_digest);
        put_digest(&mut raw, 128, value.effect_digest);
        out.extend_from_slice(&raw);
    }
}

fn append_key_components(out: &mut Vec<u8>, values: &[RetainedKeyComponent]) {
    for value in values {
        let mut raw = [0_u8; 128];
        put_u32(&mut raw, 0, value.component_ref);
        put_u32(&mut raw, 4, value.effect_ref);
        raw[8] = value.side;
        raw[9] = value.validity;
        put_u32(&mut raw, 12, value.component_ordinal);
        put_u32(&mut raw, 16, value.key_column_ref);
        put_u32(&mut raw, 20, value.source_catalog_ordinal);
        put_u64(&mut raw, 24, value.value_arena_offset);
        put_u32(&mut raw, 32, value.value_bytes);
        raw[36..40].copy_from_slice(&value.storage);
        put_u32(&mut raw, 40, value.declared_type_oid);
        put_i16(&mut raw, 44, value.signed_type_size);
        put_digest(&mut raw, 48, value.typed_value_digest);
        put_digest(&mut raw, 80, value.component_digest);
        out.extend_from_slice(&raw);
    }
}

fn append_projections(out: &mut Vec<u8>, values: &[RetainedProjectionBinding]) {
    for value in values {
        let mut raw = [0_u8; 128];
        put_u32(&mut raw, 0, value.projection_ref);
        put_u32(&mut raw, 4, value.statement_ordinal);
        put_u32(&mut raw, 8, value.projection_ordinal);
        put_u32(&mut raw, 12, value.source_catalog_ordinal);
        put_u32(&mut raw, 16, value.stable_column_id);
        put_u32(&mut raw, 20, value.table_ref);
        put_i16(&mut raw, 24, value.attnum);
        raw[28..32].copy_from_slice(&value.storage);
        put_u32(&mut raw, 32, value.declared_type_oid);
        put_i16(&mut raw, 36, value.signed_type_size);
        put_u16(&mut raw, 38, value.result_format);
        put_u32(&mut raw, 40, value.s2_projection_ordinal);
        put_digest(&mut raw, 64, value.name_digest);
        put_digest(&mut raw, 96, value.projection_digest);
        out.extend_from_slice(&raw);
    }
}

fn encode_images(graph: &ReservedSemanticsV2Graph) -> Result<Vec<Vec<u8>>, EngineError> {
    let mut encoded = Vec::with_capacity(graph.images.len());
    for image in &graph.images {
        let bytes = reencode_decoded_typed_image_for_test(image)?;
        encoded.push(bytes);
    }
    Ok(encoded)
}

fn append_image_descriptors(
    out: &mut Vec<u8>,
    graph: &ReservedSemanticsV2Graph,
    images: &[Vec<u8>],
) -> Result<(), EngineError> {
    if images.len() != graph.images.len() || images.len() != graph.tables.len() {
        return Err(reencode_error(
            "retained image/table directory count differs",
        ));
    }
    let mut expected_offset = 0_u64;
    for (ordinal, ((table, image), bytes)) in graph
        .tables
        .iter()
        .zip(&graph.images)
        .zip(images)
        .enumerate()
    {
        let facts = image.facts();
        let encoded_bytes = u64::try_from(bytes.len())
            .map_err(|_| reencode_error("reconstructed image exceeds u64"))?;
        if table.image_ref != ordinal as u32
            || table.table_ref != ordinal as u32
            || facts.role != TypedImageRole::FinalTableImage
            || facts.rows != table.transition_count
            || facts.columns != table.catalog_column_count
            || facts.layout_digest != table.image_layout_digest
            || table.image_arena_offset != expected_offset
            || encoded_bytes != table.image_encoded_bytes
            || image_content_digest(bytes) != table.image_content_digest
        {
            return Err(reencode_error(
                "reconstructed image differs from retained S7 descriptor facts",
            ));
        }
        let mut raw = [0_u8; 160];
        put_u32(&mut raw, 0, table.image_ref);
        put_u32(&mut raw, 4, table.table_ref);
        put_u32(&mut raw, 8, 1);
        put_u32(&mut raw, 16, facts.rows);
        put_u32(&mut raw, 20, facts.columns);
        put_u64(&mut raw, 24, table.image_arena_offset);
        put_u64(&mut raw, 32, encoded_bytes);
        put_digest(&mut raw, 64, table.image_layout_digest);
        put_digest(&mut raw, 96, table.image_content_digest);
        let descriptor = domain_digest(
            b"gpu-db/write001/s7-image-descriptor/v2",
            &[&raw[..128], &[0; 32]],
        );
        if descriptor != table.image_descriptor_digest {
            return Err(reencode_error(
                "reconstructed image descriptor differs from retained digest",
            ));
        }
        put_digest(&mut raw, 128, descriptor);
        out.extend_from_slice(&raw);
        expected_offset = expected_offset
            .checked_add(encoded_bytes)
            .ok_or_else(|| reencode_error("reconstructed image arena offset overflows"))?;
    }
    Ok(())
}

fn encode_key_values(graph: &ReservedSemanticsV2Graph) -> Result<Vec<u8>, EngineError> {
    let mut out = Vec::new();
    for component in &graph.key_components {
        if component.value_arena_offset != out.len() as u64 {
            return Err(reencode_error(
                "retained key-value offset differs from reconstructed arena",
            ));
        }
        let effect = graph
            .key_effects
            .get(component.effect_ref as usize)
            .ok_or_else(|| reencode_error("key component has no retained effect"))?;
        let transition = graph
            .transitions
            .get(effect.transition_ref as usize)
            .ok_or_else(|| reencode_error("key effect has no retained transition"))?;
        let image = graph
            .images
            .get(transition.image_ref as usize)
            .ok_or_else(|| reencode_error("key transition has no retained image"))?;
        let (validity, value, digest) = image_cell_value(
            image,
            transition.image_row_ordinal,
            component.source_catalog_ordinal,
            component.storage,
            component.declared_type_oid,
            component.signed_type_size,
        )?;
        if validity != component.validity
            || u32::try_from(value.len()).ok() != Some(component.value_bytes)
            || digest != component.typed_value_digest
        {
            return Err(reencode_error(
                "reconstructed key value differs from retained component facts",
            ));
        }
        out.extend_from_slice(&value);
    }
    Ok(out)
}

fn image_cell_value(
    image: &DecodedTypedImage,
    row: u32,
    catalog_column: u32,
    storage: [u8; 4],
    type_oid: u32,
    type_size: i16,
) -> Result<(u8, Vec<u8>, [u8; 32]), EngineError> {
    let facts = image.facts();
    let column = image
        .columns()
        .find(|column| column.catalog_column_ordinal == catalog_column)
        .ok_or_else(|| reencode_error("key component source column is absent"))?;
    if sql_storage(column.ty) != storage
        || column.type_oid != type_oid
        || column.type_size != type_size
    {
        return Err(reencode_error("key component image metadata differs"));
    }
    let row = usize::try_from(row).map_err(|_| reencode_error("key image row exceeds usize"))?;
    if row >= facts.rows as usize {
        return Err(reencode_error("key image row is outside decoded image"));
    }
    let valid = match column.validity {
        TypedInsertColumnValidity::AllValid => true,
        TypedInsertColumnValidity::Bitmap(words) => words
            .get(row / 32)
            .is_some_and(|word| word & (1 << (row % 32)) != 0),
    };
    let value = if valid {
        scalar_image_bytes(column.values, column.ty, row)?
    } else {
        Vec::new()
    };
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/s7-typed-key-value/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(storage);
    digest.update(type_oid.to_le_bytes());
    digest.update(type_size.to_le_bytes());
    digest.update([u8::from(!valid)]);
    digest.update((value.len() as u32).to_le_bytes());
    digest.update(&value);
    Ok((u8::from(!valid), value, digest.finalize().into()))
}

fn scalar_image_bytes(
    values: &TypedInsertColumnValues,
    ty: SqlType,
    row: usize,
) -> Result<Vec<u8>, EngineError> {
    let value = match (values, ty) {
        (TypedInsertColumnValues::I32(values), SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
            values
                .get(row)
                .ok_or_else(|| reencode_error("i32 image vector is short"))?
                .to_le_bytes()
                .to_vec()
        }
        (TypedInsertColumnValues::I64(values), SqlType::Int8 | SqlType::Timestamp) => values
            .get(row)
            .ok_or_else(|| reencode_error("i64 image vector is short"))?
            .to_le_bytes()
            .to_vec(),
        (TypedInsertColumnValues::I128(values), SqlType::Numeric { .. }) => values
            .get(row)
            .ok_or_else(|| reencode_error("numeric image vector is short"))?
            .to_le_bytes()
            .to_vec(),
        (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid) => values
            .get(row)
            .ok_or_else(|| reencode_error("UUID image vector is short"))?
            .to_vec(),
        (TypedInsertColumnValues::BoolBits(words), SqlType::Bool) => vec![u8::from(
            words
                .get(row / 32)
                .ok_or_else(|| reencode_error("bool image vector is short"))?
                & (1 << (row % 32))
                != 0,
        )],
        (TypedInsertColumnValues::Text { offsets, bytes }, SqlType::Text) => {
            let start = *offsets
                .get(row)
                .ok_or_else(|| reencode_error("text image offsets are short"))?
                as usize;
            let end = *offsets
                .get(row + 1)
                .ok_or_else(|| reencode_error("text image offsets are short"))?
                as usize;
            bytes
                .get(start..end)
                .ok_or_else(|| reencode_error("text image range is invalid"))?
                .to_vec()
        }
        _ => return Err(reencode_error("image vector arm does not match SQL type")),
    };
    Ok(value)
}

fn image_content_digest(bytes: &[u8]) -> [u8; 32] {
    domain_digest(
        b"gpu-db/write001/s7-image-content/v2",
        &[&(bytes.len() as u64).to_le_bytes(), bytes],
    )
}

fn reconstructed_root_descriptor(
    graph: &ReservedSemanticsV2Graph,
) -> Result<[u8; 32], EngineError> {
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/s7-root-descriptor/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(graph.header.root_descriptor_version.to_le_bytes());
    digest.update(graph.header.catalog_before_epoch.to_le_bytes());
    digest.update(graph.header.catalog_after_epoch.to_le_bytes());
    digest.update(graph.header.catalog_before_digest);
    digest.update(graph.header.catalog_after_digest);
    digest.update(graph.header.initial_database_root);
    digest.update(graph.header.final_database_root);
    digest.update(graph.header.initial_overlay_root);
    digest.update(graph.header.final_overlay_root);
    digest.update(
        u32::try_from(graph.tables.len())
            .map_err(|_| reencode_error("root descriptor table count exceeds u32"))?
            .to_le_bytes(),
    );
    for (ordinal, table) in graph.tables.iter().enumerate() {
        if table.table_ref != ordinal as u32 {
            return Err(reencode_error("root descriptor table order differs"));
        }
        digest.update(table.table_ref.to_le_bytes());
        digest.update(
            (u32::from(table.resets_existing_rows) | (u32::from(table.initial_table_absent) << 1))
                .to_le_bytes(),
        );
        digest.update(table.stable_table_id.to_le_bytes());
        digest.update(table.data_generation_before.to_le_bytes());
        digest.update(table.data_generation_after.to_le_bytes());
        digest.update(table.initial_table_root);
        digest.update(table.final_table_root);
        digest.update(table.manifest_digest);
    }
    Ok(digest.finalize().into())
}

fn reconstructed_payload_digest(s7: &[u8]) -> Result<[u8; 32], EngineError> {
    if s7.len() < S7_HEADER_BYTES {
        return Err(reencode_error("reconstructed S7 lacks a fixed header"));
    }
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/s7-payload/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(
        u64::try_from(s7.len())
            .map_err(|_| reencode_error("reconstructed S7 length exceeds u64"))?
            .to_le_bytes(),
    );
    digest.update(&s7[..568]);
    digest.update([0; 32]);
    digest.update(&s7[600..S7_HEADER_BYTES]);
    digest.update(&s7[S7_HEADER_BYTES..]);
    Ok(digest.finalize().into())
}

fn s2_digest(bytes: &[u8]) -> [u8; 32] {
    domain_digest(
        b"gpu-db/write001/s7-s2-record/v2",
        &[&(bytes.len() as u32).to_le_bytes(), bytes],
    )
}

fn domain_digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    for field in fields {
        digest.update(field);
    }
    digest.finalize().into()
}

fn sql_storage(ty: SqlType) -> [u8; 4] {
    match ty {
        SqlType::Int2 => [1, 0, 0, 0],
        SqlType::Int4 => [2, 0, 0, 0],
        SqlType::Int8 => [3, 0, 0, 0],
        SqlType::Numeric { precision, scale } => [4, precision, scale, 0],
        SqlType::Bool => [5, 0, 0, 0],
        SqlType::Text => [6, 0, 0, 0],
        SqlType::Date => [7, 0, 0, 0],
        SqlType::Timestamp => [8, 0, 0, 0],
        SqlType::Uuid => [9, 0, 0, 0],
    }
}

fn put_u16(raw: &mut [u8], offset: usize, value: u16) {
    raw[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_i16(raw: &mut [u8], offset: usize, value: i16) {
    raw[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(raw: &mut [u8], offset: usize, value: u32) {
    raw[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(raw: &mut [u8], offset: usize, value: u64) {
    raw[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_digest(raw: &mut [u8], offset: usize, value: [u8; 32]) {
    raw[offset..offset + 32].copy_from_slice(&value);
}

fn reencode_error(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 reencode: {message}"
    ))
}
