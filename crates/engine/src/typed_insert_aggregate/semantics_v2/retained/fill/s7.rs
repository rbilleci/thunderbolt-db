//! Exact fixed-directory and nested-image fill for S7.
//!
//! Pass zero owns full grammar, ordering, digest, and arena coverage proof.  This module copies
//! its already-proven fixed facts into compact owners, then strictly rereads every image through
//! the shared measured source decoder.  It retains neither a S7 body nor any typed-key arena.

use super::fixed::{digest_at, i16_at, push_exact, u16_at, u32_at, u64_at};
use super::source::{exact_copy_scratch, fill_error, AggregateRegionSource};
use super::ObservedSourceMeasure;
use crate::typed_insert_aggregate::codec::DecodedAggregateFraming;
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedIndexDescriptor,
    RetainedIndexKeyColumn, RetainedKeyComponent, RetainedKeyEffect, RetainedProjectionBinding,
    RetainedStatementDependencyUse, RetainedStatementResolution, RetainedTable,
    RetainedTableDisposition, RetainedTransition,
};
use crate::typed_insert_batch::{
    copy_typed_image_after_measure, decode_typed_image_after_measure,
    measure_decoded_typed_image_from_source, TypedImageRole,
};
use crate::EngineError;
use sha2::{Digest, Sha256};

const S7_HEADER_BYTES: u64 = 640;

pub(super) fn fill_s7(
    framing: &DecodedAggregateFraming<'_>,
    graph: &mut ReservedSemanticsV2Graph,
    observed: &mut ObservedSourceMeasure,
) -> Result<(), EngineError> {
    framing.with_section_reader(6, |reader| {
        let header = reader.exact::<640>()?;
        let image_arena_bytes = u64_at(&header, 96);
        let image_arena_start = u64_at(&header, 104 + 13 * 16);
        validate_header(&header, graph)?;

        fill_tables(reader, graph)?;
        fill_table_dispositions(reader, graph)?;
        fill_resolutions(reader, graph)?;
        fill_dependencies(reader, graph)?;
        fill_dependency_uses(reader, graph)?;
        fill_indexes(reader, graph)?;
        fill_index_key_columns(reader, graph)?;
        fill_transitions(reader, graph)?;
        fill_key_effects(reader, graph)?;
        fill_key_components(reader, graph)?;
        fill_projections(reader, graph)?;
        fill_images(framing, reader, graph, observed, image_arena_start)?;

        // Key values are intentionally recoverable only through the typed final-image owners;
        // image copy/decode is complete before this anonymous byte arena is discarded.
        let value_arena_bytes = u64_at(&header, 88);
        reader.skip(value_arena_bytes)?;
        reader.skip(image_arena_bytes)?;
        Ok(())
    })
}

fn validate_header(
    header: &[u8; 640],
    graph: &ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    let identity = graph.header;
    if u64_at(header, 32) != identity.total_bytes
        || u16_at(header, 30) != identity.root_descriptor_version
        || u64_at(header, 328) != identity.catalog_before_epoch
        || u64_at(header, 336) != identity.catalog_after_epoch
        || digest_at(header, 344) != identity.catalog_before_digest
        || digest_at(header, 376) != identity.catalog_after_digest
        || digest_at(header, 408) != identity.initial_database_root
        || digest_at(header, 440) != identity.final_database_root
        || digest_at(header, 472) != identity.initial_overlay_root
        || digest_at(header, 504) != identity.final_overlay_root
        || digest_at(header, 536) != identity.root_descriptor
        || digest_at(header, 568) != identity.payload_digest
    {
        return Err(fill_error(
            "S7 header changed after pass-zero identity proof",
        ));
    }
    Ok(())
}

fn fill_tables(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.tables.capacity() {
        let raw = reader.exact::<384>()?;
        push_exact(
            &mut graph.tables,
            RetainedTable {
                table_ref: u32_at(&raw, 0),
                resets_existing_rows: u32_at(&raw, 4) & 1 != 0,
                initial_table_absent: u32_at(&raw, 4) & 2 != 0,
                stable_table_id: u64_at(&raw, 8),
                display_oid: u32_at(&raw, 16),
                target_dependency_ref: u32_at(&raw, 20),
                catalog_epoch: u64_at(&raw, 24),
                data_generation_before: u64_at(&raw, 32),
                data_generation_after: u64_at(&raw, 40),
                row_allocator_before: u64_at(&raw, 48),
                row_allocator_high_water: u64_at(&raw, 56),
                initial_logical_row_count: u64_at(&raw, 64),
                final_logical_row_count: u64_at(&raw, 72),
                disposition_start: u32_at(&raw, 80),
                disposition_count: u32_at(&raw, 84),
                transition_start: u32_at(&raw, 88),
                transition_count: u32_at(&raw, 92),
                owned_index_start: u32_at(&raw, 96),
                owned_index_count: u32_at(&raw, 100),
                key_effect_start: u32_at(&raw, 104),
                key_effect_count: u32_at(&raw, 108),
                image_ref: u32_at(&raw, 112),
                catalog_column_count: u32_at(&raw, 116),
                schema_digest: digest_at(&raw, 128),
                initial_table_root: digest_at(&raw, 160),
                final_table_root: digest_at(&raw, 192),
                image_layout_digest: digest_at(&raw, 224),
                image_content_digest: digest_at(&raw, 256),
                transition_root: digest_at(&raw, 288),
                index_effect_root: digest_at(&raw, 320),
                image_arena_offset: 0,
                image_encoded_bytes: 0,
                image_descriptor_digest: [0; 32],
                manifest_digest: digest_at(&raw, 352),
            },
            "S7 table directory",
        )?;
    }
    Ok(())
}

fn fill_table_dispositions(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.table_dispositions.capacity() {
        let raw = reader.exact::<32>()?;
        push_exact(
            &mut graph.table_dispositions,
            RetainedTableDisposition {
                table_ref: u32_at(&raw, 0),
                disposition_ref: u32_at(&raw, 4),
                stable_row_id: u64_at(&raw, 8),
                statement_ordinal: u32_at(&raw, 16),
                source_row_ordinal: u32_at(&raw, 20),
                disposition: raw[24],
            },
            "S7 table-disposition directory",
        )?;
    }
    Ok(())
}

fn fill_resolutions(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.resolutions.capacity() {
        let raw = reader.exact::<320>()?;
        push_exact(
            &mut graph.resolutions,
            RetainedStatementResolution {
                statement_ordinal: u32_at(&raw, 0),
                record_ref: u32_at(&raw, 8),
                outcome_ref: u32_at(&raw, 12),
                table_ref: u32_at(&raw, 16),
                flags: u32_at(&raw, 20),
                s4_start: u32_at(&raw, 24),
                s4_count: u32_at(&raw, 28),
                s5_start: u32_at(&raw, 32),
                s5_count: u32_at(&raw, 36),
                dependency_use_start: u32_at(&raw, 40),
                dependency_use_count: u32_at(&raw, 44),
                projection_start: u32_at(&raw, 48),
                projection_count: u32_at(&raw, 52),
                input_row_count: u32_at(&raw, 56),
                surviving_row_count: u32_at(&raw, 60),
                affected_row_count: u64_at(&raw, 64),
                dependency_validation_floor: u64_at(&raw, 72),
                record_bytes: u32_at(&raw, 80),
                terminal_dependency_ref: u32_at(&raw, 84),
                terminal_row_ordinal: u32_at(&raw, 88),
                terminal_source_ordinal: u32_at(&raw, 92),
                request_digest: digest_at(&raw, 96),
                typed_statement_digest: digest_at(&raw, 128),
                record_digest: digest_at(&raw, 160),
                returning_digest: digest_at(&raw, 192),
                overlay_before: digest_at(&raw, 224),
                overlay_after: digest_at(&raw, 256),
                outcome_digest: digest_at(&raw, 288),
            },
            "S7 statement-resolution directory",
        )?;
    }
    Ok(())
}

fn fill_dependencies(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.dependencies.capacity() {
        let raw = reader.exact::<224>()?;
        push_exact(
            &mut graph.dependencies,
            RetainedDependencyToken {
                dependency_ref: u32_at(&raw, 0),
                kind: raw[4],
                access: raw[5],
                flags: u16_at(&raw, 6),
                stable_object_id: u64_at(&raw, 8),
                display_oid: u32_at(&raw, 16),
                target_table_ref: u32_at(&raw, 20),
                base_generation: u64_at(&raw, 24),
                snapshot_floor: u64_at(&raw, 32),
                key_effect_ref: u32_at(&raw, 40),
                descriptor_ref: u32_at(&raw, 44),
                catalog_epoch: u64_at(&raw, 48),
                schema_digest: digest_at(&raw, 64),
                base_root: digest_at(&raw, 96),
                name_digest: digest_at(&raw, 128),
                identity_digest: digest_at(&raw, 160),
                token_digest: digest_at(&raw, 192),
            },
            "S7 dependency-token directory",
        )?;
    }
    Ok(())
}

fn fill_dependency_uses(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.dependency_uses.capacity() {
        let raw = reader.exact::<32>()?;
        push_exact(
            &mut graph.dependency_uses,
            RetainedStatementDependencyUse {
                statement_ordinal: u32_at(&raw, 0),
                dependency_ref: u32_at(&raw, 4),
                role: u16_at(&raw, 8),
                source_ordinal: u32_at(&raw, 12),
                transition_ref: u32_at(&raw, 16),
                key_effect_ref: u32_at(&raw, 20),
            },
            "S7 statement-dependency-use directory",
        )?;
    }
    Ok(())
}

fn fill_indexes(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.indexes.capacity() {
        let raw = reader.exact::<384>()?;
        push_exact(
            &mut graph.indexes,
            RetainedIndexDescriptor {
                index_ref: u32_at(&raw, 0),
                owner_table_ref: u32_at(&raw, 8),
                raw_catalog_ordinal: u32_at(&raw, 12),
                stable_index_id: u64_at(&raw, 16),
                display_oid: u32_at(&raw, 24),
                constraint_display_oid: u32_at(&raw, 28),
                stable_constraint_id: u64_at(&raw, 32),
                flags: u32_at(&raw, 4),
                null_equality_policy: raw[48],
                key_start: u32_at(&raw, 40),
                key_count: u32_at(&raw, 44),
                owner_stable_table_id: u64_at(&raw, 64),
                catalog_epoch: u64_at(&raw, 72),
                owner_schema_digest: digest_at(&raw, 80),
                owner_name_digest: digest_at(&raw, 112),
                index_name_digest: digest_at(&raw, 144),
                constraint_name_digest: digest_at(&raw, 176),
                owner_table_base_root: digest_at(&raw, 208),
                base_index_root: digest_at(&raw, 240),
                final_index_root: digest_at(&raw, 272),
                descriptor_digest: digest_at(&raw, 304),
                owner_display_oid: u32_at(&raw, 336),
                owner_data_generation: u64_at(&raw, 344),
                base_index_generation: u64_at(&raw, 352),
                final_index_generation: u64_at(&raw, 360),
            },
            "S7 index-descriptor directory",
        )?;
    }
    Ok(())
}

fn fill_index_key_columns(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.index_key_columns.capacity() {
        let raw = reader.exact::<112>()?;
        push_exact(
            &mut graph.index_key_columns,
            RetainedIndexKeyColumn {
                key_column_ref: u32_at(&raw, 0),
                index_ref: u32_at(&raw, 4),
                key_ordinal: u32_at(&raw, 8),
                owner_catalog_column_ordinal: u32_at(&raw, 12),
                stable_column_id: u32_at(&raw, 16),
                owner_display_table_oid: u32_at(&raw, 20),
                attnum: i16_at(&raw, 24),
                storage: raw[28..32].try_into().expect("fixed key storage"),
                declared_type_oid: u32_at(&raw, 32),
                signed_type_size: i16_at(&raw, 36),
                column_name_digest: digest_at(&raw, 40),
                key_digest: digest_at(&raw, 72),
            },
            "S7 index-key-column directory",
        )?;
    }
    Ok(())
}

fn fill_transitions(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.transitions.capacity() {
        let raw = reader.exact::<192>()?;
        push_exact(
            &mut graph.transitions,
            RetainedTransition {
                transition_ref: u32_at(&raw, 0),
                table_ref: u32_at(&raw, 4),
                stable_row_id: u64_at(&raw, 8),
                source_disposition_ref: u32_at(&raw, 20),
                source_statement_ordinal: u32_at(&raw, 24),
                source_row_ordinal: u32_at(&raw, 28),
                image_ref: u32_at(&raw, 32),
                image_row_ordinal: u32_at(&raw, 36),
                key_effect_start: u32_at(&raw, 40),
                key_effect_count: u32_at(&raw, 44),
                final_writer_statement_ordinal: u32_at(&raw, 48),
                final_writer_statement_digest: digest_at(&raw, 160),
                typed_statement_digest: digest_at(&raw, 64),
                final_row_digest: digest_at(&raw, 96),
                transition_digest: digest_at(&raw, 128),
            },
            "S7 transition directory",
        )?;
    }
    Ok(())
}

fn fill_key_effects(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.key_effects.capacity() {
        let raw = reader.exact::<192>()?;
        push_exact(
            &mut graph.key_effects,
            RetainedKeyEffect {
                effect_ref: u32_at(&raw, 0),
                role: raw[4],
                action: raw[5],
                transition_ref: u32_at(&raw, 8),
                index_ref: u32_at(&raw, 12),
                dependency_ref: u32_at(&raw, 16),
                new_component_start: u32_at(&raw, 28),
                new_component_count: u32_at(&raw, 32),
                key_arity: u32_at(&raw, 36),
                participates: raw[43] != 0,
                contains_null: raw[44] != 0,
                source_catalog_ordinal: u32_at(&raw, 48),
                typed_key_digest: digest_at(&raw, 96),
                effect_digest: digest_at(&raw, 128),
            },
            "S7 key-effect directory",
        )?;
    }
    Ok(())
}

fn fill_key_components(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.key_components.capacity() {
        let raw = reader.exact::<128>()?;
        push_exact(
            &mut graph.key_components,
            RetainedKeyComponent {
                component_ref: u32_at(&raw, 0),
                effect_ref: u32_at(&raw, 4),
                side: raw[8],
                validity: raw[9],
                component_ordinal: u32_at(&raw, 12),
                key_column_ref: u32_at(&raw, 16),
                source_catalog_ordinal: u32_at(&raw, 20),
                value_arena_offset: u64_at(&raw, 24),
                value_bytes: u32_at(&raw, 32),
                storage: raw[36..40].try_into().expect("fixed component storage"),
                declared_type_oid: u32_at(&raw, 40),
                signed_type_size: i16_at(&raw, 44),
                typed_value_digest: digest_at(&raw, 48),
                component_digest: digest_at(&raw, 80),
            },
            "S7 typed-key-component directory",
        )?;
    }
    Ok(())
}

fn fill_projections(
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
) -> Result<(), EngineError> {
    for _ in 0..graph.projections.capacity() {
        let raw = reader.exact::<128>()?;
        let statement_ordinal = u32_at(&raw, 4);
        let record_ref = graph
            .resolutions
            .get(usize::try_from(statement_ordinal).map_err(|_| {
                fill_error("S7 projection statement ordinal exceeds host addressability")
            })?)
            .ok_or_else(|| fill_error("S7 projection has no retained statement resolution"))?
            .record_ref;
        push_exact(
            &mut graph.projections,
            RetainedProjectionBinding {
                projection_ref: u32_at(&raw, 0),
                statement_ordinal,
                projection_ordinal: u32_at(&raw, 8),
                source_catalog_ordinal: u32_at(&raw, 12),
                stable_column_id: u32_at(&raw, 16),
                table_ref: u32_at(&raw, 20),
                attnum: i16_at(&raw, 24),
                storage: raw[28..32].try_into().expect("fixed projection storage"),
                declared_type_oid: u32_at(&raw, 32),
                signed_type_size: i16_at(&raw, 36),
                record_ref,
                result_format: u16_at(&raw, 38),
                s2_projection_ordinal: u32_at(&raw, 40),
                name_digest: digest_at(&raw, 64),
                projection_digest: digest_at(&raw, 96),
            },
            "S7 projection-binding directory",
        )?;
    }
    Ok(())
}

fn fill_images(
    framing: &DecodedAggregateFraming<'_>,
    reader: &mut crate::typed_insert_aggregate::codec::DecodedAggregateSectionReader<'_>,
    graph: &mut ReservedSemanticsV2Graph,
    observed: &mut ObservedSourceMeasure,
    image_arena_start: u64,
) -> Result<(), EngineError> {
    for _ in 0..graph.images.capacity() {
        let raw = reader.exact::<160>()?;
        let image_ref = u32_at(&raw, 0);
        let table_ref = u32_at(&raw, 4);
        let bytes = u64_at(&raw, 32);
        let source = AggregateRegionSource::new(
            framing,
            6,
            image_arena_start
                .checked_add(u64_at(&raw, 24))
                .ok_or_else(|| fill_error("S7 image arena source offset overflows"))?,
            bytes,
        );
        let measure = measure_decoded_typed_image_from_source(&source)
            .map_err(|_| fill_error("S7 retained image source measurement fails strict decode"))?;
        if measure.image_bytes() != bytes {
            return Err(fill_error(
                "S7 retained image source measure length drifted",
            ));
        }
        let scratch_bytes = measure
            .maximum_with_image_copy_bytes()
            .map_err(|_| fill_error("S7 retained image copy scratch measure overflows"))?;
        let scratch_slots = measure
            .maximum_with_image_copy_allocation_slots()
            .map_err(|_| fill_error("S7 retained image copy scratch slot measure overflows"))?;
        observed.checked_add_image(
            measure.persistent_bytes(),
            measure.persistent_allocation_slots(),
            scratch_bytes,
            scratch_slots,
        )?;

        let mut scratch = exact_copy_scratch(bytes, "S7 exact image source copy")?;
        copy_typed_image_after_measure(&source, measure, &mut scratch)
            .map_err(|_| fill_error("S7 retained image source changed after its measurement"))?;
        let content_digest = image_content_digest(&scratch);
        let decoded = decode_typed_image_after_measure(&scratch, measure)
            .map_err(|_| fill_error("S7 retained exact image copy fails strict decode"))?;
        drop(scratch);

        let table =
            graph
                .tables
                .get_mut(usize::try_from(table_ref).map_err(|_| {
                    fill_error("S7 image table reference exceeds host addressability")
                })?)
                .ok_or_else(|| fill_error("S7 image has no retained table descriptor"))?;
        let facts = decoded.facts();
        if image_ref != graph.images.len() as u32
            || table.table_ref != table_ref
            || table.image_ref != image_ref
            || raw[8] != 1
            || u32_at(&raw, 16) != facts.rows
            || u32_at(&raw, 20) != facts.columns
            || digest_at(&raw, 64) != facts.layout_digest
            || digest_at(&raw, 96) != content_digest
            || table.image_layout_digest != facts.layout_digest
            || table.image_content_digest != content_digest
            || facts.role != TypedImageRole::FinalTableImage
        {
            return Err(fill_error(
                "decoded S7 image does not match its retained descriptor",
            ));
        }
        table.image_arena_offset = u64_at(&raw, 24);
        table.image_encoded_bytes = bytes;
        table.image_descriptor_digest = digest_at(&raw, 128);
        push_exact(&mut graph.images, decoded, "S7 decoded image directory")?;
    }
    Ok(())
}

fn image_content_digest(bytes: &[u8]) -> [u8; 32] {
    let domain = b"gpu-db/write001/s7-image-content/v2";
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    digest.finalize().into()
}
