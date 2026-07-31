//! Index descriptors, exact effect inventory, and typed-component/image closure.

use super::error;
use super::rows::typed_value_facts_from_image;
use super::statements::{begin, exact, range};
use crate::typed_insert_aggregate::semantics_v2::retained::graph::{
    ReservedSemanticsV2Graph, RetainedIndexDescriptor, RetainedKeyComponent, RetainedKeyEffect,
};
use crate::EngineError;
use sha2::Digest;

const ABSENT_U32: u32 = u32::MAX;
const MAINTENANCE: u8 = 1;
const UNIQUE_GUARD: u8 = 2;
const FOREIGN_KEY_GUARD: u8 = 3;

pub(super) fn validate(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    validate_indexes(graph)?;
    validate_effect_directory(graph)?;
    let mut expected_table_effect = 0_u32;
    for table in &graph.tables {
        let transitions = range(
            &graph.transitions,
            table.transition_start,
            table.transition_count,
            "index table transition",
        )?;
        if table.key_effect_start != expected_table_effect {
            return Err(error(
                "table key-effect ranges are not globally concatenated",
            ));
        }
        let mut expected_transition_effect = table.key_effect_start;
        for transition in transitions {
            let effects = range(
                &graph.key_effects,
                transition.key_effect_start,
                transition.key_effect_count,
                "transition key effect",
            )?;
            if transition.key_effect_start != expected_transition_effect {
                return Err(error(
                    "transition key-effect ranges are not table-concatenated",
                ));
            }
            let record = graph
                .records
                .get(transition.source_statement_ordinal as usize)
                .ok_or_else(|| error("effect source S2 is absent"))?;
            let mut expected = 0_usize;
            for index in graph
                .indexes
                .iter()
                .filter(|index| index.owner_table_ref == table.table_ref)
            {
                exact_role_effect(
                    graph,
                    effects,
                    MAINTENANCE,
                    index.index_ref,
                    index.raw_catalog_ordinal,
                    transition.transition_ref,
                )?;
                expected += 1;
                if index.flags & 1 != 0 {
                    exact_role_effect(
                        graph,
                        effects,
                        UNIQUE_GUARD,
                        index.index_ref,
                        index.raw_catalog_ordinal,
                        transition.transition_ref,
                    )?;
                    expected += 1;
                }
            }
            for foreign_key in record.foreign_keys() {
                let descriptor = graph
                    .indexes
                    .iter()
                    .find(|index| index.display_oid == foreign_key.supporting_index.oid)
                    .ok_or_else(|| error("S2 FK supporting index has no S7 descriptor"))?;
                exact_role_effect(
                    graph,
                    effects,
                    FOREIGN_KEY_GUARD,
                    descriptor.index_ref,
                    foreign_key.raw_ordinal,
                    transition.transition_ref,
                )?;
            }
            let expected = expected + record.foreign_keys().len();
            if effects.len() != expected {
                return Err(error(
                    "transition has an extra or missing maintenance/unique/FK effect",
                ));
            }
            for effect in effects {
                validate_effect(graph, transition, effect)?;
            }
            if transition.transition_digest != transition_digest(transition, effects) {
                return Err(error(
                    "transition digest differs from final-row/effect closure",
                ));
            }
            expected_transition_effect = expected_transition_effect
                .checked_add(transition.key_effect_count)
                .ok_or_else(|| error("transition key-effect count overflows"))?;
        }
        if expected_transition_effect
            != table
                .key_effect_start
                .checked_add(table.key_effect_count)
                .ok_or_else(|| error("table key-effect count overflows"))?
        {
            return Err(error(
                "table key-effect range does not exactly exhaust its transitions",
            ));
        }
        expected_table_effect = expected_transition_effect;
    }
    if expected_table_effect as usize != graph.key_effects.len() {
        return Err(error(
            "table key-effect ranges do not exhaust effect directory",
        ));
    }
    Ok(())
}

fn validate_indexes(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    let mut next_key = 0_u32;
    let mut previous_identity = None;
    for (ordinal, index) in graph.indexes.iter().enumerate() {
        let keys = range(
            &graph.index_key_columns,
            index.key_start,
            index.key_count,
            "index key",
        )?;
        if index.owner_table_ref != ABSENT_U32 {
            let owner = graph
                .tables
                .get(index.owner_table_ref as usize)
                .ok_or_else(|| error("target-owned index table is absent"))?;
            if index.owner_stable_table_id != owner.stable_table_id
                || index.owner_display_oid != owner.display_oid
                || index.catalog_epoch != owner.catalog_epoch
                || index.owner_schema_digest != owner.schema_digest
                || index.owner_table_base_root != owner.initial_table_root
                || index.owner_data_generation != owner.data_generation_before
            {
                return Err(error("target-owned index does not close its table block"));
            }
        }
        let maintenance_effects = graph
            .key_effects
            .iter()
            .filter(|effect| effect.index_ref == index.index_ref && effect.role == MAINTENANCE)
            .count();
        if (index.owner_table_ref == ABSENT_U32 && maintenance_effects != 0)
            || (maintenance_effects == 0
                && (index.final_index_generation != index.base_index_generation
                    || index.final_index_root != index.base_index_root))
            || (maintenance_effects != 0
                && (index.final_index_generation <= index.base_index_generation
                    || index.final_index_root == index.base_index_root))
        {
            return Err(error(
                "index generation/root law does not match maintenance effects",
            ));
        }
        if index.index_ref != ordinal as u32
            || index.key_start != next_key
            || index.key_count == 0
            || previous_identity.is_some_and(|previous| {
                previous >= (index.owner_stable_table_id, index.stable_index_id)
            })
            || keys.iter().enumerate().any(|(key_ordinal, key)| {
                key.key_column_ref != index.key_start + key_ordinal as u32
                    || key.index_ref != index.index_ref
                    || key.key_ordinal != key_ordinal as u32
            })
        {
            return Err(error("index descriptor/key ranges are not dense"));
        }
        for key in keys {
            if key.key_digest != index_key_digest(key) {
                return Err(error(
                    "index-key descriptor digest differs from retained fields",
                ));
            }
        }
        let mut raw = [0_u8; 384];
        raw[..4].copy_from_slice(&index.index_ref.to_le_bytes());
        raw[4..8].copy_from_slice(&index.flags.to_le_bytes());
        raw[8..12].copy_from_slice(&index.owner_table_ref.to_le_bytes());
        raw[12..16].copy_from_slice(&index.raw_catalog_ordinal.to_le_bytes());
        raw[16..24].copy_from_slice(&index.stable_index_id.to_le_bytes());
        raw[24..28].copy_from_slice(&index.display_oid.to_le_bytes());
        raw[28..32].copy_from_slice(&index.constraint_display_oid.to_le_bytes());
        raw[32..40].copy_from_slice(&index.stable_constraint_id.to_le_bytes());
        raw[40..44].copy_from_slice(&index.key_start.to_le_bytes());
        raw[44..48].copy_from_slice(&index.key_count.to_le_bytes());
        raw[48] = index.null_equality_policy;
        raw[64..72].copy_from_slice(&index.owner_stable_table_id.to_le_bytes());
        raw[72..80].copy_from_slice(&index.catalog_epoch.to_le_bytes());
        raw[80..112].copy_from_slice(&index.owner_schema_digest);
        raw[112..144].copy_from_slice(&index.owner_name_digest);
        raw[144..176].copy_from_slice(&index.index_name_digest);
        raw[176..208].copy_from_slice(&index.constraint_name_digest);
        raw[208..240].copy_from_slice(&index.owner_table_base_root);
        raw[240..272].copy_from_slice(&index.base_index_root);
        raw[272..304].copy_from_slice(&index.final_index_root);
        raw[336..340].copy_from_slice(&index.owner_display_oid.to_le_bytes());
        raw[344..352].copy_from_slice(&index.owner_data_generation.to_le_bytes());
        raw[352..360].copy_from_slice(&index.base_index_generation.to_le_bytes());
        raw[360..368].copy_from_slice(&index.final_index_generation.to_le_bytes());
        let mut digest = begin(b"gpu-db/write001/s7-index-descriptor/v2");
        digest.update(&raw[..304]);
        digest.update([0; 32]);
        digest.update(&raw[336..]);
        for key in keys {
            digest.update(key.key_digest);
        }
        let digest: [u8; 32] = digest.finalize().into();
        if digest != index.descriptor_digest {
            return Err(error("index descriptor digest differs from retained keys"));
        }
        previous_identity = Some((index.owner_stable_table_id, index.stable_index_id));
        next_key = next_key
            .checked_add(index.key_count)
            .ok_or_else(|| error("index key count overflows"))?;
    }
    if next_key as usize != graph.index_key_columns.len() {
        return Err(error("index key ranges do not exhaust key directory"));
    }
    for table in &graph.tables {
        let expected_start = graph
            .indexes
            .iter()
            .take_while(|index| {
                (index.owner_stable_table_id, index.stable_index_id) < (table.stable_table_id, 0)
            })
            .count();
        let expected_count = graph
            .indexes
            .iter()
            .filter(|index| index.owner_table_ref == table.table_ref)
            .count();
        let expected_start = u32::try_from(expected_start)
            .map_err(|_| error("owned-index lower bound exceeds u32"))?;
        let expected_count =
            u32::try_from(expected_count).map_err(|_| error("owned-index count exceeds u32"))?;
        let owned = range(
            &graph.indexes,
            table.owned_index_start,
            table.owned_index_count,
            "owned-index run",
        )?;
        if table.owned_index_start != expected_start
            || table.owned_index_count != expected_count
            || owned
                .iter()
                .any(|index| index.owner_table_ref != table.table_ref)
            || owned.len() != expected_count as usize
            || graph
                .indexes
                .iter()
                .filter(|index| index.owner_table_ref == table.table_ref)
                .any(|index| {
                    index.index_ref < table.owned_index_start
                        || index.index_ref
                            >= table
                                .owned_index_start
                                .saturating_add(table.owned_index_count)
                })
        {
            return Err(error(
                "table owned-index range omits, overlaps, or includes a parent-only descriptor",
            ));
        }
    }
    validate_descriptor_source_union(graph)?;
    Ok(())
}

/// S7 owns one deduplicated descriptor directory for target indexes and FK supporting
/// indexes. Static dependency uses prove the reverse direction; this pass prevents an
/// unattached descriptor (or a duplicate display OID) from escaping through a parent-only
/// owner shape.
fn validate_descriptor_source_union(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    for (ordinal, descriptor) in graph.indexes.iter().enumerate() {
        if graph.indexes[..ordinal]
            .iter()
            .any(|prior| prior.display_oid == descriptor.display_oid)
        {
            return Err(error("index descriptor display OIDs are not unique"));
        }
        let mut source_count = 0_u32;
        for resolution in &graph.resolutions {
            let record = graph
                .records
                .get(resolution.record_ref as usize)
                .ok_or_else(|| error("descriptor-union S2 record is absent"))?;
            for source in record
                .indexes()
                .filter(|source| source.owner_dependency_ordinal == 0)
            {
                if super::dependencies::descriptor_matches_s2_index(
                    graph, record, descriptor, source,
                )? && super::dependencies::descriptor_keys_match_s2_index(
                    graph, record, descriptor, source,
                )? {
                    source_count = source_count
                        .checked_add(1)
                        .ok_or_else(|| error("descriptor source count overflows"))?;
                }
            }
            for foreign_key in record.foreign_keys() {
                if super::dependencies::descriptor_matches_foreign_key(
                    graph,
                    record,
                    descriptor,
                    foreign_key,
                )? && super::dependencies::descriptor_keys_match_foreign_key(
                    graph,
                    record,
                    descriptor,
                    foreign_key,
                )? {
                    source_count = source_count
                        .checked_add(1)
                        .ok_or_else(|| error("descriptor source count overflows"))?;
                }
            }
        }
        if source_count == 0 {
            return Err(error(
                "index descriptor is outside the exact S2 target/FK supporting-index union",
            ));
        }
    }
    Ok(())
}

fn validate_effect_directory(graph: &ReservedSemanticsV2Graph) -> Result<(), EngineError> {
    let mut next_component = 0_u32;
    let mut next_value_offset = 0_u64;
    for (ordinal, effect) in graph.key_effects.iter().enumerate() {
        let components = range(
            &graph.key_components,
            effect.new_component_start,
            effect.new_component_count,
            "key-effect component",
        )?;
        if effect.effect_ref != ordinal as u32 || effect.new_component_start != next_component {
            return Err(error(
                "key-effect/component directories or typed-key value arena are not dense",
            ));
        }
        for (component_ordinal, component) in components.iter().enumerate() {
            if component.component_ref != effect.new_component_start + component_ordinal as u32
                || component.effect_ref != effect.effect_ref
                || component.value_arena_offset != next_value_offset
                || (component.validity == 1 && component.value_bytes != 0)
            {
                return Err(error(
                    "key-effect/component directories or typed-key value arena are not dense",
                ));
            }
            next_value_offset = next_value_offset
                .checked_add(u64::from(component.value_bytes))
                .ok_or_else(|| error("typed-key value arena offset overflows"))?;
        }
        next_component = next_component
            .checked_add(effect.new_component_count)
            .ok_or_else(|| error("key-effect component count overflows"))?;
    }
    if next_component as usize != graph.key_components.len() {
        return Err(error(
            "key-effect component ranges do not exhaust component directory",
        ));
    }
    Ok(())
}

fn exact_role_effect(
    graph: &ReservedSemanticsV2Graph,
    effects: &[RetainedKeyEffect],
    role: u8,
    index_ref: u32,
    source_ordinal: u32,
    transition_ref: u32,
) -> Result<(), EngineError> {
    let mut candidates = effects.iter().filter(|effect| {
        effect.role == role
            && effect.index_ref == index_ref
            && effect.source_catalog_ordinal == source_ordinal
    });
    let effect = candidates
        .next()
        .ok_or_else(|| error("expected index effect is absent"))?;
    if candidates.next().is_some() || effect.transition_ref != transition_ref {
        return Err(error("index effect inventory is not bijective"));
    }
    let index = graph
        .indexes
        .get(index_ref as usize)
        .ok_or_else(|| error("effect index is absent"))?;
    if effect.key_arity != index.key_count || effect.new_component_count != index.key_count {
        return Err(error("effect key arity differs from descriptor"));
    }
    Ok(())
}

fn validate_effect(
    graph: &ReservedSemanticsV2Graph,
    transition: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedTransition,
    effect: &RetainedKeyEffect,
) -> Result<(), EngineError> {
    let index = graph
        .indexes
        .get(effect.index_ref as usize)
        .ok_or_else(|| error("effect index is absent"))?;
    let table = graph
        .tables
        .get(transition.table_ref as usize)
        .ok_or_else(|| error("effect table is absent"))?;
    let image = graph
        .images
        .get(table.image_ref as usize)
        .ok_or_else(|| error("effect image is absent"))?;
    let record = graph
        .records
        .get(transition.source_statement_ordinal as usize)
        .ok_or_else(|| error("effect S2 record is absent"))?;
    let components = range(
        &graph.key_components,
        effect.new_component_start,
        effect.new_component_count,
        "effect component",
    )?;
    let keys = range(
        &graph.index_key_columns,
        index.key_start,
        index.key_count,
        "effect index key",
    )?;
    if effect.effect_ref as usize >= graph.key_effects.len()
        || effect.transition_ref != transition.transition_ref
        || effect.action != effect.role
        || !matches!(effect.role, MAINTENANCE | UNIQUE_GUARD | FOREIGN_KEY_GUARD)
        || effect.key_arity != components.len() as u32
        || effect.key_arity != keys.len() as u32
        || effect.new_component_start
            != components
                .first()
                .map_or(effect.new_component_start, |first| first.component_ref)
    {
        return Err(error("key effect scalar closure is invalid"));
    }
    let contains_null = components.iter().any(|component| component.validity == 1);
    let participates = effect.role == MAINTENANCE || !contains_null;
    if effect.contains_null != contains_null || effect.participates != participates {
        return Err(error("key effect NULL participation is invalid"));
    }
    for (ordinal, (component, key)) in components.iter().zip(keys).enumerate() {
        validate_component(
            graph,
            image,
            transition.image_row_ordinal,
            effect,
            component,
            key,
            ordinal as u32,
        )?;
    }
    validate_effect_source(record, index, effect, components, keys)?;
    let mut typed_key = begin(b"gpu-db/write001/s7-typed-key/v2");
    typed_key.update(effect.effect_ref.to_le_bytes());
    typed_key.update([2]);
    typed_key.update(effect.key_arity.to_le_bytes());
    for component in components {
        typed_key.update(component.component_digest);
    }
    let typed_key: [u8; 32] = typed_key.finalize().into();
    if typed_key != effect.typed_key_digest {
        return Err(error("typed key digest differs from components"));
    }
    if effect.effect_digest != effect_digest(effect, index, components) {
        return Err(error("key effect digest differs from retained components"));
    }
    Ok(())
}

fn validate_effect_source(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    index: &RetainedIndexDescriptor,
    effect: &RetainedKeyEffect,
    components: &[RetainedKeyComponent],
    keys: &[crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedIndexKeyColumn],
) -> Result<(), EngineError> {
    match effect.role {
        MAINTENANCE | UNIQUE_GUARD => {
            if components.iter().zip(keys).any(|(component, key)| {
                component.source_catalog_ordinal != key.owner_catalog_column_ordinal
                    || component.storage != key.storage
                    || component.declared_type_oid != key.declared_type_oid
                    || component.signed_type_size != key.signed_type_size
            }) {
                return Err(error(
                    "target index effect does not use its exact catalog-order key",
                ));
            }
            let mut matching = record.indexes().filter(|source| {
                source.owner_dependency_ordinal == 0
                    && source.raw_ordinal == effect.source_catalog_ordinal
                    && source.oid == index.display_oid
                    && source.unique == (index.flags & 1 != 0)
                    && source.primary_key == (index.flags & 2 != 0)
                    && source.unique_constraint == (index.flags & 4 != 0)
                    && source.key_count == index.key_count
            });
            if matching.next().is_none() || matching.next().is_some() {
                return Err(error("target index effect has no exact S2 index source"));
            }
        }
        FOREIGN_KEY_GUARD => {
            let foreign_key = record
                .foreign_keys()
                .find(|foreign_key| foreign_key.raw_ordinal == effect.source_catalog_ordinal)
                .ok_or_else(|| error("FK effect has no exact S2 foreign-key source"))?;
            if foreign_key.supporting_index.oid != index.display_oid
                || foreign_key.supporting_index.raw_ordinal != index.raw_catalog_ordinal
                || foreign_key.supporting_index.key_count != index.key_count
            {
                return Err(error(
                    "FK effect does not use its exact supporting parent index",
                ));
            }
            let source_keys = record.foreign_key_supporting_index_keys(foreign_key.raw_ordinal)?;
            if components.len() != 1 || keys.len() != 1 || source_keys.len() != 1 {
                return Err(error(
                    "current S2 FK closure admits exactly one supporting key",
                ));
            }
            for ((component, key), source_key) in components.iter().zip(keys).zip(source_keys) {
                if key.owner_catalog_column_ordinal != source_key.catalog_column_ordinal
                    || key.stable_column_id != source_key.column_id
                    || key.attnum != source_key.attnum
                    || key.storage != super::statements::storage(source_key.ty)
                    || key.declared_type_oid != source_key.type_oid
                    || key.signed_type_size != source_key.type_size
                    || component.source_catalog_ordinal
                        != foreign_key.child_column.catalog_column_ordinal
                    || component.storage != super::statements::storage(foreign_key.child_column.ty)
                    || component.declared_type_oid != foreign_key.child_column.type_oid
                    || component.signed_type_size != foreign_key.child_column.type_size
                {
                    return Err(error(
                        "FK component does not bind child values to parent key order",
                    ));
                }
            }
        }
        _ => return Err(error("key effect role is invalid")),
    }
    Ok(())
}

fn validate_component(
    graph: &ReservedSemanticsV2Graph,
    image: &crate::typed_insert_batch::DecodedTypedImage,
    image_row: u32,
    effect: &RetainedKeyEffect,
    component: &RetainedKeyComponent,
    key: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedIndexKeyColumn,
    ordinal: u32,
) -> Result<(), EngineError> {
    let (expected_validity, expected_bytes, expected) = typed_value_facts_from_image(
        image,
        image_row,
        component.source_catalog_ordinal,
        component.storage,
        component.declared_type_oid,
        component.signed_type_size,
    )?;
    if component.component_ref != effect.new_component_start + ordinal
        || component.effect_ref != effect.effect_ref
        || component.side != 2
        || component.component_ordinal != ordinal
        || component.key_column_ref != key.key_column_ref
        || component.validity > 1
        || component.validity != expected_validity
        || component.value_bytes != expected_bytes
        || expected != component.typed_value_digest
        || component.component_digest != component_digest(component, key)
    {
        return Err(error(
            "typed key component differs from image/key descriptor",
        ));
    }
    let _ = graph;
    Ok(())
}

fn component_digest(
    component: &RetainedKeyComponent,
    key: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedIndexKeyColumn,
) -> [u8; 32] {
    let mut raw = [0_u8; 80];
    raw[..4].copy_from_slice(&component.component_ref.to_le_bytes());
    raw[4..8].copy_from_slice(&component.effect_ref.to_le_bytes());
    raw[8] = component.side;
    raw[9] = component.validity;
    raw[12..16].copy_from_slice(&component.component_ordinal.to_le_bytes());
    raw[16..20].copy_from_slice(&component.key_column_ref.to_le_bytes());
    raw[20..24].copy_from_slice(&component.source_catalog_ordinal.to_le_bytes());
    raw[24..32].copy_from_slice(&component.value_arena_offset.to_le_bytes());
    raw[32..36].copy_from_slice(&component.value_bytes.to_le_bytes());
    raw[36..40].copy_from_slice(&component.storage);
    raw[40..44].copy_from_slice(&component.declared_type_oid.to_le_bytes());
    raw[44..46].copy_from_slice(&component.signed_type_size.to_le_bytes());
    raw[48..80].copy_from_slice(&component.typed_value_digest);
    exact(
        b"gpu-db/write001/s7-typed-key-component/v2",
        &[&raw, &[0; 32], &[0; 16], &key.key_digest],
    )
}

fn index_key_digest(
    key: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedIndexKeyColumn,
) -> [u8; 32] {
    let mut raw = [0_u8; 112];
    raw[..4].copy_from_slice(&key.key_column_ref.to_le_bytes());
    raw[4..8].copy_from_slice(&key.index_ref.to_le_bytes());
    raw[8..12].copy_from_slice(&key.key_ordinal.to_le_bytes());
    raw[12..16].copy_from_slice(&key.owner_catalog_column_ordinal.to_le_bytes());
    raw[16..20].copy_from_slice(&key.stable_column_id.to_le_bytes());
    raw[20..24].copy_from_slice(&key.owner_display_table_oid.to_le_bytes());
    raw[24..26].copy_from_slice(&key.attnum.to_le_bytes());
    raw[28..32].copy_from_slice(&key.storage);
    raw[32..36].copy_from_slice(&key.declared_type_oid.to_le_bytes());
    raw[36..38].copy_from_slice(&key.signed_type_size.to_le_bytes());
    raw[40..72].copy_from_slice(&key.column_name_digest);
    exact(
        b"gpu-db/write001/s7-index-key-column/v2",
        &[&raw[..72], &[0; 32], &raw[104..]],
    )
}

fn transition_digest(
    transition: &crate::typed_insert_aggregate::semantics_v2::retained::graph::RetainedTransition,
    effects: &[RetainedKeyEffect],
) -> [u8; 32] {
    let mut raw = [0_u8; 192];
    raw[..4].copy_from_slice(&transition.transition_ref.to_le_bytes());
    raw[4..8].copy_from_slice(&transition.table_ref.to_le_bytes());
    raw[8..16].copy_from_slice(&transition.stable_row_id.to_le_bytes());
    raw[16] = 1;
    raw[20..24].copy_from_slice(&transition.source_disposition_ref.to_le_bytes());
    raw[24..28].copy_from_slice(&transition.source_statement_ordinal.to_le_bytes());
    raw[28..32].copy_from_slice(&transition.source_row_ordinal.to_le_bytes());
    raw[32..36].copy_from_slice(&transition.image_ref.to_le_bytes());
    raw[36..40].copy_from_slice(&transition.image_row_ordinal.to_le_bytes());
    raw[40..44].copy_from_slice(&transition.key_effect_start.to_le_bytes());
    raw[44..48].copy_from_slice(&transition.key_effect_count.to_le_bytes());
    raw[48..52].copy_from_slice(&transition.final_writer_statement_ordinal.to_le_bytes());
    raw[64..96].copy_from_slice(&transition.typed_statement_digest);
    raw[96..128].copy_from_slice(&transition.final_row_digest);
    let mut digest = begin(b"gpu-db/write001/s7-transition/v2");
    digest.update(&raw[..128]);
    digest.update([0; 32]);
    digest.update(&raw[160..]);
    for effect in effects {
        digest.update(effect.effect_digest);
    }
    digest.finalize().into()
}

fn effect_digest(
    effect: &RetainedKeyEffect,
    index: &RetainedIndexDescriptor,
    components: &[RetainedKeyComponent],
) -> [u8; 32] {
    let mut raw = [0_u8; 192];
    raw[..4].copy_from_slice(&effect.effect_ref.to_le_bytes());
    raw[4] = effect.role;
    raw[5] = effect.action;
    raw[8..12].copy_from_slice(&effect.transition_ref.to_le_bytes());
    raw[12..16].copy_from_slice(&effect.index_ref.to_le_bytes());
    raw[16..20].copy_from_slice(&ABSENT_U32.to_le_bytes());
    raw[20..24].copy_from_slice(&ABSENT_U32.to_le_bytes());
    raw[28..32].copy_from_slice(&effect.new_component_start.to_le_bytes());
    raw[32..36].copy_from_slice(&effect.new_component_count.to_le_bytes());
    raw[36..40].copy_from_slice(&effect.key_arity.to_le_bytes());
    raw[41] = 1;
    raw[42] = 1;
    raw[43] = u8::from(effect.participates);
    raw[44] = u8::from(effect.contains_null);
    raw[48..52].copy_from_slice(&effect.source_catalog_ordinal.to_le_bytes());
    raw[96..128].copy_from_slice(&effect.typed_key_digest);
    let mut digest = begin(b"gpu-db/write001/s7-key-effect/v2");
    digest.update(&raw[..16]);
    digest.update(ABSENT_U32.to_le_bytes());
    digest.update(&raw[20..128]);
    digest.update([0; 32]);
    digest.update(&raw[160..]);
    digest.update(index.descriptor_digest);
    for component in components {
        digest.update(component.component_digest);
    }
    digest.finalize().into()
}
