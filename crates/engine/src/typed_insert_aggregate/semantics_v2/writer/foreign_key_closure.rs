//! Foreign-key dependency closure for the canonical INSERT writer.
//!
//! The transaction finalizer has already obtained the device FK verdict. This module serializes
//! the exact sealed S2 child-to-parent binding and immutable parent table/index generations into
//! S7. Parent descriptors are proof inputs only: they have no maintenance effect, successor, or
//! publication authority.

use super::*;
use std::collections::BTreeSet;

struct ForeignDescriptor<'a> {
    generation: &'a LiveTypedInsertForeignIndexGeneration<'a>,
    raw: [u8; 384],
    digest: [u8; 32],
    keys: Vec<IndexedS7Key>,
    shared_initial_parent: bool,
}

struct OwnedDescriptor {
    raw_ordinal: u32,
    index_ref: u32,
    static_dependency_ref: u32,
    unique: bool,
    keys: Vec<IndexedS7Key>,
    raw: [u8; 384],
    digest: [u8; 32],
}

#[derive(Clone, Copy)]
struct GuardBinding {
    descriptor_ref: u32,
    effect_ref: u32,
    transition_ref: u32,
    statement_ordinal: u32,
    source_ordinal: u32,
}

pub(super) fn from_sealed_records(
    input: &LiveTypedInsertView<'_>,
    records: &[DecodedTypedInsertRecord],
    rows: &[BoundRow],
    final_row_digests: &[[u8; 32]],
    bases: IndexedS7Bases,
    shared_initial_foreign_indexes: &[SharedInitialForeignIndexDescriptor],
) -> Result<IndexedS7Closure, EngineError> {
    if input.foreign_indexes.is_empty() || final_row_digests.len() != rows.len() {
        return Err(error("foreign-key S7 closure input geometry is invalid"));
    }
    validate_records(input, records)?;
    let final_image = crate::typed_insert_batch::decode_typed_image(input.final_image)?;
    if final_image.facts().role != crate::typed_insert_batch::TypedImageRole::FinalTableImage
        || usize::try_from(final_image.facts().rows).ok() != Some(rows.len())
    {
        return Err(error(
            "foreign-key S7 final image differs from its surviving row set",
        ));
    }

    let mut generations = input.foreign_indexes.iter().collect::<Vec<_>>();
    generations.sort_unstable_by_key(|generation| {
        (
            generation.parent.stable_table_id,
            u64::from(generation.catalog.oid),
        )
    });
    if generations.windows(2).any(|pair| {
        (pair[0].parent.stable_table_id, pair[0].catalog.oid)
            == (pair[1].parent.stable_table_id, pair[1].catalog.oid)
    }) {
        return Err(error(
            "foreign-key S7 parent index generations are duplicated",
        ));
    }

    // S7 has one globally identity-ordered, deduplicated descriptor directory. An FK may refer
    // to an initial parent created earlier in this same transaction; that parent has already
    // emitted its owned descriptor, which is reused here as the one shared index identity.
    // Only genuinely parent-only descriptors occupy this closure's before/after runs.
    let shared_for_generation = generations
        .iter()
        .map(|generation| {
            shared_initial_foreign_indexes
                .iter()
                .find(|shared| shared_matches_generation(shared, generation))
        })
        .collect::<Vec<_>>();
    if shared_for_generation
        .iter()
        .flatten()
        .any(|shared| !shared_initial_descriptor_is_final(shared))
    {
        return Err(error(
            "shared initial FK descriptor is not its parent table's final index generation",
        ));
    }
    let foreign_before_count = generations
        .iter()
        .zip(&shared_for_generation)
        .filter(|(generation, shared)| {
            shared.is_none() && generation.parent.stable_table_id < input.table.stable_table_id
        })
        .count();
    let foreign_before_count_u32 = u32::try_from(foreign_before_count)
        .map_err(|_| error("foreign-key S7 descriptor count exceeds u32"))?;
    let owned_index_count = u32::try_from(input.table.indexes.len())
        .map_err(|_| error("S7 owned-index count exceeds u32"))?;
    let owned_index_start = bases
        .index_ref
        .checked_add(foreign_before_count_u32)
        .ok_or_else(|| error("S7 owned-index lower bound overflows"))?;
    let owned_key_start = bases
        .key_start
        .checked_add(foreign_before_count_u32)
        .ok_or_else(|| error("S7 owned-index key lower bound overflows"))?;
    let parent_count = u32::try_from(
        generations
            .iter()
            .map(|generation| generation.parent.stable_table_id)
            .collect::<BTreeSet<_>>()
            .len(),
    )
    .map_err(|_| error("foreign-key S7 parent count exceeds u32"))?;
    let owned_dependency_start = bases
        .dependency_start
        .checked_add(parent_count)
        .ok_or_else(|| error("S7 owned-index dependency lower bound overflows"))?;
    let owned_skeleton = if input.table.indexes.is_empty() {
        IndexedS7Closure {
            dependencies: Vec::new(),
            dependency_uses: Vec::new(),
            indexes: Vec::new(),
            index_digests: Vec::new(),
            index_keys: Vec::new(),
            transitions: Vec::new(),
            effects: Vec::new(),
            effect_digests: Vec::new(),
            components: Vec::new(),
            values: Vec::new(),
            owned_index_start,
            owned_index_count: 0,
        }
    } else {
        IndexedS7Closure::from_sealed_records(
            input,
            records,
            &[],
            &[],
            IndexedS7Bases {
                table_ref: bases.table_ref,
                index_ref: owned_index_start,
                key_start: owned_key_start,
                effect_start: bases.effect_start,
                component_start: bases.component_start,
                value_offset: bases.value_offset,
                dependency_start: owned_dependency_start,
            },
        )?
    };
    if owned_skeleton.owned_index_start != owned_index_start
        || owned_skeleton.owned_index_count != owned_index_count
    {
        return Err(error("combined S7 owned-index skeleton drifted"));
    }
    let owned = owned_descriptors(&owned_skeleton, owned_dependency_start)?;
    let owned_key_count = u32::try_from(owned_skeleton.index_keys.len())
        .map_err(|_| error("S7 owned-index key count exceeds u32"))?;

    let mut descriptors = Vec::with_capacity(generations.len());
    let mut next_key_ref = bases.key_start;
    let mut before_ordinal = 0_u32;
    for (generation, shared) in generations.iter().zip(&shared_for_generation) {
        if generation.parent.stable_table_id >= input.table.stable_table_id {
            break;
        }
        if let Some(shared) = shared {
            descriptors.push(shared_descriptor_for_generation(generation, shared));
            continue;
        }
        descriptors.push(descriptor_for_generation(
            input,
            records,
            generation,
            bases
                .index_ref
                .checked_add(before_ordinal)
                .ok_or_else(|| error("foreign-key S7 descriptor reference overflows"))?,
            &mut next_key_ref,
        )?);
        before_ordinal = before_ordinal
            .checked_add(1)
            .ok_or_else(|| error("foreign-key S7 descriptor count exceeds u32"))?;
    }
    if before_ordinal != foreign_before_count_u32 {
        return Err(error("foreign-key S7 before-descriptor count drifted"));
    }
    next_key_ref = owned_key_start
        .checked_add(owned_key_count)
        .ok_or_else(|| error("foreign-key S7 key reference overflows"))?;
    let mut after_ordinal = 0_u32;
    for (generation, shared) in generations.iter().zip(&shared_for_generation) {
        if generation.parent.stable_table_id < input.table.stable_table_id {
            continue;
        }
        if let Some(shared) = shared {
            descriptors.push(shared_descriptor_for_generation(generation, shared));
            continue;
        }
        descriptors.push(descriptor_for_generation(
            input,
            records,
            generation,
            owned_index_start
                .checked_add(owned_index_count)
                .and_then(|start| start.checked_add(after_ordinal))
                .ok_or_else(|| error("foreign-key S7 descriptor reference overflows"))?,
            &mut next_key_ref,
        )?);
        after_ordinal = after_ordinal
            .checked_add(1)
            .ok_or_else(|| error("foreign-key S7 descriptor count exceeds u32"))?;
    }

    for statement in input.statements {
        let record = records
            .get(statement.statement_ordinal as usize)
            .ok_or_else(|| error("foreign-key S7 statement record is absent"))?;
        for foreign_key in record.foreign_keys() {
            if descriptor_ref(record, foreign_key, &descriptors).is_none() {
                return Err(error(
                    "sealed S2 foreign key has no immutable parent index generation",
                ));
            }
        }
    }

    let mut components = Vec::new();
    let mut values = Vec::new();
    let mut effects = Vec::new();
    let mut effect_digests = Vec::new();
    let mut guards = Vec::new();
    let unique_dependency_start = owned_dependency_start
        .checked_add(owned_index_count)
        .ok_or_else(|| error("S7 unique dependency lower bound overflows"))?;
    let mut unique_bindings = Vec::new();
    let mut row_effect_ranges = Vec::with_capacity(rows.len());
    for row in rows {
        let record = records
            .get(row.statement_index)
            .ok_or_else(|| error("foreign-key S7 row statement record is absent"))?;
        let statement = input
            .statements
            .iter()
            .find(|statement| statement.statement_ordinal as usize == row.statement_index)
            .ok_or_else(|| error("foreign-key S7 row statement binding is absent"))?;
        let image_row = row
            .image_row
            .ok_or_else(|| error("foreign-key S7 row has no final image coordinate"))?;
        let effect_start = bases
            .effect_start
            .checked_add(
                u32::try_from(effects.len()).map_err(|_| error("S7 effect count exceeds u32"))?,
            )
            .ok_or_else(|| error("S7 effect reference overflows"))?;
        for (owned_ordinal, index) in owned.iter().enumerate() {
            append_index_effect(
                &final_image,
                &index.keys,
                index.digest,
                row.transition_ref,
                image_row,
                1,
                index.static_dependency_ref,
                index.index_ref,
                index.raw_ordinal,
                bases.effect_start,
                bases.component_start,
                bases.value_offset,
                &mut components,
                &mut values,
                &mut effects,
                &mut effect_digests,
            )?;
            if index.unique {
                let dependency_ref = unique_dependency_start
                    .checked_add(
                        u32::try_from(unique_bindings.len())
                            .map_err(|_| error("S7 unique dependency count exceeds u32"))?,
                    )
                    .ok_or_else(|| error("S7 unique dependency reference overflows"))?;
                let (effect_ref, participates) = append_index_effect(
                    &final_image,
                    &index.keys,
                    index.digest,
                    row.transition_ref,
                    image_row,
                    2,
                    dependency_ref,
                    index.index_ref,
                    index.raw_ordinal,
                    bases.effect_start,
                    bases.component_start,
                    bases.value_offset,
                    &mut components,
                    &mut values,
                    &mut effects,
                    &mut effect_digests,
                )?;
                if participates {
                    unique_bindings.push((
                        dependency_ref,
                        effect_ref,
                        row.transition_ref,
                        statement.statement_ordinal,
                        owned_ordinal,
                    ));
                }
            }
        }
        for foreign_key in record.foreign_keys() {
            let descriptor_ref = descriptor_ref(record, foreign_key, &descriptors)
                .ok_or_else(|| error("foreign-key S7 descriptor binding is absent"))?;
            let descriptor = &descriptors[descriptor_ref as usize];
            let (effect_ref, participates) = append_foreign_effect(
                &final_image,
                foreign_key,
                descriptor_index_ref(descriptor),
                &descriptor.keys,
                descriptor.digest,
                row.transition_ref,
                image_row,
                bases.effect_start,
                bases.component_start,
                bases.value_offset,
                &mut components,
                &mut values,
                &mut effects,
                &mut effect_digests,
            )?;
            if participates {
                guards.push(GuardBinding {
                    descriptor_ref,
                    effect_ref,
                    transition_ref: row.transition_ref,
                    statement_ordinal: statement.statement_ordinal,
                    source_ordinal: foreign_key.raw_ordinal,
                });
            }
        }
        let effect_count = bases
            .effect_start
            .checked_add(
                u32::try_from(effects.len()).map_err(|_| error("S7 effect count exceeds u32"))?,
            )
            .ok_or_else(|| error("S7 effect reference overflows"))?
            .checked_sub(effect_start)
            .ok_or_else(|| error("S7 row effect range underflows"))?;
        row_effect_ranges.push((effect_start, effect_count));
    }

    let mut transitions = Vec::with_capacity(rows.len());
    for ((row, final_row_digest), (effect_start, effect_count)) in
        rows.iter().zip(final_row_digests).zip(&row_effect_ranges)
    {
        let statement = input
            .statements
            .iter()
            .find(|statement| statement.statement_ordinal as usize == row.statement_index)
            .ok_or_else(|| error("foreign-key S7 transition statement is absent"))?;
        let mut transition = transition_bytes(
            row.transition_ref,
            row.table_ref,
            row.stable_row_id,
            row.s4_ref,
            statement.statement_ordinal,
            row.source_row,
            row.table_ref,
            row.image_row.expect("foreign-key row has a final image"),
            *effect_start,
            *effect_count,
            row.final_writer_statement_ordinal,
            statement.typed_statement_digest,
            *final_row_digest,
            [0; 32],
            row.final_writer_statement_digest,
        );
        let end = effect_start
            .checked_add(*effect_count)
            .ok_or_else(|| error("S7 transition effect range overflows"))?;
        let digests = effect_digests
            .get(
                effect_start
                    .checked_sub(bases.effect_start)
                    .ok_or_else(|| error("S7 transition effect range underflows"))?
                    as usize
                    ..end
                        .checked_sub(bases.effect_start)
                        .ok_or_else(|| error("S7 transition effect range underflows"))?
                        as usize,
            )
            .ok_or_else(|| error("S7 transition effect range is absent"))?;
        let digest = v2_digest(
            b"gpu-db/write001/s7-transition/v2",
            &[
                &transition[..128],
                &[0; 32],
                &transition[160..],
                &digests.concat(),
            ],
        );
        put_digest(&mut transition, 128, digest);
        transitions.push(transition);
    }

    let parents = parent_generations(&descriptors);
    let mut dependencies = Vec::new();
    for generation in &parents {
        dependencies.push(parent_dependency(
            bases
                .dependency_start
                .checked_add(
                    u32::try_from(dependencies.len())
                        .map_err(|_| error("S7 dependency count exceeds u32"))?,
                )
                .ok_or_else(|| error("S7 dependency reference overflows"))?,
            bases.table_ref,
            input,
            generation,
        )?);
    }

    dependencies.extend(owned_skeleton.dependencies.iter().copied());
    let mut dependency_uses = owned_skeleton.dependency_uses.clone();
    for (dependency_ref, effect_ref, transition_ref, statement_ordinal, owned_ordinal) in
        unique_bindings
    {
        let local_effect_ref = effect_ref
            .checked_sub(bases.effect_start)
            .ok_or_else(|| error("S7 unique effect reference underflows"))?;
        let index = owned
            .get(owned_ordinal)
            .ok_or_else(|| error("S7 unique dependency lost its index descriptor"))?;
        dependencies.push(index_dependency(
            dependency_ref,
            4,
            &index.raw,
            index.digest,
            input.identity.dependency_validation_floor,
            effect_ref,
            effect_digests[local_effect_ref as usize],
        ));
        dependency_uses.push(dependency_use(
            statement_ordinal,
            dependency_ref,
            3,
            index.raw_ordinal,
            transition_ref,
            effect_ref,
        ));
    }

    let mut static_descriptor_order = (0..descriptors.len()).collect::<Vec<_>>();
    static_descriptor_order.sort_unstable_by_key(|ordinal| {
        let generation = descriptors[*ordinal].generation;
        (
            u64::from(generation.catalog.oid),
            generation.catalog.oid,
            *ordinal,
        )
    });
    let mut static_dependency_refs = vec![ABSENT_U32; descriptors.len()];
    for descriptor_ref in static_descriptor_order {
        let dependency_ref = bases
            .dependency_start
            .checked_add(
                u32::try_from(dependencies.len())
                    .map_err(|_| error("S7 dependency count exceeds u32"))?,
            )
            .ok_or_else(|| error("S7 dependency reference overflows"))?;
        let descriptor = &descriptors[descriptor_ref];
        dependencies.push(foreign_index_dependency(
            dependency_ref,
            5,
            descriptor,
            bases.table_ref,
            input.identity.dependency_validation_floor,
            ABSENT_U32,
            [0; 32],
        ));
        static_dependency_refs[descriptor_ref] = dependency_ref;
    }

    guards.sort_unstable_by_key(|guard| {
        let generation = descriptors[guard.descriptor_ref as usize].generation;
        (
            u64::from(generation.catalog.oid),
            generation.catalog.oid,
            guard.effect_ref,
            guard.descriptor_ref,
        )
    });
    for statement in input.statements {
        let statement_ordinal = statement.statement_ordinal;
        let record = records
            .get(statement_ordinal as usize)
            .ok_or_else(|| error("foreign-key S7 statement record is absent"))?;
        let mut parent_sources = record
            .foreign_keys()
            .map(|foreign_key| foreign_key.parent_dependency_ordinal)
            .collect::<Vec<_>>();
        parent_sources.sort_unstable();
        parent_sources.dedup();
        for source_ordinal in parent_sources {
            let source = record
                .dependencies()
                .nth(source_ordinal as usize)
                .ok_or_else(|| error("S2 foreign parent dependency is absent"))?;
            let parent_ref = parents
                .iter()
                .position(|generation| parent_matches_source(generation, source))
                .and_then(|ordinal| u32::try_from(ordinal).ok())
                .and_then(|ordinal| bases.dependency_start.checked_add(ordinal))
                .ok_or_else(|| error("S2 foreign parent has no S7 table dependency"))?;
            dependency_uses.push(dependency_use(
                statement_ordinal,
                parent_ref,
                4,
                source_ordinal,
                ABSENT_U32,
                ABSENT_U32,
            ));
        }
        for foreign_key in record.foreign_keys() {
            let descriptor_ref = descriptor_ref(record, foreign_key, &descriptors)
                .ok_or_else(|| error("foreign-key S7 descriptor binding is absent"))?;
            dependency_uses.push(dependency_use(
                statement_ordinal,
                static_dependency_refs[descriptor_ref as usize],
                5,
                foreign_key.raw_ordinal,
                ABSENT_U32,
                ABSENT_U32,
            ));
        }
    }
    for guard in guards {
        let dependency_ref = bases
            .dependency_start
            .checked_add(
                u32::try_from(dependencies.len())
                    .map_err(|_| error("S7 dependency count exceeds u32"))?,
            )
            .ok_or_else(|| error("S7 dependency reference overflows"))?;
        let local_effect_ref = guard
            .effect_ref
            .checked_sub(bases.effect_start)
            .ok_or_else(|| error("S7 FK guard effect reference underflows"))?;
        put_u32(
            effects
                .get_mut(local_effect_ref as usize)
                .ok_or_else(|| error("S7 FK guard effect is absent"))?,
            16,
            dependency_ref,
        );
        let descriptor = &descriptors[guard.descriptor_ref as usize];
        dependencies.push(foreign_index_dependency(
            dependency_ref,
            6,
            descriptor,
            bases.table_ref,
            input.identity.dependency_validation_floor,
            guard.effect_ref,
            effect_digests[local_effect_ref as usize],
        ));
        dependency_uses.push(dependency_use(
            guard.statement_ordinal,
            dependency_ref,
            6,
            guard.source_ordinal,
            guard.transition_ref,
            guard.effect_ref,
        ));
    }

    let foreign_before = descriptors.iter().filter(|descriptor| {
        !descriptor.shared_initial_parent
            && descriptor.generation.parent.stable_table_id < input.table.stable_table_id
    });
    let foreign_after = descriptors.iter().filter(|descriptor| {
        !descriptor.shared_initial_parent
            && descriptor.generation.parent.stable_table_id >= input.table.stable_table_id
    });
    let mut indexes = Vec::with_capacity(
        foreign_before_count
            .checked_add(usize::try_from(after_ordinal).expect("u32 fits usize"))
            .and_then(|count| count.checked_add(owned_skeleton.indexes.len()))
            .ok_or_else(|| error("S7 descriptor count overflows"))?,
    );
    indexes.extend(foreign_before.map(|descriptor| descriptor.raw));
    indexes.extend(owned_skeleton.indexes.iter().copied());
    indexes.extend(foreign_after.map(|descriptor| descriptor.raw));
    let mut index_digests = Vec::with_capacity(indexes.len());
    index_digests.extend(descriptors.iter().filter_map(|descriptor| {
        (!descriptor.shared_initial_parent
            && descriptor.generation.parent.stable_table_id < input.table.stable_table_id)
            .then_some(descriptor.digest)
    }));
    index_digests.extend(owned_skeleton.index_digests.iter().copied());
    index_digests.extend(descriptors.iter().filter_map(|descriptor| {
        (!descriptor.shared_initial_parent
            && descriptor.generation.parent.stable_table_id >= input.table.stable_table_id)
            .then_some(descriptor.digest)
    }));
    let mut index_keys = Vec::new();
    index_keys.extend(
        descriptors
            .iter()
            .filter(|descriptor| {
                !descriptor.shared_initial_parent
                    && descriptor.generation.parent.stable_table_id < input.table.stable_table_id
            })
            .flat_map(|descriptor| descriptor.keys.iter().map(|key| key.raw)),
    );
    index_keys.extend(owned_skeleton.index_keys.iter().copied());
    index_keys.extend(
        descriptors
            .iter()
            .filter(|descriptor| {
                !descriptor.shared_initial_parent
                    && descriptor.generation.parent.stable_table_id >= input.table.stable_table_id
            })
            .flat_map(|descriptor| descriptor.keys.iter().map(|key| key.raw)),
    );
    Ok(IndexedS7Closure {
        dependencies,
        dependency_uses,
        indexes,
        index_digests,
        index_keys,
        transitions,
        effects,
        effect_digests,
        components,
        values,
        owned_index_start,
        owned_index_count,
    })
}

fn descriptor_for_generation<'a>(
    input: &LiveTypedInsertView<'_>,
    records: &[DecodedTypedInsertRecord],
    generation: &'a LiveTypedInsertForeignIndexGeneration<'a>,
    index_ref: u32,
    next_key_ref: &mut u32,
) -> Result<ForeignDescriptor<'a>, EngineError> {
    let (record, foreign_key) = input
        .statements
        .iter()
        .find_map(|statement| {
            let record = records.get(statement.statement_ordinal as usize)?;
            record.foreign_keys().find_map(|foreign_key| {
                foreign_matches_generation(record, foreign_key, generation)
                    .then_some((record, foreign_key))
            })
        })
        .ok_or_else(|| error("foreign-key S7 parent index has no sealed S2 source"))?;
    descriptor(
        input,
        record,
        foreign_key,
        generation,
        index_ref,
        next_key_ref,
    )
}

fn shared_matches_generation(
    shared: &SharedInitialForeignIndexDescriptor,
    generation: &LiveTypedInsertForeignIndexGeneration<'_>,
) -> bool {
    shared.owner_stable_table_id == generation.parent.stable_table_id
        && shared.owner_display_oid == generation.parent.oid
        && shared.stable_index_id == u64::from(generation.catalog.oid)
        && shared.display_oid == generation.catalog.oid
        && shared.raw[272..304] == generation.index_root
        && u64::from_le_bytes(
            shared.raw[360..368]
                .try_into()
                .expect("fixed index generation"),
        ) == generation.index_generation
}

fn shared_initial_descriptor_is_final(shared: &SharedInitialForeignIndexDescriptor) -> bool {
    u32::from_le_bytes(shared.raw[8..12].try_into().expect("fixed index owner ref")) != ABSENT_U32
        && shared.raw[208..240] == [0; 32]
        && shared.raw[240..272] == [0; 32]
        && u64::from_le_bytes(
            shared.raw[344..352]
                .try_into()
                .expect("fixed owner generation"),
        ) == 0
        && u64::from_le_bytes(
            shared.raw[352..360]
                .try_into()
                .expect("fixed index generation"),
        ) == 0
        && shared.raw[272..304] != [0; 32]
        && u64::from_le_bytes(
            shared.raw[360..368]
                .try_into()
                .expect("fixed index generation"),
        ) != 0
}

fn shared_descriptor_for_generation<'a>(
    generation: &'a LiveTypedInsertForeignIndexGeneration<'a>,
    shared: &SharedInitialForeignIndexDescriptor,
) -> ForeignDescriptor<'a> {
    ForeignDescriptor {
        generation,
        raw: shared.raw,
        digest: shared.digest,
        keys: shared.keys.clone(),
        shared_initial_parent: true,
    }
}

fn descriptor_index_ref(descriptor: &ForeignDescriptor<'_>) -> u32 {
    u32::from_le_bytes(
        descriptor.raw[..4]
            .try_into()
            .expect("fixed S7 descriptor reference"),
    )
}

fn owned_descriptors(
    closure: &IndexedS7Closure,
    dependency_start: u32,
) -> Result<Vec<OwnedDescriptor>, EngineError> {
    if closure.indexes.len() != closure.index_digests.len()
        || closure.dependencies.len() != closure.indexes.len()
    {
        return Err(error("combined S7 owned-index skeleton is incomplete"));
    }
    let mut descriptors = Vec::with_capacity(closure.indexes.len());
    let mut key_offset = 0_usize;
    for (ordinal, (raw, digest)) in closure
        .indexes
        .iter()
        .zip(&closure.index_digests)
        .enumerate()
    {
        let key_count = usize::try_from(u32::from_le_bytes(
            raw[44..48]
                .try_into()
                .expect("fixed S7 descriptor key count"),
        ))
        .map_err(|_| error("S7 owned-index key count exceeds usize"))?;
        let key_end = key_offset
            .checked_add(key_count)
            .ok_or_else(|| error("S7 owned-index key range overflows"))?;
        let keys = closure
            .index_keys
            .get(key_offset..key_end)
            .ok_or_else(|| error("S7 owned-index key range is absent"))?
            .iter()
            .map(indexed_key_from_raw)
            .collect::<Vec<_>>();
        let static_dependency_ref = dependency_start
            .checked_add(
                u32::try_from(ordinal)
                    .map_err(|_| error("S7 owned-index dependency ordinal exceeds u32"))?,
            )
            .ok_or_else(|| error("S7 owned-index dependency reference overflows"))?;
        if u32::from_le_bytes(
            closure.dependencies[ordinal][..4]
                .try_into()
                .expect("fixed S7 dependency reference"),
        ) != static_dependency_ref
        {
            return Err(error(
                "combined S7 owned-index static dependency reference drifted",
            ));
        }
        descriptors.push(OwnedDescriptor {
            raw_ordinal: u32::from_le_bytes(
                raw[12..16]
                    .try_into()
                    .expect("fixed S7 descriptor catalog ordinal"),
            ),
            index_ref: u32::from_le_bytes(
                raw[..4].try_into().expect("fixed S7 descriptor reference"),
            ),
            static_dependency_ref,
            unique: u32::from_le_bytes(raw[4..8].try_into().expect("fixed S7 descriptor flags"))
                & 1
                != 0,
            keys,
            raw: *raw,
            digest: *digest,
        });
        key_offset = key_end;
    }
    if key_offset != closure.index_keys.len() {
        return Err(error(
            "combined S7 owned-index key directory is not exhausted",
        ));
    }
    Ok(descriptors)
}

fn indexed_key_from_raw(raw: &[u8; 112]) -> IndexedS7Key {
    IndexedS7Key {
        raw: *raw,
        digest: raw[72..104].try_into().expect("fixed S7 index-key digest"),
        catalog_column_ordinal: u32::from_le_bytes(
            raw[12..16]
                .try_into()
                .expect("fixed S7 key catalog ordinal"),
        ),
        stable_column_id: u32::from_le_bytes(
            raw[16..20]
                .try_into()
                .expect("fixed S7 key stable column id"),
        ),
        attnum: i16::from_le_bytes(raw[24..26].try_into().expect("fixed S7 key attnum")),
        storage: raw[28..32].try_into().expect("fixed S7 key storage kind"),
        declared_type_oid: u32::from_le_bytes(
            raw[32..36].try_into().expect("fixed S7 key type oid"),
        ),
        signed_type_size: i16::from_le_bytes(
            raw[36..38].try_into().expect("fixed S7 key type size"),
        ),
    }
}

fn validate_records(
    input: &LiveTypedInsertView<'_>,
    records: &[DecodedTypedInsertRecord],
) -> Result<(), EngineError> {
    for statement in input.statements {
        let record = records
            .get(statement.statement_ordinal as usize)
            .ok_or_else(|| error("sealed S2 FK statement record is absent"))?;
        let target = record.target_identity();
        if record.facts().statement_ordinal.as_u32() != statement.statement_ordinal
            || record.facts().typed_statement_digest != statement.typed_statement_digest
            || target.schema != input.table.schema
            || target.name != input.table.name
            || target.oid != input.table.display_oid
            || record.foreign_keys().len() == 0
        {
            return Err(error(
                "sealed S2 FK statement target differs from the live catalog",
            ));
        }
        // S2 preserves the statement-time target digest. A later stable-OID sequence rename in
        // this same S3 composition can change the final table's default-expression digest without
        // changing the target identity or this statement's FK closure. S3 authenticates that
        // final catalog transition; the checks below still bind every sealed FK to the final
        // immutable parent generation, so this does not admit a changed or omitted FK.
        for foreign_key in record.foreign_keys() {
            if !input
                .foreign_indexes
                .iter()
                .any(|generation| foreign_matches_generation(record, foreign_key, generation))
            {
                return Err(error(
                    "sealed S2 foreign key differs from the live parent catalog",
                ));
            }
        }
    }
    Ok(())
}

fn foreign_matches_generation(
    record: &DecodedTypedInsertRecord,
    foreign_key: crate::typed_insert_batch::DecodedForeignKeyFacts<'_>,
    generation: &LiveTypedInsertForeignIndexGeneration<'_>,
) -> bool {
    let Some(parent) = record
        .dependencies()
        .nth(foreign_key.parent_dependency_ordinal as usize)
    else {
        return false;
    };
    parent.schema == generation.parent.schema
        && parent.name == generation.parent.name
        && parent.oid == generation.parent.oid
        && parent.schema_digest == generation.parent_schema_digest
        && foreign_key.referenced_table_name == generation.parent.name
        && foreign_key.supporting_index.oid == generation.catalog.oid
        && foreign_key.supporting_index.name == generation.catalog.name
        && foreign_key.supporting_index.table_name == generation.parent.name
        && foreign_key.supporting_index.unique == generation.catalog.unique
        && foreign_key.supporting_index.primary_key == generation.catalog.primary_key
        && foreign_key.supporting_index.unique_constraint == generation.catalog.unique_constraint
        && foreign_key.supporting_index.key_count == 1
        && generation.catalog.key_columns.len() == 1
        && generation.catalog.key_columns[0] == foreign_key.referenced_column_name
        && generation.catalog.oid != 0
        && generation.parent.stable_table_id != 0
        && generation.parent_generation != 0
        && generation.parent_root != [0; 32]
        && generation.index_generation != 0
        && generation.index_root != [0; 32]
}

fn descriptor_ref(
    record: &DecodedTypedInsertRecord,
    foreign_key: crate::typed_insert_batch::DecodedForeignKeyFacts<'_>,
    descriptors: &[ForeignDescriptor<'_>],
) -> Option<u32> {
    descriptors
        .iter()
        .position(|descriptor| {
            foreign_matches_generation(record, foreign_key, descriptor.generation)
        })
        .and_then(|ordinal| u32::try_from(ordinal).ok())
}

fn descriptor<'a>(
    input: &LiveTypedInsertView<'_>,
    record: &DecodedTypedInsertRecord,
    foreign_key: crate::typed_insert_batch::DecodedForeignKeyFacts<'_>,
    generation: &'a LiveTypedInsertForeignIndexGeneration<'a>,
    index_ref: u32,
    next_key_ref: &mut u32,
) -> Result<ForeignDescriptor<'a>, EngineError> {
    let bindings = record
        .foreign_key_supporting_index_keys(foreign_key.raw_ordinal)?
        .collect::<Vec<_>>();
    if bindings.len() != 1 || generation.catalog.key_columns.as_slice() != [bindings[0].name] {
        return Err(error(
            "foreign-key S7 supporting index is not the exact one-column S2 binding",
        ));
    }
    let binding = bindings[0];
    let mut key = [0_u8; 112];
    put_u32(&mut key, 0, *next_key_ref);
    put_u32(&mut key, 4, index_ref);
    put_u32(&mut key, 8, 0);
    put_u32(&mut key, 12, binding.catalog_column_ordinal);
    put_u32(&mut key, 16, binding.column_id);
    put_u32(&mut key, 20, generation.parent.oid);
    put_i16(&mut key, 24, binding.attnum);
    let storage = crate::typed_insert_batch::typed_image_sql_storage(binding.ty);
    key[28..32].copy_from_slice(&storage);
    put_u32(&mut key, 32, binding.type_oid);
    put_i16(&mut key, 36, binding.type_size);
    put_digest(&mut key, 40, write001_identifier_digest(binding.name)?);
    let key_digest = v2_digest(
        b"gpu-db/write001/s7-index-key-column/v2",
        &[&key[..72], &[0; 32], &key[104..]],
    );
    put_digest(&mut key, 72, key_digest);
    let keys = vec![IndexedS7Key {
        raw: key,
        digest: key_digest,
        catalog_column_ordinal: binding.catalog_column_ordinal,
        stable_column_id: binding.column_id,
        attnum: binding.attnum,
        storage,
        declared_type_oid: binding.type_oid,
        signed_type_size: binding.type_size,
    }];
    *next_key_ref = next_key_ref
        .checked_add(1)
        .ok_or_else(|| error("foreign-key S7 key reference overflows"))?;

    let catalog = generation.catalog;
    let constraint_backed = catalog.primary_key || catalog.unique_constraint;
    let mut raw = [0_u8; 384];
    put_u32(&mut raw, 0, index_ref);
    put_u32(
        &mut raw,
        4,
        u32::from(catalog.unique)
            | (u32::from(catalog.primary_key) << 1)
            | (u32::from(catalog.unique_constraint) << 2),
    );
    put_u32(&mut raw, 8, ABSENT_U32);
    put_u32(&mut raw, 12, foreign_key.supporting_index.raw_ordinal);
    put_u64(&mut raw, 16, u64::from(catalog.oid));
    put_u32(&mut raw, 24, catalog.oid);
    put_u32(
        &mut raw,
        28,
        if constraint_backed { catalog.oid } else { 0 },
    );
    put_u64(
        &mut raw,
        32,
        if constraint_backed {
            u64::from(catalog.oid)
        } else {
            u64::MAX
        },
    );
    put_u32(&mut raw, 40, *next_key_ref - 1);
    put_u32(&mut raw, 44, 1);
    raw[48] = 1;
    put_u64(&mut raw, 64, generation.parent.stable_table_id);
    put_u64(&mut raw, 72, input.identity.catalog_epoch);
    put_digest(&mut raw, 80, generation.parent_schema_digest);
    put_digest(
        &mut raw,
        112,
        qualified_name_digest(&generation.parent.schema, &generation.parent.name)?,
    );
    let index_name = qualified_name_digest(&generation.parent.schema, &catalog.name)?;
    put_digest(&mut raw, 144, index_name);
    if constraint_backed {
        put_digest(&mut raw, 176, index_name);
    }
    put_digest(&mut raw, 208, generation.parent_root);
    put_digest(&mut raw, 240, generation.index_root);
    put_digest(&mut raw, 272, generation.index_root);
    put_u32(&mut raw, 336, generation.parent.oid);
    put_u64(&mut raw, 344, generation.parent_generation);
    put_u64(&mut raw, 352, generation.index_generation);
    put_u64(&mut raw, 360, generation.index_generation);
    let digest = index_descriptor_digest(&raw, &keys);
    put_digest(&mut raw, 304, digest);
    Ok(ForeignDescriptor {
        generation,
        raw,
        digest,
        keys,
        shared_initial_parent: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_foreign_effect(
    image: &crate::typed_insert_batch::DecodedTypedImage,
    foreign_key: crate::typed_insert_batch::DecodedForeignKeyFacts<'_>,
    index_ref: u32,
    keys: &[IndexedS7Key],
    descriptor_digest: [u8; 32],
    transition_ref: u32,
    image_row: u32,
    effect_ref_base: u32,
    component_ref_base: u32,
    value_offset_base: u64,
    components: &mut Vec<[u8; 128]>,
    values: &mut Vec<u8>,
    effects: &mut Vec<[u8; 192]>,
    effect_digests: &mut Vec<[u8; 32]>,
) -> Result<(u32, bool), EngineError> {
    if keys.len() != 1 {
        return Err(error("foreign-key S7 effect requires one parent key"));
    }
    let key = &keys[0];
    let effect_ref = effect_ref_base
        .checked_add(
            u32::try_from(effects.len()).map_err(|_| error("S7 effect count exceeds u32"))?,
        )
        .ok_or_else(|| error("S7 effect reference overflows"))?;
    let component_ref = component_ref_base
        .checked_add(
            u32::try_from(components.len()).map_err(|_| error("S7 component count exceeds u32"))?,
        )
        .ok_or_else(|| error("S7 component reference overflows"))?;
    let source_column = foreign_key.child_column.catalog_column_ordinal;
    let child = image
        .columns()
        .find(|column| column.catalog_column_ordinal == source_column)
        .ok_or_else(|| error("foreign-key S7 final-image child column is absent"))?;
    if child.stable_column_id != foreign_key.child_column.column_id
        || child.attnum != foreign_key.child_column.attnum
        || child.ty != foreign_key.child_column.ty
        || child.type_oid != foreign_key.child_column.type_oid
        || child.type_size != foreign_key.child_column.type_size
        || crate::typed_insert_batch::typed_image_sql_storage(child.ty) != key.storage
        || key.declared_type_oid != foreign_key.parent_column.type_oid
        || key.signed_type_size != foreign_key.parent_column.type_size
    {
        return Err(error(
            "foreign-key S7 final-image child metadata differs from its binding",
        ));
    }
    let row = usize::try_from(image_row)
        .map_err(|_| error("foreign-key S7 final-image row exceeds addressability"))?;
    let (is_null, bytes) =
        child.with_logical_cell_at(row, |is_null, bytes| (is_null, bytes.to_vec()))?;
    let valid = !is_null;
    let typed_value_digest = typed_key_value_digest_from_bytes(
        valid,
        &bytes,
        key.storage,
        key.declared_type_oid,
        key.signed_type_size,
    );
    let mut component = [0_u8; 128];
    put_u32(&mut component, 0, component_ref);
    put_u32(&mut component, 4, effect_ref);
    component[8] = 2;
    component[9] = u8::from(!valid);
    put_u32(&mut component, 16, key_column_ref(key));
    put_u32(&mut component, 20, source_column);
    put_u64(
        &mut component,
        24,
        value_offset_base
            .checked_add(
                u64::try_from(values.len()).map_err(|_| error("S7 key value arena exceeds u64"))?,
            )
            .ok_or_else(|| error("S7 key value offset overflows"))?,
    );
    put_u32(
        &mut component,
        32,
        u32::try_from(bytes.len()).map_err(|_| error("S7 key value exceeds u32"))?,
    );
    component[36..40].copy_from_slice(&key.storage);
    put_u32(&mut component, 40, key.declared_type_oid);
    put_i16(&mut component, 44, key.signed_type_size);
    put_digest(&mut component, 48, typed_value_digest);
    let component_digest = v2_digest(
        b"gpu-db/write001/s7-typed-key-component/v2",
        &[&component[..80], &[0; 32], &component[112..], &key.digest],
    );
    put_digest(&mut component, 80, component_digest);
    components.push(component);
    values.extend_from_slice(&bytes);

    let participates = valid;
    let effect_digest = key_effect_digest(
        effect_ref,
        3,
        transition_ref,
        index_ref,
        foreign_key.raw_ordinal,
        component_ref,
        &[component_digest],
        descriptor_digest,
        !valid,
        participates,
    );
    let mut effect = [0_u8; 192];
    put_u32(&mut effect, 0, effect_ref);
    effect[4] = 3;
    effect[5] = 3;
    put_u32(&mut effect, 8, transition_ref);
    put_u32(&mut effect, 12, index_ref);
    put_u32(&mut effect, 16, ABSENT_U32);
    put_u32(&mut effect, 20, ABSENT_U32);
    put_u32(&mut effect, 28, component_ref);
    put_u32(&mut effect, 32, 1);
    put_u32(&mut effect, 36, 1);
    effect[41] = 1;
    effect[42] = 1;
    effect[43] = u8::from(participates);
    effect[44] = u8::from(!valid);
    put_u32(&mut effect, 48, foreign_key.raw_ordinal);
    put_digest(
        &mut effect,
        96,
        typed_key_digest(effect_ref, &[component_digest]),
    );
    put_digest(&mut effect, 128, effect_digest);
    effects.push(effect);
    effect_digests.push(effect_digest);
    Ok((effect_ref, participates))
}

fn parent_generations<'a>(
    descriptors: &'a [ForeignDescriptor<'a>],
) -> Vec<&'a LiveTypedInsertForeignIndexGeneration<'a>> {
    let mut parents = descriptors
        .iter()
        .map(|descriptor| descriptor.generation)
        .collect::<Vec<_>>();
    parents.sort_unstable_by_key(|generation| generation.parent.stable_table_id);
    parents.dedup_by_key(|generation| generation.parent.stable_table_id);
    parents
}

fn parent_matches_source(
    generation: &LiveTypedInsertForeignIndexGeneration<'_>,
    source: crate::typed_insert_batch::DecodedDependencyFacts<'_>,
) -> bool {
    generation.parent.schema == source.schema
        && generation.parent.name == source.name
        && generation.parent.oid == source.oid
        && generation.parent_schema_digest == source.schema_digest
}

fn parent_dependency(
    reference: u32,
    target_table_ref: u32,
    input: &LiveTypedInsertView<'_>,
    generation: &LiveTypedInsertForeignIndexGeneration<'_>,
) -> Result<[u8; 224], EngineError> {
    let name_digest = qualified_name_digest(&generation.parent.schema, &generation.parent.name)?;
    let identity = v2_digest(
        b"gpu-db/write001/s7-table-object/v2",
        &[
            &[2],
            &generation.parent.stable_table_id.to_le_bytes(),
            &generation.parent.oid.to_le_bytes(),
            &input.identity.catalog_epoch.to_le_bytes(),
            &generation.parent_generation.to_le_bytes(),
            &generation.parent_schema_digest,
            &generation.parent_root,
            &name_digest,
        ],
    );
    let mut raw = [0_u8; 224];
    put_u32(&mut raw, 0, reference);
    raw[4] = 2;
    raw[5] = 2;
    put_u64(&mut raw, 8, generation.parent.stable_table_id);
    put_u32(&mut raw, 16, generation.parent.oid);
    put_u32(&mut raw, 20, target_table_ref);
    put_u64(&mut raw, 24, generation.parent_generation);
    put_u64(&mut raw, 32, input.identity.dependency_validation_floor);
    put_u32(&mut raw, 40, ABSENT_U32);
    put_u32(&mut raw, 44, ABSENT_U32);
    put_u64(&mut raw, 48, input.identity.catalog_epoch);
    put_digest(&mut raw, 64, generation.parent_schema_digest);
    put_digest(&mut raw, 96, generation.parent_root);
    put_digest(&mut raw, 128, name_digest);
    put_digest(&mut raw, 160, identity);
    let digest = v2_digest(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&raw[..192], &[0; 32], &[0; 32], &[0; 32]],
    );
    put_digest(&mut raw, 192, digest);
    Ok(raw)
}

#[allow(clippy::too_many_arguments)]
fn foreign_index_dependency(
    reference: u32,
    kind: u8,
    descriptor: &ForeignDescriptor<'_>,
    target_table_ref: u32,
    validation_floor: u64,
    key_effect_ref: u32,
    effect_digest: [u8; 32],
) -> [u8; 224] {
    let index = &descriptor.raw;
    let stable_index_id = u64::from_le_bytes(index[16..24].try_into().unwrap());
    let display_oid = u32::from_le_bytes(index[24..28].try_into().unwrap());
    // A shared descriptor belongs to an initial parent table in this exact S7 transaction.
    // Its FK proof therefore pins the parent's already-computed successor index rather than
    // its absent predecessor; the parent table token carries the matching successor generation.
    let base_generation = u64::from_le_bytes(
        index[if descriptor.shared_initial_parent {
            360
        } else {
            352
        }..if descriptor.shared_initial_parent {
            368
        } else {
            360
        }]
            .try_into()
            .unwrap(),
    );
    let catalog_epoch = u64::from_le_bytes(index[72..80].try_into().unwrap());
    let index_ref = u32::from_le_bytes(index[..4].try_into().unwrap());
    let live = key_effect_ref != ABSENT_U32;
    let mut raw = [0_u8; 224];
    put_u32(&mut raw, 0, reference);
    raw[4] = kind;
    raw[5] = 2;
    put_u16(&mut raw, 6, u16::from(live));
    put_u64(&mut raw, 8, stable_index_id);
    put_u32(&mut raw, 16, display_oid);
    put_u32(&mut raw, 20, target_table_ref);
    put_u64(&mut raw, 24, base_generation);
    put_u64(&mut raw, 32, validation_floor);
    put_u32(&mut raw, 40, key_effect_ref);
    put_u32(&mut raw, 44, index_ref);
    put_u64(&mut raw, 48, catalog_epoch);
    raw[64..96].copy_from_slice(&index[80..112]);
    raw[96..128].copy_from_slice(if descriptor.shared_initial_parent {
        &index[272..304]
    } else {
        &index[240..272]
    });
    raw[128..160].copy_from_slice(&index[144..176]);
    let identity = v2_digest(
        b"gpu-db/write001/s7-index-object/v2",
        &[
            &[kind],
            &stable_index_id.to_le_bytes(),
            &display_oid.to_le_bytes(),
            &catalog_epoch.to_le_bytes(),
            &base_generation.to_le_bytes(),
            &raw[64..96],
            &raw[96..128],
            &raw[128..160],
            &descriptor.digest,
            &effect_digest,
        ],
    );
    put_digest(&mut raw, 160, identity);
    let digest = v2_digest(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&raw[..192], &[0; 32], &descriptor.digest, &effect_digest],
    );
    put_digest(&mut raw, 192, digest);
    raw
}

fn key_column_ref(key: &IndexedS7Key) -> u32 {
    u32::from_le_bytes(key.raw[..4].try_into().expect("fixed key reference"))
}
