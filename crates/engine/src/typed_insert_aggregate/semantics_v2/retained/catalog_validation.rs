//! Exact pinned-catalog and durable allocator closure for the first retained typestate.
//!
//! This leaf has no generation-builder, WAL, recovery, apply, or publication authority.  It
//! validates borrowed external evidence only, then permits the caller to move the quarantined
//! graph into `GenerationPendingSemanticsV2`.

#[path = "catalog_validation/allocator.rs"]
mod allocator;
#[path = "catalog_validation/guards.rs"]
mod guards;

use super::{
    graph::{
        ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedIndexDescriptor, RetainedTable,
    },
    retained_error, validate_catalog_allocator_witness_identity, SemanticsV2BoundIdentity,
    SemanticsV2CatalogAllocatorWitness, SemanticsV2CatalogColumnWitness,
    SemanticsV2CatalogDomainWitness, SemanticsV2CatalogIndexKeyWitness,
    SemanticsV2CatalogIndexWitness, SemanticsV2CatalogSequenceWitness, SemanticsV2CatalogWitness,
};
use crate::{
    typed_insert_batch::{
        DecodedCatalogBindingFacts, DecodedCatalogColumnFacts, DecodedForeignKeyFacts,
        DecodedIndexFacts,
    },
    SqlType,
};
use sha2::{Digest, Sha256};

const TARGET_TABLE: u8 = 1;
const FOREIGN_PARENT_TABLE: u8 = 2;
const MAINTAINED_INDEX: u8 = 3;
const UNIQUE_KEY_GUARD: u8 = 4;
const FOREIGN_PARENT_INDEX: u8 = 5;
const FOREIGN_KEY_GUARD: u8 = 6;
const DOMAIN: u8 = 7;
const PUBLISHED_SEQUENCE: u8 = 8;
const NOT_NULL_GUARD: u8 = 9;
const CHECK_GUARD: u8 = 10;
const DOMAIN_CONSTRAINT_GUARD: u8 = 11;

const FOREIGN_PARENT_TABLE_ROLE: u16 = 4;

pub(super) fn validate(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    witness: &SemanticsV2CatalogAllocatorWitness<'_>,
) -> Result<(), crate::EngineError> {
    validate_catalog_allocator_witness_identity(identity, witness)?;
    validate_retained_header_identity(identity, graph)?;
    validate_catalog_order(&witness.catalog)?;
    allocator::validate_allocator_closure(identity, graph, witness.allocator_index.leases())?;
    allocator::validate_table_closure(identity, graph, &witness.catalog)?;
    validate_index_closure(identity, graph, &witness.catalog)?;
    validate_domain_closure(identity, graph, &witness.catalog)?;
    guards::validate_guard_closure(identity, graph, &witness.catalog)?;
    validate_sequence_closure(identity, graph, &witness.catalog)
}

fn validate_retained_header_identity(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
) -> Result<(), crate::EngineError> {
    let header = &graph.header;
    require(
        header.root_descriptor_version == 1
            && header.catalog_before_epoch == identity.catalog_epoch
            && header.catalog_after_epoch == identity.catalog_epoch
            && header.catalog_before_digest == identity.catalog_digest
            && header.catalog_after_digest == identity.catalog_digest
            && header.initial_database_root == identity.initial_database_root,
        "retained S7 header does not match the sealed catalog identity",
    )
}

fn validate_catalog_order(
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    allocator::validate_table_order(catalog.tables)?;
    validate_index_order(catalog.indexes)?;
    validate_domain_order(catalog.domains)?;
    guards::validate_guard_order(catalog.guards)?;
    validate_sequence_order(catalog.sequences)
}

fn validate_index_order(
    rows: &[SemanticsV2CatalogIndexWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    for row in rows {
        require_stable_identity(row.stable_index_id, row.display_oid, "catalog index")?;
        require_stable_identity(
            row.owner_stable_table_id,
            row.owner_display_oid,
            "catalog index owner",
        )?;
        require(
            valid_identifier(row.schema)
                && valid_identifier(row.name)
                && valid_identifier(row.owner_schema)
                && valid_identifier(row.owner_name)
                && previous.is_none_or(|prior| prior < row.stable_index_id),
            "catalog index identity/order is invalid",
        )?;
        previous = Some(row.stable_index_id);
        validate_index_key_order(row.key_columns)?;
    }
    Ok(())
}

fn validate_domain_order(
    rows: &[SemanticsV2CatalogDomainWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    for row in rows {
        require_stable_identity(row.stable_domain_id, row.display_oid, "catalog domain")?;
        require(
            valid_identifier(row.schema)
                && valid_identifier(row.name)
                && previous.is_none_or(|prior| prior < row.stable_domain_id),
            "catalog domain identity/order is invalid",
        )?;
        previous = Some(row.stable_domain_id);
        guards::validate_guard_list_order(
            row.constraints,
            DOMAIN_CONSTRAINT_GUARD,
            "domain constraint",
        )?;
    }
    Ok(())
}

fn validate_sequence_order(
    rows: &[SemanticsV2CatalogSequenceWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    for row in rows {
        require_stable_identity(row.stable_sequence_id, row.display_oid, "catalog sequence")?;
        require(
            valid_identifier(row.schema)
                && valid_identifier(row.name)
                && previous.is_none_or(|prior| prior < row.stable_sequence_id),
            "catalog sequence identity/order is invalid",
        )?;
        previous = Some(row.stable_sequence_id);
    }
    Ok(())
}

fn validate_index_closure(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    require(
        graph.indexes.len() == catalog.indexes.len(),
        "pinned catalog index count does not equal the exact S2/S7 index closure",
    )?;
    for descriptor in &graph.indexes {
        let mut matched = None;
        let mut count = 0_u32;
        for candidate in catalog.indexes {
            if candidate.stable_index_id == descriptor.stable_index_id
                && candidate.display_oid == descriptor.display_oid
            {
                count = count
                    .checked_add(1)
                    .ok_or_else(|| validation_error("catalog index match count overflows"))?;
                matched = Some(candidate);
            }
        }
        let Some(candidate) = matched else {
            return Err(validation_error(
                "S7 index descriptor is absent from the pinned catalog",
            ));
        };
        require(
            count == 1,
            "S7 index descriptor is ambiguous in the pinned catalog",
        )?;
        validate_index_descriptor(identity, graph, descriptor, candidate)?;
        require(
            catalog_index_has_s2_source(graph, candidate)?,
            "S7 index descriptor has no exact S2 target/FK supporting-index source",
        )?;
    }
    for record in &graph.records {
        for source in record.indexes() {
            validate_s2_index_source(catalog.indexes, record, source)?;
        }
        for foreign_key in record.foreign_keys() {
            validate_s2_fk_supporting_index_source(catalog.indexes, record, foreign_key)?;
        }
    }
    Ok(())
}

fn validate_index_descriptor(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    descriptor: &RetainedIndexDescriptor,
    catalog: &SemanticsV2CatalogIndexWitness<'_>,
) -> Result<(), crate::EngineError> {
    require(
        catalog.owner_stable_table_id == descriptor.owner_stable_table_id
            && catalog.owner_display_oid == descriptor.owner_display_oid
            && catalog.raw_catalog_ordinal == descriptor.raw_catalog_ordinal
            && catalog.index_flags == descriptor.flags
            && catalog.null_equality_policy == descriptor.null_equality_policy
            && catalog.catalog_epoch == identity.catalog_epoch
            && descriptor.catalog_epoch == identity.catalog_epoch
            && catalog.base_generation == descriptor.base_index_generation
            && catalog.base_root == descriptor.base_index_root
            && catalog.schema_digest == descriptor.owner_schema_digest
            && catalog.owner_schema == catalog.schema
            && qualified_name_digest(catalog.schema, catalog.name)? == descriptor.index_name_digest
            && qualified_name_digest(catalog.owner_schema, catalog.owner_name)?
                == descriptor.owner_name_digest
            && descriptor.owner_table_base_root != [0; 32]
            && descriptor.base_index_root != [0; 32]
            && descriptor.descriptor_digest != [0; 32]
            && descriptor.null_equality_policy == 1,
        "pinned catalog index does not match its S7 descriptor identity",
    )?;
    validate_index_constraint_identity(descriptor, catalog)?;
    let owner = catalog_table_by_identity(
        graph,
        catalog.owner_stable_table_id,
        catalog.owner_display_oid,
    );
    if let Some(owner) = owner {
        require(
            owner.initial_table_root == descriptor.owner_table_base_root
                && owner.data_generation_before == descriptor.owner_data_generation,
            "target-owned index descriptor does not match its S7 table base identity",
        )?;
    } else {
        validate_parent_only_index_owner(identity, graph, descriptor, catalog)?;
    }
    validate_descriptor_keys(graph, descriptor, catalog.key_columns)
}

fn validate_parent_only_index_owner(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    descriptor: &RetainedIndexDescriptor,
    catalog: &SemanticsV2CatalogIndexWitness<'_>,
) -> Result<(), crate::EngineError> {
    let mut source_count = 0_u32;
    for record in &graph.records {
        let statement_ordinal = record.facts().statement_ordinal.as_u32();
        for foreign_key in record.foreign_keys() {
            if !catalog_index_matches_s2(catalog, foreign_key.supporting_index) {
                continue;
            }
            let parent = record
                .dependencies()
                .nth(
                    usize::try_from(foreign_key.parent_dependency_ordinal).map_err(|_| {
                        validation_error("FK parent dependency ordinal exceeds addressability")
                    })?,
                )
                .ok_or_else(|| validation_error("FK parent dependency is absent"))?;
            require(
                parent.oid == catalog.owner_display_oid
                    && parent.schema == catalog.owner_schema
                    && parent.name == catalog.owner_name,
                "FK supporting index owner differs from its exact S2 parent dependency",
            )?;
            let token = graph.dependencies.iter().find(|token| {
                token.kind == FOREIGN_PARENT_TABLE
                    && token.stable_object_id == catalog.owner_stable_table_id
                    && token.display_oid == catalog.owner_display_oid
                    && token.catalog_epoch == identity.catalog_epoch
                    && token.base_generation == descriptor.owner_data_generation
                    && token.base_root == descriptor.owner_table_base_root
                    && token.schema_digest == descriptor.owner_schema_digest
                    && qualified_name_digest(catalog.owner_schema, catalog.owner_name)
                        .is_ok_and(|digest| token.name_digest == digest)
                    && graph.dependency_uses.iter().any(|usage| {
                        usage.dependency_ref == token.dependency_ref
                            && usage.role == FOREIGN_PARENT_TABLE_ROLE
                            && usage.statement_ordinal == statement_ordinal
                            && usage.source_ordinal == foreign_key.parent_dependency_ordinal
                    })
            });
            require(
                token.is_some(),
                "parent-only index is not joined to the same S2 FK parent-table token",
            )?;
            source_count = source_count
                .checked_add(1)
                .ok_or_else(|| validation_error("parent-only index source count overflows"))?;
        }
    }
    require(
        source_count != 0,
        "parent-only index descriptor has no exact S2 FK supporting-index source",
    )
}

fn validate_index_constraint_identity(
    descriptor: &RetainedIndexDescriptor,
    catalog: &SemanticsV2CatalogIndexWitness<'_>,
) -> Result<(), crate::EngineError> {
    let backed_by_constraint = catalog.index_flags & 0b110 != 0;
    if backed_by_constraint {
        require(
            catalog.constraint_stable_id != 0
                && catalog.constraint_stable_id != u64::MAX
                && catalog.constraint_display_oid != 0
                && catalog.constraint_display_oid <= 0x7fff_ffff
                && valid_identifier(catalog.constraint_schema)
                && valid_identifier(catalog.constraint_name)
                && catalog.constraint_stable_id == descriptor.stable_constraint_id
                && catalog.constraint_display_oid == descriptor.constraint_display_oid
                && qualified_name_digest(catalog.constraint_schema, catalog.constraint_name)?
                    == descriptor.constraint_name_digest,
            "constraint-backed index does not close its catalog constraint identity",
        )
    } else {
        require(
            catalog.constraint_stable_id == 0
                && catalog.constraint_display_oid == 0
                && catalog.constraint_schema.is_empty()
                && catalog.constraint_name.is_empty()
                && descriptor.stable_constraint_id == u64::MAX
                && descriptor.constraint_display_oid == 0
                && descriptor.constraint_name_digest == [0; 32],
            "non-constraint index has a fabricated catalog constraint identity",
        )
    }
}

fn catalog_table_by_identity(
    graph: &ReservedSemanticsV2Graph,
    stable_id: u64,
    display_oid: u32,
) -> Option<&RetainedTable> {
    graph
        .tables
        .iter()
        .find(|table| table.stable_table_id == stable_id && table.display_oid == display_oid)
}

fn validate_descriptor_keys(
    graph: &ReservedSemanticsV2Graph,
    descriptor: &RetainedIndexDescriptor,
    catalog: &[SemanticsV2CatalogIndexKeyWitness<'_>],
) -> Result<(), crate::EngineError> {
    let start = usize::try_from(descriptor.key_start)
        .map_err(|_| validation_error("S7 index key start exceeds addressability"))?;
    let count = usize::try_from(descriptor.key_count)
        .map_err(|_| validation_error("S7 index key count exceeds addressability"))?;
    let end = start
        .checked_add(count)
        .ok_or_else(|| validation_error("S7 index key range overflows"))?;
    let keys = graph
        .index_key_columns
        .get(start..end)
        .ok_or_else(|| validation_error("S7 index key range is absent"))?;
    require(
        keys.len() == catalog.len(),
        "pinned catalog index key count differs from its S7 descriptor",
    )?;
    for (key, catalog_key) in keys.iter().zip(catalog) {
        require(
            catalog_key.key_ordinal == key.key_ordinal
                && catalog_key.owner_catalog_column_ordinal == key.owner_catalog_column_ordinal
                && catalog_key.stable_column_id == key.stable_column_id
                && catalog_key.attnum == key.attnum
                && catalog_key.storage == key.storage
                && catalog_key.declared_type_oid == key.declared_type_oid
                && catalog_key.signed_type_size == key.signed_type_size
                && catalog_key.column_name_digest == key.column_name_digest,
            "pinned catalog index key differs from its S7 descriptor key",
        )?;
    }
    Ok(())
}

fn catalog_index_has_s2_source(
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogIndexWitness<'_>,
) -> Result<bool, crate::EngineError> {
    for record in &graph.records {
        for source in record.indexes() {
            if catalog_index_matches_s2(catalog, source) {
                return Ok(true);
            }
        }
        for foreign_key in record.foreign_keys() {
            if catalog_index_matches_s2(catalog, foreign_key.supporting_index) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn validate_s2_index_source(
    catalog: &[SemanticsV2CatalogIndexWitness<'_>],
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    source: DecodedIndexFacts<'_>,
) -> Result<(), crate::EngineError> {
    let count = catalog
        .iter()
        .filter(|candidate| catalog_index_matches_s2(candidate, source))
        .count();
    require(
        count == 1,
        "S2 index source does not select exactly one pinned catalog index",
    )?;
    let catalog = catalog
        .iter()
        .find(|candidate| catalog_index_matches_s2(candidate, source))
        .expect("count checked");
    let keys = record.index_key_columns(source.raw_ordinal)?;
    require(
        keys.len() == catalog.key_columns.len(),
        "S2 index key count does not close the pinned catalog index keys",
    )?;
    for (source_key, catalog_key) in keys.zip(catalog.key_columns) {
        require(
            catalog_index_key_matches_binding(catalog_key, source_key),
            "S2 index key differs from its pinned catalog key",
        )?;
    }
    Ok(())
}

fn validate_s2_fk_supporting_index_source(
    catalog: &[SemanticsV2CatalogIndexWitness<'_>],
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    foreign_key: DecodedForeignKeyFacts<'_>,
) -> Result<(), crate::EngineError> {
    let source = foreign_key.supporting_index;
    let count = catalog
        .iter()
        .filter(|candidate| catalog_index_matches_s2(candidate, source))
        .count();
    require(
        count == 1,
        "S2 FK supporting index does not select exactly one pinned catalog index",
    )?;
    let catalog = catalog
        .iter()
        .find(|candidate| catalog_index_matches_s2(candidate, source))
        .expect("count checked");
    let keys = record.foreign_key_supporting_index_keys(foreign_key.raw_ordinal)?;
    require(
        keys.len() == catalog.key_columns.len(),
        "S2 FK supporting-key count does not close the pinned catalog index keys",
    )?;
    for (source_key, catalog_key) in keys.zip(catalog.key_columns) {
        require(
            catalog_index_key_matches_binding(catalog_key, source_key),
            "S2 FK supporting key differs from its pinned catalog key",
        )?;
    }
    Ok(())
}

fn catalog_index_matches_s2(
    catalog: &SemanticsV2CatalogIndexWitness<'_>,
    source: DecodedIndexFacts<'_>,
) -> bool {
    catalog.display_oid == source.oid
        && catalog.raw_catalog_ordinal == source.raw_ordinal
        && catalog.name == source.name
        && catalog.owner_name == source.table_name
        && (catalog.index_flags & 1 != 0) == source.unique
        && (catalog.index_flags & 2 != 0) == source.primary_key
        && (catalog.index_flags & 4 != 0) == source.unique_constraint
        && usize::try_from(source.key_count)
            .ok()
            .is_some_and(|count| count == catalog.key_columns.len())
}

fn validate_domain_closure(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for domain in catalog.domains {
        require(
            catalog_domain_has_s2_source(graph, domain),
            "pinned catalog contains a domain outside the exact S2 closure",
        )?;
        let token_count = graph
            .dependencies
            .iter()
            .filter(|dependency| domain_matches_dependency(identity, domain, dependency))
            .count();
        require(
            token_count != 0,
            "pinned catalog domain has no matching S7 dependency token",
        )?;
    }
    for dependency in graph
        .dependencies
        .iter()
        .filter(|dependency| dependency.kind == DOMAIN)
    {
        let count = catalog
            .domains
            .iter()
            .filter(|domain| domain_matches_dependency(identity, domain, dependency))
            .count();
        require(
            count == 1,
            "S7 domain dependency does not select exactly one pinned catalog domain",
        )?;
    }
    Ok(())
}

fn catalog_domain_has_s2_source(
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogDomainWitness<'_>,
) -> bool {
    graph
        .records
        .iter()
        .flat_map(|record| record.domains())
        .any(|source| {
            source.oid == catalog.display_oid
                && source.schema == catalog.schema
                && source.name == catalog.name
                && storage_bytes(source.base_type) == catalog.storage
        })
}

fn domain_matches_dependency(
    identity: SemanticsV2BoundIdentity,
    domain: &SemanticsV2CatalogDomainWitness<'_>,
    dependency: &RetainedDependencyToken,
) -> bool {
    let storage_shape = domain_shape_digest(
        domain.storage,
        domain.declared_type_oid,
        domain.signed_type_size,
    );
    dependency.kind == DOMAIN
        && dependency.stable_object_id == domain.stable_domain_id
        && dependency.display_oid == domain.display_oid
        && dependency.catalog_epoch == identity.catalog_epoch
        && dependency.base_generation == domain.catalog_generation
        && dependency.base_root == [0; 32]
        && domain.storage_shape_digest == storage_shape
        && dependency.schema_digest == domain.storage_shape_digest
        && qualified_name_digest(domain.schema, domain.name)
            .is_ok_and(|digest| dependency.name_digest == digest)
}

fn validate_sequence_closure(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for sequence in catalog.sequences {
        require(
            sequence_has_s2_source(graph, sequence),
            "pinned catalog sequence is outside the exact S2/S5 closure",
        )?;
    }
    for effect in &graph.sequence_effects {
        validate_published_sequence_effect(identity, graph, catalog, effect)?;
    }
    for dependency in graph
        .dependencies
        .iter()
        .filter(|dependency| dependency.kind == PUBLISHED_SEQUENCE)
    {
        let count = graph
            .sequence_effects
            .iter()
            .filter(|effect| dependency_matches_sequence_effect(dependency, effect))
            .count();
        require(
            count == 1,
            "S7 published sequence token does not select exactly one retained S5 effect",
        )?;
    }
    Ok(())
}

fn sequence_has_s2_source(
    graph: &ReservedSemanticsV2Graph,
    sequence: &SemanticsV2CatalogSequenceWitness<'_>,
) -> bool {
    graph
        .records
        .iter()
        .flat_map(|record| record.sequence_bindings())
        .any(|source| {
            source.request.sequence_oid == sequence.display_oid
                && source.effective_name == sequence.name
                && source.descriptor_digest == sequence.descriptor_digest
        })
}

fn validate_published_sequence_effect(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
    effect: &super::graph::RetainedSequenceEffect,
) -> Result<(), crate::EngineError> {
    let sequence = catalog
        .sequences
        .iter()
        .find(|sequence| sequence.display_oid == effect.reference.sequence_oid)
        .ok_or_else(|| validation_error("S5 sequence effect is absent from the pinned catalog"))?;
    require(
        effect.reference.parent_txn_id == identity.stable_transaction_id
            && effect.reference.transition_txn_id != identity.stable_transaction_id
            && effect.body_digest != [0; 32],
        "S5 sequence effect has an invalid durable transaction/body identity",
    )?;
    require(
        sequence_has_exact_s2_effect_source(graph, sequence, effect)?,
        "S5 sequence effect does not match its exact S2 published request",
    )?;
    let matches = graph
        .dependencies
        .iter()
        .filter(|dependency| {
            sequence_dependency_matches_effect(identity, sequence, dependency, effect)
        })
        .count();
    require(
        matches == 1,
        "S5 sequence effect does not select exactly one published-sequence token",
    )?;
    let dependency = graph
        .dependencies
        .iter()
        .find(|dependency| {
            sequence_dependency_matches_effect(identity, sequence, dependency, effect)
        })
        .expect("count checked");
    require(
        graph
            .dependency_uses
            .iter()
            .filter(|usage| {
                usage.dependency_ref == dependency.dependency_ref
                    && usage.role == u16::from(PUBLISHED_SEQUENCE)
                    && usage.statement_ordinal == effect.statement_ordinal
                    && usage.source_ordinal == effect.effect_ordinal
            })
            .count()
            == 1
            && graph
                .dependency_uses
                .iter()
                .filter(|usage| usage.dependency_ref == dependency.dependency_ref)
                .count()
                == 1,
        "published-sequence token/use does not biject its exact retained S5 effect",
    )
}

fn dependency_matches_sequence_effect(
    dependency: &RetainedDependencyToken,
    effect: &super::graph::RetainedSequenceEffect,
) -> bool {
    dependency.kind == PUBLISHED_SEQUENCE
        && dependency.display_oid == effect.reference.sequence_oid
        && dependency.base_generation == effect.reference.transition_txn_id
        && dependency.base_root == effect.body_digest
}

fn sequence_dependency_matches_effect(
    identity: SemanticsV2BoundIdentity,
    sequence: &SemanticsV2CatalogSequenceWitness<'_>,
    dependency: &RetainedDependencyToken,
    effect: &super::graph::RetainedSequenceEffect,
) -> bool {
    if !dependency_matches_sequence_effect(dependency, effect)
        || dependency.stable_object_id != sequence.stable_sequence_id
        || dependency.catalog_epoch != identity.catalog_epoch
        || dependency.schema_digest != [0; 32]
        || qualified_name_digest(sequence.schema, sequence.name)
            .map_or(true, |digest| dependency.name_digest != digest)
    {
        return false;
    }
    published_sequence_identity_digest(sequence, identity.catalog_epoch, effect)
        .is_ok_and(|digest| dependency.identity_digest == digest)
}

fn sequence_has_exact_s2_effect_source(
    graph: &ReservedSemanticsV2Graph,
    sequence: &SemanticsV2CatalogSequenceWitness<'_>,
    effect: &super::graph::RetainedSequenceEffect,
) -> Result<bool, crate::EngineError> {
    let record = graph
        .records
        .iter()
        .find(|record| record.facts().statement_ordinal.as_u32() == effect.statement_ordinal)
        .ok_or_else(|| validation_error("S5 effect source record is absent"))?;
    let source = record
        .sequence_effects()
        .find(|source| source.request.effect_ordinal == effect.effect_ordinal)
        .ok_or_else(|| validation_error("S5 effect has no S2 sequence source"))?;
    let binding = record
        .sequence_bindings()
        .find(|binding| binding.effect_ordinal == effect.effect_ordinal)
        .ok_or_else(|| validation_error("S5 effect has no S2 sequence binding"))?;
    let disposition = graph
        .dispositions
        .get(
            usize::try_from(effect.disposition_ref)
                .map_err(|_| validation_error("S5 disposition reference exceeds addressability"))?,
        )
        .ok_or_else(|| validation_error("S5 disposition source is absent"))?;
    let published = match source.kind {
        crate::typed_insert_batch::DecodedSequenceEffectKindFacts::Published {
            transition_txn_id,
            input_digest,
            returned_value,
        } => {
            transition_txn_id == effect.reference.transition_txn_id
                && input_digest == effect.reference.input_digest
                && returned_value == effect.reference.returned_value
        }
        crate::typed_insert_batch::DecodedSequenceEffectKindFacts::Private { .. } => false,
    };
    Ok(published
        && effect.reference.parent_txn_id != 0
        && effect.reference.default_expression
        && effect.reference.statement_ordinal == effect.statement_ordinal
        && effect.reference.expression_ordinal == source.request.expression_ordinal
        && effect.reference.sequence_oid == source.request.sequence_oid
        && effect.reference.table_oid == source.request.target_table_oid
        && effect.reference.column_id == source.request.column_id
        && effect.reference.row_id == disposition.stable_row_id
        && disposition.statement_ordinal == effect.statement_ordinal
        && binding.request.sequence_oid == sequence.display_oid
        && binding.effective_name == sequence.name
        && binding.descriptor_digest == sequence.descriptor_digest
        && source.request == binding.request)
}

fn published_sequence_identity_digest(
    sequence: &SemanticsV2CatalogSequenceWitness<'_>,
    catalog_epoch: u64,
    effect: &super::graph::RetainedSequenceEffect,
) -> Result<[u8; 32], crate::EngineError> {
    let name_digest = qualified_name_digest(sequence.schema, sequence.name)?;
    let mut reference = [0_u8; crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES];
    crate::encode_sequence_value_reference_into_exact(&effect.reference, &mut reference)?;
    Ok(v2_digest(
        b"gpu-db/write001/s7-published-sequence/v2",
        &[
            &sequence.stable_sequence_id.to_le_bytes(),
            &sequence.display_oid.to_le_bytes(),
            &catalog_epoch.to_le_bytes(),
            &effect.reference.transition_txn_id.to_le_bytes(),
            &name_digest,
            &effect.body_digest,
            &reference,
        ],
    ))
}

fn catalog_column_matches_binding(
    catalog: &SemanticsV2CatalogColumnWitness<'_>,
    binding: DecodedCatalogBindingFacts<'_>,
) -> bool {
    catalog.catalog_column_ordinal == binding.catalog_column_ordinal
        && catalog.stable_column_id == binding.column_id
        && catalog.attnum == binding.attnum
        && catalog.name == binding.name
        && catalog.storage == storage_bytes(binding.ty)
        && catalog.declared_type_oid == binding.type_oid
        && catalog.signed_type_size == binding.type_size
}

fn catalog_column_matches_source(
    catalog: &SemanticsV2CatalogColumnWitness<'_>,
    source: DecodedCatalogColumnFacts<'_>,
) -> bool {
    catalog.catalog_column_ordinal == source.catalog_column_ordinal
        && catalog.stable_column_id == source.column_id
        && catalog.attnum == source.attnum
        && catalog.name == source.name
        && catalog.storage == storage_bytes(source.ty)
        && catalog.declared_type_oid == source.type_oid
        && catalog.signed_type_size == source.type_size
}

fn catalog_index_key_matches_binding(
    catalog: &SemanticsV2CatalogIndexKeyWitness<'_>,
    binding: DecodedCatalogBindingFacts<'_>,
) -> bool {
    catalog.owner_catalog_column_ordinal == binding.catalog_column_ordinal
        && catalog.stable_column_id == binding.column_id
        && catalog.attnum == binding.attnum
        && catalog.name == binding.name
        && catalog.storage == storage_bytes(binding.ty)
        && catalog.declared_type_oid == binding.type_oid
        && catalog.signed_type_size == binding.type_size
        && identifier_digest(binding.name).is_ok_and(|digest| catalog.column_name_digest == digest)
}

fn storage_bytes(ty: SqlType) -> [u8; 4] {
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

fn domain_shape_digest(
    storage: [u8; 4],
    declared_type_oid: u32,
    signed_type_size: i16,
) -> [u8; 32] {
    v2_digest(
        b"gpu-db/write001/s7-domain-shape/v2",
        &[
            &storage,
            &declared_type_oid.to_le_bytes(),
            &signed_type_size.to_le_bytes(),
        ],
    )
}

fn identifier_digest(value: &str) -> Result<[u8; 32], crate::EngineError> {
    let length = u32::try_from(value.len())
        .map_err(|_| validation_error("identifier length exceeds the canonical domain"))?;
    Ok(v2_digest(
        b"gpu-db/write001/s7-identifier/v2",
        &[&length.to_le_bytes(), value.as_bytes()],
    ))
}

fn qualified_name_digest(schema: &str, name: &str) -> Result<[u8; 32], crate::EngineError> {
    let schema_length = u32::try_from(schema.len())
        .map_err(|_| validation_error("schema identifier length exceeds the canonical domain"))?;
    let name_length = u32::try_from(name.len())
        .map_err(|_| validation_error("object identifier length exceeds the canonical domain"))?;
    Ok(v2_digest(
        b"gpu-db/write001/s7-qualified-name/v2",
        &[
            &schema_length.to_le_bytes(),
            schema.as_bytes(),
            &name_length.to_le_bytes(),
            name.as_bytes(),
        ],
    ))
}

fn v2_digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(
        u64::try_from(domain.len())
            .expect("domain length fits u64")
            .to_le_bytes(),
    );
    digest.update(domain);
    for field in fields {
        digest.update(field);
    }
    digest.finalize().into()
}

fn validate_index_key_order(
    rows: &[SemanticsV2CatalogIndexKeyWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut expected = 0_u32;
    for row in rows {
        require(
            row.key_ordinal == expected
                && row.stable_column_id != 0
                && row.attnum != 0
                && valid_identifier(row.name)
                && row.column_name_digest != [0; 32],
            "catalog index keys are not dense and resolved",
        )?;
        expected = expected
            .checked_add(1)
            .ok_or_else(|| validation_error("catalog index key ordinal overflows"))?;
    }
    Ok(())
}

fn require_stable_identity(
    stable: u64,
    display: u32,
    owner: &str,
) -> Result<(), crate::EngineError> {
    require(
        stable != 0 && stable != u64::MAX && display != 0 && display <= 0x7fff_ffff,
        owner,
    )
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty() && !value.as_bytes().contains(&0)
}

fn require(condition: bool, message: &str) -> Result<(), crate::EngineError> {
    condition
        .then_some(())
        .ok_or_else(|| validation_error(message))
}

fn validation_error(message: &str) -> crate::EngineError {
    retained_error(&format!("catalog/allocator validation: {message}"))
}
