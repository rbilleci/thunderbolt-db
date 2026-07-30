//! Exact pinned-catalog and durable allocator closure for the first retained typestate.
//!
//! This leaf has no generation-builder, WAL, recovery, apply, or publication authority.  It
//! validates borrowed external evidence only, then permits the caller to move the quarantined
//! graph into `GenerationPendingSemanticsV2`.

use super::{
    graph::{
        ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedIndexDescriptor, RetainedTable,
    },
    retained_error, validate_catalog_allocator_witness_identity, SemanticsV2BoundIdentity,
    SemanticsV2CatalogAllocatorWitness, SemanticsV2CatalogColumnWitness,
    SemanticsV2CatalogDomainWitness, SemanticsV2CatalogForeignKeyWitness,
    SemanticsV2CatalogGuardWitness, SemanticsV2CatalogIndexKeyWitness,
    SemanticsV2CatalogIndexWitness, SemanticsV2CatalogSequenceWitness,
    SemanticsV2CatalogTableWitness, SemanticsV2CatalogWitness, SemanticsV2RowAllocatorLeaseWitness,
};
use crate::{
    typed_insert_batch::{
        DecodedCatalogBindingFacts, DecodedCatalogColumnFacts, DecodedDependencyFacts,
        DecodedForeignKeyFacts, DecodedIndexFacts,
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
    validate_allocator_closure(identity, graph, witness.allocator_index.leases())?;
    validate_table_closure(identity, graph, &witness.catalog)?;
    validate_index_closure(identity, graph, &witness.catalog)?;
    validate_domain_closure(identity, graph, &witness.catalog)?;
    validate_guard_closure(identity, graph, &witness.catalog)?;
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

/// The durable allocator index has already established marker provenance.  This codec boundary
/// only closes each proven lease against exactly one retained S7 table and its contained row
/// interval; it never tries to reconstruct allocator state from the transaction bytes.
fn validate_allocator_closure(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    leases: &[SemanticsV2RowAllocatorLeaseWitness],
) -> Result<(), crate::EngineError> {
    require(
        leases.len() == graph.tables.len(),
        "durable allocator proof does not have exactly one lease per S7 table",
    )?;
    let mut prior_table_id = None;
    for (table, lease) in graph.tables.iter().zip(leases) {
        require(
            prior_table_id.is_none_or(|prior| prior < table.stable_table_id)
                && lease.database_id == identity.database_id
                && lease.allocator_kind == 1
                && lease.stable_allocator_id == table.stable_table_id
                && lease.marker_commit_sequence < identity.commit_sequence
                && lease.marker_system_transaction_id != identity.stable_transaction_id
                && lease.lease_start <= table.row_allocator_before
                && table.row_allocator_high_water <= lease.lease_end,
            "durable allocator lease order, marker precedence, or table interval is invalid",
        )?;
        prior_table_id = Some(table.stable_table_id);
    }
    Ok(())
}

fn validate_catalog_order(
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    validate_table_order(catalog.tables)?;
    validate_index_order(catalog.indexes)?;
    validate_domain_order(catalog.domains)?;
    validate_guard_order(catalog.guards)?;
    validate_sequence_order(catalog.sequences)
}

fn validate_table_order(
    rows: &[SemanticsV2CatalogTableWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    for row in rows {
        require_stable_identity(row.stable_table_id, row.display_oid, "catalog table")?;
        require(
            valid_identifier(row.schema) && valid_identifier(row.name),
            "catalog table name is not a resolved identifier",
        )?;
        require(
            previous.is_none_or(|prior| prior < row.stable_table_id),
            "catalog tables are not in strict stable-ID order",
        )?;
        previous = Some(row.stable_table_id);
        validate_column_order(row.catalog_columns)?;
        validate_guard_list_order(row.not_null_guards, NOT_NULL_GUARD, "NOT NULL")?;
        validate_guard_list_order(row.check_guards, CHECK_GUARD, "CHECK")?;
        validate_foreign_key_order(row.foreign_keys)?;
    }
    Ok(())
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
        validate_guard_list_order(
            row.constraints,
            DOMAIN_CONSTRAINT_GUARD,
            "domain constraint",
        )?;
    }
    Ok(())
}

fn validate_guard_order(
    rows: &[SemanticsV2CatalogGuardWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    for row in rows {
        require(
            row.stable_guard_id != 0
                && row.stable_guard_id != u64::MAX
                && matches!(
                    row.kind,
                    NOT_NULL_GUARD | CHECK_GUARD | DOMAIN_CONSTRAINT_GUARD
                )
                && previous.is_none_or(|prior| prior < row.stable_guard_id),
            "catalog guards are not in strict stable-ID order",
        )?;
        if row.synthesized_not_null {
            require(
                row.display_oid == 0
                    && matches!(row.kind, NOT_NULL_GUARD | DOMAIN_CONSTRAINT_GUARD),
                "only synthesized NOT NULL guards may omit a display OID",
            )?;
        } else {
            require(
                row.display_oid != 0
                    && row.display_oid <= 0x7fff_ffff
                    && valid_identifier(row.schema)
                    && valid_identifier(row.name),
                "named catalog guard identity is invalid",
            )?;
        }
        previous = Some(row.stable_guard_id);
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

fn validate_table_closure(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for table in &graph.tables {
        let mut matched = None;
        let mut count = 0_u32;
        for candidate in catalog.tables {
            if candidate.stable_table_id == table.stable_table_id
                && candidate.display_oid == table.display_oid
            {
                count = count.checked_add(1).ok_or_else(|| {
                    validation_error("catalog target-table match count overflows")
                })?;
                matched = Some(candidate);
            }
        }
        let Some(candidate) = matched else {
            return Err(validation_error(
                "S7 target table is absent from the pinned catalog",
            ));
        };
        require(
            count == 1,
            "S7 target table is ambiguous in the pinned catalog",
        )?;
        validate_target_table(identity, graph, catalog, table, candidate)?;
    }

    for dependency in &graph.dependencies {
        if matches!(dependency.kind, TARGET_TABLE | FOREIGN_PARENT_TABLE) {
            let matches = catalog
                .tables
                .iter()
                .filter(|table| {
                    table.stable_table_id == dependency.stable_object_id
                        && table.display_oid == dependency.display_oid
                })
                .count();
            require(
                matches == 1,
                "table dependency does not select exactly one pinned catalog table",
            )?;
            if dependency.kind == FOREIGN_PARENT_TABLE {
                let table = catalog
                    .tables
                    .iter()
                    .find(|table| {
                        table.stable_table_id == dependency.stable_object_id
                            && table.display_oid == dependency.display_oid
                    })
                    .expect("count checked");
                validate_foreign_parent_table(identity, graph, dependency, table)?;
            }
        }
    }

    for table in catalog.tables {
        let is_target = graph.tables.iter().any(|retained| {
            retained.stable_table_id == table.stable_table_id
                && retained.display_oid == table.display_oid
        });
        let is_parent = graph.dependencies.iter().any(|dependency| {
            dependency.kind == FOREIGN_PARENT_TABLE
                && dependency.stable_object_id == table.stable_table_id
                && dependency.display_oid == table.display_oid
        });
        require(
            is_target || is_parent,
            "pinned catalog contains a table outside the exact S2 dependency closure",
        )?;
        if is_parent {
            validate_parent_table_columns(graph, table)?;
        }
    }
    Ok(())
}

fn validate_target_table(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    all_catalog: &SemanticsV2CatalogWitness<'_>,
    retained: &RetainedTable,
    catalog: &SemanticsV2CatalogTableWitness<'_>,
) -> Result<(), crate::EngineError> {
    require(
        catalog.data_generation == retained.data_generation_before
            && catalog.data_root == retained.initial_table_root
            && catalog.schema_digest == retained.schema_digest
            && retained.catalog_epoch == identity.catalog_epoch
            && retained.data_generation_before != 0
            && retained.initial_table_root != [0; 32]
            && usize::try_from(retained.catalog_column_count)
                .ok()
                .is_some_and(|count| count == catalog.catalog_columns.len()),
        "pinned target table does not match its S7 initial identity",
    )?;

    let mut resolution_count = 0_u32;
    for resolution in graph
        .resolutions
        .iter()
        .filter(|resolution| resolution.table_ref == retained.table_ref)
    {
        resolution_count = resolution_count
            .checked_add(1)
            .ok_or_else(|| validation_error("target-table resolution count overflows"))?;
        let record =
            graph
                .records
                .get(usize::try_from(resolution.record_ref).map_err(|_| {
                    validation_error("S7 record reference exceeds host addressability")
                })?)
                .ok_or_else(|| validation_error("S7 target-table record is absent"))?;
        let target = record.target_identity();
        require(
            target.oid == catalog.display_oid
                && target.schema == catalog.schema
                && target.name == catalog.name
                && target.schema_digest == catalog.schema_digest,
            "S2 target identity does not match its pinned catalog table",
        )?;
        validate_target_columns(record, catalog.catalog_columns)?;
        validate_target_foreign_keys(graph, all_catalog, retained, record, catalog)?;
    }
    require(
        resolution_count != 0,
        "S7 target table has no S2 statement resolution",
    )
}

fn validate_target_columns(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    columns: &[SemanticsV2CatalogColumnWitness<'_>],
) -> Result<(), crate::EngineError> {
    require(
        record.catalog_columns().len() == columns.len(),
        "S2 target columns do not exactly close the pinned catalog columns",
    )?;
    for (source, catalog) in record.catalog_columns().zip(columns) {
        require(
            catalog_column_matches_source(catalog, source),
            "S2 target column differs from its pinned catalog column",
        )?;
    }
    Ok(())
}

fn validate_foreign_parent_table(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    dependency: &RetainedDependencyToken,
    catalog: &SemanticsV2CatalogTableWitness<'_>,
) -> Result<(), crate::EngineError> {
    require(
        dependency.catalog_epoch == identity.catalog_epoch
            && dependency.base_generation == catalog.data_generation
            && dependency.base_root == catalog.data_root
            && dependency.schema_digest == catalog.schema_digest
            && dependency.name_digest == qualified_name_digest(catalog.schema, catalog.name)?,
        "foreign-parent table token does not match the pinned catalog identity",
    )?;
    let mut source_count = 0_u32;
    for usage in graph.dependency_uses.iter().filter(|usage| {
        usage.dependency_ref == dependency.dependency_ref && usage.role == FOREIGN_PARENT_TABLE_ROLE
    }) {
        let source =
            source_dependency_for_use(graph, usage.statement_ordinal, usage.source_ordinal)?;
        source_count = source_count
            .checked_add(1)
            .ok_or_else(|| validation_error("foreign-parent source count overflows"))?;
        require(
            source.oid == catalog.display_oid
                && source.schema == catalog.schema
                && source.name == catalog.name
                && source.schema_digest == catalog.schema_digest,
            "S2 foreign-parent dependency does not match its pinned catalog table",
        )?;
    }
    require(
        source_count != 0,
        "foreign-parent table token has no exact S2 dependency use",
    )
}

fn validate_parent_table_columns(
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogTableWitness<'_>,
) -> Result<(), crate::EngineError> {
    for record in &graph.records {
        for foreign_key in record.foreign_keys() {
            let dependency = record
                .dependencies()
                .nth(
                    usize::try_from(foreign_key.parent_dependency_ordinal).map_err(|_| {
                        validation_error(
                            "S2 foreign-parent dependency ordinal exceeds addressability",
                        )
                    })?,
                )
                .ok_or_else(|| validation_error("S2 foreign-parent dependency is absent"))?;
            if dependency.oid == catalog.display_oid
                && dependency.schema == catalog.schema
                && dependency.name == catalog.name
            {
                require(
                    catalog.catalog_columns.iter().any(|column| {
                        catalog_column_matches_binding(column, foreign_key.parent_column)
                    }),
                    "S2 FK parent column is absent from its pinned catalog table",
                )?;
                for key in record.foreign_key_supporting_index_keys(foreign_key.raw_ordinal)? {
                    require(
                        catalog
                            .catalog_columns
                            .iter()
                            .any(|column| catalog_column_matches_binding(column, key)),
                        "S2 FK supporting-index key is absent from its parent catalog table",
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn source_dependency_for_use(
    graph: &ReservedSemanticsV2Graph,
    statement_ordinal: u32,
    source_ordinal: u32,
) -> Result<DecodedDependencyFacts<'_>, crate::EngineError> {
    let resolution = graph
        .resolutions
        .iter()
        .find(|resolution| resolution.statement_ordinal == statement_ordinal)
        .ok_or_else(|| validation_error("dependency use has no statement resolution"))?;
    let record = graph
        .records
        .get(usize::try_from(resolution.record_ref).map_err(|_| {
            validation_error("dependency-use record reference exceeds addressability")
        })?)
        .ok_or_else(|| validation_error("dependency-use record is absent"))?;
    record
        .dependencies()
        .nth(usize::try_from(source_ordinal).map_err(|_| {
            validation_error("dependency-use source ordinal exceeds addressability")
        })?)
        .ok_or_else(|| validation_error("dependency-use source dependency is absent"))
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

fn validate_guard_closure(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for guard in catalog.guards {
        let count = graph
            .dependencies
            .iter()
            .filter(|dependency| guard_matches_dependency(identity, guard, dependency))
            .count();
        require(
            count != 0,
            "pinned catalog guard is outside the exact S2/catalog guard closure",
        )?;
        validate_guard_owner(catalog, guard)?;
    }
    for dependency in graph.dependencies.iter().filter(|dependency| {
        matches!(
            dependency.kind,
            NOT_NULL_GUARD | CHECK_GUARD | DOMAIN_CONSTRAINT_GUARD
        )
    }) {
        let count = catalog
            .guards
            .iter()
            .filter(|guard| guard_matches_dependency(identity, guard, dependency))
            .count();
        require(
            count == 1,
            "S7 guard dependency does not select exactly one pinned catalog guard",
        )?;
    }
    validate_guard_nested_membership(catalog)
}

fn guard_matches_dependency(
    identity: SemanticsV2BoundIdentity,
    guard: &SemanticsV2CatalogGuardWitness<'_>,
    dependency: &RetainedDependencyToken,
) -> bool {
    guard.kind == dependency.kind
        && guard.stable_guard_id == dependency.stable_object_id
        && guard.display_oid == dependency.display_oid
        && guard.catalog_generation == dependency.base_generation
        && guard.shape_digest == dependency.schema_digest
        && guard.program_or_descriptor_root == dependency.base_root
        && dependency.catalog_epoch == identity.catalog_epoch
        && guard_name_digest(guard).is_ok_and(|digest| dependency.name_digest == digest)
}

fn validate_guard_owner(
    catalog: &SemanticsV2CatalogWitness<'_>,
    guard: &SemanticsV2CatalogGuardWitness<'_>,
) -> Result<(), crate::EngineError> {
    match guard.kind {
        NOT_NULL_GUARD | CHECK_GUARD => {
            require(
                guard.owner_kind == 1,
                "table guard has an invalid owner kind",
            )?;
            let table = catalog
                .tables
                .iter()
                .find(|table| {
                    table.stable_table_id == guard.owner_stable_id
                        && table.display_oid == guard.owner_display_oid
                })
                .ok_or_else(|| validation_error("table guard owner is absent from the catalog"))?;
            if guard.kind == NOT_NULL_GUARD {
                let column = table
                    .catalog_columns
                    .iter()
                    .find(|column| {
                        column.catalog_column_ordinal == guard.owner_catalog_column_ordinal
                    })
                    .ok_or_else(|| validation_error("NOT NULL guard owner column is absent"))?;
                require(
                    guard.synthesized_not_null
                        && guard.raw_constraint_ordinal == 0
                        && guard.shape_digest == column.column_shape_digest
                        && guard.program_or_descriptor_root == column.column_root,
                    "NOT NULL guard differs from its pinned catalog column descriptor",
                )?;
            } else {
                require(
                    !guard.synthesized_not_null
                        && guard.owner_catalog_column_ordinal == u32::MAX
                        && guard.domain_ordinal == u32::MAX,
                    "CHECK guard has an invalid table-column/domain binding",
                )?;
            }
        }
        DOMAIN_CONSTRAINT_GUARD => {
            require(
                guard.owner_kind == 2 && guard.owner_catalog_column_ordinal == u32::MAX,
                "domain guard has an invalid owner binding",
            )?;
            let domain = catalog
                .domains
                .iter()
                .find(|domain| {
                    domain.stable_domain_id == guard.owner_stable_id
                        && domain.display_oid == guard.owner_display_oid
                })
                .ok_or_else(|| validation_error("domain guard owner is absent from the catalog"))?;
            require(
                domain
                    .constraints
                    .iter()
                    .any(|candidate| same_guard(candidate, guard)),
                "domain guard is absent from its ordered domain-constraint list",
            )?;
        }
        _ => return Err(validation_error("catalog guard kind is invalid")),
    }
    Ok(())
}

fn validate_guard_nested_membership(
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for table in catalog.tables {
        for guard in table.not_null_guards.iter().chain(table.check_guards) {
            require(
                catalog
                    .guards
                    .iter()
                    .filter(|candidate| same_guard(candidate, guard))
                    .count()
                    == 1
                    && guard.owner_stable_id == table.stable_table_id
                    && guard.owner_display_oid == table.display_oid,
                "table guard list has an extra, missing, or foreign guard",
            )?;
        }
    }
    for domain in catalog.domains {
        for guard in domain.constraints {
            require(
                catalog
                    .guards
                    .iter()
                    .filter(|candidate| same_guard(candidate, guard))
                    .count()
                    == 1
                    && guard.owner_stable_id == domain.stable_domain_id
                    && guard.owner_display_oid == domain.display_oid,
                "domain guard list has an extra, missing, or foreign guard",
            )?;
        }
    }
    for guard in catalog.guards {
        let nested_count = match guard.kind {
            NOT_NULL_GUARD => catalog
                .tables
                .iter()
                .flat_map(|table| table.not_null_guards)
                .filter(|candidate| same_guard(candidate, guard))
                .count(),
            CHECK_GUARD => catalog
                .tables
                .iter()
                .flat_map(|table| table.check_guards)
                .filter(|candidate| same_guard(candidate, guard))
                .count(),
            DOMAIN_CONSTRAINT_GUARD => catalog
                .domains
                .iter()
                .flat_map(|domain| domain.constraints)
                .filter(|candidate| same_guard(candidate, guard))
                .count(),
            _ => 0,
        };
        require(
            nested_count == 1,
            "catalog guard does not occur exactly once in its owner list",
        )?;
    }
    Ok(())
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

fn validate_target_foreign_keys(
    graph: &ReservedSemanticsV2Graph,
    all_catalog: &SemanticsV2CatalogWitness<'_>,
    retained: &RetainedTable,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    catalog: &SemanticsV2CatalogTableWitness<'_>,
) -> Result<(), crate::EngineError> {
    require(
        record.foreign_keys().len() == catalog.foreign_keys.len(),
        "S2 target foreign-key count differs from the pinned catalog table",
    )?;
    for (source, catalog_fk) in record.foreign_keys().zip(catalog.foreign_keys) {
        require(
            catalog_fk_matches_source(
                graph,
                all_catalog,
                retained,
                record,
                catalog,
                catalog_fk,
                source,
            )?,
            "S2 foreign key differs from its exact pinned catalog closure",
        )?;
    }
    Ok(())
}

fn catalog_fk_matches_source(
    graph: &ReservedSemanticsV2Graph,
    all_catalog: &SemanticsV2CatalogWitness<'_>,
    retained: &RetainedTable,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    owner: &SemanticsV2CatalogTableWitness<'_>,
    catalog: &SemanticsV2CatalogForeignKeyWitness<'_>,
    source: DecodedForeignKeyFacts<'_>,
) -> Result<bool, crate::EngineError> {
    if catalog.raw_foreign_key_ordinal != source.raw_ordinal
        || catalog.schema != owner.schema
        || catalog.name != source.name
        || !owner.catalog_columns.iter().any(|column| {
            catalog_column_matches_binding(column, source.child_column)
                && column.catalog_column_ordinal == catalog.child_catalog_column_ordinal
                && column.stable_column_id == catalog.child_stable_column_id
        })
    {
        return Ok(false);
    }
    let parent_dependency = record
        .dependencies()
        .nth(
            usize::try_from(source.parent_dependency_ordinal).map_err(|_| {
                validation_error("FK parent dependency ordinal exceeds addressability")
            })?,
        )
        .ok_or_else(|| validation_error("FK parent dependency is absent"))?;
    let Some(parent) = all_catalog.tables.iter().find(|table| {
        table.stable_table_id == catalog.parent_stable_table_id
            && table.display_oid == catalog.parent_display_oid
    }) else {
        return Ok(false);
    };
    if parent.schema != parent_dependency.schema
        || parent.name != parent_dependency.name
        || parent.schema_digest != parent_dependency.schema_digest
        || !parent.catalog_columns.iter().any(|column| {
            catalog_column_matches_binding(column, source.parent_column)
                && column.catalog_column_ordinal == catalog.parent_catalog_column_ordinal
                && column.stable_column_id == catalog.parent_stable_column_id
        })
    {
        return Ok(false);
    }
    let supporting_index = all_catalog
        .indexes
        .iter()
        .find(|index| index.stable_index_id == catalog.supporting_stable_index_id);
    let Some(supporting_index) = supporting_index else {
        return Ok(false);
    };
    if !catalog_index_matches_s2(supporting_index, source.supporting_index)
        || supporting_index.owner_stable_table_id != catalog.parent_stable_table_id
        || supporting_index.owner_display_oid != catalog.parent_display_oid
        || !graph.dependencies.iter().any(|dependency| {
            dependency.kind == FOREIGN_KEY_GUARD
                && dependency.stable_object_id == catalog.stable_constraint_id
                && dependency.display_oid == catalog.display_oid
                && dependency.target_table_ref == retained.table_ref
        })
    {
        return Ok(false);
    }
    Ok(true)
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

fn guard_name_digest(
    guard: &SemanticsV2CatalogGuardWitness<'_>,
) -> Result<[u8; 32], crate::EngineError> {
    if guard.synthesized_not_null {
        let source_ordinal = match guard.owner_kind {
            1 => guard.owner_catalog_column_ordinal,
            2 => 0,
            _ => return Err(validation_error("synthesized guard owner kind is invalid")),
        };
        Ok(v2_digest(
            b"gpu-db/write001/s7-synthesized-not-null-name/v2",
            &[
                &[guard.owner_kind],
                &guard.owner_stable_id.to_le_bytes(),
                &source_ordinal.to_le_bytes(),
            ],
        ))
    } else {
        qualified_name_digest(guard.schema, guard.name)
    }
}

fn same_guard(
    left: &SemanticsV2CatalogGuardWitness<'_>,
    right: &SemanticsV2CatalogGuardWitness<'_>,
) -> bool {
    left.kind == right.kind
        && left.stable_guard_id == right.stable_guard_id
        && left.display_oid == right.display_oid
        && left.schema == right.schema
        && left.name == right.name
        && left.synthesized_not_null == right.synthesized_not_null
        && left.owner_kind == right.owner_kind
        && left.owner_stable_id == right.owner_stable_id
        && left.owner_display_oid == right.owner_display_oid
        && left.owner_catalog_column_ordinal == right.owner_catalog_column_ordinal
        && left.domain_ordinal == right.domain_ordinal
        && left.raw_constraint_ordinal == right.raw_constraint_ordinal
        && left.source_ordinal == right.source_ordinal
        && left.shape_digest == right.shape_digest
        && left.program_or_descriptor_root == right.program_or_descriptor_root
        && left.catalog_generation == right.catalog_generation
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

fn validate_column_order(
    rows: &[SemanticsV2CatalogColumnWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut expected = 0_u32;
    for row in rows {
        require(
            row.catalog_column_ordinal == expected
                && row.stable_column_id != 0
                && row.attnum != 0
                && valid_identifier(row.name)
                && row.column_shape_digest != [0; 32]
                && row.column_root != [0; 32],
            "catalog columns are not dense, resolved, and complete",
        )?;
        expected = expected
            .checked_add(1)
            .ok_or_else(|| validation_error("catalog column ordinal overflows"))?;
    }
    Ok(())
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

fn validate_guard_list_order(
    rows: &[SemanticsV2CatalogGuardWitness<'_>],
    kind: u8,
    owner: &str,
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    let mut expected_raw_constraint = 0_u32;
    for row in rows {
        let raw_constraint_is_dense = kind != NOT_NULL_GUARD;
        require(
            row.kind == kind
                && previous.is_none_or(|prior| prior < row.source_ordinal)
                && row.stable_guard_id != 0
                && row.stable_guard_id != u64::MAX
                && (!raw_constraint_is_dense
                    || (row.raw_constraint_ordinal == expected_raw_constraint
                        && row.source_ordinal == expected_raw_constraint)),
            "catalog guard list is not in exact source order",
        )?;
        previous = Some(row.source_ordinal);
        if raw_constraint_is_dense {
            expected_raw_constraint = expected_raw_constraint
                .checked_add(1)
                .ok_or_else(|| validation_error("catalog guard raw ordinal overflows"))?;
        }
    }
    let _ = owner;
    Ok(())
}

fn validate_foreign_key_order(
    rows: &[SemanticsV2CatalogForeignKeyWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut expected = 0_u32;
    for row in rows {
        require(
            row.raw_foreign_key_ordinal == expected
                && row.stable_constraint_id != 0
                && row.stable_constraint_id != u64::MAX
                && row.display_oid != 0
                && row.display_oid <= 0x7fff_ffff
                && valid_identifier(row.schema)
                && valid_identifier(row.name)
                && row.child_stable_column_id != 0
                && row.parent_stable_table_id != 0
                && row.parent_stable_column_id != 0
                && row.supporting_stable_index_id != 0,
            "catalog foreign keys are not dense and complete",
        )?;
        expected = expected
            .checked_add(1)
            .ok_or_else(|| validation_error("catalog FK ordinal overflows"))?;
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
