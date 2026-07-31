//! Guard applicability, terminal catalog matching, and foreign-key closure.

use super::super::{
    graph::{ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedTable},
    SemanticsV2BoundIdentity, SemanticsV2CatalogForeignKeyWitness, SemanticsV2CatalogGuardWitness,
    SemanticsV2CatalogTableWitness, SemanticsV2CatalogWitness,
};
use crate::typed_insert_batch::DecodedForeignKeyFacts;

pub(super) fn validate_guard_order(
    rows: &[SemanticsV2CatalogGuardWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    for row in rows {
        super::require(
            row.stable_guard_id != 0
                && row.stable_guard_id != u64::MAX
                && matches!(
                    row.kind,
                    super::NOT_NULL_GUARD | super::CHECK_GUARD | super::DOMAIN_CONSTRAINT_GUARD
                )
                && previous.is_none_or(|prior| prior < row.stable_guard_id),
            "catalog guards are not in strict stable-ID order",
        )?;
        if row.synthesized_not_null {
            super::require(
                row.display_oid == 0
                    && matches!(
                        row.kind,
                        super::NOT_NULL_GUARD | super::DOMAIN_CONSTRAINT_GUARD
                    ),
                "only synthesized NOT NULL guards may omit a display OID",
            )?;
        } else {
            super::require(
                row.display_oid != 0
                    && row.display_oid <= 0x7fff_ffff
                    && super::valid_identifier(row.schema)
                    && super::valid_identifier(row.name),
                "named catalog guard identity is invalid",
            )?;
        }
        previous = Some(row.stable_guard_id);
    }
    Ok(())
}

pub(super) fn validate_guard_list_order(
    rows: &[SemanticsV2CatalogGuardWitness<'_>],
    kind: u8,
    owner: &str,
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    let mut expected_raw_constraint = 0_u32;
    for row in rows {
        let raw_constraint_is_dense = kind != super::NOT_NULL_GUARD;
        super::require(
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
                .ok_or_else(|| super::validation_error("catalog guard raw ordinal overflows"))?;
        }
    }
    let _ = owner;
    Ok(())
}

pub(super) fn validate_guard_closure(
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
        super::require(
            count != 0,
            "pinned catalog guard is outside the exact S2/catalog guard closure",
        )?;
        validate_guard_owner(catalog, guard)?;
    }
    for dependency in graph.dependencies.iter().filter(|dependency| {
        matches!(
            dependency.kind,
            super::NOT_NULL_GUARD | super::CHECK_GUARD | super::DOMAIN_CONSTRAINT_GUARD
        )
    }) {
        let count = catalog
            .guards
            .iter()
            .filter(|guard| guard_matches_dependency(identity, guard, dependency))
            .count();
        super::require(
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
        super::NOT_NULL_GUARD | super::CHECK_GUARD => {
            super::require(
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
                .ok_or_else(|| {
                    super::validation_error("table guard owner is absent from the catalog")
                })?;
            if guard.kind == super::NOT_NULL_GUARD {
                let column = table
                    .catalog_columns
                    .iter()
                    .find(|column| {
                        column.catalog_column_ordinal == guard.owner_catalog_column_ordinal
                    })
                    .ok_or_else(|| {
                        super::validation_error("NOT NULL guard owner column is absent")
                    })?;
                super::require(
                    guard.synthesized_not_null
                        && guard.raw_constraint_ordinal == 0
                        && guard.shape_digest == column.column_shape_digest
                        && guard.program_or_descriptor_root == column.column_root,
                    "NOT NULL guard differs from its pinned catalog column descriptor",
                )?;
            } else {
                super::require(
                    !guard.synthesized_not_null
                        && guard.owner_catalog_column_ordinal == u32::MAX
                        && guard.domain_ordinal == u32::MAX,
                    "CHECK guard has an invalid table-column/domain binding",
                )?;
            }
        }
        super::DOMAIN_CONSTRAINT_GUARD => {
            super::require(
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
                .ok_or_else(|| {
                    super::validation_error("domain guard owner is absent from the catalog")
                })?;
            super::require(
                domain
                    .constraints
                    .iter()
                    .any(|candidate| same_guard(candidate, guard)),
                "domain guard is absent from its ordered domain-constraint list",
            )?;
        }
        _ => return Err(super::validation_error("catalog guard kind is invalid")),
    }
    Ok(())
}

fn validate_guard_nested_membership(
    catalog: &SemanticsV2CatalogWitness<'_>,
) -> Result<(), crate::EngineError> {
    for table in catalog.tables {
        for guard in table.not_null_guards.iter().chain(table.check_guards) {
            super::require(
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
            super::require(
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
            super::NOT_NULL_GUARD => catalog
                .tables
                .iter()
                .flat_map(|table| table.not_null_guards)
                .filter(|candidate| same_guard(candidate, guard))
                .count(),
            super::CHECK_GUARD => catalog
                .tables
                .iter()
                .flat_map(|table| table.check_guards)
                .filter(|candidate| same_guard(candidate, guard))
                .count(),
            super::DOMAIN_CONSTRAINT_GUARD => catalog
                .domains
                .iter()
                .flat_map(|domain| domain.constraints)
                .filter(|candidate| same_guard(candidate, guard))
                .count(),
            _ => 0,
        };
        super::require(
            nested_count == 1,
            "catalog guard does not occur exactly once in its owner list",
        )?;
    }
    Ok(())
}

pub(super) fn validate_target_foreign_keys(
    graph: &ReservedSemanticsV2Graph,
    all_catalog: &SemanticsV2CatalogWitness<'_>,
    retained: &RetainedTable,
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    catalog: &SemanticsV2CatalogTableWitness<'_>,
) -> Result<(), crate::EngineError> {
    super::require(
        record.foreign_keys().len() == catalog.foreign_keys.len(),
        "S2 target foreign-key count differs from the pinned catalog table",
    )?;
    for (source, catalog_fk) in record.foreign_keys().zip(catalog.foreign_keys) {
        super::require(
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
            super::catalog_column_matches_binding(column, source.child_column)
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
                super::validation_error("FK parent dependency ordinal exceeds addressability")
            })?,
        )
        .ok_or_else(|| super::validation_error("FK parent dependency is absent"))?;
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
            super::catalog_column_matches_binding(column, source.parent_column)
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
    if !super::catalog_index_matches_s2(supporting_index, source.supporting_index)
        || supporting_index.owner_stable_table_id != catalog.parent_stable_table_id
        || supporting_index.owner_display_oid != catalog.parent_display_oid
        || !graph.dependencies.iter().any(|dependency| {
            dependency.kind == super::FOREIGN_KEY_GUARD
                && dependency.stable_object_id == supporting_index.stable_index_id
                && dependency.display_oid == supporting_index.display_oid
                && dependency.base_generation == supporting_index.base_generation
                && dependency.schema_digest == supporting_index.schema_digest
                && dependency.base_root == supporting_index.base_root
                && super::qualified_name_digest(supporting_index.schema, supporting_index.name)
                    .is_ok_and(|digest| dependency.name_digest == digest)
                && graph.indexes.iter().any(|descriptor| {
                    dependency.descriptor_ref == descriptor.index_ref
                        && descriptor.stable_index_id == supporting_index.stable_index_id
                        && descriptor.display_oid == supporting_index.display_oid
                        && descriptor.descriptor_digest != [0; 32]
                })
                && dependency.target_table_ref == retained.table_ref
        })
    {
        return Ok(false);
    }
    Ok(true)
}

fn guard_name_digest(
    guard: &SemanticsV2CatalogGuardWitness<'_>,
) -> Result<[u8; 32], crate::EngineError> {
    if guard.synthesized_not_null {
        let source_ordinal = match guard.owner_kind {
            1 => guard.owner_catalog_column_ordinal,
            2 => 0,
            _ => {
                return Err(super::validation_error(
                    "synthesized guard owner kind is invalid",
                ))
            }
        };
        Ok(super::v2_digest(
            b"gpu-db/write001/s7-synthesized-not-null-name/v2",
            &[
                &[guard.owner_kind],
                &guard.owner_stable_id.to_le_bytes(),
                &source_ordinal.to_le_bytes(),
            ],
        ))
    } else {
        super::qualified_name_digest(guard.schema, guard.name)
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
