//! Durable allocator leases and exact table closure for pinned catalog evidence.

use super::super::{
    graph::{ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedTable},
    SemanticsV2BoundIdentity, SemanticsV2CatalogColumnWitness, SemanticsV2CatalogTableWitness,
    SemanticsV2CatalogWitness, SemanticsV2RowAllocatorLeaseWitness,
};
use crate::typed_insert_batch::DecodedDependencyFacts;

/// The durable allocator index has already established marker provenance.  This codec boundary
/// only closes each proven lease against exactly one retained S7 table and its contained row
/// interval; it never tries to reconstruct allocator state from the transaction bytes.
pub(super) fn validate_allocator_closure(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    leases: &[SemanticsV2RowAllocatorLeaseWitness],
) -> Result<(), crate::EngineError> {
    super::require(
        leases.len() == graph.tables.len(),
        "durable allocator proof does not have exactly one lease per S7 table",
    )?;
    let mut prior_table_id = None;
    for (table, lease) in graph.tables.iter().zip(leases) {
        super::require(
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

pub(super) fn validate_table_order(
    rows: &[SemanticsV2CatalogTableWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut previous = None;
    for row in rows {
        super::require_stable_identity(row.stable_table_id, row.display_oid, "catalog table")?;
        super::require(
            super::valid_identifier(row.schema) && super::valid_identifier(row.name),
            "catalog table name is not a resolved identifier",
        )?;
        super::require(
            previous.is_none_or(|prior| prior < row.stable_table_id),
            "catalog tables are not in strict stable-ID order",
        )?;
        previous = Some(row.stable_table_id);
        validate_column_order(row.catalog_columns)?;
        super::guards::validate_guard_list_order(
            row.not_null_guards,
            super::NOT_NULL_GUARD,
            "NOT NULL",
        )?;
        super::guards::validate_guard_list_order(row.check_guards, super::CHECK_GUARD, "CHECK")?;
        validate_foreign_key_order(row.foreign_keys)?;
    }
    Ok(())
}

pub(super) fn validate_table_closure(
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
                    super::validation_error("catalog target-table match count overflows")
                })?;
                matched = Some(candidate);
            }
        }
        let Some(candidate) = matched else {
            return Err(super::validation_error(
                "S7 target table is absent from the pinned catalog",
            ));
        };
        super::require(
            count == 1,
            "S7 target table is ambiguous in the pinned catalog",
        )?;
        validate_target_table(identity, graph, catalog, table, candidate)?;
    }

    for dependency in &graph.dependencies {
        if matches!(
            dependency.kind,
            super::TARGET_TABLE | super::FOREIGN_PARENT_TABLE
        ) {
            let matches = catalog
                .tables
                .iter()
                .filter(|table| {
                    table.stable_table_id == dependency.stable_object_id
                        && table.display_oid == dependency.display_oid
                })
                .count();
            super::require(
                matches == 1,
                "table dependency does not select exactly one pinned catalog table",
            )?;
            if dependency.kind == super::FOREIGN_PARENT_TABLE {
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
            dependency.kind == super::FOREIGN_PARENT_TABLE
                && dependency.stable_object_id == table.stable_table_id
                && dependency.display_oid == table.display_oid
        });
        super::require(
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
    super::require(
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
            .ok_or_else(|| super::validation_error("target-table resolution count overflows"))?;
        let record = graph
            .records
            .get(usize::try_from(resolution.record_ref).map_err(|_| {
                super::validation_error("S7 record reference exceeds host addressability")
            })?)
            .ok_or_else(|| super::validation_error("S7 target-table record is absent"))?;
        let target = record.target_identity();
        super::require(
            target.oid == catalog.display_oid
                && target.schema == catalog.schema
                && target.name == catalog.name
                && target.schema_digest == catalog.schema_digest,
            "S2 target identity does not match its pinned catalog table",
        )?;
        validate_target_columns(record, catalog.catalog_columns)?;
        super::guards::validate_target_foreign_keys(graph, all_catalog, retained, record, catalog)?;
    }
    super::require(
        resolution_count != 0,
        "S7 target table has no S2 statement resolution",
    )
}

fn validate_target_columns(
    record: &crate::typed_insert_batch::DecodedTypedInsertRecord,
    columns: &[SemanticsV2CatalogColumnWitness<'_>],
) -> Result<(), crate::EngineError> {
    super::require(
        record.catalog_columns().len() == columns.len(),
        "S2 target columns do not exactly close the pinned catalog columns",
    )?;
    for (source, catalog) in record.catalog_columns().zip(columns) {
        super::require(
            super::catalog_column_matches_source(catalog, source),
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
    super::require(
        dependency.catalog_epoch == identity.catalog_epoch
            && dependency.base_generation == catalog.data_generation
            && dependency.base_root == catalog.data_root
            && dependency.schema_digest == catalog.schema_digest
            && dependency.name_digest
                == super::qualified_name_digest(catalog.schema, catalog.name)?,
        "foreign-parent table token does not match the pinned catalog identity",
    )?;
    let mut source_count = 0_u32;
    for usage in graph.dependency_uses.iter().filter(|usage| {
        usage.dependency_ref == dependency.dependency_ref
            && usage.role == super::FOREIGN_PARENT_TABLE_ROLE
    }) {
        let source =
            source_dependency_for_use(graph, usage.statement_ordinal, usage.source_ordinal)?;
        source_count = source_count
            .checked_add(1)
            .ok_or_else(|| super::validation_error("foreign-parent source count overflows"))?;
        super::require(
            source.oid == catalog.display_oid
                && source.schema == catalog.schema
                && source.name == catalog.name
                && source.schema_digest == catalog.schema_digest,
            "S2 foreign-parent dependency does not match its pinned catalog table",
        )?;
    }
    super::require(
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
                        super::validation_error(
                            "S2 foreign-parent dependency ordinal exceeds addressability",
                        )
                    })?,
                )
                .ok_or_else(|| super::validation_error("S2 foreign-parent dependency is absent"))?;
            if dependency.oid == catalog.display_oid
                && dependency.schema == catalog.schema
                && dependency.name == catalog.name
            {
                super::require(
                    catalog.catalog_columns.iter().any(|column| {
                        super::catalog_column_matches_binding(column, foreign_key.parent_column)
                    }),
                    "S2 FK parent column is absent from its pinned catalog table",
                )?;
                for key in record.foreign_key_supporting_index_keys(foreign_key.raw_ordinal)? {
                    super::require(
                        catalog
                            .catalog_columns
                            .iter()
                            .any(|column| super::catalog_column_matches_binding(column, key)),
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
        .ok_or_else(|| super::validation_error("dependency use has no statement resolution"))?;
    let record = graph
        .records
        .get(usize::try_from(resolution.record_ref).map_err(|_| {
            super::validation_error("dependency-use record reference exceeds addressability")
        })?)
        .ok_or_else(|| super::validation_error("dependency-use record is absent"))?;
    record
        .dependencies()
        .nth(usize::try_from(source_ordinal).map_err(|_| {
            super::validation_error("dependency-use source ordinal exceeds addressability")
        })?)
        .ok_or_else(|| super::validation_error("dependency-use source dependency is absent"))
}

fn validate_column_order(
    rows: &[SemanticsV2CatalogColumnWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut expected = 0_u32;
    for row in rows {
        super::require(
            row.catalog_column_ordinal == expected
                && row.stable_column_id != 0
                && row.attnum != 0
                && super::valid_identifier(row.name)
                && row.column_shape_digest != [0; 32]
                && row.column_root != [0; 32],
            "catalog columns are not dense, resolved, and complete",
        )?;
        expected = expected
            .checked_add(1)
            .ok_or_else(|| super::validation_error("catalog column ordinal overflows"))?;
    }
    Ok(())
}

fn validate_foreign_key_order(
    rows: &[super::super::SemanticsV2CatalogForeignKeyWitness<'_>],
) -> Result<(), crate::EngineError> {
    let mut expected = 0_u32;
    for row in rows {
        super::require(
            row.raw_foreign_key_ordinal == expected
                && row.stable_constraint_id != 0
                && row.stable_constraint_id != u64::MAX
                && row.display_oid != 0
                && row.display_oid <= 0x7fff_ffff
                && super::valid_identifier(row.schema)
                && super::valid_identifier(row.name)
                && row.child_stable_column_id != 0
                && row.parent_stable_table_id != 0
                && row.parent_stable_column_id != 0
                && row.supporting_stable_index_id != 0,
            "catalog foreign keys are not dense and complete",
        )?;
        expected = expected
            .checked_add(1)
            .ok_or_else(|| super::validation_error("catalog FK ordinal overflows"))?;
    }
    Ok(())
}
