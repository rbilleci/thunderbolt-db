//! Durable allocator leases and exact table closure for pinned catalog evidence.

use super::super::{
    graph::{ReservedSemanticsV2Graph, RetainedDependencyToken, RetainedTable},
    SemanticsV2BoundIdentity, SemanticsV2CatalogColumnWitness, SemanticsV2CatalogTableWitness,
    SemanticsV2CatalogWitness,
};
use crate::typed_insert_batch::DecodedDependencyFacts;
use sha2::{Digest, Sha256};

#[cfg(test)]
#[path = "allocator_tests.rs"]
mod allocator_tests;

/// Concrete borrowed authority from the durable allocator index.  It carries both the complete
/// immutable index and the transaction's explicit frozen selection; the latter is never inferred
/// from an epoch.  The non-`Copy` proof borrows an opaque pin, so its lifetime keeps the pinned
/// checkpoint/index generation alive until generation validation consumes the pending owner.
pub(in crate::typed_insert_aggregate::semantics_v2::retained) struct SemanticsV2DurableAllocatorIndexProof<
    'a,
> {
    snapshot: &'a SemanticsV2ImmutableDurableAllocatorIndex<'a>,
    checkpoint_pin: &'a SemanticsV2PinnedAllocatorIndexGeneration,
    selected: &'a [SemanticsV2SelectedAllocatorLeaseWitness],
}

/// The immutable complete index carries authenticated marker frontiers and a root recomputed
/// over every row.  Its fields are private to the durable-index adapter; codec bytes cannot
/// manufacture either a root or a record selection from this owner.
#[allow(dead_code)]
pub(super) struct SemanticsV2ImmutableDurableAllocatorIndex<'a> {
    database_id: [u8; 16],
    index_generation: u64,
    index_root: [u8; 32],
    complete_next_commit_sequence: u64,
    durable_next_commit_sequence: u64,
    published_next_commit_sequence: u64,
    records: &'a [SemanticsV2DurableAllocatorLeaseRecord],
}

/// Opaque active checkpoint/index-generation ownership.  Its non-`Copy` shape is intentional:
/// a proof can borrow this guard but cannot duplicate, retire, or recreate its retention claim.
#[allow(dead_code)]
pub(super) struct SemanticsV2PinnedAllocatorIndexGeneration {
    database_id: [u8; 16],
    cluster_id: [u8; 16],
    timeline_id: [u8; 16],
    format_epoch: u64,
    leader_epoch: u64,
    index_generation: u64,
    index_root: [u8; 32],
    retained_through_commit_sequence: u64,
}

/// One record in the authoritative complete index.  It stores only the frozen allocator-marker
/// tuple; complete/durable/published state comes from authenticated snapshot frontiers instead
/// of self-described copies on every row.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2DurableAllocatorLeaseRecord {
    database_id: [u8; 16],
    allocator_kind: u8,
    stable_allocator_id: u64,
    lease_epoch: u64,
    lease_start: u64,
    lease_end: u64,
    prior_high_water: u64,
    new_high_water: u64,
    marker_system_transaction_id: u64,
    marker_commit_sequence: u64,
}

/// Exact transaction-selected witness.  The index ordinal is a bounded handle, while the full
/// frozen tuple prevents a narrowed/reconstructed lease from posing as the authoritative row.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SemanticsV2SelectedAllocatorLeaseWitness {
    record_ordinal: u32,
    database_id: [u8; 16],
    allocator_kind: u8,
    stable_allocator_id: u64,
    lease_epoch: u64,
    lease_start: u64,
    lease_end: u64,
    prior_high_water: u64,
    new_high_water: u64,
    marker_system_transaction_id: u64,
    marker_commit_sequence: u64,
}

/// Test-only adapter for immutable authoritative records and explicit selected witnesses.  It
/// is the sole local construction seam; production has no callback or codec-visible constructor.
#[cfg(test)]
pub(super) fn proof_from_immutable_checked_records_for_test<'a>(
    snapshot: &'a SemanticsV2ImmutableDurableAllocatorIndex<'a>,
    checkpoint_pin: &'a SemanticsV2PinnedAllocatorIndexGeneration,
    selected: &'a [SemanticsV2SelectedAllocatorLeaseWitness],
) -> SemanticsV2DurableAllocatorIndexProof<'a> {
    SemanticsV2DurableAllocatorIndexProof {
        snapshot,
        checkpoint_pin,
        selected,
    }
}

/// Closure-scoped input for a retained-boundary test proof.  It carries only the table allocator
/// interval; the durable snapshot, authenticated root, active pin, selected rows, and all marker
/// lifecycle fields are constructed locally below and cannot escape this adapter.
#[cfg(test)]
#[derive(Clone, Copy)]
pub(in crate::typed_insert_aggregate::semantics_v2::retained) struct AllocatorLeaseSpecForTest {
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) stable_allocator_id: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) lease_start: u64,
    pub(in crate::typed_insert_aggregate::semantics_v2::retained) lease_end: u64,
}

/// Build an authoritative allocator proof only for the dynamic extent of `operation`.  The proof
/// borrows local records/snapshot/pin, so neither it nor its fixture backing can escape; callers
/// must consume it immediately through the ordinary catalog-and-allocator validation path.
#[cfg(test)]
pub(super) fn with_allocator_proof_for_test<T>(
    identity: SemanticsV2BoundIdentity,
    leases: &[AllocatorLeaseSpecForTest],
    operation: impl FnOnce(SemanticsV2DurableAllocatorIndexProof<'_>) -> T,
) -> T {
    assert!(
        identity.commit_sequence > 2,
        "test allocator proof needs a prior durable marker"
    );
    let marker_transaction = if identity.stable_transaction_id == 1 {
        2
    } else {
        1
    };
    let records: Vec<_> = leases
        .iter()
        .map(|lease| {
            assert!(
                lease.stable_allocator_id != 0
                    && lease.lease_start != 0
                    && lease.lease_start < lease.lease_end,
                "test allocator lease has an invalid exact interval"
            );
            SemanticsV2DurableAllocatorLeaseRecord {
                database_id: identity.database_id,
                allocator_kind: 1,
                stable_allocator_id: lease.stable_allocator_id,
                lease_epoch: 1,
                lease_start: lease.lease_start,
                lease_end: lease.lease_end,
                prior_high_water: lease.lease_start,
                new_high_water: lease.lease_end,
                marker_system_transaction_id: marker_transaction,
                marker_commit_sequence: 1,
            }
        })
        .collect();
    let selected: Vec<_> = records
        .iter()
        .enumerate()
        .map(
            |(ordinal, record)| SemanticsV2SelectedAllocatorLeaseWitness {
                record_ordinal: u32::try_from(ordinal).expect("test allocator record ordinal fits"),
                database_id: record.database_id,
                allocator_kind: record.allocator_kind,
                stable_allocator_id: record.stable_allocator_id,
                lease_epoch: record.lease_epoch,
                lease_start: record.lease_start,
                lease_end: record.lease_end,
                prior_high_water: record.prior_high_water,
                new_high_water: record.new_high_water,
                marker_system_transaction_id: record.marker_system_transaction_id,
                marker_commit_sequence: record.marker_commit_sequence,
            },
        )
        .collect();
    let mut snapshot = SemanticsV2ImmutableDurableAllocatorIndex {
        database_id: identity.database_id,
        index_generation: 1,
        index_root: [0; 32],
        complete_next_commit_sequence: identity.commit_sequence + 1,
        durable_next_commit_sequence: identity.commit_sequence + 1,
        published_next_commit_sequence: identity.commit_sequence + 1,
        records: &records,
    };
    snapshot.index_root = immutable_index_root(&snapshot);
    let pin = SemanticsV2PinnedAllocatorIndexGeneration {
        database_id: identity.database_id,
        cluster_id: identity.cluster_id,
        timeline_id: identity.timeline_id,
        format_epoch: identity.format_epoch,
        leader_epoch: identity.leader_epoch,
        index_generation: snapshot.index_generation,
        index_root: snapshot.index_root,
        retained_through_commit_sequence: identity.commit_sequence,
    };
    operation(SemanticsV2DurableAllocatorIndexProof {
        snapshot: &snapshot,
        checkpoint_pin: &pin,
        selected: &selected,
    })
}

pub(super) fn validate_durable_allocator_index_identity(
    identity: SemanticsV2BoundIdentity,
    proof: &SemanticsV2DurableAllocatorIndexProof<'_>,
) -> Result<(), crate::EngineError> {
    validate_complete_index(identity, proof)?;
    validate_selected_marker_lifecycle(identity, proof)
}

fn validate_complete_index(
    identity: SemanticsV2BoundIdentity,
    proof: &SemanticsV2DurableAllocatorIndexProof<'_>,
) -> Result<(), crate::EngineError> {
    let snapshot = proof.snapshot;
    let pin = proof.checkpoint_pin;
    super::require(
        snapshot.database_id == identity.database_id
            && snapshot.index_generation != 0
            && snapshot.index_generation != u64::MAX
            && snapshot.index_root != [0; 32]
            && snapshot.complete_next_commit_sequence != 0
            && snapshot.durable_next_commit_sequence != 0
            && snapshot.published_next_commit_sequence != 0
            && pin.database_id == identity.database_id
            && pin.cluster_id == identity.cluster_id
            && pin.timeline_id == identity.timeline_id
            && pin.format_epoch == identity.format_epoch
            && pin.leader_epoch == identity.leader_epoch
            && pin.leader_epoch != 0
            && pin.index_generation == snapshot.index_generation
            && pin.index_root == snapshot.index_root
            && pin.retained_through_commit_sequence != 0
            && immutable_index_root(snapshot) == snapshot.index_root,
        "durable allocator pinned index identity or authenticated root is invalid",
    )?;

    for (record_ordinal, record) in snapshot.records.iter().enumerate() {
        validate_complete_record(identity, record)?;
        for earlier in &snapshot.records[..record_ordinal] {
            if earlier.stable_allocator_id == record.stable_allocator_id
                && earlier.lease_epoch == record.lease_epoch
            {
                super::require(
                    earlier.lease_end <= record.lease_start
                        || record.lease_end <= earlier.lease_start,
                    "complete durable allocator index has hidden allocator lease overlap",
                )?;
            }
        }
    }
    Ok(())
}

fn validate_complete_record(
    identity: SemanticsV2BoundIdentity,
    record: &SemanticsV2DurableAllocatorLeaseRecord,
) -> Result<(), crate::EngineError> {
    super::require(
        record.database_id == identity.database_id
            && record.allocator_kind == 1
            && record.stable_allocator_id != 0
            && record.stable_allocator_id != u64::MAX
            && record.lease_epoch != 0
            && record.lease_epoch != u64::MAX
            && record.lease_start != 0
            && record.lease_start != u64::MAX
            && record.lease_end != 0
            && record.lease_end != u64::MAX
            && record.prior_high_water <= record.lease_start
            && record.lease_start < record.lease_end
            && record.new_high_water == record.lease_end
            && record.marker_system_transaction_id != 0
            && record.marker_system_transaction_id != u64::MAX
            && record.marker_commit_sequence != 0
            && record.marker_commit_sequence != u64::MAX,
        "complete durable allocator index record has an invalid immutable shape",
    )
}

fn validate_selected_marker_lifecycle(
    identity: SemanticsV2BoundIdentity,
    proof: &SemanticsV2DurableAllocatorIndexProof<'_>,
) -> Result<(), crate::EngineError> {
    for (selected_ordinal, selected) in proof.selected.iter().enumerate() {
        let record = proof
            .snapshot
            .records
            .get(usize::try_from(selected.record_ordinal).map_err(|_| {
                super::validation_error(
                    "selected durable allocator record ordinal is unaddressable",
                )
            })?)
            .ok_or_else(|| {
                super::validation_error("selected durable allocator record is absent")
            })?;
        super::require(
            selected_matches_record(selected, record)
                && selected.marker_system_transaction_id != identity.stable_transaction_id
                && selected.marker_commit_sequence < identity.commit_sequence
                && selected.marker_commit_sequence < proof.snapshot.complete_next_commit_sequence
                && selected.marker_commit_sequence < proof.snapshot.durable_next_commit_sequence
                && selected.marker_commit_sequence < proof.snapshot.published_next_commit_sequence
                && selected.marker_commit_sequence
                    <= proof.checkpoint_pin.retained_through_commit_sequence,
            "selected durable allocator witness lacks exact member, lifecycle, or checkpoint evidence",
        )?;
        for earlier in &proof.selected[..selected_ordinal] {
            super::require(
                earlier.record_ordinal != selected.record_ordinal,
                "selected durable allocator witness duplicates an authoritative record",
            )?;
        }
    }
    Ok(())
}

fn selected_matches_record(
    selected: &SemanticsV2SelectedAllocatorLeaseWitness,
    record: &SemanticsV2DurableAllocatorLeaseRecord,
) -> bool {
    selected.database_id == record.database_id
        && selected.allocator_kind == record.allocator_kind
        && selected.stable_allocator_id == record.stable_allocator_id
        && selected.lease_epoch == record.lease_epoch
        && selected.lease_start == record.lease_start
        && selected.lease_end == record.lease_end
        && selected.prior_high_water == record.prior_high_water
        && selected.new_high_water == record.new_high_water
        && selected.marker_system_transaction_id == record.marker_system_transaction_id
        && selected.marker_commit_sequence == record.marker_commit_sequence
}

fn immutable_index_root(snapshot: &SemanticsV2ImmutableDurableAllocatorIndex<'_>) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"gpu-db/write001/durable-allocator-index/v1");
    digest.update(snapshot.database_id);
    digest.update(snapshot.index_generation.to_le_bytes());
    digest.update(snapshot.complete_next_commit_sequence.to_le_bytes());
    digest.update(snapshot.durable_next_commit_sequence.to_le_bytes());
    digest.update(snapshot.published_next_commit_sequence.to_le_bytes());
    digest.update(
        u64::try_from(snapshot.records.len())
            .expect("durable allocator index record count fits u64")
            .to_le_bytes(),
    );
    for record in snapshot.records {
        digest.update(record.database_id);
        digest.update([record.allocator_kind]);
        digest.update(record.stable_allocator_id.to_le_bytes());
        digest.update(record.lease_epoch.to_le_bytes());
        digest.update(record.lease_start.to_le_bytes());
        digest.update(record.lease_end.to_le_bytes());
        digest.update(record.prior_high_water.to_le_bytes());
        digest.update(record.new_high_water.to_le_bytes());
        digest.update(record.marker_system_transaction_id.to_le_bytes());
        digest.update(record.marker_commit_sequence.to_le_bytes());
    }
    digest.finalize().into()
}

/// Match the explicit selected witnesses, in stable S7 table order, to the complete immutable
/// index.  Unrelated index records at any epoch remain valid evidence and are never selected by
/// a heuristic scan.
pub(super) fn validate_allocator_closure(
    identity: SemanticsV2BoundIdentity,
    graph: &ReservedSemanticsV2Graph,
    proof: &SemanticsV2DurableAllocatorIndexProof<'_>,
) -> Result<(), crate::EngineError> {
    validate_durable_allocator_index_identity(identity, proof)?;
    super::require(
        proof.selected.len() == graph.tables.len(),
        "durable allocator proof does not have exactly one selected lease per S7 table",
    )?;
    let mut prior_table_id = None;
    for (table, selected) in graph.tables.iter().zip(proof.selected) {
        super::require(
            prior_table_id.is_none_or(|prior| prior < table.stable_table_id)
                && selected.stable_allocator_id == table.stable_table_id
                && selected.lease_start <= table.row_allocator_before
                && table.row_allocator_high_water <= selected.lease_end,
            "selected durable allocator lease order or full S7 table interval is invalid",
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
