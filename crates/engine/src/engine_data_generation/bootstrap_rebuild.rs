//! Root-free compiler input and proof contract for the first private bootstrap rebuild seam.
//!
//! This is deliberately preparation only. It has no CUDA submission, logical-root input,
//! result decoding, installation, publication, reader, cache, or host hash path.

use std::collections::{BTreeMap, BTreeSet};

use gpu_db_sql::SqlType;

use super::{
    bootstrap_publication::{
        BootstrapMaterializationLease, BootstrapPhysicalLayoutRoleKind,
        BootstrapPhysicalStorageType, BootstrapRebuildExpectationSource,
        BootstrapRebuildRootFreeResource, BootstrapRebuildRootFreeRole,
        BootstrapRebuildRootFreeSource, BootstrapResourceLedgerKind, BootstrapResourceLedgerOwner,
        BootstrapResourceStorageTier,
    },
    digest::{RootFormatVersion, StableColumnId, StableIndexId, StableTableId},
    resources::{
        self, BootstrapAttachedUninstalledResources, BootstrapRebuildAttachedOwners,
        BootstrapRebuildRuntimeIdentity,
    },
    DataGenerationError,
};

const V1_BOOTSTRAP_LAYOUT_ENCODING: u16 = 1;
const RADIX_DEPTH_COUNT: u8 = 64;
const V1_EMPTY_TYPED_RESOURCE_SENTINEL_BYTES: u64 = 1;
const MAX_BOOTSTRAP_REBUILD_OUTPUT_SLOTS: usize = 1_048_576;

type BootstrapRebuildTableRows = BTreeMap<StableTableId, u64>;
type BootstrapRebuildColumnTypes = BTreeMap<StableTableId, BTreeMap<StableColumnId, SqlType>>;
type BootstrapRebuildIndexKey = (u16, StableColumnId, SqlType);
type BootstrapRebuildIndexKeys =
    BTreeMap<(StableTableId, StableIndexId), Vec<BootstrapRebuildIndexKey>>;

struct BootstrapRebuildSemanticTables {
    rows: BootstrapRebuildTableRows,
    columns: BootstrapRebuildColumnTypes,
    index_keys: BootstrapRebuildIndexKeys,
}

/// The only compiler capability formed from an attached bootstrap source. It retains opaque,
/// exact owners and the root-free semantic source, plus the closed proof-slot contract a later
/// GPU completion must bind. It intentionally has neither `Clone` nor `Debug`.
pub(super) struct BootstrapRebuildCompilerInput {
    owners: BootstrapRebuildAttachedOwners,
    semantic: BootstrapRebuildSemanticLayout,
    prepared: BootstrapRebuildPreparedProofLayout,
}

/// Consumed compiler payload for the execution-only V1 adapter. It contains no expected roots;
/// comparator evidence remains in [`BootstrapRebuildExpectations`] and never enters CUDA input.
pub(super) struct BootstrapRebuildGpuParts {
    pub(super) owners: BootstrapRebuildAttachedOwners,
    pub(super) source: BootstrapRebuildRootFreeSource,
    pub(super) proof: BootstrapRebuildPreparedProofLayout,
}

impl BootstrapRebuildGpuParts {
    pub(super) fn into_compiler_input(self) -> BootstrapRebuildCompilerInput {
        let runtime = self.owners.runtime_identity();
        BootstrapRebuildCompilerInput {
            owners: self.owners,
            semantic: BootstrapRebuildSemanticLayout {
                source: self.source,
                runtime,
            },
            prepared: self.proof,
        }
    }
}

/// Comparator-only source kept separate from the compiler capability. A future completion
/// checker may consume it after GPU work, but no expected root is available to this module's
/// compiler-input layout. It intentionally has neither `Clone` nor `Debug`.
pub(super) struct BootstrapRebuildExpectations {
    expected: BootstrapRebuildExpectationSource,
}

/// A failed atomic preparation retains every consumed capability for a private retry. It has no
/// public constructor and offers no path for partially prepared compiler state to escape.
pub(super) struct BootstrapRebuildPreparationFailure {
    error: DataGenerationError,
    owners: BootstrapRebuildAttachedOwners,
    source: BootstrapRebuildRootFreeSource,
    expected: BootstrapRebuildExpectationSource,
}

struct BootstrapRebuildSemanticLayout {
    source: BootstrapRebuildRootFreeSource,
    runtime: BootstrapRebuildRuntimeIdentity,
}

/// A closed, ordered V1 output contract. Each slot is a proof binding coordinate, not a root or
/// CPU-readable value. The vocabulary follows the logical runtime-generation grammar rather than
/// physical resource roles, so layout offsets and source tier can never become commitments.
pub(super) struct BootstrapRebuildPreparedProofLayout {
    exact_output_count: u32,
    slots: Box<[BootstrapRebuildOutputSlot]>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum BootstrapRebuildOutputSlot {
    ColumnShape {
        table_id: StableTableId,
        column_id: StableColumnId,
    },
    TypedValueVector {
        table_id: StableTableId,
        column_id: StableColumnId,
    },
    CurrentRowLeaves {
        table_id: StableTableId,
    },
    RowMapDepth {
        table_id: StableTableId,
        depth: u8,
    },
    TableRoot {
        table_id: StableTableId,
    },
    DatabaseMapDepth {
        depth: u8,
    },
    DatabaseRoot,
}

/// Atomically consume an attached uninstalled source. Success creates exactly the root-free
/// compiler capability and the comparator-only expectation capability. Failure owns all input
/// pieces, so none can be published, reattached, or partially reused.
pub(super) fn prepare_bootstrap_rebuild(
    attached: BootstrapAttachedUninstalledResources,
) -> Result<
    (BootstrapRebuildCompilerInput, BootstrapRebuildExpectations),
    Box<BootstrapRebuildPreparationFailure>,
> {
    let (lease, owners) = resources::into_bootstrap_rebuild_attached_owners(attached);
    let (source, expected) = into_rebuild_sources(lease);
    prepare_from_parts(owners, source, expected)
}

impl BootstrapRebuildPreparationFailure {
    /// Retry without recovering any source component to sibling modules. This is primarily the
    /// failure-preserving seam for a future in-module admission repair; current V1 validation is
    /// deterministic and therefore retry only succeeds after such a repair changes this module.
    #[allow(dead_code)]
    fn retry(
        self,
    ) -> Result<
        (BootstrapRebuildCompilerInput, BootstrapRebuildExpectations),
        Box<BootstrapRebuildPreparationFailure>,
    > {
        prepare_from_parts(self.owners, self.source, self.expected)
    }
}

impl BootstrapRebuildCompilerInput {
    /// Consume the root-free compiler capability at the one engine-to-execution bridge.
    pub(super) fn into_gpu_parts(self) -> BootstrapRebuildGpuParts {
        BootstrapRebuildGpuParts {
            owners: self.owners,
            source: self.semantic.source,
            proof: self.prepared,
        }
    }
}

impl BootstrapRebuildPreparedProofLayout {
    pub(super) fn exact_output_count(&self) -> u32 {
        self.exact_output_count
    }

    pub(super) fn is_v1_single_table_int4_contract(&self) -> bool {
        if self.exact_output_count != 135 || self.slots.len() != 135 {
            return false;
        }
        let Some(BootstrapRebuildOutputSlot::ColumnShape {
            table_id,
            column_id,
        }) = self.slots.first()
        else {
            return false;
        };
        if self.slots.get(1)
            != Some(&BootstrapRebuildOutputSlot::TypedValueVector {
                table_id: *table_id,
                column_id: *column_id,
            })
            || self.slots.get(2)
                != Some(&BootstrapRebuildOutputSlot::CurrentRowLeaves {
                    table_id: *table_id,
                })
        {
            return false;
        }
        for depth in 0..=RADIX_DEPTH_COUNT {
            if self.slots.get(3 + usize::from(depth))
                != Some(&BootstrapRebuildOutputSlot::RowMapDepth {
                    table_id: *table_id,
                    depth,
                })
                || self.slots.get(69 + usize::from(depth))
                    != Some(&BootstrapRebuildOutputSlot::DatabaseMapDepth { depth })
            {
                return false;
            }
        }
        self.slots.get(68)
            == Some(&BootstrapRebuildOutputSlot::TableRoot {
                table_id: *table_id,
            })
            && self.slots.get(134) == Some(&BootstrapRebuildOutputSlot::DatabaseRoot)
    }
}

fn into_rebuild_sources(
    lease: BootstrapMaterializationLease,
) -> (
    BootstrapRebuildRootFreeSource,
    BootstrapRebuildExpectationSource,
) {
    lease.into_bootstrap_rebuild_sources()
}

fn prepare_from_parts(
    owners: BootstrapRebuildAttachedOwners,
    source: BootstrapRebuildRootFreeSource,
    expected: BootstrapRebuildExpectationSource,
) -> Result<
    (BootstrapRebuildCompilerInput, BootstrapRebuildExpectations),
    Box<BootstrapRebuildPreparationFailure>,
> {
    let prepared = match prepare_proof_layout(&owners, &source) {
        Ok(prepared) => prepared,
        Err(error) => {
            return Err(Box::new(BootstrapRebuildPreparationFailure {
                error,
                owners,
                source,
                expected,
            }))
        }
    };
    let runtime = owners.runtime_identity();
    Ok((
        BootstrapRebuildCompilerInput {
            owners,
            semantic: BootstrapRebuildSemanticLayout { source, runtime },
            prepared,
        },
        BootstrapRebuildExpectations { expected },
    ))
}

fn prepare_proof_layout(
    owners: &BootstrapRebuildAttachedOwners,
    source: &BootstrapRebuildRootFreeSource,
) -> Result<BootstrapRebuildPreparedProofLayout, DataGenerationError> {
    if source.root_format != RootFormatVersion::V1 {
        return Err(DataGenerationError::UnsupportedRootFormat(
            source.root_format.get(),
        ));
    }
    if owners.attachment_count() != source.resources.len() {
        return Err(DataGenerationError::PredecessorMismatch(
            "bootstrap rebuild attached resource coverage",
        ));
    }

    let tables = validate_semantic_tables(source)?;
    validate_resource_groups(source)?;
    validate_root_free_resources(source, &tables)?;
    build_prepared_proof_layout(source)
}

fn validate_semantic_tables(
    source: &BootstrapRebuildRootFreeSource,
) -> Result<BootstrapRebuildSemanticTables, DataGenerationError> {
    let mut table_rows = BTreeMap::new();
    let mut column_types = BTreeMap::new();
    let mut index_keys = BTreeMap::new();
    let mut previous_table = None;
    for table in source.tables.iter() {
        if previous_table.is_some_and(|prior| prior >= table.table_id) {
            return Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap rebuild table order",
            ));
        }
        previous_table = Some(table.table_id);
        if table_rows
            .insert(table.table_id, table.logical_row_count)
            .is_some()
        {
            return Err(DataGenerationError::Invalid(
                "duplicate bootstrap rebuild table",
            ));
        }

        let mut columns = BTreeMap::new();
        let mut previous_column = None;
        for (position, column) in table.columns.iter().enumerate() {
            let ordinal =
                u16::try_from(position).map_err(|_| DataGenerationError::CountOverflow)?;
            if column.ordinal != ordinal
                || previous_column.is_some_and(|prior| prior >= column.column_id)
                || column.attnum <= 0
                || column.declared_type_oid != column.sql_type.postgres_oid()
                || column.signed_type_size != column.sql_type.type_size()
            {
                return Err(DataGenerationError::NonCanonicalOrder(
                    "bootstrap rebuild table column order",
                ));
            }
            previous_column = Some(column.column_id);
            if columns.insert(column.column_id, column.sql_type).is_some() {
                return Err(DataGenerationError::Invalid(
                    "duplicate bootstrap rebuild table column",
                ));
            }
        }

        let mut previous_index = None;
        for index in table.indexes.iter() {
            if previous_index.is_some_and(|prior| prior >= index.index_id) {
                return Err(DataGenerationError::NonCanonicalOrder(
                    "bootstrap rebuild index order",
                ));
            }
            previous_index = Some(index.index_id);
            if index.keys.is_empty() {
                return Err(DataGenerationError::Missing("bootstrap rebuild index keys"));
            }
            let mut keys = Vec::with_capacity(index.keys.len());
            for (position, key) in index.keys.iter().enumerate() {
                let ordinal =
                    u16::try_from(position).map_err(|_| DataGenerationError::CountOverflow)?;
                if key.key_ordinal != ordinal {
                    return Err(DataGenerationError::NonCanonicalOrder(
                        "bootstrap rebuild index key order",
                    ));
                }
                let sql_type = columns
                    .get(&key.column_id)
                    .ok_or(DataGenerationError::Missing(
                        "bootstrap rebuild index key column enrollment",
                    ))?;
                if key.sql_type != *sql_type {
                    return Err(DataGenerationError::PredecessorMismatch(
                        "bootstrap rebuild index key column type",
                    ));
                }
                keys.push((key.key_ordinal, key.column_id, key.sql_type));
            }
            if index_keys
                .insert((table.table_id, index.index_id), keys)
                .is_some()
            {
                return Err(DataGenerationError::Invalid(
                    "duplicate bootstrap rebuild index",
                ));
            }
        }
        if column_types.insert(table.table_id, columns).is_some() {
            return Err(DataGenerationError::Invalid(
                "duplicate bootstrap rebuild table columns",
            ));
        }
    }
    Ok(BootstrapRebuildSemanticTables {
        rows: table_rows,
        columns: column_types,
        index_keys,
    })
}

fn validate_resource_groups(
    source: &BootstrapRebuildRootFreeSource,
) -> Result<(), DataGenerationError> {
    let mut database_resources = 0_u32;
    let mut status_resources = 0_u32;
    let mut resource_ids = BTreeSet::new();
    let mut descriptors = BTreeSet::new();
    let mut start = 0;
    while start < source.resources.len() {
        let first = &source.resources[start];
        if first.member_count == 0 {
            return Err(DataGenerationError::Invalid(
                "bootstrap rebuild resource member count",
            ));
        }
        let mut end = start + 1;
        while end < source.resources.len()
            && source.resources[end].kind == first.kind
            && source.resources[end].owner == first.owner
        {
            end += 1;
        }
        let count = u32::try_from(end - start).map_err(|_| DataGenerationError::CountOverflow)?;
        if first.member_count != count {
            return Err(DataGenerationError::Invalid(
                "bootstrap rebuild resource member count",
            ));
        }
        for (position, resource) in source.resources[start..end].iter().enumerate() {
            let ordinal =
                u32::try_from(position).map_err(|_| DataGenerationError::CountOverflow)?;
            if resource.member_count != count || resource.ordinal != ordinal {
                return Err(DataGenerationError::NonCanonicalOrder(
                    "bootstrap rebuild resource group order",
                ));
            }
            if resource.resource_id == 0 || !resource_ids.insert(resource.resource_id) {
                return Err(DataGenerationError::Invalid(
                    "bootstrap rebuild resource identity",
                ));
            }
            if !descriptors.insert(resource.layout.descriptor_id) {
                return Err(DataGenerationError::Invalid(
                    "duplicate bootstrap rebuild layout descriptor",
                ));
            }
            match (resource.kind, resource.owner) {
                (
                    BootstrapResourceLedgerKind::DatabaseManifest,
                    BootstrapResourceLedgerOwner::Database,
                ) => {
                    database_resources = database_resources
                        .checked_add(1)
                        .ok_or(DataGenerationError::CountOverflow)?;
                }
                (BootstrapResourceLedgerKind::StatusView, BootstrapResourceLedgerOwner::Status) => {
                    status_resources = status_resources
                        .checked_add(1)
                        .ok_or(DataGenerationError::CountOverflow)?;
                }
                _ => {}
            }
        }
        start = end;
    }
    if database_resources != 1 || status_resources != 1 {
        return Err(DataGenerationError::Missing(
            "bootstrap rebuild base resource coverage",
        ));
    }
    Ok(())
}

fn validate_root_free_resources(
    source: &BootstrapRebuildRootFreeSource,
    tables: &BootstrapRebuildSemanticTables,
) -> Result<(), DataGenerationError> {
    let mut table_payloads =
        BTreeMap::<StableTableId, Vec<&BootstrapRebuildRootFreeResource>>::new();
    let mut payload_tables = BTreeSet::new();
    let mut payload_indexes = BTreeSet::new();

    for resource in source.resources.iter() {
        validate_resource_provenance(source, resource)?;
        validate_resource_layout(resource)?;
        match (resource.kind, resource.owner) {
            (
                BootstrapResourceLedgerKind::DatabaseManifest,
                BootstrapResourceLedgerOwner::Database,
            )
            | (BootstrapResourceLedgerKind::StatusView, BootstrapResourceLedgerOwner::Status) => {
                validate_base_resource(resource)?;
            }
            (
                BootstrapResourceLedgerKind::TablePayload | BootstrapResourceLedgerKind::Sidecar,
                BootstrapResourceLedgerOwner::Table(table_id),
            ) => {
                let rows = tables
                    .rows
                    .get(&table_id)
                    .ok_or(DataGenerationError::Missing(
                        "bootstrap rebuild table resource enrollment",
                    ))?;
                let columns = tables
                    .columns
                    .get(&table_id)
                    .ok_or(DataGenerationError::Missing(
                        "bootstrap rebuild table column enrollment",
                    ))?;
                validate_typed_resource_geometry(resource, *rows)?;
                validate_table_roles(
                    resource,
                    columns,
                    resource.kind == BootstrapResourceLedgerKind::TablePayload,
                )?;
                if resource.kind == BootstrapResourceLedgerKind::TablePayload {
                    payload_tables.insert(table_id);
                    table_payloads.entry(table_id).or_default().push(resource);
                }
            }
            (
                BootstrapResourceLedgerKind::IndexPayload | BootstrapResourceLedgerKind::Sidecar,
                BootstrapResourceLedgerOwner::Index { table_id, index_id },
            ) => {
                let rows = tables
                    .rows
                    .get(&table_id)
                    .ok_or(DataGenerationError::Missing(
                        "bootstrap rebuild index table enrollment",
                    ))?;
                let keys = tables.index_keys.get(&(table_id, index_id)).ok_or(
                    DataGenerationError::Missing("bootstrap rebuild index enrollment"),
                )?;
                validate_typed_resource_geometry(resource, *rows)?;
                validate_index_roles(resource, keys)?;
                if resource.kind == BootstrapResourceLedgerKind::IndexPayload {
                    payload_indexes.insert((table_id, index_id));
                }
            }
            _ => {
                return Err(DataGenerationError::Invalid(
                    "bootstrap rebuild resource owner",
                ))
            }
        }
    }
    let expected_tables = tables.rows.keys().copied().collect::<BTreeSet<_>>();
    let expected_indexes = tables.index_keys.keys().copied().collect::<BTreeSet<_>>();
    if payload_tables != expected_tables {
        return Err(DataGenerationError::Missing(
            "bootstrap rebuild table payload coverage",
        ));
    }
    if payload_indexes != expected_indexes {
        return Err(DataGenerationError::Missing(
            "bootstrap rebuild index payload coverage",
        ));
    }
    for (table_id, resources) in table_payloads {
        validate_table_payload_partition(table_id, &resources, tables.rows[&table_id])?;
    }
    Ok(())
}

fn validate_resource_provenance(
    source: &BootstrapRebuildRootFreeSource,
    resource: &BootstrapRebuildRootFreeResource,
) -> Result<(), DataGenerationError> {
    if resource.database_id != source.database_id
        || resource.root_format != source.root_format
        || resource.covered_through != source.covered_through
    {
        return Err(DataGenerationError::PredecessorMismatch(
            "bootstrap rebuild resource provenance",
        ));
    }
    resource
        .byte_offset
        .checked_add(resource.byte_len.get())
        .ok_or(DataGenerationError::CountOverflow)?;
    if resource.tier == BootstrapResourceStorageTier::Resident && resource.byte_offset != 0 {
        return Err(DataGenerationError::Invalid(
            "bootstrap rebuild resident resource range",
        ));
    }
    Ok(())
}

fn validate_resource_layout(
    resource: &BootstrapRebuildRootFreeResource,
) -> Result<(), DataGenerationError> {
    if resource.layout.encoding_version != V1_BOOTSTRAP_LAYOUT_ENCODING {
        return Err(DataGenerationError::Invalid(
            "unsupported bootstrap rebuild V1 layout encoding",
        ));
    }
    if resource.layout.roles.is_empty() {
        return Err(DataGenerationError::Missing(
            "bootstrap rebuild layout roles",
        ));
    }
    resource
        .layout
        .row_start
        .checked_add(resource.layout.row_count)
        .ok_or(DataGenerationError::CountOverflow)?;
    let mut prior_end = 0_u64;
    let mut expected_index_ordinal = 0_u16;
    for (position, role) in resource.layout.roles.iter().enumerate() {
        let ordinal = u16::try_from(position).map_err(|_| DataGenerationError::CountOverflow)?;
        if role.ordinal != ordinal {
            return Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap rebuild layout role order",
            ));
        }
        let end = role
            .byte_offset
            .checked_add(role.byte_len)
            .ok_or(DataGenerationError::CountOverflow)?;
        if end > resource.byte_len.get() || role.byte_offset < prior_end {
            return Err(DataGenerationError::Invalid(
                "bootstrap rebuild layout role range",
            ));
        }
        prior_end = end;
        validate_role_shape_and_storage(
            role,
            resource.layout.row_count,
            &mut expected_index_ordinal,
        )?;
    }
    Ok(())
}

fn validate_role_shape_and_storage(
    role: &BootstrapRebuildRootFreeRole,
    row_count: u64,
    expected_index_ordinal: &mut u16,
) -> Result<(), DataGenerationError> {
    match role.kind {
        BootstrapPhysicalLayoutRoleKind::Opaque => {
            if role.column_id.is_some() || role.sql_type.is_some() || role.key_ordinal.is_some() {
                return Err(DataGenerationError::Invalid(
                    "bootstrap rebuild opaque role",
                ));
            }
            if role.storage_type != BootstrapPhysicalStorageType::Bytes
                || role.byte_stride.get() != 1
            {
                return Err(DataGenerationError::Invalid(
                    "bootstrap rebuild opaque storage",
                ));
            }
        }
        BootstrapPhysicalLayoutRoleKind::StableRowId => {
            if role.column_id.is_some() || role.sql_type.is_some() || role.key_ordinal.is_some() {
                return Err(DataGenerationError::Invalid(
                    "bootstrap rebuild stable-row role",
                ));
            }
            validate_storage_geometry(
                role,
                BootstrapPhysicalStorageType::Int8,
                8,
                checked_bytes(row_count, 8)?,
            )?;
        }
        BootstrapPhysicalLayoutRoleKind::Validity => {
            if role.column_id.is_none() || role.sql_type.is_none() || role.key_ordinal.is_some() {
                return Err(DataGenerationError::Invalid(
                    "bootstrap rebuild validity role",
                ));
            }
            validate_storage_geometry(
                role,
                BootstrapPhysicalStorageType::Bit,
                1,
                bitmap_bytes(row_count)?,
            )?;
        }
        BootstrapPhysicalLayoutRoleKind::Value => {
            if role.column_id.is_none() || role.sql_type.is_none() || role.key_ordinal.is_some() {
                return Err(DataGenerationError::Invalid("bootstrap rebuild value role"));
            }
            validate_value_storage_geometry(role, row_count)?;
        }
        BootstrapPhysicalLayoutRoleKind::CreatedBy | BootstrapPhysicalLayoutRoleKind::DeletedBy => {
            if role.column_id.is_some() || role.sql_type.is_some() || role.key_ordinal.is_some() {
                return Err(DataGenerationError::Invalid("bootstrap rebuild MVCC role"));
            }
            validate_storage_geometry(
                role,
                BootstrapPhysicalStorageType::Int8,
                8,
                checked_bytes(row_count, 8)?,
            )?;
        }
        BootstrapPhysicalLayoutRoleKind::IndexKey => {
            if role.column_id.is_none()
                || role.sql_type.is_none()
                || role.key_ordinal != Some(*expected_index_ordinal)
            {
                return Err(DataGenerationError::Invalid(
                    "bootstrap rebuild index-key role",
                ));
            }
            *expected_index_ordinal = expected_index_ordinal
                .checked_add(1)
                .ok_or(DataGenerationError::CountOverflow)?;
            validate_value_storage_geometry(role, row_count)?;
        }
    }
    Ok(())
}

fn validate_value_storage_geometry(
    role: &BootstrapRebuildRootFreeRole,
    row_count: u64,
) -> Result<(), DataGenerationError> {
    let sql_type = role
        .sql_type
        .ok_or(DataGenerationError::Invalid("bootstrap rebuild value role"))?;
    let (storage, stride) = fixed_value_storage(sql_type)?;
    let byte_len = if storage == BootstrapPhysicalStorageType::Bit {
        bitmap_bytes(row_count)?
    } else {
        checked_bytes(row_count, stride)?
    };
    validate_storage_geometry(role, storage, stride, byte_len)
}

fn fixed_value_storage(
    sql_type: SqlType,
) -> Result<(BootstrapPhysicalStorageType, u64), DataGenerationError> {
    Ok(match sql_type {
        SqlType::Int2 | SqlType::Int4 | SqlType::Date => (BootstrapPhysicalStorageType::Int4, 4),
        SqlType::Int8 | SqlType::Timestamp => (BootstrapPhysicalStorageType::Int8, 8),
        SqlType::Bool => (BootstrapPhysicalStorageType::Bit, 1),
        SqlType::Numeric { .. } | SqlType::Uuid => (BootstrapPhysicalStorageType::Int128, 16),
        SqlType::Text => {
            return Err(DataGenerationError::Invalid(
                "unsupported bootstrap V1 variable-width layout",
            ))
        }
    })
}

fn validate_storage_geometry(
    role: &BootstrapRebuildRootFreeRole,
    storage: BootstrapPhysicalStorageType,
    stride: u64,
    byte_len: u64,
) -> Result<(), DataGenerationError> {
    if role.storage_type != storage || role.byte_stride.get() != stride || role.byte_len != byte_len
    {
        return Err(DataGenerationError::Invalid(
            "bootstrap rebuild storage geometry",
        ));
    }
    Ok(())
}

fn bitmap_bytes(row_count: u64) -> Result<u64, DataGenerationError> {
    row_count
        .checked_add(31)
        .ok_or(DataGenerationError::CountOverflow)?
        .checked_div(32)
        .ok_or(DataGenerationError::CountOverflow)?
        .checked_mul(4)
        .ok_or(DataGenerationError::CountOverflow)
}

fn checked_bytes(row_count: u64, stride: u64) -> Result<u64, DataGenerationError> {
    row_count
        .checked_mul(stride)
        .ok_or(DataGenerationError::CountOverflow)
}

fn validate_base_resource(
    resource: &BootstrapRebuildRootFreeResource,
) -> Result<(), DataGenerationError> {
    if resource.layout.row_start != 0
        || resource.layout.row_count != 0
        || resource.layout.roles.len() != 1
        || resource.layout.roles[0].kind != BootstrapPhysicalLayoutRoleKind::Opaque
        || resource.layout.roles[0].byte_offset != 0
        || resource.layout.roles[0].byte_len != resource.byte_len.get()
    {
        return Err(DataGenerationError::Invalid(
            "bootstrap rebuild opaque layout grammar",
        ));
    }
    Ok(())
}

fn validate_typed_resource_geometry(
    resource: &BootstrapRebuildRootFreeResource,
    table_rows: u64,
) -> Result<(), DataGenerationError> {
    let row_end = resource
        .layout
        .row_start
        .checked_add(resource.layout.row_count)
        .ok_or(DataGenerationError::CountOverflow)?;
    if row_end > table_rows {
        return Err(DataGenerationError::Invalid(
            "bootstrap rebuild typed resource row bounds",
        ));
    }
    if table_rows == 0
        && (resource.byte_len.get() != V1_EMPTY_TYPED_RESOURCE_SENTINEL_BYTES
            || resource.layout.row_start != 0
            || resource.layout.row_count != 0
            || resource
                .layout
                .roles
                .iter()
                .any(|role| role.byte_offset != 0 || role.byte_len != 0))
    {
        return Err(DataGenerationError::Invalid(
            "bootstrap rebuild empty typed resource V1 geometry",
        ));
    }
    Ok(())
}

fn validate_table_roles(
    resource: &BootstrapRebuildRootFreeResource,
    expected_columns: &BTreeMap<StableColumnId, SqlType>,
    require_mvcc: bool,
) -> Result<(), DataGenerationError> {
    let mut validity = BTreeMap::new();
    let mut values = BTreeMap::new();
    let mut stable_row_ids = 0_u16;
    let mut created = 0_u16;
    let mut deleted = 0_u16;
    let mut previous_key = None;
    for role in resource.layout.roles.iter() {
        let role_key = match role.kind {
            BootstrapPhysicalLayoutRoleKind::StableRowId => {
                stable_row_ids = stable_row_ids
                    .checked_add(1)
                    .ok_or(DataGenerationError::CountOverflow)?;
                (0, 0, 0)
            }
            BootstrapPhysicalLayoutRoleKind::Validity => {
                let column_id = role.column_id.ok_or(DataGenerationError::Invalid(
                    "bootstrap rebuild table column role",
                ))?;
                let sql_type = role.sql_type.ok_or(DataGenerationError::Invalid(
                    "bootstrap rebuild table column type",
                ))?;
                if validity.insert(column_id, sql_type).is_some() {
                    return Err(DataGenerationError::Invalid(
                        "duplicate bootstrap rebuild table validity role",
                    ));
                }
                (1, column_id.get(), 0)
            }
            BootstrapPhysicalLayoutRoleKind::Value => {
                let column_id = role.column_id.ok_or(DataGenerationError::Invalid(
                    "bootstrap rebuild table column role",
                ))?;
                let sql_type = role.sql_type.ok_or(DataGenerationError::Invalid(
                    "bootstrap rebuild table column type",
                ))?;
                if values.insert(column_id, sql_type).is_some() {
                    return Err(DataGenerationError::Invalid(
                        "duplicate bootstrap rebuild table value role",
                    ));
                }
                (1, column_id.get(), 1)
            }
            BootstrapPhysicalLayoutRoleKind::CreatedBy => {
                created = created
                    .checked_add(1)
                    .ok_or(DataGenerationError::CountOverflow)?;
                (2, 0, 0)
            }
            BootstrapPhysicalLayoutRoleKind::DeletedBy => {
                deleted = deleted
                    .checked_add(1)
                    .ok_or(DataGenerationError::CountOverflow)?;
                (3, 0, 0)
            }
            BootstrapPhysicalLayoutRoleKind::Opaque | BootstrapPhysicalLayoutRoleKind::IndexKey => {
                return Err(DataGenerationError::Invalid(
                    "bootstrap rebuild table role grammar",
                ))
            }
        };
        if previous_key.is_some_and(|prior| prior >= role_key) {
            return Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap rebuild table role order",
            ));
        }
        previous_key = Some(role_key);
    }
    if validity != values {
        return Err(DataGenerationError::Missing(
            "bootstrap rebuild table value validity pairing",
        ));
    }
    for (column_id, sql_type) in &values {
        let expected_type = expected_columns
            .get(column_id)
            .ok_or(DataGenerationError::Missing(
                "bootstrap rebuild table column enrollment",
            ))?;
        if sql_type != expected_type {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap rebuild table column type",
            ));
        }
    }
    if require_mvcc && values != *expected_columns {
        return Err(DataGenerationError::Missing(
            "bootstrap rebuild table column coverage",
        ));
    }
    if require_mvcc && stable_row_ids != 1 {
        return Err(DataGenerationError::Missing(
            "bootstrap rebuild stable row coverage",
        ));
    }
    if created > 1 || deleted > 1 || created != deleted || (require_mvcc && created != 1) {
        return Err(DataGenerationError::Invalid(
            "bootstrap rebuild table MVCC pairing",
        ));
    }
    Ok(())
}

fn validate_index_roles(
    resource: &BootstrapRebuildRootFreeResource,
    expected_keys: &[BootstrapRebuildIndexKey],
) -> Result<(), DataGenerationError> {
    if resource.layout.roles.len() != expected_keys.len() {
        return Err(DataGenerationError::Missing(
            "bootstrap rebuild index key coverage",
        ));
    }
    for (role, (key_ordinal, column_id, sql_type)) in
        resource.layout.roles.iter().zip(expected_keys)
    {
        let (storage_type, byte_stride) = fixed_value_storage(*sql_type)?;
        if role.kind != BootstrapPhysicalLayoutRoleKind::IndexKey
            || role.key_ordinal != Some(*key_ordinal)
            || role.column_id != Some(*column_id)
            || role.sql_type != Some(*sql_type)
            || role.storage_type != storage_type
            || role.byte_stride.get() != byte_stride
        {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap rebuild index key type",
            ));
        }
    }
    Ok(())
}

fn validate_table_payload_partition(
    table_id: StableTableId,
    resources: &[&BootstrapRebuildRootFreeResource],
    table_rows: u64,
) -> Result<(), DataGenerationError> {
    if table_rows == 0 {
        if resources.len() != 1 {
            return Err(DataGenerationError::Invalid(
                "bootstrap rebuild empty table payload multiplicity",
            ));
        }
        return Ok(());
    }
    let mut next_row = 0_u64;
    for resource in resources {
        if resource.layout.row_count == 0 || resource.layout.row_start != next_row {
            return Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap rebuild table payload row coverage",
            ));
        }
        next_row = next_row
            .checked_add(resource.layout.row_count)
            .ok_or(DataGenerationError::CountOverflow)?;
    }
    if next_row != table_rows {
        return Err(DataGenerationError::Missing(
            "bootstrap rebuild table payload row coverage",
        ));
    }
    let _ = table_id;
    Ok(())
}

fn build_prepared_proof_layout(
    source: &BootstrapRebuildRootFreeSource,
) -> Result<BootstrapRebuildPreparedProofLayout, DataGenerationError> {
    let mut slots = BTreeSet::new();
    let mut expected_count = 0_usize;
    for table in source.tables.iter() {
        for column in table.columns.iter() {
            insert_output_slot(
                &mut slots,
                &mut expected_count,
                BootstrapRebuildOutputSlot::ColumnShape {
                    table_id: table.table_id,
                    column_id: column.column_id,
                },
            )?;
            insert_output_slot(
                &mut slots,
                &mut expected_count,
                BootstrapRebuildOutputSlot::TypedValueVector {
                    table_id: table.table_id,
                    column_id: column.column_id,
                },
            )?;
        }
        insert_output_slot(
            &mut slots,
            &mut expected_count,
            BootstrapRebuildOutputSlot::CurrentRowLeaves {
                table_id: table.table_id,
            },
        )?;
        for depth in 0..=RADIX_DEPTH_COUNT {
            insert_output_slot(
                &mut slots,
                &mut expected_count,
                BootstrapRebuildOutputSlot::RowMapDepth {
                    table_id: table.table_id,
                    depth,
                },
            )?;
        }
        insert_output_slot(
            &mut slots,
            &mut expected_count,
            BootstrapRebuildOutputSlot::TableRoot {
                table_id: table.table_id,
            },
        )?;
    }
    for depth in 0..=RADIX_DEPTH_COUNT {
        insert_output_slot(
            &mut slots,
            &mut expected_count,
            BootstrapRebuildOutputSlot::DatabaseMapDepth { depth },
        )?;
    }
    insert_output_slot(
        &mut slots,
        &mut expected_count,
        BootstrapRebuildOutputSlot::DatabaseRoot,
    )?;
    if slots.len() != expected_count {
        return Err(DataGenerationError::Invalid(
            "duplicate bootstrap rebuild proof slot",
        ));
    }
    let exact_output_count =
        u32::try_from(expected_count).map_err(|_| DataGenerationError::CountOverflow)?;
    let prepared = BootstrapRebuildPreparedProofLayout {
        exact_output_count,
        slots: slots.into_iter().collect::<Vec<_>>().into_boxed_slice(),
    };
    prepared.validate_observed_bindings(&prepared.slots)?;
    Ok(prepared)
}

fn insert_output_slot(
    slots: &mut BTreeSet<BootstrapRebuildOutputSlot>,
    expected_count: &mut usize,
    slot: BootstrapRebuildOutputSlot,
) -> Result<(), DataGenerationError> {
    *expected_count = expected_count
        .checked_add(1)
        .ok_or(DataGenerationError::CountOverflow)?;
    if *expected_count > MAX_BOOTSTRAP_REBUILD_OUTPUT_SLOTS {
        return Err(DataGenerationError::Invalid(
            "bootstrap rebuild output slot bound",
        ));
    }
    if !slots.insert(slot) {
        return Err(DataGenerationError::Invalid(
            "duplicate bootstrap rebuild proof slot",
        ));
    }
    Ok(())
}

impl BootstrapRebuildPreparedProofLayout {
    fn validate_observed_bindings(
        &self,
        observed: &[BootstrapRebuildOutputSlot],
    ) -> Result<(), DataGenerationError> {
        if usize::try_from(self.exact_output_count)
            .map_err(|_| DataGenerationError::CountOverflow)?
            != self.slots.len()
            || self.slots.len() > MAX_BOOTSTRAP_REBUILD_OUTPUT_SLOTS
        {
            return Err(DataGenerationError::Invalid(
                "bootstrap rebuild exact output count",
            ));
        }
        if observed.len() < self.slots.len() {
            return Err(DataGenerationError::Missing("bootstrap rebuild proof slot"));
        }
        if observed.len() > self.slots.len() {
            return Err(DataGenerationError::Unexpected(
                "bootstrap rebuild proof slot",
            ));
        }
        let mut previous = None;
        for slot in observed {
            if let Some(prior) = previous {
                if prior == slot {
                    return Err(DataGenerationError::Invalid(
                        "duplicate bootstrap rebuild proof slot",
                    ));
                }
                if prior > slot {
                    return Err(DataGenerationError::NonCanonicalOrder(
                        "bootstrap rebuild proof slot order",
                    ));
                }
            }
            previous = Some(slot);
        }
        if observed != self.slots.as_ref() {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap rebuild proof binding",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attached_source_becomes_root_free_compiler_and_comparator_capabilities() {
        let attached = resources::attached_resources_for_bootstrap_rebuild_test();
        let (compiler, expectations) =
            prepare_bootstrap_rebuild(attached).unwrap_or_else(|_| panic!("prepared rebuild"));

        assert_eq!(
            compiler.owners.attachment_count(),
            compiler.semantic.source.resources.len()
        );
        assert_eq!(compiler.semantic.source.tables.len(), 1);
        let table = &compiler.semantic.source.tables[0];
        assert_eq!(table.columns.len(), 1);
        assert_eq!(table.columns[0].ordinal, 0);
        assert_eq!(table.columns[0].column_id.get(), 7);
        assert_eq!(table.columns[0].sql_type, SqlType::Int4);
        assert_eq!(table.columns[0].attnum, 1);
        assert_eq!(
            table.columns[0].declared_type_oid,
            SqlType::Int4.postgres_oid()
        );
        assert_eq!(table.columns[0].signed_type_size, SqlType::Int4.type_size());
        assert!(compiler.prepared.slots.iter().any(|slot| {
            matches!(
                slot,
                BootstrapRebuildOutputSlot::ColumnShape { column_id, .. } if column_id.get() == 7
            )
        }));
        assert!(compiler.prepared.slots.iter().any(|slot| {
            matches!(
                slot,
                BootstrapRebuildOutputSlot::TypedValueVector { column_id, .. }
                    if column_id.get() == 7
            )
        }));
        assert_eq!(
            compiler.prepared.exact_output_count as usize,
            compiler.prepared.slots.len()
        );
        assert_eq!(compiler.prepared.slots.len(), 135);
        let _ = expectations;
    }

    #[test]
    fn proof_contract_rejects_missing_duplicate_and_out_of_order_bindings() {
        let table_id = StableTableId::new(7).expect("table id");
        let contract = BootstrapRebuildPreparedProofLayout {
            exact_output_count: 2,
            slots: vec![
                BootstrapRebuildOutputSlot::RowMapDepth { table_id, depth: 0 },
                BootstrapRebuildOutputSlot::RowMapDepth { table_id, depth: 1 },
            ]
            .into_boxed_slice(),
        };
        let missing = [BootstrapRebuildOutputSlot::RowMapDepth { table_id, depth: 0 }];
        assert_eq!(
            contract.validate_observed_bindings(&missing),
            Err(DataGenerationError::Missing("bootstrap rebuild proof slot"))
        );
        let duplicate = [
            BootstrapRebuildOutputSlot::RowMapDepth { table_id, depth: 0 },
            BootstrapRebuildOutputSlot::RowMapDepth { table_id, depth: 0 },
        ];
        assert_eq!(
            contract.validate_observed_bindings(&duplicate),
            Err(DataGenerationError::Invalid(
                "duplicate bootstrap rebuild proof slot"
            ))
        );
        let out_of_order = [
            BootstrapRebuildOutputSlot::RowMapDepth { table_id, depth: 1 },
            BootstrapRebuildOutputSlot::RowMapDepth { table_id, depth: 0 },
        ];
        assert_eq!(
            contract.validate_observed_bindings(&out_of_order),
            Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap rebuild proof slot order"
            ))
        );
    }

    #[test]
    fn coherent_table_role_reordering_is_rejected_before_compiler_input() {
        let attached = resources::attached_resources_for_bootstrap_rebuild_test();
        let (lease, owners) = resources::into_bootstrap_rebuild_attached_owners(attached);
        let (mut source, expected) = into_rebuild_sources(lease);
        let roles = &mut source.resources[2].layout.roles;
        roles.swap(1, 2);
        roles[1].ordinal = 1;
        roles[2].ordinal = 2;
        roles[1].byte_offset = 8;
        roles[2].byte_offset = 12;

        match prepare_from_parts(owners, source, expected) {
            Ok(_) => panic!("role substitution must fail"),
            Err(failure) => assert_eq!(
                failure.error,
                DataGenerationError::NonCanonicalOrder("bootstrap rebuild table role order")
            ),
        }
    }

    #[test]
    fn table_payload_cannot_omit_both_validity_and_value_roles() {
        let attached = resources::attached_resources_for_bootstrap_rebuild_test();
        let (lease, owners) = resources::into_bootstrap_rebuild_attached_owners(attached);
        let (mut source, expected) = into_rebuild_sources(lease);
        let retained_roles = std::mem::replace(&mut source.resources[2].layout.roles, Box::new([]))
            .into_vec()
            .into_iter()
            .enumerate()
            .filter_map(|(position, role)| (position != 1 && position != 2).then_some(role))
            .enumerate()
            .map(|(ordinal, mut role)| {
                role.ordinal = u16::try_from(ordinal).expect("test role ordinal");
                role
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        source.resources[2].layout.roles = retained_roles;

        match prepare_from_parts(owners, source, expected) {
            Ok(_) => panic!("missing table columns must fail"),
            Err(failure) => assert_eq!(
                failure.error,
                DataGenerationError::Missing("bootstrap rebuild table column coverage")
            ),
        }
    }

    #[test]
    fn table_payload_same_width_sql_type_substitution_is_rejected() {
        let attached = resources::attached_resources_for_bootstrap_rebuild_test();
        let (lease, owners) = resources::into_bootstrap_rebuild_attached_owners(attached);
        let (mut source, expected) = into_rebuild_sources(lease);
        source.resources[2].layout.roles[1].sql_type = Some(SqlType::Date);
        source.resources[2].layout.roles[2].sql_type = Some(SqlType::Date);

        match prepare_from_parts(owners, source, expected) {
            Ok(_) => panic!("same-width table SQL type substitution must fail"),
            Err(failure) => assert_eq!(
                failure.error,
                DataGenerationError::PredecessorMismatch("bootstrap rebuild table column type")
            ),
        }
    }

    #[test]
    fn index_key_same_width_sql_storage_substitution_is_rejected() {
        let attached = resources::attached_resources_for_bootstrap_rebuild_test();
        let (lease, owners) = resources::into_bootstrap_rebuild_attached_owners(attached);
        let (mut source, expected) = into_rebuild_sources(lease);
        source.resources[3].layout.roles[0].sql_type = Some(SqlType::Date);

        match prepare_from_parts(owners, source, expected) {
            Ok(_) => panic!("same-width index SQL storage substitution must fail"),
            Err(failure) => assert_eq!(
                failure.error,
                DataGenerationError::PredecessorMismatch("bootstrap rebuild index key type")
            ),
        }
    }

    #[test]
    fn compiler_input_cannot_carry_comparator_source_or_clone_debug_surface() {
        let source = include_str!("bootstrap_rebuild.rs");
        let compiler = source
            .split("pub(super) struct BootstrapRebuildCompilerInput")
            .nth(1)
            .and_then(|section| {
                section
                    .split("pub(super) struct BootstrapRebuildExpectations")
                    .next()
            })
            .expect("compiler source section");
        assert!(!compiler.contains("BootstrapRebuildExpectationSource"));
        assert!(!compiler.contains("expected_"));
        assert!(!compiler.contains("impl Clone for BootstrapRebuildCompilerInput"));
        assert!(!compiler.contains("impl std::fmt::Debug for BootstrapRebuildCompilerInput"));
        assert!(!compiler.contains("fn new("));
    }
}
