//! Immutable, validation-only bootstrap source for one quiescent logical publication cut.
//!
//! This is intentionally a source contract, not a materializer. It owns only authenticated
//! durable facts and physical-resource provenance placeholders after validation; it cannot create
//! GPU roots, acquire a runtime object, or make any source visible.

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroU64,
    sync::Arc,
};

use gpu_db_sql::SqlType;

use super::digest::{
    CatalogIdentity, ColumnShapeRoot, DataGeneration, DatabaseId, DatabaseRoot, IndexGeneration,
    IndexRoot, IndexShapeRoot, RootFormatVersion, StableColumnId, StableIndexId, StableTableId,
    StatusEntryLeafRoot, StatusViewRoot, TableRoot, VisibleNext,
};
use super::status::PublishedStatusEntry;
use super::DataGenerationError;

/// The common durable cut `C` and the only visibility boundary that may describe it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BootstrapDurableCut {
    covered_through: u64,
    visible_next: VisibleNext,
}

impl BootstrapDurableCut {
    fn validate(self) -> Result<(), DataGenerationError> {
        let expected_visible_next = self
            .covered_through
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        if self.visible_next.get() != expected_visible_next {
            return Err(DataGenerationError::Invalid(
                "bootstrap visibility boundary",
            ));
        }
        Ok(())
    }
}

/// The durable catalog identity and the one retained catalog identity must byte-identify exactly.
/// This pair is held only inside the sealed replay witness; the bootstrap source never accepts an
/// independently supplied identity next to a snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BootstrapCatalogPair {
    durable: CatalogIdentity,
    retained: CatalogIdentity,
}

impl BootstrapCatalogPair {
    fn validate(self) -> Result<(), DataGenerationError> {
        if self.durable != self.retained {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap catalog identity",
            ));
        }
        Ok(())
    }
}

/// Relation-shaped stable IDs remain distinct allocator spaces even when their display OIDs
/// happen to match. The tag is part of the persisted migration key, never inferred at bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BootstrapRelationKind {
    Table,
    Index,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BootstrapRelationMigration {
    kind: BootstrapRelationKind,
    display_oid: u32,
    stable_id: u64,
}

/// The legacy column key cannot be collapsed into the relation key. `attnum` is positive for a
/// persisted user column, and the stable column ID remains nonzero after widening at the root
/// boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BootstrapColumnMigration {
    owner_display_table_oid: u32,
    legacy_column_id: u32,
    attnum: i16,
    stable_column_id: u32,
}

/// Complete persisted migration and allocator state at the bootstrap cut. High waters may exceed
/// the observed map because dropped objects are never recycled; they may not be below an observed
/// stable ID. `migration_complete` distinguishes an empty migrated catalog from absent state.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BootstrapStableIdState {
    migration_complete: bool,
    relation_migrations: Vec<BootstrapRelationMigration>,
    column_migrations: Vec<BootstrapColumnMigration>,
    table_high_water: u64,
    index_high_water: u64,
    column_high_water: u32,
    checkpoint_table_high_water: u64,
    checkpoint_index_high_water: u64,
    checkpoint_column_high_water: u32,
}

impl BootstrapStableIdState {
    fn validate(&self) -> Result<(BTreeSet<u64>, BTreeSet<u64>), DataGenerationError> {
        if !self.migration_complete {
            return Err(DataGenerationError::Missing("stable-ID migration state"));
        }
        if self.table_high_water != self.checkpoint_table_high_water
            || self.index_high_water != self.checkpoint_index_high_water
            || self.column_high_water != self.checkpoint_column_high_water
        {
            return Err(DataGenerationError::PredecessorMismatch(
                "stable-ID checkpoint high waters",
            ));
        }
        validate_strictly_ascending(
            self.relation_migrations
                .iter()
                .map(|entry| (entry.kind, entry.display_oid)),
            "stable relation migration keys",
        )?;
        validate_strictly_ascending(
            self.column_migrations.iter().map(|entry| {
                (
                    entry.owner_display_table_oid,
                    entry.legacy_column_id,
                    entry.attnum,
                )
            }),
            "stable column migration keys",
        )?;

        let mut table_ids = BTreeSet::new();
        let mut index_ids = BTreeSet::new();
        for entry in &self.relation_migrations {
            if entry.display_oid == 0 || entry.stable_id == 0 {
                return Err(DataGenerationError::ZeroIdentity(
                    "stable relation migration",
                ));
            }
            match entry.kind {
                BootstrapRelationKind::Table => {
                    if entry.stable_id > self.table_high_water {
                        return Err(DataGenerationError::Invalid("table stable-ID high water"));
                    }
                    if !table_ids.insert(entry.stable_id) {
                        return Err(DataGenerationError::Invalid("duplicate stable table ID"));
                    }
                }
                BootstrapRelationKind::Index => {
                    if entry.stable_id > self.index_high_water {
                        return Err(DataGenerationError::Invalid("index stable-ID high water"));
                    }
                    if !index_ids.insert(entry.stable_id) {
                        return Err(DataGenerationError::Invalid("duplicate stable index ID"));
                    }
                }
            }
        }

        let mut column_ids = BTreeSet::new();
        for entry in &self.column_migrations {
            if entry.owner_display_table_oid == 0
                || entry.legacy_column_id == 0
                || entry.attnum <= 0
                || entry.stable_column_id == 0
            {
                return Err(DataGenerationError::ZeroIdentity("stable column migration"));
            }
            if entry.stable_column_id > self.column_high_water {
                return Err(DataGenerationError::Invalid("column stable-ID high water"));
            }
            if !column_ids.insert(entry.stable_column_id) {
                return Err(DataGenerationError::Invalid("duplicate stable column ID"));
            }
        }
        Ok((table_ids, index_ids))
    }
}

/// The current logical table set at the durable cut. It is intentionally distinct from the
/// complete migration map: stable IDs for dropped tables or indexes stay allocated but cannot
/// receive physical resources in this generation.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BootstrapCurrentColumn {
    owner_display_table_oid: u32,
    legacy_column_id: u32,
    attnum: i16,
    stable_column_id: StableColumnId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BootstrapExpectedIndex {
    display_index_oid: u32,
    index_id: StableIndexId,
    generation: IndexGeneration,
    root: IndexRoot,
    shape_root: IndexShapeRoot,
    key_descriptors: Vec<BootstrapIndexKeyDescriptor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BootstrapIndexKeyDescriptor {
    key_ordinal: u16,
    column_id: StableColumnId,
    shape_root: ColumnShapeRoot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BootstrapResourceIndexKeyBinding {
    key_ordinal: u16,
    column_id: StableColumnId,
    sql_type: SqlType,
    shape_root: ColumnShapeRoot,
}

/// Current logical enrollment and replay-authenticated manifest expectations for one table.
/// Roots are opaque comparison targets for the later GPU materializer, never host inputs to a
/// root calculation.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BootstrapCurrentTable {
    table_id: StableTableId,
    display_table_oid: u32,
    logical_row_count: u64,
    data_generation: DataGeneration,
    expected_table_root: TableRoot,
    enrolled_columns: Vec<BootstrapCurrentColumn>,
    enrolled_indexes: Vec<BootstrapExpectedIndex>,
}

struct CurrentTableValidation {
    table_ids: BTreeSet<StableTableId>,
    enrolled_indexes: BTreeSet<(StableTableId, StableIndexId)>,
    column_types: BTreeMap<StableTableId, BTreeMap<StableColumnId, SqlType>>,
    table_row_counts: BTreeMap<StableTableId, u64>,
    index_keys: BTreeMap<(StableTableId, StableIndexId), Box<[BootstrapResourceIndexKeyBinding]>>,
}

fn validate_current_tables(
    tables: &[BootstrapCurrentTable],
    migrated_table_ids: &BTreeSet<u64>,
    migrated_index_ids: &BTreeSet<u64>,
    stable_ids: &BootstrapStableIdState,
    relation_migrations: &[BootstrapRelationMigration],
    catalog_snapshot: &crate::engine_state::CatalogSnapshot,
) -> Result<CurrentTableValidation, DataGenerationError> {
    if tables.is_empty() {
        return Err(DataGenerationError::Missing("bootstrap current tables"));
    }
    validate_strictly_ascending(
        tables.iter().map(|table| table.table_id),
        "bootstrap current tables",
    )?;

    let migrated_table_bindings = relation_migrations
        .iter()
        .filter(|entry| entry.kind == BootstrapRelationKind::Table)
        .map(|entry| (entry.display_oid, entry.stable_id))
        .collect::<BTreeSet<_>>();
    let migrated_index_bindings = relation_migrations
        .iter()
        .filter(|entry| entry.kind == BootstrapRelationKind::Index)
        .map(|entry| (entry.display_oid, entry.stable_id))
        .collect::<BTreeSet<_>>();
    let catalog_table_oids = catalog_snapshot
        .relational_catalog
        .values()
        .map(|table| table.oid)
        .collect::<BTreeSet<_>>();
    if catalog_table_oids.len() != catalog_snapshot.relational_catalog.len() {
        return Err(DataGenerationError::Invalid("bootstrap catalog table OIDs"));
    }

    let migrated_column_bindings = stable_ids
        .column_migrations
        .iter()
        .map(|entry| {
            (
                entry.owner_display_table_oid,
                entry.legacy_column_id,
                entry.attnum,
                entry.stable_column_id,
            )
        })
        .collect::<BTreeSet<_>>();
    let mut current_table_ids = BTreeSet::new();
    let mut current_table_oids = BTreeSet::new();
    let mut enrolled_indexes = BTreeSet::new();
    let mut column_types = BTreeMap::new();
    let mut table_row_counts = BTreeMap::new();
    let mut index_keys = BTreeMap::new();
    for table in tables {
        if table.display_table_oid == 0
            || !migrated_table_ids.contains(&table.table_id.get())
            || !migrated_table_bindings.contains(&(table.display_table_oid, table.table_id.get()))
        {
            return Err(DataGenerationError::Missing(
                "bootstrap current table mapping",
            ));
        }
        current_table_ids.insert(table.table_id);
        current_table_oids.insert(table.display_table_oid);
        validate_strictly_ascending(
            table
                .enrolled_columns
                .iter()
                .map(|column| column.stable_column_id),
            "bootstrap current stable column IDs",
        )?;
        validate_strictly_ascending(
            table.enrolled_indexes.iter().map(|index| index.index_id),
            "bootstrap enrolled stable index IDs",
        )?;
        let catalog_table = catalog_snapshot
            .relational_catalog
            .values()
            .find(|catalog_table| catalog_table.oid == table.display_table_oid);
        let Some(catalog_table) = catalog_table else {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap catalog table enrollment",
            ));
        };

        let mut enrolled_catalog_columns = BTreeSet::new();
        for column in &table.enrolled_columns {
            let stable_column_id = u32::try_from(column.stable_column_id.get()).map_err(|_| {
                DataGenerationError::Invalid("bootstrap current stable column width")
            })?;
            if column.owner_display_table_oid != table.display_table_oid
                || column.legacy_column_id == 0
                || column.attnum <= 0
                || !migrated_column_bindings.contains(&(
                    column.owner_display_table_oid,
                    column.legacy_column_id,
                    column.attnum,
                    stable_column_id,
                ))
            {
                return Err(DataGenerationError::Missing(
                    "bootstrap current column mapping",
                ));
            }
            enrolled_catalog_columns.insert((
                column.owner_display_table_oid,
                stable_column_id,
                column.attnum,
            ));
        }
        let catalog_columns = catalog_table
            .columns
            .iter()
            .map(|column| (column.table_oid, column.id, column.attnum))
            .collect::<BTreeSet<_>>();
        if catalog_columns.len() != catalog_table.columns.len() {
            return Err(DataGenerationError::Invalid(
                "bootstrap catalog column identities",
            ));
        }
        if enrolled_catalog_columns != catalog_columns {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap catalog column enrollment",
            ));
        }
        let table_column_types = table
            .enrolled_columns
            .iter()
            .map(|enrolled| {
                let catalog = catalog_table
                    .columns
                    .iter()
                    .find(|catalog| {
                        catalog.table_oid == enrolled.owner_display_table_oid
                            && u64::from(catalog.id) == enrolled.stable_column_id.get()
                            && catalog.attnum == enrolled.attnum
                    })
                    .ok_or(DataGenerationError::PredecessorMismatch(
                        "bootstrap catalog column type enrollment",
                    ))?;
                Ok((enrolled.stable_column_id, catalog.ty))
            })
            .collect::<Result<BTreeMap<_, _>, DataGenerationError>>()?;
        let catalog_columns_by_name = catalog_table
            .columns
            .iter()
            .map(|catalog| {
                let stable_column_id = table
                    .enrolled_columns
                    .iter()
                    .find(|enrolled| {
                        enrolled.owner_display_table_oid == catalog.table_oid
                            && u64::from(catalog.id) == enrolled.stable_column_id.get()
                            && enrolled.attnum == catalog.attnum
                    })
                    .map(|enrolled| enrolled.stable_column_id)
                    .ok_or(DataGenerationError::PredecessorMismatch(
                        "bootstrap catalog column name enrollment",
                    ))?;
                Ok((catalog.name.as_str(), (stable_column_id, catalog.ty)))
            })
            .collect::<Result<BTreeMap<_, _>, DataGenerationError>>()?;
        if catalog_columns_by_name.len() != catalog_table.columns.len() {
            return Err(DataGenerationError::Invalid(
                "bootstrap catalog column names",
            ));
        }

        let catalog_indexes = catalog_table
            .indexes
            .iter()
            .map(|index| index.oid)
            .collect::<BTreeSet<_>>();
        if catalog_indexes.len() != catalog_table.indexes.len() {
            return Err(DataGenerationError::Invalid("bootstrap catalog index OIDs"));
        }
        let mut enrolled_index_oids = BTreeSet::new();
        for index in &table.enrolled_indexes {
            if index.display_index_oid == 0
                || !migrated_index_ids.contains(&index.index_id.get())
                || !migrated_index_bindings
                    .contains(&(index.display_index_oid, index.index_id.get()))
            {
                return Err(DataGenerationError::Missing(
                    "bootstrap enrolled index mapping",
                ));
            }
            let catalog_index = catalog_table
                .indexes
                .iter()
                .find(|catalog| catalog.oid == index.display_index_oid)
                .ok_or(DataGenerationError::PredecessorMismatch(
                    "bootstrap catalog index enrollment",
                ))?;
            validate_index_key_descriptors(
                &index.key_descriptors,
                catalog_index,
                &catalog_columns_by_name,
            )?;
            let resource_key_bindings = index
                .key_descriptors
                .iter()
                .map(|descriptor| {
                    let sql_type = table_column_types
                        .get(&descriptor.column_id)
                        .copied()
                        .ok_or(DataGenerationError::Missing(
                            "bootstrap index key column type enrollment",
                        ))?;
                    Ok(BootstrapResourceIndexKeyBinding {
                        key_ordinal: descriptor.key_ordinal,
                        column_id: descriptor.column_id,
                        sql_type,
                        shape_root: descriptor.shape_root,
                    })
                })
                .collect::<Result<Vec<_>, DataGenerationError>>()
                .map(Vec::into_boxed_slice)?;
            enrolled_index_oids.insert(index.display_index_oid);
            enrolled_indexes.insert((table.table_id, index.index_id));
            if index_keys
                .insert((table.table_id, index.index_id), resource_key_bindings)
                .is_some()
            {
                return Err(DataGenerationError::Invalid(
                    "duplicate bootstrap index resource binding",
                ));
            }
            let _ = (index.generation, index.root, index.shape_root);
        }
        if enrolled_index_oids != catalog_indexes {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap catalog index enrollment",
            ));
        }
        let _ = (
            table.logical_row_count,
            table.data_generation,
            table.expected_table_root,
        );
        if column_types
            .insert(table.table_id, table_column_types)
            .is_some()
        {
            return Err(DataGenerationError::Invalid(
                "duplicate bootstrap table resource binding",
            ));
        }
        if table_row_counts
            .insert(table.table_id, table.logical_row_count)
            .is_some()
        {
            return Err(DataGenerationError::Invalid(
                "duplicate bootstrap table row binding",
            ));
        }
    }
    if current_table_oids != catalog_table_oids {
        return Err(DataGenerationError::PredecessorMismatch(
            "bootstrap catalog table enrollment",
        ));
    }
    Ok(CurrentTableValidation {
        table_ids: current_table_ids,
        enrolled_indexes,
        column_types,
        table_row_counts,
        index_keys,
    })
}

fn validate_index_key_descriptors(
    descriptors: &[BootstrapIndexKeyDescriptor],
    catalog_index: &crate::RelationalIndex,
    catalog_columns_by_name: &BTreeMap<&str, (StableColumnId, SqlType)>,
) -> Result<(), DataGenerationError> {
    let Some(first_key) = catalog_index.key_columns.first() else {
        return Err(DataGenerationError::Missing("bootstrap catalog index keys"));
    };
    if catalog_index.column != *first_key {
        return Err(DataGenerationError::PredecessorMismatch(
            "bootstrap catalog index first key",
        ));
    }
    if descriptors.len() != catalog_index.key_columns.len() {
        return Err(DataGenerationError::Missing(
            "bootstrap index key descriptor coverage",
        ));
    }
    for (position, descriptor) in descriptors.iter().enumerate() {
        let expected_ordinal =
            u16::try_from(position).map_err(|_| DataGenerationError::CountOverflow)?;
        if descriptor.key_ordinal != expected_ordinal {
            return Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap index key descriptors",
            ));
        }
        let catalog_name = &catalog_index.key_columns[position];
        let (expected_column_id, _) = catalog_columns_by_name.get(catalog_name.as_str()).ok_or(
            DataGenerationError::Missing("bootstrap catalog index key column"),
        )?;
        if descriptor.column_id != *expected_column_id {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap index key catalog order",
            ));
        }
        // The shape root is an opaque GPU-produced expected comparator. It is deliberately not
        // recomputed from catalog data and is never passed into a CPU or GPU layout compiler here.
        let _ = descriptor.shape_root;
    }
    Ok(())
}

/// One exact canonical envelope replayed at one terminal sequence. The opaque envelope bytes are
/// retained for the later materializer/recovery handoff; this foundation neither decodes nor
/// re-hashes them. The duplicated typed facts are a sealed replay projection used only to bind the
/// terminal status map to those exact bytes.
#[derive(Clone, Debug)]
struct CanonicalTerminalEnvelopeProvenance {
    exact_canonical_envelope: Arc<[u8]>,
    entry_leaf_root: StatusEntryLeafRoot,
    entry: PublishedStatusEntry,
}

/// A terminal status entry has no construction path outside the sealed replay witness.
#[derive(Clone, Debug)]
struct ReplayedTerminalEnvelope {
    entry_leaf_root: StatusEntryLeafRoot,
    entry: PublishedStatusEntry,
    canonical: CanonicalTerminalEnvelopeProvenance,
}

/// The complete canonical terminal-envelope prefix through the witness cut.
#[derive(Clone, Debug)]
struct ReplayedTerminalStatusWitness {
    database_id: DatabaseId,
    root_format: RootFormatVersion,
    covered_through: u64,
    status_view_root: StatusViewRoot,
    last_terminal_envelope: Option<super::digest::TerminalEnvelopeDigest>,
    entries: Vec<ReplayedTerminalEnvelope>,
}

impl ReplayedTerminalStatusWitness {
    fn validate(
        &self,
        cut: BootstrapDurableCut,
        database_id: DatabaseId,
        root_format: RootFormatVersion,
    ) -> Result<(), DataGenerationError> {
        if self.database_id != database_id
            || self.root_format != root_format
            || self.covered_through != cut.covered_through
        {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap status identity",
            ));
        }
        if u64::try_from(self.entries.len()).map_err(|_| DataGenerationError::CountOverflow)?
            != cut.covered_through
        {
            return Err(DataGenerationError::Invalid("bootstrap status coverage"));
        }
        let mut transaction_ids = BTreeSet::new();
        for (position, completed) in self.entries.iter().enumerate() {
            let expected_sequence = u64::try_from(position)
                .map_err(|_| DataGenerationError::CountOverflow)?
                .checked_add(1)
                .ok_or(DataGenerationError::CountOverflow)?;
            if completed.entry.commit_sequence.get() != expected_sequence {
                return Err(DataGenerationError::NonCanonicalOrder(
                    "bootstrap terminal status sequence",
                ));
            }
            if completed.canonical.exact_canonical_envelope.is_empty()
                || completed.entry_leaf_root != completed.canonical.entry_leaf_root
                || completed.entry != completed.canonical.entry
            {
                return Err(DataGenerationError::PredecessorMismatch(
                    "bootstrap terminal envelope provenance",
                ));
            }
            completed.entry.validate()?;
            if !transaction_ids.insert(completed.entry.transaction_id) {
                return Err(DataGenerationError::Invalid(
                    "duplicate bootstrap status transaction",
                ));
            }
        }
        let observed_last = self
            .entries
            .last()
            .map(|entry| entry.entry.terminal_envelope_digest);
        if observed_last != self.last_terminal_envelope {
            return Err(DataGenerationError::Invalid(
                "bootstrap terminal-status tail",
            ));
        }
        let _ = self.status_view_root;
        Ok(())
    }
}

/// A physical resource placeholder is keyed by logical owner and durable provenance, never by a
/// device address, cache generation, or placement. Future materialization must replace this with
/// a separately owned, typed resource handoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BootstrapPhysicalResourceKind {
    DatabaseManifest,
    StatusView,
    TablePayload,
    IndexPayload,
    Sidecar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BootstrapPhysicalResourceOwner {
    Database,
    Status,
    Table(StableTableId),
    Index {
        table_id: StableTableId,
        index_id: StableIndexId,
    },
}

/// Authenticated source representation for one physical descriptor. This is not a live-cache
/// class or a placement decision: it says which detached owner the replay source sealed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum BootstrapResourceStorageTier {
    Resident,
    DetachedRam,
}

/// Nonzero, placement-neutral identity of one persisted physical layout descriptor. It binds a
/// resource range to the source format without contributing to any logical root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct BootstrapPhysicalLayoutDescriptorId(NonZeroU64);

impl BootstrapPhysicalLayoutDescriptorId {
    pub(super) fn new(value: u64) -> Result<Self, DataGenerationError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(DataGenerationError::ZeroIdentity(
                "bootstrap physical layout descriptor",
            ))
    }
}

/// Fixed typed role vocabulary retained with every source layout. It intentionally describes
/// source bytes only; it is neither a relational execution plan nor a live-cache lookup key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum BootstrapPhysicalLayoutRoleKind {
    Opaque,
    /// The persisted stable row identity. It is never inferred from a payload shard offset.
    StableRowId,
    Validity,
    Value,
    CreatedBy,
    DeletedBy,
    IndexKey,
}

/// Storage representation declared by the sealed physical layout rather than inferred from host
/// values during rebuild.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum BootstrapPhysicalStorageType {
    Bytes,
    Bit,
    Int4,
    Int8,
    Int128,
}

/// One exact relative byte region in the descriptor's enclosing resource. Column/type facts are
/// present for value, validity, and index-key roles; stable-row and MVCC roles carry no column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BootstrapPhysicalLayoutRole {
    pub(super) kind: BootstrapPhysicalLayoutRoleKind,
    pub(super) ordinal: u16,
    pub(super) column_id: Option<StableColumnId>,
    pub(super) sql_type: Option<SqlType>,
    pub(super) storage_type: BootstrapPhysicalStorageType,
    pub(super) key_ordinal: Option<u16>,
    /// Exact catalog-derived shape root for this ordered index key. A nonzero layout descriptor
    /// ID names the enclosing physical descriptor; this root binds the key's semantic shape.
    pub(super) key_shape_root: Option<ColumnShapeRoot>,
    pub(super) byte_offset: u64,
    /// Role extents may be zero for an empty row span; the enclosing detached resource remains
    /// nonempty and is the ownership/range unit.
    pub(super) byte_len: u64,
    pub(super) byte_stride: NonZeroU64,
}

/// Immutable sealed physical layout facts for exactly one resource descriptor. The initial
/// rebuild boundary retains this fixed header/role vector fail-closed; a later row-layout builder
/// may consume these facts, but must not reload or infer a descriptor from live engine state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BootstrapPhysicalLayoutDescriptor {
    pub(super) id: BootstrapPhysicalLayoutDescriptorId,
    /// Persisted source encoding. V1 is the sole admitted grammar; zero and every future value
    /// fail closed until that version has its own complete source-layout validator.
    pub(super) encoding_version: u16,
    pub(super) row_start: u64,
    pub(super) row_count: u64,
    pub(super) roles: Box<[BootstrapPhysicalLayoutRole]>,
}

impl BootstrapPhysicalLayoutDescriptor {
    fn validate(
        &self,
        resource_len: u64,
        resource_kind: BootstrapPhysicalResourceKind,
        resource_owner: BootstrapPhysicalResourceOwner,
    ) -> Result<(), DataGenerationError> {
        if self.encoding_version != 1 {
            return Err(DataGenerationError::Invalid(
                "unsupported bootstrap physical layout encoding version",
            ));
        }
        if self.roles.is_empty() {
            return Err(DataGenerationError::Missing(
                "bootstrap physical layout roles",
            ));
        }
        self.row_start
            .checked_add(self.row_count)
            .ok_or(DataGenerationError::CountOverflow)?;
        let mut index_key_ordinal = 0_u16;
        let mut prior_end = 0_u64;
        for (position, role) in self.roles.iter().enumerate() {
            let expected_ordinal =
                u16::try_from(position).map_err(|_| DataGenerationError::CountOverflow)?;
            if role.ordinal != expected_ordinal {
                return Err(DataGenerationError::NonCanonicalOrder(
                    "bootstrap physical layout roles",
                ));
            }
            let role_end = role
                .byte_offset
                .checked_add(role.byte_len)
                .ok_or(DataGenerationError::CountOverflow)?;
            if role_end > resource_len || role.byte_offset < prior_end {
                return Err(DataGenerationError::Invalid(
                    "bootstrap physical layout role range",
                ));
            }
            prior_end = role_end;
            match role.kind {
                BootstrapPhysicalLayoutRoleKind::Opaque => {
                    if role.column_id.is_some()
                        || role.sql_type.is_some()
                        || role.key_ordinal.is_some()
                        || role.key_shape_root.is_some()
                    {
                        return Err(DataGenerationError::Invalid(
                            "bootstrap opaque physical layout role",
                        ));
                    }
                }
                BootstrapPhysicalLayoutRoleKind::StableRowId => {
                    if role.column_id.is_some()
                        || role.sql_type.is_some()
                        || role.key_ordinal.is_some()
                        || role.key_shape_root.is_some()
                    {
                        return Err(DataGenerationError::Invalid(
                            "bootstrap stable-row physical layout role",
                        ));
                    }
                }
                BootstrapPhysicalLayoutRoleKind::Validity
                | BootstrapPhysicalLayoutRoleKind::Value => {
                    if role.column_id.is_none()
                        || role.sql_type.is_none()
                        || role.key_ordinal.is_some()
                        || role.key_shape_root.is_some()
                    {
                        return Err(DataGenerationError::Invalid(
                            "bootstrap column physical layout role",
                        ));
                    }
                }
                BootstrapPhysicalLayoutRoleKind::CreatedBy
                | BootstrapPhysicalLayoutRoleKind::DeletedBy => {
                    if role.column_id.is_some()
                        || role.sql_type.is_some()
                        || role.key_ordinal.is_some()
                        || role.key_shape_root.is_some()
                    {
                        return Err(DataGenerationError::Invalid(
                            "bootstrap MVCC physical layout role",
                        ));
                    }
                }
                BootstrapPhysicalLayoutRoleKind::IndexKey => {
                    if role.column_id.is_none()
                        || role.sql_type.is_none()
                        || role.key_ordinal != Some(index_key_ordinal)
                        || role.key_shape_root.is_none()
                    {
                        return Err(DataGenerationError::Invalid(
                            "bootstrap index-key physical layout role",
                        ));
                    }
                    index_key_ordinal = index_key_ordinal
                        .checked_add(1)
                        .ok_or(DataGenerationError::CountOverflow)?;
                }
            }
            validate_physical_role_storage(role, self.row_count)?;
        }
        match (resource_kind, resource_owner) {
            (
                BootstrapPhysicalResourceKind::DatabaseManifest,
                BootstrapPhysicalResourceOwner::Database,
            )
            | (BootstrapPhysicalResourceKind::StatusView, BootstrapPhysicalResourceOwner::Status) => {
                if self.row_start != 0
                    || self.row_count != 0
                    || self.roles.len() != 1
                    || self.roles[0].kind != BootstrapPhysicalLayoutRoleKind::Opaque
                    || self.roles[0].byte_offset != 0
                    || self.roles[0].byte_len != resource_len
                {
                    return Err(DataGenerationError::Invalid(
                        "bootstrap opaque physical layout grammar",
                    ));
                }
            }
            (
                BootstrapPhysicalResourceKind::TablePayload,
                BootstrapPhysicalResourceOwner::Table(_),
            )
            | (BootstrapPhysicalResourceKind::Sidecar, BootstrapPhysicalResourceOwner::Table(_)) => {
                if self.roles.iter().any(|role| {
                    !matches!(
                        role.kind,
                        BootstrapPhysicalLayoutRoleKind::StableRowId
                            | BootstrapPhysicalLayoutRoleKind::Validity
                            | BootstrapPhysicalLayoutRoleKind::Value
                            | BootstrapPhysicalLayoutRoleKind::CreatedBy
                            | BootstrapPhysicalLayoutRoleKind::DeletedBy
                    )
                }) {
                    return Err(DataGenerationError::Invalid(
                        "bootstrap table physical layout grammar",
                    ));
                }
            }
            (
                BootstrapPhysicalResourceKind::IndexPayload,
                BootstrapPhysicalResourceOwner::Index { .. },
            )
            | (
                BootstrapPhysicalResourceKind::Sidecar,
                BootstrapPhysicalResourceOwner::Index { .. },
            ) => {
                if self
                    .roles
                    .iter()
                    .any(|role| role.kind != BootstrapPhysicalLayoutRoleKind::IndexKey)
                {
                    return Err(DataGenerationError::Invalid(
                        "bootstrap index physical layout grammar",
                    ));
                }
            }
            _ => {
                return Err(DataGenerationError::Invalid(
                    "bootstrap physical layout owner",
                ))
            }
        }
        Ok(())
    }

    fn validate_catalog_binding(
        &self,
        resource_kind: BootstrapPhysicalResourceKind,
        resource_owner: BootstrapPhysicalResourceOwner,
        bindings: &CurrentTableValidation,
    ) -> Result<(), DataGenerationError> {
        match resource_owner {
            BootstrapPhysicalResourceOwner::Database | BootstrapPhysicalResourceOwner::Status => {
                return Ok(())
            }
            BootstrapPhysicalResourceOwner::Table(table_id) => {
                return self.validate_table_catalog_binding(
                    resource_kind == BootstrapPhysicalResourceKind::TablePayload,
                    table_id,
                    bindings,
                )
            }
            BootstrapPhysicalResourceOwner::Index { table_id, index_id } => {
                self.validate_index_catalog_binding(table_id, index_id, bindings)?;
            }
        }
        Ok(())
    }

    fn validate_table_catalog_binding(
        &self,
        require_complete_columns: bool,
        table_id: StableTableId,
        bindings: &CurrentTableValidation,
    ) -> Result<(), DataGenerationError> {
        let row_end = self
            .row_start
            .checked_add(self.row_count)
            .ok_or(DataGenerationError::CountOverflow)?;
        let expected_row_count =
            bindings
                .table_row_counts
                .get(&table_id)
                .ok_or(DataGenerationError::Missing(
                    "bootstrap physical layout table enrollment",
                ))?;
        if row_end > *expected_row_count {
            return Err(DataGenerationError::Invalid(
                "bootstrap table physical layout row bounds",
            ));
        }
        let expected_columns =
            bindings
                .column_types
                .get(&table_id)
                .ok_or(DataGenerationError::Missing(
                    "bootstrap physical layout table enrollment",
                ))?;
        let mut values = BTreeSet::new();
        let mut validity = BTreeSet::new();
        let mut stable_row_ids = 0_u16;
        let mut created_by = 0_u16;
        let mut deleted_by = 0_u16;
        for role in &self.roles {
            match role.kind {
                BootstrapPhysicalLayoutRoleKind::StableRowId => {
                    stable_row_ids = stable_row_ids
                        .checked_add(1)
                        .ok_or(DataGenerationError::CountOverflow)?
                }
                BootstrapPhysicalLayoutRoleKind::Validity
                | BootstrapPhysicalLayoutRoleKind::Value => {
                    let column_id = role.column_id.ok_or(DataGenerationError::Invalid(
                        "bootstrap physical layout column role",
                    ))?;
                    let expected_type =
                        expected_columns
                            .get(&column_id)
                            .ok_or(DataGenerationError::Missing(
                                "bootstrap physical layout column enrollment",
                            ))?;
                    if role.sql_type.as_ref() != Some(expected_type) {
                        return Err(DataGenerationError::PredecessorMismatch(
                            "bootstrap physical layout column type",
                        ));
                    }
                    let set = if role.kind == BootstrapPhysicalLayoutRoleKind::Value {
                        &mut values
                    } else {
                        &mut validity
                    };
                    if !set.insert(column_id) {
                        return Err(DataGenerationError::Invalid(
                            "duplicate bootstrap table physical layout role",
                        ));
                    }
                }
                BootstrapPhysicalLayoutRoleKind::CreatedBy => {
                    created_by = created_by
                        .checked_add(1)
                        .ok_or(DataGenerationError::CountOverflow)?
                }
                BootstrapPhysicalLayoutRoleKind::DeletedBy => {
                    deleted_by = deleted_by
                        .checked_add(1)
                        .ok_or(DataGenerationError::CountOverflow)?
                }
                BootstrapPhysicalLayoutRoleKind::Opaque
                | BootstrapPhysicalLayoutRoleKind::IndexKey => {
                    return Err(DataGenerationError::Invalid(
                        "bootstrap table physical layout grammar",
                    ))
                }
            }
        }
        if values != validity {
            return Err(DataGenerationError::Missing(
                "bootstrap table physical layout value validity pairing",
            ));
        }
        if created_by > 1 || deleted_by > 1 || created_by != deleted_by {
            return Err(DataGenerationError::Invalid(
                "bootstrap table physical layout MVCC pairing",
            ));
        }
        if require_complete_columns {
            if stable_row_ids != 1 {
                return Err(DataGenerationError::Missing(
                    "bootstrap table physical layout stable row coverage",
                ));
            }
            let all_columns = expected_columns.keys().copied().collect::<BTreeSet<_>>();
            if values != all_columns {
                return Err(DataGenerationError::Missing(
                    "bootstrap table physical layout column coverage",
                ));
            }
            if created_by != 1 {
                return Err(DataGenerationError::Missing(
                    "bootstrap table physical layout MVCC coverage",
                ));
            }
        }
        Ok(())
    }

    fn validate_index_catalog_binding(
        &self,
        table_id: StableTableId,
        index_id: StableIndexId,
        bindings: &CurrentTableValidation,
    ) -> Result<(), DataGenerationError> {
        let row_end = self
            .row_start
            .checked_add(self.row_count)
            .ok_or(DataGenerationError::CountOverflow)?;
        if row_end
            > *bindings
                .table_row_counts
                .get(&table_id)
                .ok_or(DataGenerationError::Missing(
                    "bootstrap index physical layout table enrollment",
                ))?
        {
            return Err(DataGenerationError::Invalid(
                "bootstrap index physical layout row bounds",
            ));
        }
        let expected_keys =
            bindings
                .index_keys
                .get(&(table_id, index_id))
                .ok_or(DataGenerationError::Missing(
                    "bootstrap index physical layout enrollment",
                ))?;
        let actual_keys = &self.roles;
        if actual_keys.len() != expected_keys.len() {
            return Err(DataGenerationError::Missing(
                "bootstrap index physical layout key coverage",
            ));
        }
        for (actual, expected) in actual_keys.iter().zip(expected_keys.iter()) {
            if actual.key_ordinal != Some(expected.key_ordinal)
                || actual.column_id != Some(expected.column_id)
            {
                return Err(DataGenerationError::PredecessorMismatch(
                    "bootstrap index physical layout key order",
                ));
            }
            if actual.sql_type.as_ref() != Some(&expected.sql_type) {
                return Err(DataGenerationError::PredecessorMismatch(
                    "bootstrap index physical layout key type",
                ));
            }
            if actual.key_shape_root != Some(expected.shape_root) {
                return Err(DataGenerationError::PredecessorMismatch(
                    "bootstrap index physical layout key shape",
                ));
            }
        }
        Ok(())
    }
}

/// The sealed V1 physical encoding admits only fixed-width, device-ready sections. SQL type
/// identity remains in the descriptor; this mapping proves that its storage class and stride
/// agree with V1. Text needs offsets/blob grammar that V1 never sealed, so it must fail before
/// an ambiguous source reaches materialization or attachment.
fn physical_value_storage(
    sql_type: SqlType,
) -> Result<(BootstrapPhysicalStorageType, u64), DataGenerationError> {
    Ok(match sql_type {
        // int2 and date deliberately use the engine's widened int4 section.
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

fn bitmap_bytes(row_count: u64) -> Result<u64, DataGenerationError> {
    row_count
        .checked_add(31)
        .ok_or(DataGenerationError::CountOverflow)
        .map(|rows| rows / 32)
        .and_then(|words| {
            words
                .checked_mul(4)
                .ok_or(DataGenerationError::CountOverflow)
        })
}

fn validate_physical_role_storage(
    role: &BootstrapPhysicalLayoutRole,
    row_count: u64,
) -> Result<(), DataGenerationError> {
    let expected = match role.kind {
        BootstrapPhysicalLayoutRoleKind::Opaque => {
            if role.storage_type != BootstrapPhysicalStorageType::Bytes
                || role.byte_stride.get() != 1
            {
                return Err(DataGenerationError::Invalid(
                    "bootstrap opaque physical layout storage",
                ));
            }
            return Ok(());
        }
        BootstrapPhysicalLayoutRoleKind::StableRowId => (
            BootstrapPhysicalStorageType::Int8,
            8,
            row_count
                .checked_mul(8)
                .ok_or(DataGenerationError::CountOverflow)?,
        ),
        BootstrapPhysicalLayoutRoleKind::Validity => (
            BootstrapPhysicalStorageType::Bit,
            1,
            bitmap_bytes(row_count)?,
        ),
        BootstrapPhysicalLayoutRoleKind::Value | BootstrapPhysicalLayoutRoleKind::IndexKey => {
            let sql_type = role.sql_type.ok_or(DataGenerationError::Invalid(
                "bootstrap column physical layout role",
            ))?;
            let (storage, stride) = physical_value_storage(sql_type)?;
            let byte_len = if storage == BootstrapPhysicalStorageType::Bit {
                bitmap_bytes(row_count)?
            } else {
                row_count
                    .checked_mul(stride)
                    .ok_or(DataGenerationError::CountOverflow)?
            };
            (storage, stride, byte_len)
        }
        BootstrapPhysicalLayoutRoleKind::CreatedBy | BootstrapPhysicalLayoutRoleKind::DeletedBy => {
            (
                BootstrapPhysicalStorageType::Int8,
                8,
                row_count
                    .checked_mul(8)
                    .ok_or(DataGenerationError::CountOverflow)?,
            )
        }
    };
    if role.storage_type != expected.0
        || role.byte_stride.get() != expected.1
        || role.byte_len != expected.2
    {
        return Err(DataGenerationError::Invalid(
            "bootstrap physical layout storage geometry",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BootstrapPhysicalResource {
    kind: BootstrapPhysicalResourceKind,
    owner: BootstrapPhysicalResourceOwner,
    /// A durable resource-ledger identity, not a pointer or an identity/root input.
    resource_id: u64,
    /// Canonical per-owner resource ordinal. A logical payload may have several shards/chunks.
    ordinal: u32,
    /// The exact number of members in this `(kind, owner)` group, repeated by every member.
    member_count: u32,
    /// Source-authenticated detached representation and exact physical source extent.
    tier: BootstrapResourceStorageTier,
    byte_offset: u64,
    byte_len: u64,
    layout: BootstrapPhysicalLayoutDescriptor,
    database_id: DatabaseId,
    root_format: RootFormatVersion,
    covered_through: u64,
}

impl BootstrapPhysicalResource {
    fn validate_kind_owner(&self) -> Result<(), DataGenerationError> {
        let valid = matches!(
            (self.kind, self.owner),
            (
                BootstrapPhysicalResourceKind::DatabaseManifest,
                BootstrapPhysicalResourceOwner::Database
            ) | (
                BootstrapPhysicalResourceKind::StatusView,
                BootstrapPhysicalResourceOwner::Status
            ) | (
                BootstrapPhysicalResourceKind::TablePayload,
                BootstrapPhysicalResourceOwner::Table(_)
            ) | (
                BootstrapPhysicalResourceKind::IndexPayload,
                BootstrapPhysicalResourceOwner::Index { .. }
            ) | (
                BootstrapPhysicalResourceKind::Sidecar,
                BootstrapPhysicalResourceOwner::Table(_)
                    | BootstrapPhysicalResourceOwner::Index { .. }
            )
        );
        if !valid {
            return Err(DataGenerationError::Invalid("bootstrap resource owner"));
        }
        Ok(())
    }
}

/// The one sealed result of WAL/checkpoint replay at a quiescent cut. It owns every fact that a
/// bootstrap needs, rather than accepting independently reloadable catalog, migration, status, or
/// resource inputs. There is deliberately no production constructor: integration must mint this
/// capability at the existing replay-authority seam after it has checked the canonical envelope,
/// checkpoint manifest, and catalog identity together.
#[derive(Debug)]
struct BootstrapReplayWitness {
    cut: BootstrapDurableCut,
    root_format: RootFormatVersion,
    database_id: DatabaseId,
    expected_database_root: DatabaseRoot,
    expected_status_view_root: StatusViewRoot,
    catalog: BootstrapCatalogPair,
    catalog_snapshot: Arc<crate::engine_state::CatalogSnapshot>,
    stable_ids: BootstrapStableIdState,
    current_tables: Vec<BootstrapCurrentTable>,
    terminal_status: ReplayedTerminalStatusWitness,
    resources: Box<[BootstrapPhysicalResource]>,
    _sealed: BootstrapReplayWitnessSeal,
}

#[derive(Debug)]
struct BootstrapReplayWitnessSeal;

impl BootstrapReplayWitness {
    fn validate(&self) -> Result<CurrentTableValidation, DataGenerationError> {
        self.cut.validate()?;
        if self.root_format != RootFormatVersion::V1 {
            return Err(DataGenerationError::UnsupportedRootFormat(
                self.root_format.get(),
            ));
        }
        self.catalog.validate()?;
        if self.catalog_snapshot.commit_seq != self.cut.covered_through {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap replayed catalog witness",
            ));
        }
        let (table_ids, index_ids) = self.stable_ids.validate()?;
        let enrolled = validate_current_tables(
            &self.current_tables,
            &table_ids,
            &index_ids,
            &self.stable_ids,
            &self.stable_ids.relation_migrations,
            &self.catalog_snapshot,
        )?;
        self.terminal_status
            .validate(self.cut, self.database_id, self.root_format)?;
        if self.terminal_status.status_view_root != self.expected_status_view_root {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap expected status root",
            ));
        }
        validate_resources(
            &self.resources,
            self.cut,
            self.database_id,
            self.root_format,
            &enrolled,
        )?;
        Ok(enrolled)
    }
}

/// All facts required to validate one quiescent bootstrap are consumed in one replay witness.
/// The source owns no callback, engine handle, reader, or mutable state.
#[derive(Debug)]
struct BootstrapPublicationSource {
    replay: BootstrapReplayWitness,
}

impl BootstrapPublicationSource {
    /// Validate the sealed replay witness atomically, then transfer ownership into an explicitly
    /// uninstalled candidate. This does not materialize manifests or expose a runtime read path.
    fn validate_into_uninstalled(
        self,
    ) -> Result<UninstalledPublicationGeneration, DataGenerationError> {
        self.replay.validate()?;
        Ok(UninstalledPublicationGeneration {
            replay: self.replay,
        })
    }

    /// The sole cross-module bootstrap handoff. The resources capability proves that this source
    /// is being consumed at the private resource boundary; validation and the uninstalled
    /// intermediate move happen atomically before the lease is exposed.
    pub(super) fn validate_into_materialization_lease(
        self,
        _access: &super::resources::BootstrapResourceLeaseAccess,
    ) -> Result<BootstrapMaterializationLease, DataGenerationError> {
        self.validate_into_uninstalled()
            .map(UninstalledPublicationGeneration::into_materialization_lease)
    }
}

fn validate_resources(
    resources: &[BootstrapPhysicalResource],
    cut: BootstrapDurableCut,
    database_id: DatabaseId,
    root_format: RootFormatVersion,
    bindings: &CurrentTableValidation,
) -> Result<(), DataGenerationError> {
    if resources.is_empty() {
        return Err(DataGenerationError::Missing("bootstrap physical resources"));
    }
    validate_strictly_ascending(
        resources
            .iter()
            .map(|resource| (resource.kind, resource.owner, resource.ordinal)),
        "bootstrap physical resources",
    )?;
    let mut resource_ids = BTreeSet::new();
    let mut layout_descriptor_ids = BTreeSet::new();
    for resource in resources {
        if resource.resource_id == 0 {
            return Err(DataGenerationError::ZeroIdentity("bootstrap resource"));
        }
        if resource.database_id != database_id
            || resource.root_format != root_format
            || resource.covered_through != cut.covered_through
        {
            return Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource provenance",
            ));
        }
        if !resource_ids.insert(resource.resource_id) {
            return Err(DataGenerationError::Invalid("duplicate bootstrap resource"));
        }
        if resource.byte_len == 0
            || resource
                .byte_offset
                .checked_add(resource.byte_len)
                .is_none()
        {
            return Err(DataGenerationError::Invalid(
                "bootstrap resource source range",
            ));
        }
        if resource.tier == BootstrapResourceStorageTier::Resident && resource.byte_offset != 0 {
            return Err(DataGenerationError::Invalid(
                "bootstrap resident source range",
            ));
        }
        resource
            .layout
            .validate(resource.byte_len, resource.kind, resource.owner)?;
        if !layout_descriptor_ids.insert(resource.layout.id) {
            return Err(DataGenerationError::Invalid(
                "duplicate bootstrap physical layout descriptor",
            ));
        }
        resource.validate_kind_owner()?;
    }

    let base_groups = validate_resource_groups(resources)?;
    if base_groups.database_manifest.is_none() || base_groups.status_view.is_none() {
        return Err(DataGenerationError::Missing(
            "bootstrap base resource coverage",
        ));
    }
    if base_groups.database_manifest != Some(1) || base_groups.status_view != Some(1) {
        return Err(DataGenerationError::Invalid(
            "bootstrap base resource multiplicity",
        ));
    }

    let mut table_payloads = BTreeSet::new();
    let mut table_payload_groups =
        BTreeMap::<StableTableId, Vec<&BootstrapPhysicalResource>>::new();
    let mut index_payloads = BTreeSet::new();
    for resource in resources {
        match (resource.kind, resource.owner) {
            (_, BootstrapPhysicalResourceOwner::Table(table_id)) => {
                if !bindings.table_ids.contains(&table_id) {
                    return Err(DataGenerationError::Missing(
                        "bootstrap resource current table",
                    ));
                }
                validate_empty_table_typed_resource_geometry(resource, table_id, bindings)?;
                resource.layout.validate_catalog_binding(
                    resource.kind,
                    resource.owner,
                    bindings,
                )?;
                if resource.kind == BootstrapPhysicalResourceKind::TablePayload {
                    table_payloads.insert(table_id);
                    table_payload_groups
                        .entry(table_id)
                        .or_default()
                        .push(resource);
                }
            }
            (_, BootstrapPhysicalResourceOwner::Index { table_id, index_id }) => {
                if !bindings.enrolled_indexes.contains(&(table_id, index_id)) {
                    return Err(DataGenerationError::Missing(
                        "bootstrap resource index enrollment",
                    ));
                }
                validate_empty_table_typed_resource_geometry(resource, table_id, bindings)?;
                resource.layout.validate_catalog_binding(
                    resource.kind,
                    resource.owner,
                    bindings,
                )?;
                if resource.kind == BootstrapPhysicalResourceKind::IndexPayload {
                    index_payloads.insert((table_id, index_id));
                }
            }
            _ => {
                resource
                    .layout
                    .validate_catalog_binding(resource.kind, resource.owner, bindings)?
            }
        }
    }
    if table_payloads != bindings.table_ids {
        return Err(DataGenerationError::Missing(
            "bootstrap table resource coverage",
        ));
    }
    if index_payloads != bindings.enrolled_indexes {
        return Err(DataGenerationError::Missing(
            "bootstrap index resource coverage",
        ));
    }
    for (table_id, group) in table_payload_groups {
        validate_table_payload_group(table_id, &group, bindings)?;
    }
    Ok(())
}

/// Table payload shards are a canonical row partition. Column/MVCC completeness is checked for
/// every shard above; this group gate prevents gaps, overlap, or an incomplete final table span.
///
/// An empty V1 payload still retains one byte: it is the exact detached-owner sentinel required
/// by the resource ledger's nonempty range invariant, not unmodeled row data. All typed roles
/// must remain zero-length at offset zero, so no arbitrary source bytes can be relabeled as an
/// empty table payload.
const V1_EMPTY_TABLE_PAYLOAD_SENTINEL_BYTES: u64 = 1;

/// Every non-base resource for an empty table has the same exact V1 detached-owner sentinel.
/// This applies to table/index payloads and sidecars alike: their source bytes cannot carry an
/// untyped, nonempty range merely because no logical rows currently reference it.
fn validate_empty_table_typed_resource_geometry(
    resource: &BootstrapPhysicalResource,
    table_id: StableTableId,
    bindings: &CurrentTableValidation,
) -> Result<(), DataGenerationError> {
    let logical_row_count =
        bindings
            .table_row_counts
            .get(&table_id)
            .ok_or(DataGenerationError::Missing(
                "bootstrap empty table resource enrollment",
            ))?;
    if *logical_row_count != 0 {
        return Ok(());
    }
    if resource.byte_len != V1_EMPTY_TABLE_PAYLOAD_SENTINEL_BYTES
        || resource.layout.row_start != 0
        || resource.layout.row_count != 0
        || resource
            .layout
            .roles
            .iter()
            .any(|role| role.byte_offset != 0 || role.byte_len != 0)
    {
        return Err(DataGenerationError::Invalid(
            "bootstrap empty table typed resource V1 geometry",
        ));
    }
    Ok(())
}

fn validate_table_payload_group(
    table_id: StableTableId,
    resources: &[&BootstrapPhysicalResource],
    bindings: &CurrentTableValidation,
) -> Result<(), DataGenerationError> {
    let expected_rows =
        bindings
            .table_row_counts
            .get(&table_id)
            .ok_or(DataGenerationError::Missing(
                "bootstrap table payload row enrollment",
            ))?;
    if *expected_rows == 0 {
        if resources.len() != 1 {
            return Err(DataGenerationError::Invalid(
                "bootstrap empty table payload multiplicity",
            ));
        }
        return Ok(());
    }
    let mut next_row = 0_u64;
    for resource in resources {
        if resource.layout.row_count == 0 {
            return Err(DataGenerationError::Invalid(
                "bootstrap nonempty table payload zero-row shard",
            ));
        }
        if resource.layout.row_start != next_row {
            return Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap table payload row coverage",
            ));
        }
        next_row = next_row
            .checked_add(resource.layout.row_count)
            .ok_or(DataGenerationError::CountOverflow)?;
    }
    if next_row != *expected_rows {
        return Err(DataGenerationError::Missing(
            "bootstrap table payload row coverage",
        ));
    }
    Ok(())
}

#[derive(Default)]
struct BootstrapBaseResourceGroups {
    database_manifest: Option<u32>,
    status_view: Option<u32>,
}

fn validate_resource_groups(
    resources: &[BootstrapPhysicalResource],
) -> Result<BootstrapBaseResourceGroups, DataGenerationError> {
    let mut base_groups = BootstrapBaseResourceGroups::default();
    let mut start = 0;
    while start < resources.len() {
        let first = &resources[start];
        if first.member_count == 0 {
            return Err(DataGenerationError::Invalid(
                "bootstrap resource member count",
            ));
        }
        let mut end = start + 1;
        while end < resources.len()
            && resources[end].kind == first.kind
            && resources[end].owner == first.owner
        {
            end += 1;
        }
        let actual_count =
            u32::try_from(end - start).map_err(|_| DataGenerationError::CountOverflow)?;
        if actual_count != first.member_count {
            return Err(DataGenerationError::Invalid(
                "bootstrap resource member count",
            ));
        }
        for (position, resource) in resources[start..end].iter().enumerate() {
            let expected_ordinal =
                u32::try_from(position).map_err(|_| DataGenerationError::CountOverflow)?;
            if resource.member_count != first.member_count {
                return Err(DataGenerationError::Invalid(
                    "bootstrap resource member count",
                ));
            }
            if resource.ordinal != expected_ordinal {
                return Err(DataGenerationError::NonCanonicalOrder(
                    "bootstrap resource group ordinals",
                ));
            }
        }
        match (first.kind, first.owner) {
            (
                BootstrapPhysicalResourceKind::DatabaseManifest,
                BootstrapPhysicalResourceOwner::Database,
            ) => base_groups.database_manifest = Some(first.member_count),
            (BootstrapPhysicalResourceKind::StatusView, BootstrapPhysicalResourceOwner::Status) => {
                base_groups.status_view = Some(first.member_count)
            }
            _ => {}
        }
        start = end;
    }
    Ok(base_groups)
}

fn validate_strictly_ascending<T: Ord>(
    values: impl Iterator<Item = T>,
    label: &'static str,
) -> Result<(), DataGenerationError> {
    let mut previous = None;
    for value in values {
        if previous.as_ref().is_some_and(|prior| prior >= &value) {
            return Err(DataGenerationError::NonCanonicalOrder(label));
        }
        previous = Some(value);
    }
    Ok(())
}

/// The placement-neutral physical-resource claim issued by the sealed replay witness. It is a
/// resource-validation/lifetime key only; neither its resource ID nor its ordinal enters any
/// logical root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum BootstrapResourceLedgerKind {
    DatabaseManifest,
    StatusView,
    TablePayload,
    IndexPayload,
    Sidecar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum BootstrapResourceLedgerOwner {
    Database,
    Status,
    Table(StableTableId),
    Index {
        table_id: StableTableId,
        index_id: StableIndexId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BootstrapResourceLedgerEntry {
    pub(super) kind: BootstrapResourceLedgerKind,
    pub(super) owner: BootstrapResourceLedgerOwner,
    pub(super) resource_id: u64,
    pub(super) ordinal: u32,
    pub(super) member_count: u32,
    pub(super) tier: BootstrapResourceStorageTier,
    pub(super) byte_offset: u64,
    pub(super) byte_len: NonZeroU64,
    /// Immutable source layout facts, including the nonzero descriptor ID and exact role ranges.
    pub(super) layout: BootstrapPhysicalLayoutDescriptor,
    /// Durable source provenance is part of an exact resource claim. Resource IDs are unique
    /// only within one replay source, so a materializer must not accept an otherwise-shaped
    /// record from another database, root format, or quiescent cut.
    pub(super) database_id: DatabaseId,
    pub(super) root_format: RootFormatVersion,
    pub(super) covered_through: u64,
}

impl From<BootstrapPhysicalResource> for BootstrapResourceLedgerEntry {
    fn from(resource: BootstrapPhysicalResource) -> Self {
        let kind = match resource.kind {
            BootstrapPhysicalResourceKind::DatabaseManifest => {
                BootstrapResourceLedgerKind::DatabaseManifest
            }
            BootstrapPhysicalResourceKind::StatusView => BootstrapResourceLedgerKind::StatusView,
            BootstrapPhysicalResourceKind::TablePayload => {
                BootstrapResourceLedgerKind::TablePayload
            }
            BootstrapPhysicalResourceKind::IndexPayload => {
                BootstrapResourceLedgerKind::IndexPayload
            }
            BootstrapPhysicalResourceKind::Sidecar => BootstrapResourceLedgerKind::Sidecar,
        };
        let owner = match resource.owner {
            BootstrapPhysicalResourceOwner::Database => BootstrapResourceLedgerOwner::Database,
            BootstrapPhysicalResourceOwner::Status => BootstrapResourceLedgerOwner::Status,
            BootstrapPhysicalResourceOwner::Table(table_id) => {
                BootstrapResourceLedgerOwner::Table(table_id)
            }
            BootstrapPhysicalResourceOwner::Index { table_id, index_id } => {
                BootstrapResourceLedgerOwner::Index { table_id, index_id }
            }
        };
        Self {
            kind,
            owner,
            resource_id: resource.resource_id,
            ordinal: resource.ordinal,
            member_count: resource.member_count,
            tier: resource.tier,
            byte_offset: resource.byte_offset,
            byte_len: NonZeroU64::new(resource.byte_len)
                .expect("validated bootstrap resource range must be nonempty"),
            layout: resource.layout,
            database_id: resource.database_id,
            root_format: resource.root_format,
            covered_through: resource.covered_through,
        }
    }
}

/// Root-free source facts for the private V1 rebuild compiler. This intentionally carries only
/// stable identities, exact sealed physical geometry, and storage types; no expected commitment,
/// terminal-envelope fact, or raw source byte crosses this boundary.
pub(super) struct BootstrapRebuildRootFreeSource {
    pub(super) database_id: DatabaseId,
    pub(super) root_format: RootFormatVersion,
    pub(super) covered_through: u64,
    pub(super) tables: Box<[BootstrapRebuildRootFreeTable]>,
    pub(super) resources: Box<[BootstrapRebuildRootFreeResource]>,
}

pub(super) struct BootstrapRebuildRootFreeTable {
    pub(super) table_id: StableTableId,
    pub(super) data_generation: DataGeneration,
    pub(super) logical_row_count: u64,
    /// Canonical stable-column inventory copied from the sealed catalog enrollment. These are
    /// semantic compiler facts, not shape roots or a host catalog lookup surface.
    pub(super) columns: Box<[BootstrapRebuildRootFreeColumn]>,
    pub(super) indexes: Box<[BootstrapRebuildRootFreeIndex]>,
}

pub(super) struct BootstrapRebuildRootFreeColumn {
    pub(super) ordinal: u16,
    pub(super) column_id: StableColumnId,
    pub(super) sql_type: SqlType,
    pub(super) attnum: i16,
    pub(super) declared_type_oid: u32,
    pub(super) signed_type_size: i16,
}

pub(super) struct BootstrapRebuildRootFreeIndex {
    pub(super) index_id: StableIndexId,
    pub(super) keys: Box<[BootstrapRebuildRootFreeIndexKey]>,
}

pub(super) struct BootstrapRebuildRootFreeIndexKey {
    pub(super) key_ordinal: u16,
    pub(super) column_id: StableColumnId,
    pub(super) sql_type: SqlType,
}

/// One opaque-owner binding stripped of every comparator root. The resource ID/range remains a
/// bounded source-ownership coordinate only; it is never a logical identity or root input.
pub(super) struct BootstrapRebuildRootFreeResource {
    pub(super) kind: BootstrapResourceLedgerKind,
    pub(super) owner: BootstrapResourceLedgerOwner,
    pub(super) resource_id: u64,
    pub(super) ordinal: u32,
    pub(super) member_count: u32,
    pub(super) tier: BootstrapResourceStorageTier,
    pub(super) byte_offset: u64,
    pub(super) byte_len: NonZeroU64,
    /// Source provenance has no logical commitment content, but it prevents a physically
    /// well-shaped descriptor from a different sealed source entering this compiler input.
    pub(super) database_id: DatabaseId,
    pub(super) root_format: RootFormatVersion,
    pub(super) covered_through: u64,
    pub(super) layout: BootstrapRebuildRootFreeLayout,
}

pub(super) struct BootstrapRebuildRootFreeLayout {
    pub(super) descriptor_id: BootstrapPhysicalLayoutDescriptorId,
    pub(super) encoding_version: u16,
    pub(super) row_start: u64,
    pub(super) row_count: u64,
    pub(super) roles: Box<[BootstrapRebuildRootFreeRole]>,
}

pub(super) struct BootstrapRebuildRootFreeRole {
    pub(super) kind: BootstrapPhysicalLayoutRoleKind,
    pub(super) ordinal: u16,
    pub(super) column_id: Option<StableColumnId>,
    pub(super) sql_type: Option<SqlType>,
    pub(super) storage_type: BootstrapPhysicalStorageType,
    pub(super) key_ordinal: Option<u16>,
    pub(super) byte_offset: u64,
    pub(super) byte_len: u64,
    pub(super) byte_stride: NonZeroU64,
}

impl From<&BootstrapResourceLedgerEntry> for BootstrapRebuildRootFreeResource {
    fn from(entry: &BootstrapResourceLedgerEntry) -> Self {
        Self {
            kind: entry.kind,
            owner: entry.owner,
            resource_id: entry.resource_id,
            ordinal: entry.ordinal,
            member_count: entry.member_count,
            tier: entry.tier,
            byte_offset: entry.byte_offset,
            byte_len: entry.byte_len,
            database_id: entry.database_id,
            root_format: entry.root_format,
            covered_through: entry.covered_through,
            layout: BootstrapRebuildRootFreeLayout {
                descriptor_id: entry.layout.id,
                encoding_version: entry.layout.encoding_version,
                row_start: entry.layout.row_start,
                row_count: entry.layout.row_count,
                roles: entry
                    .layout
                    .roles
                    .iter()
                    .map(|role| BootstrapRebuildRootFreeRole {
                        kind: role.kind,
                        ordinal: role.ordinal,
                        column_id: role.column_id,
                        sql_type: role.sql_type,
                        storage_type: role.storage_type,
                        key_ordinal: role.key_ordinal,
                        byte_offset: role.byte_offset,
                        byte_len: role.byte_len,
                        byte_stride: role.byte_stride,
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            },
        }
    }
}

/// Comparator-only checkpoint facts. The rebuild compiler never receives this type: a later GPU
/// completion validator consumes it only to compare the separately produced table/index/database
/// commitments with the sealed replay expectations.
pub(super) struct BootstrapRebuildExpectationSource {
    expected_database_root: DatabaseRoot,
    expected_status_view_root: StatusViewRoot,
    tables: Box<[BootstrapRebuildExpectedTable]>,
}

struct BootstrapRebuildExpectedTable {
    table_id: StableTableId,
    data_generation: DataGeneration,
    table_root: TableRoot,
    indexes: Box<[BootstrapRebuildExpectedIndex]>,
}

struct BootstrapRebuildExpectedIndex {
    index_id: StableIndexId,
    generation: IndexGeneration,
    root: IndexRoot,
    shape_root: IndexShapeRoot,
    key_shape_roots: Box<[ColumnShapeRoot]>,
}

/// A fully checked bootstrap payload with no installation, publication, or read capability. Its
/// fields remain private so resources can consume it only through the sealed lease below.
#[derive(Debug)]
struct UninstalledPublicationGeneration {
    replay: BootstrapReplayWitness,
}

/// The only sibling-visible carrier of an uninstalled generation's physical ledger. It owns the
/// whole logical candidate so resource records from another replay source cannot be substituted
/// for this source's canonical claims. The resource module may inspect the immutable claim list,
/// but cannot reload replay state or install the candidate.
pub(super) struct BootstrapMaterializationLease {
    _candidate: UninstalledPublicationGeneration,
    claims: Box<[BootstrapResourceLedgerEntry]>,
}

impl UninstalledPublicationGeneration {
    /// Consume this uninstalled candidate into the one move-only materialization lease. There is
    /// deliberately no reverse conversion or public constructor.
    fn into_materialization_lease(self) -> BootstrapMaterializationLease {
        let claims = self
            .replay
            .resources
            .iter()
            .cloned()
            .map(BootstrapResourceLedgerEntry::from)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        BootstrapMaterializationLease {
            _candidate: self,
            claims,
        }
    }
}

impl BootstrapMaterializationLease {
    /// Canonical source claims for `resources` validation only. This exposes no replay facts,
    /// logical roots, runtime, device pointer, or ownership construction path. The private
    /// resources access capability prevents other generation siblings from reading the claims.
    pub(super) fn resource_claims(
        &self,
        _access: &super::resources::BootstrapResourceLeaseAccess,
    ) -> &[BootstrapResourceLedgerEntry] {
        &self.claims
    }

    /// Split a consumed attached lease into the only two rebuild-domain projections. Exact owner
    /// retention is handled by `resources`; this method strips comparator roots from the compiler
    /// source and drops replay/catalog/terminal payloads after extracting their expected outputs.
    pub(super) fn into_bootstrap_rebuild_sources(
        self,
    ) -> (
        BootstrapRebuildRootFreeSource,
        BootstrapRebuildExpectationSource,
    ) {
        let BootstrapMaterializationLease { _candidate, claims } = self;
        let UninstalledPublicationGeneration { replay } = _candidate;
        let BootstrapReplayWitness {
            cut,
            root_format,
            database_id,
            catalog_snapshot,
            expected_database_root,
            expected_status_view_root,
            current_tables,
            ..
        } = replay;
        let mut root_free_tables = Vec::with_capacity(current_tables.len());
        let mut expected_tables = Vec::with_capacity(current_tables.len());
        for table in current_tables {
            let catalog_table = catalog_snapshot
                .relational_catalog
                .values()
                .find(|catalog| catalog.oid == table.display_table_oid)
                .expect("validated bootstrap table enrollment");
            let root_free_columns = table
                .enrolled_columns
                .iter()
                .enumerate()
                .map(|(position, enrolled)| {
                    let ordinal = u16::try_from(position)
                        .expect("validated bootstrap column count fits V1 ordinal");
                    let catalog = catalog_table
                        .columns
                        .iter()
                        .find(|catalog| {
                            catalog.table_oid == enrolled.owner_display_table_oid
                                && u64::from(catalog.id) == enrolled.stable_column_id.get()
                                && catalog.attnum == enrolled.attnum
                        })
                        .expect("validated bootstrap column enrollment");
                    BootstrapRebuildRootFreeColumn {
                        ordinal,
                        column_id: enrolled.stable_column_id,
                        sql_type: catalog.ty,
                        attnum: enrolled.attnum,
                        declared_type_oid: catalog.type_oid,
                        signed_type_size: catalog.type_size,
                    }
                })
                .collect::<Vec<_>>();
            let root_free_indexes = table
                .enrolled_indexes
                .iter()
                .map(|index| BootstrapRebuildRootFreeIndex {
                    index_id: index.index_id,
                    keys: index
                        .key_descriptors
                        .iter()
                        .map(|key| {
                            let sql_type = root_free_columns
                                .iter()
                                .find(|column| column.column_id == key.column_id)
                                .map(|column| column.sql_type)
                                .expect("validated bootstrap index key column enrollment");
                            BootstrapRebuildRootFreeIndexKey {
                                key_ordinal: key.key_ordinal,
                                column_id: key.column_id,
                                sql_type,
                            }
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let expected_indexes = table
                .enrolled_indexes
                .iter()
                .map(|index| BootstrapRebuildExpectedIndex {
                    index_id: index.index_id,
                    generation: index.generation,
                    root: index.root,
                    shape_root: index.shape_root,
                    key_shape_roots: index
                        .key_descriptors
                        .iter()
                        .map(|key| key.shape_root)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            root_free_tables.push(BootstrapRebuildRootFreeTable {
                table_id: table.table_id,
                data_generation: table.data_generation,
                logical_row_count: table.logical_row_count,
                columns: root_free_columns.into_boxed_slice(),
                indexes: root_free_indexes,
            });
            expected_tables.push(BootstrapRebuildExpectedTable {
                table_id: table.table_id,
                data_generation: table.data_generation,
                table_root: table.expected_table_root,
                indexes: expected_indexes,
            });
        }
        (
            BootstrapRebuildRootFreeSource {
                database_id,
                root_format,
                covered_through: cut.covered_through,
                tables: root_free_tables.into_boxed_slice(),
                resources: claims
                    .iter()
                    .map(BootstrapRebuildRootFreeResource::from)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            },
            BootstrapRebuildExpectationSource {
                expected_database_root,
                expected_status_view_root,
                tables: expected_tables.into_boxed_slice(),
            },
        )
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use crate::{
        engine_state::CatalogSnapshot, RelationalColumn, RelationalIndex, RelationalTable,
    };
    use gpu_db_sql::SqlType;

    use super::super::digest::{
        synthetic_gpu_completion_for_test, CanonicalCatalogDigest, CatalogEpoch,
        SyntheticRootForTest,
    };
    use super::super::input::{TerminalOutcome, TerminalOutcomeKind};
    use super::*;

    fn root<R: SyntheticRootForTest>(seed: u64) -> R {
        R::from_synthetic(synthetic_gpu_completion_for_test(seed))
    }

    fn database_id() -> DatabaseId {
        DatabaseId::new([0x42; 16]).expect("database ID")
    }

    fn catalog(seed: u64) -> CatalogIdentity {
        CatalogIdentity::new(CatalogEpoch::new(7), root::<CanonicalCatalogDigest>(seed))
    }

    fn catalog_snapshot(commit_seq: u64) -> Arc<CatalogSnapshot> {
        let table_name = "bootstrap_table".to_owned();
        let table = RelationalTable {
            schema: "public".to_owned(),
            name: table_name.clone(),
            oid: 100,
            columns: vec![RelationalColumn {
                id: 7,
                table_oid: 100,
                attnum: 1,
                name: "id".to_owned(),
                ty: SqlType::Int4,
                domain: None,
                default: None,
                type_oid: SqlType::Int4.postgres_oid(),
                type_size: SqlType::Int4.type_size(),
            }],
            indexes: vec![RelationalIndex {
                oid: 200,
                name: "bootstrap_index".to_owned(),
                table: table_name.clone(),
                column: "id".to_owned(),
                key_columns: vec!["id".to_owned()],
                unique: true,
                primary_key: true,
                unique_constraint: true,
            }],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
            acl: BTreeMap::new(),
        };
        let mut snapshot = CatalogSnapshot {
            commit_seq,
            ..CatalogSnapshot::default()
        };
        snapshot.relational_catalog.insert(table_name, table);
        Arc::new(snapshot)
    }

    fn terminal_envelope(sequence: u64) -> ReplayedTerminalEnvelope {
        let entry_leaf_root = root(sequence + 100);
        let entry = PublishedStatusEntry {
            transaction_id: super::super::digest::StableTransactionId::new(sequence + 10)
                .expect("transaction ID"),
            request_digest: root(sequence + 200),
            commit_sequence: super::super::digest::CommitSequence::new(sequence)
                .expect("commit sequence"),
            outcome: TerminalOutcome {
                kind: TerminalOutcomeKind::CommitSuccess,
                affected_rows: 1,
                sqlstate: None,
                constraint_id: 0,
            },
            target_digest: root(sequence + 300),
            returning_digest: root(sequence + 400),
            terminal_envelope_digest: root(sequence + 500),
        };
        ReplayedTerminalEnvelope {
            entry_leaf_root,
            canonical: CanonicalTerminalEnvelopeProvenance {
                exact_canonical_envelope: Arc::from(vec![sequence as u8]),
                entry_leaf_root,
                entry: entry.clone(),
            },
            entry,
        }
    }

    type LayoutRoleInput = (
        BootstrapPhysicalLayoutRoleKind,
        u16,
        Option<StableColumnId>,
        Option<SqlType>,
        BootstrapPhysicalStorageType,
        Option<u16>,
        Option<ColumnShapeRoot>,
        u64,
        u64,
        u64,
    );

    fn layout_role(input: LayoutRoleInput) -> BootstrapPhysicalLayoutRole {
        let (
            kind,
            ordinal,
            column_id,
            sql_type,
            storage_type,
            key_ordinal,
            key_shape_root,
            byte_offset,
            byte_len,
            byte_stride,
        ) = input;
        BootstrapPhysicalLayoutRole {
            kind,
            ordinal,
            column_id,
            sql_type,
            storage_type,
            key_ordinal,
            key_shape_root,
            byte_offset,
            byte_len,
            byte_stride: NonZeroU64::new(byte_stride).expect("nonzero layout stride"),
        }
    }

    fn layout(
        id: u64,
        roles: Vec<BootstrapPhysicalLayoutRole>,
    ) -> BootstrapPhysicalLayoutDescriptor {
        BootstrapPhysicalLayoutDescriptor {
            id: BootstrapPhysicalLayoutDescriptorId::new(id).expect("layout descriptor"),
            encoding_version: 1,
            row_start: 0,
            row_count: 0,
            roles: roles.into_boxed_slice(),
        }
    }

    fn opaque_layout(id: u64) -> BootstrapPhysicalLayoutDescriptor {
        layout(
            id,
            vec![layout_role((
                BootstrapPhysicalLayoutRoleKind::Opaque,
                0,
                None,
                None,
                BootstrapPhysicalStorageType::Bytes,
                None,
                None,
                0,
                16,
                1,
            ))],
        )
    }

    fn table_layout(id: u64) -> BootstrapPhysicalLayoutDescriptor {
        let column_id = StableColumnId::new(7).expect("stable column ID");
        let mut descriptor = layout(
            id,
            vec![
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::StableRowId,
                    0,
                    None,
                    None,
                    BootstrapPhysicalStorageType::Int8,
                    None,
                    None,
                    0,
                    8,
                    8,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::Validity,
                    1,
                    Some(column_id),
                    Some(SqlType::Int4),
                    BootstrapPhysicalStorageType::Bit,
                    None,
                    None,
                    8,
                    4,
                    1,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::Value,
                    2,
                    Some(column_id),
                    Some(SqlType::Int4),
                    BootstrapPhysicalStorageType::Int4,
                    None,
                    None,
                    12,
                    4,
                    4,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::CreatedBy,
                    3,
                    None,
                    None,
                    BootstrapPhysicalStorageType::Int8,
                    None,
                    None,
                    16,
                    8,
                    8,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::DeletedBy,
                    4,
                    None,
                    None,
                    BootstrapPhysicalStorageType::Int8,
                    None,
                    None,
                    24,
                    8,
                    8,
                )),
            ],
        );
        descriptor.row_count = 1;
        descriptor
    }

    fn index_layout(id: u64) -> BootstrapPhysicalLayoutDescriptor {
        let mut descriptor = layout(
            id,
            vec![layout_role((
                BootstrapPhysicalLayoutRoleKind::IndexKey,
                0,
                Some(StableColumnId::new(7).expect("stable column ID")),
                Some(SqlType::Int4),
                BootstrapPhysicalStorageType::Int4,
                Some(0),
                Some(root(7)),
                0,
                4,
                4,
            ))],
        );
        descriptor.row_count = 1;
        descriptor
    }

    fn compound_table_layout(id: u64) -> BootstrapPhysicalLayoutDescriptor {
        let first = StableColumnId::new(7).expect("stable column ID");
        let second = StableColumnId::new(8).expect("stable column ID");
        let mut descriptor = layout(
            id,
            vec![
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::StableRowId,
                    0,
                    None,
                    None,
                    BootstrapPhysicalStorageType::Int8,
                    None,
                    None,
                    0,
                    8,
                    8,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::Validity,
                    1,
                    Some(first),
                    Some(SqlType::Int4),
                    BootstrapPhysicalStorageType::Bit,
                    None,
                    None,
                    8,
                    4,
                    1,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::Value,
                    2,
                    Some(first),
                    Some(SqlType::Int4),
                    BootstrapPhysicalStorageType::Int4,
                    None,
                    None,
                    12,
                    4,
                    4,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::Validity,
                    3,
                    Some(second),
                    Some(SqlType::Date),
                    BootstrapPhysicalStorageType::Bit,
                    None,
                    None,
                    16,
                    4,
                    1,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::Value,
                    4,
                    Some(second),
                    Some(SqlType::Date),
                    BootstrapPhysicalStorageType::Int4,
                    None,
                    None,
                    20,
                    4,
                    4,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::CreatedBy,
                    5,
                    None,
                    None,
                    BootstrapPhysicalStorageType::Int8,
                    None,
                    None,
                    24,
                    8,
                    8,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::DeletedBy,
                    6,
                    None,
                    None,
                    BootstrapPhysicalStorageType::Int8,
                    None,
                    None,
                    32,
                    8,
                    8,
                )),
            ],
        );
        descriptor.row_count = 1;
        descriptor
    }

    fn compound_index_layout(id: u64) -> BootstrapPhysicalLayoutDescriptor {
        let mut descriptor = layout(
            id,
            vec![
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::IndexKey,
                    0,
                    Some(StableColumnId::new(7).expect("stable column ID")),
                    Some(SqlType::Int4),
                    BootstrapPhysicalStorageType::Int4,
                    Some(0),
                    Some(root(7)),
                    0,
                    4,
                    4,
                )),
                layout_role((
                    BootstrapPhysicalLayoutRoleKind::IndexKey,
                    1,
                    Some(StableColumnId::new(8).expect("stable column ID")),
                    Some(SqlType::Date),
                    BootstrapPhysicalStorageType::Int4,
                    Some(1),
                    Some(root(8)),
                    4,
                    4,
                    4,
                )),
            ],
        );
        descriptor.row_count = 1;
        descriptor
    }

    fn source() -> BootstrapPublicationSource {
        let database_id = database_id();
        let table_id = StableTableId::new(10).expect("table ID");
        let index_id = StableIndexId::new(20).expect("index ID");
        let un_enrolled_index_id = StableIndexId::new(21).expect("index ID");
        let cut = BootstrapDurableCut {
            covered_through: 2,
            visible_next: VisibleNext::new(3).expect("visible next"),
        };
        let entries = vec![terminal_envelope(1), terminal_envelope(2)];
        BootstrapPublicationSource {
            replay: BootstrapReplayWitness {
                cut,
                root_format: RootFormatVersion::V1,
                database_id,
                expected_database_root: root(1),
                expected_status_view_root: root(3),
                catalog: BootstrapCatalogPair {
                    durable: catalog(2),
                    retained: catalog(2),
                },
                catalog_snapshot: catalog_snapshot(2),
                stable_ids: BootstrapStableIdState {
                    migration_complete: true,
                    relation_migrations: vec![
                        BootstrapRelationMigration {
                            kind: BootstrapRelationKind::Table,
                            display_oid: 100,
                            stable_id: table_id.get(),
                        },
                        BootstrapRelationMigration {
                            kind: BootstrapRelationKind::Index,
                            display_oid: 200,
                            stable_id: index_id.get(),
                        },
                        BootstrapRelationMigration {
                            kind: BootstrapRelationKind::Index,
                            display_oid: 201,
                            stable_id: un_enrolled_index_id.get(),
                        },
                    ],
                    column_migrations: vec![BootstrapColumnMigration {
                        owner_display_table_oid: 100,
                        legacy_column_id: 7,
                        attnum: 1,
                        stable_column_id: 7,
                    }],
                    table_high_water: 10,
                    index_high_water: 21,
                    column_high_water: 7,
                    checkpoint_table_high_water: 10,
                    checkpoint_index_high_water: 21,
                    checkpoint_column_high_water: 7,
                },
                current_tables: vec![BootstrapCurrentTable {
                    table_id,
                    display_table_oid: 100,
                    logical_row_count: 1,
                    data_generation: DataGeneration::new(2).expect("data generation"),
                    expected_table_root: root(4),
                    enrolled_columns: vec![BootstrapCurrentColumn {
                        owner_display_table_oid: 100,
                        legacy_column_id: 7,
                        attnum: 1,
                        stable_column_id: StableColumnId::new(7).expect("stable column ID"),
                    }],
                    enrolled_indexes: vec![BootstrapExpectedIndex {
                        display_index_oid: 200,
                        index_id,
                        generation: IndexGeneration::new(2).expect("index generation"),
                        root: root(5),
                        shape_root: root(6),
                        key_descriptors: vec![BootstrapIndexKeyDescriptor {
                            key_ordinal: 0,
                            column_id: StableColumnId::new(7).expect("stable column ID"),
                            shape_root: root(7),
                        }],
                    }],
                }],
                terminal_status: ReplayedTerminalStatusWitness {
                    database_id,
                    root_format: RootFormatVersion::V1,
                    covered_through: 2,
                    status_view_root: root(3),
                    last_terminal_envelope: Some(entries[1].entry.terminal_envelope_digest),
                    entries,
                },
                resources: vec![
                    BootstrapPhysicalResource {
                        kind: BootstrapPhysicalResourceKind::DatabaseManifest,
                        owner: BootstrapPhysicalResourceOwner::Database,
                        resource_id: 1,
                        ordinal: 0,
                        member_count: 1,
                        tier: BootstrapResourceStorageTier::DetachedRam,
                        byte_offset: 4,
                        byte_len: 16,
                        layout: opaque_layout(1),
                        database_id,
                        root_format: RootFormatVersion::V1,
                        covered_through: 2,
                    },
                    BootstrapPhysicalResource {
                        kind: BootstrapPhysicalResourceKind::StatusView,
                        owner: BootstrapPhysicalResourceOwner::Status,
                        resource_id: 2,
                        ordinal: 0,
                        member_count: 1,
                        tier: BootstrapResourceStorageTier::DetachedRam,
                        byte_offset: 4,
                        byte_len: 16,
                        layout: opaque_layout(2),
                        database_id,
                        root_format: RootFormatVersion::V1,
                        covered_through: 2,
                    },
                    BootstrapPhysicalResource {
                        kind: BootstrapPhysicalResourceKind::TablePayload,
                        owner: BootstrapPhysicalResourceOwner::Table(table_id),
                        resource_id: 3,
                        ordinal: 0,
                        member_count: 1,
                        tier: BootstrapResourceStorageTier::DetachedRam,
                        byte_offset: 4,
                        byte_len: 32,
                        layout: table_layout(3),
                        database_id,
                        root_format: RootFormatVersion::V1,
                        covered_through: 2,
                    },
                    BootstrapPhysicalResource {
                        kind: BootstrapPhysicalResourceKind::IndexPayload,
                        owner: BootstrapPhysicalResourceOwner::Index { table_id, index_id },
                        resource_id: 4,
                        ordinal: 0,
                        member_count: 1,
                        tier: BootstrapResourceStorageTier::DetachedRam,
                        byte_offset: 4,
                        byte_len: 4,
                        layout: index_layout(4),
                        database_id,
                        root_format: RootFormatVersion::V1,
                        covered_through: 2,
                    },
                ]
                .into_boxed_slice(),
                _sealed: BootstrapReplayWitnessSeal,
            },
        }
    }

    fn compound_source() -> BootstrapPublicationSource {
        let mut source = source();
        let catalog = Arc::make_mut(&mut source.replay.catalog_snapshot)
            .relational_catalog
            .get_mut("bootstrap_table")
            .expect("fixture table");
        catalog.columns.push(RelationalColumn {
            id: 8,
            table_oid: 100,
            attnum: 2,
            name: "secondary".to_owned(),
            ty: SqlType::Date,
            domain: None,
            default: None,
            type_oid: SqlType::Date.postgres_oid(),
            type_size: SqlType::Date.type_size(),
        });
        catalog.indexes[0].key_columns.push("secondary".to_owned());
        source
            .replay
            .stable_ids
            .column_migrations
            .push(BootstrapColumnMigration {
                owner_display_table_oid: 100,
                legacy_column_id: 8,
                attnum: 2,
                stable_column_id: 8,
            });
        source.replay.stable_ids.column_high_water = 8;
        source.replay.stable_ids.checkpoint_column_high_water = 8;
        source.replay.current_tables[0]
            .enrolled_columns
            .push(BootstrapCurrentColumn {
                owner_display_table_oid: 100,
                legacy_column_id: 8,
                attnum: 2,
                stable_column_id: StableColumnId::new(8).expect("stable column ID"),
            });
        source.replay.current_tables[0].enrolled_indexes[0]
            .key_descriptors
            .push(BootstrapIndexKeyDescriptor {
                key_ordinal: 1,
                column_id: StableColumnId::new(8).expect("stable column ID"),
                shape_root: root(8),
            });
        source.replay.resources[2].byte_len = 40;
        source.replay.resources[2].layout = compound_table_layout(3);
        source.replay.resources[3].byte_len = 8;
        source.replay.resources[3].layout = compound_index_layout(4);
        source
    }

    fn empty_table_source() -> BootstrapPublicationSource {
        let mut source = source();
        source.replay.current_tables[0].logical_row_count = 0;
        let table = &mut source.replay.resources[2];
        table.byte_len = V1_EMPTY_TABLE_PAYLOAD_SENTINEL_BYTES;
        table.layout.row_count = 0;
        for role in &mut table.layout.roles {
            role.byte_offset = 0;
            role.byte_len = 0;
        }
        let index = &mut source.replay.resources[3];
        index.byte_len = V1_EMPTY_TABLE_PAYLOAD_SENTINEL_BYTES;
        index.layout.row_count = 0;
        for role in &mut index.layout.roles {
            role.byte_offset = 0;
            role.byte_len = 0;
        }
        source
    }

    fn empty_table_sidecar_source() -> BootstrapPublicationSource {
        let mut source = empty_table_source();
        let mut sidecar = source.replay.resources[2].clone();
        sidecar.kind = BootstrapPhysicalResourceKind::Sidecar;
        sidecar.resource_id = 5;
        sidecar.layout.id = BootstrapPhysicalLayoutDescriptorId::new(5).expect("layout descriptor");
        sidecar.ordinal = 0;
        sidecar.member_count = 1;
        source.replay.resources = source
            .replay
            .resources
            .into_vec()
            .into_iter()
            .chain(std::iter::once(sidecar))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        source
    }

    fn empty_index_sidecar_source() -> BootstrapPublicationSource {
        let mut source = empty_table_source();
        let mut sidecar = source.replay.resources[3].clone();
        sidecar.kind = BootstrapPhysicalResourceKind::Sidecar;
        sidecar.resource_id = 5;
        sidecar.layout.id = BootstrapPhysicalLayoutDescriptorId::new(5).expect("layout descriptor");
        sidecar.ordinal = 0;
        sidecar.member_count = 1;
        source.replay.resources = source
            .replay
            .resources
            .into_vec()
            .into_iter()
            .chain(std::iter::once(sidecar))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        source
    }

    /// Test support only: exercise the production source-to-lease handoff without exposing a
    /// synthetic lease constructor or any private replay/source fields to sibling test modules.
    pub(in crate::engine_data_generation) fn validated_materialization_lease_for_resources(
        access: &super::super::resources::BootstrapResourceLeaseAccess,
    ) -> Result<BootstrapMaterializationLease, DataGenerationError> {
        source().validate_into_materialization_lease(access)
    }

    /// Test support for the ignored CUDA owner-lifetime proof. It still takes the full sealed
    /// replay-validation path; only the authenticated detached tiers differ from the RAM fixture.
    pub(in crate::engine_data_generation) fn validated_materialization_lease_with_resident_payloads_for_resources(
        access: &super::super::resources::BootstrapResourceLeaseAccess,
    ) -> Result<BootstrapMaterializationLease, DataGenerationError> {
        let mut source = source();
        source.replay.resources[2].tier = BootstrapResourceStorageTier::Resident;
        source.replay.resources[3].tier = BootstrapResourceStorageTier::Resident;
        source.replay.resources[2].byte_offset = 0;
        source.replay.resources[3].byte_offset = 0;
        source.replay.resources[2].byte_len = 32;
        source.replay.resources[3].byte_len = 32;
        source.validate_into_materialization_lease(access)
    }

    #[test]
    fn bootstrap_source_moves_only_validated_owned_facts_into_an_uninstalled_candidate() {
        let candidate = source()
            .validate_into_uninstalled()
            .expect("valid bootstrap source");
        assert_eq!(candidate.replay.cut.covered_through, 2);
        assert_eq!(candidate.replay.resources.len(), 4);
        assert_eq!(candidate.replay.current_tables.len(), 1);
        assert_eq!(candidate.replay.terminal_status.entries.len(), 2);
        assert_eq!(
            candidate.replay.catalog.durable,
            candidate.replay.catalog.retained
        );
        assert_eq!(candidate.replay.catalog_snapshot.commit_seq, 2);
        assert_eq!(candidate.replay.root_format, RootFormatVersion::V1);
        assert_eq!(candidate.replay.database_id, database_id());
        let _ = candidate.replay.expected_database_root;
        let _ = candidate.replay.stable_ids;
    }

    #[test]
    fn bootstrap_rejects_invalid_cut_and_visibility_boundaries() {
        let mut stale_visible = source();
        stale_visible.replay.cut.visible_next = VisibleNext::new(2).expect("visible next");
        assert!(matches!(
            stale_visible.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap visibility boundary"
            ))
        ));

        let mut status_cut = source();
        status_cut.replay.terminal_status.covered_through = 1;
        assert!(matches!(
            status_cut.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap status identity"
            ))
        ));

        let mut foreign_status = source();
        foreign_status.replay.terminal_status.database_id =
            DatabaseId::new([0x43; 16]).expect("foreign database ID");
        assert!(matches!(
            foreign_status.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap status identity"
            ))
        ));
    }

    #[test]
    fn bootstrap_rejects_catalog_and_stable_id_migration_sabotage() {
        let mut catalog_mismatch = source();
        catalog_mismatch.replay.catalog.retained = catalog(99);
        assert!(matches!(
            catalog_mismatch.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap catalog identity"
            ))
        ));

        let mut absent_migration = source();
        absent_migration.replay.stable_ids.migration_complete = false;
        assert!(matches!(
            absent_migration.validate_into_uninstalled(),
            Err(DataGenerationError::Missing("stable-ID migration state"))
        ));

        let mut duplicate_stable_id = source();
        duplicate_stable_id
            .replay
            .stable_ids
            .relation_migrations
            .insert(
                1,
                BootstrapRelationMigration {
                    kind: BootstrapRelationKind::Table,
                    display_oid: 101,
                    stable_id: 10,
                },
            );
        assert!(matches!(
            duplicate_stable_id.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid("duplicate stable table ID"))
        ));

        let mut high_water = source();
        high_water.replay.stable_ids.index_high_water = 19;
        high_water.replay.stable_ids.checkpoint_index_high_water = 19;
        assert!(matches!(
            high_water.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid("index stable-ID high water"))
        ));

        let mut coherently_inflated_high_waters = source();
        coherently_inflated_high_waters
            .replay
            .stable_ids
            .table_high_water = 11;
        coherently_inflated_high_waters
            .replay
            .stable_ids
            .index_high_water = 22;
        coherently_inflated_high_waters
            .replay
            .stable_ids
            .column_high_water = 8;
        assert!(matches!(
            coherently_inflated_high_waters.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "stable-ID checkpoint high waters"
            ))
        ));

        let mut noncanonical_columns = source();
        noncanonical_columns
            .replay
            .stable_ids
            .column_migrations
            .push(BootstrapColumnMigration {
                owner_display_table_oid: 100,
                legacy_column_id: 1,
                attnum: 1,
                stable_column_id: 8,
            });
        noncanonical_columns.replay.stable_ids.column_high_water = 8;
        noncanonical_columns
            .replay
            .stable_ids
            .checkpoint_column_high_water = 8;
        assert!(matches!(
            noncanonical_columns.validate_into_uninstalled(),
            Err(DataGenerationError::NonCanonicalOrder(
                "stable column migration keys"
            ))
        ));

        let mut absent_current_table = source();
        absent_current_table.replay.current_tables[0].table_id =
            StableTableId::new(999).expect("absent current table ID");
        assert!(matches!(
            absent_current_table.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap current table mapping"
            ))
        ));

        let mut empty_current_tables = source();
        empty_current_tables.replay.current_tables.clear();
        assert!(matches!(
            empty_current_tables.validate_into_uninstalled(),
            Err(DataGenerationError::Missing("bootstrap current tables"))
        ));

        let mut duplicate_current_table = source();
        duplicate_current_table
            .replay
            .current_tables
            .push(duplicate_current_table.replay.current_tables[0].clone());
        assert!(matches!(
            duplicate_current_table.validate_into_uninstalled(),
            Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap current tables"
            ))
        ));

        let mut unordered_indexes = source();
        unordered_indexes.replay.current_tables[0].enrolled_indexes = vec![
            BootstrapExpectedIndex {
                display_index_oid: 201,
                index_id: StableIndexId::new(21).expect("index ID"),
                generation: IndexGeneration::new(2).expect("index generation"),
                root: root(8),
                shape_root: root(9),
                key_descriptors: vec![BootstrapIndexKeyDescriptor {
                    key_ordinal: 0,
                    column_id: StableColumnId::new(7).expect("stable column ID"),
                    shape_root: root(10),
                }],
            },
            BootstrapExpectedIndex {
                display_index_oid: 200,
                index_id: StableIndexId::new(20).expect("index ID"),
                generation: IndexGeneration::new(2).expect("index generation"),
                root: root(11),
                shape_root: root(12),
                key_descriptors: vec![BootstrapIndexKeyDescriptor {
                    key_ordinal: 0,
                    column_id: StableColumnId::new(7).expect("stable column ID"),
                    shape_root: root(13),
                }],
            },
        ];
        assert!(matches!(
            unordered_indexes.validate_into_uninstalled(),
            Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap enrolled stable index IDs"
            ))
        ));
    }

    #[test]
    fn bootstrap_rejects_missing_or_mismatched_current_column_enrollment() {
        let mut missing_column = source();
        missing_column.replay.current_tables[0]
            .enrolled_columns
            .clear();
        assert!(matches!(
            missing_column.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap catalog column enrollment"
            ))
        ));

        let mut mismatched_column = source();
        mismatched_column.replay.current_tables[0].enrolled_columns[0].stable_column_id =
            StableColumnId::new(8).expect("stable column ID");
        assert!(matches!(
            mismatched_column.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap current column mapping"
            ))
        ));

        let mut substituted_catalog_column = source();
        Arc::make_mut(&mut substituted_catalog_column.replay.catalog_snapshot)
            .relational_catalog
            .get_mut("bootstrap_table")
            .expect("fixture table")
            .columns[0]
            .id = 8;
        assert!(matches!(
            substituted_catalog_column.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap catalog column enrollment"
            ))
        ));
    }

    #[test]
    fn bootstrap_rejects_substituted_catalog_snapshot_or_enrollment() {
        let mut wrong_snapshot_cut = source();
        wrong_snapshot_cut.replay.catalog_snapshot = catalog_snapshot(1);
        assert!(matches!(
            wrong_snapshot_cut.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap replayed catalog witness"
            ))
        ));

        let mut substituted_catalog = source();
        Arc::make_mut(&mut substituted_catalog.replay.catalog_snapshot)
            .relational_catalog
            .get_mut("bootstrap_table")
            .expect("fixture table")
            .indexes[0]
            .oid = 201;
        assert!(matches!(
            substituted_catalog.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap catalog index enrollment"
            ))
        ));
    }

    #[test]
    fn bootstrap_rejects_incomplete_or_reordered_terminal_status_source() {
        let mut missing = source();
        missing.replay.terminal_status.entries.pop();
        assert!(matches!(
            missing.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid("bootstrap status coverage"))
        ));

        let mut reordered = source();
        reordered.replay.terminal_status.entries.swap(0, 1);
        assert!(matches!(
            reordered.validate_into_uninstalled(),
            Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap terminal status sequence"
            ))
        ));

        let mut duplicate_transaction = source();
        duplicate_transaction.replay.terminal_status.entries[1]
            .entry
            .transaction_id = duplicate_transaction.replay.terminal_status.entries[0]
            .entry
            .transaction_id;
        duplicate_transaction.replay.terminal_status.entries[1]
            .canonical
            .entry
            .transaction_id = duplicate_transaction.replay.terminal_status.entries[0]
            .entry
            .transaction_id;
        assert!(matches!(
            duplicate_transaction.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "duplicate bootstrap status transaction"
            ))
        ));

        let mut early_envelope_substitution = source();
        let substituted = early_envelope_substitution.replay.terminal_status.entries[1]
            .canonical
            .clone();
        early_envelope_substitution.replay.terminal_status.entries[0].canonical = substituted;
        assert!(matches!(
            early_envelope_substitution.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap terminal envelope provenance"
            ))
        ));

        let mut retry_commitment_substitution = source();
        retry_commitment_substitution.replay.terminal_status.entries[1]
            .canonical
            .entry
            .request_digest = root(9_998);
        assert!(matches!(
            retry_commitment_substitution.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap terminal envelope provenance"
            ))
        ));

        let mut tail = source();
        tail.replay.terminal_status.last_terminal_envelope = Some(root(9_999));
        assert!(matches!(
            tail.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap terminal-status tail"
            ))
        ));
    }

    #[test]
    fn bootstrap_rejects_resource_substitution_duplicate_and_noncurrent_owner() {
        let mut stale_resource = source();
        stale_resource.replay.resources[2].covered_through = 1;
        assert!(matches!(
            stale_resource.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource provenance"
            ))
        ));

        let mut duplicate_resource = source();
        duplicate_resource.replay.resources[3].resource_id = 3;
        assert!(matches!(
            duplicate_resource.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid("duplicate bootstrap resource"))
        ));

        let mut duplicate_layout = source();
        duplicate_layout.replay.resources[3].layout.id =
            duplicate_layout.replay.resources[2].layout.id;
        assert!(matches!(
            duplicate_layout.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "duplicate bootstrap physical layout descriptor"
            ))
        ));

        let mut empty_range = source();
        empty_range.replay.resources[2].byte_len = 0;
        assert!(matches!(
            empty_range.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap resource source range"
            ))
        ));

        let mut overflowing_range = source();
        overflowing_range.replay.resources[2].byte_offset = u64::MAX;
        assert!(matches!(
            overflowing_range.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap resource source range"
            ))
        ));

        let mut noncurrent_owner = source();
        noncurrent_owner.replay.resources[2].owner = BootstrapPhysicalResourceOwner::Table(
            StableTableId::new(999).expect("unknown table ID"),
        );
        assert!(matches!(
            noncurrent_owner.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap resource current table"
            ))
        ));

        let mut foreign_database = source();
        foreign_database.replay.resources[2].database_id =
            DatabaseId::new([0x43; 16]).expect("foreign database ID");
        assert!(matches!(
            foreign_database.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap resource provenance"
            ))
        ));

        let mut foreign_layout_column = source();
        foreign_layout_column.replay.resources[2].layout.roles[2].column_id =
            Some(StableColumnId::new(8).expect("foreign stable column ID"));
        assert!(matches!(
            foreign_layout_column.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap physical layout column enrollment"
            ))
        ));

        let mut wrong_layout_type = source();
        wrong_layout_type.replay.resources[2].layout.roles[2].sql_type = Some(SqlType::Date);
        assert!(matches!(
            wrong_layout_type.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap physical layout column type"
            ))
        ));

        let mut wrong_index_key_ordinal = source();
        wrong_index_key_ordinal.replay.resources[3].layout.roles[0].key_ordinal = Some(1);
        assert!(matches!(
            wrong_index_key_ordinal.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap index-key physical layout role"
            ))
        ));

        let mut wrong_index_key_shape = source();
        wrong_index_key_shape.replay.resources[3].layout.roles[0].key_shape_root = Some(root(99));
        assert!(matches!(
            wrong_index_key_shape.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap index physical layout key shape"
            ))
        ));
    }

    #[test]
    fn bootstrap_rejects_incomplete_or_misencoded_table_layout_roles() {
        let mut opaque_payload = source();
        let mut opaque = opaque_layout(3);
        opaque.roles[0].byte_len = opaque_payload.replay.resources[2].byte_len;
        opaque_payload.replay.resources[2].layout = opaque;
        assert!(matches!(
            opaque_payload.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap table physical layout grammar"
            ))
        ));

        let mut missing_value = source();
        let mut roles =
            std::mem::take(&mut missing_value.replay.resources[2].layout.roles).into_vec();
        roles.remove(1);
        for (ordinal, role) in roles.iter_mut().enumerate() {
            role.ordinal = u16::try_from(ordinal).expect("test role ordinal");
        }
        missing_value.replay.resources[2].layout.roles = roles.into_boxed_slice();
        assert!(matches!(
            missing_value.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap table physical layout value validity pairing"
            ))
        ));

        let mut missing_columns = source();
        let mut roles =
            std::mem::take(&mut missing_columns.replay.resources[2].layout.roles).into_vec();
        // Keep the mandatory stable-row-id role and remove the complete column pair so this
        // remains a direct coverage sabotage rather than failing the row-id invariant first.
        roles.drain(1..3);
        for (ordinal, role) in roles.iter_mut().enumerate() {
            role.ordinal = u16::try_from(ordinal).expect("test role ordinal");
        }
        missing_columns.replay.resources[2].layout.roles = roles.into_boxed_slice();
        assert!(matches!(
            missing_columns.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap table physical layout column coverage"
            ))
        ));

        let mut missing_mvcc = source();
        let mut roles =
            std::mem::take(&mut missing_mvcc.replay.resources[2].layout.roles).into_vec();
        roles.drain(3..);
        missing_mvcc.replay.resources[2].layout.roles = roles.into_boxed_slice();
        assert!(matches!(
            missing_mvcc.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap table physical layout MVCC coverage"
            ))
        ));

        let mut mismatched_pair = compound_source();
        let mut roles =
            std::mem::take(&mut mismatched_pair.replay.resources[2].layout.roles).into_vec();
        roles.remove(3);
        roles[1].column_id = Some(StableColumnId::new(8).expect("stable column ID"));
        roles[1].sql_type = Some(SqlType::Date);
        for (ordinal, role) in roles.iter_mut().enumerate() {
            role.ordinal = u16::try_from(ordinal).expect("test role ordinal");
        }
        mismatched_pair.replay.resources[2].layout.roles = roles.into_boxed_slice();
        assert!(matches!(
            mismatched_pair.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap table physical layout value validity pairing"
            ))
        ));

        for (storage_type, byte_stride, byte_len) in [
            (BootstrapPhysicalStorageType::Int8, 4, 4),
            (BootstrapPhysicalStorageType::Int4, 8, 4),
            (BootstrapPhysicalStorageType::Int4, 4, 3),
        ] {
            let mut malformed = source();
            let value = &mut malformed.replay.resources[2].layout.roles[2];
            value.storage_type = storage_type;
            value.byte_stride = NonZeroU64::new(byte_stride).expect("test stride");
            value.byte_len = byte_len;
            assert!(matches!(
                malformed.validate_into_uninstalled(),
                Err(DataGenerationError::Invalid(
                    "bootstrap physical layout storage geometry"
                ))
            ));
        }

        let mut opaque_index_payload = source();
        let mut opaque = opaque_layout(4);
        opaque.roles[0].byte_len = opaque_index_payload.replay.resources[3].byte_len;
        opaque_index_payload.replay.resources[3].layout = opaque;
        assert!(matches!(
            opaque_index_payload.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap index physical layout grammar"
            ))
        ));

        let mut incomplete_index_payload = compound_source();
        let mut roles =
            std::mem::take(&mut incomplete_index_payload.replay.resources[3].layout.roles)
                .into_vec();
        roles.pop();
        incomplete_index_payload.replay.resources[3].layout.roles = roles.into_boxed_slice();
        assert!(matches!(
            incomplete_index_payload.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap index physical layout key coverage"
            ))
        ));
    }

    #[test]
    fn bootstrap_catalog_index_order_is_independent_of_replay_and_physical_layout() {
        assert!(compound_source().validate_into_uninstalled().is_ok());

        let mut missing_descriptor = compound_source();
        missing_descriptor.replay.current_tables[0].enrolled_indexes[0]
            .key_descriptors
            .pop();
        assert!(matches!(
            missing_descriptor.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap index key descriptor coverage"
            ))
        ));

        let mut bad_catalog_first_key = compound_source();
        Arc::make_mut(&mut bad_catalog_first_key.replay.catalog_snapshot)
            .relational_catalog
            .get_mut("bootstrap_table")
            .expect("fixture table")
            .indexes[0]
            .column = "secondary".to_owned();
        assert!(matches!(
            bad_catalog_first_key.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap catalog index first key"
            ))
        ));

        let mut coherent_reorder = compound_source();
        let expected =
            &mut coherent_reorder.replay.current_tables[0].enrolled_indexes[0].key_descriptors;
        expected.swap(0, 1);
        for (ordinal, descriptor) in expected.iter_mut().enumerate() {
            descriptor.key_ordinal = u16::try_from(ordinal).expect("test key ordinal");
        }
        let roles = &mut coherent_reorder.replay.resources[3].layout.roles;
        roles.swap(0, 1);
        for (ordinal, role) in roles.iter_mut().enumerate() {
            let ordinal = u16::try_from(ordinal).expect("test role ordinal");
            role.ordinal = ordinal;
            role.key_ordinal = Some(ordinal);
        }
        assert!(matches!(
            coherent_reorder.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap index key catalog order"
            ))
        ));

        let mut coherent_replacement = compound_source();
        coherent_replacement.replay.current_tables[0].enrolled_indexes[0].key_descriptors[0]
            .column_id = StableColumnId::new(8).expect("stable column ID");
        coherent_replacement.replay.current_tables[0].enrolled_indexes[0].key_descriptors[0]
            .shape_root = root(8);
        let role = &mut coherent_replacement.replay.resources[3].layout.roles[0];
        role.column_id = Some(StableColumnId::new(8).expect("stable column ID"));
        role.sql_type = Some(SqlType::Date);
        role.key_shape_root = Some(root(8));
        assert!(matches!(
            coherent_replacement.validate_into_uninstalled(),
            Err(DataGenerationError::PredecessorMismatch(
                "bootstrap index key catalog order"
            ))
        ));
    }

    #[test]
    fn physical_storage_mapping_covers_every_admitted_sql_type() {
        let fixed = |storage, stride| Ok((storage, stride));
        assert_eq!(
            physical_value_storage(SqlType::Int2),
            fixed(BootstrapPhysicalStorageType::Int4, 4)
        );
        assert_eq!(
            physical_value_storage(SqlType::Int4),
            fixed(BootstrapPhysicalStorageType::Int4, 4)
        );
        assert_eq!(
            physical_value_storage(SqlType::Date),
            fixed(BootstrapPhysicalStorageType::Int4, 4)
        );
        assert_eq!(
            physical_value_storage(SqlType::Int8),
            fixed(BootstrapPhysicalStorageType::Int8, 8)
        );
        assert_eq!(
            physical_value_storage(SqlType::Timestamp),
            fixed(BootstrapPhysicalStorageType::Int8, 8)
        );
        assert_eq!(
            physical_value_storage(SqlType::Bool),
            fixed(BootstrapPhysicalStorageType::Bit, 1)
        );
        assert_eq!(
            physical_value_storage(SqlType::Numeric {
                precision: 9,
                scale: 2
            }),
            fixed(BootstrapPhysicalStorageType::Int128, 16)
        );
        assert_eq!(
            physical_value_storage(SqlType::Uuid),
            fixed(BootstrapPhysicalStorageType::Int128, 16)
        );
        assert!(matches!(
            physical_value_storage(SqlType::Text),
            Err(DataGenerationError::Invalid(
                "unsupported bootstrap V1 variable-width layout"
            ))
        ));
    }

    #[test]
    fn bootstrap_v1_rejects_ambiguous_text_roles_before_materialization() {
        let mut multi_row_value = source();
        multi_row_value.replay.current_tables[0].logical_row_count = 2;
        let catalog = Arc::make_mut(&mut multi_row_value.replay.catalog_snapshot)
            .relational_catalog
            .get_mut("bootstrap_table")
            .expect("fixture table");
        catalog.columns[0].ty = SqlType::Text;
        catalog.columns[0].type_oid = SqlType::Text.postgres_oid();
        catalog.columns[0].type_size = SqlType::Text.type_size();
        let table = &mut multi_row_value.replay.resources[2];
        table.byte_len = 55;
        table.layout.row_count = 2;
        table.layout.roles[0].byte_len = 16;
        table.layout.roles[1].sql_type = Some(SqlType::Text);
        table.layout.roles[1].byte_offset = 16;
        table.layout.roles[1].byte_len = 4;
        let value = &mut table.layout.roles[2];
        value.sql_type = Some(SqlType::Text);
        value.storage_type = BootstrapPhysicalStorageType::Bytes;
        value.byte_stride = NonZeroU64::new(1).expect("text stride");
        value.byte_offset = 20;
        value.byte_len = 3;
        table.layout.roles[3].byte_offset = 23;
        table.layout.roles[3].byte_len = 16;
        table.layout.roles[4].byte_offset = 39;
        table.layout.roles[4].byte_len = 16;
        assert!(matches!(
            multi_row_value.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "unsupported bootstrap V1 variable-width layout"
            ))
        ));

        let mut multi_row_index = source();
        let index = &mut multi_row_index.replay.resources[3];
        index.byte_len = 3;
        index.layout.row_count = 2;
        let key = &mut index.layout.roles[0];
        key.sql_type = Some(SqlType::Text);
        key.storage_type = BootstrapPhysicalStorageType::Bytes;
        key.byte_stride = NonZeroU64::new(1).expect("text stride");
        key.byte_len = 3;
        assert!(matches!(
            multi_row_index.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "unsupported bootstrap V1 variable-width layout"
            ))
        ));

        let mut zero_row_value = empty_table_source();
        let catalog = Arc::make_mut(&mut zero_row_value.replay.catalog_snapshot)
            .relational_catalog
            .get_mut("bootstrap_table")
            .expect("fixture table");
        catalog.columns[0].ty = SqlType::Text;
        catalog.columns[0].type_oid = SqlType::Text.postgres_oid();
        catalog.columns[0].type_size = SqlType::Text.type_size();
        zero_row_value.replay.resources[2].layout.roles[1].sql_type = Some(SqlType::Text);
        let value = &mut zero_row_value.replay.resources[2].layout.roles[2];
        value.sql_type = Some(SqlType::Text);
        value.storage_type = BootstrapPhysicalStorageType::Bytes;
        value.byte_stride = NonZeroU64::new(1).expect("text stride");
        assert!(matches!(
            zero_row_value.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "unsupported bootstrap V1 variable-width layout"
            ))
        ));

        let mut zero_row_index = empty_table_source();
        let key = &mut zero_row_index.replay.resources[3].layout.roles[0];
        key.sql_type = Some(SqlType::Text);
        key.storage_type = BootstrapPhysicalStorageType::Bytes;
        key.byte_stride = NonZeroU64::new(1).expect("text stride");
        assert!(matches!(
            zero_row_index.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "unsupported bootstrap V1 variable-width layout"
            ))
        ));
    }

    #[test]
    fn bootstrap_rejects_unsupported_physical_layout_encoding_versions_before_roles() {
        for version in [0, 2] {
            let mut malformed = source();
            malformed.replay.resources[2].layout.encoding_version = version;
            malformed.replay.resources[2].layout.roles = Box::new([]);
            assert!(matches!(
                malformed.validate_into_uninstalled(),
                Err(DataGenerationError::Invalid(
                    "unsupported bootstrap physical layout encoding version"
                ))
            ));
        }
    }

    #[test]
    fn bootstrap_requires_complete_current_table_and_index_resource_coverage() {
        let mut missing_table_payload = source();
        missing_table_payload.replay.resources = missing_table_payload
            .replay
            .resources
            .into_vec()
            .into_iter()
            .filter(|resource| resource.kind != BootstrapPhysicalResourceKind::TablePayload)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        assert!(matches!(
            missing_table_payload.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap table resource coverage"
            ))
        ));

        let mut missing_index_payload = source();
        missing_index_payload.replay.resources = missing_index_payload
            .replay
            .resources
            .into_vec()
            .into_iter()
            .filter(|resource| resource.kind != BootstrapPhysicalResourceKind::IndexPayload)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        assert!(matches!(
            missing_index_payload.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap index resource coverage"
            ))
        ));

        let mut un_enrolled_index = source();
        un_enrolled_index.replay.resources[3].owner = BootstrapPhysicalResourceOwner::Index {
            table_id: un_enrolled_index.replay.current_tables[0].table_id,
            index_id: StableIndexId::new(21).expect("un-enrolled index ID"),
        };
        assert!(matches!(
            un_enrolled_index.validate_into_uninstalled(),
            Err(DataGenerationError::Missing(
                "bootstrap resource index enrollment"
            ))
        ));
    }

    #[test]
    fn bootstrap_accepts_canonically_ordered_multi_shard_resources() {
        let mut multi_shard = source();
        multi_shard.replay.current_tables[0].logical_row_count = 2;
        let mut second_shard = multi_shard.replay.resources[2].clone();
        second_shard.resource_id = 5;
        second_shard.layout.id =
            BootstrapPhysicalLayoutDescriptorId::new(5).expect("layout descriptor");
        second_shard.ordinal = 1;
        second_shard.member_count = 2;
        second_shard.layout.row_start = 1;
        let mut resources = multi_shard.replay.resources.into_vec();
        resources[2].member_count = 2;
        resources.insert(3, second_shard);
        multi_shard.replay.resources = resources.into_boxed_slice();
        let candidate = multi_shard
            .validate_into_uninstalled()
            .expect("canonically ordered table shards");
        assert_eq!(candidate.replay.resources.len(), 5);

        let mut duplicate_ordinal = source();
        let mut repeated_shard = duplicate_ordinal.replay.resources[2].clone();
        repeated_shard.resource_id = 5;
        repeated_shard.layout.id =
            BootstrapPhysicalLayoutDescriptorId::new(5).expect("layout descriptor");
        repeated_shard.member_count = 2;
        let mut resources = duplicate_ordinal.replay.resources.into_vec();
        resources[2].member_count = 2;
        resources.insert(3, repeated_shard);
        duplicate_ordinal.replay.resources = resources.into_boxed_slice();
        assert!(matches!(
            duplicate_ordinal.validate_into_uninstalled(),
            Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap physical resources"
            ))
        ));
    }

    #[test]
    fn bootstrap_empty_table_typed_resources_use_canonical_v1_geometry() {
        let mut redundant_zero_shard = source();
        let mut zero_shard = redundant_zero_shard.replay.resources[2].clone();
        zero_shard.resource_id = 5;
        zero_shard.layout.id =
            BootstrapPhysicalLayoutDescriptorId::new(5).expect("layout descriptor");
        zero_shard.ordinal = 1;
        zero_shard.member_count = 2;
        zero_shard.layout.row_start = 1;
        zero_shard.layout.row_count = 0;
        for role in &mut zero_shard.layout.roles {
            role.byte_len = 0;
        }
        let mut resources = redundant_zero_shard.replay.resources.into_vec();
        resources[2].member_count = 2;
        resources.insert(3, zero_shard);
        redundant_zero_shard.replay.resources = resources.into_boxed_slice();
        assert!(matches!(
            redundant_zero_shard.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap nonempty table payload zero-row shard"
            ))
        ));

        assert!(empty_table_source().validate_into_uninstalled().is_ok());

        let mut duplicate_empty_resource = empty_table_source();
        let mut duplicate = duplicate_empty_resource.replay.resources[2].clone();
        duplicate.resource_id = 5;
        duplicate.layout.id =
            BootstrapPhysicalLayoutDescriptorId::new(5).expect("layout descriptor");
        duplicate.ordinal = 1;
        duplicate.member_count = 2;
        let mut resources = duplicate_empty_resource.replay.resources.into_vec();
        resources[2].member_count = 2;
        resources.insert(3, duplicate);
        duplicate_empty_resource.replay.resources = resources.into_boxed_slice();
        assert!(matches!(
            duplicate_empty_resource.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap empty table payload multiplicity"
            ))
        ));

        let mut noncanonical_empty_extent = empty_table_source();
        noncanonical_empty_extent.replay.resources[2].byte_len =
            V1_EMPTY_TABLE_PAYLOAD_SENTINEL_BYTES + 1;
        assert!(matches!(
            noncanonical_empty_extent.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap empty table typed resource V1 geometry"
            ))
        ));

        let mut noncanonical_empty_index_extent = empty_table_source();
        noncanonical_empty_index_extent.replay.resources[3].byte_len =
            V1_EMPTY_TABLE_PAYLOAD_SENTINEL_BYTES + 1;
        assert!(matches!(
            noncanonical_empty_index_extent.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap empty table typed resource V1 geometry"
            ))
        ));

        assert!(empty_table_sidecar_source()
            .validate_into_uninstalled()
            .is_ok());
        let mut noncanonical_table_sidecar = empty_table_sidecar_source();
        noncanonical_table_sidecar.replay.resources[4].byte_len =
            V1_EMPTY_TABLE_PAYLOAD_SENTINEL_BYTES + 1;
        assert!(matches!(
            noncanonical_table_sidecar.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap empty table typed resource V1 geometry"
            ))
        ));

        assert!(empty_index_sidecar_source()
            .validate_into_uninstalled()
            .is_ok());
        let mut noncanonical_index_sidecar = empty_index_sidecar_source();
        noncanonical_index_sidecar.replay.resources[4].byte_len =
            V1_EMPTY_TABLE_PAYLOAD_SENTINEL_BYTES + 1;
        assert!(matches!(
            noncanonical_index_sidecar.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap empty table typed resource V1 geometry"
            ))
        ));
    }

    #[test]
    fn bootstrap_rejects_nonexact_resource_group_membership() {
        let mut sole_nonzero_ordinal = source();
        sole_nonzero_ordinal.replay.resources[2].ordinal = 7;
        assert!(matches!(
            sole_nonzero_ordinal.validate_into_uninstalled(),
            Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap resource group ordinals"
            ))
        ));

        let mut ordinal_gap = source();
        let mut shard = ordinal_gap.replay.resources[2].clone();
        ordinal_gap.replay.resources[2].member_count = 2;
        shard.resource_id = 5;
        shard.layout.id = BootstrapPhysicalLayoutDescriptorId::new(5).expect("layout descriptor");
        shard.ordinal = 2;
        shard.member_count = 2;
        let mut resources = ordinal_gap.replay.resources.into_vec();
        resources.insert(3, shard);
        ordinal_gap.replay.resources = resources.into_boxed_slice();
        assert!(matches!(
            ordinal_gap.validate_into_uninstalled(),
            Err(DataGenerationError::NonCanonicalOrder(
                "bootstrap resource group ordinals"
            ))
        ));

        let mut inconsistent_member_count = source();
        let mut shard = inconsistent_member_count.replay.resources[2].clone();
        inconsistent_member_count.replay.resources[2].member_count = 2;
        shard.resource_id = 5;
        shard.layout.id = BootstrapPhysicalLayoutDescriptorId::new(5).expect("layout descriptor");
        shard.ordinal = 1;
        shard.member_count = 3;
        let mut resources = inconsistent_member_count.replay.resources.into_vec();
        resources.insert(3, shard);
        inconsistent_member_count.replay.resources = resources.into_boxed_slice();
        assert!(matches!(
            inconsistent_member_count.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap resource member count"
            ))
        ));

        let mut mismatched_base_multiplicity = source();
        let mut resources = mismatched_base_multiplicity.replay.resources.into_vec();
        let mut database_manifest = resources[0].clone();
        resources[0].member_count = 2;
        database_manifest.resource_id = 5;
        database_manifest.layout.id =
            BootstrapPhysicalLayoutDescriptorId::new(5).expect("layout descriptor");
        database_manifest.ordinal = 1;
        database_manifest.member_count = 2;
        resources.insert(1, database_manifest);
        let mut status_view = resources[2].clone();
        resources[2].member_count = 2;
        status_view.resource_id = 6;
        status_view.layout.id =
            BootstrapPhysicalLayoutDescriptorId::new(6).expect("layout descriptor");
        status_view.ordinal = 1;
        status_view.member_count = 2;
        resources.insert(3, status_view);
        mismatched_base_multiplicity.replay.resources = resources.into_boxed_slice();
        assert!(matches!(
            mismatched_base_multiplicity.validate_into_uninstalled(),
            Err(DataGenerationError::Invalid(
                "bootstrap base resource multiplicity"
            ))
        ));
    }

    #[test]
    fn bootstrap_contract_has_no_runtime_install_or_read_surface() {
        let source = include_str!("bootstrap_publication.rs");
        for forbidden in [
            ["Arc", "Swap"].concat(),
            ["Read", "State"].concat(),
            ["fn ", "install"].concat(),
            ["fn ", "publish"].concat(),
            ["fn ", "acquire"].concat(),
            ["fn ", "materialize"].concat(),
        ] {
            assert!(
                !source.contains(&forbidden),
                "bootstrap source unexpectedly exposes {forbidden}"
            );
        }
        assert!(
            source.contains(
                "struct BootstrapPublicationSource {\n    replay: BootstrapReplayWitness"
            ),
            "bootstrap source must consume exactly one sealed replay witness"
        );
        assert!(
            source.contains("exact_canonical_envelope")
                && source.contains("checkpoint_table_high_water")
                && source.contains("expected_database_root"),
            "sealed replay witness lost authenticated replay/checkpoint facts"
        );
        assert!(
            source.contains("pub(super) struct BootstrapMaterializationLease")
                && source.contains("pub(super) fn validate_into_materialization_lease")
                && source.contains("pub(super) fn resource_claims"),
            "bootstrap source must expose only the sealed materialization lease"
        );
        for forbidden in [
            ["pub(super) struct ", "BootstrapReplayWitness"].concat(),
            ["pub(super) struct ", "BootstrapStableIdState"].concat(),
            ["pub(super) struct ", "ReplayedTerminalStatusWitness"].concat(),
            ["pub(super) struct ", "BootstrapPhysicalResource"].concat(),
            ["pub(super) struct ", "UninstalledPublicationGeneration"].concat(),
            ["pub(super) fn ", "validate_into_uninstalled"].concat(),
            ["pub(super) fn ", "into_materialization_lease"].concat(),
            ["fn ", "from_snapshot"].concat(),
            ["fn ", "from_identity"].concat(),
            ["fn ", "from_entries"].concat(),
        ] {
            assert!(
                !source.contains(&forbidden),
                "bootstrap source exposes a generic replay relabel surface: {forbidden}"
            );
        }
    }
}
