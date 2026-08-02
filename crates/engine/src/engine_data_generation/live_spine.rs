//! The deliberately narrow, read-only WRITE-001 publication object.
//!
//! This imports exactly one immutable `FixedRadixMap` table entry from the V1 Rebuild witness.
//! It is not a generic mutable map importer: the witness carries one leaf and one 64-node path
//! only, with no row-map, status, index, or multi-table completion. A later DML publication must
//! invalidate this route through the one commit coordinator before advancing `committed_seq`.

use std::sync::{atomic::Ordering, Arc};

use gpu_db_execution::{
    OpaqueRuntimeGenerationRebuildProof, PreparedRuntimeGenerationRebuild, RecompactFill,
    RecompactSegment, RuntimeGenerationRebuildAttempt, RuntimeGenerationRebuildCompletion,
    RuntimeGenerationRebuildError, RuntimeGenerationRebuildInput,
    RuntimeGenerationRebuildPrepareError, RuntimeGenerationRebuildRoleSpan,
    RuntimeGenerationRebuildShard, RuntimeGenerationRebuildShardRoleSource,
    RuntimeGenerationRebuildShardRoleSources, RuntimeGenerationRebuildSource,
};
use gpu_db_sql::{Select, SqlType};
use gpu_db_wal::SealedInt4RebuildManifestV1;

use crate::{
    engine_state::CatalogSnapshot,
    rel_exec_helpers::{bind_relational_select, select_is_plain_view_scan},
    relational_model::{RelationalSelectResult, RelationalTable, ResidentDeviceNullBitmapLayout},
    resident_storage::RelationalResidentShard,
    Engine, EngineError, ExecuteError, Index,
};

use super::digest::{
    DatabaseId, DatabaseRoot, GpuCompletedDigest, RootFormatVersion, StableTableId, TableMapRoot,
    TableRoot,
};
use super::manifest::{
    FixedRadixMap, GpuRadixEmptyRoots, GpuRadixPathCompletion, GpuRadixPathNode, RadixLeafValue,
};
use super::DataGenerationError;

const SEALED_INT4_V1_STABLE_TABLE_ID: u64 = 1;
const SEALED_INT4_V1_STABLE_COLUMN_ID: u64 = 1;
const SEALED_INT4_V1_DATA_GENERATION: u64 = 1;

fn rebuild_error_message(error: &RuntimeGenerationRebuildError) -> String {
    match error {
        RuntimeGenerationRebuildError::Runtime(error) => error.to_string(),
        RuntimeGenerationRebuildError::DeviceRejected(status) => {
            format!("runtime-generation rebuild device rejected status {status}")
        }
    }
}

fn rebuild_prepare_error_message(error: &RuntimeGenerationRebuildPrepareError) -> String {
    match error {
        RuntimeGenerationRebuildPrepareError::Runtime(error) => error.to_string(),
        RuntimeGenerationRebuildPrepareError::AsyncTransportUnavailable => {
            "runtime-generation rebuild async transport unavailable".to_string()
        }
        RuntimeGenerationRebuildPrepareError::PinnedHostStagingUnavailable => {
            "runtime-generation rebuild pinned host staging unavailable".to_string()
        }
        RuntimeGenerationRebuildPrepareError::InvalidInput(detail) => {
            format!("runtime-generation rebuild invalid input: {detail}")
        }
    }
}

/// Root-free Rebuild inputs for the one sealed V1 grammar. Expected roots remain exclusively in
/// the durable manifest and enter this module only for the later comparator; sealing therefore
/// never fabricates a placeholder commitment before CUDA has produced one.
#[derive(Clone, Copy)]
pub(crate) struct SealedInt4RebuildMetadataV1 {
    database_id: [u8; 16],
    table_oid: u32,
    stable_table_id: u64,
    stable_table_high_water: u64,
    index_high_water: u64,
    data_generation: u64,
    logical_row_count: u64,
    column_owner_table_oid: u32,
    legacy_column_id: u32,
    stable_column_id: u64,
    stable_column_high_water: u64,
    attnum: i16,
    covered_through: Index,
    visible_next: Index,
}

impl SealedInt4RebuildMetadataV1 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        database_id: [u8; 16],
        table_oid: u32,
        stable_table_id: u64,
        stable_table_high_water: u64,
        index_high_water: u64,
        data_generation: u64,
        logical_row_count: u64,
        column_owner_table_oid: u32,
        legacy_column_id: u32,
        stable_column_id: u64,
        stable_column_high_water: u64,
        attnum: i16,
        covered_through: Index,
    ) -> Result<Self, DataGenerationError> {
        let visible_next = covered_through
            .checked_add(1)
            .ok_or(DataGenerationError::CountOverflow)?;
        if database_id == [0; 16]
            || table_oid == 0
            || stable_table_id == 0
            || stable_table_high_water < stable_table_id
            || data_generation == 0
            || logical_row_count == 0
            || column_owner_table_oid != table_oid
            || legacy_column_id == 0
            || stable_column_id == 0
            || stable_column_high_water < stable_column_id
            || attnum <= 0
            || covered_through == 0
        {
            return Err(DataGenerationError::Invalid(
                "sealed nullable-int4 rebuild metadata",
            ));
        }
        Ok(Self {
            database_id,
            table_oid,
            stable_table_id,
            stable_table_high_water,
            index_high_water,
            data_generation,
            logical_row_count,
            column_owner_table_oid,
            legacy_column_id,
            stable_column_id,
            stable_column_high_water,
            attnum,
            covered_through,
            visible_next,
        })
    }

    fn from_manifest(manifest: &SealedInt4RebuildManifestV1) -> Result<Self, DataGenerationError> {
        manifest
            .validate()
            .map_err(|_| DataGenerationError::Invalid("sealed nullable-int4 rebuild manifest"))?;
        let metadata = Self::new(
            manifest.database_id(),
            manifest.table_oid(),
            manifest.stable_table_id(),
            manifest.stable_table_high_water(),
            manifest.index_high_water(),
            manifest.data_generation(),
            manifest.logical_row_count(),
            manifest.column_owner_table_oid(),
            manifest.legacy_column_id(),
            manifest.stable_column_id(),
            manifest.stable_column_high_water(),
            manifest.attnum(),
            manifest.covered_through(),
        )?;
        if metadata.visible_next != manifest.visible_next() {
            return Err(DataGenerationError::Invalid(
                "sealed nullable-int4 manifest visibility",
            ));
        }
        Ok(metadata)
    }
}

/// A GPU-authenticated, one-table immutable read base retained as the sole leaf in the sealed V1
/// table map.
#[derive(Debug)]
struct SealedRecoveredReadBaseV1 {
    database_root: DatabaseRoot,
    table_id: StableTableId,
    table_root: TableRoot,
}

/// The installed read generation for the first live WRITE-001 spine.  The exact immutable shard
/// owners are captured once here and then handed directly to the GPU general read source.
#[derive(Debug)]
struct SealedInt4TableEntryV1 {
    root_format: RootFormatVersion,
    database_id: DatabaseId,
    visible_through: Index,
    table_oid: u32,
    table_name: String,
    column_id: u64,
    attnum: i16,
    data_generation: u64,
    catalog: Arc<CatalogSnapshot>,
    sealed_base: SealedRecoveredReadBaseV1,
    shards: Arc<[RelationalResidentShard]>,
}

/// The one V1 table-map leaf.  Its root is supplied by the device witness; this value merely
/// retains the immutable table generation that the map entry names.
#[derive(Clone, Debug)]
struct SealedInt4TableMapLeafV1 {
    publication: Arc<SealedInt4TableEntryV1>,
}

impl RadixLeafValue for SealedInt4TableMapLeafV1 {
    fn same_commitment(&self, other: &Self) -> bool {
        self.publication.table_id() == other.publication.table_id()
            && self.publication.table_root() == other.publication.table_root()
    }
}

type SealedInt4TableMapV1 = FixedRadixMap<StableTableId, TableMapRoot, SealedInt4TableMapLeafV1>;

/// The installed immutable publication-generation authority for the closed V1 recovery route.
/// It owns the persistent map itself, not just a detached table object.  The map contains exactly
/// one recovered nullable-INT4 table; broader table/status/index map import remains deferred.
#[derive(Debug)]
pub(crate) struct SealedInt4PublicationGenerationV1 {
    root_format: RootFormatVersion,
    database_id: DatabaseId,
    visible_through: Index,
    database_root: DatabaseRoot,
    table_id: StableTableId,
    table_oid: u32,
    tables: SealedInt4TableMapV1,
}

impl SealedInt4TableEntryV1 {
    /// Build the installed, immutable import only after the execution comparator has confirmed
    /// both semantic roots against the durable checkpoint manifest.  The opaque typed roots have
    /// already crossed the closed GPU handoff at this point; this constructor deliberately does
    /// no digest construction or comparison.
    fn from_verified_roots(
        manifest: &SealedInt4RebuildManifestV1,
        catalog: Arc<CatalogSnapshot>,
        table_name: String,
        table_root: TableRoot,
        database_root: DatabaseRoot,
        shards: Vec<RelationalResidentShard>,
    ) -> Result<Self, DataGenerationError> {
        let metadata = SealedInt4RebuildMetadataV1::from_manifest(manifest)?;
        validate_sealed_int4_catalog(&metadata, &catalog, &table_name)?;
        let database_id = DatabaseId::new(metadata.database_id)?;
        let table_id = StableTableId::new(metadata.stable_table_id)?;
        if catalog
            .relational_catalog
            .get(&table_name)
            .is_none_or(|table| table.oid != metadata.table_oid)
            || shards.is_empty()
        {
            return Err(DataGenerationError::PredecessorMismatch(
                "sealed nullable-int4 catalog or resident shard",
            ));
        }
        Ok(Self {
            root_format: RootFormatVersion::V1,
            database_id,
            visible_through: metadata.covered_through,
            table_oid: metadata.table_oid,
            table_name,
            column_id: metadata.stable_column_id,
            attnum: metadata.attnum,
            data_generation: metadata.data_generation,
            catalog,
            sealed_base: SealedRecoveredReadBaseV1 {
                database_root,
                table_id,
                table_root,
            },
            shards: shards.into(),
        })
    }

    /// Reader acceptance is exact on the persisted catalog object identity, visibility cut, and
    /// stable table identity.  It returns the retained generation itself so callers make one load
    /// and never combine it with later residency/catalog map loads.
    fn matches_reader(
        &self,
        catalog: &Arc<CatalogSnapshot>,
        visible_through: Index,
        table_name: &str,
    ) -> bool {
        self.root_format == RootFormatVersion::V1
            && self.visible_through == visible_through
            && self.table_name == table_name
            && Arc::ptr_eq(&self.catalog, catalog)
            && self
                .catalog
                .relational_catalog
                .get(table_name)
                .is_some_and(|table| table.oid == self.table_oid)
    }

    fn visible_through(&self) -> Index {
        self.visible_through
    }

    fn table_oid(&self) -> u32 {
        self.table_oid
    }

    fn table_id(&self) -> StableTableId {
        self.sealed_base.table_id
    }

    fn table_root(&self) -> TableRoot {
        self.sealed_base.table_root
    }

    fn database_root(&self) -> DatabaseRoot {
        self.sealed_base.database_root
    }

    /// The one served V1 route: an unfiltered `SELECT *` over the recovered sealed table.  Its
    /// source is assembled from this object's captured shard owners only; it never reloads a
    /// residency map by table name after acquiring the generation.
    fn execute_plain_select(
        &self,
        engine: &Engine,
        select: &Select,
        catalog: &Arc<CatalogSnapshot>,
        copin_s: Index,
    ) -> Option<Result<RelationalSelectResult, ExecuteError>> {
        if !select_is_plain_view_scan(select)
            || !self.matches_reader(catalog, copin_s, &select.table)
        {
            return None;
        }
        let table = match catalog.relational_catalog.get(&self.table_name).cloned() {
            Some(table) => table,
            None => {
                return Some(Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "sealed nullable-int4 reader lost its pinned table catalog".to_string(),
                ))));
            }
        };
        let bound = match bind_relational_select(&table, select) {
            Ok(bound) => bound,
            Err(error) => return Some(Err(error)),
        };
        let unified = match engine.build_sharded_unified_exec_source_from_shards(
            &table,
            None,
            copin_s,
            self.shards.iter().cloned().collect(),
        ) {
            Ok(unified) => unified,
            Err(error) => return Some(Err(error)),
        };
        #[cfg(any(test, feature = "probe-timing"))]
        engine.note_sealed_int4_direct_source_gpu_serve();
        Some(engine.execute_resident_grouped_via_general_with_binding(
            select,
            &table,
            bound,
            copin_s,
            Some(&unified.src),
            unified.visibility,
        ))
    }
}

impl SealedInt4PublicationGenerationV1 {
    /// Consume the exact one-entry table-map witness from a proof whose durable table/database
    /// roots have already matched. The host validates only map topology, cardinalities, and
    /// identity links while retaining opaque device completions; it never hashes a root.
    fn from_verified_table_map(
        publication: Arc<SealedInt4TableEntryV1>,
        completion: gpu_db_execution::RuntimeGenerationRebuildV1TableMapCompletion,
    ) -> Result<Self, DataGenerationError> {
        let table_id = publication.table_id();
        let table_oid = publication.table_oid();
        let root_format = publication.root_format;
        let database_id = publication.database_id;
        let visible_through = publication.visible_through();
        let database_root = publication.database_root();
        completion.consume(|empty_roots, leaf_root, path_roots| {
            if empty_roots.len() != 65 || path_roots.len() != 64 {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "sealed nullable-int4 table-map witness shape",
                ));
            }
            let empty_roots = GpuRadixEmptyRoots::from_verified_depths(
                empty_roots
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(depth, root)| {
                        (
                            depth as u8,
                            TableMapRoot::from_gpu_completion(
                                GpuCompletedDigest::from_cuda_completion(root),
                            ),
                        )
                    })
                    .collect(),
            )?;
            let path = GpuRadixPathCompletion {
                leaf_root: TableMapRoot::from_gpu_completion(
                    GpuCompletedDigest::from_cuda_completion(leaf_root),
                ),
                nodes: path_roots
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(depth, root)| GpuRadixPathNode {
                        depth: depth as u8,
                        subtree_count: 1,
                        root: TableMapRoot::from_gpu_completion(
                            GpuCompletedDigest::from_cuda_completion(root),
                        ),
                    })
                    .collect(),
            };
            let expected_root = path
                .nodes
                .first()
                .ok_or(DataGenerationError::GpuCompletionMismatch(
                    "sealed nullable-int4 table-map root",
                ))?
                .root;
            let leaf = SealedInt4TableMapLeafV1 {
                publication: Arc::clone(&publication),
            };
            let tables = SealedInt4TableMapV1::empty(empty_roots)?.substitute(
                table_id,
                None,
                Some(leaf),
                &path,
            )?;
            tables.validate()?;
            if tables.count() != 1 || tables.root() != expected_root {
                return Err(DataGenerationError::GpuCompletionMismatch(
                    "sealed nullable-int4 table-map import",
                ));
            }
            let installed = tables.get(table_id).ok_or(DataGenerationError::Missing(
                "sealed nullable-int4 table-map leaf",
            ))?;
            if !Arc::ptr_eq(&installed.publication, &publication) {
                return Err(DataGenerationError::PredecessorMismatch(
                    "sealed nullable-int4 table-map leaf publication",
                ));
            }
            Ok(Self {
                root_format,
                database_id,
                visible_through,
                database_root,
                table_id,
                table_oid,
                tables,
            })
        })
    }

    pub(crate) fn visible_through(&self) -> Index {
        self.visible_through
    }

    pub(crate) fn table_oid(&self) -> u32 {
        self.table_oid
    }

    /// Route through the imported persistent map, then use the exact immutable leaf captured by
    /// that map. No later table-name/residency lookup may assemble a different generation.
    pub(crate) fn execute_plain_select(
        &self,
        engine: &Engine,
        select: &Select,
        catalog: &Arc<CatalogSnapshot>,
        copin_s: Index,
    ) -> Option<Result<RelationalSelectResult, ExecuteError>> {
        if self.root_format != RootFormatVersion::V1 || self.visible_through != copin_s {
            return None;
        }
        let publication = &self.tables.get(self.table_id)?.publication;
        if publication.table_oid() != self.table_oid
            || publication.database_id != self.database_id
            || publication.database_root() != self.database_root
        {
            return None;
        }
        publication.execute_plain_select(engine, select, catalog, copin_s)
    }

    #[cfg(test)]
    pub(crate) fn validate_table_map_import_for_test(&self) -> Result<(), DataGenerationError> {
        self.tables.validate()?;
        if self.tables.count() != 1 || self.tables.get(self.table_id).is_none() {
            return Err(DataGenerationError::Invalid(
                "sealed nullable-int4 table-map test import",
            ));
        }
        Ok(())
    }
}

impl Engine {
    #[cfg(any(test, feature = "probe-timing"))]
    fn note_sealed_int4_direct_source_gpu_serve(&self) {
        self.read_state
            .sealed_int4_direct_source_gpu_served_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn sealed_int4_direct_source_gpu_served_total_for_test(&self) -> u64 {
        self.read_state
            .sealed_int4_direct_source_gpu_served_total
            .load(Ordering::Relaxed)
    }
}

/// Validate the deliberately closed V1 catalog/migration grammar.  The stable IDs are not
/// inferred from OID or attnum during recovery: they are read from the durable migration record,
/// whose one-table bootstrap allocation is explicitly fixed at one.
fn validate_sealed_int4_catalog(
    metadata: &SealedInt4RebuildMetadataV1,
    catalog: &CatalogSnapshot,
    table_name: &str,
) -> Result<(), DataGenerationError> {
    if catalog.relational_catalog.len() != 1
        || metadata.stable_table_id != SEALED_INT4_V1_STABLE_TABLE_ID
        || metadata.stable_table_high_water != SEALED_INT4_V1_STABLE_TABLE_ID
        || metadata.stable_column_id != SEALED_INT4_V1_STABLE_COLUMN_ID
        || metadata.stable_column_high_water != SEALED_INT4_V1_STABLE_COLUMN_ID
        || metadata.index_high_water != 0
        || metadata.data_generation != SEALED_INT4_V1_DATA_GENERATION
    {
        return Err(DataGenerationError::Invalid(
            "sealed nullable-int4 v1 migration grammar",
        ));
    }
    let table = catalog
        .relational_catalog
        .get(table_name)
        .filter(|table| table.oid == metadata.table_oid)
        .ok_or(DataGenerationError::PredecessorMismatch(
            "sealed nullable-int4 table OID",
        ))?;
    validate_sealed_int4_table_shape(table, metadata)?;
    Ok(())
}

fn validate_sealed_int4_table_shape(
    table: &RelationalTable,
    metadata: &SealedInt4RebuildMetadataV1,
) -> Result<(), DataGenerationError> {
    let Some(column) = table.columns.first() else {
        return Err(DataGenerationError::Invalid("sealed nullable-int4 column"));
    };
    if table.columns.len() != 1
        || !table.indexes.is_empty()
        || table.oid != metadata.table_oid
        || column.id != metadata.legacy_column_id
        || column.table_oid != metadata.column_owner_table_oid
        || column.attnum != metadata.attnum
        || column.ty != SqlType::Int4
        || column.type_oid != 23
        || column.type_size != 4
    {
        return Err(DataGenerationError::Invalid(
            "sealed nullable-int4 v1 table shape",
        ));
    }
    Ok(())
}

impl Engine {
    /// Compact the final replayed device generation into a private dense immutable base before
    /// Rebuild.  Open shards reserve append headroom, so authenticating their capacity-strided
    /// payload directly would make the proof depend on unwritten slots.  This performs only
    /// device-to-device copies (apart from the eight-byte row-count header), retains row-id and
    /// creation-stamp owners, and never reads or recreates relational values on the host.
    pub(crate) fn seal_recovered_int4_shards_for_rebuild(
        &self,
        table: &RelationalTable,
        shards: &[RelationalResidentShard],
    ) -> Result<Vec<RelationalResidentShard>, EngineError> {
        if table.columns.len() != 1 || table.columns[0].ty != SqlType::Int4 || shards.is_empty() {
            return Err(EngineError::ApplyFailed(
                "sealed nullable-int4 dense recovery source shape".to_string(),
            ));
        }
        let runtime = self.cuda_driver_probe_runtime();
        let mut sealed = Vec::with_capacity(shards.len());
        for shard in shards {
            let row_count = u64::try_from(shard.row_count).map_err(|_| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 dense recovery row count overflow".to_string(),
                )
            })?;
            if shard.schema != table.schema
                || shard.table != table.name
                || shard.resident_device_int4_columns.len() != 1
                || shard.resident_device_int4_columns.first() != Some(&table.columns[0].name)
                || !shard.resident_device_int8_columns.is_empty()
                || !shard.resident_device_numeric_columns.is_empty()
                || !shard.resident_device_bool_columns.is_empty()
                || !shard.resident_device_text_columns.is_empty()
                || shard.resident_device_null_columns.len() > 1
                || shard
                    .resident_device_null_columns
                    .iter()
                    .any(|layout| layout.name != table.columns[0].name)
                // V1 seals only a current-live image.  Retaining a tombstone sidecar would require
                // a visibility-filtering compaction, which is intentionally outside this narrow
                // immutable base rather than something a proof-only path may silently omit.
                || shard.deleted_by_region.is_some()
            {
                return Err(EngineError::ApplyFailed(format!(
                    "sealed nullable-int4 dense recovery shard shape: rows={row_count}, schema_matches={}, table_matches={}, has_tombstones={}",
                    shard.schema == table.schema,
                    shard.table == table.name,
                    shard.deleted_by_region.is_some(),
                )));
            }
            // CREATE can leave a zero-row bootstrap shard in front of its first dense rollover.
            // It contributes no row/value/validity/sidecar bytes to the current image, and the
            // successor starts at the same logical row offset, so omit it rather than requiring
            // a fake empty Rebuild role source.
            if row_count == 0 {
                continue;
            }
            let payload = Arc::clone(shard.device_memory.as_ref().ok_or_else(|| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 dense recovery payload owner".to_string(),
                )
            })?);
            let row_ids = Arc::clone(shard.row_id_region.as_ref().ok_or_else(|| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 dense recovery row-id owner".to_string(),
                )
            })?);
            let created_by = Arc::clone(shard.created_by_region.as_ref().ok_or_else(|| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 dense recovery created-by owner".to_string(),
                )
            })?);
            let values_bytes = row_count.checked_mul(4).ok_or_else(|| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 dense recovery value span overflow".to_string(),
                )
            })?;
            let ids_bytes = row_count.checked_mul(8).ok_or_else(|| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 dense recovery ID span overflow".to_string(),
                )
            })?;
            let validity_bytes = row_count
                .checked_add(31)
                .and_then(|rows| rows.checked_div(32))
                .and_then(|words| words.checked_mul(4))
                .ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "sealed nullable-int4 dense recovery validity span overflow".to_string(),
                    )
                })?;
            let nullable = shard
                .resident_device_null_columns
                .iter()
                .find(|layout| layout.name == table.columns[0].name)
                .cloned();
            let value_end = 8_u64.checked_add(values_bytes).ok_or_else(|| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 dense recovery value range overflow".to_string(),
                )
            })?;
            let validity_end = nullable
                .as_ref()
                .map(|layout| {
                    layout
                        .bitmap_byte_offset
                        .checked_add(validity_bytes)
                        .ok_or_else(|| {
                            EngineError::ApplyFailed(
                                "sealed nullable-int4 dense recovery validity range overflow"
                                    .to_string(),
                            )
                        })
                })
                .transpose()?;
            if value_end > payload.metadata().allocated_bytes
                || validity_end.is_some_and(|end| end > payload.metadata().allocated_bytes)
                || ids_bytes > row_ids.metadata().allocated_bytes
                || ids_bytes > created_by.metadata().allocated_bytes
            {
                return Err(EngineError::ApplyFailed(
                    "sealed nullable-int4 dense recovery source range".to_string(),
                ));
            }
            let payload_bytes = 8_u64
                .checked_add(values_bytes)
                .and_then(|bytes| {
                    nullable
                        .as_ref()
                        .map_or(Some(bytes), |_| bytes.checked_add(validity_bytes))
                })
                .ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "sealed nullable-int4 dense recovery payload overflow".to_string(),
                    )
                })?;
            let mut header = Vec::with_capacity(8);
            header.extend_from_slice(&row_count.to_le_bytes());
            let mut segments = vec![RecompactSegment {
                src_device_ptr: payload.device_ptr(),
                src_byte_offset: 8,
                dst_byte_offset: 8,
                byte_len: values_bytes,
            }];
            if let Some(layout) = &nullable {
                segments.push(RecompactSegment {
                    src_device_ptr: payload.device_ptr(),
                    src_byte_offset: layout.bitmap_byte_offset,
                    dst_byte_offset: 8 + values_bytes,
                    byte_len: validity_bytes,
                });
            }
            let dense_payload = Arc::new(
                runtime
                    .retain_device_memory_recompacted(
                        shard.gpu_id,
                        payload_bytes,
                        &header,
                        &[],
                        &segments,
                    )
                    .map_err(|error| {
                        EngineError::ApplyFailed(format!(
                            "sealed nullable-int4 dense payload copy: {error}"
                        ))
                    })?,
            );
            let copy_sidecar = |source: &Arc<gpu_db_execution::CudaResidentDeviceMemory>| {
                runtime
                    .retain_device_memory_recompacted(
                        shard.gpu_id,
                        ids_bytes,
                        &[],
                        &[],
                        &[RecompactSegment {
                            src_device_ptr: source.device_ptr(),
                            src_byte_offset: 0,
                            dst_byte_offset: 0,
                            byte_len: ids_bytes,
                        }],
                    )
                    .map(Arc::new)
                    .map_err(|error| {
                        EngineError::ApplyFailed(format!(
                            "sealed nullable-int4 dense sidecar copy: {error}"
                        ))
                    })
            };
            let dense_row_ids = copy_sidecar(&row_ids)?;
            let dense_created_by = copy_sidecar(&created_by)?;
            sealed.push(RelationalResidentShard {
                shard_id: shard.shard_id,
                row_start: shard.row_start,
                row_count: shard.row_count,
                history_floor_index: shard.history_floor_index,
                capacity: shard.row_count,
                int4_appendable: false,
                resident_device_int4_column_stats: shard.resident_device_int4_column_stats.clone(),
                resident_bytes: payload_bytes,
                allocated_bytes: payload_bytes,
                count_header_byte_offset: 0,
                resident_device_int4_columns: shard.resident_device_int4_columns.clone(),
                resident_device_int8_columns: Vec::new(),
                resident_device_numeric_columns: Vec::new(),
                resident_device_bool_columns: Vec::new(),
                resident_device_text_columns: Vec::new(),
                resident_device_null_columns: nullable
                    .map(|_| ResidentDeviceNullBitmapLayout {
                        name: table.columns[0].name.clone(),
                        bitmap_byte_offset: 8 + values_bytes,
                    })
                    .into_iter()
                    .collect(),
                gpu_id: shard.gpu_id,
                schema: shard.schema.clone(),
                table: shard.table.clone(),
                point_route_generation: Arc::new(()),
                device_memory_proof: Some(dense_payload.metadata().clone()),
                invalidated_by_txn_id: None,
                invalidated_at_index: None,
                invalidated_by_memory_pressure: false,
                memory_pressure_active: false,
                device_memory: Some(dense_payload),
                deleted_by_region: None,
                created_by_region: Some(dense_created_by),
                row_id_region: Some(dense_row_ids),
                max_created_by: shard.max_created_by,
            });
        }
        if sealed.is_empty() {
            return Err(EngineError::ApplyFailed(
                "sealed nullable-int4 recovery has no live rows".to_string(),
            ));
        }
        Ok(sealed)
    }

    /// Run the sole production GPU Rebuild operator against direct shard owners.  This is shared
    /// by sealing and recovery comparison; neither caller has a CPU hashing or alternate import
    /// path. `metadata` is root-free and already carries the explicit one-table migration IDs.
    pub(crate) fn rebuild_sealed_int4_v1(
        &self,
        metadata: &SealedInt4RebuildMetadataV1,
        catalog: &Arc<CatalogSnapshot>,
        table_name: &str,
        shards: &[RelationalResidentShard],
    ) -> Result<OpaqueRuntimeGenerationRebuildProof, EngineError> {
        validate_sealed_int4_catalog(metadata, catalog, table_name)
            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
        let table = catalog.relational_catalog.get(table_name).ok_or_else(|| {
            EngineError::ApplyFailed("sealed nullable-int4 table disappeared".to_string())
        })?;
        let runtime = self.cuda_driver_probe_runtime();
        if shards.is_empty() {
            return Err(EngineError::ApplyFailed(
                "sealed nullable-int4 rebuild has no resident shards".to_string(),
            ));
        }

        let mut row_start = 0_u64;
        let mut rebuild_shards = Vec::with_capacity(shards.len());
        for shard in shards {
            let row_count = u64::try_from(shard.row_count).map_err(|_| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 shard row count overflow".to_string(),
                )
            })?;
            if row_count == 0
                || shard.capacity != shard.row_count
                || u64::try_from(shard.row_start).ok() != Some(row_start)
                || shard.schema != table.schema
                || shard.table != table.name
                || shard.deleted_by_region.is_some()
                || shard.max_created_by > metadata.covered_through
            {
                return Err(EngineError::ApplyFailed(
                    "sealed nullable-int4 shard is not a dense current-live image".to_string(),
                ));
            }
            let values = Arc::clone(shard.device_memory.as_ref().ok_or_else(|| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 shard has no payload owner".to_string(),
                )
            })?);
            let row_ids = Arc::clone(shard.row_id_region.as_ref().ok_or_else(|| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 shard has no row-id owner".to_string(),
                )
            })?);
            let created_by = Arc::clone(shard.created_by_region.as_ref().ok_or_else(|| {
                EngineError::ApplyFailed(
                    "sealed nullable-int4 shard has no created-by owner".to_string(),
                )
            })?);
            let ids_bytes = row_count.checked_mul(8).ok_or_else(|| {
                EngineError::ApplyFailed("sealed nullable-int4 row-id byte overflow".to_string())
            })?;
            let values_bytes = row_count.checked_mul(4).ok_or_else(|| {
                EngineError::ApplyFailed("sealed nullable-int4 value byte overflow".to_string())
            })?;
            let validity_bytes = row_count
                .checked_add(31)
                .and_then(|rows| rows.checked_div(32))
                .and_then(|words| words.checked_mul(4))
                .ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "sealed nullable-int4 validity byte overflow".to_string(),
                    )
                })?;
            let target = runtime
                .runtime_generation_rebuild_target(shard.gpu_id)
                .map_err(|error| {
                    EngineError::ApplyFailed(format!(
                        "sealed nullable-int4 rebuild target: {error}"
                    ))
                })?;
            let deleted_by = Arc::new(
                runtime
                    .retain_device_memory_recompacted(
                        shard.gpu_id,
                        ids_bytes,
                        &[],
                        &[RecompactFill {
                            byte_offset: 0,
                            len: ids_bytes,
                            // Rebuild hashes the unsigned deletion stamp, where all-one is the
                            // canonical immutable current-live sentinel.  This buffer is private
                            // to the proof fence and is never a serving visibility sidecar.
                            fill_byte: u8::MAX,
                        }],
                        &[],
                    )
                    .map_err(|error| {
                        EngineError::ApplyFailed(format!(
                            "sealed nullable-int4 deleted-by fill: {error}"
                        ))
                    })?,
            );
            let (validity, validity_offset) = match shard
                .resident_device_null_columns
                .iter()
                .find(|layout| layout.name == table.columns[0].name)
            {
                Some(layout) => (Arc::clone(&values), layout.bitmap_byte_offset),
                None => {
                    // Residency elides the bitmap when every row is valid. Rebuild still hashes
                    // its canonical bitmap role, whose unused final-word bits MUST be zero. Fill
                    // whole words on the device and D2D-copy one four-byte control constant for
                    // a partial final word; never synthesize or read relational values on host.
                    let complete_words = row_count / 32;
                    let tail_bits = row_count % 32;
                    let validity = if tail_bits == 0 {
                        Arc::new(
                            runtime
                                .retain_device_memory_recompacted(
                                    shard.gpu_id,
                                    validity_bytes,
                                    &[],
                                    &[RecompactFill {
                                        byte_offset: 0,
                                        len: validity_bytes,
                                        fill_byte: u8::MAX,
                                    }],
                                    &[],
                                )
                                .map_err(|error| {
                                    EngineError::ApplyFailed(format!(
                                        "sealed nullable-int4 all-valid fill: {error}"
                                    ))
                                })?,
                        )
                    } else {
                        let tail_word = (1_u32 << tail_bits) - 1;
                        let tail_source = runtime
                            .retain_device_memory_recompacted(
                                shard.gpu_id,
                                4,
                                &tail_word.to_le_bytes(),
                                &[],
                                &[],
                            )
                            .map_err(|error| {
                                EngineError::ApplyFailed(format!(
                                    "sealed nullable-int4 all-valid tail source: {error}"
                                ))
                            })?;
                        let mut fills = Vec::new();
                        if complete_words != 0 {
                            fills.push(RecompactFill {
                                byte_offset: 0,
                                len: complete_words * 4,
                                fill_byte: u8::MAX,
                            });
                        }
                        Arc::new(
                            runtime
                                .retain_device_memory_recompacted(
                                    shard.gpu_id,
                                    validity_bytes,
                                    &[],
                                    &fills,
                                    &[RecompactSegment {
                                        src_device_ptr: tail_source.device_ptr(),
                                        src_byte_offset: 0,
                                        dst_byte_offset: complete_words * 4,
                                        byte_len: 4,
                                    }],
                                )
                                .map_err(|error| {
                                    EngineError::ApplyFailed(format!(
                                        "sealed nullable-int4 all-valid tail fill: {error}"
                                    ))
                                })?,
                        )
                    };
                    (validity, 0)
                }
            };
            let source = |memory: Arc<gpu_db_execution::CudaResidentDeviceMemory>,
                          span: RuntimeGenerationRebuildRoleSpan| {
                RuntimeGenerationRebuildShardRoleSource::new(
                    RuntimeGenerationRebuildSource::resident(
                        Arc::clone(&memory),
                        0,
                        memory.metadata().allocated_bytes,
                    ),
                    span,
                )
            };
            rebuild_shards.push(RuntimeGenerationRebuildShard::from_role_sources(
                row_start,
                row_count,
                RuntimeGenerationRebuildShardRoleSources {
                    stable_row_ids: source(
                        row_ids,
                        RuntimeGenerationRebuildRoleSpan {
                            byte_offset: 0,
                            byte_len: ids_bytes,
                        },
                    ),
                    validity: source(
                        validity,
                        RuntimeGenerationRebuildRoleSpan {
                            byte_offset: validity_offset,
                            byte_len: validity_bytes,
                        },
                    ),
                    values: source(
                        values,
                        RuntimeGenerationRebuildRoleSpan {
                            byte_offset: 8,
                            byte_len: values_bytes,
                        },
                    ),
                    created_by: source(
                        created_by,
                        RuntimeGenerationRebuildRoleSpan {
                            byte_offset: 0,
                            byte_len: ids_bytes,
                        },
                    ),
                    deleted_by: source(
                        deleted_by,
                        RuntimeGenerationRebuildRoleSpan {
                            byte_offset: 0,
                            byte_len: ids_bytes,
                        },
                    ),
                },
            ));
            // Every V1 table is intentionally one GPU.  Rebuild cannot combine primary
            // contexts; a second shard on another GPU is rejected before any enqueue.
            if target.device_ordinal() != shards[0].gpu_id {
                return Err(EngineError::ApplyFailed(
                    "sealed nullable-int4 v1 does not span GPUs".to_string(),
                ));
            }
            row_start = row_start.checked_add(row_count).ok_or_else(|| {
                EngineError::ApplyFailed("sealed nullable-int4 row coverage overflow".to_string())
            })?;
        }
        if row_start != metadata.logical_row_count {
            return Err(EngineError::ApplyFailed(
                "sealed nullable-int4 logical row count mismatch".to_string(),
            ));
        }
        let target = runtime
            .runtime_generation_rebuild_target(shards[0].gpu_id)
            .map_err(|error| {
                EngineError::ApplyFailed(format!("sealed nullable-int4 rebuild target: {error}"))
            })?;
        let input = RuntimeGenerationRebuildInput::new(
            target,
            RuntimeGenerationRebuildAttempt::new(metadata.visible_next).map_err(|error| {
                EngineError::ApplyFailed(format!("sealed nullable-int4 rebuild attempt: {error:?}"))
            })?,
            metadata.database_id,
            metadata.stable_table_id,
            metadata.data_generation,
            metadata.covered_through,
            metadata.logical_row_count,
            metadata.stable_column_id,
            metadata.attnum,
            23,
            4,
            rebuild_shards.into_boxed_slice(),
        );
        let prepared = PreparedRuntimeGenerationRebuild::prepare(input).map_err(|failure| {
            EngineError::ApplyFailed(format!(
                "sealed nullable-int4 rebuild prepare: {}",
                rebuild_prepare_error_message(failure.error())
            ))
        })?;
        match prepared.enqueue().complete() {
            RuntimeGenerationRebuildCompletion::Quiesced(Ok(proof)) => Ok(proof),
            RuntimeGenerationRebuildCompletion::Quiesced(Err(error)) => {
                Err(EngineError::ApplyFailed(format!(
                    "sealed nullable-int4 rebuild completion: {}",
                    rebuild_error_message(&error)
                )))
            }
            RuntimeGenerationRebuildCompletion::UnknownQuiescence(unknown) => {
                // The owner is dropped here, which makes one bounded drain attempt and parks
                // resources if that still cannot prove quiescence. Do not spin on a poisoned
                // context: the lifecycle wrapper recognizes the preserved CUDA text and
                // retries the immutable recovery source once with a fresh context.
                Err(EngineError::ApplyFailed(format!(
                    "sealed nullable-int4 rebuild unknown quiescence: {}",
                    rebuild_error_message(unknown.error())
                )))
            }
        }
    }

    /// Consume a successful recovery proof only after its two semantic commitments match the
    /// durable checkpoint. The comparison closure is the only place opaque GPU roots become
    /// typed engine roots, then imports the one sealed table-map path into the immutable reader
    /// generation.
    pub(crate) fn import_recovered_sealed_int4_generation(
        &self,
        manifest: Arc<SealedInt4RebuildManifestV1>,
        catalog: Arc<CatalogSnapshot>,
        table_name: String,
        shards: Arc<[RelationalResidentShard]>,
    ) -> Result<Arc<SealedInt4PublicationGenerationV1>, EngineError> {
        let table = catalog.relational_catalog.get(&table_name).ok_or_else(|| {
            EngineError::Durability(
                "sealed nullable-int4 recovery table is absent from pinned catalog".to_string(),
            )
        })?;
        let sealed_shards = self.seal_recovered_int4_shards_for_rebuild(table, &shards)?;
        let metadata = SealedInt4RebuildMetadataV1::from_manifest(&manifest)
            .map_err(|error| EngineError::Durability(error.to_string()))?;
        let proof =
            self.rebuild_sealed_int4_v1(&metadata, &catalog, &table_name, &sealed_shards)?;
        let expected = manifest
            .with_expected_roots(|table_root, database_root| {
                let mut bytes = [0_u8;
                    gpu_db_execution::RUNTIME_GENERATION_REBUILD_V1_DURABLE_COMMITMENT_BYTES];
                bytes[..32].copy_from_slice(table_root);
                bytes[32..].copy_from_slice(database_root);
                gpu_db_execution::RuntimeGenerationRebuildV1DurableCommitments::decode_durable(
                    &bytes,
                )
            })
            .map_err(|error| {
                EngineError::Durability(format!(
                    "sealed nullable-int4 durable commitment decode: {error:?}"
                ))
            })?;
        let generation = proof
            .compare_durable_v1_commitments_with_table_map(
                &expected,
                |_, table_root, table_map, database_root| {
                    let table_root = TableRoot::from_gpu_completion(
                        GpuCompletedDigest::from_cuda_completion(table_root),
                    );
                    let database_root = DatabaseRoot::from_gpu_completion(
                        GpuCompletedDigest::from_cuda_completion(database_root),
                    );
                    let publication = SealedInt4TableEntryV1::from_verified_roots(
                        &manifest,
                        catalog,
                        table_name,
                        table_root,
                        database_root,
                        sealed_shards,
                    )?;
                    SealedInt4PublicationGenerationV1::from_verified_table_map(
                        Arc::new(publication),
                        table_map,
                    )
                    .map(Arc::new)
                },
            )
            .map_err(|_| {
                EngineError::Durability(
                    "sealed nullable-int4 rebuild root mismatch during recovery".to_string(),
                )
            })?;
        generation.map_err(|error| EngineError::Durability(error.to_string()))
    }
}

impl Engine {
    pub(crate) fn sealed_int4_recovery_capture_is_armed_for(&self, table_oid: u32) -> bool {
        table_oid != 0
            && self
                .read_state
                .sealed_int4_capture_table_oid
                .load(Ordering::Acquire)
                == table_oid
    }

    /// Arm a private/offline recovery builder to capture the exact shard owners that admission
    /// publishes for the one table.  The caller is quiescent; a second arm or stale capture is an
    /// invariant failure rather than a chance to mix generations.
    pub(crate) fn arm_sealed_int4_recovery_capture(
        &self,
        table_oid: u32,
    ) -> Result<(), DataGenerationError> {
        if table_oid == 0
            || self
                .read_state
                .sealed_int4_capture_table_oid
                .compare_exchange(0, table_oid, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(DataGenerationError::Invalid(
                "sealed nullable-int4 recovery capture arm",
            ));
        }
        let capture = self
            .read_state
            .sealed_int4_recovery_capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if capture.is_some() {
            self.read_state
                .sealed_int4_capture_table_oid
                .store(0, Ordering::Release);
            return Err(DataGenerationError::Invalid(
                "stale sealed nullable-int4 recovery capture",
            ));
        }
        Ok(())
    }

    pub(crate) fn stage_sealed_int4_recovery_manifest(
        &self,
        manifest: Arc<SealedInt4RebuildManifestV1>,
    ) -> Result<(), DataGenerationError> {
        manifest
            .validate()
            .map_err(|_| DataGenerationError::Invalid("sealed nullable-int4 rebuild manifest"))?;
        self.arm_sealed_int4_recovery_capture(manifest.table_oid())?;
        let mut pending = self
            .read_state
            .sealed_int4_recovery_manifest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if pending.replace(manifest).is_some() {
            self.read_state
                .sealed_int4_capture_table_oid
                .store(0, Ordering::Release);
            return Err(DataGenerationError::Invalid(
                "multiple sealed nullable-int4 recovery manifests",
            ));
        }
        Ok(())
    }

    /// Called by the admission owner while it still holds the exact shard value it is about to
    /// publish. A clone is an Arc-owner capture, not a name-map reload, and it preserves every
    /// payload/sidecar allocation needed by Rebuild and the eventual reader. Recovery can publish
    /// a CREATE's empty bootstrap descriptor before the final all-row admission; retain the most
    /// recent direct owner so the post-replay capture is the final immutable image, never that
    /// bootstrap.
    pub(crate) fn capture_sealed_int4_recovery_shard(
        &self,
        table_oid: u32,
        shards: Vec<RelationalResidentShard>,
    ) -> Result<(), DataGenerationError> {
        let armed = self
            .read_state
            .sealed_int4_capture_table_oid
            .load(Ordering::Acquire);
        if armed == 0 || armed != table_oid || shards.is_empty() {
            return Ok(());
        }
        let mut capture = self
            .read_state
            .sealed_int4_recovery_capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        capture.replace(shards.into());
        Ok(())
    }

    pub(crate) fn take_sealed_int4_recovery_capture(
        &self,
    ) -> Result<Arc<[RelationalResidentShard]>, DataGenerationError> {
        self.read_state
            .sealed_int4_capture_table_oid
            .store(0, Ordering::Release);
        self.read_state
            .sealed_int4_recovery_capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .ok_or(DataGenerationError::Missing(
                "sealed nullable-int4 recovery shard capture",
            ))
    }

    pub(crate) fn take_staged_sealed_int4_recovery_manifest(
        &self,
    ) -> Option<Arc<SealedInt4RebuildManifestV1>> {
        self.read_state
            .sealed_int4_recovery_manifest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }
}
