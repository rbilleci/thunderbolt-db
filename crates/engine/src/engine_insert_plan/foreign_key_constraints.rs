//! Inert immediate foreign-key proof metadata.
//!
//! This production-compiled leaf binds exact catalog identities for the future GPU FK operator.
//! It owns neither a live admission route nor WAL, apply, allocator, or publication authority.

#![allow(dead_code)] // compiled proof metadata; the consuming inspection seam is test-only

use crate::relational_model::{RelationalColumn, RelationalForeignKey, RelationalTable};
use crate::typed_insert_batch::TypedInsertBatch;
use crate::{
    CatalogSnapshot, Engine, EngineError, ExecuteError, Index, RelationalResidentShard, SqlType,
};

use std::sync::{Arc, MutexGuard};

#[cfg(test)]
use std::cell::Cell;

use super::{pre_wal_constraints::ConstraintCandidate, resident_constraint_generation};
use gpu_db_execution::{
    insert_foreign_key_verdict_scratch_bytes, CudaAllocationScope, CudaInsertForeignKeyParentShard,
    CudaInsertForeignKeySelfProvider, CudaInsertResidentKeySidecar,
    INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES,
};

#[cfg(test)]
thread_local! {
    static CURRENT_GENERATION_VALIDATION_ENTRIES: Cell<usize> = const { Cell::new(0) };
}

/// Test-only proof that a rejection reached the GPU FK evaluator rather than short-circuiting in
/// the caller's semantic revalidation.  This remains scalar-only and has no production route.
#[cfg(test)]
pub(super) fn reset_current_generation_validation_entries() {
    CURRENT_GENERATION_VALIDATION_ENTRIES.with(|entries| entries.set(0));
}

#[cfg(test)]
pub(super) fn current_generation_validation_entries() -> usize {
    CURRENT_GENERATION_VALIDATION_ENTRIES.with(Cell::get)
}

#[derive(Clone)]
struct TableBinding {
    schema: String,
    name: String,
    oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
}

#[derive(Clone)]
struct ColumnBinding {
    id: u32,
    table_oid: u32,
    attnum: i16,
    name: String,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
}

#[derive(Clone)]
struct SupportingIndexBinding {
    raw_ordinal: usize,
    oid: u32,
    name: String,
    table: String,
    column: String,
    key_columns: Box<[String]>,
    unique: bool,
    primary_key: bool,
    unique_constraint: bool,
}

#[derive(Clone)]
pub(super) struct ForeignKeyBinding {
    raw_ordinal: usize,
    name: String,
    column: String,
    referenced_table: String,
    referenced_column: String,
    child_column: ColumnBinding,
    parent: TableBinding,
    parent_column: ColumnBinding,
    supporting_index: SupportingIndexBinding,
}

/// Move-only exact catalog witness for every child FK in raw catalog-vector order.
pub(super) struct ForeignKeyConstraintProof {
    child: TableBinding,
    prepared_catalog_seq: Index,
    original_read_snapshot: Index,
    autocommit_scope: bool,
    bindings: Box<[ForeignKeyBinding]>,
}

impl ForeignKeyConstraintProof {
    pub(super) fn compile(
        engine: &Engine,
        batch: &TypedInsertBatch,
        catalog: &CatalogSnapshot,
    ) -> Result<Self, EngineError> {
        let child = bound_child(batch, catalog)?;
        let child_binding = table_binding(child)?;
        let bindings = child
            .foreign_keys
            .iter()
            .enumerate()
            .map(|(raw_ordinal, foreign_key)| {
                bind_foreign_key(raw_ordinal, child, foreign_key, catalog)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            child: child_binding,
            prepared_catalog_seq: catalog.commit_seq,
            original_read_snapshot: engine.committed_seq(),
            autocommit_scope: engine.current_transaction_read_snapshot().is_none(),
            bindings: bindings.into(),
        })
    }

    pub(super) fn access_table_names(&self) -> impl Iterator<Item = String> + '_ {
        std::iter::once(self.child.name.clone()).chain(
            self.bindings
                .iter()
                .map(|binding| binding.parent.name.clone()),
        )
    }

    pub(super) fn original_read_snapshot(&self) -> Index {
        self.original_read_snapshot
    }

    pub(super) fn bindings(&self) -> &[ForeignKeyBinding] {
        &self.bindings
    }

    pub(super) fn validate_current_catalog(
        &self,
        catalog: &CatalogSnapshot,
        predecessor_boundary: Index,
    ) -> Result<(), EngineError> {
        if !self.autocommit_scope || predecessor_boundary != catalog.commit_seq {
            return Err(EngineError::ApplyFailed(
                "resident INSERT foreign-key proof requires one autocommit catalog boundary"
                    .to_string(),
            ));
        }
        if catalog.commit_seq < self.prepared_catalog_seq {
            return Err(EngineError::ApplyFailed(
                "resident INSERT foreign-key proof catalog generation moved backward".to_string(),
            ));
        }
        let child = table_at_binding(catalog, &self.child)?;
        if child.foreign_keys.len() != self.bindings.len() {
            return Err(EngineError::ApplyFailed(
                "resident INSERT foreign-key proof FK vector drifted".to_string(),
            ));
        }
        for binding in self.bindings.iter() {
            let live = child.foreign_keys.get(binding.raw_ordinal).ok_or_else(|| {
                EngineError::ApplyFailed(
                    "resident INSERT foreign-key proof FK ordinal disappeared".to_string(),
                )
            })?;
            if live.name != binding.name
                || live.column != binding.column
                || live.referenced_table != binding.referenced_table
                || live.referenced_column != binding.referenced_column
                || !column_matches(child, &binding.child_column)
            {
                return Err(EngineError::ApplyFailed(
                    "resident INSERT foreign-key proof child binding drifted".to_string(),
                ));
            }
            let parent = table_at_binding(catalog, &binding.parent)?;
            if !column_matches(parent, &binding.parent_column)
                || !index_matches(parent, &binding.supporting_index)
            {
                return Err(EngineError::ApplyFailed(
                    "resident INSERT foreign-key proof parent binding drifted".to_string(),
                ));
            }
        }
        Ok(())
    }
}

fn bound_child<'a>(
    batch: &TypedInsertBatch,
    catalog: &'a CatalogSnapshot,
) -> Result<&'a RelationalTable, EngineError> {
    let (oid, digest, sequence) = batch.row_local_constraint_target();
    let child = catalog
        .relational_catalog
        .values()
        .find(|table| table.oid == oid)
        .ok_or_else(|| {
            EngineError::ApplyFailed("device pre-WAL target relation is absent".to_string())
        })?;
    if catalog.commit_seq != sequence
        || crate::engine_transaction_reset::table_schema_digest(child).ok() != Some(digest)
    {
        return Err(EngineError::ApplyFailed(
            "device pre-WAL target binding drifted before off-lock preparation".to_string(),
        ));
    }
    Ok(child)
}

fn bind_foreign_key(
    raw_ordinal: usize,
    child: &RelationalTable,
    foreign_key: &RelationalForeignKey,
    catalog: &CatalogSnapshot,
) -> Result<ForeignKeyBinding, EngineError> {
    let child_column = bind_column(child, &foreign_key.column)?;
    let parent = catalog
        .relational_catalog
        .get(&foreign_key.referenced_table)
        .ok_or_else(|| {
            EngineError::ApplyFailed("resident INSERT foreign-key parent is absent".to_string())
        })?;
    let parent_column = bind_column(parent, &foreign_key.referenced_column)?;
    if child_column.ty != parent_column.ty {
        return Err(EngineError::ApplyFailed(
            "resident INSERT foreign-key columns have incompatible types".to_string(),
        ));
    }
    let (index_ordinal, index) = parent
        .indexes
        .iter()
        .enumerate()
        .find(|(_, index)| {
            index.table == parent.name
                && index.column == foreign_key.referenced_column
                && index.unique
                && (index.primary_key || index.unique_constraint)
                && index.key_columns.as_slice() == [foreign_key.referenced_column.as_str()]
        })
        .ok_or_else(|| {
            EngineError::ApplyFailed(
                "resident INSERT foreign-key parent has no single-column UNIQUE/PRIMARY index"
                    .to_string(),
            )
        })?;
    Ok(ForeignKeyBinding {
        raw_ordinal,
        name: foreign_key.name.clone(),
        column: foreign_key.column.clone(),
        referenced_table: foreign_key.referenced_table.clone(),
        referenced_column: foreign_key.referenced_column.clone(),
        child_column,
        parent: table_binding(parent)?,
        parent_column,
        supporting_index: SupportingIndexBinding {
            raw_ordinal: index_ordinal,
            oid: index.oid,
            name: index.name.clone(),
            table: index.table.clone(),
            column: index.column.clone(),
            key_columns: index.key_columns.clone().into(),
            unique: index.unique,
            primary_key: index.primary_key,
            unique_constraint: index.unique_constraint,
        },
    })
}

fn table_binding(table: &RelationalTable) -> Result<TableBinding, EngineError> {
    Ok(TableBinding {
        schema: table.schema.clone(),
        name: table.name.clone(),
        oid: table.oid,
        schema_digest: crate::engine_transaction_reset::table_schema_digest(table)
            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?,
    })
}

fn bind_column(table: &RelationalTable, name: &str) -> Result<ColumnBinding, EngineError> {
    let column = table
        .columns
        .iter()
        .find(|column| column.name == name)
        .ok_or_else(|| {
            EngineError::ApplyFailed(format!(
                "resident INSERT foreign-key column \"{name}\" is absent from relation \"{}\"",
                table.name
            ))
        })?;
    Ok(ColumnBinding {
        id: column.id,
        table_oid: column.table_oid,
        attnum: column.attnum,
        name: column.name.clone(),
        ty: column.ty,
        type_oid: column.type_oid,
        type_size: column.type_size,
    })
}

fn table_at_binding<'a>(
    catalog: &'a CatalogSnapshot,
    binding: &TableBinding,
) -> Result<&'a RelationalTable, EngineError> {
    let table = catalog
        .relational_catalog
        .values()
        .find(|table| table.oid == binding.oid)
        .ok_or_else(|| {
            EngineError::ApplyFailed("resident INSERT foreign-key table disappeared".to_string())
        })?;
    if table.schema != binding.schema
        || table.name != binding.name
        || crate::engine_transaction_reset::table_schema_digest(table).ok()
            != Some(binding.schema_digest)
    {
        return Err(EngineError::ApplyFailed(
            "resident INSERT foreign-key table binding drifted".to_string(),
        ));
    }
    Ok(table)
}

fn column_matches(table: &RelationalTable, binding: &ColumnBinding) -> bool {
    table.columns.iter().any(|column| {
        column.id == binding.id
            && column.table_oid == binding.table_oid
            && column.attnum == binding.attnum
            && column.name == binding.name
            && column.ty == binding.ty
            && column.type_oid == binding.type_oid
            && column.type_size == binding.type_size
    })
}

fn index_matches(table: &RelationalTable, binding: &SupportingIndexBinding) -> bool {
    table.indexes.get(binding.raw_ordinal).is_some_and(|index| {
        index.table == table.name
            && index.column == binding.column
            && index.unique
            && (index.primary_key || index.unique_constraint)
            && index.key_columns.as_slice() == [binding.column.as_str()]
            && index.oid == binding.oid
            && index.name == binding.name
            && index.table == binding.table
            && index.column == binding.column
            && index.key_columns.as_slice() == binding.key_columns.as_ref()
            && index.unique == binding.unique
            && index.primary_key == binding.primary_key
            && index.unique_constraint == binding.unique_constraint
    })
}

/// Move-only successful FK validation.  Keeping the access lease, exact map pin, and mutation
/// guard borrow together prevents a test inspection from outliving the authority that proved it.
pub(super) struct ForeignKeyValidationSeal<'guard, 'mutex> {
    _table_access: Arc<crate::TableAccessLease>,
    _pinned_generation:
        resident_constraint_generation::PinnedResidentConstraintGeneration<'guard, 'mutex>,
    _proof: ForeignKeyConstraintProof,
    _parents: Box<[ForeignKeyParentGenerationEvidence]>,
    child_table: String,
    predecessor_boundary: Index,
    original_read_snapshot: Index,
    gpu_id: u16,
    parent_shard_count: usize,
    allocation_peak_bytes: u64,
    expected_allocation_peak_bytes: u64,
}

/// Explicit pins/evidence for every parent generation used by an FK operator.  The enclosing map
/// pin is the authority; these clones make the seal's resource dependency auditable and prevent
/// a future refactor from retaining only scalar identity after successful GPU validation.
struct ForeignKeyParentGenerationEvidence {
    raw_fk_ordinal: usize,
    parent_oid: u32,
    generation: Arc<()>,
    shards: Box<[ForeignKeyParentShardResourceEvidence]>,
}

struct ForeignKeyParentShardResourceEvidence {
    shard_id: u32,
    row_count: usize,
    capacity: usize,
    history_floor_index: Index,
    payload: Arc<gpu_db_execution::CudaResidentDeviceMemory>,
    payload_ptr: u64,
    created_by: Option<Arc<gpu_db_execution::CudaResidentDeviceMemory>>,
    deleted_by: Option<Arc<gpu_db_execution::CudaResidentDeviceMemory>>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ForeignKeyProofReport {
    child_table: String,
    predecessor_boundary: Index,
    original_read_snapshot: Index,
    gpu_id: u16,
    parent_shard_count: usize,
    allocation_peak_bytes: u64,
    expected_allocation_peak_bytes: u64,
}

impl<'guard, 'mutex> ForeignKeyValidationSeal<'guard, 'mutex> {
    /// The report deliberately contains no row ids, mutation authority, WAL payload, or resident
    /// allocation handles.  Consuming the seal also releases the map/access/gate witnesses before
    /// arbitrary test inspection can run.
    #[cfg(test)]
    pub(super) fn into_report(self) -> ForeignKeyProofReport {
        ForeignKeyProofReport {
            child_table: self.child_table,
            predecessor_boundary: self.predecessor_boundary,
            original_read_snapshot: self.original_read_snapshot,
            gpu_id: self.gpu_id,
            parent_shard_count: self.parent_shard_count,
            allocation_peak_bytes: self.allocation_peak_bytes,
            expected_allocation_peak_bytes: self.expected_allocation_peak_bytes,
        }
    }
}

struct ResolvedForeignKey<'catalog, 'pin> {
    binding: &'catalog ForeignKeyBinding,
    parent: &'catalog RelationalTable,
    parent_generation: resident_constraint_generation::ValidatedResidentConstraintTable<'pin>,
    child_columns: Box<[resident_constraint_generation::ResidentConstraintColumnBinding]>,
    parent_columns: Box<[resident_constraint_generation::ResidentConstraintColumnBinding]>,
}

/// Evaluate every raw-ordinal FK using the one map pinned below the residency mutation gate.
/// It returns a move-only validation seal and cannot reach WAL, apply, allocator reservation, or
/// residency publication. The sole scalar inspection caller is test-only.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_current_generation<'guard, 'mutex>(
    engine: &Engine,
    batch: &TypedInsertBatch,
    proof: ForeignKeyConstraintProof,
    local_candidate: Option<ConstraintCandidate>,
    current_catalog: &CatalogSnapshot,
    predecessor_boundary: Index,
    held_mutation_gate: &'guard MutexGuard<'mutex, ()>,
    table_access: Arc<crate::TableAccessLease>,
) -> Result<ForeignKeyValidationSeal<'guard, 'mutex>, ExecuteError> {
    #[cfg(test)]
    CURRENT_GENERATION_VALIDATION_ENTRIES.with(|entries| {
        entries.set(entries.get().saturating_add(1));
    });
    if engine.current_transaction_read_snapshot().is_some() {
        return Err(decline(
            "resident INSERT foreign-key proof is restricted to an autocommit snapshot",
        ));
    }
    proof
        .validate_current_catalog(current_catalog, predecessor_boundary)
        .map_err(ExecuteError::Engine)?;
    if predecessor_boundary < proof.original_read_snapshot() {
        return Err(decline(
            "resident INSERT foreign-key proof predecessor predates its read snapshot",
        ));
    }
    let child = table_at_binding(current_catalog, &proof.child).map_err(ExecuteError::Engine)?;
    // Capture non-authoritative residency state once for the entire FK closure before pinning
    // the authoritative shard map.  Parent validation below must not reload either source.
    let chunk_authoritative = engine
        .read_state
        .residency
        .chunk_authoritative_tables
        .load();
    let cold_chunks = engine.read_streaming_cold_chunks();
    if chunk_authoritative.contains_key(&child.name)
        || cold_chunks.contains_key(&child.name)
        || engine.intent_lanes.is_some()
    {
        return Err(decline(
            "resident INSERT foreign-key proof requires one hot non-lane autocommit generation",
        ));
    }

    let expected_gpu = engine.planner.default_gpu_id();
    let pinned_generation = resident_constraint_generation::pin_hot_shard_generation(
        engine,
        child,
        proof.original_read_snapshot(),
        expected_gpu,
        held_mutation_gate,
    )?;
    let mut resolved = Vec::with_capacity(proof.bindings.len());
    for binding in proof.bindings() {
        let parent =
            table_at_binding(current_catalog, &binding.parent).map_err(ExecuteError::Engine)?;
        if chunk_authoritative.contains_key(&parent.name) || cold_chunks.contains_key(&parent.name)
        {
            return Err(decline(
                "resident INSERT foreign-key proof requires hot parent generations",
            ));
        }
        let parent_generation = pinned_generation.validate_table(
            engine,
            parent,
            proof.original_read_snapshot(),
            expected_gpu,
        )?;
        let child_column = column_at(child, &binding.child_column).map_err(ExecuteError::Engine)?;
        let parent_column =
            column_at(parent, &binding.parent_column).map_err(ExecuteError::Engine)?;
        resolved.push(ResolvedForeignKey {
            binding,
            parent,
            parent_generation,
            child_columns: resident_constraint_generation::bind_catalog_columns(
                child,
                [child_column],
            )?,
            parent_columns: resident_constraint_generation::bind_catalog_columns(
                parent,
                [parent_column],
            )?,
        });
    }

    let source_bytes = batch
        .row_local_constraint_device_payload_bytes()
        .map_err(ExecuteError::Engine)?;
    let rows = usize::try_from(batch.binary_insert_template_row_count())
        .expect("u32 row count fits usize on supported hosts");
    let mut max_scratch = 0_u64;
    for fk in &resolved {
        let child_descriptor_count =
            resident_constraint_generation::descriptor_count_for_batch(batch, &fk.child_columns)
                .map_err(ExecuteError::Engine)?;
        let mut max_parent_descriptor_count = 0;
        for shard in fk.parent_generation.shards() {
            let snapshot = engine.resident_snapshot_for_shard(shard, fk.parent);
            max_parent_descriptor_count = max_parent_descriptor_count.max(
                resident_constraint_generation::descriptor_count_for_resident(
                    fk.parent,
                    &snapshot,
                    &fk.parent_columns,
                )?,
            );
        }
        let self_descriptor_count = (fk.parent.oid == child.oid)
            .then(|| {
                resident_constraint_generation::descriptor_count_for_batch(
                    batch,
                    &fk.parent_columns,
                )
            })
            .transpose()
            .map_err(ExecuteError::Engine)?;
        let scratch = insert_foreign_key_verdict_scratch_bytes(
            rows,
            child_descriptor_count,
            max_parent_descriptor_count,
            self_descriptor_count,
        )
        .ok_or_else(|| decline("resident INSERT foreign-key scratch extent overflows"))?;
        max_scratch = max_scratch.max(scratch);
    }
    let peak = source_bytes
        .checked_add(max_scratch)
        .ok_or_else(|| decline("resident INSERT foreign-key allocation peak overflows"))?;
    let budget = match engine.relational_residency_budget_bytes(expected_gpu) {
        Some(limit) => limit
            .checked_sub(engine.relational_resident_bytes_for_gpu(expected_gpu))
            .ok_or_else(|| {
                decline("resident INSERT foreign-key proof is refused under device pressure")
            })?,
        None => peak,
    };
    let allocation_scope = CudaAllocationScope::with_budget(budget);
    CudaAllocationScope::ensure_available(peak).map_err(|error| {
        decline(format!(
            "resident INSERT foreign-key allocation is unavailable: {error}"
        ))
    })?;
    // Exactly one typed child upload is shared by every raw FK operator invocation below.
    let source = batch
        .row_local_constraint_device_source(engine, child)
        .map_err(|error| {
            decline(format!(
                "resident INSERT foreign-key source is unavailable: {error}"
            ))
        })?;

    let mut foreign_candidate = None;
    let mut first_history = None;
    // A child floor pertains to duplicate/key-history proof, not a parent FK absence. Only a
    // parent floor paired with that FK's unresolved absence may require retry below.
    let mut history_floor_requires_retry = false;
    let mut parent_shard_count = 0_usize;
    for fk in &resolved {
        let child_columns =
            resident_constraint_generation::incoming_columns(&source, &fk.child_columns)
                .map_err(ExecuteError::Engine)?;
        let self_columns = (fk.parent.oid == child.oid)
            .then(|| resident_constraint_generation::incoming_columns(&source, &fk.parent_columns))
            .transpose()
            .map_err(ExecuteError::Engine)?;
        let parent_shards = fk.parent_generation.shards();
        let parent_columns = parent_shards
            .iter()
            .map(|shard| {
                resident_constraint_generation::resident_columns(
                    engine,
                    fk.parent,
                    shard,
                    &fk.parent_columns,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let parent_inputs = parent_shards
            .iter()
            .zip(parent_columns.iter())
            .map(|(shard, columns)| foreign_key_parent_shard(shard, columns))
            .collect::<Result<Vec<_>, _>>()?;
        parent_shard_count = parent_shard_count
            .checked_add(parent_inputs.len())
            .ok_or_else(|| decline("resident INSERT foreign-key parent shard count overflows"))?;
        let verdict = source
            .memory()
            .insert_foreign_key_verdict_against_shards(
                &child_columns,
                batch.binary_insert_template_row_count(),
                &parent_inputs,
                self_columns
                    .as_deref()
                    .map(|columns| CudaInsertForeignKeySelfProvider { columns }),
                predecessor_boundary,
                proof.original_read_snapshot(),
            )
            .map_err(|error| {
                decline(format!(
                    "resident INSERT foreign-key verdict declined for constraint \"{}\": {error}",
                    fk.binding.name
                ))
            })?;
        if verdict.readback_bytes != INSERT_FOREIGN_KEY_VERDICT_READBACK_BYTES
            && batch.binary_insert_template_row_count() != 0
        {
            return Err(decline(
                "resident INSERT foreign-key verdict lost its bounded terminal",
            ));
        }
        if let Some(row) = verdict.first_missing_row {
            if fk.parent_generation.history_floor_requires_retry() {
                // A dense parent image cannot make an absence authoritative below its retained
                // floor. Positive both-world matches remain valid despite this floor.
                history_floor_requires_retry = true;
            } else {
                foreign_candidate = ConstraintCandidate::choose(
                    foreign_candidate,
                    Some(ConstraintCandidate::foreign_key(
                        row,
                        fk.binding.raw_ordinal,
                        child.name.clone(),
                        fk.binding.name.clone(),
                    )),
                );
            }
        }
        if let Some(row) = verdict.first_history_row {
            first_history = Some(first_history.map_or(row, |current: u32| current.min(row)));
        }
    }
    let allocation_peak_bytes = allocation_scope.peak_bytes();
    if allocation_peak_bytes != peak {
        return Err(decline(
            "resident INSERT foreign-key allocation scope lost its exact peak",
        ));
    }
    drop(source);
    drop(allocation_scope);

    if let Some(candidate) = ConstraintCandidate::choose(local_candidate, foreign_candidate) {
        return Err(ExecuteError::Engine(candidate.into_error()));
    }
    if let Some(row) = first_history {
        return Err(ExecuteError::Serialization(format!(
            "resident foreign-key history changed after read snapshot {} at incoming row {row}",
            proof.original_read_snapshot()
        )));
    }
    if history_floor_requires_retry {
        return Err(ExecuteError::Serialization(format!(
            "resident foreign-key history floor is newer than read snapshot {}",
            proof.original_read_snapshot()
        )));
    }
    let parent_evidence = resolved
        .iter()
        .map(foreign_key_parent_generation_evidence)
        .collect::<Result<Vec<_>, _>>()?;
    let child_table = child.name.clone();
    let original_read_snapshot = proof.original_read_snapshot();
    // The seal owns the proof, not a borrow of it.  Release all raw binding/table views first;
    // only the opaque map pin and explicit resource evidence survive this boundary.
    drop(resolved);
    Ok(ForeignKeyValidationSeal {
        _table_access: table_access,
        _pinned_generation: pinned_generation,
        _proof: proof,
        _parents: parent_evidence.into(),
        child_table,
        predecessor_boundary,
        original_read_snapshot,
        gpu_id: expected_gpu,
        parent_shard_count,
        allocation_peak_bytes,
        expected_allocation_peak_bytes: peak,
    })
}

fn foreign_key_parent_generation_evidence(
    foreign_key: &ResolvedForeignKey<'_, '_>,
) -> Result<ForeignKeyParentGenerationEvidence, ExecuteError> {
    let shards = foreign_key
        .parent_generation
        .shards()
        .iter()
        .map(|shard| {
            let payload = shard.device_memory.as_ref().ok_or_else(|| {
                decline("resident INSERT foreign-key seal lost a validated parent payload")
            })?;
            Ok(ForeignKeyParentShardResourceEvidence {
                shard_id: shard.shard_id,
                row_count: shard.row_count,
                capacity: shard.capacity,
                history_floor_index: shard.history_floor_index,
                payload: Arc::clone(payload),
                payload_ptr: payload.device_ptr(),
                created_by: shard.created_by_region.as_ref().map(Arc::clone),
                deleted_by: shard.deleted_by_region.as_ref().map(Arc::clone),
            })
        })
        .collect::<Result<Vec<_>, ExecuteError>>()?;
    Ok(ForeignKeyParentGenerationEvidence {
        raw_fk_ordinal: foreign_key.binding.raw_ordinal,
        parent_oid: foreign_key.parent.oid,
        generation: Arc::clone(foreign_key.parent_generation.generation()),
        shards: shards.into(),
    })
}

fn column_at<'a>(
    table: &'a RelationalTable,
    binding: &ColumnBinding,
) -> Result<&'a RelationalColumn, EngineError> {
    table
        .columns
        .iter()
        .find(|column| {
            column.id == binding.id
                && column.table_oid == binding.table_oid
                && column.attnum == binding.attnum
                && column.name == binding.name
                && column.ty == binding.ty
                && column.type_oid == binding.type_oid
                && column.type_size == binding.type_size
        })
        .ok_or_else(|| {
            EngineError::ApplyFailed(
                "resident INSERT foreign-key column binding drifted".to_string(),
            )
        })
}

fn foreign_key_parent_shard<'a>(
    shard: &'a RelationalResidentShard,
    columns: &'a [gpu_db_execution::CudaCompoundFoldColumn],
) -> Result<CudaInsertForeignKeyParentShard<'a>, ExecuteError> {
    let payload = shard
        .device_memory
        .as_deref()
        .ok_or_else(|| decline("resident INSERT foreign-key proof lost a pinned parent payload"))?;
    let deleted_live = u64::from_le_bytes(
        [crate::engine_residency::DELETED_BY_LIVE_FILL_BYTE; std::mem::size_of::<u64>()],
    );
    Ok(CudaInsertForeignKeyParentShard {
        payload,
        columns,
        row_count: u32::try_from(shard.row_count)
            .map_err(|_| decline("resident parent shard row count exceeds CUDA FK domain"))?,
        created_by: shard
            .created_by_region
            .as_deref()
            .map(|memory| CudaInsertResidentKeySidecar {
                memory,
                byte_offset: 0,
            }),
        created_default: u64::from_le_bytes(
            [crate::engine_residency::CREATED_BY_VISIBLE_FILL_BYTE; std::mem::size_of::<u64>()],
        ),
        deleted_by: shard
            .deleted_by_region
            .as_deref()
            .map(|memory| CudaInsertResidentKeySidecar {
                memory,
                byte_offset: 0,
            }),
        deleted_default: deleted_live,
        deleted_live,
    })
}

fn decline(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Serialization(message.into())
}

#[cfg(test)]
#[path = "foreign_key_constraints_tests.rs"]
mod tests;
