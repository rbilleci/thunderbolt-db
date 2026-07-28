//! Test-only current-generation UNIQUE/PRIMARY KEY proof for typed INSERT.
//!
//! This module owns no WAL, row identities, apply, or publication.  It proves one deliberately
//! narrow autocommit shape under the residency mutation gate so the future live handoff has an
//! exact GPU-resident semantic reference rather than a host duplicate check.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::MutexGuard;

use super::batch_key_constraints::{BatchKeyConstraintProof, IndexBinding, KeyColumnBinding};
use super::pre_wal_constraints::{self, ConstraintCandidate};
use crate::relational_model::RelationalTable;
use crate::typed_insert_batch::{TypedInsertBatch, TypedInsertConstraintDeviceSource};
use crate::{Engine, EngineError, ExecuteError, Index, RelationalResidentShard, SqlType};
use gpu_db_execution::{
    insert_resident_key_verdict_scratch_bytes, CudaAllocationScope, CudaCompoundFoldColumn,
    CudaInsertResidentKeyShard, CudaInsertResidentKeySidecar,
    INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES,
};

/// Move-only context for the proof-only current-generation pass.  The sole sealed raw-index
/// proof stays inside the pre-WAL carrier until that carrier is consumed beneath the canonical
/// commit gate; this context contains only snapshot scope, never a cloned catalog authority.
pub(super) struct ResidentKeyConstraintProof {
    original_read_snapshot: Index,
    autocommit_scope: bool,
}

pub(crate) struct ResidentKeyValidationSeal {
    table_name: String,
    catalog_seq: Index,
    original_read_snapshot: Index,
    predecessor_boundary: Index,
    gpu_id: u16,
    generation: Arc<()>,
    shards: Box<[ResidentKeyShardResourceEvidence]>,
}

struct ResidentKeyShardResourceEvidence {
    shard_id: u32,
    payload_ptr: u64,
    row_count: usize,
    capacity: usize,
}

impl ResidentKeyValidationSeal {
    pub(crate) fn matches_in_place_append(
        &self,
        table: &RelationalTable,
        catalog_seq: Index,
        shard: &RelationalResidentShard,
        predecessor_boundary: Index,
    ) -> bool {
        self.table_name == table.name
            && self.catalog_seq == catalog_seq
            && self.predecessor_boundary == predecessor_boundary
            && shard.gpu_id == self.gpu_id
            && Arc::ptr_eq(&self.generation, &shard.point_route_generation)
            && self.shards.iter().any(|evidence| {
                evidence.shard_id == shard.shard_id
                    && evidence.row_count == shard.row_count
                    && evidence.capacity == shard.capacity
                    && shard
                        .device_memory
                        .as_ref()
                        .is_some_and(|memory| memory.device_ptr() == evidence.payload_ptr)
            })
    }

    pub(crate) fn original_read_snapshot(&self) -> Index {
        self.original_read_snapshot
    }

    pub(crate) fn predecessor_boundary(&self) -> Index {
        self.predecessor_boundary
    }
}

pub(super) fn compile(engine: &Engine) -> ResidentKeyConstraintProof {
    ResidentKeyConstraintProof {
        original_read_snapshot: engine.committed_seq(),
        autocommit_scope: engine.current_transaction_read_snapshot().is_none(),
    }
}

/// Evaluate every UNIQUE/PK binding against every shard in the one generation captured beneath
/// the mutation gate.  A SQL candidate always wins over a history retry, regardless of incoming
/// row order; the returned error is therefore the final product-order terminal for this proof
/// seam.
#[allow(clippy::too_many_arguments)] // each authority stays explicit at the commit/mutation gate
pub(super) fn validate_current_generation(
    engine: &Engine,
    batch: &TypedInsertBatch,
    proof: &ResidentKeyConstraintProof,
    keys: &BatchKeyConstraintProof,
    local_candidate: Option<ConstraintCandidate>,
    current_catalog: &crate::CatalogSnapshot,
    predecessor_boundary: Index,
    _held_mutation_gate: &MutexGuard<'_, ()>,
) -> Result<ResidentKeyValidationSeal, ExecuteError> {
    // The caller holds commit -> residency mutation in the established writer order.  This inner
    // validator deliberately takes neither lock and never reloads `committed_seq`: its explicit
    // predecessor boundary is the only publication decision it is allowed to make.
    if !proof.autocommit_scope || engine.current_transaction_read_snapshot().is_some() {
        return Err(decline(
            "resident INSERT key proof is restricted to an autocommit snapshot",
        ));
    }
    let original_read_snapshot = proof.original_read_snapshot;
    if predecessor_boundary < original_read_snapshot
        || !keys.matches_current_target_binding(current_catalog)
    {
        return Err(decline(
            "resident INSERT key proof target/catalog generation changed before the current gate",
        ));
    }
    if keys
        .indexes()
        .iter()
        .any(|index| index.is_unique_or_primary() && index.key_columns().is_empty())
    {
        return Err(decline(
            "resident key proof found an empty UNIQUE/PRIMARY key",
        ));
    }
    let table = pre_wal_constraints::bound_table_current_generation(batch, current_catalog)
        .map_err(ExecuteError::Engine)?;
    if !table.foreign_keys.is_empty() {
        return Err(decline(
            "resident INSERT key proof does not serve foreign-key tables",
        ));
    }
    if engine
        .read_state
        .residency
        .chunk_authoritative_tables
        .load()
        .contains_key(&table.name)
        || engine
            .read_streaming_cold_chunks()
            .contains_key(&table.name)
        || engine.intent_lanes.is_some()
    {
        return Err(decline(
            "resident INSERT key proof requires one hot non-lane autocommit generation",
        ));
    }

    // Pin the exact immutable map while the mutation gate excludes every descriptor/sidecar
    // publisher.  Never reload this map inside the shard/index loops.
    let shard_map = engine.read_state.residency.shards.load_full();
    let shards = shard_map.get(&table.name).ok_or_else(|| {
        decline("resident INSERT key proof found no hot shard generation for its table")
    })?;
    let runtime = engine.router.runtime().snapshot();
    let expected_gpu = engine.planner.default_gpu_id();
    let history_floor_requires_retry = validate_shard_generation(
        engine,
        table,
        shards,
        original_read_snapshot,
        expected_gpu,
        &runtime,
    )?;

    let scratch = max_scratch_bytes(engine, batch, table, shards, keys)?;
    let source_bytes = batch
        .row_local_constraint_device_payload_bytes()
        .map_err(ExecuteError::Engine)?;
    let peak = source_bytes
        .checked_add(scratch)
        .ok_or_else(|| decline("resident INSERT key proof device allocation peak overflows"))?;
    let budget = match engine.relational_residency_budget_bytes(expected_gpu) {
        Some(limit) => limit
            .checked_sub(engine.relational_resident_bytes_for_gpu(expected_gpu))
            .ok_or_else(|| decline("resident INSERT key proof is refused under device pressure"))?,
        None => peak,
    };
    let allocation_scope = CudaAllocationScope::with_budget(budget);
    CudaAllocationScope::ensure_available(peak).map_err(|error| {
        decline(format!(
            "resident INSERT key proof allocation is unavailable: {error}"
        ))
    })?;
    let source = batch
        .row_local_constraint_device_source(engine, table)
        .map_err(|error| {
            decline(format!(
                "resident INSERT key source is unavailable: {error}"
            ))
        })?;

    let mut resident_candidate = None;
    let mut first_history = None;
    for index in keys
        .indexes()
        .iter()
        .filter(|index| index.is_unique_or_primary())
    {
        let incoming_columns = incoming_columns(&source, index).map_err(ExecuteError::Engine)?;
        for shard in shards {
            if shard.row_count == 0 {
                // The generation validator above makes this a history-floor decision; the CUDA
                // primitive's zero-readback fast return is never itself used as history evidence.
                continue;
            }
            let resident_columns = resident_columns(engine, table, shard, index)?;
            let payload = shard
                .device_memory
                .as_deref()
                .ok_or_else(|| decline("resident INSERT key proof lost a pinned shard payload"))?;
            let deleted_live = u64::from_le_bytes(
                [crate::engine_residency::DELETED_BY_LIVE_FILL_BYTE; std::mem::size_of::<u64>()],
            );
            let shard_source = CudaInsertResidentKeyShard {
                payload,
                columns: &resident_columns,
                row_count: u32::try_from(shard.row_count)
                    .map_err(|_| decline("resident shard row count exceeds CUDA key domain"))?,
                created_by: shard.created_by_region.as_deref().map(|memory| {
                    CudaInsertResidentKeySidecar {
                        memory,
                        byte_offset: 0,
                    }
                }),
                created_default: u64::from_le_bytes(
                    [crate::engine_residency::CREATED_BY_VISIBLE_FILL_BYTE;
                        std::mem::size_of::<u64>()],
                ),
                deleted_by: shard.deleted_by_region.as_deref().map(|memory| {
                    CudaInsertResidentKeySidecar {
                        memory,
                        byte_offset: 0,
                    }
                }),
                deleted_default: deleted_live,
                deleted_live,
            };
            let verdict = source
                .memory()
                .insert_resident_key_verdict_against_shard(
                    &incoming_columns,
                    batch.binary_insert_template_row_count(),
                    &shard_source,
                    predecessor_boundary,
                    original_read_snapshot,
                )
                .map_err(|error| {
                    decline(format!(
                        "resident INSERT key verdict declined for index \"{}\": {error}",
                        index.name
                    ))
                })?;
            if verdict.readback_bytes != INSERT_RESIDENT_KEY_VERDICT_READBACK_BYTES {
                return Err(decline(
                    "resident INSERT key verdict lost its bounded terminal",
                ));
            }
            if let Some(row) = verdict.first_visible_conflict_row {
                resident_candidate = ConstraintCandidate::choose(
                    resident_candidate,
                    Some(ConstraintCandidate::unique(
                        row,
                        index.raw_ordinal(),
                        index.name.clone(),
                    )),
                );
            }
            if let Some(row) = verdict.first_history_conflict_row {
                first_history = Some(first_history.map_or(row, |current: u32| current.min(row)));
            }
        }
    }
    drop(source);
    drop(allocation_scope);

    if let Some(candidate) = ConstraintCandidate::choose(local_candidate, resident_candidate) {
        return Err(ExecuteError::Engine(candidate.into_error()));
    }
    if let Some(row) = first_history {
        return Err(ExecuteError::Serialization(format!(
            "resident UNIQUE/PRIMARY key history changed after read snapshot {original_read_snapshot} at incoming row {row}"
        )));
    }
    if history_floor_requires_retry {
        return Err(ExecuteError::Serialization(format!(
            "resident UNIQUE/PRIMARY key history floor is newer than read snapshot {original_read_snapshot}"
        )));
    }
    let generation = shards
        .first()
        .map(|shard| Arc::clone(&shard.point_route_generation))
        .ok_or_else(|| decline("resident INSERT key proof lost its shard generation"))?;
    let shard_evidence = shards
        .iter()
        .map(|shard| {
            let payload = shard
                .device_memory
                .as_ref()
                .expect("generation validation proved every shard payload");
            ResidentKeyShardResourceEvidence {
                shard_id: shard.shard_id,
                payload_ptr: payload.device_ptr(),
                row_count: shard.row_count,
                capacity: shard.capacity,
            }
        })
        .collect();
    Ok(ResidentKeyValidationSeal {
        table_name: table.name.clone(),
        catalog_seq: current_catalog.commit_seq,
        original_read_snapshot,
        predecessor_boundary,
        gpu_id: expected_gpu,
        generation,
        shards: shard_evidence,
    })
}

fn validate_shard_generation(
    engine: &Engine,
    table: &RelationalTable,
    shards: &[RelationalResidentShard],
    original_read_snapshot: Index,
    expected_gpu: u16,
    runtime: &gpu_db_execution::GpuRuntimeSnapshot,
) -> Result<bool, ExecuteError> {
    if shards.is_empty() {
        return Err(decline(
            "resident INSERT key proof has an empty shard generation",
        ));
    }
    let mut shard_ids = BTreeSet::new();
    let mut generation = None::<Arc<()>>;
    let mut history_floor_requires_retry = false;
    for shard in shards {
        let pressured = runtime.memory_pressured_gpu_ids.contains(&shard.gpu_id);
        if shard.schema != table.schema
            || shard.table != table.name
            || shard.gpu_id != expected_gpu
            || !shard.is_valid(pressured)
            || shard.capacity < shard.row_count
            || !shard_ids.insert(shard.shard_id)
        {
            return Err(decline(
                "resident INSERT key proof found a pressured, stale, or incomplete shard generation",
            ));
        }
        history_floor_requires_retry |= shard.history_floor_index > original_read_snapshot;
        let payload = shard.device_memory.as_ref().ok_or_else(|| {
            decline("resident INSERT key proof found a shard without a device payload")
        })?;
        let proof = payload.metadata();
        if proof.gpu_id != expected_gpu
            || !proof.retained
            || proof.copied_bytes > proof.allocated_bytes
            || shard.device_memory_proof.as_ref() != Some(proof)
            || !engine.shard_write_locate_cell_live(&table.name, shard.shard_id, payload)
        {
            return Err(decline(
                "resident INSERT key proof found an invalid payload ownership witness",
            ));
        }
        let sidecar_bytes = u64::try_from(shard.capacity)
            .ok()
            .and_then(|capacity| capacity.checked_mul(std::mem::size_of::<u64>() as u64))
            .ok_or_else(|| decline("resident INSERT key sidecar extent overflows"))?;
        let valid_sidecar = |memory: &Arc<gpu_db_execution::CudaResidentDeviceMemory>| {
            let proof = memory.metadata();
            proof.gpu_id == expected_gpu
                && proof.retained
                && proof.copied_bytes >= sidecar_bytes
                && proof.copied_bytes <= proof.allocated_bytes
        };
        if shard
            .deleted_by_region
            .as_ref()
            .is_some_and(|memory| !valid_sidecar(memory))
            || shard
                .created_by_region
                .as_ref()
                .is_some_and(|memory| !valid_sidecar(memory))
            || (shard.max_created_by > 0 && shard.created_by_region.is_none())
        {
            return Err(decline(
                "resident INSERT key proof found incomplete version sidecars",
            ));
        }
        if let Some(current) = generation.as_ref() {
            if !Arc::ptr_eq(current, &shard.point_route_generation) {
                return Err(decline(
                    "resident INSERT key proof found a torn table generation",
                ));
            }
        } else {
            generation = Some(Arc::clone(&shard.point_route_generation));
        }
    }
    Ok(history_floor_requires_retry)
}

fn max_scratch_bytes(
    engine: &Engine,
    batch: &TypedInsertBatch,
    table: &RelationalTable,
    shards: &[RelationalResidentShard],
    keys: &BatchKeyConstraintProof,
) -> Result<u64, ExecuteError> {
    let rows = usize::try_from(batch.binary_insert_template_row_count())
        .expect("u32 row count fits usize on supported hosts");
    let mut maximum = 0_u64;
    for index in keys
        .indexes()
        .iter()
        .filter(|index| index.is_unique_or_primary())
    {
        let incoming = descriptor_count_for_batch(batch, index).map_err(ExecuteError::Engine)?;
        for shard in shards.iter().filter(|shard| shard.row_count != 0) {
            let snapshot = engine.resident_snapshot_for_shard(shard, table);
            let resident = descriptor_count_for_resident(table, &snapshot, index)?;
            let bytes = insert_resident_key_verdict_scratch_bytes(rows, incoming, resident)
                .ok_or_else(|| decline("resident INSERT key scratch extent overflows"))?;
            maximum = maximum.max(bytes);
        }
    }
    Ok(maximum)
}

pub(crate) fn descriptor_count_for_batch(
    batch: &TypedInsertBatch,
    index: &IndexBinding,
) -> Result<usize, EngineError> {
    let mut validity = BTreeSet::new();
    for column in index.key_columns() {
        if batch.row_local_constraint_column_has_validity_bitmap(column.id)? {
            validity.insert((column.attnum, column.id));
        }
    }
    index
        .key_columns()
        .len()
        .checked_add(validity.len())
        .ok_or_else(|| {
            EngineError::ApplyFailed("resident key descriptor count overflows".to_string())
        })
}

fn descriptor_count_for_resident(
    table: &RelationalTable,
    snapshot: &crate::RelationalResidencySnapshot,
    index: &IndexBinding,
) -> Result<usize, ExecuteError> {
    let mut validity = BTreeSet::new();
    for column in index.key_columns() {
        let column_idx = table_column_index(table, column)?;
        if crate::resident_device_null_column_offset(snapshot, table, column_idx)
            .map_err(|error| decline(format!("resident key layout declined: {error}")))?
            .is_some()
        {
            validity.insert((column.attnum, column.id));
        }
    }
    index
        .key_columns()
        .len()
        .checked_add(validity.len())
        .ok_or_else(|| decline("resident key descriptor count overflows"))
}

pub(crate) fn incoming_columns(
    source: &TypedInsertConstraintDeviceSource,
    index: &IndexBinding,
) -> Result<Vec<CudaCompoundFoldColumn>, EngineError> {
    let mut descriptors = Vec::with_capacity(index.key_columns().len() * 2);
    let mut validity = BTreeSet::new();
    for column in index.key_columns() {
        let (data, valid) = source.key_column_layout(column.id, &column.name)?;
        descriptors.push(data);
        if let Some(offset) = valid {
            validity.insert((column.attnum, column.id, offset));
        }
    }
    descriptors.extend(
        validity.into_iter().map(
            |(_, _, bitmap_byte_offset)| CudaCompoundFoldColumn::Validity { bitmap_byte_offset },
        ),
    );
    Ok(descriptors)
}

pub(crate) fn resident_columns(
    engine: &Engine,
    table: &RelationalTable,
    shard: &RelationalResidentShard,
    index: &IndexBinding,
) -> Result<Vec<CudaCompoundFoldColumn>, ExecuteError> {
    let snapshot = engine.resident_snapshot_for_shard(shard, table);
    let mut descriptors = Vec::with_capacity(index.key_columns().len() * 2);
    let mut validity = BTreeSet::new();
    for column in index.key_columns() {
        let column_idx = table_column_index(table, column)?;
        let data = match column.ty() {
            SqlType::Int2 | SqlType::Int4 | SqlType::Date => CudaCompoundFoldColumn::Fixed {
                byte_offset: crate::resident_device_int4_column_offset(
                    &snapshot, table, column_idx,
                )
                .map_err(|error| decline(format!("resident int4 key layout declined: {error}")))?,
                width_words: 1,
            },
            SqlType::Int8 | SqlType::Timestamp => CudaCompoundFoldColumn::Fixed {
                byte_offset: crate::resident_device_int8_column_offset(
                    &snapshot, table, column_idx,
                )
                .map_err(|error| decline(format!("resident int8 key layout declined: {error}")))?,
                width_words: 2,
            },
            SqlType::Numeric { .. } | SqlType::Uuid => CudaCompoundFoldColumn::Fixed {
                byte_offset: crate::resident_device_numeric_column_offset(
                    &snapshot, table, column_idx,
                )
                .map_err(|error| decline(format!("resident wide key layout declined: {error}")))?,
                width_words: 4,
            },
            SqlType::Bool => CudaCompoundFoldColumn::Bool {
                bitmap_byte_offset: crate::resident_device_bool_column_offset(
                    &snapshot, table, column_idx,
                )
                .map_err(|error| decline(format!("resident bool key layout declined: {error}")))?,
            },
            SqlType::Text => {
                let layout =
                    crate::resident_device_text_column_layout(&snapshot, table, column_idx)
                        .map_err(|error| {
                            decline(format!("resident text key layout declined: {error}"))
                        })?;
                CudaCompoundFoldColumn::Text {
                    offsets_byte_offset: layout.offsets_byte_offset,
                    bytes_byte_offset: layout.bytes_byte_offset,
                    bytes_len: layout.bytes_len,
                }
            }
        };
        descriptors.push(data);
        if let Some(offset) = crate::resident_device_null_column_offset(
            &snapshot, table, column_idx,
        )
        .map_err(|error| decline(format!("resident key validity layout declined: {error}")))?
        {
            validity.insert((column.attnum, column.id, offset));
        }
    }
    descriptors.extend(
        validity.into_iter().map(
            |(_, _, bitmap_byte_offset)| CudaCompoundFoldColumn::Validity { bitmap_byte_offset },
        ),
    );
    Ok(descriptors)
}

fn table_column_index(
    table: &RelationalTable,
    binding: &KeyColumnBinding,
) -> Result<usize, ExecuteError> {
    table
        .columns
        .iter()
        .position(|column| {
            column.id == binding.id
                && column.attnum == binding.attnum
                && column.name == binding.name
                && column.ty == binding.ty()
        })
        .ok_or_else(|| decline("resident key binding lost its catalog column"))
}

fn decline(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Serialization(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_validation_uses_the_callers_boundary_and_mutation_context() {
        let source = include_str!("resident_key_constraints.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("implementation precedes tests");
        let inner = source
            .split("pub(super) fn validate_current_generation")
            .nth(1)
            .and_then(|section| section.split("\nfn validate_shard_generation").next())
            .expect("current-generation validator");
        assert!(inner.contains("predecessor_boundary: Index"));
        assert!(inner.contains("_held_mutation_gate: &MutexGuard"));
        assert!(!inner.contains("engine.commit_state()"));
        assert!(!inner.contains("engine.committed_seq()"));
        assert!(!inner.contains("mutation_gate\n        .lock"));
        assert!(inner.contains("ResidentKeyValidationSeal"));
    }

    fn proof_only_plan(engine: &Engine, sql: &str) -> super::super::PreparedDeviceInsertPlan {
        let catalog = engine.catalog_snapshot();
        let command = gpu_db_sql::parse_command(sql).expect("test INSERT parses");
        let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
            &command,
            &catalog,
            catalog.commit_seq,
        )
        .expect("proof-only typed builder succeeds")
        .expect("indexed proof-only shape remains eligible");
        super::super::PreparedDeviceInsertPlan::from_typed_batch(batch, engine, &catalog)
            .expect("proof-only pre-WAL preparation succeeds")
    }

    fn gpu_resident_unique_engine(table: &str) -> Option<Engine> {
        let mut engine = Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return None;
        }
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);
        engine
            .execute_text(1, &format!("CREATE TABLE {table} (id int4 UNIQUE)"))
            .unwrap();
        engine
            .execute_text(2, &format!("INSERT INTO {table} VALUES (7)"))
            .unwrap();
        engine
            .populate_relational_residency_snapshot(table)
            .unwrap();
        Some(engine)
    }

    fn assert_serialization_without_side_effects(
        engine: &Engine,
        plan: &super::super::PreparedDeviceInsertPlan,
    ) {
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let boundary_before = engine.committed_seq();
        assert!(matches!(
            plan.validate_current_resident_key_constraints(engine),
            Err(ExecuteError::Serialization(_))
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.committed_seq(), boundary_before);
    }

    #[test]
    fn resident_key_seam_is_unavailable_to_an_ordinary_live_batch() {
        let engine = Engine::new_local_test_engine();
        engine
            .execute_text(1, "CREATE TABLE resident_key_live_gate (id int4)")
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let command =
            gpu_db_sql::parse_command("INSERT INTO resident_key_live_gate VALUES (1)").unwrap();
        let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch(
            &command,
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .expect("ordinary unindexed batch remains on the live typed route");
        let plan =
            super::super::PreparedDeviceInsertPlan::from_typed_batch(batch, &engine, &catalog)
                .unwrap();
        assert!(matches!(
            plan.validate_current_resident_key_constraints(&engine),
            Err(ExecuteError::Unsupported(message))
                if message == "resident INSERT key proof is not enabled for this batch"
        ));
        let proof_only_batch =
            crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
                &command,
                &catalog,
                catalog.commit_seq,
            )
            .unwrap()
            .expect("unindexed ProofOnly batch still has ordinary typed semantics");
        let proof_only_plan = super::super::PreparedDeviceInsertPlan::from_typed_batch(
            proof_only_batch,
            &engine,
            &catalog,
        )
        .unwrap();
        assert!(matches!(
            proof_only_plan.validate_current_resident_key_constraints(&engine),
            Err(ExecuteError::Unsupported(message))
                if message == "resident INSERT key proof is not enabled for this batch"
        ));
    }

    #[test]
    fn proof_only_seam_keeps_both_live_index_gates_in_place() {
        let row_local = include_str!("row_local_constraints.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production row-local gate precedes its tests");
        assert!(row_local.contains("table.indexes.is_empty()"));

        let fixed_insert = include_str!("../engine_residency/fixed_insert.rs");
        let source_matches_table = fixed_insert
            .split("fn source_matches_table")
            .nth(1)
            .and_then(|source| source.split("\n}\n").next())
            .expect("fixed insert source matcher exists");
        assert!(source_matches_table.contains("table.indexes.is_empty()"));
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_resident_key_proof_declines_sabotaged_current_generation() {
        let Some(engine) = gpu_resident_unique_engine("resident_key_history_floor") else {
            return;
        };
        let original = engine.committed_seq();
        let plan = proof_only_plan(&engine, "INSERT INTO resident_key_history_floor VALUES (8)");
        let visible_plan =
            proof_only_plan(&engine, "INSERT INTO resident_key_history_floor VALUES (7)");
        engine.read_state.residency.with_shards_mut_for_table(
            "resident_key_history_floor",
            |tables| {
                let shard = tables
                    .get_mut("resident_key_history_floor")
                    .expect("fixture table remains resident")
                    .iter_mut()
                    .find(|shard| shard.row_count != 0)
                    .expect("fixture has a non-empty shard");
                shard.history_floor_index = original.saturating_add(1);
            },
        );
        assert_serialization_without_side_effects(&engine, &plan);
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let boundary_before = engine.committed_seq();
        let error = visible_plan
            .validate_current_resident_key_constraints(&engine)
            .expect_err("visible resident duplicate must outrank a deferred history floor");
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::UniqueViolation(message))
                if message.contains("resident_key_history_floor_id_key")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.committed_seq(), boundary_before);

        let Some(engine) = gpu_resident_unique_engine("resident_key_missing_created") else {
            return;
        };
        let plan = proof_only_plan(
            &engine,
            "INSERT INTO resident_key_missing_created VALUES (8)",
        );
        engine.read_state.residency.with_shards_mut_for_table(
            "resident_key_missing_created",
            |tables| {
                let shard = tables
                    .get_mut("resident_key_missing_created")
                    .expect("fixture table remains resident")
                    .iter_mut()
                    .find(|shard| shard.row_count != 0)
                    .expect("fixture has a non-empty shard");
                shard.max_created_by = shard.max_created_by.max(1);
                shard.created_by_region = None;
            },
        );
        assert_serialization_without_side_effects(&engine, &plan);

        let Some(mut engine) = gpu_resident_unique_engine("resident_key_pressure") else {
            return;
        };
        let plan = proof_only_plan(&engine, "INSERT INTO resident_key_pressure VALUES (8)");
        engine.mark_gpu_memory_pressured(0);
        assert_serialization_without_side_effects(&engine, &plan);

        let Some(engine) = gpu_resident_unique_engine("resident_key_catalog_drift") else {
            return;
        };
        let plan = proof_only_plan(&engine, "INSERT INTO resident_key_catalog_drift VALUES (8)");
        engine
            .execute_text(
                3,
                "CREATE INDEX resident_key_catalog_drift_extra ON resident_key_catalog_drift (id)",
            )
            .unwrap();
        assert_serialization_without_side_effects(&engine, &plan);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_resident_key_proof_seals_autocommit_scope_at_compile() {
        let Some(engine) = gpu_resident_unique_engine("resident_key_transaction_scope") else {
            return;
        };
        engine.execute_text(3, "BEGIN").unwrap();
        let transaction = engine
            .transaction_snapshot_handle(3)
            .expect("BEGIN retains an explicit transaction snapshot");
        let scope = engine.enter_transaction_read(transaction);
        let plan = proof_only_plan(
            &engine,
            "INSERT INTO resident_key_transaction_scope VALUES (8)",
        );
        drop(scope);
        assert_serialization_without_side_effects(&engine, &plan);
        engine.execute_text(3, "ROLLBACK").unwrap();
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_resident_key_proof_covers_compound_nullable_text_and_wide_layouts() {
        let mut engine = Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return;
        }
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);
        engine
            .execute_text(
                1,
                "CREATE TABLE resident_wide_text (id int4 PRIMARY KEY, note text, \
                 amount numeric(10,2), \
                 CONSTRAINT resident_wide_text_compound UNIQUE (note, amount))",
            )
            .unwrap();
        engine
            .execute_text(
                2,
                "INSERT INTO resident_wide_text VALUES (1, 'alpha', 12.34)",
            )
            .unwrap();
        engine
            .populate_relational_residency_snapshot("resident_wide_text")
            .unwrap();
        let shards = engine.read_residency_shards();
        assert!(shards["resident_wide_text"].iter().any(|shard| {
            shard.row_count == 1
                && !shard.resident_device_text_columns.is_empty()
                && !shard.resident_device_numeric_columns.is_empty()
                && shard.device_memory.is_some()
        }));

        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let boundary_before = engine.committed_seq();
        let error = proof_only_plan(
            &engine,
            "INSERT INTO resident_wide_text VALUES (2, 'alpha', 12.34)",
        )
        .validate_current_resident_key_constraints(&engine)
        .expect_err("compound text/numeric resident duplicate must be found on GPU");
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::UniqueViolation(message))
                if message.contains("resident_wide_text_compound")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.committed_seq(), boundary_before);

        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let boundary_before = engine.committed_seq();
        proof_only_plan(
            &engine,
            "INSERT INTO resident_wide_text VALUES (2, NULL, 12.34)",
        )
        .validate_current_resident_key_constraints(&engine)
        .expect("ordinary UNIQUE semantics allow any compound key containing NULL");
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.committed_seq(), boundary_before);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_resident_key_proof_orders_sql_errors_without_side_effects() {
        let mut engine = Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return;
        }
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);
        engine
            .execute_text(
                1,
                "CREATE TABLE resident_key (id int4, code int4, value int4, \
                 CONSTRAINT resident_key_pk PRIMARY KEY (id), \
                 CONSTRAINT resident_key_code_unique UNIQUE (code), \
                 CONSTRAINT resident_key_positive CHECK (value > 0))",
            )
            .unwrap();
        engine
            .execute_text(2, "INSERT INTO resident_key VALUES (7, 70, 1)")
            .unwrap();
        engine
            .populate_relational_residency_snapshot("resident_key")
            .unwrap();
        let shards = engine.read_residency_shards();
        assert!(
            shards["resident_key"]
                .iter()
                .any(|shard| shard.row_count != 0)
                && shards["resident_key"].iter().all(|shard| {
                    shard.device_memory.is_some()
                        && engine.shard_write_locate_cell_live(
                            "resident_key",
                            shard.shard_id,
                            shard.device_memory.as_ref().unwrap(),
                        )
                }),
            "proof test requires a live, non-empty sharded generation: {:?}",
            shards["resident_key"]
        );

        let assert_error = |sql: &str, expected_constraint: &str| {
            let wal_before = engine.durable_wal_records().len();
            let row_id_before = engine.read_state.mvcc.current_row_id();
            let boundary_before = engine.committed_seq();
            let plan = proof_only_plan(&engine, sql);
            let error = plan
                .validate_current_resident_key_constraints(&engine)
                .expect_err("the proof-only seam must return its ordered SQL terminal");
            assert!(
                matches!(
                error,
                ExecuteError::Engine(EngineError::CheckViolation(ref message))
                        if expected_constraint == "resident_key_positive"
                            && message.contains(expected_constraint)
                ) || matches!(
                error,
                ExecuteError::Engine(EngineError::UniqueViolation(ref message))
                        if expected_constraint != "resident_key_positive"
                            && message.contains(expected_constraint)
                )
            );
            assert_eq!(engine.durable_wal_records().len(), wal_before);
            assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
            assert_eq!(engine.committed_seq(), boundary_before);
        };

        // Same-row CHECK beats both resident unique terminals; a resident row at row 0 beats a
        // later CHECK; and raw catalog index order breaks a same-row PK/UNIQUE tie.
        assert_error(
            "INSERT INTO resident_key VALUES (7, 70, -1)",
            "resident_key_positive",
        );
        assert_error(
            "INSERT INTO resident_key VALUES (8, 80, -1), (7, 71, 1)",
            "resident_key_positive",
        );
        assert_error(
            "INSERT INTO resident_key VALUES (7, 71, 1), (8, 80, -1)",
            "resident_key_pk",
        );
        assert_error(
            "INSERT INTO resident_key VALUES (8, 70, 1), (8, 80, 1)",
            "resident_key_code_unique",
        );
        assert_error(
            "INSERT INTO resident_key VALUES (8, 80, 1), (8, 80, 1)",
            "resident_key_pk",
        );
        assert_error(
            "INSERT INTO resident_key VALUES (7, 70, 1)",
            "resident_key_pk",
        );

        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let boundary_before = engine.committed_seq();
        proof_only_plan(&engine, "INSERT INTO resident_key VALUES (8, 80, 1)")
            .validate_current_resident_key_constraints(&engine)
            .expect("the proof-only validation has no apply, WAL, or publication side effect");
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.committed_seq(), boundary_before);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_resident_key_proof_preserves_raw_unique_order_across_nonunique_indexes() {
        let mut engine = Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return;
        }
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);
        engine
            .execute_text(1, "CREATE TABLE resident_index_order (id int4)")
            .unwrap();
        engine
            .execute_text(
                2,
                "CREATE UNIQUE INDEX resident_index_order_first ON resident_index_order (id)",
            )
            .unwrap();
        engine
            .execute_text(
                3,
                "CREATE INDEX resident_index_order_middle ON resident_index_order (id)",
            )
            .unwrap();
        engine
            .execute_text(
                4,
                "CREATE UNIQUE INDEX resident_index_order_second ON resident_index_order (id)",
            )
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let indexes = &catalog.relational_catalog["resident_index_order"].indexes;
        assert_eq!(
            indexes
                .iter()
                .map(|index| index.name.as_str())
                .collect::<Vec<_>>(),
            [
                "resident_index_order_first",
                "resident_index_order_middle",
                "resident_index_order_second",
            ]
        );
        assert_eq!(
            indexes.iter().map(|index| index.unique).collect::<Vec<_>>(),
            [true, false, true]
        );
        engine
            .execute_text(5, "INSERT INTO resident_index_order VALUES (7)")
            .unwrap();
        engine
            .populate_relational_residency_snapshot("resident_index_order")
            .unwrap();

        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let boundary_before = engine.committed_seq();
        let error = proof_only_plan(&engine, "INSERT INTO resident_index_order VALUES (7)")
            .validate_current_resident_key_constraints(&engine)
            .expect_err("both raw UNIQUE bindings find the resident row");
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::UniqueViolation(message))
                if message.contains("resident_index_order_first")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.committed_seq(), boundary_before);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_resident_key_proof_rereads_post_snapshot_generation_and_orders_history() {
        let mut engine = Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return;
        }
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);
        engine
            .execute_text(1, "CREATE TABLE resident_history (id int4 UNIQUE)")
            .unwrap();
        engine
            .populate_relational_residency_snapshot("resident_history")
            .unwrap();
        let original = engine.committed_seq();
        let plan = proof_only_plan(&engine, "INSERT INTO resident_history VALUES (7)");
        let prior_payload = engine.read_residency_shards()["resident_history"]
            .iter()
            .find_map(|shard| shard.device_memory.as_ref().cloned())
            .expect("S generation has a payload");

        // This is deliberately a plan compiled at S. The later INSERT republishes an unchanged
        // target catalog at a newer sequence and replaces the resident generation. Exact target
        // revalidation must permit that monotonic sequence advance and scan the replacement.
        engine
            .execute_text(2, "INSERT INTO resident_history VALUES (7)")
            .unwrap();
        engine
            .populate_relational_residency_snapshot("resident_history")
            .unwrap();
        let current_shards = engine.read_residency_shards();
        let current = current_shards["resident_history"]
            .iter()
            .find(|shard| shard.row_count == 1)
            .expect("current generation contains the post-snapshot key");
        assert!(
            !Arc::ptr_eq(current.device_memory.as_ref().unwrap(), &prior_payload),
            "the test must observe a replacement generation rather than reuse the old absence"
        );
        assert!(current.max_created_by > original);

        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let boundary_before = engine.committed_seq();
        let error = plan
            .validate_current_resident_key_constraints(&engine)
            .expect_err("the replacement generation must not be treated as the old miss");
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::UniqueViolation(ref message))
                if message.contains("resident_history_id_key")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.committed_seq(), boundary_before);

        // Claim-release is independently retryable: the proof saw the claim at S, then the
        // current generation released it. The test asserts the release cannot turn into a clean
        // miss or produce any write-side effect.
        engine
            .execute_text(3, "CREATE TABLE resident_claim_release (id int4 UNIQUE)")
            .unwrap();
        engine
            .execute_text(4, "INSERT INTO resident_claim_release VALUES (7)")
            .unwrap();
        engine
            .populate_relational_residency_snapshot("resident_claim_release")
            .unwrap();
        let claim_plan = proof_only_plan(&engine, "INSERT INTO resident_claim_release VALUES (7)");
        engine
            .execute_text(5, "DELETE FROM resident_claim_release WHERE id = 7")
            .unwrap();
        let released_shards = engine.read_residency_shards();
        assert!(released_shards["resident_claim_release"]
            .iter()
            .any(|shard| { shard.row_count != 0 && shard.deleted_by_region.is_some() }));
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let boundary_before = engine.committed_seq();
        let error = claim_plan
            .validate_current_resident_key_constraints(&engine)
            .expect_err("post-S claim-release must serialize rather than become a clean miss");
        assert!(matches!(
            error,
            ExecuteError::Serialization(message) if message.contains("history changed")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert_eq!(engine.committed_seq(), boundary_before);
    }
}
