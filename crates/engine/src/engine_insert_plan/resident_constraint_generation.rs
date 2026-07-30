//! Neutral GPU-resident constraint-generation and key-layout validation.
//!
//! This module pins one immutable hot-shard generation and derives device fold descriptors from
//! the sealed batch and resident layout. It owns neither constraint semantics nor a CUDA verdict:
//! UNIQUE/PK and a future inert FK proof consume these validated inputs independently.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, MutexGuard};

use crate::relational_model::{RelationalColumn, RelationalResidencySnapshot, RelationalTable};
use crate::typed_insert_batch::{TypedInsertBatch, TypedInsertConstraintDeviceSource};
use crate::{Engine, EngineError, ExecuteError, Index, RelationalResidentShard, SqlType};
use gpu_db_execution::{CudaCompoundFoldColumn, CudaResidentDeviceMemory};

/// An immutable shard-map pin validated beneath the caller's residency mutation gate.
///
/// The map Arc keeps the exact generation alive across descriptor construction and device proof
/// launches. Its retained guard borrow makes the pin impossible to retain after the caller's
/// residency mutation interval; the caller still owns every SQL constraint decision.
pub(super) struct PinnedResidentConstraintGeneration<'guard, 'mutex> {
    shard_map: Arc<BTreeMap<String, Vec<RelationalResidentShard>>>,
    table_name: String,
    history_floor_requires_retry: bool,
    runtime: gpu_db_execution::GpuRuntimeSnapshot,
    _held_mutation_gate: &'guard MutexGuard<'mutex, ()>,
}

/// Opaque validated table slice from the exact map and pressure snapshot held by a generation
/// pin.  Callers can inspect only the generation they just validated; the map itself remains
/// private to prevent a later name lookup from bypassing the same-map invariant.
pub(super) struct ValidatedResidentConstraintTable<'pin> {
    shards: &'pin [RelationalResidentShard],
    history_floor_requires_retry: bool,
    generation: Arc<()>,
}

impl<'pin> ValidatedResidentConstraintTable<'pin> {
    pub(super) fn shards(&self) -> &'pin [RelationalResidentShard] {
        self.shards
    }

    pub(super) fn history_floor_requires_retry(&self) -> bool {
        self.history_floor_requires_retry
    }

    pub(super) fn generation(&self) -> &Arc<()> {
        &self.generation
    }
}

/// One exact catalog column admitted to a resident constraint layout.
///
/// The fields are private and construction always rechecks the complete source catalog column
/// against its owning table. Consumers therefore cannot manufacture a constraint key from an
/// arbitrary name, type, or ordinal.
#[derive(Clone)]
pub(crate) struct ResidentConstraintColumnBinding {
    id: u32,
    table_oid: u32,
    attnum: i16,
    name: String,
    ty: SqlType,
    type_oid: u32,
    type_size: i16,
}

/// Bind an ordered catalog column slice for a resident constraint proof.
///
/// UNIQUE adapts its sealed raw-index witness here. A future FK compiler can supply its exactly
/// resolved referencing/referenced catalog members without fabricating a synthetic index.
pub(super) fn bind_catalog_columns<'a>(
    table: &RelationalTable,
    columns: impl IntoIterator<Item = &'a RelationalColumn>,
) -> Result<Box<[ResidentConstraintColumnBinding]>, ExecuteError> {
    columns
        .into_iter()
        .map(|column| {
            table
                .columns
                .iter()
                .find(|candidate| *candidate == column)
                .map(|catalog| ResidentConstraintColumnBinding {
                    id: catalog.id,
                    table_oid: catalog.table_oid,
                    attnum: catalog.attnum,
                    name: catalog.name.clone(),
                    ty: catalog.ty,
                    type_oid: catalog.type_oid,
                    type_size: catalog.type_size,
                })
                .ok_or_else(|| decline("resident key binding lost its catalog column"))
        })
        .collect()
}

impl<'guard, 'mutex> PinnedResidentConstraintGeneration<'guard, 'mutex> {
    pub(super) fn shards(&self) -> &[RelationalResidentShard] {
        self.shard_map
            .get(&self.table_name)
            .expect("validated pinned resident constraint generation lost its table")
    }

    pub(super) fn history_floor_requires_retry(&self) -> bool {
        self.history_floor_requires_retry
    }

    /// Validate another table against the same immutable shard-map Arc captured for the child.
    /// No caller may reload residency while one FK proof is in flight.
    pub(super) fn validate_table(
        &self,
        engine: &Engine,
        table: &RelationalTable,
        original_read_snapshot: Index,
        expected_gpu: u16,
    ) -> Result<ValidatedResidentConstraintTable<'_>, ExecuteError> {
        let shards = self.shard_map.get(&table.name).ok_or_else(|| {
            decline("resident INSERT key proof found no hot shard generation for its table")
        })?;
        let history_floor_requires_retry = validate_hot_shard_generation(
            engine,
            table,
            shards,
            original_read_snapshot,
            expected_gpu,
            &self.runtime,
        )?;
        Ok(ValidatedResidentConstraintTable {
            shards,
            history_floor_requires_retry,
            generation: Arc::clone(
                &shards
                    .first()
                    .expect("validated resident constraint table unexpectedly has no shard")
                    .point_route_generation,
            ),
        })
    }
}

/// Pin and validate the exact authoritative hot generation for a later device constraint proof.
///
/// The caller holds the established mutation gate, so this deliberately does not acquire a lock,
/// reload a snapshot, or make an admission/publication decision.
pub(super) fn pin_hot_shard_generation<'guard, 'mutex>(
    engine: &Engine,
    table: &RelationalTable,
    original_read_snapshot: Index,
    expected_gpu: u16,
    held_mutation_gate: &'guard MutexGuard<'mutex, ()>,
) -> Result<PinnedResidentConstraintGeneration<'guard, 'mutex>, ExecuteError> {
    let shard_map = engine.read_state.residency.shards.load_full();
    let shards = shard_map.get(&table.name).ok_or_else(|| {
        decline("resident INSERT key proof found no hot shard generation for its table")
    })?;
    let runtime = engine.router.runtime().snapshot();
    let history_floor_requires_retry = validate_hot_shard_generation(
        engine,
        table,
        shards,
        original_read_snapshot,
        expected_gpu,
        &runtime,
    )?;
    Ok(PinnedResidentConstraintGeneration {
        shard_map,
        table_name: table.name.clone(),
        history_floor_requires_retry,
        runtime,
        _held_mutation_gate: held_mutation_gate,
    })
}

fn validate_hot_shard_generation(
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
        let valid_sidecar = |memory: &Arc<CudaResidentDeviceMemory>| {
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

/// The descriptor count for the incoming batch side of one physical key proof.
pub(crate) fn descriptor_count_for_batch(
    batch: &TypedInsertBatch,
    columns: &[ResidentConstraintColumnBinding],
) -> Result<usize, EngineError> {
    let mut validity = BTreeSet::new();
    for column in columns {
        if batch.row_local_constraint_column_has_validity_bitmap(column.id)? {
            validity.insert((column.attnum, column.id));
        }
    }
    columns.len().checked_add(validity.len()).ok_or_else(|| {
        EngineError::ApplyFailed("resident key descriptor count overflows".to_string())
    })
}

/// The descriptor count for the resident side of one physical key proof.
pub(crate) fn descriptor_count_for_resident(
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    columns: &[ResidentConstraintColumnBinding],
) -> Result<usize, ExecuteError> {
    let mut validity = BTreeSet::new();
    for column in columns {
        let column_idx = table_column_index(table, column)?;
        if crate::resident_device_null_column_offset(snapshot, table, column_idx)
            .map_err(|error| decline(format!("resident key layout declined: {error}")))?
            .is_some()
        {
            validity.insert((column.attnum, column.id));
        }
    }
    columns
        .len()
        .checked_add(validity.len())
        .ok_or_else(|| decline("resident key descriptor count overflows"))
}

/// Build the incoming device descriptors in the catalog key order, followed by deduplicated
/// validity descriptors in stable attribute order.
pub(crate) fn incoming_columns(
    source: &TypedInsertConstraintDeviceSource,
    columns: &[ResidentConstraintColumnBinding],
) -> Result<Vec<CudaCompoundFoldColumn>, EngineError> {
    let mut descriptors = Vec::with_capacity(columns.len() * 2);
    let mut validity = BTreeSet::new();
    for column in columns {
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

/// Build the resident descriptors for one pinned shard using its exact published layout.
pub(crate) fn resident_columns(
    engine: &Engine,
    table: &RelationalTable,
    shard: &RelationalResidentShard,
    columns: &[ResidentConstraintColumnBinding],
) -> Result<Vec<CudaCompoundFoldColumn>, ExecuteError> {
    let snapshot = engine.resident_snapshot_for_shard(shard, table);
    resident_columns_for_snapshot(table, &snapshot, columns)
}

/// Build the resident descriptors for one private or published resident snapshot.
pub(crate) fn resident_columns_for_snapshot(
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    columns: &[ResidentConstraintColumnBinding],
) -> Result<Vec<CudaCompoundFoldColumn>, ExecuteError> {
    let mut descriptors = Vec::with_capacity(columns.len() * 2);
    let mut validity = BTreeSet::new();
    for column in columns {
        let column_idx = table_column_index(table, column)?;
        let data = match column.ty {
            SqlType::Int2 | SqlType::Int4 | SqlType::Date => CudaCompoundFoldColumn::Fixed {
                byte_offset: crate::resident_device_int4_column_offset(snapshot, table, column_idx)
                    .map_err(|error| {
                        decline(format!("resident int4 key layout declined: {error}"))
                    })?,
                width_words: 1,
            },
            SqlType::Int8 | SqlType::Timestamp => CudaCompoundFoldColumn::Fixed {
                byte_offset: crate::resident_device_int8_column_offset(snapshot, table, column_idx)
                    .map_err(|error| {
                        decline(format!("resident int8 key layout declined: {error}"))
                    })?,
                width_words: 2,
            },
            SqlType::Numeric { .. } | SqlType::Uuid => CudaCompoundFoldColumn::Fixed {
                byte_offset: crate::resident_device_numeric_column_offset(
                    snapshot, table, column_idx,
                )
                .map_err(|error| decline(format!("resident wide key layout declined: {error}")))?,
                width_words: 4,
            },
            SqlType::Bool => CudaCompoundFoldColumn::Bool {
                bitmap_byte_offset: crate::resident_device_bool_column_offset(
                    snapshot, table, column_idx,
                )
                .map_err(|error| decline(format!("resident bool key layout declined: {error}")))?,
            },
            SqlType::Text => {
                let layout = crate::resident_device_text_column_layout(snapshot, table, column_idx)
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
        if let Some(offset) = crate::resident_device_null_column_offset(snapshot, table, column_idx)
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
    binding: &ResidentConstraintColumnBinding,
) -> Result<usize, ExecuteError> {
    table
        .columns
        .iter()
        .position(|column| {
            column.id == binding.id
                && column.table_oid == binding.table_oid
                && column.attnum == binding.attnum
                && column.name == binding.name
                && column.ty == binding.ty
                && column.type_oid == binding.type_oid
                && column.type_size == binding.type_size
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
    fn resident_constraint_generation_is_neutral_and_owns_the_pinned_layout_seam() {
        let source = include_str!("resident_constraint_generation.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production generation seam precedes tests");
        assert!(source.contains("struct PinnedResidentConstraintGeneration"));
        assert!(source.contains("shard_map: Arc<BTreeMap<String, Vec<RelationalResidentShard>>>"));
        assert!(source.contains("fn pin_hot_shard_generation"));
        assert!(source.contains("held_mutation_gate: &'guard MutexGuard<'mutex, ()>"));
        assert!(source.contains("_held_mutation_gate: &'guard MutexGuard<'mutex, ()>"));
        assert!(source.contains("_held_mutation_gate: held_mutation_gate"));
        assert!(source.contains("pub(super) fn shards"));
        assert!(source.contains("pub(super) fn history_floor_requires_retry"));
        assert!(source.contains("fn descriptor_count_for_batch"));
        assert!(source.contains("fn incoming_columns"));
        assert!(source.contains("fn resident_columns_for_snapshot"));
        assert!(source.contains("fn table_column_index"));
        assert!(source.contains("struct ResidentConstraintColumnBinding"));
        assert!(source.contains("fn bind_catalog_columns"));
        assert!(source.contains("table_oid: u32"));
        assert!(source.contains("type_oid: u32"));
        assert!(source.contains("type_size: i16"));
        assert!(!source.contains(concat!("Index", "Binding")));
        assert!(!source.contains(concat!("KeyColumn", "Binding")));
        for forbidden in [
            "CudaInsertResidentKeyShard",
            "insert_resident_key_verdict",
            concat!("CudaInsert", "ForeignKey"),
            "wal_",
            "apply_",
        ] {
            assert!(
                !source.contains(forbidden),
                "neutral generation seam contains {forbidden}"
            );
        }
    }

    #[test]
    fn pin_constructor_signature_retains_the_held_mutation_guard() {
        fn construct_with_guard<'guard, 'mutex>(
            engine: &Engine,
            table: &RelationalTable,
            guard: &'guard MutexGuard<'mutex, ()>,
        ) -> Result<PinnedResidentConstraintGeneration<'guard, 'mutex>, ExecuteError> {
            pin_hot_shard_generation(engine, table, 0, 0, guard)
        }

        let _constructor = construct_with_guard;
    }

    #[test]
    fn resident_constraint_column_binding_rejects_every_catalog_identity_drift() {
        let engine = Engine::new_local_test_engine();
        engine
            .execute_text(1, "CREATE TABLE resident_constraint_identity (value int4)")
            .unwrap();
        let table = engine
            .relational_catalog_table("resident_constraint_identity")
            .expect("created table remains in the catalog");
        let binding = bind_catalog_columns(&table, [&table.columns[0]])
            .expect("catalog column binds exactly")
            .into_vec()
            .pop()
            .expect("one bound column");

        let mut id = binding.clone();
        id.id += 1;
        assert!(table_column_index(&table, &id).is_err());

        let mut table_oid = binding.clone();
        table_oid.table_oid += 1;
        assert!(table_column_index(&table, &table_oid).is_err());

        let mut attnum = binding.clone();
        attnum.attnum += 1;
        assert!(table_column_index(&table, &attnum).is_err());

        let mut name = binding.clone();
        name.name.push_str("_drift");
        assert!(table_column_index(&table, &name).is_err());

        let mut ty = binding.clone();
        ty.ty = SqlType::Bool;
        assert!(table_column_index(&table, &ty).is_err());

        let mut type_oid = binding.clone();
        type_oid.type_oid += 1;
        assert!(table_column_index(&table, &type_oid).is_err());

        let mut type_size = binding;
        type_size.type_size += 1;
        assert!(table_column_index(&table, &type_size).is_err());
    }
}
