//! Current-generation UNIQUE/PRIMARY KEY validation ingredients for typed INSERT.
//!
//! This module owns no WAL, row identities, apply, or publication. It pins the exact resident
//! generation consumed by the codec-5 indexed physical tail after semantic validation closes.

use std::sync::Arc;
use std::sync::MutexGuard;

use super::host_retention::{HostRetentionGeometry, HostRetentionReport};
use super::resident_constraint_generation;
use crate::relational_model::RelationalTable;
use crate::{Engine, ExecuteError, Index, RelationalResidentShard};

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
    /// Retained host backing for the validation seal. Its point-route generation is a per-GPU
    /// generation pin, never a generic host owner or host-generation pin.
    pub(crate) fn append_host_retention(
        &self,
        report: &mut HostRetentionReport,
    ) -> Result<(), crate::EngineError> {
        report.retain_string(&self.table_name)?;
        report.retain_boxed_slice(&self.shards)?;
        Ok(())
    }

    /// Allocation-free pre-lease geometry. The string and sealed shard box are independent
    /// owners; the generation Arc remains a device-generation pin rather than host storage.
    pub(crate) fn host_retention_geometry(
        &self,
    ) -> Result<HostRetentionGeometry, crate::EngineError> {
        let mut geometry = HostRetentionGeometry::default();
        append_string_geometry(&mut geometry, &self.table_name, "resident key table name")?;
        geometry.checked_add_backing_elements::<ResidentKeyShardResourceEvidence>(
            self.shards.len(),
            "resident key shard evidence",
        )?;
        Ok(geometry)
    }

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

    pub(crate) fn matches_fixed_rollover_predecessor(
        &self,
        table: &RelationalTable,
        catalog_seq: Index,
        shard: &RelationalResidentShard,
        predecessor_boundary: Index,
    ) -> bool {
        self.matches_in_place_append(table, catalog_seq, shard, predecessor_boundary)
    }

    pub(crate) fn original_read_snapshot(&self) -> Index {
        self.original_read_snapshot
    }

    #[cfg(test)]
    pub(crate) fn predecessor_boundary(&self) -> Index {
        self.predecessor_boundary
    }
}

/// Bind the hot generation required by the WRITE-001 indexed terminal.
///
/// There is intentionally no key verdict here: the generic transaction finalizer has already
/// closed UNIQUE/PRIMARY conflicts on the GPU. The seal pins only the exact resident generation,
/// sidecar geometry, and point-route identity consumed by the physical index tail.
pub(crate) fn validate_current_index_in_place_generation(
    engine: &Engine,
    table: &RelationalTable,
    current_catalog: &crate::CatalogSnapshot,
    predecessor_boundary: Index,
    held_mutation_gate: &MutexGuard<'_, ()>,
) -> Result<ResidentKeyValidationSeal, ExecuteError> {
    validate_indexed_generation_against_public_catalog(
        engine,
        table,
        table,
        current_catalog,
        predecessor_boundary,
        held_mutation_gate,
    )
}

/// S3 CREATE INDEX over a populated relation has one public predecessor catalog shape and one
/// final composed shape.  The private index identity has already passed the transaction's GPU
/// UNIQUE validation; this seals only the unchanged hot resident generation without pretending
/// the final index was published before WAL.
pub(crate) fn validate_s3_created_index_generation(
    engine: &Engine,
    public_table: &RelationalTable,
    final_table: &RelationalTable,
    current_catalog: &crate::CatalogSnapshot,
    predecessor_boundary: Index,
    held_mutation_gate: &MutexGuard<'_, ()>,
) -> Result<ResidentKeyValidationSeal, ExecuteError> {
    let mut final_without_new_indexes = final_table.clone();
    final_without_new_indexes.indexes = public_table.indexes.clone();
    if final_without_new_indexes != *public_table {
        return Err(decline(
            "S3-created index changed the public relation outside its index list",
        ));
    }
    validate_indexed_generation_against_public_catalog(
        engine,
        public_table,
        final_table,
        current_catalog,
        predecessor_boundary,
        held_mutation_gate,
    )
}

fn validate_indexed_generation_against_public_catalog(
    engine: &Engine,
    public_table: &RelationalTable,
    table: &RelationalTable,
    current_catalog: &crate::CatalogSnapshot,
    predecessor_boundary: Index,
    held_mutation_gate: &MutexGuard<'_, ()>,
) -> Result<ResidentKeyValidationSeal, ExecuteError> {
    if current_catalog.commit_seq != predecessor_boundary
        || current_catalog.relational_catalog.get(&public_table.name) != Some(public_table)
        || table.indexes.is_empty()
        || table.indexes.iter().any(|index| {
            (index.primary_key && !index.unique)
                || (index.unique_constraint && !index.unique)
                || index.key_columns.is_empty()
        })
    {
        return Err(decline(
            "indexed codec-5 path lost its exact maintained-index table shape",
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
    {
        return Err(decline(
            "indexed codec-5 canary requires one hot non-lane generation",
        ));
    }

    let expected_gpu = engine.planner.default_gpu_id();
    // The current predecessor is the only safe floor for the cache/sidecar generation the later
    // prepared tail may mutate. The transaction finalizer owns any historical conflict verdict;
    // this pin remains independent of the transaction's earlier SQL read snapshot.
    let pinned_generation = resident_constraint_generation::pin_hot_shard_generation(
        engine,
        table,
        predecessor_boundary,
        expected_gpu,
        held_mutation_gate,
    )?;
    if pinned_generation.history_floor_requires_retry() {
        return Err(decline(
            "indexed codec-5 canary hot generation history floor advanced",
        ));
    }
    let shards = pinned_generation.shards();
    let generation = shards
        .first()
        .map(|shard| Arc::clone(&shard.point_route_generation))
        .ok_or_else(|| decline("indexed codec-5 canary lost its hot shard generation"))?;
    let shard_evidence = shards
        .iter()
        .map(|shard| {
            let payload = shard
                .device_memory
                .as_ref()
                .ok_or_else(|| decline("indexed codec-5 canary lost a hot shard payload"))?;
            Ok(ResidentKeyShardResourceEvidence {
                shard_id: shard.shard_id,
                payload_ptr: payload.device_ptr(),
                row_count: shard.row_count,
                capacity: shard.capacity,
            })
        })
        .collect::<Result<Box<[_]>, ExecuteError>>()?;
    Ok(ResidentKeyValidationSeal {
        table_name: table.name.clone(),
        catalog_seq: current_catalog.commit_seq,
        original_read_snapshot: predecessor_boundary,
        predecessor_boundary,
        gpu_id: expected_gpu,
        generation,
        shards: shard_evidence,
    })
}

fn append_string_geometry(
    geometry: &mut HostRetentionGeometry,
    value: &String,
    domain: &'static str,
) -> Result<(), crate::EngineError> {
    if value.capacity() == 0 {
        return Ok(());
    }
    let bytes = u64::try_from(value.capacity()).map_err(|_| {
        crate::EngineError::Durability("resident key string capacity overflows".to_string())
    })?;
    geometry.checked_add_backing_bytes_slots(bytes, 1, domain)
}

fn decline(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Serialization(message.into())
}
