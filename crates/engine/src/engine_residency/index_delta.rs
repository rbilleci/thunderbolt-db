//! Inert, test-only ownership for one already-published indexed in-place append proof.
//!
//! This leaf prepares no canonical operation and exposes no device mutation entry point.  It
//! exists to make the exact resource/lifetime handoff auditable before WRITE-001 intentionally
//! connects it to a live commit path.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use super::fixed_insert::ResidentOpenShardAppendPlan;
use crate::engine_insert_plan::batch_key_constraints::BatchKeyConstraintProof;
use crate::engine_insert_plan::resident_key_constraints::{self, ResidentKeyValidationSeal};
use crate::engine_state::TransactionNamedIndexPublicationGuard;
use crate::relational_model::RelationalTable;
use crate::typed_insert_batch::TypedInsertBatch;
use crate::{Engine, ExecuteError, Index, RelationalResidentShard};
use gpu_db_execution::{
    resident_index_allocated_bytes, resident_typed_indexes_insert_preparation_bytes,
    CudaAllocationScope, CudaResidentDeviceMemory, CudaResidentTypedIndexInsert,
    PreparedResidentTypedIndexesInsert,
};

/// Every catalog index is represented here in raw catalog order.  Multiple raw indexes may use
/// one physical directory only when the established `index_probe_key_id` says they do.
struct RawIndexLogicalBinding {
    raw_ordinal: usize,
    key_id: usize,
    physical_ordinal: usize,
}

struct PhysicalIndexLogicalBinding {
    raw_ordinal: usize,
    key_id: usize,
    descriptor_count: usize,
}

pub(super) struct PreparedIndexDeltaLogicalBindings {
    raw: Box<[RawIndexLogicalBinding]>,
    physical: Box<[PhysicalIndexLogicalBinding]>,
    preparation_bytes: u64,
}

impl PreparedIndexDeltaLogicalBindings {
    pub(super) fn preparation_bytes(&self) -> u64 {
        self.preparation_bytes
    }
}

/// Cache evidence for one distinct physical request.  The Arc pins are deliberately retained
/// independently of the cache map so retirement cannot invalidate the prepared token's basis.
struct PhysicalIndexBinding {
    cache_key: (String, u32, usize),
    key_id: usize,
    device_index: Arc<CudaResidentDeviceMemory>,
    published_row_count: Arc<AtomicUsize>,
    published_has_postings: Arc<AtomicBool>,
    has_postings_at_prepare: bool,
    source_ptr: u64,
    base_row: usize,
    end_row: usize,
    capacity: usize,
    table_mask: u32,
    hash_shift: u32,
    gc_boundary: Index,
    allocated_bytes: u64,
}

/// Exact resource accounting for one inert prepared token.  Persistent index allocations are
/// pinned rather than allocated here; transient bytes are the two pooled leases retained by the
/// existing execution token until this owner drops.
struct IndexDeltaResourceLedger {
    pinned_persistent_index_bytes: u64,
    transient_preparation_bytes: u64,
    descriptor_bytes: u64,
    descriptor_count: usize,
    bounded_readback_bytes: u64,
    allocation_pin_count: usize,
    raw_index_count: usize,
    physical_index_count: usize,
}

/// Scalar-only observation available to proof tests.  Device pointers, CUDA tokens, append
/// sources, cache entries, and lifecycle guards never cross this reporting boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexedInPlaceProofReport {
    pub(crate) raw_index_count: usize,
    pub(crate) physical_index_count: usize,
    pub(crate) base_row: usize,
    pub(crate) incoming_rows: usize,
    pub(crate) preparation_bytes: u64,
    pub(crate) original_read_snapshot: Index,
    pub(crate) predecessor_boundary: Index,
    pub(crate) pinned_persistent_index_bytes: u64,
    pub(crate) descriptor_bytes: u64,
    pub(crate) descriptor_count: usize,
    pub(crate) bounded_readback_bytes: u64,
    pub(crate) allocation_pin_count: usize,
}

/// Move-only device-index preparation and its exact proof/pin ledger.  Launch setup drops before
/// semantic witness and cache pins, so no retained basis can outlive the prepared CUDA drain.
struct PreparedResidentIndexDelta {
    launch: PreparedResidentTypedIndexesInsert,
    source_payload: Arc<CudaResidentDeviceMemory>,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    raw_bindings: Box<[RawIndexLogicalBinding]>,
    physical_bindings: Box<[PhysicalIndexBinding]>,
    mutation_epoch: Arc<AtomicU64>,
    mutation_epoch_expected_even: u64,
    ledger: IndexDeltaResourceLedger,
}

/// Move-only inert owner.  Declaration order is load-bearing: abandoning the proof drains its
/// whole prepared index delta first, then releases the ordinary append reservation, and only then
/// releases named-index lifecycle protection.
pub(super) struct PreparedIndexedInPlaceProof<'a> {
    index_delta: PreparedResidentIndexDelta,
    append: ResidentOpenShardAppendPlan<'a>,
    report: IndexedInPlaceProofReport,
    _named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
}

impl PreparedIndexedInPlaceProof<'_> {
    /// Consume the inert owner after exposing only its scalar accounting report.  There is no
    /// forwarding surface for the prepared CUDA token or append carrier.
    pub(super) fn inspect<R>(
        self,
        inspect: impl FnOnce(IndexedInPlaceProofReport) -> R,
    ) -> Result<R, ExecuteError> {
        let report = self.scalar_report_if_intact()?;
        Ok(inspect(report))
    }

    fn scalar_report_if_intact(&self) -> Result<IndexedInPlaceProofReport, ExecuteError> {
        let delta = &self.index_delta;
        let (shard_id, base_row, _) = self.append.proof_only_open_shard_basis();
        if !self.append.holds_budget_reservation()
            || base_row != self.report.base_row
            || self.append.row_count() != self.report.incoming_rows
            || delta.launch.preparation_bytes() != delta.ledger.transient_preparation_bytes
            || self.report.preparation_bytes != delta.ledger.transient_preparation_bytes
            || self.report.original_read_snapshot != delta.validation.original_read_snapshot()
            || self.report.predecessor_boundary != delta.validation.predecessor_boundary()
            || delta.mutation_epoch.load(Ordering::Acquire) != delta.mutation_epoch_expected_even
            || delta.mutation_epoch_expected_even & 1 != 0
            || delta.key_proof.indexes().len() != delta.raw_bindings.len()
            || delta.source_payload.device_ptr() == 0
        {
            return Err(decline(
                "indexed append proof retained resource integrity drifted",
            ));
        }
        let pinned_bytes = delta
            .physical_bindings
            .iter()
            .try_fold(0_u64, |total, binding| {
                total
                    .checked_add(binding.allocated_bytes)
                    .ok_or_else(|| decline("indexed append proof retained byte sum overflows"))
            })?;
        if pinned_bytes != delta.ledger.pinned_persistent_index_bytes
            || delta.ledger.raw_index_count != delta.raw_bindings.len()
            || delta.ledger.physical_index_count != delta.physical_bindings.len()
            || delta.ledger.allocation_pin_count != delta.physical_bindings.len() + 1
            || delta.ledger.bounded_readback_bytes != std::mem::size_of::<u32>() as u64
            || self.report.raw_index_count != delta.ledger.raw_index_count
            || self.report.physical_index_count != delta.ledger.physical_index_count
            || self.report.pinned_persistent_index_bytes
                != delta.ledger.pinned_persistent_index_bytes
            || self.report.descriptor_bytes != delta.ledger.descriptor_bytes
            || self.report.descriptor_count != delta.ledger.descriptor_count
            || self.report.bounded_readback_bytes != delta.ledger.bounded_readback_bytes
            || self.report.allocation_pin_count != delta.ledger.allocation_pin_count
        {
            return Err(decline("indexed append proof retained ledger drifted"));
        }
        for (raw_ordinal, raw) in delta.raw_bindings.iter().enumerate() {
            let physical = delta
                .physical_bindings
                .get(raw.physical_ordinal)
                .ok_or_else(|| decline("indexed append proof raw mapping lost a physical pin"))?;
            if raw.raw_ordinal != raw_ordinal
                || raw.key_id != physical.key_id
                || delta
                    .key_proof
                    .indexes()
                    .get(raw_ordinal)
                    .is_none_or(|binding| binding.raw_ordinal() != raw_ordinal)
            {
                return Err(decline("indexed append proof raw mapping drifted"));
            }
        }
        for physical in delta.physical_bindings.iter() {
            if physical.cache_key.1 != shard_id
                || physical.cache_key.2 != physical.key_id
                || physical.source_ptr != delta.source_payload.device_ptr()
                || physical.base_row != base_row
                || physical.end_row < physical.base_row
                || physical.end_row > physical.capacity
                || physical.device_index.device_ptr() == physical.source_ptr
                || physical.device_index.metadata().allocated_bytes != physical.allocated_bytes
                || physical.published_row_count.load(Ordering::Acquire) != physical.base_row
                || physical.published_has_postings.load(Ordering::Acquire)
                    != physical.has_postings_at_prepare
                || physical.gc_boundary > delta.validation.original_read_snapshot()
                || physical.table_mask == 0
                || physical.hash_shift != 32 - (u64::from(physical.table_mask) + 1).trailing_zeros()
            {
                return Err(decline("indexed append proof physical pin drifted"));
            }
        }
        Ok(self.report)
    }
}

/// Assemble and inspect the inert proof while the caller retains the canonical commit guard and
/// has passed its already-held residency mutation gate into the normal append core.
#[allow(clippy::too_many_arguments)] // proof ownership stays explicit; no opaque live-operation carrier
pub(crate) fn inspect_prepared_indexed_in_place<'a, R>(
    engine: &'a Engine,
    table: &RelationalTable,
    predecessor_boundary: Index,
    batch: TypedInsertBatch,
    row_ids: super::DeviceInsertRowIds,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    mutation_gate: std::sync::MutexGuard<'a, ()>,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
    inspect: impl FnOnce(IndexedInPlaceProofReport) -> R,
) -> Result<R, ExecuteError> {
    let logical = prepare_logical_bindings(engine, table, &key_proof)?;
    let source = batch
        .into_resident_append_source()
        .ok_or_else(|| decline("indexed append proof lost its resident append source"))?;
    let append = engine
        .prepare_resident_open_shard_append_indexed_in_place_proof(
            source,
            row_ids,
            mutation_gate,
            logical.preparation_bytes(),
        )
        .map_err(|_| {
            decline("indexed append proof is ineligible for fixed in-place preparation")
        })?;
    prepare(
        engine,
        table,
        predecessor_boundary,
        append,
        key_proof,
        validation,
        logical,
        named_index_lifecycle,
    )
    .and_then(|prepared| prepared.inspect(inspect))
}

pub(super) fn prepare_logical_bindings(
    engine: &Engine,
    table: &RelationalTable,
    keys: &BatchKeyConstraintProof,
) -> Result<PreparedIndexDeltaLogicalBindings, ExecuteError> {
    if keys.indexes().len() != table.indexes.len() || table.indexes.is_empty() {
        return Err(decline(
            "indexed append proof lost exact raw catalog enrollment",
        ));
    }
    let mut raw = Vec::with_capacity(table.indexes.len());
    let mut physical = Vec::with_capacity(table.indexes.len());
    let mut physical_by_key = BTreeMap::new();
    let mut descriptor_count = 0_usize;
    let shard_map = engine.read_state.residency.shards.load_full();
    let open = shard_map
        .get(&table.name)
        .and_then(|shards| shards.last())
        .ok_or_else(|| decline("indexed append proof has no current open shard"))?;
    for (raw_ordinal, index) in table.indexes.iter().enumerate() {
        let binding = keys
            .indexes()
            .get(raw_ordinal)
            .filter(|binding| binding.raw_ordinal() == raw_ordinal)
            .ok_or_else(|| decline("indexed append proof lost a raw catalog binding"))?;
        if !crate::engine_residency::index_all_key_columns_foldable(table, index) {
            return Err(decline(
                "indexed append proof found a non-foldable named index",
            ));
        }
        let key_id = crate::engine_residency::index_probe_key_id(table, index, raw_ordinal)
            .ok_or_else(|| decline("indexed append proof found no resident index key id"))?;
        let physical_ordinal = if let Some(&ordinal) = physical_by_key.get(&key_id) {
            ordinal
        } else {
            let ordinal = physical.len();
            let count =
                resident_key_constraints::resident_columns(engine, table, open, binding)?.len();
            descriptor_count = descriptor_count
                .checked_add(count)
                .ok_or_else(|| decline("indexed append proof descriptor count overflows"))?;
            physical.push(PhysicalIndexLogicalBinding {
                raw_ordinal,
                key_id,
                descriptor_count: count,
            });
            physical_by_key.insert(key_id, ordinal);
            ordinal
        };
        raw.push(RawIndexLogicalBinding {
            raw_ordinal,
            key_id,
            physical_ordinal,
        });
    }
    let preparation_bytes =
        resident_typed_indexes_insert_preparation_bytes(physical.len(), descriptor_count)
            .ok_or_else(|| decline("indexed append proof preparation geometry overflows"))?;
    Ok(PreparedIndexDeltaLogicalBindings {
        raw: raw.into(),
        physical: physical.into(),
        preparation_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare<'a>(
    engine: &'a Engine,
    table: &RelationalTable,
    predecessor_boundary: Index,
    append: ResidentOpenShardAppendPlan<'a>,
    key_proof: BatchKeyConstraintProof,
    validation: ResidentKeyValidationSeal,
    logical: PreparedIndexDeltaLogicalBindings,
    named_index_lifecycle: TransactionNamedIndexPublicationGuard<'a>,
) -> Result<PreparedIndexedInPlaceProof<'a>, ExecuteError> {
    let (shard_id, base_row, catalog_seq) = append.proof_only_open_shard_basis();
    let shard_map = engine.read_state.residency.shards.load_full();
    let open = shard_map
        .get(&table.name)
        .and_then(|shards| shards.iter().find(|shard| shard.shard_id == shard_id))
        .ok_or_else(|| decline("indexed append proof lost its open shard"))?;
    if !validation.matches_in_place_append(table, catalog_seq, open, predecessor_boundary)
        || validation.original_read_snapshot() > predecessor_boundary
        || open.row_count != base_row
    {
        return Err(decline("indexed append proof generation witness drifted"));
    }
    let source_ptr = open
        .device_memory
        .as_ref()
        .map(|memory| memory.device_ptr())
        .filter(|pointer| *pointer != 0)
        .ok_or_else(|| decline("indexed append proof lost its resident source allocation"))?;
    let end_row = base_row
        .checked_add(append.row_count())
        .filter(|end| *end <= open.capacity)
        .ok_or_else(|| decline("indexed append proof escaped its pinned open-shard extent"))?;
    let (requests, physical_bindings) = bind_physical_indexes(
        engine,
        table,
        open,
        source_ptr,
        base_row,
        end_row,
        &key_proof,
        &logical,
        validation.original_read_snapshot(),
    )?;
    let expected_columns = logical
        .physical
        .iter()
        .try_fold(0_usize, |total, binding| {
            total.checked_add(binding.descriptor_count)
        })
        .ok_or_else(|| decline("indexed append proof descriptor total overflows"))?;
    let observed_columns = requests
        .iter()
        .map(|request| request.columns.len())
        .sum::<usize>();
    if requests.len() != logical.physical.len() || observed_columns != expected_columns {
        return Err(decline("indexed append proof descriptor authority drifted"));
    }
    let mutation_epoch = engine
        .read_state
        .residency
        .point_index_mutation_epoch(&table.name);
    let mutation_epoch_expected_even = mutation_epoch.load(Ordering::Acquire);
    if mutation_epoch_expected_even & 1 != 0 {
        return Err(decline(
            "indexed append proof observed an active mutation epoch",
        ));
    }
    let allocation_scope = CudaAllocationScope::with_budget(logical.preparation_bytes);
    let source_payload = open
        .device_memory
        .as_ref()
        .cloned()
        .ok_or_else(|| decline("indexed append proof lost pinned open payload"))?;
    let index_delta = source_payload
        .prepare_resident_typed_indexes_insert(&requests, base_row, append.row_count())
        .map_err(|_| decline("indexed append proof CUDA preparation declined"))?;
    if index_delta.preparation_bytes() != logical.preparation_bytes {
        return Err(decline(
            "indexed append proof pooled preparation size drifted",
        ));
    }
    if mutation_epoch.load(Ordering::Acquire) != mutation_epoch_expected_even {
        return Err(decline(
            "indexed append proof mutation epoch changed during preparation",
        ));
    }
    if allocation_scope.peak_bytes() != logical.preparation_bytes {
        return Err(decline(
            "indexed append proof pooled preparation peak drifted",
        ));
    }
    drop(allocation_scope);
    let index_delta_ledger = resource_ledger(&logical, &physical_bindings)?;
    let report = IndexedInPlaceProofReport {
        raw_index_count: index_delta_ledger.raw_index_count,
        physical_index_count: index_delta_ledger.physical_index_count,
        base_row,
        incoming_rows: append.row_count(),
        preparation_bytes: index_delta_ledger.transient_preparation_bytes,
        original_read_snapshot: validation.original_read_snapshot(),
        predecessor_boundary,
        pinned_persistent_index_bytes: index_delta_ledger.pinned_persistent_index_bytes,
        descriptor_bytes: index_delta_ledger.descriptor_bytes,
        descriptor_count: index_delta_ledger.descriptor_count,
        bounded_readback_bytes: index_delta_ledger.bounded_readback_bytes,
        allocation_pin_count: index_delta_ledger.allocation_pin_count,
    };
    Ok(PreparedIndexedInPlaceProof {
        index_delta: PreparedResidentIndexDelta {
            launch: index_delta,
            source_payload,
            key_proof,
            validation,
            raw_bindings: logical.raw,
            physical_bindings: physical_bindings.into(),
            mutation_epoch,
            mutation_epoch_expected_even,
            ledger: index_delta_ledger,
        },
        append,
        report,
        _named_index_lifecycle: named_index_lifecycle,
    })
}

#[allow(clippy::too_many_arguments)]
fn bind_physical_indexes(
    engine: &Engine,
    table: &RelationalTable,
    open: &RelationalResidentShard,
    source_ptr: u64,
    base_row: usize,
    end_row: usize,
    keys: &BatchKeyConstraintProof,
    logical: &PreparedIndexDeltaLogicalBindings,
    original_read_snapshot: Index,
) -> Result<(Vec<CudaResidentTypedIndexInsert>, Vec<PhysicalIndexBinding>), ExecuteError> {
    let horizon = expected_index_horizon(open)?;
    let open_payload = open
        .device_memory
        .as_ref()
        .ok_or_else(|| decline("indexed append proof lost its open payload before cache bind"))?;
    let route_publish = engine
        .read_state
        .residency
        .sharded_point_route_publish_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let cache = engine
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let coverage = engine
        .read_state
        .residency
        .named_index_coverage
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let complete = engine
        .read_state
        .residency
        .named_index_coverage_complete
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let publications = engine
        .read_state
        .residency
        .named_index_publications
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if publications.get(&table.oid) != Some(&table.indexes)
        || !complete
            .get(&table.name)
            .is_some_and(|(oid, indexes)| *oid == table.oid && indexes == &table.indexes)
    {
        return Err(decline(
            "indexed append proof requires complete named-index enrollment",
        ));
    }
    let mut requests = Vec::with_capacity(logical.physical.len());
    let mut physical = Vec::with_capacity(logical.physical.len());
    let mut destinations = BTreeSet::new();
    for logical_binding in logical.physical.iter() {
        let binding = keys
            .indexes()
            .get(logical_binding.raw_ordinal)
            .filter(|binding| binding.raw_ordinal() == logical_binding.raw_ordinal)
            .ok_or_else(|| decline("indexed append proof raw binding changed before cache bind"))?;
        let index = table
            .indexes
            .get(logical_binding.raw_ordinal)
            .ok_or_else(|| decline("indexed append proof catalog index disappeared"))?;
        if crate::engine_residency::index_probe_key_id(table, index, logical_binding.raw_ordinal)
            != Some(logical_binding.key_id)
        {
            return Err(decline("indexed append proof logical key id drifted"));
        }
        let cache_key = (table.name.clone(), open.shard_id, logical_binding.key_id);
        let covered = coverage.get(&cache_key) == Some(&(source_ptr, base_row));
        let entry = cache
            .get(&cache_key)
            .ok_or_else(|| decline("indexed append proof has no enrolled physical index"))?;
        let device_index = entry
            .device_index
            .as_ref()
            .filter(|memory| {
                entry.resident_device_ptr == source_ptr
                    && entry.row_count == base_row
                    && Arc::ptr_eq(&entry._resident_guard, open_payload)
                    && entry.gc_boundary <= original_read_snapshot
                    && entry.published_row_count.load(Ordering::Acquire) == base_row
                    && entry.published_has_postings.load(Ordering::Acquire) == entry.has_postings
                    && entry.table_mask == horizon.table_mask
                    && entry.hash_shift == horizon.hash_shift
                    && memory.metadata().allocated_bytes == horizon.allocated_bytes
            })
            .cloned()
            .ok_or_else(|| decline("indexed append proof physical index basis drifted"))?;
        if !covered
            || device_index.device_ptr() == source_ptr
            || !destinations.insert(device_index.device_ptr())
        {
            return Err(decline(
                "indexed append proof lost complete distinct cache coverage",
            ));
        }
        let columns = resident_key_constraints::resident_columns(engine, table, open, binding)?;
        if columns.len() != logical_binding.descriptor_count {
            return Err(decline(
                "indexed append proof input descriptor count drifted",
            ));
        }
        requests.push(CudaResidentTypedIndexInsert {
            index: Arc::clone(&device_index),
            table_mask: entry.table_mask,
            hash_shift: entry.hash_shift,
            columns,
        });
        physical.push(PhysicalIndexBinding {
            cache_key,
            key_id: logical_binding.key_id,
            device_index,
            published_row_count: Arc::clone(&entry.published_row_count),
            published_has_postings: Arc::clone(&entry.published_has_postings),
            has_postings_at_prepare: entry.published_has_postings.load(Ordering::Acquire),
            source_ptr,
            base_row,
            end_row,
            capacity: open.capacity,
            table_mask: entry.table_mask,
            hash_shift: entry.hash_shift,
            gc_boundary: entry.gc_boundary,
            allocated_bytes: horizon.allocated_bytes,
        });
    }
    drop(publications);
    drop(complete);
    drop(coverage);
    drop(cache);
    drop(route_publish);
    Ok((requests, physical))
}

struct IndexHorizon {
    table_mask: u32,
    hash_shift: u32,
    allocated_bytes: u64,
}

fn expected_index_horizon(open: &RelationalResidentShard) -> Result<IndexHorizon, ExecuteError> {
    let rows = u64::try_from(open.row_count)
        .map_err(|_| decline("indexed append proof row count overflows"))?;
    let capacity = u64::try_from(open.capacity)
        .map_err(|_| decline("indexed append proof capacity overflows"))?;
    let table_size = crate::engine_residency::resident_shard_index_table_size(rows, capacity)
        .ok_or_else(|| decline("indexed append proof has no index horizon geometry"))?;
    let table_mask = (table_size - 1) as u32;
    let allocated_bytes = resident_index_allocated_bytes(table_mask, capacity.max(rows))
        .ok_or_else(|| decline("indexed append proof index allocation geometry overflows"))?;
    Ok(IndexHorizon {
        table_mask,
        hash_shift: 32 - table_size.trailing_zeros(),
        allocated_bytes,
    })
}

fn resource_ledger(
    logical: &PreparedIndexDeltaLogicalBindings,
    physical: &[PhysicalIndexBinding],
) -> Result<IndexDeltaResourceLedger, ExecuteError> {
    let descriptor_count = logical
        .physical
        .iter()
        .try_fold(0_usize, |total, binding| {
            total.checked_add(binding.descriptor_count)
        })
        .ok_or_else(|| decline("indexed append proof ledger descriptor count overflows"))?;
    let descriptor_words = logical
        .physical
        .len()
        .checked_mul(3)
        .and_then(|words| words.checked_add(descriptor_count.checked_mul(4)?))
        .ok_or_else(|| decline("indexed append proof ledger descriptor words overflow"))?;
    let descriptor_bytes = u64::try_from(descriptor_words)
        .ok()
        .and_then(|words| words.checked_mul(std::mem::size_of::<u64>() as u64))
        .ok_or_else(|| decline("indexed append proof ledger descriptor bytes overflow"))?;
    let pinned_persistent_index_bytes = physical.iter().try_fold(0_u64, |total, binding| {
        total
            .checked_add(binding.allocated_bytes)
            .ok_or_else(|| decline("indexed append proof ledger persistent bytes overflow"))
    })?;
    let allocation_pin_count = physical
        .len()
        .checked_add(1)
        .ok_or_else(|| decline("indexed append proof ledger pin count overflows"))?;
    Ok(IndexDeltaResourceLedger {
        pinned_persistent_index_bytes,
        transient_preparation_bytes: logical.preparation_bytes,
        descriptor_bytes,
        descriptor_count,
        bounded_readback_bytes: std::mem::size_of::<u32>() as u64,
        allocation_pin_count,
        raw_index_count: logical.raw.len(),
        physical_index_count: logical.physical.len(),
    })
}

fn decline(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Serialization(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indexed_resident_engine() -> Option<crate::Engine> {
        let mut engine = crate::Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return None;
        }
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);
        engine
            .execute_text(1, "CREATE TABLE inert_index_delta_other (value int4)")
            .unwrap();
        engine
            .execute_text(
                2,
                "CREATE TABLE inert_index_delta (id int4 PRIMARY KEY, shared int4, code int4 UNIQUE)",
            )
            .unwrap();
        engine
            .execute_text(
                3,
                "CREATE INDEX inert_index_delta_shared_a ON inert_index_delta (shared)",
            )
            .unwrap();
        engine
            .execute_text(
                4,
                "CREATE INDEX inert_index_delta_shared_b ON inert_index_delta (shared)",
            )
            .unwrap();
        engine
            .execute_text(
                5,
                "CREATE INDEX inert_index_delta_compound ON inert_index_delta (shared, code)",
            )
            .unwrap();
        engine
            .execute_text(6, "INSERT INTO inert_index_delta VALUES (1, 10, 100)")
            .unwrap();
        engine
            .populate_relational_residency_snapshot("inert_index_delta")
            .unwrap();
        engine
            .publish_relational_resident_indexes("inert_index_delta")
            .unwrap();
        Some(engine)
    }

    fn indexed_proof_plan(
        engine: &crate::Engine,
    ) -> crate::engine_insert_plan::PreparedDeviceInsertPlan {
        let catalog = engine.catalog_snapshot();
        let command =
            gpu_db_sql::parse_command("INSERT INTO inert_index_delta VALUES (2, 20, 200)").unwrap();
        let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
            &command,
            &catalog,
            catalog.commit_seq,
        )
        .unwrap()
        .expect("indexed proof-only builder remains eligible");
        crate::engine_insert_plan::PreparedDeviceInsertPlan::from_typed_batch(
            batch, engine, &catalog,
        )
        .unwrap()
    }

    #[test]
    fn indexed_in_place_preparation_is_inert_and_pins_the_open_generation() {
        let Some(engine) = indexed_resident_engine() else {
            return;
        };
        let catalog = engine.catalog_snapshot();
        let plan = indexed_proof_plan(&engine);
        let epoch = engine
            .read_state
            .residency
            .point_index_mutation_epoch("inert_index_delta");
        let epoch_before = epoch.load(Ordering::Acquire);
        let wal_before = engine.durable_wal_records().len();
        let boundary_before = engine.committed_seq();
        let allocator_before = engine.read_state.mvcc.current_row_id();
        let (shard_id, source_ptr, source_header) = {
            let shards = engine.read_state.residency.shards.load_full();
            let open = shards
                .get("inert_index_delta")
                .and_then(|shards| shards.last())
                .unwrap();
            let payload = open.device_memory.as_ref().unwrap();
            (
                open.shard_id,
                payload.device_ptr(),
                payload.read_resident_i32_column(0, 2).unwrap(),
            )
        };
        let key_ids = catalog
            .relational_catalog
            .get("inert_index_delta")
            .unwrap()
            .indexes
            .iter()
            .enumerate()
            .map(|(ordinal, index)| {
                crate::engine_residency::index_probe_key_id(
                    catalog.relational_catalog.get("inert_index_delta").unwrap(),
                    index,
                    ordinal,
                )
                .unwrap()
            })
            .collect::<BTreeSet<_>>();
        let primary_key_id = crate::engine_residency::index_probe_key_id(
            catalog.relational_catalog.get("inert_index_delta").unwrap(),
            &catalog
                .relational_catalog
                .get("inert_index_delta")
                .unwrap()
                .indexes[0],
            0,
        )
        .unwrap();
        let (primary_index, primary_table_mask, primary_hash_shift, primary_row_count) =
            super::super::capacity_payload_tests::resident_named_index_cache_entry(
                &engine,
                "inert_index_delta",
                shard_id,
                primary_key_id,
            );
        assert_eq!(
            super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                &primary_index,
                primary_table_mask,
                primary_hash_shift,
                primary_row_count,
                1,
            ),
            1
        );
        assert_eq!(
            super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                &primary_index,
                primary_table_mask,
                primary_hash_shift,
                primary_row_count,
                2,
            ),
            0
        );
        let publication_before = engine
            .read_state
            .residency
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let proposal = plan.prepare_row_id_proposal(allocator_before).unwrap();
        let report = plan
            .inspect_current_resident_index_delta(&engine, proposal, |report| {
                let cache = engine
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let coverage = engine
                    .read_state
                    .residency
                    .named_index_coverage
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                for key_id in &key_ids {
                    let key = ("inert_index_delta".to_string(), shard_id, *key_id);
                    let entry = cache.get(&key).unwrap();
                    assert_eq!(entry.resident_device_ptr, source_ptr);
                    assert_eq!(entry.row_count, 1);
                    assert_eq!(entry.published_row_count.load(Ordering::Acquire), 1);
                    assert_eq!(coverage.get(&key), Some(&(source_ptr, 1)));
                }
                drop(coverage);
                drop(cache);
                let shards = engine.read_state.residency.shards.load_full();
                let payload = shards
                    .get("inert_index_delta")
                    .and_then(|shards| shards.last())
                    .and_then(|shard| shard.device_memory.as_ref())
                    .unwrap();
                assert_eq!(payload.device_ptr(), source_ptr);
                assert_eq!(
                    payload.read_resident_i32_column(0, 2).unwrap(),
                    source_header
                );
                assert_eq!(
                    super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                        &primary_index,
                        primary_table_mask,
                        primary_hash_shift,
                        primary_row_count,
                        1,
                    ),
                    1
                );
                assert_eq!(
                    super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                        &primary_index,
                        primary_table_mask,
                        primary_hash_shift,
                        primary_row_count,
                        2,
                    ),
                    0
                );
                engine
                    .read_state
                    .residency
                    .purge_shard_pk_index_for_table("inert_index_delta");
                assert!(engine
                    .read_state
                    .residency
                    .shard_pk_device_index
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .keys()
                    .any(|(table, _, _)| table == "inert_index_delta"));
                report
            })
            .unwrap();
        assert_eq!(report.raw_index_count, 5);
        assert_eq!(report.physical_index_count, 4);
        assert_eq!(report.base_row, 1);
        assert_eq!(report.incoming_rows, 1);
        assert_eq!(report.preparation_bytes, 512);
        assert_eq!(report.bounded_readback_bytes, 4);
        assert_eq!(report.allocation_pin_count, 5);
        assert!(report.pinned_persistent_index_bytes > 0);
        assert_eq!(report.descriptor_bytes, 256);
        assert_eq!(report.descriptor_count, 5);
        assert_eq!(epoch.load(Ordering::Acquire), epoch_before);
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.committed_seq(), boundary_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), allocator_before);
        // The cache is now retired, but this pre-proof Arc remains a live device pin.  It must
        // still answer the old key and not the merely proposed key; no write launch occurred.
        assert_eq!(
            super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                &primary_index,
                primary_table_mask,
                primary_hash_shift,
                primary_row_count,
                1,
            ),
            1
        );
        assert_eq!(
            super::super::capacity_payload_tests::resident_named_index_physical_hit_count(
                &primary_index,
                primary_table_mask,
                primary_hash_shift,
                primary_row_count,
                2,
            ),
            0
        );
        assert_eq!(
            *engine
                .read_state
                .residency
                .named_index_publications
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            publication_before
        );
        assert!(!engine
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .any(|(table, _, _)| table == "inert_index_delta"));
        assert!(!engine
            .read_state
            .residency
            .named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .any(|(table, _, _)| table == "inert_index_delta"));
    }

    #[test]
    fn indexed_preparation_declines_missing_stale_aliased_and_incomplete_cache_evidence() {
        for sabotage in [
            "missing",
            "stale_geometry",
            "short_published_geometry",
            "stale_coverage",
            "destination_alias",
            "source_arc_pin_mismatch",
            "horizon_metadata_mismatch",
            "shared_posting_verdict_mismatch",
            "incomplete",
        ] {
            let Some(engine) = indexed_resident_engine() else {
                return;
            };
            let catalog = engine.catalog_snapshot();
            let table = catalog.relational_catalog.get("inert_index_delta").unwrap();
            let key_id =
                crate::engine_residency::index_probe_key_id(table, &table.indexes[0], 0).unwrap();
            let distinct_key_id =
                crate::engine_residency::index_probe_key_id(table, &table.indexes[1], 1).unwrap();
            let shard_id = engine
                .read_state
                .residency
                .shards
                .load_full()
                .get("inert_index_delta")
                .and_then(|shards| shards.last())
                .unwrap()
                .shard_id;
            let cache_key = ("inert_index_delta".to_string(), shard_id, key_id);
            match sabotage {
                "missing" => {
                    engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .remove(&cache_key);
                }
                "stale_geometry" => {
                    engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .get_mut(&cache_key)
                        .unwrap()
                        .row_count = 0;
                }
                "short_published_geometry" => {
                    engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .get(&cache_key)
                        .unwrap()
                        .published_row_count
                        .store(0, Ordering::Release);
                }
                "stale_coverage" => {
                    engine
                        .read_state
                        .residency
                        .named_index_coverage
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .insert(cache_key.clone(), (0, 0));
                }
                "destination_alias" => {
                    let distinct_cache_key =
                        ("inert_index_delta".to_string(), shard_id, distinct_key_id);
                    let mut cache = engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let aliased_index = cache
                        .get(&cache_key)
                        .and_then(|entry| entry.device_index.as_ref())
                        .cloned()
                        .unwrap();
                    cache.get_mut(&distinct_cache_key).unwrap().device_index = Some(aliased_index);
                }
                "source_arc_pin_mismatch" => {
                    let mut cache = engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let different_allocation = cache
                        .get(&cache_key)
                        .and_then(|entry| entry.device_index.as_ref())
                        .cloned()
                        .unwrap();
                    cache.get_mut(&cache_key).unwrap()._resident_guard = different_allocation;
                }
                "horizon_metadata_mismatch" => {
                    let mut cache = engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let entry = cache.get_mut(&cache_key).unwrap();
                    entry.table_mask ^= 1;
                    assert_ne!(entry.table_mask, 0);
                }
                "shared_posting_verdict_mismatch" => {
                    let cache = engine
                        .read_state
                        .residency
                        .shard_pk_device_index
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let entry = cache.get(&cache_key).unwrap();
                    entry
                        .published_has_postings
                        .store(!entry.has_postings, Ordering::Release);
                }
                "incomplete" => {
                    engine
                        .read_state
                        .residency
                        .named_index_coverage
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .remove(&cache_key);
                }
                _ => unreachable!(),
            }
            let plan = indexed_proof_plan(&engine);
            let proposal = plan
                .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
                .unwrap();
            let wal_before = engine.durable_wal_records().len();
            let boundary_before = engine.committed_seq();
            assert!(matches!(
                plan.inspect_current_resident_index_delta(&engine, proposal, |_| ()),
                Err(ExecuteError::Serialization(_))
            ));
            assert_eq!(engine.durable_wal_records().len(), wal_before, "{sabotage}");
            assert_eq!(engine.committed_seq(), boundary_before, "{sabotage}");
        }
    }

    #[test]
    fn indexed_preparation_declines_when_exact_pooled_bytes_exceed_remaining_budget() {
        let Some(mut engine) = indexed_resident_engine() else {
            return;
        };
        let plan = indexed_proof_plan(&engine);
        let gpu_id = engine
            .read_state
            .residency
            .shards
            .load_full()
            .get("inert_index_delta")
            .and_then(|shards| shards.last())
            .unwrap()
            .gpu_id;
        let resident = engine.relational_resident_bytes_for_gpu(gpu_id);
        // The exact fanout geometry is a 512-byte two-lease pool reservation on this shape.
        // Keep validation's small source/verdict allocation available, then refuse the append
        // core's sidecar(0)+prepared-token(512) reservation before the token can allocate.
        engine.set_relational_residency_budget_bytes(gpu_id, resident + 511);
        let proposal = plan
            .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        let boundary_before = engine.committed_seq();
        assert!(matches!(
            plan.inspect_current_resident_index_delta(&engine, proposal, |_| ()),
            Err(ExecuteError::Serialization(_))
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.committed_seq(), boundary_before);
    }

    #[test]
    fn indexed_preparation_revalidates_target_binding_at_an_advanced_predecessor_boundary() {
        let Some(engine) = indexed_resident_engine() else {
            return;
        };
        let original_read_snapshot = engine.committed_seq();
        let plan = indexed_proof_plan(&engine);
        // Advance the global catalog/commit sequence through unrelated DML only.  The target
        // relation, index enrollment, and resident generation remain unchanged.
        engine
            .execute_text(77, "INSERT INTO inert_index_delta_other VALUES (7)")
            .unwrap();
        let predecessor_boundary = engine.committed_seq();
        assert!(predecessor_boundary > original_read_snapshot);
        let proposal = plan
            .prepare_row_id_proposal(engine.read_state.mvcc.current_row_id())
            .unwrap();
        let report = plan
            .inspect_current_resident_index_delta(&engine, proposal, |report| report)
            .unwrap();
        assert_eq!(report.original_read_snapshot, original_read_snapshot);
        assert_eq!(report.predecessor_boundary, predecessor_boundary);
        assert!(report.original_read_snapshot < report.predecessor_boundary);
        assert_eq!(engine.committed_seq(), predecessor_boundary);
    }

    #[test]
    fn inert_owner_exposes_only_scalar_inspection() {
        let source = include_str!("index_delta.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production proof owner precedes tests");
        assert!(source.contains("struct PreparedIndexedInPlaceProof"));
        assert!(source.contains("struct PreparedResidentIndexDelta"));
        assert!(source.contains("launch: PreparedResidentTypedIndexesInsert"));
        assert!(source.contains("index_delta: PreparedResidentIndexDelta"));
        assert!(source.contains("append: ResidentOpenShardAppendPlan"));
        assert!(source.contains("key_proof: BatchKeyConstraintProof"));
        assert!(source.contains("validation: ResidentKeyValidationSeal"));
        assert!(source.contains("struct IndexDeltaResourceLedger"));
        assert!(source.contains("mutation_epoch_expected_even"));
        assert!(source.contains("_named_index_lifecycle"));
        assert!(!source.contains("begin_point_index_mutation"));
        let wrapper = source
            .split("struct PreparedIndexedInPlaceProof")
            .nth(1)
            .and_then(|section| {
                section
                    .split("\n}\n\nimpl PreparedIndexedInPlaceProof")
                    .next()
            })
            .expect("wrapper declaration");
        assert!(
            wrapper.find("index_delta:") < wrapper.find("append:")
                && wrapper.find("append:") < wrapper.find("_named_index_lifecycle:"),
            "drop order must remain index delta -> append -> lifecycle"
        );
        let owner = source
            .split("impl PreparedIndexedInPlaceProof")
            .nth(1)
            .and_then(|section| section.split("\n/// Assemble").next())
            .expect("inert owner inspection impl");
        for forbidden in ["submit", "apply", "into_parts", "wal", "status", "poison"] {
            assert!(
                !owner.contains(forbidden),
                "inert owner must not expose {forbidden}"
            );
        }
        assert!(owner.contains("fn inspect"));
    }
}
