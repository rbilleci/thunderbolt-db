//! Allocation-complete table/index publication capability for the future indexed INSERT handoff.
//!
//! The capability is production-compiled so its ownership, lock order, and terminal behavior are
//! the surface a live `DeviceInsertPlan` will adopt.  It intentionally has no production
//! constructor: test-only pre-WAL constructors exercise the real residency authorities without
//! creating a second INSERT/WAL/apply path.

#![allow(dead_code)]

use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[cfg(test)]
use crate::engine_state::POINT_INDEX_MUTATION_POISON;
use crate::engine_state::{
    NamedIndexCoverage, PreparedPointIndexMutationGuard, ReadState, ResidencyReadState,
    TablePointSlot, TerminalNamedIndexPublicationGuard,
};
#[cfg(test)]
use crate::engine_transaction_reset::table_schema_digest;
use crate::relational_model::{RelationalIndex, RelationalTable};
use crate::resident_storage::{
    CachedCompoundI32I64PointRoute, CachedShardPkDeviceIndex, CachedShardedPointRoute,
    RelationalResidentShard, ResidentDeviceMemoryCell, ShardResidentDeviceMemoryMap,
};
use gpu_db_execution::CudaResidentDeviceMemory;

type ShardPkIndexCache = BTreeMap<(String, u32, usize), CachedShardPkDeviceIndex>;
type NamedIndexComplete = BTreeMap<String, (u32, Vec<RelationalIndex>)>;
type NamedIndexEnrollment = BTreeMap<u32, Vec<RelationalIndex>>;
type ShardCellMap = BTreeMap<(String, u32), ResidentDeviceMemoryCell>;

/// Borrowed roots protected by the publication locks. Grouping them keeps the currentness check
/// tied to one coherent lock-held view without allocating or cloning any map.
struct LockedIndexState<'a> {
    cache: &'a ShardPkIndexCache,
    coverage: &'a NamedIndexCoverage,
    complete: &'a NamedIndexComplete,
    enrollment: &'a NamedIndexEnrollment,
}

/// A static failure label keeps the post-preparation path allocation-free.  Before terminal arm,
/// callers may decline this inert proof; after arm, the owned guards fail closed on every error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PreparedTableIndexManifestError {
    CatalogOrSlotStale,
    EpochStale,
    DescriptorStale,
    SideCellsStale,
    IndexStateStale,
    PointEpochStartFailed,
}

/// Exact table-scoped point-index mutation owner.  Name, OID, table-slot Arc, separately retained
/// slot identity, and the selected even epoch are captured before the terminal boundary.  The
/// consuming publisher rechecks all of them while holding the route-publication lock.
struct PreparedTablePointMutation {
    table_name: Box<str>,
    table_oid: u32,
    /// Full catalog-shaped target retained before the durable cut.  Name/OID alone do not prove
    /// index enrollment, column layout, constraints, or schema identity survived an ALTER.
    table_shape: RelationalTable,
    _schema_digest: gpu_db_wal::CanonicalDigest,
    slot: Arc<TablePointSlot>,
    slot_identity: Arc<()>,
    epoch: Arc<std::sync::atomic::AtomicU64>,
    expected_even: u64,
    transition: PreparedPointIndexMutationGuard,
}

impl PreparedTablePointMutation {
    #[cfg(test)]
    fn prepare(
        residency: &ResidencyReadState,
        read_state: &ReadState,
        table: &RelationalTable,
    ) -> Result<Self, PreparedTableIndexManifestError> {
        let slot = residency
            .ensure_table_point_slot(read_state, table)
            .ok_or(PreparedTableIndexManifestError::CatalogOrSlotStale)?;
        let slot_identity = Arc::clone(&slot.slot_identity);
        let epoch = Arc::clone(&slot.index_epoch);
        let expected_even = epoch.load(Ordering::Acquire);
        if expected_even & 1 != 0 || expected_even == POINT_INDEX_MUTATION_POISON {
            return Err(PreparedTableIndexManifestError::EpochStale);
        }
        let schema_digest = table_schema_digest(table)
            .map_err(|_| PreparedTableIndexManifestError::CatalogOrSlotStale)?;
        if !Self::catalog_is_exact(read_state, table)
            || !residency.table_point_slot_is_current(&table.name, table.oid, &slot, &slot_identity)
        {
            return Err(PreparedTableIndexManifestError::CatalogOrSlotStale);
        }
        Ok(Self {
            table_name: Box::from(table.name.as_str()),
            table_oid: table.oid,
            table_shape: table.clone(),
            _schema_digest: schema_digest,
            slot,
            slot_identity,
            epoch: Arc::clone(&epoch),
            expected_even,
            transition: residency.prepare_exact_point_index_mutation(epoch, expected_even),
        })
    }

    fn catalog_is_exact(read_state: &ReadState, expected: &RelationalTable) -> bool {
        read_state
            .latest_catalog()
            .relational_catalog
            .get(&expected.name)
            .is_some_and(|current| current.oid == expected.oid && current == expected)
    }

    /// Caller holds the route-publication lock.  Epoch validation deliberately stays separate so
    /// the clean pre-WAL phase can require the captured even value while the terminal finalizer
    /// requires the exact writing-odd value after its successful CAS.
    fn target_is_current_under_publish_lock(
        &self,
        residency: &ResidencyReadState,
        read_state: &ReadState,
    ) -> bool {
        Self::catalog_is_exact(read_state, &self.table_shape)
            && residency.table_point_slot_is_current(
                &self.table_name,
                self.table_oid,
                &self.slot,
                &self.slot_identity,
            )
    }

    /// The caller has crossed the test-only terminal simulation.  An exact-CAS failure is a
    /// fail-closed event because `PreparedPointIndexMutationGuard` is already armed.
    fn start_terminal(&mut self) -> Result<(), PreparedTableIndexManifestError> {
        self.transition
            .start()
            .map_err(|()| PreparedTableIndexManifestError::PointEpochStartFailed)
    }

    fn arm_terminal(&mut self) {
        self.transition.arm_post_wal();
    }

    /// `start_terminal` succeeded on this owned transition, so final completion cannot be a
    /// recoverable branch after descriptor-last stores.  Any violated internal premise is a
    /// process bug and the guard's Drop remains fail-closed while unwinding.
    fn complete_terminal(self) {
        self.transition.complete_started();
    }

    fn writing_epoch(&self) -> u64 {
        self.expected_even
            .checked_add(1)
            .expect("prepared point-index mutation start proved its writing epoch fits")
    }
}

/// One prebuilt cell-map root.  The successor map is constructed before terminal arm; publication
/// only verifies the captured root and performs one ArcSwap store.
struct PreparedShardCellMap {
    expected: Arc<ShardCellMap>,
    successor: Arc<ShardCellMap>,
    cells: Box<[PreparedShardCellWitness]>,
}

/// Roots and witnesses displaced by one terminal side-cell-map swap.  They are retained until
/// every publication lock is released and lifecycle completion has made the new generation final.
struct RetiredShardCellMapRoots {
    published_root: Arc<ShardCellMap>,
    expected_root: Arc<ShardCellMap>,
    witnesses: Box<[PreparedShardCellWitness]>,
}

/// Root identity alone is not a sidecar currentness proof: a writer can publish a fresh payload
/// through an existing `SnapshotCell`.  Capture the exact cell Arc, generation, and payload Arc
/// (or None) for every entry before the terminal boundary.
struct PreparedShardCellWitness {
    key: (String, u32),
    cell: ResidentDeviceMemoryCell,
    generation: u64,
    payload: Option<Arc<CudaResidentDeviceMemory>>,
}

impl PreparedShardCellMap {
    #[cfg(test)]
    fn capture(map: &ShardResidentDeviceMemoryMap) -> Self {
        let expected = map.cells.load_full();
        let mut cells = Vec::with_capacity(expected.len());
        for (key, cell) in expected.iter() {
            let generation = cell.load();
            cells.push(PreparedShardCellWitness {
                key: key.clone(),
                cell: Arc::clone(cell),
                generation: generation.generation(),
                payload: generation.get().clone(),
            });
        }
        Self {
            successor: Arc::new((*expected).clone()),
            expected,
            cells: cells.into_boxed_slice(),
        }
    }

    fn is_current(&self, map: &ShardResidentDeviceMemoryMap) -> bool {
        Arc::ptr_eq(&self.expected, &map.cells.load_full())
            && self.cells.iter().all(|expected| {
                map.cells
                    .load()
                    .get(&expected.key)
                    .is_some_and(|current_cell| {
                        Arc::ptr_eq(current_cell, &expected.cell) && {
                            let current = current_cell.load();
                            current.generation() == expected.generation
                                && match (current.get(), &expected.payload) {
                                    (Some(current), Some(expected)) => {
                                        Arc::ptr_eq(current, expected)
                                    }
                                    (None, None) => true,
                                    _ => false,
                                }
                        }
                    })
            })
    }

    fn publish(self, map: &ShardResidentDeviceMemoryMap) -> RetiredShardCellMapRoots {
        let Self {
            expected,
            successor,
            cells,
        } = self;
        RetiredShardCellMapRoots {
            published_root: map.cells.swap(successor),
            expected_root: expected,
            witnesses: cells,
        }
    }
}

/// Four independent descriptor-adjacent cell maps travel as one manifest: payload, deleted-by,
/// created-by, and durable row-id.  Successor roots are deliberately retained independently so a
/// later rollover can replace any subset without an allocation on the terminal path.
struct PreparedShardCellPublications {
    payload: PreparedShardCellMap,
    deleted_by: PreparedShardCellMap,
    created_by: PreparedShardCellMap,
    row_id: PreparedShardCellMap,
}

/// All four side-cell roots displaced by a descriptor-last finalization.
struct RetiredShardCellPublications {
    payload: RetiredShardCellMapRoots,
    deleted_by: RetiredShardCellMapRoots,
    created_by: RetiredShardCellMapRoots,
    row_id: RetiredShardCellMapRoots,
}

impl PreparedShardCellPublications {
    #[cfg(test)]
    fn capture(residency: &ResidencyReadState) -> Self {
        Self {
            payload: PreparedShardCellMap::capture(&residency.shard_device_memory),
            deleted_by: PreparedShardCellMap::capture(&residency.shard_deleted_by_memory),
            created_by: PreparedShardCellMap::capture(&residency.shard_created_by_memory),
            row_id: PreparedShardCellMap::capture(&residency.shard_row_id_memory),
        }
    }

    fn is_current(&self, residency: &ResidencyReadState) -> bool {
        self.payload.is_current(&residency.shard_device_memory)
            && self
                .deleted_by
                .is_current(&residency.shard_deleted_by_memory)
            && self
                .created_by
                .is_current(&residency.shard_created_by_memory)
            && self.row_id.is_current(&residency.shard_row_id_memory)
    }

    fn publish(self, residency: &ResidencyReadState) -> RetiredShardCellPublications {
        RetiredShardCellPublications {
            payload: self.payload.publish(&residency.shard_device_memory),
            deleted_by: self.deleted_by.publish(&residency.shard_deleted_by_memory),
            created_by: self.created_by.publish(&residency.shard_created_by_memory),
            row_id: self.row_id.publish(&residency.shard_row_id_memory),
        }
    }
}

/// Unforgeable evidence that the durable cut has been crossed.  Production has no issuer until
/// the sole live INSERT carrier owns the canonical WAL record.
struct PreparedTableIndexManifestPostWalPermit(());

/// A branch-neutral, complete successor for all state an indexed table mutation must linearize.
/// There is deliberately no production builder or call site; the sole consumer below is the
/// intended live publication authority once `DeviceInsertPlan` owns the actual branch resources.
struct PreparedTableIndexManifest<'a> {
    point: PreparedTablePointMutation,
    expected_shards: Arc<BTreeMap<String, Vec<RelationalResidentShard>>>,
    successor_shards: Arc<BTreeMap<String, Vec<RelationalResidentShard>>>,
    side_cells: PreparedShardCellPublications,
    expected_index_cache: ShardPkIndexCache,
    successor_index_cache: ShardPkIndexCache,
    index_cache_tail_witnesses: Box<[PreparedIndexCacheTailWitness]>,
    expected_coverage: NamedIndexCoverage,
    successor_coverage: NamedIndexCoverage,
    expected_complete: NamedIndexComplete,
    successor_complete: NamedIndexComplete,
    expected_enrollment: NamedIndexEnrollment,
    successor_enrollment: NamedIndexEnrollment,
    /// Future conversion consumes the reservation's allocation owner.  Holding the existing
    /// budget serialization guard now prevents a map-finalizer preparation from racing a sidecar
    /// allocation/replacement before the live carrier takes over this exact lifetime.
    budget_guard: std::sync::MutexGuard<'a, ()>,
    /// Acquired before the budget guard and retained until point poisoning/completion has fenced
    /// every descriptor/side-cell authority.  Fields drop in declaration order, so it releases
    /// after `budget_guard` and before the lifecycle owner.
    mutation_guard: std::sync::MutexGuard<'a, ()>,
    /// Last on purpose: global acquisition is lifecycle → mutation → budget, and both success
    /// and abandonment release in the inverse order budget → mutation → lifecycle.
    lifecycle: TerminalNamedIndexPublicationGuard<'a>,
}

/// Atomic tail values must be frozen independently of their shared Arc identities.  A successor
/// cache record intentionally aliases the live counters; comparing two late loads through those
/// aliases would otherwise accept a tail mutation that happened after preparation.
struct PreparedIndexCacheTailWitness {
    key: (String, u32, usize),
    published_row_count: usize,
    published_has_postings: bool,
}

/// Terminal owner after the exact point epoch has transitioned even→odd.  Dropping this before a
/// physical-completion seal is fail-closed through the armed point/lifecycle guards.
struct ArmedTableIndexManifest<'a> {
    manifest: PreparedTableIndexManifest<'a>,
}

/// The only input accepted by the final map publisher.  A future branch conversion must consume
/// its real prepared fused-tail or fixed-header capability to create this seal while the epoch is
/// odd; the test-only mint below proves only finalizer ownership/currentness, not CUDA execution.
struct CompletedPhysicalTableIndexManifest<'a> {
    manifest: PreparedTableIndexManifest<'a>,
}

fn clone_index_cache_for_prepared_successor(current: &ShardPkIndexCache) -> ShardPkIndexCache {
    current
        .iter()
        .map(|(key, entry)| (key.clone(), entry.clone_for_prepared_successor()))
        .collect()
}

fn capture_index_cache_tail_witnesses(
    current: &ShardPkIndexCache,
) -> Box<[PreparedIndexCacheTailWitness]> {
    let mut witnesses = Vec::with_capacity(current.len());
    for (key, entry) in current {
        witnesses.push(PreparedIndexCacheTailWitness {
            key: key.clone(),
            published_row_count: entry.published_row_count.load(Ordering::Acquire),
            published_has_postings: entry.published_has_postings.load(Ordering::Acquire),
        });
    }
    witnesses.into_boxed_slice()
}

impl<'a> PreparedTableIndexManifest<'a> {
    /// Prebuild every persistent root while the table is still pre-terminal.  Locking follows the
    /// consuming order so the captured roots form one real residency authority, not a detached
    /// model.  Every allocation (map clone, retoken, and lifecycle protected-set) occurs here.
    #[cfg(test)]
    fn prepare_for_test(
        residency: &'a ResidencyReadState,
        read_state: &ReadState,
        table: &RelationalTable,
    ) -> Result<Self, PreparedTableIndexManifestError> {
        let point = PreparedTablePointMutation::prepare(residency, read_state, table)?;
        let mut protected_tables = BTreeSet::new();
        protected_tables.insert(table.name.clone());
        let lifecycle = residency
            .begin_transaction_named_index_publication(protected_tables)
            .into_terminal();
        let mutation_guard = residency
            .mutation_gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let budget_guard = residency
            .budget_allocation_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _descriptor = residency
            .descriptor_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _route = residency
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let index_cache = residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let coverage = residency
            .named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let complete = residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let enrollment = residency
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !point.target_is_current_under_publish_lock(residency, read_state) {
            return Err(PreparedTableIndexManifestError::CatalogOrSlotStale);
        }

        let expected_shards = residency.shards.load_full();
        let mut next_shards = (*expected_shards).clone();
        if let Some(shards) = next_shards.get_mut(table.name.as_str()) {
            let next_generation = Arc::new(());
            for shard in shards {
                shard.point_route_generation = Arc::clone(&next_generation);
            }
        }
        let side_cells = PreparedShardCellPublications::capture(residency);
        let expected_index_cache = clone_index_cache_for_prepared_successor(&index_cache);
        let successor_index_cache = clone_index_cache_for_prepared_successor(&index_cache);
        let index_cache_tail_witnesses = capture_index_cache_tail_witnesses(&index_cache);
        let expected_coverage = coverage.clone();
        let successor_coverage = coverage.clone();
        let expected_complete = complete.clone();
        let successor_complete = complete.clone();
        let expected_enrollment = enrollment.clone();
        let successor_enrollment = enrollment.clone();
        drop(enrollment);
        drop(complete);
        drop(coverage);
        drop(index_cache);
        drop(_route);
        drop(_descriptor);
        Ok(Self {
            point,
            successor_shards: Arc::new(next_shards),
            expected_shards,
            side_cells,
            expected_index_cache,
            successor_index_cache,
            index_cache_tail_witnesses,
            expected_coverage,
            successor_coverage,
            expected_complete,
            successor_complete,
            expected_enrollment,
            successor_enrollment,
            budget_guard,
            mutation_guard,
            lifecycle,
        })
    }

    #[cfg(test)]
    fn successor_roots_are_distinct(&self) -> bool {
        !Arc::ptr_eq(&self.expected_shards, &self.successor_shards)
            && !Arc::ptr_eq(
                &self.side_cells.payload.expected,
                &self.side_cells.payload.successor,
            )
            && !Arc::ptr_eq(
                &self.side_cells.created_by.expected,
                &self.side_cells.created_by.successor,
            )
            && !Arc::ptr_eq(
                &self.side_cells.row_id.expected,
                &self.side_cells.row_id.successor,
            )
    }

    /// Exact addresses of the preallocated ArcSwap roots.  The focused runtime proof compares
    /// these against the roots after finalization, demonstrating the consumer stored the owned
    /// successors rather than allocating/rebuilding a replacement map.
    #[cfg(test)]
    fn successor_authority_roots(&self) -> [usize; 5] {
        [
            Arc::as_ptr(&self.successor_shards) as usize,
            Arc::as_ptr(&self.side_cells.payload.successor) as usize,
            Arc::as_ptr(&self.side_cells.deleted_by.successor) as usize,
            Arc::as_ptr(&self.side_cells.created_by.successor) as usize,
            Arc::as_ptr(&self.side_cells.row_id.successor) as usize,
        ]
    }

    #[cfg(test)]
    fn prebuilt_geometry(&self) -> PreparedTableIndexManifestGeometry {
        PreparedTableIndexManifestGeometry {
            descriptor_entries: self.successor_shards.len(),
            payload_cell_entries: self.side_cells.payload.successor.len(),
            deleted_by_cell_entries: self.side_cells.deleted_by.successor.len(),
            created_by_cell_entries: self.side_cells.created_by.successor.len(),
            row_id_cell_entries: self.side_cells.row_id.successor.len(),
            cell_generation_witnesses: self.side_cells.payload.cells.len()
                + self.side_cells.deleted_by.cells.len()
                + self.side_cells.created_by.cells.len()
                + self.side_cells.row_id.cells.len(),
            index_cache_entries: self.successor_index_cache.len(),
            coverage_entries: self.successor_coverage.len(),
            completeness_entries: self.successor_complete.len(),
            enrollment_entries: self.successor_enrollment.len(),
            successor_arc_roots: 5,
        }
    }

    fn all_current_under_publish_locks(
        &self,
        residency: &ResidencyReadState,
        read_state: &ReadState,
        expected_epoch: u64,
        index_state: LockedIndexState<'_>,
    ) -> Result<(), PreparedTableIndexManifestError> {
        if !self
            .point
            .target_is_current_under_publish_lock(residency, read_state)
        {
            return Err(PreparedTableIndexManifestError::CatalogOrSlotStale);
        }
        if self.point.epoch.load(Ordering::Acquire) != expected_epoch {
            return Err(PreparedTableIndexManifestError::EpochStale);
        }
        if !Arc::ptr_eq(&self.expected_shards, &residency.shards.load_full()) {
            return Err(PreparedTableIndexManifestError::DescriptorStale);
        }
        if !self.side_cells.is_current(residency) {
            return Err(PreparedTableIndexManifestError::SideCellsStale);
        }
        if !same_index_cache(
            index_state.cache,
            &self.expected_index_cache,
            &self.index_cache_tail_witnesses,
        ) || index_state.coverage != &self.expected_coverage
            || index_state.complete != &self.expected_complete
            || index_state.enrollment != &self.expected_enrollment
        {
            return Err(PreparedTableIndexManifestError::IndexStateStale);
        }
        Ok(())
    }

    /// Clean pre-terminal validation.  It uses the same real locks/currentness checks as the
    /// consuming publisher but neither arms terminal guards nor mutates any authority.
    #[cfg(test)]
    fn validate_pre_wal(
        &self,
        residency: &ResidencyReadState,
        read_state: &ReadState,
    ) -> Result<(), PreparedTableIndexManifestError> {
        let _descriptor = residency
            .descriptor_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _route = residency
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let cache = residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let coverage = residency
            .named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let complete = residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let enrollment = residency
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.all_current_under_publish_locks(
            residency,
            read_state,
            self.point.expected_even,
            LockedIndexState {
                cache: &cache,
                coverage: &coverage,
                complete: &complete,
                enrollment: &enrollment,
            },
        )
    }

    fn arm_post_wal(
        mut self,
        _permit: PreparedTableIndexManifestPostWalPermit,
    ) -> Result<ArmedTableIndexManifest<'a>, PreparedTableIndexManifestError> {
        self.lifecycle.arm_post_wal();
        self.point.arm_terminal();
        self.point.start_terminal()?;
        Ok(ArmedTableIndexManifest { manifest: self })
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreparedTableIndexManifestGeometry {
    descriptor_entries: usize,
    payload_cell_entries: usize,
    deleted_by_cell_entries: usize,
    created_by_cell_entries: usize,
    row_id_cell_entries: usize,
    cell_generation_witnesses: usize,
    index_cache_entries: usize,
    coverage_entries: usize,
    completeness_entries: usize,
    enrollment_entries: usize,
    successor_arc_roots: usize,
}

impl<'a> ArmedTableIndexManifest<'a> {
    /// Test-only completion mint.  It cannot stand in for CUDA execution; it only lets focused
    /// tests exercise the production-compiled terminal map finalizer and its failure behavior.
    #[cfg(test)]
    fn complete_physical_for_test(self) -> CompletedPhysicalTableIndexManifest<'a> {
        CompletedPhysicalTableIndexManifest {
            manifest: self.manifest,
        }
    }
}

#[cfg(test)]
impl<'a> PreparedTableIndexManifest<'a> {
    fn arm_post_wal_for_test(
        self,
    ) -> Result<ArmedTableIndexManifest<'a>, PreparedTableIndexManifestError> {
        self.arm_post_wal(PreparedTableIndexManifestPostWalPermit(()))
    }
}

fn same_index_cache(
    current: &ShardPkIndexCache,
    expected: &ShardPkIndexCache,
    tails: &[PreparedIndexCacheTailWitness],
) -> bool {
    current.len() == expected.len()
        && tails.len() == expected.len()
        && current.iter().all(|(key, entry)| {
            expected.get(key).is_some_and(|expected_entry| {
                entry.resident_device_ptr == expected_entry.resident_device_ptr
                    && entry.row_count == expected_entry.row_count
                    && Arc::ptr_eq(
                        &entry.published_row_count,
                        &expected_entry.published_row_count,
                    )
                    && entry.gc_boundary == expected_entry.gc_boundary
                    && entry.duplicate_tolerant == expected_entry.duplicate_tolerant
                    && entry.has_postings == expected_entry.has_postings
                    && Arc::ptr_eq(
                        &entry.published_has_postings,
                        &expected_entry.published_has_postings,
                    )
                    && Arc::ptr_eq(&entry._resident_guard, &expected_entry._resident_guard)
                    && match (&entry.device_index, &expected_entry.device_index) {
                        (Some(current), Some(expected)) => Arc::ptr_eq(current, expected),
                        (None, None) => true,
                        _ => false,
                    }
                    && entry.table_mask == expected_entry.table_mask
                    && entry.hash_shift == expected_entry.hash_shift
            })
        })
        && tails.iter().all(|witness| {
            current.get(&witness.key).is_some_and(|entry| {
                entry.published_row_count.load(Ordering::Acquire) == witness.published_row_count
                    && entry.published_has_postings.load(Ordering::Acquire)
                        == witness.published_has_postings
            })
        })
}

/// Every authority displaced by descriptor-last publication.  The finalizer keeps this bundle
/// alive until all publication locks have been released and the exact lifecycle generation has
/// completed; an ArcSwap/store therefore never drops an old route, descriptor, or side-cell map
/// while the publication is still observed as in-flight.
struct RetiredTableIndexManifestRoots {
    sharded_route: Option<Arc<CachedShardedPointRoute>>,
    compound_route: Option<Arc<CachedCompoundI32I64PointRoute>>,
    side_cells: RetiredShardCellPublications,
    index_cache: ShardPkIndexCache,
    coverage: NamedIndexCoverage,
    complete: NamedIndexComplete,
    enrollment: NamedIndexEnrollment,
    descriptor: Arc<BTreeMap<String, Vec<RelationalResidentShard>>>,
}

impl ResidencyReadState {
    /// Sole manifest consumer.  It accepts only an already-prepared owner, obtains locks in the
    /// deliberate descriptor→route→index/coverage/completeness/enrollment order, and then
    /// performs only preowned stores/swaps and drops.  The caller already armed its terminal
    /// guards, so a mismatch here is intentionally fatal rather than a clean reprepare signal.
    fn publish_completed_table_index_manifest(
        &self,
        read_state: &ReadState,
        completed: CompletedPhysicalTableIndexManifest<'_>,
    ) -> Result<(), PreparedTableIndexManifestError> {
        let mut manifest = completed.manifest;
        manifest.lifecycle.enter_final_publication();

        let _descriptor = self
            .descriptor_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _route = self
            .sharded_point_route_publish_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut cache = self
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut coverage = self
            .named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut complete = self
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut enrollment = self
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // This check intentionally occurs *after* terminal arm.  It can return an error to the
        // caller for observability, but dropping this armed owner poisons both lifecycle and
        // point epoch; it is never a clean opportunity to rebuild/reprepare.
        manifest.all_current_under_publish_locks(
            self,
            read_state,
            manifest.point.writing_epoch(),
            LockedIndexState {
                cache: &cache,
                coverage: &coverage,
                complete: &complete,
                enrollment: &enrollment,
            },
        )?;

        // Route retirement shares the exact slot identity and lock held above.  Every successor
        // root was built before arm; no COW clone, insertion, key construction, or cache rebuild
        // occurs after this point.  Retain every returned old owner until locks and lifecycle
        // completion release the completed generation to external retirement.
        let PreparedTableIndexManifest {
            point,
            expected_shards: _,
            successor_shards,
            side_cells,
            expected_index_cache: _,
            successor_index_cache,
            index_cache_tail_witnesses: _,
            expected_coverage: _,
            successor_coverage,
            expected_complete: _,
            successor_complete,
            expected_enrollment: _,
            successor_enrollment,
            budget_guard,
            mutation_guard,
            lifecycle,
        } = manifest;
        let retired = RetiredTableIndexManifestRoots {
            sharded_route: point.slot.sharded_route.swap(None),
            compound_route: point.slot.compound_route.swap(None),
            side_cells: side_cells.publish(self),
            index_cache: std::mem::replace(&mut *cache, successor_index_cache),
            coverage: std::mem::replace(&mut *coverage, successor_coverage),
            complete: std::mem::replace(&mut *complete, successor_complete),
            enrollment: std::mem::replace(&mut *enrollment, successor_enrollment),
            // The descriptor is the last residency authority published before its point epoch
            // becomes visible as a completed even generation.
            descriptor: self.shards.swap(successor_shards),
        };
        point.complete_terminal();

        drop(enrollment);
        drop(complete);
        drop(coverage);
        drop(cache);
        drop(_route);
        drop(_descriptor);
        // Global acquisition was lifecycle → mutation → budget.  The terminal point guard above
        // is consumed while both local guards remain held; now release in strict reverse order.
        drop(budget_guard);
        drop(mutation_guard);
        lifecycle.complete_success();
        drop(retired);
        Ok(())
    }
}

/// Shared test-binary allocation gate.  There is one process-wide allocator, while each proof is
/// scoped to its calling thread so concurrent engine tests remain independent.
#[cfg(test)]
pub(crate) mod allocation_test_support {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    /// Test-binary allocator wrapper for the terminal finalizer gate.  It is thread-scoped so
    /// parallel engine tests and background runtime threads do not contaminate this proof.  The
    /// const TLS cell avoids a first-use heap allocation from the allocator itself.
    struct ThreadScopedCountingAllocator;

    std::thread_local! {
        static TERMINAL_FINALIZER_ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
    }

    fn record_terminal_finalizer_allocation() {
        let _ = TERMINAL_FINALIZER_ALLOCATIONS.try_with(|count| {
            if let Some(current) = count.get() {
                count.set(Some(current.wrapping_add(1)));
            }
        });
    }

    unsafe impl GlobalAlloc for ThreadScopedCountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_terminal_finalizer_allocation();
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record_terminal_finalizer_allocation();
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record_terminal_finalizer_allocation();
            unsafe { System.realloc(ptr, layout, new_size) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static THREAD_SCOPED_COUNTING_ALLOCATOR: ThreadScopedCountingAllocator =
        ThreadScopedCountingAllocator;

    struct ThreadAllocationScope;

    impl ThreadAllocationScope {
        fn begin() -> Self {
            TERMINAL_FINALIZER_ALLOCATIONS.with(|count| {
                assert!(
                    count.get().is_none(),
                    "thread allocation scopes may not nest"
                );
                count.set(Some(0));
            });
            Self
        }

        fn allocation_count(&self) -> usize {
            TERMINAL_FINALIZER_ALLOCATIONS
                .with(|count| count.get().expect("thread allocation scope remains active"))
        }
    }

    impl Drop for ThreadAllocationScope {
        fn drop(&mut self) {
            let _ = TERMINAL_FINALIZER_ALLOCATIONS.try_with(|count| count.set(None));
        }
    }

    pub(crate) fn assert_no_thread_allocations<T>(operation: impl FnOnce() -> T) -> T {
        let scope = ThreadAllocationScope::begin();
        let output = operation();
        assert_eq!(
            scope.allocation_count(),
            0,
            "the allocation-free terminal path allocated on its host thread"
        );
        output
    }
}

#[cfg(test)]
mod tests {
    use super::allocation_test_support::assert_no_thread_allocations as assert_terminal_finalizer_has_no_thread_allocations;
    use super::*;

    use gpu_db_snapshot::SnapshotCell;

    use crate::Engine;

    fn engine_and_table(name: &str) -> (Engine, RelationalTable) {
        let engine = Engine::new_local_test_engine();
        engine
            .execute_text(1, &format!("CREATE TABLE {name} (id int4 PRIMARY KEY)"))
            .expect("manifest fixture table");
        let table = engine
            .relational_catalog_table(name)
            .expect("manifest fixture table remains catalog-visible");
        (engine, table)
    }

    fn install_none_cell(map: &ShardResidentDeviceMemoryMap, table: &str, shard_id: u32) {
        let mut cells = BTreeMap::new();
        cells.insert(
            (table.to_owned(), shard_id),
            Arc::new(SnapshotCell::new(None)),
        );
        map.cells.store(Arc::new(cells));
    }

    fn seed_real_side_cell_authorities(engine: &Engine, table: &RelationalTable) {
        let residency = &engine.read_state.residency;
        install_none_cell(&residency.shard_device_memory, &table.name, 1);
        install_none_cell(&residency.shard_deleted_by_memory, &table.name, 1);
        install_none_cell(&residency.shard_created_by_memory, &table.name, 1);
        install_none_cell(&residency.shard_row_id_memory, &table.name, 1);
    }

    fn authority_roots(residency: &ResidencyReadState) -> [usize; 5] {
        [
            Arc::as_ptr(&residency.shards.load_full()) as usize,
            Arc::as_ptr(&residency.shard_device_memory.cells.load_full()) as usize,
            Arc::as_ptr(&residency.shard_deleted_by_memory.cells.load_full()) as usize,
            Arc::as_ptr(&residency.shard_created_by_memory.cells.load_full()) as usize,
            Arc::as_ptr(&residency.shard_row_id_memory.cells.load_full()) as usize,
        ]
    }

    /// Force the one-time ArcSwap/lock/lifecycle machinery before allocation measurement.  The
    /// measured owner below is independently prepared after this completed generation.
    fn warm_terminal_finalizer(engine: &Engine, table: &RelationalTable) {
        let residency = &engine.read_state.residency;
        let completed =
            PreparedTableIndexManifest::prepare_for_test(residency, &engine.read_state, table)
                .expect("warm terminal map finalizer prebuilds")
                .arm_post_wal_for_test()
                .expect("warm terminal map finalizer arms")
                .complete_physical_for_test();
        residency
            .publish_completed_table_index_manifest(&engine.read_state, completed)
            .expect("warm terminal map finalizer publishes");
    }

    /// Build a real retained compound route, but deliberately return no cloned route/plan owner.
    /// The slot's ArcSwap is therefore the route plan's last owner when the terminal manifest
    /// swaps it into its retirement bundle.
    fn prepare_gpu_compound_route_fixture(
        engine: &mut Engine,
        name: &str,
        first_transaction_id: u64,
    ) -> RelationalTable {
        engine
            .execute_text(
                first_transaction_id,
                &format!(
                    "CREATE TABLE {name} (tenant int4, account int8, balance int4, PRIMARY KEY (tenant, account))"
                ),
            )
            .expect("GPU compound-route fixture table");
        engine
            .execute_text(
                first_transaction_id
                    .checked_add(1)
                    .expect("fixture transaction ids remain distinct"),
                &format!("INSERT INTO {name} VALUES (1, 10, 100)"),
            )
            .expect("GPU compound-route fixture row");
        engine
            .populate_relational_residency_snapshot(name)
            .expect("GPU compound-route fixture residency");
        engine
            .publish_relational_resident_indexes(name)
            .expect("GPU compound-route fixture named indexes");
        let table = engine
            .relational_catalog_table(name)
            .expect("GPU compound-route fixture remains catalog-visible");
        warm_terminal_finalizer(engine, &table);
        let template = engine
            .prepare_relational_compound_i32_i64_point_read_template(
                "public",
                name,
                ["tenant", "account"],
                &["balance"],
            )
            .expect("GPU compound route prepares");
        drop(template);
        assert_eq!(
            engine.read_state.residency.compound_point_route_count(),
            1,
            "the fixture publishes exactly one compound route and leaks no test-side owner"
        );
        table
    }

    #[test]
    fn point_preparation_rejects_stale_name_and_oid_before_any_lifecycle_arm() {
        let (engine, table) = engine_and_table("manifest_stale_input");
        let mut wrong_name = table.clone();
        wrong_name.name = "manifest_stale_other".to_owned();
        assert!(matches!(
            PreparedTablePointMutation::prepare(
                &engine.read_state.residency,
                &engine.read_state,
                &wrong_name,
            ),
            Err(PreparedTableIndexManifestError::CatalogOrSlotStale)
        ));
        let mut wrong_oid = table.clone();
        wrong_oid.oid = wrong_oid
            .oid
            .checked_add(1)
            .expect("fixture OID increments");
        assert!(matches!(
            PreparedTablePointMutation::prepare(
                &engine.read_state.residency,
                &engine.read_state,
                &wrong_oid,
            ),
            Err(PreparedTableIndexManifestError::CatalogOrSlotStale)
        ));
    }

    #[test]
    fn stale_slot_identity_and_epoch_reject_pre_wal_without_partial_publication() {
        let (engine, table) = engine_and_table("manifest_stale_slot");
        seed_real_side_cell_authorities(&engine, &table);
        let manifest = PreparedTableIndexManifest::prepare_for_test(
            &engine.read_state.residency,
            &engine.read_state,
            &table,
        )
        .expect("current table prepares");
        let roots_before = authority_roots(&engine.read_state.residency);
        let slot = engine
            .read_state
            .residency
            .table_point_slot(&table.name, table.oid)
            .expect("preparation installs exact slot");
        let mut replacement = (*engine.read_state.residency.table_point_slots.load_full()).clone();
        replacement.insert(
            table.name.clone(),
            Arc::new(TablePointSlot::new(slot.table_oid)),
        );
        engine
            .read_state
            .residency
            .table_point_slots
            .store(Arc::new(replacement));
        assert_eq!(
            manifest.validate_pre_wal(&engine.read_state.residency, &engine.read_state),
            Err(PreparedTableIndexManifestError::CatalogOrSlotStale)
        );
        assert_eq!(authority_roots(&engine.read_state.residency), roots_before);
        drop(manifest);

        let manifest = PreparedTableIndexManifest::prepare_for_test(
            &engine.read_state.residency,
            &engine.read_state,
            &table,
        )
        .expect("replacement slot prepares");
        let epoch = engine
            .read_state
            .residency
            .point_index_mutation_epoch_for_table(&engine.read_state, &table)
            .expect("exact slot supplies epoch");
        epoch.fetch_add(2, Ordering::AcqRel);
        assert_eq!(
            manifest.validate_pre_wal(&engine.read_state.residency, &engine.read_state),
            Err(PreparedTableIndexManifestError::EpochStale)
        );
        assert_eq!(authority_roots(&engine.read_state.residency), roots_before);
        drop(manifest);
    }

    #[test]
    fn armed_owner_abandonment_before_physical_completion_wedges_without_map_changes() {
        let (engine, table) = engine_and_table("manifest_armed_abandonment");
        seed_real_side_cell_authorities(&engine, &table);
        let residency = &engine.read_state.residency;
        let roots_before = authority_roots(residency);
        let manifest =
            PreparedTableIndexManifest::prepare_for_test(residency, &engine.read_state, &table)
                .expect("terminal map finalizer prebuilds while current");
        let epoch = Arc::clone(&manifest.point.epoch);
        let armed = manifest
            .arm_post_wal_for_test()
            .expect("exact even epoch arms before physical work");
        drop(armed);
        assert_eq!(authority_roots(residency), roots_before);
        assert_eq!(
            epoch.load(Ordering::Acquire),
            POINT_INDEX_MUTATION_POISON,
            "an armed failure wedges the exact captured epoch"
        );
    }

    #[test]
    fn terminal_currentness_drift_after_physical_completion_wedges_instead_of_repreparing() {
        let (engine, table) = engine_and_table("manifest_terminal_stale");
        seed_real_side_cell_authorities(&engine, &table);
        let residency = &engine.read_state.residency;
        warm_terminal_finalizer(&engine, &table);
        let roots_before = authority_roots(residency);
        let manifest =
            PreparedTableIndexManifest::prepare_for_test(residency, &engine.read_state, &table)
                .expect("current rollover shape prepares");
        let epoch = Arc::clone(&manifest.point.epoch);
        let armed = manifest
            .arm_post_wal_for_test()
            .expect("terminal owner starts the exact writing epoch");
        epoch.fetch_add(2, Ordering::AcqRel);
        let completed = armed.complete_physical_for_test();
        assert_eq!(
            assert_terminal_finalizer_has_no_thread_allocations(|| {
                residency.publish_completed_table_index_manifest(&engine.read_state, completed)
            }),
            Err(PreparedTableIndexManifestError::EpochStale)
        );
        assert_eq!(authority_roots(residency), roots_before);
        assert_eq!(epoch.load(Ordering::Acquire), POINT_INDEX_MUTATION_POISON);
    }

    #[test]
    fn terminal_side_cell_generation_drift_wedges_without_replacing_any_root() {
        let (engine, table) = engine_and_table("manifest_terminal_side_cell");
        seed_real_side_cell_authorities(&engine, &table);
        let residency = &engine.read_state.residency;
        warm_terminal_finalizer(&engine, &table);
        let roots_before = authority_roots(residency);
        let manifest =
            PreparedTableIndexManifest::prepare_for_test(residency, &engine.read_state, &table)
                .expect("current terminal map finalizer prepares");
        let epoch = Arc::clone(&manifest.point.epoch);
        let armed = manifest
            .arm_post_wal_for_test()
            .expect("terminal owner arms before physical completion");
        let cell = residency
            .shard_created_by_memory
            .cells
            .load()
            .get(&(table.name.clone(), 1))
            .cloned()
            .expect("real created-by side-cell authority is seeded");
        cell.publish(None);
        let completed = armed.complete_physical_for_test();
        assert_eq!(
            assert_terminal_finalizer_has_no_thread_allocations(|| {
                residency.publish_completed_table_index_manifest(&engine.read_state, completed)
            }),
            Err(PreparedTableIndexManifestError::SideCellsStale)
        );
        assert_eq!(authority_roots(residency), roots_before);
        assert_eq!(epoch.load(Ordering::Acquire), POINT_INDEX_MUTATION_POISON);
    }

    #[test]
    fn two_prebuilt_terminal_map_finalizers_linearize_through_the_one_consumer() {
        let (engine, table) = engine_and_table("manifest_linearize");
        seed_real_side_cell_authorities(&engine, &table);
        let residency = &engine.read_state.residency;
        let before = authority_roots(residency);
        let first =
            PreparedTableIndexManifest::prepare_for_test(residency, &engine.read_state, &table)
                .expect("in-place shape prepares");
        assert!(first.successor_roots_are_distinct());
        let first_successor_roots = first.successor_authority_roots();
        let geometry = first.prebuilt_geometry();
        assert_eq!(geometry.successor_arc_roots, 5);
        assert_eq!(geometry.payload_cell_entries, 1);
        assert_eq!(geometry.deleted_by_cell_entries, 1);
        assert_eq!(geometry.created_by_cell_entries, 1);
        assert_eq!(geometry.row_id_cell_entries, 1);
        assert_eq!(geometry.cell_generation_witnesses, 4);
        let first = first
            .arm_post_wal_for_test()
            .expect("first terminal map finalizer arms")
            .complete_physical_for_test();
        residency
            .publish_completed_table_index_manifest(&engine.read_state, first)
            .expect("first finalizer uses the sole consumer");
        let after_in_place = authority_roots(residency);
        assert_eq!(
            after_in_place, first_successor_roots,
            "the consumer installed exactly the roots prebuilt before terminal arm"
        );
        assert_ne!(
            after_in_place, before,
            "consumer stored prebuilt successor roots"
        );

        let second =
            PreparedTableIndexManifest::prepare_for_test(residency, &engine.read_state, &table)
                .expect("second terminal map finalizer prepares after first publication");
        assert!(second.successor_roots_are_distinct());
        let second = second
            .arm_post_wal_for_test()
            .expect("second terminal map finalizer arms")
            .complete_physical_for_test();
        assert_terminal_finalizer_has_no_thread_allocations(|| {
            residency
                .publish_completed_table_index_manifest(&engine.read_state, second)
                .expect("second finalizer uses the same consumer");
        });
        assert_ne!(authority_roots(residency), after_in_place);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU-resident named-index cache"]
    fn terminal_cached_index_tail_drift_wedges_the_real_physical_cache_authority() {
        let mut engine = Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        assert!(
            hardware.driver_available && hardware.device_count != 0,
            "ignored cache-tail sabotage requires a local CUDA device"
        );
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);
        engine
            .execute_text(
                1,
                "CREATE TABLE manifest_tail (id int4 PRIMARY KEY, code int4 UNIQUE)",
            )
            .expect("GPU tail fixture table");
        engine
            .execute_text(2, "INSERT INTO manifest_tail VALUES (1, 10)")
            .expect("GPU tail fixture row");
        engine
            .populate_relational_residency_snapshot("manifest_tail")
            .expect("GPU tail fixture residency");
        engine
            .publish_relational_resident_indexes("manifest_tail")
            .expect("GPU tail fixture named indexes");
        let table = engine
            .relational_catalog_table("manifest_tail")
            .expect("GPU tail table remains current");
        let residency = &engine.read_state.residency;
        let roots_before = authority_roots(residency);
        let manifest =
            PreparedTableIndexManifest::prepare_for_test(residency, &engine.read_state, &table)
                .expect("real cache map prebuilds");
        let epoch = Arc::clone(&manifest.point.epoch);
        let armed = manifest
            .arm_post_wal_for_test()
            .expect("terminal owner arms before physical completion");
        {
            let cache = residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let (_, entry) = cache
                .iter()
                .find(|((cached_table, _, _), _)| cached_table == "manifest_tail")
                .expect("named-index publication retains a physical cache entry");
            entry.published_row_count.fetch_add(1, Ordering::AcqRel);
        }
        let completed = armed.complete_physical_for_test();
        assert_eq!(
            residency.publish_completed_table_index_manifest(&engine.read_state, completed),
            Err(PreparedTableIndexManifestError::IndexStateStale)
        );
        assert_eq!(authority_roots(residency), roots_before);
        assert_eq!(epoch.load(Ordering::Acquire), POINT_INDEX_MUTATION_POISON);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU-resident compound point route"]
    fn terminal_finalizer_retires_a_last_owner_compound_route_without_allocating() {
        let mut engine = Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        assert!(
            hardware.driver_available && hardware.device_count != 0,
            "ignored compound-route retirement proof requires a local CUDA device"
        );
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);

        let table = prepare_gpu_compound_route_fixture(&mut engine, "manifest_compound_success", 1);
        {
            let residency = &engine.read_state.residency;
            let charge_key = (0, table.name.clone());
            let charged_bytes = *residency
                .live_compound_point_route_bytes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&charge_key)
                .expect("the cached compound route owns a live-byte charge");
            let completed =
                PreparedTableIndexManifest::prepare_for_test(residency, &engine.read_state, &table)
                    .expect("manifest captures the populated compound-route slot")
                    .arm_post_wal_for_test()
                    .expect("manifest arms the exact point epoch")
                    .complete_physical_for_test();
            assert_terminal_finalizer_has_no_thread_allocations(|| {
                residency
                    .publish_completed_table_index_manifest(&engine.read_state, completed)
                    .expect("terminal publication retires the route as its last owner");
            });
            assert_eq!(
                residency.compound_point_route_count(),
                0,
                "the finalizer removed the exact table slot route"
            );
            assert!(
                residency
                    .live_compound_point_route_bytes
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .get(&charge_key)
                    .is_none(),
                "retiring the last route owner drains its exact live-byte charge ({charged_bytes} bytes)"
            );
        }

        // The terminal error path must remain allocation-free and leave a route owner live until
        // its explicit drain.  This mirrors a post-WAL validation failure: it wedges the manifest
        // rather than publishing a replacement or silently releasing a still-current route.
        let failed_table =
            prepare_gpu_compound_route_fixture(&mut engine, "manifest_compound_terminal_error", 3);
        let residency = &engine.read_state.residency;
        let failed_charge_key = (0, failed_table.name.clone());
        let failed_charge_bytes = *residency
            .live_compound_point_route_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&failed_charge_key)
            .expect("the terminal-error fixture owns a live-byte charge");
        let manifest = PreparedTableIndexManifest::prepare_for_test(
            residency,
            &engine.read_state,
            &failed_table,
        )
        .expect("terminal-error manifest captures the exact route slot");
        let epoch = Arc::clone(&manifest.point.epoch);
        let armed = manifest
            .arm_post_wal_for_test()
            .expect("terminal-error manifest arms the exact point epoch");
        epoch.fetch_add(2, Ordering::AcqRel);
        let completed = armed.complete_physical_for_test();
        assert_eq!(
            assert_terminal_finalizer_has_no_thread_allocations(|| {
                residency.publish_completed_table_index_manifest(&engine.read_state, completed)
            }),
            Err(PreparedTableIndexManifestError::EpochStale)
        );
        assert_eq!(epoch.load(Ordering::Acquire), POINT_INDEX_MUTATION_POISON);
        assert_eq!(
            residency.compound_point_route_count(),
            1,
            "failed terminal validation does not detach a still-current route owner"
        );
        assert_terminal_finalizer_has_no_thread_allocations(|| {
            residency.reset_point_routes_for_test();
        });
        assert!(
            residency
                .live_compound_point_route_bytes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&failed_charge_key)
                .is_none(),
            "the explicit terminal-error drain releases the last route charge ({failed_charge_bytes} bytes)"
        );
    }

    #[test]
    fn manifest_consumer_has_no_second_builder_or_allocating_publish_path() {
        let source = include_str!("prepared_table_index_manifest.rs");
        let consumer = source
            .split("fn publish_completed_table_index_manifest")
            .nth(1)
            .and_then(|section| section.split("#[cfg(test)]").next())
            .expect("sole consumer precedes tests");
        for forbidden in [
            "with_shards_mut_for_table",
            "clone(",
            ".insert(",
            "to_string(",
            "collect(",
            "Box<dyn Fn",
            "fn()",
            "DeviceInsertPlanKind::Indexed",
            "encode_",
            "purge_table_point_routes_under_publish_lock",
            "complete_terminal()?",
        ] {
            assert!(
                !consumer.contains(forbidden),
                "manifest consumer must remain one prebuilt authority: {forbidden}"
            );
        }
        assert_eq!(
            source
                .matches("\n    fn publish_completed_table_index_manifest")
                .count(),
            1,
            "no alternate manifest publisher may become authority"
        );
        let lifecycle = include_str!("../engine_state.rs");
        let final_phase = lifecycle
            .split("fn enter_transaction_named_index_final_publication")
            .nth(1)
            .and_then(|section| section.split("#[cfg(test)]").next())
            .expect("terminal lifecycle final phase");
        assert!(
            final_phase.contains("std::mem::take(&mut lifecycle.protected_tables)"),
            "the terminal lifecycle phase must move its sole protected table set"
        );
        assert!(
            !final_phase.contains("protected_tables.clone()")
                && !lifecycle.contains("prepared_final_publication_tables"),
            "the terminal lifecycle phase must not clone protected table names"
        );
        assert!(
            consumer.contains("point.slot.sharded_route.swap(None)")
                && consumer.contains("point.slot.compound_route.swap(None)"),
            "the exact prepared slot must return both old route owners to retirement"
        );
        assert!(
            consumer.contains("drop(budget_guard);")
                && consumer.contains("drop(mutation_guard);")
                && consumer.contains("lifecycle.complete_success();"),
            "terminal success must release budget then mutation before lifecycle"
        );
    }
}
