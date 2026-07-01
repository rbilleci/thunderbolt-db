//! Resident device-memory + MVCC version storage internals (P0 §9.6
//! decomposition, behavior-preserving): the per-table resident device-memory
//! cell map (ResidentDeviceMemoryMap, ShardResidentDeviceMemoryMap), the
//! MVCC version store (ColumnValueKey, TableVersionData, MvccData), the rows
//! view, and the resident cache + read-pin (TableRowsView, RelationalReadPin,
//! RelationalResidentCache + decision/observation/shard). Storage the
//! Engine owns and drives; not the commit-critical lock state.

use super::*;

/// ADR-009 R1: a cached GPU hash index over one resident int4 key column, for the index-probe
/// point-lookup route. Built lazily by DtoH-reading the key column from the resident device buffer
/// (NOT host rows) so the index's row→value mapping is inherently consistent with the SAME bytes the
/// scan reads — the audit's host_rows↔device_memory cross-generation hazard cannot arise. Validity:
/// `column_idx` + `resident_device_ptr` tag WHICH resident buffer (a unique allocation per residency
/// generation) this index mirrors. `_resident_guard` pins that buffer so its address can never be
/// freed-then-reused while cached — making `resident_device_ptr` an unambiguous identity check: a
/// re-admission allocates a NEW buffer (new ptr) → cache miss → rebuild against the live bytes.
/// `index_memory` is `None` when the buffer is NOT cleanly indexable for this column (duplicate keys
/// — incl. multiple NULLs materialized as 0 — since the scan returns EVERY match but a hash index
/// holds one row per key; or a key would exceed the kernel's probe cap; or the build failed), so the
/// route transparently falls back to the scan. The index `Arc<CudaResidentDeviceMemory>` is what a
/// submission pins (R1b) so the device index buffer outlives an in-flight kernel even if evicted.
#[derive(Debug)]
pub(crate) struct WaveResidentIndex {
    pub(crate) column_idx: usize,
    pub(crate) resident_device_ptr: u64,
    pub(crate) _resident_guard: Arc<CudaResidentDeviceMemory>,
    pub(crate) index_memory: Option<Arc<CudaResidentDeviceMemory>>,
    pub(crate) table_mask: u32,
    pub(crate) hash_shift: u32,
}

/// Per-table GPU-resident device memory, each table behind its own [`SnapshotCell`]
/// generation. A reader `get`s an owned `Arc` (a refcount bump, no borrow of the map)
/// so it pins the owner for its whole read; the serialized writer publishes a new
/// generation on (re)population and a `None` tombstone on invalidation instead of
/// freeing in place, so an in-flight reader's generation is never dropped under it
/// (P1-M3 slice B; doc 14). `Some` = resident, `None` = tombstoned (not resident).
///
/// The cell *map* itself is an [`arc_swap::ArcSwap`] so its structural mutations
/// (first-time `insert`, `remove`) are wait-free copy-on-write through `&self` — needed
/// now that this map lives inside the shared `Arc<ReadState>`, where even a holder of
/// `&mut Engine` only ever has `&self` access (lock-free read path, write-half Stage 4).
/// The COW clones only `Arc<SnapshotCell>` pointers, not the cells, and structural
/// changes happen solely on the serialized catalog-latch path (warm-up / DDL / drop), so
/// reads stay wait-free: a reader loads the map snapshot, finds its cell, and loads the
/// cell — never blocking and never contending with the publisher.
///
/// One table's resident device-memory cell: a [`SnapshotCell`] whose payload is `Some(owner)` when
/// resident and `None` when tombstoned, behind an `Arc` so the cell map's copy-on-write store only
/// clones pointers (not cells).
pub(crate) type ResidentDeviceMemoryCell = Arc<SnapshotCell<Option<Arc<CudaResidentDeviceMemory>>>>;

#[derive(Debug, Default)]
pub(crate) struct ResidentDeviceMemoryMap {
    pub(crate) cells: ArcSwap<BTreeMap<String, ResidentDeviceMemoryCell>>,
}

impl ResidentDeviceMemoryMap {
    /// Load the currently-published resident owner for `table`, if any (owned `Arc`).
    pub(crate) fn get(&self, table: &str) -> Option<Arc<CudaResidentDeviceMemory>> {
        self.cells
            .load()
            .get(table)
            .and_then(|cell| cell.load().get().clone())
    }

    /// Whether `table` currently has a published resident owner.
    pub(crate) fn contains_key(&self, table: &str) -> bool {
        self.cells
            .load()
            .get(table)
            .is_some_and(|cell| cell.load().get().is_some())
    }

    /// Number of tables with a published resident owner (tombstones excluded).
    /// Test-only accessor (residency counts are asserted in tests).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.cells
            .load()
            .values()
            .filter(|cell| cell.load().get().is_some())
            .count()
    }

    /// Publish a table's resident owner as a new generation (creating the cell on
    /// first residency); an in-flight reader keeps the generation it already loaded.
    /// `&self`: if the cell exists we publish on it directly (no map change); only a
    /// first-time residency COW-installs a new cell into the map.
    pub(crate) fn insert(&self, table: String, owner: CudaResidentDeviceMemory) {
        let owner = Some(Arc::new(owner));
        if let Some(cell) = self.cells.load().get(&table) {
            cell.publish(owner);
            return;
        }
        let mut next = (**self.cells.load()).clone();
        next.insert(table, Arc::new(SnapshotCell::new(owner)));
        self.cells.store(Arc::new(next));
    }

    /// Publish a `None` tombstone (invalidation): the cell is retained so in-flight
    /// readers keep their generation; new loads see "not resident". No-op if the table
    /// has no cell. `&self` (the cell publishes via `&self`) so the concurrent commit path can
    /// invalidate a mutated table's GPU residency without an engine write lock (write-half Stage 4).
    pub(crate) fn invalidate(&self, table: &str) {
        if let Some(cell) = self.cells.load().get(table) {
            cell.publish(None);
        }
    }

    /// Remove a table's cell entirely (DROP TABLE) via a COW store. In-flight readers
    /// retain their own loaded generation via its `Arc`, so this never frees memory under
    /// a reader.
    pub(crate) fn remove(&self, table: &str) {
        if !self.cells.load().contains_key(table) {
            return;
        }
        let mut next = (**self.cells.load()).clone();
        next.remove(table);
        self.cells.store(Arc::new(next));
    }
}

/// Per-shard resident device memory, keyed by `(table, shard_id)`, under the same
/// publish-don't-mutate discipline as `ResidentDeviceMemoryMap` (P1-M3 slice B; doc 14).
/// Before this, shards were a plain `BTreeMap<(String,u32), CudaResidentDeviceMemory>`
/// freed *in place* on invalidate/replace/drop while a reader borrowed `&owner` across a
/// kernel launch — a use-after-free the moment a writer overlaps a reader. Now each
/// shard is a `SnapshotCell<Option<Arc<…>>>`: readers `get()` an owned `Arc` (no map
/// borrow held across the launch), and the writer publishes a new generation / `None`
/// tombstone, so an in-flight reader's generation is freed only after it drains.
/// The cell map is an [`arc_swap::ArcSwap`] (same rationale as [`ResidentDeviceMemoryMap`]):
/// structural mutation (`install_table_shards`, `remove_table`) is wait-free copy-on-write
/// through `&self`, so it works from inside the shared `Arc<ReadState>`, while `get`/`invalidate`
/// stay wait-free.
#[derive(Debug, Default)]
pub(crate) struct ShardResidentDeviceMemoryMap {
    pub(crate) cells: ArcSwap<BTreeMap<(String, u32), ResidentDeviceMemoryCell>>,
}

impl ShardResidentDeviceMemoryMap {
    /// Load the published owner for one shard, if any (owned `Arc` — the borrow of the
    /// map ends here, so it is never held across a kernel launch).
    pub(crate) fn get(&self, key: &(String, u32)) -> Option<Arc<CudaResidentDeviceMemory>> {
        self.cells
            .load()
            .get(key)
            .and_then(|cell| cell.load().get().clone())
    }

    /// Whether this shard currently has a published owner (tombstones excluded).
    pub(crate) fn contains_key(&self, key: &(String, u32)) -> bool {
        self.cells
            .load()
            .get(key)
            .is_some_and(|cell| cell.load().get().is_some())
    }

    /// Published owners for every shard of `table` (tombstones excluded), owned `Arc`s.
    pub(crate) fn published_owners_for_table(
        &self,
        table: &str,
    ) -> Vec<Arc<CudaResidentDeviceMemory>> {
        self.cells
            .load()
            .iter()
            .filter(|((cell_table, _), _)| cell_table == table)
            .filter_map(|(_, cell)| cell.load().get().clone())
            .collect()
    }

    /// Replace a table's shards: publish each new shard as a new generation
    /// (creating the cell on first residency) and publish a `None` tombstone for any prior
    /// shard of this table not in the new set. In-flight readers keep the generation
    /// they already loaded. `&self`: existing cells are republished in place; only newly-keyed
    /// shards COW-install a cell, so the map is stored at most once per call.
    pub(crate) fn install_table_shards(
        &self,
        table: &str,
        device_memory: BTreeMap<u32, CudaResidentDeviceMemory>,
    ) {
        let snapshot = self.cells.load();
        let prior_ids: Vec<u32> = snapshot
            .keys()
            .filter(|(cell_table, _)| cell_table == table)
            .map(|(_, shard_id)| *shard_id)
            .collect();
        for shard_id in prior_ids {
            if !device_memory.contains_key(&shard_id) {
                if let Some(cell) = snapshot.get(&(table.to_string(), shard_id)) {
                    cell.publish(None);
                }
            }
        }
        let mut new_cells: BTreeMap<(String, u32), ResidentDeviceMemoryCell> = BTreeMap::new();
        for (shard_id, memory) in device_memory {
            let owner = Some(Arc::new(memory));
            let key = (table.to_string(), shard_id);
            if let Some(cell) = snapshot.get(&key) {
                cell.publish(owner);
            } else {
                new_cells.insert(key, Arc::new(SnapshotCell::new(owner)));
            }
        }
        if !new_cells.is_empty() {
            let mut next = (**snapshot).clone();
            next.extend(new_cells);
            self.cells.store(Arc::new(next));
        }
    }

    /// S-d2c: publish ONE shard's device memory (rollover's new open shard) WITHOUT tombstoning the table's
    /// other shards — a COW add (or in-place publish if the cell already exists). Unlike
    /// `install_table_shards`, leaves the existing sealed shards' cells untouched.
    pub(crate) fn insert_shard(&self, table: &str, shard_id: u32, memory: CudaResidentDeviceMemory) {
        let owner = Some(Arc::new(memory));
        let key = (table.to_string(), shard_id);
        if let Some(cell) = self.cells.load().get(&key) {
            cell.publish(owner);
            return;
        }
        let mut next = (**self.cells.load()).clone();
        next.insert(key, Arc::new(SnapshotCell::new(owner)));
        self.cells.store(Arc::new(next));
    }

    /// Publish a `None` tombstone for every shard of `table` (invalidation): cells are
    /// retained so in-flight readers keep their generation; new loads see "not resident". `&self`
    /// (cells publish via `&self`) for the concurrent commit path (write-half Stage 4).
    pub(crate) fn invalidate_table(&self, table: &str) {
        for ((cell_table, _), cell) in self.cells.load().iter() {
            if cell_table == table {
                cell.publish(None);
            }
        }
    }

    /// Remove every shard cell of `table` (DROP TABLE) via a COW store. In-flight readers
    /// retain their own loaded generation via its `Arc`, so this never frees memory under a reader.
    pub(crate) fn remove_table(&self, table: &str) {
        if !self
            .cells
            .load()
            .keys()
            .any(|(cell_table, _)| cell_table == table)
        {
            return;
        }
        let mut next = (**self.cells.load()).clone();
        next.retain(|(cell_table, _), _| cell_table != table);
        self.cells.store(Arc::new(next));
    }
}

/// One `(column, value)` slot of a table's equality value-index. The owning table is implied by
/// the per-table [`SnapshotCell`] the index lives in (write-half MVCC, Stage 3), so unlike the old
/// global `RelationalIndexKey` this carries no `table` field.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ColumnValueKey {
    pub(crate) column: String,
    pub(crate) value: String,
}

/// One table's publish-on-commit MVCC payload: its row version chains **and** its (now versioned)
/// equality value-index, held together so a reader that `load()`s one generation sees rows and
/// value-index that are mutually consistent at the same `commit_seq` (write-half MVCC, Stage 3,
/// design §3.1). The serialized writer publishes ONE new `Arc<TableVersionData>` for the mutated
/// table per commit; readers run `&self` against the immutable payload with no lock held, exactly
/// as the GPU residency `SnapshotCell`s already do.
///
/// `rows` is an [`InMemoryTupleStore`] holding ONLY this table's `rel/<table>/…` version chains
/// (the KV namespace lives in its own [`MvccData::kv`] partition). Tuple ids are still allocated
/// from one process-wide monotonic space ([`MvccData::next_tuple_id`]) so they remain globally
/// unique and byte-identical to the pre-partition single store — recovery / `all_versions` / prune
/// are unchanged. `value_index` is the per-table equality index the fast-path reads; it is
/// append-only under DML (stale entries are filtered out by row visibility + the final predicate
/// recheck, exactly as before), so a loaded generation's index is always a *superset* consistent
/// with that generation's rows.
#[derive(Debug, Clone, Default)]
pub(crate) struct TableVersionData {
    pub(crate) rows: InMemoryTupleStore,
    // Persistent immutable ordered map (`imbl::OrdMap`): O(1) clone (so `with_table_mut`'s per-commit
    // `TableVersionData::clone` no longer deep-copies every `Vec<String>`) and O(log n)
    // structurally-shared update. Each slot's row-key list is `Arc`-wrapped, so a clone shares lists
    // until one is mutated, at which point `Arc::make_mut` copies ONLY that list (copy-on-write).
    // Iteration stays in `ColumnValueKey` order, matching the old `BTreeMap`.
    pub(crate) value_index: imbl::OrdMap<ColumnValueKey, std::sync::Arc<Vec<String>>>,
}

impl TableVersionData {
    /// The row keys the value-index records for `(column, value)` (the equality fast-path lookup).
    /// Empty when the slot has no entries — matching the old `relational_value_index.get(...)`
    /// `.cloned().unwrap_or_default()`.
    pub(crate) fn index_keys(&self, column: &str, value: &str) -> Vec<String> {
        self.value_index
            .get(&ColumnValueKey {
                column: column.to_string(),
                value: value.to_string(),
            })
            .map(|keys| keys.as_ref().clone())
            .unwrap_or_default()
    }
}

/// The whole engine's versioned, publish-on-commit MVCC data: a per-table map of
/// [`SnapshotCell<Arc<TableVersionData>>`] (rows + value-index together) plus a single KV-namespace
/// partition, behind the residency `SnapshotCell` discipline — monotonic generation ids and epoch
/// reclamation, so an old `Arc<TableVersionData>` is freed only after its last reader handle drops
/// (write-half MVCC, Stage 3).
///
/// Reads `load()` a partition's current generation once and run `&self` against the immutable
/// payload. The serialized writer (still under the existing commit lock until Stage 4) mutates a
/// cheap clone of the relevant partition and `publish`es a fresh `Arc`; in-flight readers keep the
/// generation they already loaded. `next_row_id`/`next_tuple_id` are atomics so the writer advances
/// them through `&self` (and so a `prepare_*` can snapshot `next_row_id` off-lock in Stage 4).
#[derive(Debug)]
pub(crate) struct MvccData {
    /// `rel/<table>/…` row chains + the table's value-index, one published generation per table.
    ///
    /// Behind a `RwLock` for STRUCTURAL access only (write-half Stage 4): the per-cell publish is
    /// already `&self` (`SnapshotCell::publish`), so the lock is held only briefly to find a cell
    /// (read-lock, then `load()` clones an owned handle and the lock drops — never held across a read
    /// body or a kernel launch) or to INSERT a new cell on a table's first write (write-lock). This
    /// lets the concurrent commit critical section publish a mutated table's generation through
    /// `&self`, so lock-free readers run concurrently with a committer.
    pub(crate) tables: RwLock<BTreeMap<String, SnapshotCell<Arc<TableVersionData>>>>,
    /// The non-relational KV namespace (`SET`/`DELETE` keys), published as its own generation. Has
    /// no value-index (the equality fast-path is relational-only).
    pub(crate) kv: SnapshotCell<Arc<InMemoryTupleStore>>,
    /// Process-wide monotonic tuple-id allocator shared across ALL partitions, so tuple ids stay
    /// globally unique and identical to the pre-partition single `InMemoryTupleStore`.
    pub(crate) next_tuple_id: AtomicU64,
    /// The relational row-id allocator (was `relational_next_row_id`), now an atomic.
    pub(crate) next_row_id: AtomicU64,
}

impl Default for MvccData {
    fn default() -> Self {
        Self {
            tables: RwLock::new(BTreeMap::new()),
            kv: SnapshotCell::new(Arc::new(InMemoryTupleStore::new())),
            next_tuple_id: AtomicU64::new(1),
            next_row_id: AtomicU64::new(1),
        }
    }
}

impl MvccData {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Reserve the next globally-unique tuple id (serialized writer; `&self` for Stage 4 symmetry).
    pub(crate) fn reserve_tuple_id(&self) -> TupleId {
        self.next_tuple_id.fetch_add(1, AtomicOrdering::Relaxed)
    }

    /// The current relational row-id (the value a `prepare_*` snapshots as its insert base).
    pub(crate) fn current_row_id(&self) -> u64 {
        self.next_row_id.load(AtomicOrdering::Relaxed)
    }

    /// Advance the relational row-id allocator by `n` (serialized writer, on apply).
    pub(crate) fn advance_row_id(&self, n: u64) {
        self.next_row_id.fetch_add(n, AtomicOrdering::Relaxed);
    }

    /// Load the KV partition's currently-published generation (a refcount bump; the read body runs
    /// lock-free afterward and pins the generation until the handle drops).
    pub(crate) fn load_kv(&self) -> SnapshotHandle<Arc<InMemoryTupleStore>> {
        self.kv.load()
    }

    /// Lock the `tables` map for reading, recovering from poison (the only mutation under the lock is
    /// a `BTreeMap` insert of a fresh cell, which cannot leave the map torn, so continuing past a
    /// poisoned lock is safe — and one panicking committer must not wedge every reader).
    pub(crate) fn tables_read(
        &self,
    ) -> std::sync::RwLockReadGuard<'_, BTreeMap<String, SnapshotCell<Arc<TableVersionData>>>> {
        self.tables
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn tables_write(
        &self,
    ) -> std::sync::RwLockWriteGuard<'_, BTreeMap<String, SnapshotCell<Arc<TableVersionData>>>>
    {
        self.tables
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Load `table`'s currently-published generation, if the table has any MVCC data yet. Returns
    /// `None` before the table's first write (no cell published) — callers treat that as "empty
    /// table" exactly as a prefix scan over a store with no matching keys did. The map read-lock is
    /// held only to clone the cell's current `Arc` (`load()`), then released — the lock-free read
    /// body runs against the pinned handle with no lock held.
    pub(crate) fn load_table(&self, table: &str) -> Option<SnapshotHandle<Arc<TableVersionData>>> {
        let g = self.tables_read();
        g.get(table).map(|cell| cell.load())
    }

    /// A loaded, immutable view of `table`'s rows for the read path — an empty store when the table
    /// has no cell yet. Returned as an owned value (either the published `Arc`'s payload borrow via
    /// the handle, or a shared empty store) so callers resolve a `MvccReadQuery` against it.
    pub(crate) fn table_rows(&self, table: &str) -> TableRowsView {
        match self.load_table(table) {
            Some(handle) => TableRowsView::Resident(handle),
            None => TableRowsView::Empty(Arc::clone(&EMPTY_TABLE_VERSION_DATA)),
        }
    }

    /// Mutate a table's payload via copy-on-write and publish the new generation: clone the current
    /// `TableVersionData` (or start empty), run `mutate`, then `publish` a fresh `Arc`. In-flight
    /// readers keep the generation they already loaded (epoch reclamation).
    ///
    /// `&self` (write-half Stage 4): the COW mutate + the per-cell `publish` are `&self`; the map
    /// read-lock is held only to clone the existing cell's `Arc` and again to publish onto it, and a
    /// write-lock is taken only to insert a cell on a table's FIRST write. The CALLER (the commit
    /// critical section or a serialized DDL apply) provides the serialization that makes the
    /// read-clone → mutate → publish sequence atomic w.r.t. other writers; lock-free readers are
    /// never blocked by it.
    pub(crate) fn with_table_mut<R>(
        &self,
        table: &str,
        mutate: impl FnOnce(&mut TableVersionData) -> R,
    ) -> R {
        let mut data = match self.tables_read().get(table) {
            Some(cell) => TableVersionData::clone(cell.load().get()),
            None => TableVersionData::default(),
        };
        let result = mutate(&mut data);
        let data = Arc::new(data);
        // Re-resolve the cell: publish onto an existing one (read-lock), or insert a new one
        // (write-lock) on first write. Under the caller's serialization no other writer raced in.
        if let Some(cell) = self.tables_read().get(table) {
            cell.publish(data);
            return result;
        }
        let mut map = self.tables_write();
        match map.get(table) {
            // A concurrent first-writer beat us to creating the cell between the read-unlock and the
            // write-lock (does not happen under the caller's serialization, but is correct anyway):
            // publish our generation onto it.
            Some(cell) => {
                cell.publish(data);
            }
            None => {
                map.insert(table.to_string(), SnapshotCell::new(data));
            }
        }
        result
    }

    /// Mutate the KV partition via copy-on-write and publish the new generation. `&self` (the cell
    /// publishes via `&self`); the caller serializes (commit critical section / serialized DDL apply).
    pub(crate) fn with_kv_mut<R>(&self, mutate: impl FnOnce(&mut InMemoryTupleStore) -> R) -> R {
        let mut store = InMemoryTupleStore::clone(self.kv.load().get());
        let result = mutate(&mut store);
        self.kv.publish(Arc::new(store));
        result
    }

    /// Fetch the visible version of a KV-namespace `key` (test-only helper mirroring the old
    /// `mvcc_store.tuple_fetch_by_key` for KV keys).
    #[cfg(test)]
    pub(crate) fn kv_tuple_fetch_by_key(
        &self,
        key: &str,
        visibility: StorageVisibility,
    ) -> Result<Option<TupleVersion>, StorageError> {
        self.load_kv().get().tuple_fetch_by_key(key, visibility)
    }

    /// Reconstruct the whole-engine value-index keyed by `(table, column, value)` from every
    /// per-table generation (test-only — the production value-index is the per-table one).
    #[cfg(test)]
    pub(crate) fn value_index_snapshot(&self) -> BTreeMap<RelationalIndexKey, Vec<String>> {
        let mut index = BTreeMap::new();
        for (table, cell) in self.tables_read().iter() {
            for (key, row_keys) in cell.load().get().value_index.iter() {
                index.insert(
                    RelationalIndexKey {
                        table: table.clone(),
                        column: key.column.clone(),
                        value: key.value.clone(),
                    },
                    row_keys.as_ref().clone(),
                );
            }
        }
        index
    }

    /// Every version in every partition (KV + all tables), sorted by tuple id so the order matches
    /// the pre-partition single `BTreeMap<TupleId, …>` store — recovery / CUDA all-versions / the
    /// stamp-determinism tests depend on this order. Test-only (the production all-versions reads
    /// go through `resolve_mvcc_all_versions` on a single partition).
    #[cfg(test)]
    pub(crate) fn all_versions(&self) -> Vec<TupleVersion> {
        let mut versions = self.kv.load().get().all_versions();
        for cell in self.tables_read().values() {
            versions.extend(cell.load().get().rows.all_versions());
        }
        versions.sort_by_key(|version| version.tuple_id);
        versions
    }

    /// Total live version count across all partitions (test-only).
    #[cfg(test)]
    pub(crate) fn version_count(&self) -> usize {
        let mut count = self.kv.load().get().version_count();
        for cell in self.tables_read().values() {
            count += cell.load().get().rows.version_count();
        }
        count
    }

    /// Prune versions deleted at or before `safe_txn_id` across every partition, republishing each
    /// touched partition. Aggregates the per-partition [`PruneStats`].
    pub(crate) fn prune_versions_deleted_at_or_before(&self, safe_txn_id: TxnId) -> PruneStats {
        let mut total = PruneStats {
            removed_versions: 0,
            removed_tuples: 0,
            remaining_versions: 0,
        };
        let kv_stats =
            self.with_kv_mut(|store| store.prune_versions_deleted_at_or_before(safe_txn_id));
        total.removed_versions += kv_stats.removed_versions;
        total.removed_tuples += kv_stats.removed_tuples;
        total.remaining_versions += kv_stats.remaining_versions;
        let table_names: Vec<String> = self.tables_read().keys().cloned().collect();
        for table in table_names {
            let stats = self.with_table_mut(&table, |data| {
                data.rows.prune_versions_deleted_at_or_before(safe_txn_id)
            });
            total.removed_versions += stats.removed_versions;
            total.removed_tuples += stats.removed_tuples;
            total.remaining_versions += stats.remaining_versions;
        }
        total
    }
}

/// A shared, permanently-empty `TableVersionData` so [`MvccData::table_rows`] can hand out a view of
/// a not-yet-written table without allocating a new store per read.
pub(crate) static EMPTY_TABLE_VERSION_DATA: std::sync::LazyLock<Arc<TableVersionData>> =
    std::sync::LazyLock::new(|| Arc::new(TableVersionData::default()));

thread_local! {
    /// Set (RAII-scoped) while this thread runs an internal relational read from INSIDE the commit
    /// critical section (materialized-view create/refresh applying a committed entry). The deep read
    /// executor reads this to skip its leader re-check, which would otherwise re-lock the already-held
    /// commit_mutex and self-deadlock. False for every client read. See
    /// [`Engine::skip_leader_check_during_internal_read`].
    pub(crate) static MVCC_READ_SKIPS_LEADER_CHECK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A loaded, immutable view of one table's rows for the read path: either the pinned published
/// generation (the common case) or a shared empty payload for a table with no MVCC data yet.
pub(crate) enum TableRowsView {
    Resident(SnapshotHandle<Arc<TableVersionData>>),
    Empty(Arc<TableVersionData>),
}

impl TableRowsView {
    /// Borrow the underlying row store to resolve a `MvccReadQuery` against.
    pub(crate) fn store(&self) -> &InMemoryTupleStore {
        match self {
            TableRowsView::Resident(handle) => &handle.get().rows,
            TableRowsView::Empty(data) => &data.rows,
        }
    }

    /// Borrow the value-index of THIS pinned generation. Prereq #1 (Stage 4): a relational read
    /// pins ONE `TableVersionData` generation for the whole statement and reads BOTH its
    /// `value_index` (the equality fast-path) and its `rows` (resolution) from it, so a concurrent
    /// publish can never interleave the index of one generation with the rows of another.
    pub(crate) fn payload(&self) -> &TableVersionData {
        match self {
            TableRowsView::Resident(handle) => handle.get(),
            TableRowsView::Empty(data) => data,
        }
    }

    /// The value-index row keys for `(column, value)` in this pinned generation.
    pub(crate) fn index_keys(&self, column: &str, value: &str) -> Vec<String> {
        self.payload().index_keys(column, value)
    }
}

/// One pinned, statement-stable relational read snapshot (prereq #1, write-half Stage 4). Holds the
/// single visibility boundary (`committed_seq` read once at statement start) AND one pinned
/// generation of the read table (its rows + value-index together). Every part of a relational
/// statement — the equality fast-path value-index lookup, key resolution, ordering scans, the final
/// row materialization — reads from THIS one generation, so a concurrent committer that publishes a
/// new generation mid-statement can never make the read see the value-index of one `commit_seq` and
/// the rows of another (the exact hazard the audit flagged). The handle keeps its generation alive
/// (epoch reclamation) until the statement drops the pin.
pub(crate) struct RelationalReadPin {
    pub(crate) visibility: StorageVisibility,
    pub(crate) table_rows: TableRowsView,
}

impl RelationalReadPin {
    pub(crate) fn store(&self) -> &InMemoryTupleStore {
        self.table_rows.store()
    }

    pub(crate) fn index_keys(&self, column: &str, value: &str) -> Vec<String> {
        self.table_rows.index_keys(column, value)
    }
}

/// The serialized-path resident-route metadata kept on [`Engine`] (mutated only by the catalog-latch
/// path). The lock-free read path's parts all moved to `Engine::read_state`: the device-memory maps
/// (the tombstone gate) and the snapshot/shard metadata to [`ResidencyReadState`] (Stage 3 —
/// blocker #2), the route telemetry to [`RouteTelemetry`]. What remains here is purely the
/// admission-time accounting the read path never consults: per-GPU residency budgets and the last
/// admission decision per table.
#[derive(Debug, Default)]
pub(crate) struct RelationalResidentCache {
    pub(crate) budget_bytes_by_gpu: BTreeMap<u16, u64>,
    pub(crate) last_decisions: BTreeMap<String, RelationalResidentCacheDecision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RelationalResidentCacheDecision {
    pub(crate) table: String,
    pub(crate) gpu_id: u16,
    pub(crate) accepted: bool,
    pub(crate) reason: String,
    pub(crate) resident_bytes: u64,
    pub(crate) budget_bytes: Option<u64>,
    pub(crate) current_bytes_before: u64,
    pub(crate) current_bytes_after: u64,
    pub(crate) evicted_tables: Vec<String>,
}

pub(crate) struct RelationalResidentRouteExecutionObservation {
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_samples: u64,
    pub(crate) kernel_ms: u64,
    pub(crate) kernel_event_elapsed_us: Option<u64>,
    pub(crate) rows: usize,
    pub(crate) wall_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RelationalResidentShard {
    pub(crate) shard_id: u32,
    pub(crate) row_start: usize,
    pub(crate) row_count: usize,
    /// Capacity-padded column stride (S-d2): the OPEN shard carries headroom (`capacity > row_count`) for
    /// in-place appends; sealed/benchmark shards are dense (`capacity == row_count`). The sharded read's
    /// recompaction gather reads each column slice at this stride (a column's live rows sit at its
    /// capacity-strided start), mirroring the single buffer's `capacity` field (Slice 1b-i).
    pub(crate) capacity: usize,
    /// S-d2b: true iff this shard is an int4-only open shard with NO null bitmap / text / non-int4 sections
    /// — i.e. a committed INSERT's applied rows can be appended in place into its headroom (the same
    /// eligibility the single-buffer append checks). Set at admission from the shard's column layout; the
    /// append is declined (re-admit) when false. Benchmark/sealed shards are not append targets (false).
    pub(crate) int4_appendable: bool,
    /// S-d3: per-int4-column ZONE MAP (min/max over THIS shard's live rows, in int4-ordinal order). A point
    /// lookup / range query PRUNES shards whose `[min,max]` for the filter column excludes the needle, so it
    /// recompacts ~1 shard instead of all of them (O(1), not O(num_shards)). Maintained on append (merge the
    /// new rows' min/max). Empty = not pruned (always gathered) — e.g. benchmark shards.
    pub(crate) resident_device_int4_column_stats: Vec<ResidentDeviceInt4ColumnStats>,
    pub(crate) resident_bytes: u64,
    pub(crate) allocated_bytes: u64,
    pub(crate) count_header_byte_offset: u64,
    pub(crate) resident_device_int4_columns: Vec<String>,
    pub(crate) resident_device_text_columns: Vec<ResidentDeviceTextColumnLayout>,
    pub(crate) gpu_id: u16,
    pub(crate) schema: String,
    pub(crate) table: String,
    pub(crate) device_memory_proof: Option<CudaDeviceMemoryProof>,
    pub(crate) invalidated_by_txn_id: Option<TxnId>,
    pub(crate) invalidated_at_index: Option<Index>,
    pub(crate) invalidated_by_memory_pressure: bool,
    pub(crate) memory_pressure_active: bool,
}

impl RelationalResidentShard {
    pub(crate) fn is_valid(&self, memory_pressure_active: bool) -> bool {
        self.invalidated_by_txn_id.is_none()
            && self.invalidated_at_index.is_none()
            && !self.invalidated_by_memory_pressure
            && !memory_pressure_active
    }
}

impl RelationalResidentCache {
    pub(crate) fn record_decision(&mut self, decision: RelationalResidentCacheDecision) {
        self.last_decisions.insert(decision.table.clone(), decision);
    }

    pub(crate) fn last_decision(&self, table: &str) -> Option<&RelationalResidentCacheDecision> {
        self.last_decisions.get(table)
    }

    /// Drop a table's serialized-path metadata, plus its device memory (now in `residency`) and its
    /// route telemetry (now in `telemetry`). The device-memory/route-telemetry stores live in the
    /// shared `Arc<ReadState>`; the catalog-latch caller passes `&self`-views of them in.
    /// Drop a table's resident metadata. The snapshot/shard maps + device memory now live in the
    /// shared `residency` (Stage 3); the route telemetry in `telemetry`. `&self` (nothing on the
    /// `RelationalResidentCache` itself is removed — its budget/last-decision accounting is keyed
    /// independently and pruned elsewhere); the catalog-latch caller passes `&self`-views in.
    pub(crate) fn remove_table(
        &self,
        table: &str,
        residency: &ResidencyReadState,
        telemetry: &RouteTelemetry,
    ) {
        residency.with_snapshots_mut(|snapshots| snapshots.remove(table));
        residency.device_memory.remove(table);
        residency.with_shards_mut(|shards| shards.remove(table));
        residency.shard_device_memory.remove_table(table);
        telemetry.remove_table(table);
    }

    pub(crate) fn install_snapshot(
        &self,
        table: String,
        descriptor: RelationalResidencySnapshot,
        host_rows: Vec<Vec<SqlValue>>,
        device_memory: Option<CudaResidentDeviceMemory>,
        residency: &ResidencyReadState,
    ) {
        if let Some(device_memory) = device_memory {
            residency.device_memory.insert(table.clone(), device_memory);
        } else {
            // Refreshed without device memory (e.g. no GPU): publish a tombstone so any
            // in-flight reader of a prior resident generation keeps it.
            residency.device_memory.invalidate(&table);
        }
        // Co-publish the lightweight descriptor + the heavy host rows as one Arc-shared entry, so
        // the COW map clone (every reader + every invalidation) bumps refcounts, not row data. Admit lays
        // the rows down as ONE segment; an INSERT commit appends further segments (Slice 1b-ii-d).
        let entry = RelationalResidencyEntry::from_dense_host_rows(Arc::new(descriptor), host_rows);
        residency.with_snapshots_mut(|snapshots| snapshots.insert(table, entry));
    }

    pub(crate) fn install_shards(
        &self,
        table: String,
        shards: Vec<RelationalResidentShard>,
        device_memory: BTreeMap<u32, CudaResidentDeviceMemory>,
        residency: &ResidencyReadState,
    ) {
        residency
            .shard_device_memory
            .install_table_shards(&table, device_memory);
        residency.with_shards_mut(|map| map.insert(table, shards));
    }
}
