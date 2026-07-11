pub type TupleId = u64;
pub type TxnId = u64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TupleVersion {
    pub tuple_id: TupleId,
    pub key: String,
    pub value: String,
    pub created_by: TxnId,
    pub deleted_by: Option<TxnId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Visibility {
    pub read_txn_id: TxnId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewTuple {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PruneStats {
    pub removed_versions: usize,
    pub removed_tuples: usize,
    pub remaining_versions: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StorageError {
    #[error("tuple not found")]
    NotFound,
    #[error("tuple already exists")]
    AlreadyExists,
    #[error("invalid visibility")]
    InvalidVisibility,
}

pub trait TupleStore {
    fn tuple_fetch(
        &self,
        tuple_id: TupleId,
        visibility: Visibility,
    ) -> Result<Option<TupleVersion>, StorageError>;

    fn tuple_insert(&mut self, tuple: NewTuple, txn_id: TxnId) -> Result<TupleId, StorageError>;

    fn tuple_update(
        &mut self,
        tuple_id: TupleId,
        new_value: String,
        txn_id: TxnId,
    ) -> Result<(), StorageError>;

    fn tuple_delete(&mut self, tuple_id: TupleId, txn_id: TxnId) -> Result<(), StorageError>;

    fn seq_scan_open(
        &self,
        visibility: Visibility,
    ) -> Result<Box<dyn SeqScanCursor + '_>, StorageError>;

    fn index_scan_open(
        &self,
        key: &str,
        visibility: Visibility,
    ) -> Result<Box<dyn IndexScanCursor + '_>, StorageError>;

    fn tuple_fetch_by_key(
        &self,
        key: &str,
        visibility: Visibility,
    ) -> Result<Option<TupleVersion>, StorageError> {
        let mut cursor = self.index_scan_open(key, visibility)?;
        Ok(cursor.next())
    }

    fn key_exists_at_visibility(
        &self,
        key: &str,
        visibility: Visibility,
    ) -> Result<bool, StorageError> {
        Ok(self.tuple_fetch_by_key(key, visibility)?.is_some())
    }

    fn visible_tuple_count(&self, visibility: Visibility) -> Result<usize, StorageError> {
        let mut cursor = self.seq_scan_open(visibility)?;
        let mut count = 0;
        while cursor.next().is_some() {
            count += 1;
        }
        Ok(count)
    }
}

pub trait SeqScanCursor {
    fn next(&mut self) -> Option<TupleVersion>;
}

pub trait IndexScanCursor {
    fn next(&mut self) -> Option<TupleVersion>;
}

#[derive(Debug, Default, Clone)]
pub struct InMemoryTupleStore {
    next_tuple_id: TupleId,
    // Persistent immutable ordered map (`imbl::OrdMap`): O(1) clone (refcount bump) and O(log n)
    // structurally-shared update. The per-commit whole-table clone the engine does becomes O(1),
    // and a commit touching k chains is O(k·log n) — no more O(table) deep copy per write. Each
    // version chain is `Arc`-wrapped so a clone shares chains until one is mutated, at which point
    // `Arc::make_mut` copies ONLY that chain (copy-on-write). Iteration stays in `TupleId` order.
    versions: imbl::OrdMap<TupleId, std::sync::Arc<Vec<TupleVersion>>>,
    // Row-`key` → `TupleId`s index, so a by-`key` fetch is O(matches·chain_depth + log n) instead of
    // scanning every chain (`versions` is keyed by `TupleId`; the row key lives inside each
    // `TupleVersion`, so without this an equality point-lookup resolves in O(rows)). Maintained in
    // lockstep with `versions` and published in the SAME immutable generation. Each slot holds a
    // `Vec<TupleId>` for defensive multiplicity, mirroring the engine value-index's append + dedup-
    // on-read contract; relational row keys are unique per table so a slot is normally a single id.
    // `Arc`-wrapped like a chain, so a clone shares id-lists until `Arc::make_mut` copies one (COW).
    key_to_tuple_ids: imbl::OrdMap<String, std::sync::Arc<Vec<TupleId>>>,
    // The WRITE-SIDE CHANGE LOG (6c-1 fix): a monotonic mutation epoch + the last
    // `CHANGE_LOG_CAP` `(epoch, tuple_id)` mutations. `changed_tuple_ids` between two
    // generations reads the log slice `(old.epoch, new.epoch]` — EXACT O(delta) ids by
    // construction. It exists because `imbl::OrdMap::diff` was PROVEN to MISS real changes
    // (repro: three sequential single-row deletes against pinned generations — the third
    // vanished from the diff while the chains' contents differed), silently corrupting any
    // consumer that trusts the structural diff. Structural diffing is BANNED for
    // correctness-bearing deltas; the out-of-window fallback is a full pointer-pruned key walk.
    epoch: u64,
    recent_changes: imbl::Vector<(u64, TupleId)>,
}

/// The change-log window. A delta wider than this (one giant bulk write, or a generation gap
/// spanning many epochs) falls back to the O(n) walk — correct, just not O(delta).
const CHANGE_LOG_CAP: usize = 65536;

impl InMemoryTupleStore {
    pub fn new() -> Self {
        Self {
            next_tuple_id: 1,
            versions: imbl::OrdMap::new(),
            key_to_tuple_ids: imbl::OrdMap::new(),
            epoch: 0,
            recent_changes: imbl::Vector::new(),
        }
    }

    /// Record one mutation in the change log (called by every `versions`-mutating write).
    fn record_change(&mut self, tuple_id: TupleId) {
        self.epoch += 1;
        self.recent_changes.push_back((self.epoch, tuple_id));
        if self.recent_changes.len() > CHANGE_LOG_CAP {
            self.recent_changes.pop_front();
        }
    }

    /// STRATA 6c-1 (chunk-granular cold-tier patching): the TupleIds whose version CHAINS differ
    /// between two store generations (`self` = the older, pinned one). PRIMARY SOURCE: the
    /// write-side change log — the slice `(self.epoch, newer.epoch]` is exactly the mutated ids,
    /// O(delta) by construction. `imbl::OrdMap::diff` is DELIBERATELY NOT USED: it was proven to
    /// MISS real changes (the pinned-generation triple-delete repro), which silently corrupts
    /// every consumer. When the log window does not cover the gap (bulk write / long-lived pin),
    /// fall back to a FULL parallel key walk with per-chain `Arc` pointer pruning — O(n) but
    /// correct by construction (a pointer-unequal but content-equal chain is a harmless spurious
    /// id: the consumer rebuilds a clean chunk). Control-plane addressing only.
    /// CONTRACT (audit LOW): the log-slice arm is truthful only when `newer` DESCENDS from
    /// `self` (every caller diffs a pinned past snapshot of a table against that table's live
    /// lineage). Two SIBLING generations sharing an epoch range would lie here — nothing in the
    /// engine produces that shape; a wholesale store replacement restarts at epoch 0 and lands
    /// in `newer.epoch < self.epoch`, which takes the fallback walk.
    pub fn changed_tuple_ids(&self, newer: &Self) -> Vec<TupleId> {
        if newer.epoch == self.epoch {
            return Vec::new();
        }
        if newer.epoch > self.epoch {
            let covered = newer
                .recent_changes
                .front()
                .is_some_and(|(first, _)| *first <= self.epoch + 1);
            if covered {
                let mut ids: Vec<TupleId> = newer
                    .recent_changes
                    .iter()
                    .filter(|(epoch, _)| *epoch > self.epoch)
                    .map(|(_, id)| *id)
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                return ids;
            }
        }
        // Out-of-window fallback: zip both key-ordered maps; a key present on one side only, or
        // present on both with pointer-UNEQUAL chains, is changed.
        let mut ids = Vec::new();
        let mut a = self.versions.iter().peekable();
        let mut b = newer.versions.iter().peekable();
        loop {
            match (a.peek(), b.peek()) {
                (Some((ka, va)), Some((kb, vb))) => match ka.cmp(kb) {
                    std::cmp::Ordering::Equal => {
                        if !std::sync::Arc::ptr_eq(va, vb) {
                            ids.push(**ka);
                        }
                        a.next();
                        b.next();
                    }
                    std::cmp::Ordering::Less => {
                        ids.push(**ka);
                        a.next();
                    }
                    std::cmp::Ordering::Greater => {
                        ids.push(**kb);
                        b.next();
                    }
                },
                (Some((ka, _)), None) => {
                    ids.push(**ka);
                    a.next();
                }
                (None, Some((kb, _))) => {
                    ids.push(**kb);
                    b.next();
                }
                (None, None) => break,
            }
        }
        ids
    }

    /// P4 RECLAMATION (chunk-authoritative tables): drop EVERY version chain and key-index slot
    /// while PRESERVING the tuple-id allocator (identities must never be reused across a later
    /// de-authoritization rebuild) and the change-log epoch continuity (a cleared store restarts
    /// its log; any stray ancestor diff lands in the out-of-window fallback walk — safe). The
    /// caller owns the invariant that nothing reads these rows afterwards (the class guard
    /// funnel + COW generation pinning for in-flight readers).
    pub fn clear_versions_preserving_allocator(&mut self) {
        self.versions = imbl::OrdMap::new();
        self.key_to_tuple_ids = imbl::OrdMap::new();
        self.epoch += 1;
        self.recent_changes = imbl::Vector::new();
    }

    /// P2 (cold-chunk tombstone sidecars): one chain's versions, for the patch classifier's
    /// old-vs-new comparison. Control-plane addressing only (no row values inspected by the
    /// caller beyond structural equality).
    pub fn chain(&self, id: TupleId) -> Option<&std::sync::Arc<Vec<TupleVersion>>> {
        self.versions.get(&id)
    }

    /// P2 (cold-chunk tombstone sidecars): COUNT of chains with a visible version in the
    /// INCLUSIVE TupleId range — the sidecar's SLOT-RANK walk (a chunk's slots are its visible
    /// rows in TupleId order, so a row's slot is the count of visible ids below it in the
    /// chunk's range). Same visibility semantics as `visible_versions_in_range`, without the
    /// clones.
    pub fn visible_count_in_range(
        &self,
        visibility: Visibility,
        lo: TupleId,
        hi: TupleId,
    ) -> Result<usize, StorageError> {
        Self::validate_visibility(visibility)?;
        if lo > hi {
            return Ok(0);
        }
        Ok(self
            .versions
            .range(lo..=hi)
            .filter(|(_, versions)| {
                versions
                    .iter()
                    .rev()
                    .any(|version| Self::is_visible(version, visibility))
            })
            .count())
    }

    /// STRATA 6c-1: the newest VISIBLE version per chain within an INCLUSIVE TupleId range — the
    /// bounded rebuild scan for ONE dirty cold chunk. Byte-identical visibility semantics to
    /// `visible_versions` (same `is_visible`, same newest-first resolution), restricted by range.
    pub fn visible_versions_in_range(
        &self,
        visibility: Visibility,
        lo: TupleId,
        hi: TupleId,
    ) -> Result<Vec<TupleVersion>, StorageError> {
        Self::validate_visibility(visibility)?;
        Ok(self
            .versions
            .range(lo..=hi)
            .filter_map(|(_, versions)| {
                versions
                    .iter()
                    .rev()
                    .find(|version| Self::is_visible(version, visibility))
                    .cloned()
            })
            .collect())
    }

    pub fn all_versions(&self) -> Vec<TupleVersion> {
        self.versions
            .values()
            .flat_map(|versions| versions.iter().cloned())
            .collect()
    }

    pub fn version_count(&self) -> usize {
        self.versions.values().map(|versions| versions.len()).sum()
    }

    pub fn tuple_chain_count(&self) -> usize {
        self.versions.len()
    }

    pub fn prune_versions_deleted_at_or_before(&mut self, safe_txn_id: TxnId) -> PruneStats {
        let before_versions = self.version_count();
        let before_tuples = self.tuple_chain_count();

        // `imbl::OrdMap` has no in-place `retain`; rebuild the surviving chains into a fresh map.
        // This is the GC path (off the hot commit path), so the rebuild cost is not critical —
        // correctness (and preserving `TupleId` order, which OrdMap maintains) is. `Arc::make_mut`
        // shrinks a chain in place when it is uniquely owned, copying only a chain shared with a
        // live snapshot (COW), exactly as the per-version mutation sites do.
        let mut pruned = imbl::OrdMap::new();
        // Keys whose chain is fully GC'd here; their id-list entry is dropped from the key index
        // after the loop (can't mutate `key_to_tuple_ids` while iterating `versions`). A chain's
        // versions all share one `key`, so the first version names it.
        let mut dropped: Vec<(String, TupleId)> = Vec::new();
        for (id, chain) in self.versions.iter() {
            let chain_key = chain.first().map(|version| version.key.clone());
            let mut chain = std::sync::Arc::clone(chain);
            std::sync::Arc::make_mut(&mut chain).retain(|version| {
                version
                    .deleted_by
                    .is_none_or(|deleted_by| deleted_by > safe_txn_id)
            });
            if !chain.is_empty() {
                pruned.insert(*id, chain);
            } else if let Some(chain_key) = chain_key {
                dropped.push((chain_key, *id));
            }
        }
        // CHANGE-LOG NOTE (audit LOW): pruning deliberately does NOT `record_change` — every
        // touched version has `deleted_by <= safe_txn_id` (the GC horizon = the minimum live
        // snapshot), so the mutation is invisible to every servable reader and cannot change a
        // served result; the DELETE that preceded the prune was already recorded at its commit,
        // and an out-of-window diff's fallback walk catches pruned keys structurally. If the
        // horizon/servability invariant ever weakens, this exemption must be revisited.
        self.versions = pruned;
        for (key, id) in dropped {
            self.index_key_remove(&key, id);
        }

        let remaining_versions = self.version_count();
        PruneStats {
            removed_versions: before_versions.saturating_sub(remaining_versions),
            removed_tuples: before_tuples.saturating_sub(self.tuple_chain_count()),
            remaining_versions,
        }
    }

    fn validate_visibility(visibility: Visibility) -> Result<(), StorageError> {
        if visibility.read_txn_id == 0 {
            return Err(StorageError::InvalidVisibility);
        }
        Ok(())
    }

    fn is_visible(version: &TupleVersion, visibility: Visibility) -> bool {
        version.created_by <= visibility.read_txn_id
            && version
                .deleted_by
                .is_none_or(|deleted_by| deleted_by > visibility.read_txn_id)
    }

    // Record a fresh chain's `tuple_id` under its row `key`. `Arc::make_mut` copies the id-list ONLY
    // if a live snapshot still shares it (copy-on-write), exactly like the version-chain mutation
    // sites, so a pinned reader's generation is never mutated. Called from the insert paths, which
    // allocate a chain at a new `tuple_id`; an update/delete reuses the same `tuple_id`+`key` (a new
    // version is pushed into the SAME chain) so the index is unchanged.
    fn index_key_insert(&mut self, key: &str, tuple_id: TupleId) {
        let mut ids = self.key_to_tuple_ids.get(key).cloned().unwrap_or_default();
        std::sync::Arc::make_mut(&mut ids).push(tuple_id);
        self.key_to_tuple_ids.insert(key.to_string(), ids);
    }

    // Drop a chain's `tuple_id` from its key's id-list (the prune path, when a chain is fully GC'd);
    // remove the slot entirely once its list is empty. COW via `Arc::make_mut`, same as inserts.
    fn index_key_remove(&mut self, key: &str, tuple_id: TupleId) {
        if let Some(ids) = self.key_to_tuple_ids.get(key).cloned() {
            let mut ids = ids;
            std::sync::Arc::make_mut(&mut ids).retain(|id| *id != tuple_id);
            if ids.is_empty() {
                self.key_to_tuple_ids.remove(key);
            } else {
                self.key_to_tuple_ids.insert(key.to_string(), ids);
            }
        }
    }

    fn current_version_mut(
        &mut self,
        tuple_id: TupleId,
    ) -> Result<&mut TupleVersion, StorageError> {
        // `OrdMap::get_mut` structurally clones the path to this entry; `Arc::make_mut` then copies
        // the chain ONLY if it is still shared with a live snapshot (copy-on-write), so an in-flight
        // reader's pinned generation is never mutated.
        self.versions
            .get_mut(&tuple_id)
            .and_then(|chain| {
                std::sync::Arc::make_mut(chain)
                    .iter_mut()
                    .rev()
                    .find(|version| version.deleted_by.is_none())
            })
            .ok_or(StorageError::NotFound)
    }

    fn visible_versions(&self, visibility: Visibility) -> Result<Vec<TupleVersion>, StorageError> {
        Self::validate_visibility(visibility)?;
        Ok(self
            .versions
            .values()
            .filter_map(|versions| {
                versions
                    .iter()
                    .rev()
                    .find(|version| Self::is_visible(version, visibility))
                    .cloned()
            })
            .collect())
    }

    // The visible versions of every chain whose row key is `key`, resolved through the key index
    // (O(matches·chain_depth + log n)) rather than scanning every chain. The per-version
    // `is_visible` predicate is byte-identical to `visible_versions`: the candidate set is narrowed
    // by KEY only, so the result is exactly the rows visible at the pinned `read_txn_id` that match
    // `key` — identical to `visible_versions().filter(|v| v.key == key)`, just without the full scan.
    // The `v.key == key` guard is defensive against a stale id-list entry. A missing key yields an
    // empty result, matching the old filtered seq-scan.
    fn index_lookup(
        &self,
        key: &str,
        visibility: Visibility,
    ) -> Result<Vec<TupleVersion>, StorageError> {
        Self::validate_visibility(visibility)?;
        let Some(tuple_ids) = self.key_to_tuple_ids.get(key) else {
            return Ok(Vec::new());
        };
        Ok(tuple_ids
            .iter()
            .filter_map(|tuple_id| {
                self.versions.get(tuple_id).and_then(|versions| {
                    versions
                        .iter()
                        .rev()
                        .find(|version| Self::is_visible(version, visibility) && version.key == key)
                        .cloned()
                })
            })
            .collect())
    }

    fn key_exists(&self, key: &str) -> bool {
        self.versions.values().any(|versions| {
            versions
                .iter()
                .rev()
                .find(|version| version.deleted_by.is_none())
                .is_some_and(|version| version.key == key)
        })
    }

    pub fn tuple_insert_reserved_key(
        &mut self,
        tuple: NewTuple,
        txn_id: TxnId,
    ) -> Result<TupleId, StorageError> {
        if txn_id == 0 {
            return Err(StorageError::InvalidVisibility);
        }

        let tuple_id = self.next_tuple_id;
        self.next_tuple_id += 1;
        let key = tuple.key.clone();
        self.versions.insert(
            tuple_id,
            std::sync::Arc::new(vec![TupleVersion {
                tuple_id,
                key: tuple.key,
                value: tuple.value,
                created_by: txn_id,
                deleted_by: None,
            }]),
        );
        self.index_key_insert(&key, tuple_id);
        self.record_change(tuple_id);
        Ok(tuple_id)
    }

    /// Insert a fresh version chain at a CALLER-supplied `tuple_id`, skipping the live-key
    /// uniqueness check (the reserved-key contract). Used when tuple ids are allocated by an
    /// external shared allocator (the engine's per-table `MvccData` partitions a single
    /// monotonic id space across partition stores, so ids stay globally unique and identical
    /// to the pre-partition single store). `tuple_id` must not already exist in this store.
    pub fn tuple_insert_reserved_key_with_id(
        &mut self,
        tuple_id: TupleId,
        tuple: NewTuple,
        txn_id: TxnId,
    ) -> Result<TupleId, StorageError> {
        if txn_id == 0 {
            return Err(StorageError::InvalidVisibility);
        }
        if self.versions.contains_key(&tuple_id) {
            return Err(StorageError::AlreadyExists);
        }
        let key = tuple.key.clone();
        self.versions.insert(
            tuple_id,
            std::sync::Arc::new(vec![TupleVersion {
                tuple_id,
                key: tuple.key,
                value: tuple.value,
                created_by: txn_id,
                deleted_by: None,
            }]),
        );
        self.index_key_insert(&key, tuple_id);
        self.record_change(tuple_id);
        Ok(tuple_id)
    }

    /// Insert a fresh version chain at a CALLER-supplied `tuple_id`, enforcing the live-key
    /// uniqueness check (the `tuple_insert` contract) within THIS partition store.
    pub fn tuple_insert_with_id(
        &mut self,
        tuple_id: TupleId,
        tuple: NewTuple,
        txn_id: TxnId,
    ) -> Result<TupleId, StorageError> {
        if txn_id == 0 {
            return Err(StorageError::InvalidVisibility);
        }
        if self.key_exists(&tuple.key) {
            return Err(StorageError::AlreadyExists);
        }
        self.tuple_insert_reserved_key_with_id(tuple_id, tuple, txn_id)
    }
}

#[derive(Debug)]
struct InMemoryCursor {
    versions: std::vec::IntoIter<TupleVersion>,
}

impl InMemoryCursor {
    fn new(versions: Vec<TupleVersion>) -> Self {
        Self {
            versions: versions.into_iter(),
        }
    }
}

impl SeqScanCursor for InMemoryCursor {
    fn next(&mut self) -> Option<TupleVersion> {
        self.versions.next()
    }
}

impl IndexScanCursor for InMemoryCursor {
    fn next(&mut self) -> Option<TupleVersion> {
        self.versions.next()
    }
}

impl TupleStore for InMemoryTupleStore {
    fn tuple_fetch(
        &self,
        tuple_id: TupleId,
        visibility: Visibility,
    ) -> Result<Option<TupleVersion>, StorageError> {
        Self::validate_visibility(visibility)?;
        Ok(self.versions.get(&tuple_id).and_then(|versions| {
            versions
                .iter()
                .rev()
                .find(|version| Self::is_visible(version, visibility))
                .cloned()
        }))
    }

    fn tuple_insert(&mut self, tuple: NewTuple, txn_id: TxnId) -> Result<TupleId, StorageError> {
        if txn_id == 0 {
            return Err(StorageError::InvalidVisibility);
        }
        if self.key_exists(&tuple.key) {
            return Err(StorageError::AlreadyExists);
        }

        self.tuple_insert_reserved_key(tuple, txn_id)
    }

    fn tuple_update(
        &mut self,
        tuple_id: TupleId,
        new_value: String,
        txn_id: TxnId,
    ) -> Result<(), StorageError> {
        if txn_id == 0 {
            return Err(StorageError::InvalidVisibility);
        }

        let (key, tuple_id) = {
            let current = self.current_version_mut(tuple_id)?;
            current.deleted_by = Some(txn_id);
            (current.key.clone(), current.tuple_id)
        };

        let versions = self
            .versions
            .get_mut(&tuple_id)
            .ok_or(StorageError::NotFound)?;
        // `current_version_mut` already made this chain uniquely owned, so this `make_mut` is the
        // O(1) refcount==1 case; it stays correct (COW) regardless.
        std::sync::Arc::make_mut(versions).push(TupleVersion {
            tuple_id,
            key,
            value: new_value,
            created_by: txn_id,
            deleted_by: None,
        });
        self.record_change(tuple_id);
        Ok(())
    }

    fn tuple_delete(&mut self, tuple_id: TupleId, txn_id: TxnId) -> Result<(), StorageError> {
        if txn_id == 0 {
            return Err(StorageError::InvalidVisibility);
        }

        let current = self.current_version_mut(tuple_id)?;
        current.deleted_by = Some(txn_id);
        self.record_change(tuple_id);
        Ok(())
    }

    fn seq_scan_open(
        &self,
        visibility: Visibility,
    ) -> Result<Box<dyn SeqScanCursor + '_>, StorageError> {
        Ok(Box::new(InMemoryCursor::new(
            self.visible_versions(visibility)?,
        )))
    }

    fn index_scan_open(
        &self,
        key: &str,
        visibility: Visibility,
    ) -> Result<Box<dyn IndexScanCursor + '_>, StorageError> {
        // Resolve via the key index — O(matches·chain_depth + log n) — instead of filtering a full
        // visible-version scan. Visibility filtering is unchanged, so the rows are identical to the
        // old `visible_versions().filter(|v| v.key == key)`.
        Ok(Box::new(InMemoryCursor::new(
            self.index_lookup(key, visibility)?,
        )))
    }

    // Override the O(rows) trait default (which opens an index scan and takes its first row): a
    // by-key point fetch resolves the newest visible version directly through the key index, so an
    // equality point-lookup is O(log n + matches) instead of scanning every chain. The newest
    // version comes first because `index_lookup` searches each chain newest-first (`.rev()`).
    fn tuple_fetch_by_key(
        &self,
        key: &str,
        visibility: Visibility,
    ) -> Result<Option<TupleVersion>, StorageError> {
        Ok(self.index_lookup(key, visibility)?.into_iter().next())
    }
}

#[cfg(test)]
mod change_log_tests {
    use super::*;

    fn store_with_rows(n: u64) -> InMemoryTupleStore {
        let mut store = InMemoryTupleStore::new();
        for i in 0..n {
            store
                .tuple_insert(
                    NewTuple {
                        key: format!("k{i:08}"),
                        value: format!("v{i}"),
                    },
                    1,
                )
                .expect("insert");
        }
        store
    }

    /// The pinned-generation triple-delete shape that `imbl::OrdMap::diff` was PROVEN to miss
    /// (the third delete's id vanished while the chains differed) — the change log must report
    /// every step, at the store level, forever.
    #[test]
    fn pinned_generation_deltas_report_every_delete() {
        let mut store = store_with_rows(1500);
        let g0 = store.clone();
        store.tuple_delete(101, 5).expect("d1");
        let g1 = store.clone();
        assert_eq!(g0.changed_tuple_ids(&g1), vec![101]);
        store.tuple_delete(701, 6).expect("d2");
        let g2 = store.clone();
        assert_eq!(g1.changed_tuple_ids(&g2), vec![701]);
        store.tuple_delete(1401, 7).expect("d3");
        let g3 = store.clone();
        assert_eq!(g2.changed_tuple_ids(&g3), vec![1401]);
        // And spanning multiple mutations: the window slice dedups + sorts.
        assert_eq!(g0.changed_tuple_ids(&g3), vec![101, 701, 1401]);
    }

    /// AUDIT MEDIUM (adopted): the OUT-OF-WINDOW fallback walk is the correctness backstop when
    /// a pin outlives `CHANGE_LOG_CAP` mutations — drive the log past the cap and assert the
    /// fallback still reports exactly the changed ids (adds, updates, deletes).
    #[test]
    fn out_of_window_fallback_walk_reports_every_delta() {
        let mut store = store_with_rows(64);
        let pinned = store.clone();
        // Blow the window: > CHANGE_LOG_CAP mutations after the pin (updates on one row).
        for i in 0..(CHANGE_LOG_CAP as u64 + 10) {
            store
                .tuple_update(1, format!("spin{i}"), 10 + i)
                .expect("update");
        }
        // Plus a delete and an insert whose ids must ALSO survive the fallback.
        store.tuple_delete(33, 999_999).expect("delete");
        let added = store
            .tuple_insert(
                NewTuple {
                    key: "fresh".to_string(),
                    value: "x".to_string(),
                },
                1_000_000,
            )
            .expect("insert");
        // The window cannot cover the pin now — this exercises the zip walk.
        assert!(
            store.recent_changes.front().map(|(e, _)| *e).unwrap_or(0) > pinned.epoch + 1,
            "premise: the log window no longer covers the pinned epoch"
        );
        let mut changed = pinned.changed_tuple_ids(&store);
        changed.sort_unstable();
        assert_eq!(changed, vec![1, 33, added]);
        // Symmetric sanity: an untouched clone diffs empty through the fast path.
        let same = store.clone();
        assert_eq!(store.changed_tuple_ids(&same), Vec::<TupleId>::new());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EmptyStore;

    impl TupleStore for EmptyStore {
        fn tuple_fetch(
            &self,
            _tuple_id: TupleId,
            _visibility: Visibility,
        ) -> Result<Option<TupleVersion>, StorageError> {
            Ok(None)
        }

        fn tuple_insert(
            &mut self,
            _tuple: NewTuple,
            _txn_id: TxnId,
        ) -> Result<TupleId, StorageError> {
            Err(StorageError::NotFound)
        }

        fn tuple_update(
            &mut self,
            _tuple_id: TupleId,
            _new_value: String,
            _txn_id: TxnId,
        ) -> Result<(), StorageError> {
            Err(StorageError::NotFound)
        }

        fn tuple_delete(&mut self, _tuple_id: TupleId, _txn_id: TxnId) -> Result<(), StorageError> {
            Err(StorageError::NotFound)
        }

        fn seq_scan_open(
            &self,
            _visibility: Visibility,
        ) -> Result<Box<dyn SeqScanCursor + '_>, StorageError> {
            Ok(Box::new(std::iter::empty::<TupleVersion>()))
        }

        fn index_scan_open(
            &self,
            _key: &str,
            _visibility: Visibility,
        ) -> Result<Box<dyn IndexScanCursor + '_>, StorageError> {
            Ok(Box::new(std::iter::empty::<TupleVersion>()))
        }
    }

    impl SeqScanCursor for std::iter::Empty<TupleVersion> {
        fn next(&mut self) -> Option<TupleVersion> {
            Iterator::next(self)
        }
    }

    impl IndexScanCursor for std::iter::Empty<TupleVersion> {
        fn next(&mut self) -> Option<TupleVersion> {
            Iterator::next(self)
        }
    }

    #[test]
    fn tuple_store_contract_allows_visibility_bound_reads() {
        let store = EmptyStore;
        let visibility = Visibility { read_txn_id: 42 };
        assert_eq!(store.tuple_fetch(1, visibility).unwrap(), None);
        assert_eq!(
            store.tuple_fetch_by_key("missing", visibility).unwrap(),
            None
        );
        assert!(!store
            .key_exists_at_visibility("missing", visibility)
            .unwrap());
        assert_eq!(store.visible_tuple_count(visibility).unwrap(), 0);
    }

    #[test]
    fn in_memory_store_insert_and_fetch_respect_visibility() {
        let mut store = InMemoryTupleStore::new();
        let tuple_id = store
            .tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "open".to_string(),
                },
                7,
            )
            .unwrap();

        assert_eq!(
            store
                .tuple_fetch(tuple_id, Visibility { read_txn_id: 6 })
                .unwrap(),
            None
        );

        let visible = store
            .tuple_fetch(tuple_id, Visibility { read_txn_id: 7 })
            .unwrap()
            .expect("version should become visible at creator txn");
        assert_eq!(visible.key, "acct:1");
        assert_eq!(visible.value, "open");
    }

    #[test]
    fn in_memory_store_update_preserves_snapshot_history() {
        let mut store = InMemoryTupleStore::new();
        let tuple_id = store
            .tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "open".to_string(),
                },
                3,
            )
            .unwrap();

        store
            .tuple_update(tuple_id, "closed".to_string(), 5)
            .unwrap();

        assert_eq!(
            store
                .tuple_fetch(tuple_id, Visibility { read_txn_id: 4 })
                .unwrap()
                .map(|version| version.value),
            Some("open".to_string())
        );
        assert_eq!(
            store
                .tuple_fetch(tuple_id, Visibility { read_txn_id: 5 })
                .unwrap()
                .map(|version| version.value),
            Some("closed".to_string())
        );
    }

    #[test]
    fn in_memory_store_delete_hides_versions_at_and_after_delete_txn() {
        let mut store = InMemoryTupleStore::new();
        let tuple_id = store
            .tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "open".to_string(),
                },
                2,
            )
            .unwrap();

        store.tuple_delete(tuple_id, 9).unwrap();

        assert!(store
            .tuple_fetch(tuple_id, Visibility { read_txn_id: 8 })
            .unwrap()
            .is_some());
        assert_eq!(
            store
                .tuple_fetch(tuple_id, Visibility { read_txn_id: 9 })
                .unwrap(),
            None
        );
    }

    #[test]
    fn in_memory_store_scans_only_visible_versions() {
        let mut store = InMemoryTupleStore::new();
        let acct_1 = store
            .tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "open".to_string(),
                },
                1,
            )
            .unwrap();
        let _acct_2 = store
            .tuple_insert(
                NewTuple {
                    key: "acct:2".to_string(),
                    value: "pending".to_string(),
                },
                2,
            )
            .unwrap();
        store.tuple_update(acct_1, "closed".to_string(), 4).unwrap();

        let mut seq = store.seq_scan_open(Visibility { read_txn_id: 3 }).unwrap();
        let mut seq_rows = Vec::new();
        while let Some(version) = seq.next() {
            seq_rows.push((version.key, version.value));
        }
        assert_eq!(
            seq_rows,
            vec![
                ("acct:1".to_string(), "open".to_string()),
                ("acct:2".to_string(), "pending".to_string())
            ]
        );

        let mut index = store
            .index_scan_open("acct:1", Visibility { read_txn_id: 4 })
            .unwrap();
        let hit = index.next().expect("acct:1 should be visible");
        assert_eq!(hit.value, "closed");
        assert_eq!(index.next(), None);
    }

    #[test]
    fn in_memory_store_rejects_duplicate_live_keys_and_zero_visibility() {
        let mut store = InMemoryTupleStore::new();
        store
            .tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "open".to_string(),
                },
                1,
            )
            .unwrap();

        assert_eq!(
            store.tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "dupe".to_string(),
                },
                2,
            ),
            Err(StorageError::AlreadyExists)
        );
        assert_eq!(
            store.tuple_fetch(1, Visibility { read_txn_id: 0 }),
            Err(StorageError::InvalidVisibility)
        );
    }

    #[test]
    fn tuple_store_fetch_by_key_uses_index_visibility() {
        let mut store = InMemoryTupleStore::new();
        store
            .tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "open".to_string(),
                },
                3,
            )
            .unwrap();

        assert_eq!(
            store
                .tuple_fetch_by_key("acct:1", Visibility { read_txn_id: 2 })
                .unwrap(),
            None
        );
        assert_eq!(
            store
                .tuple_fetch_by_key("acct:1", Visibility { read_txn_id: 3 })
                .unwrap()
                .map(|tuple| tuple.value),
            Some("open".to_string())
        );
        assert!(store
            .key_exists_at_visibility("acct:1", Visibility { read_txn_id: 3 })
            .unwrap());
        assert!(!store
            .key_exists_at_visibility("acct:2", Visibility { read_txn_id: 3 })
            .unwrap());
        assert_eq!(
            store
                .visible_tuple_count(Visibility { read_txn_id: 2 })
                .unwrap(),
            0
        );
        assert_eq!(
            store
                .visible_tuple_count(Visibility { read_txn_id: 3 })
                .unwrap(),
            1
        );
    }

    #[test]
    fn prune_versions_deleted_before_safe_boundary_keeps_current_history() {
        let mut store = InMemoryTupleStore::new();
        let tuple_id = store
            .tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "open".to_string(),
                },
                2,
            )
            .unwrap();
        store
            .tuple_update(tuple_id, "closed".to_string(), 5)
            .unwrap();

        assert_eq!(store.version_count(), 2);
        let stats = store.prune_versions_deleted_at_or_before(4);
        assert_eq!(
            stats,
            PruneStats {
                removed_versions: 0,
                removed_tuples: 0,
                remaining_versions: 2,
            }
        );
        assert_eq!(
            store
                .tuple_fetch(tuple_id, Visibility { read_txn_id: 4 })
                .unwrap()
                .map(|version| version.value),
            Some("open".to_string())
        );

        let stats = store.prune_versions_deleted_at_or_before(5);
        assert_eq!(
            stats,
            PruneStats {
                removed_versions: 1,
                removed_tuples: 0,
                remaining_versions: 1,
            }
        );
        assert_eq!(
            store
                .tuple_fetch(tuple_id, Visibility { read_txn_id: 5 })
                .unwrap()
                .map(|version| version.value),
            Some("closed".to_string())
        );
        assert_eq!(
            store
                .tuple_fetch(tuple_id, Visibility { read_txn_id: 4 })
                .unwrap(),
            None
        );
    }

    #[test]
    fn prune_versions_removes_fully_deleted_tuple_chains() {
        let mut store = InMemoryTupleStore::new();
        let tuple_id = store
            .tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "open".to_string(),
                },
                2,
            )
            .unwrap();
        store.tuple_delete(tuple_id, 5).unwrap();

        let stats = store.prune_versions_deleted_at_or_before(5);
        assert_eq!(
            stats,
            PruneStats {
                removed_versions: 1,
                removed_tuples: 1,
                remaining_versions: 0,
            }
        );
        assert_eq!(store.tuple_chain_count(), 0);
    }

    #[test]
    fn explicit_tuple_id_inserts_place_at_caller_id_and_enforce_contracts() {
        let mut store = InMemoryTupleStore::new();
        // Reserved-key variant skips the live-key uniqueness check (relational row keys are unique
        // per table by construction) and places the chain at the caller-supplied id.
        let id = store
            .tuple_insert_reserved_key_with_id(
                42,
                NewTuple {
                    key: "rel/people/0000000000000000001".to_string(),
                    value: "Ada".to_string(),
                },
                7,
            )
            .unwrap();
        assert_eq!(id, 42);
        assert_eq!(
            store
                .tuple_fetch(42, Visibility { read_txn_id: 7 })
                .unwrap()
                .map(|v| v.value),
            Some("Ada".to_string())
        );
        // A caller-supplied id that already exists is rejected (never silently overwrites a chain).
        assert_eq!(
            store.tuple_insert_reserved_key_with_id(
                42,
                NewTuple {
                    key: "rel/people/0000000000000000002".to_string(),
                    value: "dupe".to_string(),
                },
                8,
            ),
            Err(StorageError::AlreadyExists)
        );

        // The checked variant enforces live-key uniqueness within the partition (KV-namespace use).
        store
            .tuple_insert_with_id(
                43,
                NewTuple {
                    key: "kv-key".to_string(),
                    value: "v1".to_string(),
                },
                9,
            )
            .unwrap();
        assert_eq!(
            store.tuple_insert_with_id(
                44,
                NewTuple {
                    key: "kv-key".to_string(),
                    value: "v2".to_string(),
                },
                10,
            ),
            Err(StorageError::AlreadyExists)
        );
        // Zero visibility (txn id 0) is rejected by both variants.
        assert_eq!(
            store.tuple_insert_reserved_key_with_id(
                99,
                NewTuple {
                    key: "k".to_string(),
                    value: "v".to_string(),
                },
                0,
            ),
            Err(StorageError::InvalidVisibility)
        );
    }

    // The old by-key resolution: a full visible-version scan filtered by key. The key index must
    // return exactly these rows (visibility filtering is unchanged; only the candidate set narrows).
    fn seq_scan_filter_oracle(
        store: &InMemoryTupleStore,
        key: &str,
        visibility: Visibility,
    ) -> Vec<TupleVersion> {
        store
            .visible_versions(visibility)
            .unwrap()
            .into_iter()
            .filter(|version| version.key == key)
            .collect()
    }

    fn drain_index_scan(
        store: &InMemoryTupleStore,
        key: &str,
        visibility: Visibility,
    ) -> Vec<TupleVersion> {
        let mut cursor = store.index_scan_open(key, visibility).unwrap();
        let mut rows = Vec::new();
        while let Some(version) = cursor.next() {
            rows.push(version);
        }
        rows
    }

    #[test]
    fn by_key_fetch_matches_seq_scan_filter_across_chain_and_visibility() {
        let mut store = InMemoryTupleStore::new();
        // A second, unrelated chain so the index must actually discriminate by key (and a full scan
        // would have to skip it).
        store
            .tuple_insert(
                NewTuple {
                    key: "acct:2".to_string(),
                    value: "other".to_string(),
                },
                1,
            )
            .unwrap();
        // acct:1: insert@3 -> update@5 -> delete@7, exercising a multi-version chain.
        let acct_1 = store
            .tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "open".to_string(),
                },
                3,
            )
            .unwrap();
        store.tuple_update(acct_1, "closed".to_string(), 5).unwrap();
        store.tuple_delete(acct_1, 7).unwrap();

        // At every visibility boundary around the chain's events, the index path, the cursor, and the
        // seq-scan-filter oracle must all agree — including before the row exists and after it is
        // deleted (both empty), and for a key that was never inserted.
        for read_txn_id in [2, 3, 4, 5, 6, 7, 8] {
            let visibility = Visibility { read_txn_id };
            let oracle = seq_scan_filter_oracle(&store, "acct:1", visibility);
            assert_eq!(
                store.index_lookup("acct:1", visibility).unwrap(),
                oracle,
                "index_lookup diverged at read_txn_id={read_txn_id}"
            );
            assert_eq!(
                drain_index_scan(&store, "acct:1", visibility),
                oracle,
                "index_scan_open diverged at read_txn_id={read_txn_id}"
            );
            assert_eq!(
                store.tuple_fetch_by_key("acct:1", visibility).unwrap(),
                oracle.into_iter().next(),
                "tuple_fetch_by_key diverged at read_txn_id={read_txn_id}"
            );
        }

        // Spot-check the resolved values along the chain.
        assert_eq!(
            store
                .tuple_fetch_by_key("acct:1", Visibility { read_txn_id: 4 })
                .unwrap()
                .map(|v| v.value),
            Some("open".to_string())
        );
        assert_eq!(
            store
                .tuple_fetch_by_key("acct:1", Visibility { read_txn_id: 6 })
                .unwrap()
                .map(|v| v.value),
            Some("closed".to_string())
        );
        assert_eq!(
            store
                .tuple_fetch_by_key("acct:1", Visibility { read_txn_id: 7 })
                .unwrap(),
            None
        );

        // A key that was never inserted resolves empty, identical to the old filtered scan.
        let visibility = Visibility { read_txn_id: 8 };
        assert!(seq_scan_filter_oracle(&store, "missing", visibility).is_empty());
        assert_eq!(
            store.index_lookup("missing", visibility).unwrap(),
            Vec::new()
        );
        assert_eq!(
            store.tuple_fetch_by_key("missing", visibility).unwrap(),
            None
        );
    }

    #[test]
    fn by_key_fetch_resolves_multiple_distinct_keys() {
        let mut store = InMemoryTupleStore::new();
        for n in 1..=5 {
            store
                .tuple_insert(
                    NewTuple {
                        key: format!("acct:{n}"),
                        value: format!("v{n}"),
                    },
                    1,
                )
                .unwrap();
        }
        let visibility = Visibility { read_txn_id: 1 };
        for n in 1..=5 {
            let key = format!("acct:{n}");
            assert_eq!(
                store.index_lookup(&key, visibility).unwrap(),
                seq_scan_filter_oracle(&store, &key, visibility)
            );
            assert_eq!(
                store
                    .tuple_fetch_by_key(&key, visibility)
                    .unwrap()
                    .map(|v| v.value),
                Some(format!("v{n}"))
            );
        }
    }

    #[test]
    fn by_key_fetch_after_prune_that_removes_a_key() {
        let mut store = InMemoryTupleStore::new();
        let acct_1 = store
            .tuple_insert(
                NewTuple {
                    key: "acct:1".to_string(),
                    value: "open".to_string(),
                },
                2,
            )
            .unwrap();
        store
            .tuple_insert(
                NewTuple {
                    key: "acct:2".to_string(),
                    value: "keep".to_string(),
                },
                2,
            )
            .unwrap();
        store.tuple_delete(acct_1, 5).unwrap();

        // Pruning at the delete boundary drops acct:1's whole chain → its key index slot must go too.
        let stats = store.prune_versions_deleted_at_or_before(5);
        assert_eq!(stats.removed_tuples, 1);
        assert!(
            !store.key_to_tuple_ids.contains_key("acct:1"),
            "pruned key must be removed from the key index"
        );
        assert!(store.key_to_tuple_ids.contains_key("acct:2"));

        let visibility = Visibility { read_txn_id: 6 };
        // The pruned key now resolves empty, and the surviving key is unaffected — both matching the
        // seq-scan-filter oracle.
        assert_eq!(
            store.index_lookup("acct:1", visibility).unwrap(),
            seq_scan_filter_oracle(&store, "acct:1", visibility)
        );
        assert_eq!(
            store.tuple_fetch_by_key("acct:1", visibility).unwrap(),
            None
        );
        assert_eq!(
            store.index_lookup("acct:2", visibility).unwrap(),
            seq_scan_filter_oracle(&store, "acct:2", visibility)
        );
        assert_eq!(
            store
                .tuple_fetch_by_key("acct:2", visibility)
                .unwrap()
                .map(|v| v.value),
            Some("keep".to_string())
        );
    }

    // Every live version chain's current key must be present in the key index (the maintenance
    // invariant the read path depends on). Exercised across inserts, updates, deletes, and a prune.
    fn assert_key_index_covers_live_chains(store: &InMemoryTupleStore) {
        for chain in store.versions.values() {
            let key = &chain.last().expect("chain is non-empty").key;
            let ids = store
                .key_to_tuple_ids
                .get(key)
                .unwrap_or_else(|| panic!("live key {key:?} missing from key index"));
            assert!(
                ids.contains(&chain[0].tuple_id),
                "live chain {} not recorded under key {key:?}",
                chain[0].tuple_id
            );
        }
    }

    #[test]
    fn key_index_covers_every_live_chain() {
        let mut store = InMemoryTupleStore::new();
        let a = store
            .tuple_insert(
                NewTuple {
                    key: "a".to_string(),
                    value: "1".to_string(),
                },
                1,
            )
            .unwrap();
        let b = store
            .tuple_insert(
                NewTuple {
                    key: "b".to_string(),
                    value: "1".to_string(),
                },
                1,
            )
            .unwrap();
        store
            .tuple_insert_reserved_key_with_id(
                100,
                NewTuple {
                    key: "c".to_string(),
                    value: "1".to_string(),
                },
                1,
            )
            .unwrap();
        assert_key_index_covers_live_chains(&store);

        store.tuple_update(a, "2".to_string(), 2).unwrap();
        assert_key_index_covers_live_chains(&store);

        store.tuple_delete(b, 3).unwrap();
        // The chain still exists (its delete version is still retained), so its key is still indexed.
        assert_key_index_covers_live_chains(&store);

        // Prune drops b's chain; the invariant must still hold for what remains.
        store.prune_versions_deleted_at_or_before(3);
        assert_key_index_covers_live_chains(&store);
        assert!(!store.key_to_tuple_ids.contains_key("b"));
    }
}
