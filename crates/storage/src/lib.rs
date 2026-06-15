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
}

impl InMemoryTupleStore {
    pub fn new() -> Self {
        Self {
            next_tuple_id: 1,
            versions: imbl::OrdMap::new(),
        }
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
        for (id, chain) in self.versions.iter() {
            let mut chain = std::sync::Arc::clone(chain);
            std::sync::Arc::make_mut(&mut chain).retain(|version| {
                version
                    .deleted_by
                    .is_none_or(|deleted_by| deleted_by > safe_txn_id)
            });
            if !chain.is_empty() {
                pruned.insert(*id, chain);
            }
        }
        self.versions = pruned;

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
        Ok(())
    }

    fn tuple_delete(&mut self, tuple_id: TupleId, txn_id: TxnId) -> Result<(), StorageError> {
        if txn_id == 0 {
            return Err(StorageError::InvalidVisibility);
        }

        let current = self.current_version_mut(tuple_id)?;
        current.deleted_by = Some(txn_id);
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
        let versions = self
            .visible_versions(visibility)?
            .into_iter()
            .filter(|version| version.key == key)
            .collect();
        Ok(Box::new(InMemoryCursor::new(versions)))
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
}
