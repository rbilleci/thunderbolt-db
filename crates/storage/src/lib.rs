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
    versions: std::collections::BTreeMap<TupleId, Vec<TupleVersion>>,
}

impl InMemoryTupleStore {
    pub fn new() -> Self {
        Self {
            next_tuple_id: 1,
            versions: std::collections::BTreeMap::new(),
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
        self.versions
            .get_mut(&tuple_id)
            .and_then(|versions| {
                versions
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

        let tuple_id = self.next_tuple_id;
        self.next_tuple_id += 1;
        self.versions.insert(
            tuple_id,
            vec![TupleVersion {
                tuple_id,
                key: tuple.key,
                value: tuple.value,
                created_by: txn_id,
                deleted_by: None,
            }],
        );
        Ok(tuple_id)
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
        versions.push(TupleVersion {
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
    }
}
