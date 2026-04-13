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
}

pub trait SeqScanCursor {
    fn next(&mut self) -> Option<TupleVersion>;
}

pub trait IndexScanCursor {
    fn next(&mut self) -> Option<TupleVersion>;
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
    }
}
