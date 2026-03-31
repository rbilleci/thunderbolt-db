use gpu_db_types::{EngineError, TxnId};

#[derive(Debug, Clone)]
pub struct WalRecord {
    pub txn_id: TxnId,
    pub payload: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct WalBuffer {
    records: Vec<WalRecord>,
    flushed: usize,
    fail_next_flush: bool,
}

impl WalBuffer {
    pub fn append(&mut self, rec: WalRecord) {
        self.records.push(rec);
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn truncate(&mut self, len: usize) {
        self.records.truncate(len);
        if self.flushed > self.records.len() {
            self.flushed = self.records.len();
        }
    }

    pub fn flush_all(&mut self) -> Result<(), EngineError> {
        if self.fail_next_flush {
            self.fail_next_flush = false;
            return Err(EngineError::Durability(
                "simulated wal flush failure".to_string(),
            ));
        }
        self.flushed = self.records.len();
        Ok(())
    }

    pub fn flushed_count(&self) -> usize {
        self.flushed
    }

    pub fn flushed_records(&self) -> &[WalRecord] {
        &self.records[..self.flushed]
    }

    pub fn unflushed_count(&self) -> usize {
        self.records.len().saturating_sub(self.flushed)
    }

    pub fn fail_next_flush(&mut self) {
        self.fail_next_flush = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_commits_all_appended_records() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec(),
        });

        wal.flush_all().unwrap();

        assert_eq!(wal.flushed_count(), 2);
    }

    #[test]
    fn fail_next_flush_is_one_shot() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });

        wal.fail_next_flush();
        let err = wal.flush_all().unwrap_err();
        assert!(matches!(err, EngineError::Durability(_)));
        assert_eq!(wal.flushed_count(), 0);

        wal.flush_all().unwrap();
        assert_eq!(wal.flushed_count(), 1);
    }

    #[test]
    fn truncate_shrinks_records_and_adjusts_flushed_count() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec(),
        });

        wal.flush_all().unwrap();
        assert_eq!(wal.flushed_count(), 2);

        wal.truncate(1);
        assert_eq!(wal.len(), 1);
        assert_eq!(wal.flushed_count(), 1);
    }

    #[test]
    fn unflushed_count_tracks_unpersisted_tail() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec(),
        });

        assert_eq!(wal.unflushed_count(), 2);

        wal.flush_all().unwrap();
        assert_eq!(wal.unflushed_count(), 0);

        wal.append(WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec(),
        });
        assert_eq!(wal.unflushed_count(), 1);
    }

    #[test]
    fn flushed_records_expose_only_durable_prefix() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });
        wal.append(WalRecord {
            txn_id: 2,
            payload: b"SET b=2".to_vec(),
        });

        assert!(wal.flushed_records().is_empty());

        wal.flush_all().unwrap();
        wal.append(WalRecord {
            txn_id: 3,
            payload: b"SET c=3".to_vec(),
        });

        let durable = wal.flushed_records();
        assert_eq!(durable.len(), 2);
        assert_eq!(durable[0].txn_id, 1);
        assert_eq!(durable[1].txn_id, 2);
    }

    #[test]
    fn flush_failure_does_not_advance_flushed_records() {
        let mut wal = WalBuffer::default();
        wal.append(WalRecord {
            txn_id: 1,
            payload: b"SET a=1".to_vec(),
        });

        wal.fail_next_flush();
        let _ = wal.flush_all();

        assert!(wal.flushed_records().is_empty());
        assert_eq!(wal.unflushed_count(), 1);
    }
}
