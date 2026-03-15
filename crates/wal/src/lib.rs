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

    pub fn fail_next_flush(&mut self) {
        self.fail_next_flush = true;
    }
}
