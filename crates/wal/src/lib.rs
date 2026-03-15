use gpu_db_types::TxnId;

#[derive(Debug, Clone)]
pub struct WalRecord {
    pub txn_id: TxnId,
    pub payload: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct WalBuffer {
    records: Vec<WalRecord>,
    flushed: usize,
}

impl WalBuffer {
    pub fn append(&mut self, rec: WalRecord) {
        self.records.push(rec);
    }

    pub fn flush_all(&mut self) {
        self.flushed = self.records.len();
    }

    pub fn flushed_count(&self) -> usize {
        self.flushed
    }
}
