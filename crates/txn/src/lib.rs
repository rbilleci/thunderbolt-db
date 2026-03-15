use gpu_db_types::TxnId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    Active,
    Committed,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Txn {
    pub id: TxnId,
    pub state: TxnState,
}

#[derive(Debug, Default)]
pub struct TxnManager {
    next_id: TxnId,
}

impl TxnManager {
    pub fn begin(&mut self) -> Txn {
        self.next_id += 1;
        Txn {
            id: self.next_id,
            state: TxnState::Active,
        }
    }
}
