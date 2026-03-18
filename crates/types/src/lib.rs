use serde::{Deserialize, Serialize};

pub type Term = u64;
pub type Index = u64;
pub type TxnId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Leader,
    Follower,
    Candidate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEntry {
    pub term: Term,
    pub index: Index,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitToken {
    pub index: Index,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotMeta {
    pub last_included_index: Index,
    pub last_included_term: Term,
    pub snapshot_id: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("not leader")]
    NotLeader,
    #[error("proposal failed: {0}")]
    ProposalFailed(String),
    #[error("apply failed: {0}")]
    ApplyFailed(String),
    #[error("durability failure: {0}")]
    Durability(String),
}
