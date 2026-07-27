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
    /// W1a: shared, immutable statement bytes. One allocation at ingress is refcounted through
    /// the WAL record, the replication log, and the commit-wave item (previously three `Vec`
    /// copies per committed statement on the hot path). serde's `rc` feature covers ser/de
    /// (deserialize allocates fresh, as before).
    pub payload: std::sync::Arc<[u8]>,
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
    /// A relational bind resolved a target column name that is absent from the table schema.
    /// Kept typed through the engine boundary so neutral protocol adapters can emit 42703
    /// without inspecting a diagnostic string.
    #[error("column \"{0}\" does not exist")]
    UndefinedColumn(String),
    /// A relational target list named the same column more than once. This is a pre-effect
    /// binding failure (42701), not an internal apply error.
    #[error("column \"{0}\" specified more than once")]
    DuplicateColumn(String),
    #[error("unique constraint violation: {0}")]
    UniqueViolation(String),
    #[error("not-null constraint violation: {0}")]
    NotNullViolation(String),
    #[error("foreign key constraint violation: {0}")]
    ForeignKeyViolation(String),
    #[error("check constraint violation: {0}")]
    CheckViolation(String),
    #[error("numeric value out of range: {0}")]
    NumericValueOutOfRange(String),
    #[error("durability failure: {0}")]
    Durability(String),
    #[error("pending mutation queue overloaded: pending={pending} cap={cap}")]
    MutationQueueOverloaded { pending: usize, cap: usize },
}
