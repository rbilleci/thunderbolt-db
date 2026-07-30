use serde::{Deserialize, Serialize};
use std::fmt;
use std::num::NonZeroI32;
use std::sync::OnceLock;

pub type Term = u64;
pub type Index = u64;
pub type TxnId = u64;

/// Durability implementation that observed a terminal post-handoff fault.
///
/// This remains a fixed value so the committed-write fail-stop path can publish its first fault
/// without allocating a diagnostic string.  Backends that have not yet adopted the fixed-fault
/// route keep their existing compatibility diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DurabilityBackend {
    SerialWal,
    FuaWal,
}

impl fmt::Display for DurabilityBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SerialWal => "serial-wal",
            Self::FuaWal => "fua-wal",
        })
    }
}

/// Exact location in a backend's post-handoff durability state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DurabilityStage {
    PositionalWrite,
    PositionalWriteZero,
    PositionalWriteOverflow,
    PositionalWriteInvariant,
    SyncData,
    FrontierDrift,
    Abandoned,
    DescriptorPoison,
    /// FUA exact-group admission could not retain an immutable physical scatter reservation
    /// before the WAL claim crossed its rollback boundary.
    FuaReservation,
    /// The fixed FUA scatter source or its one-release publication failed after exact handoff.
    FuaScatter,
    /// FUA fence-pool work reported an immutable post-publication failure.
    FuaFence,
    /// The physical cadence controller could not preserve the sealed exact decision.
    FuaController,
    /// The exact FUA publish cursor or durable frontier diverged from the sealed group range.
    FuaFrontier,
    /// An exact FUA group was dropped between preclaim ownership and terminal settlement.
    FuaAbandoned,
    /// The permanently provisioned FUA descriptor/ledger was consumed inconsistently.
    FuaDescriptor,
}

impl fmt::Display for DurabilityStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::PositionalWrite => "positional-write",
            Self::PositionalWriteZero => "positional-write-zero",
            Self::PositionalWriteOverflow => "positional-write-overflow",
            Self::PositionalWriteInvariant => "positional-write-invariant",
            Self::SyncData => "sync-data",
            Self::FrontierDrift => "frontier-drift",
            Self::Abandoned => "abandoned",
            Self::DescriptorPoison => "descriptor-poison",
            Self::FuaReservation => "fua-reservation",
            Self::FuaScatter => "fua-scatter",
            Self::FuaFence => "fua-fence",
            Self::FuaController => "fua-controller",
            Self::FuaFrontier => "fua-frontier",
            Self::FuaAbandoned => "fua-abandoned",
            Self::FuaDescriptor => "fua-descriptor",
        })
    }
}

/// Allocation-free identity of a fail-stop durability fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurabilityFault {
    pub backend: DurabilityBackend,
    pub stage: DurabilityStage,
    /// Platform error value captured at the failing syscall.
    ///
    /// A zero OS error is not a failure. Keeping the optional value niche-backed prevents this
    /// fixed fault from enlarging the ubiquitous `EngineError`/`ExecuteError` success ABI.
    raw_os_error: Option<NonZeroI32>,
    pub segment_id: u64,
    pub group_first_record: u64,
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<DurabilityFault>() <= 24);

impl DurabilityFault {
    pub const fn new(
        backend: DurabilityBackend,
        stage: DurabilityStage,
        raw_os_error: Option<i32>,
        segment_id: u64,
        group_first_record: u64,
    ) -> Self {
        let raw_os_error = match raw_os_error {
            Some(error) => NonZeroI32::new(error),
            None => None,
        };
        Self {
            backend,
            stage,
            raw_os_error,
            segment_id,
            group_first_record,
        }
    }

    pub const fn raw_os_error(self) -> Option<i32> {
        match self.raw_os_error {
            Some(error) => Some(error.get()),
            None => None,
        }
    }
}

impl fmt::Display for DurabilityFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "backend={} stage={} raw_os_error={:?} segment_id={} group_first_record={}",
            self.backend,
            self.stage,
            self.raw_os_error(),
            self.segment_id,
            self.group_first_record
        )
    }
}

/// First-wins fault publication for allocation-free, post-handoff fail-stop paths.
#[derive(Debug, Default)]
pub struct DurabilityPoison {
    first: OnceLock<DurabilityFault>,
}

impl DurabilityPoison {
    pub const fn new() -> Self {
        Self {
            first: OnceLock::new(),
        }
    }

    /// Publish `fault` unless a concurrent path already won. Returns the immutable first fault.
    pub fn install(&self, fault: DurabilityFault) -> DurabilityFault {
        let _ = self.first.set(fault);
        *self
            .first
            .get()
            .expect("DurabilityPoison must contain the fault it just installed or observed")
    }

    pub fn snapshot(&self) -> Option<DurabilityFault> {
        self.first.get().copied()
    }
}

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
    /// A parsed value/default has a concrete source type that cannot be assigned to its target
    /// column type (PostgreSQL 42804).  Keep this distinct from syntax and conversion failures so
    /// protocol façades never have to classify a diagnostic string.
    #[error("datatype mismatch: {0}")]
    DatatypeMismatch(String),
    /// A relational bind resolved a target column name that is absent from the table schema.
    /// Kept typed through the engine boundary so neutral protocol adapters can emit 42703
    /// without inspecting a diagnostic string.
    #[error("column \"{0}\" does not exist")]
    UndefinedColumn(String),
    /// A durable/default regclass target is absent from the catalog.  Keep this distinct from
    /// generic apply failures so neutral protocol adapters can emit PostgreSQL 42P01.
    #[error("relation \"{0}\" does not exist")]
    UndefinedRelation(String),
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
    /// A date or timestamp input failed its type's textual input rules (PostgreSQL 22007).
    #[error("invalid datetime format: {0}")]
    InvalidDatetimeFormat(String),
    /// A syntactically numeric date/time field or timestamp range is not representable (22008).
    #[error("date/time field value out of range: {0}")]
    DatetimeFieldOverflow(String),
    /// A non-datetime scalar input failed its type's textual input rules (PostgreSQL 22P02).
    #[error("invalid text representation: {0}")]
    InvalidTextRepresentation(String),
    /// No comparison operator exists for the resolved operand types (PostgreSQL 42883).
    #[error("operator does not exist: {0}")]
    UndefinedOperator(String),
    #[error("durability failure: {0}")]
    Durability(String),
    #[error("durability fault: {0}")]
    DurabilityFault(DurabilityFault),
    #[error("pending mutation queue overloaded: pending={pending} cap={cap}")]
    MutationQueueOverloaded { pending: usize, cap: usize },
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<EngineError>() <= 32);
