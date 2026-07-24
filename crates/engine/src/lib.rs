use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;
use gpu_db_batching::{BatchItem, DualTriggerBatcher, FlushReason};
use gpu_db_execution::{
    CompoundI32I64ProbeShard, CudaCompoundFoldColumn, CudaDeviceMemoryChunk, CudaDeviceMemoryProof,
    CudaDriverRuntime, CudaFixedPointProjection, CudaFixedPointProjectionKind,
    CudaI32BatchProjectionColumns, CudaI32Comparison, CudaI32EqualAnyProjectSubmission,
    CudaI32I64MultiShardProbePlan, CudaI32I64PointKey, CudaI32IndexProbeDenseSubmission,
    CudaI32Stats, CudaOwnedDeviceMemoryChunk, CudaResidentDeviceMemory,
    CudaResidentDeviceMemoryReadView, DeviceRouter, DeviceTarget, MockGpuRuntime, ResidentElemType,
    RouteDecision, VisibleLocateShard, WriteLocateShard,
};
use gpu_db_metrics::{BatchFlushReason, FallbackReason, RuntimeMetrics, RuntimeMetricsSnapshot};
use gpu_db_observability::{
    ActiveFallbackReason, EngineStatusSnapshot, EngineTelemetrySnapshot, FallbackStatus,
    ReadinessStatus, RelationalResidencyStatus, RelationalResidencyTableStatus,
    RelationalResidentRouteDecisionStatus, ReplicationLagSnapshot, SnapshotStatus, TelemetrySink,
};
use gpu_db_planner::{ExecutionPlan, Planner, PlannerConfig};
use gpu_db_replication::{LocalReplicator, LogReplicator, ReplicatedStateMachine};
use gpu_db_snapshot::{SnapshotCell, SnapshotHandle};
use gpu_db_sql::{
    parse_command, parse_command_allowing_catalog, AclRelationKind, AddCheckConstraint,
    AddForeignKey, AddUniqueConstraint, AlterRoleLogin, ColumnDef, ColumnDefault, Command,
    CommentTarget, CopyColumn, CopyFromStdin, CreateDatabase, CreateDomain, CreateExtension,
    CreateIndex, CreateMaterializedView, CreatePublication, CreateRole, CreateSchema,
    CreateSequence, CreateSubscription, CreateTable, CreateTablespace, CreateView,
    DatabasePrivilege, Decimal128, Delete, DropConstraint, DropDatabase, DropDomain, DropExtension,
    DropIndex, DropMaterializedView, DropPublication, DropRole, DropSchema, DropSequence,
    DropSubscription, DropTable, DropTablespace, DropView, FunctionPrivilege, GroupedAggKind,
    GroupedAggregate, Insert, ParseError, PreparedCatalogProgram, PreparedCommand,
    PublicationTarget, RefreshMaterializedView, RenameColumn, RenameConstraint, RenameDatabase,
    RenameFunction, RenameIndex, RenameMaterializedView, RenameRole, RenameSequence, RenameTable,
    RenameTablespace, RenameView, SchemaPrivilege, Select, SelectFilterOp, SelectFunction,
    SelectLiteral, SelectProjection, SequenceNextVal, SequenceSetVal, SqlType, SqlValue,
    TablePrivilege, TablespacePrivilege, TransactionCharacteristics, TruncateTable, Update,
    NUMERIC_DEFAULT_PRECISION, PROJECTION_WILDCARD_SENTINEL,
};
#[cfg(test)]
use gpu_db_storage::TupleVersion;
use gpu_db_storage::{
    InMemoryTupleStore, NewTuple, PruneStats, StorageError, TupleId, TupleStore,
    Visibility as StorageVisibility,
};
use gpu_db_txn::{TxnError, TxnManager, TxnState};
use gpu_db_types::{CommitToken, EngineError, Index, LogEntry, Role, SnapshotMeta, Term, TxnId};
use gpu_db_wal::{
    append_wal_archive_segment_with_timestamps, apply_wal_archive_retention_from_txn,
    apply_wal_archive_retention_to_timestamp_micros, apply_wal_archive_retention_to_txn,
    apply_wal_archive_timeline_prune, export_wal_archive_object_backup,
    fork_wal_archive_timeline_to_timestamp_micros, fork_wal_archive_timeline_to_txn,
    plan_wal_archive_retention_from_txn, plan_wal_archive_retention_to_timestamp_micros,
    plan_wal_archive_retention_to_txn, plan_wal_archive_timeline_prune, read_wal_archive,
    read_wal_archive_timeline, read_wal_archive_timeline_registry,
    read_wal_archive_to_timestamp_micros, read_wal_archive_to_txn, read_wal_checkpoint,
    read_wal_segment, recover_wal_segment, register_wal_archive_timeline,
    restore_wal_archive_object_backup, select_wal_archive_timeline, write_wal_archive_timeline,
    write_wal_archive_with_timestamps, write_wal_control_file, write_wal_segment,
    WalArchiveManifest, WalArchiveObjectBackup, WalArchiveRecordTimestamp, WalArchiveRetentionPlan,
    WalArchiveTimeline, WalArchiveTimelineBranch, WalArchiveTimelinePrunePlan,
    WalArchiveTimelineRegistry, WalArchiveTimelineSelection, WalBuffer, WalCheckpointMeta,
    WalControlFile, WalDurability, WalGroupCommitStats, WalRecord,
};

mod rel_exec_catalog;
pub(crate) use rel_exec_catalog::*;
mod rel_exec_helpers;
pub(crate) use rel_exec_helpers::*;
mod wal_binary;
pub(crate) use wal_binary::*;
mod write_path;
pub(crate) use write_path::*;
mod mvcc_read_model;
pub use mvcc_read_model::*;
mod relational_model;
pub use relational_model::*;
#[cfg(test)]
mod mvcc_read_exec;
#[cfg(test)]
pub(crate) use mvcc_read_exec::*;
mod engine_state;
pub use engine_state::*;
mod resident_storage;
pub(crate) use resident_storage::*;
mod resident_route;
pub(crate) use resident_route::*;
mod engine_catalog;
pub use engine_catalog::{CopyTargetProof, TransactionCopyTargetOrigin};
mod engine_commit;
mod engine_commit_coordinator;
mod engine_commit_residency;
mod engine_ddl_acl;
mod engine_ddl_alter;
mod engine_ddl_objects;
mod engine_ddl_pubsub_role;
mod engine_ddl_table;
mod engine_dml_concurrent;
pub use engine_dml_concurrent::DmlExecutionResult;
mod engine_dml_intent;
mod engine_durability;
mod engine_intent_lanes;
pub use engine_dml_intent::{
    CoveredDeleteRoute, CoveredInsertRoute, CoveredUpdateRoute, IntentTicket, SynchronousCommit,
};
mod engine_dml_prepare;
pub(crate) use engine_dml_prepare::InsertPrepareValidation;
mod engine_expr;
mod engine_expr_ir;
mod engine_introspection;
mod engine_join_ir;
mod engine_lifecycle;
mod engine_mutation_admission;
pub use engine_mutation_admission::{
    CopyMutationRequest, MutationRequest, PredeclaredOperationResult, PredeclaredTransaction,
    PredeclaredTransactionResult, TransactionAdmissionResult, TransactionClass, TransactionRequest,
    TransactionResources,
};
pub use gpu_db_sql::{TransactionAccessMode, TransactionIsolation};
mod engine_mvcc_dispatch;
mod engine_prepared;
pub use engine_prepared::PreparedCommandDescription;
mod engine_prepared_transaction;
pub use engine_prepared_transaction::{
    BoundPreparedTransactionRoute, PreparedTransactionClassAdmissions, PreparedTransactionRoute,
};
mod engine_residency;
mod engine_resident_probe;
mod engine_result_frame;
mod engine_result_sort;
mod engine_retained_read;
pub use engine_retained_read::{
    RelationalCompoundI32I64PointReadParam, RelationalCompoundI32I64PointReadTemplate,
    RelationalResidentIndexPublication, RelationalResidentIndexPublicationEntry,
};
mod engine_select_bind;
mod engine_select_exec;
mod engine_sql_pg;
mod engine_streaming_exec;
mod engine_transaction_catalog;
mod engine_transaction_commit;
mod engine_transaction_delta;
mod engine_transaction_reset;
mod table_access;
use table_access::{TableAccessLease, TableAccessRegistry};
mod engine_wal_archive;
mod engine_write_apply;

#[derive(Debug, Default)]
pub struct KvStateMachine {
    pub applied: Vec<Vec<u8>>,
    pub kv: BTreeMap<String, String>,
}

impl ReplicatedStateMachine for KvStateMachine {
    fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError> {
        self.applied.push(entry.payload.to_vec());
        if let Some(cmd) = Engine::decode_engine_command(&entry.payload)? {
            match cmd {
                Command::SetKv { key, value } => {
                    self.kv.insert(key, value);
                }
                Command::DeleteKv { key } => {
                    self.kv.remove(&key);
                }
                Command::Begin { .. }
                | Command::Commit { .. }
                | Command::Rollback { .. }
                | Command::Flush
                | Command::ResetAll
                | Command::ShowTransactionIsolation
                | Command::SetRole { .. }
                | Command::SessionControl { .. }
                | Command::PreparedCatalog(_)
                | Command::GetKv { .. }
                | Command::CreateSchema(_)
                | Command::DropSchema(_)
                | Command::CreateDatabase(_)
                | Command::DropDatabase(_)
                | Command::RenameDatabase(_)
                | Command::CreateTablespace(_)
                | Command::DropTablespace(_)
                | Command::RenameTablespace(_)
                | Command::CreateTable(_)
                | Command::AddPrimaryKey(_)
                | Command::AddUniqueConstraint(_)
                | Command::AddCheckConstraint(_)
                | Command::AddForeignKey(_)
                | Command::AddColumn(_)
                | Command::RenameTable(_)
                | Command::RenameColumn(_)
                | Command::RenameConstraint(_)
                | Command::DropColumn(_)
                | Command::DropConstraint(_)
                | Command::CreateIndex(_)
                | Command::RenameIndex(_)
                | Command::CreateView(_)
                | Command::RenameView(_)
                | Command::CreateMaterializedView(_)
                | Command::RefreshMaterializedView(_)
                | Command::RenameMaterializedView(_)
                | Command::CreateFunction(_)
                | Command::RenameFunction(_)
                | Command::DropFunction(_)
                | Command::SelectFunction(_)
                | Command::SelectLiteral(_)
                | Command::CreateExtension(_)
                | Command::DropExtension(_)
                | Command::CreateSequence(_)
                | Command::CreateDomain(_)
                | Command::SequenceNextVal(_)
                | Command::SequenceCurrVal(_)
                | Command::SequenceSetVal(_)
                | Command::RenameSequence(_)
                | Command::CreatePublication(_)
                | Command::DropPublication(_)
                | Command::CreateSubscription(_)
                | Command::DropSubscription(_)
                | Command::CreateRole(_)
                | Command::DropRole(_)
                | Command::RenameRole(_)
                | Command::AlterRoleLogin(_)
                | Command::DropTable(_)
                | Command::TruncateTable(_)
                | Command::DropIndex(_)
                | Command::DropView(_)
                | Command::DropMaterializedView(_)
                | Command::DropSequence(_)
                | Command::DropDomain(_)
                | Command::GrantTable(_)
                | Command::RevokeTable(_)
                | Command::GrantSchema(_)
                | Command::RevokeSchema(_)
                | Command::GrantDatabase(_)
                | Command::RevokeDatabase(_)
                | Command::GrantTablespace(_)
                | Command::RevokeTablespace(_)
                | Command::GrantFunction(_)
                | Command::RevokeFunction(_)
                | Command::GrantDefaultTablePrivileges(_)
                | Command::RevokeDefaultTablePrivileges(_)
                | Command::AlterColumnDefault(_)
                | Command::CommentOn(_)
                | Command::Insert(_)
                | Command::Delete(_)
                | Command::Update(_)
                | Command::Select(_) => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecuteError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Txn(#[from] TxnError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// A retryable Snapshot-Isolation write-write serialization conflict (write-half MVCC, Stage 4):
    /// a concurrent transaction committed a write to a key in this transaction's write-set after this
    /// transaction took its read snapshot, so first-committer-wins aborts this one. The transaction
    /// made NO durable or visible change (it never published); the caller should retry it with a
    /// fresh snapshot. The façade maps this to a retryable class-40 error
    /// (`ErrorCategory::Serialization`).
    #[error("could not serialize access due to concurrent update: {0}")]
    Serialization(String),
    #[error("command is not readable via execute_read_text: {0}")]
    NonReadCommand(&'static str),
    #[error("{0}")]
    Unsupported(String),
    #[error("relation \"{0}\" does not exist")]
    UndefinedRelation(String),
    #[error("column \"{0}\" does not exist")]
    UndefinedColumn(String),
    #[error("could not determine data type of parameter ${0}")]
    IndeterminateParameterType(usize),
    #[error("datatype mismatch: {0}")]
    DatatypeMismatch(String),
    #[error("invalid request shape: {0}")]
    InvalidRequest(String),
    /// A bounded foreground admission queue could not reserve its class-specific stage credits
    /// before the pre-WAL service deadline. No transaction state, sequence, WAL, or publication
    /// effect exists; callers may retry with ordinary overload backoff.
    #[error("insufficient resources: {0}")]
    ResourceExhausted(String),
    /// The durable/log boundary was crossed, but terminal commit publication could not be proven
    /// to the caller. The engine is fail-stopped and restart recovery owns the outcome.
    #[error("indeterminate transaction outcome: {0}")]
    Indeterminate(String),
}

impl ExecuteError {
    pub fn is_unique_violation(&self) -> bool {
        matches!(self, Self::Engine(EngineError::UniqueViolation(_)))
    }
}

impl ExecuteError {
    /// Whether this error is a retryable Snapshot-Isolation serialization conflict (the caller may
    /// retry the whole statement against a fresh snapshot). Lets adapters classify the retryable
    /// class-40 case without string-matching the message.
    pub fn is_serialization_conflict(&self) -> bool {
        matches!(self, ExecuteError::Serialization(_))
    }

    pub fn is_indeterminate(&self) -> bool {
        matches!(self, ExecuteError::Indeterminate(_))
    }

    /// Whether this error is the resident-route "the table's GPU residency was invalidated out from
    /// under this statement" case (write-half MVCC, Stage 4): a concurrent committer tombstoned the
    /// table's device-memory `SnapshotCell` (`publish(None)`) between this statement's resident-route
    /// plan (which saw it published) and the GPU probe (which loaded the now-`None` cell). It is NOT a
    /// genuine GPU/CUDA failure. It is classified precisely so dispatch can decline through the
    /// fail-loud GPU-required boundary without masking a real device error.
    fn is_residency_invalidated(&self) -> bool {
        matches!(
            self,
            ExecuteError::Engine(EngineError::ApplyFailed(msg))
                if msg.contains(RESIDENT_DEVICE_MEMORY_MISSING)
                    || (msg.contains(RESIDENT_SHARD_PREFIX)
                        && (msg.contains(RESIDENT_SHARD_INVALID)
                            || msg.contains(RESIDENT_SHARD_MEMORY_MISSING)))
        )
    }
}

/// The substring every GPU resident-route probe uses when a table's device-memory cell is `None`
/// (tombstoned/never-populated). Used to detect the residency-invalidated-mid-statement case.
const RESIDENT_DEVICE_MEMORY_MISSING: &str = "has no retained resident device memory";

/// W0c (audit B2): the SHARDED unified-source errors (`build_sharded_unified_exec_source`'s
/// `source_for`: "resident shard {id} is invalid" / "resident shard {id} has no retained device
/// memory") are the same residency-invalidated-mid-statement case in per-shard form — the route
/// plan accepted an earlier generation and a concurrent commit flagged the shards before the
/// executor's own load. W0 made that window COMMON under OLTP write load (every concurrent
/// invalidation now flags descriptors), and without these matches a racing reader got a hard
/// classified route decline. Matched as (prefix AND suffix) so a genuine device/CUDA error is never
/// masked.
const RESIDENT_SHARD_PREFIX: &str = "resident shard ";
const RESIDENT_SHARD_INVALID: &str = " is invalid";
const RESIDENT_SHARD_MEMORY_MISSING: &str = " has no retained device memory";

#[derive(Debug, Clone)]
struct PendingMutation {
    txn_id: u64,
    payload: std::sync::Arc<[u8]>,
    /// Queue ownership, not the enqueue call, defines the lifetime of a table access. The lease
    /// therefore survives time/count batching, retries, cancellation, and terminal apply/drop.
    table_access: Option<Arc<TableAccessLease>>,
}

/// D3b (write-path assessment / scalability ledger #7) — group-commit flush coordination for the
/// concurrent DML path. Committers validate, append, propose, and APPLY under the commit_mutex,
/// then LEAVE the critical section and make their record durable here: the first arrival becomes
/// the FLUSHER (one `flush_all` = one fsync covering every record appended so far — the group),
/// later arrivals wait on the condvar and share that fsync. `committed_seq` is published only
/// after a committer's record is covered by the durable frontier, so WAL-before-visibility is
/// unchanged — the fsync just moved OFF the validate/apply critical path, so concurrent
/// committers overlap their WAL waits instead of serializing fsync-per-commit.
struct GroupFlushState {
    coord: Mutex<GroupFlushCoord>,
    cv: std::sync::Condvar,
    /// Mirror of the WAL's flushed record count (the durable frontier), maintained by group
    /// flushers so waiters can check durability without re-taking the commit_mutex. A stale
    /// (low) value is safe: the waiter becomes a flusher and `flush_all` with nothing unflushed
    /// is a no-op that refreshes the mirror.
    durable_records: std::sync::atomic::AtomicUsize,
    /// E1 step 2 — set once at construction when the WAL's durability backend supports MULTIPLE
    /// concurrent group flushes in flight (the FUA fence pool). When true, `wait_group_durable`
    /// takes the CONCURRENT path: it skips the single-flusher election (`flusher_active`) and lets
    /// every committer run `begin_group_flush` + `job.commit()` at once — the WAL's ticket gate
    /// keeps frames ordered and the fence pool pipelines durability. The serial backend leaves this
    /// false and keeps electing one flusher (its `io_in_flight` slot admits at most one IO).
    concurrent_durability: bool,
    /// E1 step 2 — lock-free sticky mirror of `GroupFlushCoord::failed` for the concurrent path's
    /// POLLING waiters (a committer whose records another thread already published spins on
    /// `durable_records` with no per-commit wakeup). A flusher sets this before it wedges so a
    /// polling waiter never spins forever behind a failed fence; the authoritative message stays in
    /// `coord.failed`. Unused (always false) on the serial path.
    wedged: std::sync::atomic::AtomicBool,
}

impl Default for GroupFlushState {
    fn default() -> Self {
        Self {
            coord: Mutex::new(GroupFlushCoord::default()),
            cv: std::sync::Condvar::new(),
            durable_records: std::sync::atomic::AtomicUsize::new(0),
            concurrent_durability: false,
            wedged: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

#[derive(Default)]
struct GroupFlushCoord {
    /// A flusher currently owns the fsync; arrivals wait for its result instead of stacking a
    /// second concurrent fsync on the same segment.
    flusher_active: bool,
    /// Sticky wedge: a group fsync failed AFTER its members' deltas were applied (they can never
    /// be published, and nothing later may publish over them). See
    /// [`Engine::wait_group_durable`] for the failure-semantics rationale.
    failed: Option<String>,
}

pub struct Engine {
    /// The commit-critical mutable substate — the replicator (commit-`Index` oracle), the WAL, the
    /// per-txn commit timestamps, and the recent-commits conflict ledger — bundled behind ONE mutex
    /// that IS the **commit_mutex** (write-half MVCC, Stage 4). The concurrent DML commit path locks
    /// it for its short critical section (validate → claim `commit_seq` → append WAL → synchronously
    /// apply and install terminal status), then leaves the lock for registered group durability and
    /// the sole coordinator's contiguous publication. Serialized/batch callers may perform their
    /// durability wait synchronously, but they report the same durable-and-applied index to that
    /// publication coordinator. Prepare runs off-lock and readers stay lock-free. Code that already
    /// holds `&mut self` (serialized DDL apply, recovery, checkpoint/snapshot admin) reaches it via
    /// `commit_state_mut()` (a zero-cost `Mutex::get_mut`, no actual locking).
    commit: Mutex<CommitState>,
    /// Sole logical visibility owner. Commit strategies report exact durable-and-applied indices;
    /// this join alone advances the contiguous reader-visible prefix.
    commit_publication: engine_commit_coordinator::CommitPublicationCoordinator,
    /// Sole accepted-but-not-terminal transaction-id registry across queued batch and optimized
    /// intent strategies. The canonical terminal authority remains `CommitState::transaction_status`;
    /// this shared registry only closes the admission-to-WAL interval and is removed on either
    /// clean rejection or terminal installation.
    pending_transaction_claims: Arc<Mutex<HashMap<TxnId, gpu_db_wal::CanonicalDigest>>>,
    /// Sticky fail-stop independent of mutex poisoning. A transaction whose WAL record crossed
    /// the durable/replicated boundary but could not be fully installed must never return to
    /// ordinary service: restart recovery is the only safe continuation.
    commit_path_wedged: Arc<AtomicBool>,
    /// Engine-local fault injection for explicit-transaction post-durable apply tests. Keeping this
    /// per instance prevents parallel engines from stealing one another's one-shot failure.
    #[cfg(test)]
    fail_next_transaction_post_durable_apply: AtomicBool,
    /// One-shot deterministic seam after WAL durability and private-generation retirement but
    /// before canonical apply, used to prove publication credit cannot be stolen by another
    /// allocator in that exact window.
    #[cfg(test)]
    transaction_post_durable_hook: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// In-flight statements' and explicit transactions' read snapshots (write-half MVCC, Stage 4),
    /// for the oldest-active GC boundary. Autocommit prepare registers a scalar guard; explicit
    /// BEGIN registers one keyed, generation-owned catalog/MVCC/GPU-resource bundle.
    active_snapshots: std::sync::Arc<Mutex<ActiveSnapshots>>,
    /// Stable-table-identity access guards. Explicit transactions retain their lease through
    /// terminal control; statement snapshots retain one only for the statement. A table reset
    /// upgrades its owner to exclusive and deterministic incompatible acquisition returns 40001.
    table_access: Arc<TableAccessRegistry>,
    /// GPU bytes owned only by explicit-transaction private generations, by device. These
    /// allocations are not present in the globally published residency maps, so they must be
    /// charged separately against the same admission budget until the transaction snapshot drops.
    transaction_private_gpu_bytes: Arc<Mutex<BTreeMap<u16, u64>>>,
    /// Exact payload, sidecar, and index allocations retained by explicit transaction snapshots.
    /// The allocation-identity registry unions current global ownership with transaction lifetime
    /// ownership for hard-budget accounting without double counting.
    transaction_retained_gpu_allocations: Arc<Mutex<TransactionRetainedGpuAccount>>,
    /// Lock-free mirror of the replicator role (0=Leader, 1=Follower, 2=Candidate), updated only
    /// by the rare `become_*` transitions. The per-statement leader check (`repl_role`) used to
    /// lock the commit_mutex for this one field read — measured to CONVOY every "off-lock"
    /// prepare behind the wave sequencer's mutex hold (3µs → 1.4ms per prepare at 32 writers).
    repl_role_mirror: std::sync::atomic::AtomicU8,
    /// The deterministic commit-wave queue (ledger #6 / ADR-009 host spine): concurrent DML
    /// commits are sequenced in WAVES by one promoted sequencer per wave — one commit_mutex hold,
    /// one group fsync, one publish per wave — instead of a per-commit critical section.
    commit_wave: engine_dml_concurrent::CommitWaveState,
    /// Group-commit flush coordination for the concurrent DML path (D3b): the commit fsync runs
    /// OUTSIDE the commit_mutex so concurrent committers share one fsync per group. Lock order:
    /// `group_flush.coord` may be taken only when the commit_mutex is NOT held; a flusher takes
    /// the commit_mutex briefly INSIDE (coord → commit), never the reverse.
    group_flush: GroupFlushState,
    /// E2.5b-2 — N-lane intent commit pipeline state (`GPU_DB_INTENT_LANES>=2` on a durable
    /// engine); `None` keeps every existing path byte-identical. Arc: lane pumps hold clones.
    intent_lanes: Option<std::sync::Arc<engine_intent_lanes::IntentLaneState>>,
    /// The lock-free read-path state, shared by value-`Arc` with the concurrent-dispatch façade so
    /// reads and the concurrent-DML path reach `mvcc` / `committed_seq` / resident device-memory /
    /// route telemetry WITHOUT the engine `RwLock` (lock-free read path, write-half MVCC). A holder
    /// of `&mut Engine` (serialized DDL) still only ever gets `&ReadState` through this `Arc`, and
    /// every field is interior-mutable, so a reader and a DDL writer never alias the same byte.
    read_state: Arc<ReadState>,
    /// The DDL-only catalog working state, bundled behind ONE mutex that IS the **catalog latch**
    /// (lock-free read path, write-half MVCC). It holds the working catalog/views/matviews/functions
    /// maps (which lock-free readers consult ONLY via the *published* `read_state.catalog` snapshot,
    /// never these working copies) plus the DDL-only maps the read path never consults at all
    /// (sequences / domains / publications / subscriptions / roles / databases / tablespaces / acl /
    /// comments / the oid+column-id allocators / the resident-cache admission accounting). DDL already
    /// serializes, so one mutex is correct; lock-free readers never touch it and the concurrent-DML
    /// commit path never touches it either. Lock order is fixed: `commit` (commit_mutex) FIRST, then
    /// `catalog_latch`. Code that already holds `&mut self` (construction / serialized DDL / recovery /
    /// residency admin) reaches it lock-free via `Mutex::get_mut` (`ddl_catalog_mut()`); the `&self`
    /// DDL apply path locks it (`ddl_catalog()`).
    catalog_latch: Mutex<DdlCatalogState>,
    metrics: RuntimeMetrics,
    /// Live admission telemetry for engine-proven prepared transaction service classes, indexed
    /// W1/T8/T32/General. These counters are consumed at admission, so the class is operational
    /// routing state rather than a label attached only to the returned result.
    prepared_transaction_class_admissions: [AtomicU64; 4],
    /// Class-aware, pre-BEGIN stage-credit scheduler for engine-proven prepared W1/T8/T32 work.
    /// General transactions retain the same semantically unbounded operation-count surface but do
    /// not borrow a low-latency class's reserved foreground population.
    prepared_transaction_service: engine_prepared_transaction::PreparedTransactionServiceController,
    /// The pending-mutation group-commit batcher (the legacy KV-SET batching subsystem + FLUSH),
    /// behind its OWN `Mutex` (NOT the commit_mutex) so the `&self` `execute_text` FLUSH path can drain
    /// it without `&mut self` (lock-free read path, write-half MVCC). `apply_batch` drains items under
    /// this lock, RELEASES it, then commits each via `commit_mutation` (which takes the commit_mutex
    /// separately — so the batcher lock is never held across a commit, no lock-order coupling).
    batcher: Mutex<DualTriggerBatcher<PendingMutation>>,
    planner: Planner,
    router: DeviceRouter<MockGpuRuntime>,
    // Lazily-probed CUDA runtime, behind a OnceLock so the probe getter is `&self`
    // (the read path lazily initializes it — P1-M3 step 3c).
    cached_cuda_probe_runtime: OnceLock<CudaDriverRuntime>,
    /// STRATA S-B: when true, a committing mutation maintains its tables as GPU-resident authority.
    /// Default on; a maintenance failure fails the commit path rather than acknowledging a host-only
    /// relational image. Interior-mutable (`&self`).
    auto_admit_on_commit: AtomicBool,
    /// ADR-009 R1: when true, a resident int4 unique-key equality point lookup probes a GPU hash index
    /// (built lazily from the resident column, cached per generation) instead of a full-scan kernel —
    /// the per-batch device cost drops O(rows)→O(1). DEFAULT ON (user 2026-06-29: lpb chosen over the wave
    /// engine — the production read path); falls back to the scan whenever the column is non-unique or the
    /// index cannot be built. (Was `wave_engine_enabled` — renamed after the wave retirement; it always gated
    /// the INDEX, never the persistent wave.) Interior-mutable (`&self`), read on the read path.
    index_probe_enabled: AtomicBool,
    /// DECISIONS "lpb read levers" #1: when true, the lpb unique index probe uses the DENSE-emit kernel
    /// (thread `i` -> slot `i`, no atomic, no needle_indices/row_indices/count; host compacts sequentially)
    /// instead of the atomic-compaction kernel. Byte-identical; DEFAULT ON (user 2026-06-29: strict win
    /// `>= b4096`, audit SHIP) — set false to A/B against the atomic kernel. Only the unique index route
    /// honors it — the non-unique scan always keeps the atomic kernel. Interior-mutable.
    dense_index_probe_enabled: AtomicBool,
    /// Billions-of-rows scaling (segmented layout, S-d1): when true, a table is admitted as a SEGMENTED
    /// shard list (sealed shards + one bounded open shard) routed through the sharded resident read path,
    /// instead of one capacity-padded unified buffer that caps at ~536M rows and re-admits O(table). DEFAULT
    /// ON; this flag remains an A/B/test lever. Interior-mutable.
    shard_residency_enabled: AtomicBool,
    /// Sub-slice 3b: route a shard-resident int4 UNIQUE-key equality POINT lookup through the CROSS-SHARD PK
    /// INDEX (cached hash+bloom `locate`) so the sharded read gathers ONLY the located shard(s) instead of
    /// every zone-map-non-excluded shard. DEFAULT ON, nested under `shard_residency_enabled` (the sharded read
    /// falls back to the existing zone-map scan + recompaction when disabled — byte-identical). The A/B lever
    /// for the membership-pruning win that lets the shard path stay O(1) even when zone-maps degrade under
    /// UPDATE key-scatter (scalability-ledger #4/#8). Interior-mutable (the read path reads it).
    shard_index_probe_enabled: AtomicBool,
    /// lpb-for-shards wiring: admit a shard-resident int4 point-lookup BATCH into the facade point-lookup
    /// batcher and serve it via the batched cross-shard gather (`submit_sharded_point_lookups_batched`)
    /// instead of degrading to per-query single-flight. DEFAULT ON (nested under shard residency): OFF =>
    /// `submit_sharded_point_lookups_batched` returns `None` and the batcher keeps its existing behavior
    /// (byte-identical). The A/B lever that LANDS the ~310x batched throughput on real workloads. Interior-mutable.
    shard_batched_point_read_enabled: AtomicBool,
    /// S-d2c: target row count per shard. When the open shard reaches it, an append SEALS the open shard
    /// (immutable) and ROLLS OVER to a fresh open shard (O(rows), never the O(table) re-admit), so a table
    /// grows as bounded shards to billions of rows. Caps the admit headroom + sizes a rollover shard.
    /// Default 4M (seals in ~3ms, ~250 shards/1B per the admit-scaling measurement); settable small in
    /// tests. Interior-mutable.
    /// W5a: covered inserts log RESOLVED BINARY WAL records (decode+install replay) instead of
    /// SQL text. Default OFF until the replay-differential burn-in flips it.
    binary_wal_records_enabled: std::sync::atomic::AtomicBool,
    /// TYPE-COVERAGE track 2 slice 2: sharded admission includes i64-SECTION columns
    /// (Int8/Timestamp) alongside the i32 sections — the first non-i32 shard section.
    /// DEFAULT ON (the 2026-07-03 flip). Kill switch -> int8-bearing tables admit
    /// single-buffer (the pre-slice layout).
    shard_int8_section_enabled: std::sync::atomic::AtomicBool,
    /// E2.5c 2M+ push (b): FUSED merged-apply device pass (one staging HtoD + one launch for
    /// column scatter + created_by/row-id stamps + PK index insert). Default ON (measured
    /// best-of-3 sustained 1.65M vs 1.41M unfused); always on (the unfused arm is the ineligible-shape fallback),
    /// settable per engine.
    fused_apply_enabled: std::sync::atomic::AtomicBool,
    /// M1 design B (wave-time batched validation): eligible INSERTs' PK-unique check is DEFERRED
    /// from the off-lock prepare to the wave sequencer, which batches the whole wave's PK needles
    /// into ONE device locate (the amortization win: launch cost is flat vs batch size). Default
    /// OFF. Kill switch -> per-item off-lock validation.
    device_write_locate_wave_batch_enabled: std::sync::atomic::AtomicBool,
    auto_vacuum_enabled: std::sync::atomic::AtomicBool,
    tombstone_churn_threshold_override: std::sync::atomic::AtomicU64,
    /// S-d2c: the target row count per shard (the rollover/seal threshold; default 4M). Settable
    /// small in tests. Interior-mutable.
    shard_size_target: std::sync::atomic::AtomicUsize,
}

/// The DDL-only catalog working state, serialized behind the engine's **catalog latch**
/// (`Engine::catalog_latch`) — see that field's doc. It bundles every `Engine` catalog field that is
/// mutated by DDL and is NOT interior-mutable: the working catalog maps the read path consults only
/// through the *published* snapshot (`relational_catalog`/`relational_views`/
/// `relational_materialized_views`/`relational_functions`), plus the DDL-only maps the read path never
/// consults at all. DDL serializes (one writer), so holding the latch makes a DDL's working-map
/// mutation + its published-snapshot publish atomic w.r.t. another DDL; lock-free readers and the
/// concurrent-DML path never take this latch.
#[derive(Debug, Clone)]
struct DdlCatalogState {
    relational_catalog: BTreeMap<String, RelationalTable>,
    relational_views: BTreeMap<String, RelationalView>,
    relational_materialized_views: BTreeMap<String, RelationalMaterializedView>,
    relational_functions: BTreeMap<String, RelationalFunction>,
    relational_sequences: BTreeMap<String, RelationalSequence>,
    relational_domains: BTreeMap<String, RelationalDomain>,
    relational_publications: BTreeMap<String, RelationalPublication>,
    relational_subscriptions: BTreeMap<String, RelationalSubscription>,
    relational_roles: BTreeMap<String, RelationalRole>,
    relational_databases: BTreeMap<String, RelationalDatabase>,
    relational_tablespaces: BTreeMap<String, RelationalTablespace>,
    relational_public_schema_exists: bool,
    relational_public_schema_implicit: bool,
    relational_schema_acl: BTreeMap<String, BTreeSet<SchemaPrivilege>>,
    relational_default_table_acl: BTreeMap<String, BTreeSet<TablePrivilege>>,
    relational_comments: BTreeMap<RelationalCommentTarget, String>,
    // The value-index and the relational row-id allocator are now folded into `mvcc`
    // (per-table `TableVersionData::value_index` + `MvccData::next_row_id`).
    relational_resident_cache: RelationalResidentCache,
    /// Sole `pg_class.oid` allocation authority for tables, indexes, views, sequences, and every
    /// other relation-shaped catalog object.
    relational_next_oid: u32,
    /// Recovery-only cursor for stable identities synthesized for pre-PRODUCT-001 indexes.  It is
    /// initialized above every low-range OID the complete historical prefix can allocate, starts at
    /// 20,000 for ordinary databases, and is retired into `relational_next_oid` at the first
    /// current-format command.  It is never a live/product allocation path.
    legacy_recovery_next_index_oid: u32,
    legacy_recovery_index_oids_assigned: bool,
    /// Set only after recovery has inspected the complete durable prefix.  Legacy index
    /// identities must never be synthesized from a partial chunk whose unseen suffix can still
    /// allocate a low-range `pg_class` OID.
    legacy_recovery_floor_prepared: bool,
    index_oid_epoch_current: bool,
    relational_next_column_id: u32,
}

const FIRST_LEGACY_RECOVERY_INDEX_OID: u32 = 20_000;
const MAX_CATALOG_OID: u32 = i32::MAX as u32;

/// Exact `pg_class`-style relation kind for the shared public-schema name authority. Domains and
/// functions live in distinct PostgreSQL namespaces; table-backed indexes do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PgClassRelationKind {
    Table,
    Index,
    View,
    MaterializedView,
    Sequence,
}

fn resolve_pg_class_relation_kind(
    name: &str,
    table: bool,
    index_count: usize,
    view: bool,
    materialized_view: bool,
    sequence: bool,
) -> Result<Option<PgClassRelationKind>, EngineError> {
    if index_count > 1 {
        return Err(EngineError::Durability(format!(
            "catalog contains multiple indexes named \"{name}\""
        )));
    }
    let candidates = [
        table.then_some(PgClassRelationKind::Table),
        (index_count == 1).then_some(PgClassRelationKind::Index),
        view.then_some(PgClassRelationKind::View),
        materialized_view.then_some(PgClassRelationKind::MaterializedView),
        sequence.then_some(PgClassRelationKind::Sequence),
    ];
    let mut found = None;
    for candidate in candidates.into_iter().flatten() {
        if found.replace(candidate).is_some() {
            return Err(EngineError::Durability(format!(
                "catalog contains multiple pg_class relations named \"{name}\""
            )));
        }
    }
    Ok(found)
}

impl DdlCatalogState {
    fn pg_class_relation_kind(
        &self,
        name: &str,
    ) -> Result<Option<PgClassRelationKind>, EngineError> {
        resolve_pg_class_relation_kind(
            name,
            self.relational_catalog.contains_key(name),
            self.relational_catalog
                .values()
                .map(|table| {
                    table
                        .indexes
                        .iter()
                        .filter(|index| index.name == name)
                        .count()
                })
                .sum(),
            self.relational_views.contains_key(name),
            self.relational_materialized_views.contains_key(name),
            self.relational_sequences.contains_key(name),
        )
    }

    fn relational_class_oid_in_use(&self, oid: u32) -> bool {
        self.relational_catalog
            .values()
            .any(|table| table.oid == oid || table.indexes.iter().any(|index| index.oid == oid))
            || self
                .relational_views
                .values()
                .any(|relation| relation.oid == oid)
            || self
                .relational_materialized_views
                .values()
                .any(|relation| relation.oid == oid)
            || self
                .relational_functions
                .values()
                .any(|relation| relation.oid == oid)
            || self
                .relational_sequences
                .values()
                .any(|relation| relation.oid == oid)
            || self
                .relational_domains
                .values()
                .any(|relation| relation.oid == oid)
            || self
                .relational_publications
                .values()
                .any(|relation| relation.oid == oid)
            || self
                .relational_subscriptions
                .values()
                .any(|relation| relation.oid == oid)
            || self
                .relational_roles
                .values()
                .any(|relation| relation.oid == oid)
            || self
                .relational_databases
                .values()
                .any(|relation| relation.oid == oid)
            || self
                .relational_tablespaces
                .values()
                .any(|relation| relation.oid == oid)
    }

    /// Sole allocator for every newly admitted relation-shaped identity.
    fn allocate_relational_class_oid(
        &mut self,
        exhausted_message: &'static str,
    ) -> Result<u32, EngineError> {
        let oid = self.relational_next_oid;
        if !self.index_oid_epoch_current
            || oid > MAX_CATALOG_OID
            || self.relational_class_oid_in_use(oid)
        {
            return Err(EngineError::ApplyFailed(exhausted_message.to_string()));
        }
        self.relational_next_oid = oid
            .checked_add(1)
            .ok_or_else(|| EngineError::ApplyFailed(exhausted_message.to_string()))?;
        Ok(oid)
    }

    /// Sequentially assign the compatibility identity that the original PRODUCT-001 migration used
    /// for a pre-slice index.  The cursor has already been placed above the complete old prefix's
    /// relation high-water, so these identities cannot collide with a later historical table.
    fn migrated_legacy_index_oid(&mut self, pending: &BTreeSet<u32>) -> Result<u32, EngineError> {
        if !self.legacy_recovery_floor_prepared {
            return Err(EngineError::Durability(
                "legacy index identity migration has no complete-prefix OID floor".to_string(),
            ));
        }
        if self.index_oid_epoch_current {
            return Err(EngineError::Durability(
                "legacy index catalog record follows the PRODUCT-001 OID migration boundary"
                    .to_string(),
            ));
        }
        let mut candidate = self
            .legacy_recovery_next_index_oid
            .max(FIRST_LEGACY_RECOVERY_INDEX_OID);
        loop {
            if candidate > MAX_CATALOG_OID {
                return Err(EngineError::ApplyFailed(
                    "historical index OID migration exceeds the GPU catalog Int4 domain"
                        .to_string(),
                ));
            }
            if !pending.contains(&candidate) && !self.relational_class_oid_in_use(candidate) {
                self.legacy_recovery_next_index_oid =
                    candidate.checked_add(1).ok_or_else(|| {
                        EngineError::ApplyFailed(
                            "historical index OID migration range is exhausted".to_string(),
                        )
                    })?;
                self.legacy_recovery_index_oids_assigned = true;
                return Ok(candidate);
            }
            candidate = candidate.checked_add(1).ok_or_else(|| {
                EngineError::ApplyFailed(
                    "historical index OID migration range is exhausted".to_string(),
                )
            })?;
        }
    }

    /// One-way transition from replay-only index identities to the sole shared live allocator.
    fn finalize_legacy_index_oid_migration(&mut self) -> Result<(), EngineError> {
        if self.index_oid_epoch_current {
            return Ok(());
        }
        if self.legacy_recovery_index_oids_assigned {
            self.relational_next_oid = self
                .relational_next_oid
                .max(self.legacy_recovery_next_index_oid);
        }
        if self.relational_next_oid > MAX_CATALOG_OID + 1 {
            return Err(EngineError::ApplyFailed(
                "relational catalog OID migration exceeds the GPU catalog Int4 domain".to_string(),
            ));
        }
        self.index_oid_epoch_current = true;
        Ok(())
    }

    fn prepare_legacy_index_oid_recovery_floor(
        &mut self,
        floor: u32,
        has_legacy_prefix: bool,
    ) -> Result<(), EngineError> {
        let exhausted_high_water = MAX_CATALOG_OID + 1;
        if floor > exhausted_high_water || (has_legacy_prefix && floor > MAX_CATALOG_OID) {
            return Err(EngineError::Durability(
                "historical catalog OID prefix exceeds the GPU catalog Int4 domain".to_string(),
            ));
        }
        if self.legacy_recovery_floor_prepared {
            if self.legacy_recovery_index_oids_assigned
                && floor > self.legacy_recovery_next_index_oid
            {
                return Err(EngineError::Durability(
                    "legacy index OID floor changed after recovery assigned identities".to_string(),
                ));
            }
            self.legacy_recovery_next_index_oid = self.legacy_recovery_next_index_oid.max(floor);
            return Ok(());
        }
        self.legacy_recovery_floor_prepared = true;
        if !has_legacy_prefix {
            return Ok(());
        }
        self.index_oid_epoch_current = false;
        self.legacy_recovery_next_index_oid = self
            .legacy_recovery_next_index_oid
            .max(FIRST_LEGACY_RECOVERY_INDEX_OID)
            .max(floor);
        Ok(())
    }
}

/// The commit-critical mutable substate bundled behind the engine's commit_mutex (write-half MVCC,
/// Stage 4). Holding the lock covers the common authoritative claim: validate the write-set, assign
/// a `commit_seq`, append WAL, synchronously apply, and install terminal status. Concurrent callers
/// then use registered off-lock group durability; serialized/batch callers may wait synchronously.
/// Every live strategy reports its exact durable-and-applied index to the sole contiguous
/// publication coordinator, which alone advances `committed_seq`. Code holding `&mut Engine`
/// reaches this state lock-free via `Mutex::get_mut`.
struct CommitState {
    /// Durable ADR-014 lineage copied into every canonical WAL envelope. Recovery replaces the
    /// freshly generated value from the first validated canonical record before replay.
    canonical_identity: gpu_db_wal::CanonicalIdentity,
    canonical_lineage_bound: bool,
    /// Set after the first canonical record is admitted during chunked recovery. Unlike the live
    /// status index, this is solely the one-way legacy-prefix migration barrier.
    canonical_replay_seen: bool,
    /// Non-pruned terminal claim index. The canonical WAL envelope is the durable authority; this
    /// map is its live/recovered lookup index for exact same-id retry resolution.
    transaction_status: HashMap<TxnId, DurableTransactionStatus>,
    /// Most recently applied entry and its exact relational row count. Recovery consumes this
    /// immediately to compare device/engine replay with the canonical terminal marker.
    last_applied_outcome: Option<(Index, u64)>,
    /// The commit-`Index` oracle + log: `propose` assigns the next monotonic `commit_seq` inside the
    /// critical section (Stage 0 unification — `commit_seq == commit Index`).
    repl: LocalReplicator,
    /// The sole live write-ahead log. The authoritative claim appends while holding the commit
    /// mutex; concurrent strategies make the registered record range durable off-lock through
    /// group flush, while serialized/batch strategies may flush synchronously. Publication is
    /// separate and can advance only after the sole coordinator receives the exact durable-and-
    /// applied index.
    wal: WalBuffer,
    /// Per-txn commit timestamps (durable transaction identity → wall-clock micros), for
    /// PITR-by-timestamp lookups. Keyed by the façade txn_id (the durable identity), distinct from
    /// the MVCC `commit_seq`. A HashMap: every consumer is a point lookup or a retain (archive
    /// timestamp export, checkpoint pruning) — nothing reads key order — and the per-commit insert
    /// sits on the wave sequencer's serial cut, where an unbounded BTreeMap's O(log n) insert was
    /// measured as a top per-item cost at millions of retained commits.
    wal_commit_timestamps_micros: HashMap<TxnId, u64>,
    /// O(1) running max of every value ever put into `wal_commit_timestamps_micros`. Commit
    /// timestamps are assigned monotonically (`next_commit_timestamp_micros`), so this is exactly
    /// `wal_commit_timestamps_micros.values().max()` — tracked incrementally to keep the per-commit
    /// timestamp assignment O(1) instead of an O(n) scan of the never-pruned map. Maintained ONLY
    /// via `record_commit_timestamp` so the two can never drift (the invariant the differential
    /// test asserts). `0` means "no commit yet".
    max_commit_timestamp_micros: u64,
    /// The recent-commits conflict ledger (SI write-write, first-committer-wins).
    ledger: RecentCommitsLedger,
    /// The KV replay/state machine (the `SET`/`DELETE`/`GET` key namespace's applied-record log +
    /// in-memory store), moved under the commit_mutex (A.2): the serialized commit path applies KV
    /// records here, and the few `GetKv` reads take a brief `commit_state()` shim. The concurrent DML
    /// path never touches it (it is relational-only), so it adds no contention to that path.
    sm: KvStateMachine,
    /// BEGIN/COMMIT/ROLLBACK transaction bookkeeping, moved under the commit_mutex (A.2): transaction
    /// control runs on the serialized path; the active-count / oldest-txn watermark reads take a brief
    /// `commit_state()` shim.
    txn_manager: TxnManager,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DurableTransactionStatus {
    request_digest: gpu_db_wal::CanonicalDigest,
    outcome: DurableTransactionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DurableTransactionOutcome {
    Committed {
        commit_seq: Index,
        affected_rows: u64,
    },
    AbortedDiscardedOrphan,
}

impl CommitState {
    fn resolve_transaction_retry(
        &self,
        txn_id: TxnId,
        payload: &[u8],
    ) -> Result<Option<CommitToken>, EngineError> {
        let Some(status) = self.transaction_status.get(&txn_id) else {
            return Ok(None);
        };
        let request_digest = gpu_db_wal::canonical_request_digest(payload);
        if request_digest != status.request_digest {
            return Err(EngineError::Durability(format!(
                "transaction id {txn_id} is already durably claimed by a different request"
            )));
        }
        match status.outcome {
            DurableTransactionOutcome::Committed { commit_seq, .. } => {
                Ok(Some(CommitToken { index: commit_seq }))
            }
            DurableTransactionOutcome::AbortedDiscardedOrphan => Err(EngineError::Durability(
                format!("transaction id {txn_id} was durably aborted during crash recovery"),
            )),
        }
    }

    fn record_transaction_status_digest_outcome(
        &mut self,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
        commit_seq: Index,
        affected_rows: u64,
    ) -> Result<(), EngineError> {
        let status = DurableTransactionStatus {
            request_digest,
            outcome: DurableTransactionOutcome::Committed {
                commit_seq,
                affected_rows,
            },
        };
        match self.transaction_status.entry(txn_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(status);
                Ok(())
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                Err(EngineError::Durability(format!(
                    "transaction id {txn_id} already has terminal status {:?}; refusing a second terminal claim {:?}",
                    entry.get(), status
                )))
            }
        }
    }

    fn resolve_transaction_retry_digest_outcome(
        &self,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<Option<(CommitToken, u64)>, EngineError> {
        let Some(status) = self.transaction_status.get(&txn_id) else {
            return Ok(None);
        };
        if request_digest != status.request_digest {
            return Err(EngineError::Durability(format!(
                "transaction id {txn_id} is already durably claimed by a different request"
            )));
        }
        match status.outcome {
            DurableTransactionOutcome::Committed {
                commit_seq,
                affected_rows,
            } => Ok(Some((CommitToken { index: commit_seq }, affected_rows))),
            DurableTransactionOutcome::AbortedDiscardedOrphan => Err(EngineError::Durability(
                format!("transaction id {txn_id} was durably aborted during crash recovery"),
            )),
        }
    }

    /// Record a commit's wall-clock timestamp in the PITR map AND advance the O(1) running max
    /// (`max_commit_timestamp_micros`) in lock-step. EVERY writer of `wal_commit_timestamps_micros`
    /// must go through here so the max can never lag the map — that is the invariant
    /// `next_commit_timestamp_micros` relies on to skip the old O(n) `.values().max()` scan.
    fn record_commit_timestamp(&mut self, txn_id: TxnId, timestamp_micros: u64) {
        self.wal_commit_timestamps_micros
            .insert(txn_id, timestamp_micros);
        self.max_commit_timestamp_micros = self.max_commit_timestamp_micros.max(timestamp_micros);
    }
}

impl Engine {
    /// Allocate an explicit-transaction identity that is unclaimed by every durable or accepted
    /// strategy. Callers hold the commit lock, so terminal status and the shared pending registry
    /// are observed as one admission boundary.
    fn begin_unclaimed_transaction(&self, commit: &mut CommitState) -> Result<TxnId, TxnError> {
        loop {
            let txn = commit.txn_manager.begin()?;
            let pending = self
                .pending_transaction_claims
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let claimed =
                commit.transaction_status.contains_key(&txn.id) || pending.contains_key(&txn.id);
            drop(pending);
            if !claimed {
                return Ok(txn.id);
            }
            commit.txn_manager.rollback(txn.id)?;
        }
    }

    fn resolve_pending_transaction_claim(
        &self,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<bool, EngineError> {
        let claims = self
            .pending_transaction_claims
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(existing) = claims.get(&txn_id) else {
            return Ok(false);
        };
        if *existing != request_digest {
            return Err(EngineError::Durability(format!(
                "transaction id {txn_id} is already pending with a different request"
            )));
        }
        Ok(true)
    }

    /// Reserve the admission-to-WAL interval. Returns false for an exact already-pending request;
    /// callers decide whether that means their own accepted item or an indeterminate retry.
    fn reserve_pending_transaction_claim(
        &self,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<bool, EngineError> {
        let mut claims = self
            .pending_transaction_claims
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match claims.entry(txn_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(request_digest);
                Ok(true)
            }
            std::collections::hash_map::Entry::Occupied(entry)
                if *entry.get() == request_digest =>
            {
                Ok(false)
            }
            std::collections::hash_map::Entry::Occupied(_) => Err(EngineError::Durability(
                format!("transaction id {txn_id} is already pending with a different request"),
            )),
        }
    }

    fn release_pending_transaction_claim(
        &self,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) {
        let mut claims = self
            .pending_transaction_claims
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if claims
            .get(&txn_id)
            .is_some_and(|existing| *existing == request_digest)
        {
            claims.remove(&txn_id);
        }
    }
}

/// The whole-engine value-index key `(table, column, value)`. With the value-index now folded
/// per-table into `TableVersionData` (keyed by [`ColumnValueKey`]), this whole-engine flattened
/// key is only used by test assertions that snapshot the value-index across all tables.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RelationalIndexKey {
    table: String,
    column: String,
    value: String,
}

const PUBLIC_SCHEMA_NAME: &str = "public";
const FIRST_USER_RELATION_OID: u32 = 16_384;
const FIRST_USER_COLUMN_ID: u32 = 1;

impl RelationalColumn {
    fn from_def(
        id: u32,
        table_oid: u32,
        attnum: i16,
        def: ColumnDef,
        type_oid: u32,
        type_size: i16,
    ) -> Self {
        Self {
            id,
            table_oid,
            attnum,
            name: def.name,
            ty: def.ty,
            domain: def.domain,
            default: def.default,
            type_oid,
            type_size,
        }
    }

    pub fn as_column_def(&self) -> ColumnDef {
        ColumnDef {
            name: self.name.clone(),
            ty: self.ty,
            domain: self.domain.clone(),
            default: self.default.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableWalArchiveRetentionWindowPlan {
    pub current_timestamp_micros: u64,
    pub pitr_window_micros: u64,
    pub cutoff_timestamp_micros: u64,
    pub base_txn_id: TxnId,
    pub base_timestamp_micros: u64,
    pub retention_plan: WalArchiveRetentionPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableWalArchiveMaintenancePlan {
    pub retention_window_plan: DurableWalArchiveRetentionWindowPlan,
    pub timeline_prune_plan: WalArchiveTimelinePrunePlan,
}

#[cfg(test)]
mod tests;

/// Commit-wave telemetry accessor: `[waves, items, sequencing_nanos]` (see
/// `engine_dml_concurrent::WAVE_STATS`). Read by the phase-D SLO benchmark.
pub fn engine_dml_concurrent_wave_stats() -> &'static [std::sync::atomic::AtomicU64; 3] {
    &engine_dml_concurrent::WAVE_STATS
}

/// FUSE-recon device-phase timing accessor: `[locate_nanos, append_nanos, index_insert_nanos]`
/// (see `engine_dml_concurrent::WAVE_DEVICE_STATS`; populated only under `GPU_DB_BENCH_DEVPHASE=1`).
pub fn engine_dml_concurrent_wave_device_stats() -> &'static [std::sync::atomic::AtomicU64; 3] {
    &engine_dml_concurrent::WAVE_DEVICE_STATS
}

/// HOST-sequencer per-item phase timing accessor: `[wave_validate, conflict, reresolve, sequence,
/// ledger, apply, invalidate]` nanos (see `engine_dml_concurrent::WAVE_HOST_STATS`; populated only
/// under `GPU_DB_BENCH_HOSTPHASE=1`). The serial work under the commit_mutex — the peak-throughput wall.
pub fn engine_dml_concurrent_wave_host_stats() -> &'static [std::sync::atomic::AtomicU64; 7] {
    &engine_dml_concurrent::WAVE_HOST_STATS
}
