use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;
use gpu_db_batching::{BatchItem, DualTriggerBatcher, FlushReason};
use gpu_db_execution::{
    CudaDeviceMemoryChunk, CudaDeviceMemoryProof, CudaDriverRuntime, CudaI32Comparison,
    CudaI32EqualAnyProjectSubmission, CudaMvccRowBatch, CudaOwnedDeviceMemoryChunk,
    CudaResidentDeviceMemory, CudaResidentDeviceMemoryReadView, DeviceRouter, DeviceTarget,
    FilterOperator, GroupedI64Order, GroupedI64SortColumn, LimitOperator, MockGpuRuntime, Operator,
    PlannedOp, ProjectOperator, RouteDecision, ScanOperator, SortOperator,
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
    AddForeignKey, AddUniqueConstraint, ColumnDef, ColumnDefault, Command, CommentTarget,
    CopyColumn, CopyFromStdin, CreateDatabase, CreateDomain, CreateExtension, CreateIndex,
    CreateMaterializedView, CreatePublication, CreateRole, CreateSchema, CreateSequence,
    CreateSubscription, CreateTable, CreateTablespace, CreateView, DatabasePrivilege, Decimal128,
    Delete, DropConstraint, DropDatabase, DropDomain, DropExtension, DropIndex,
    DropMaterializedView, DropPublication, DropRole, DropSchema, DropSequence, DropSubscription,
    DropTable, DropTablespace, DropView, FunctionPrivilege, Insert, ParseError, PublicationTarget,
    RefreshMaterializedView, RenameColumn, RenameConstraint, RenameDatabase, RenameFunction,
    RenameIndex, RenameMaterializedView, RenameRole, RenameSequence, RenameTable, RenameTablespace,
    RenameView, SchemaPrivilege, Select, SelectFilterOp, SelectFunction, SelectProjection,
    SequenceNextVal, SequenceSetVal, SqlType, SqlValue, TablePrivilege, TablespacePrivilege,
    TruncateTable, Update, NUMERIC_DEFAULT_PRECISION,
};
use gpu_db_storage::{
    InMemoryTupleStore, NewTuple, PruneStats, StorageError, TupleId, TupleStore, TupleVersion,
    Visibility as StorageVisibility,
};
use gpu_db_txn::{TxnError, TxnManager};
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
    read_wal_segment, register_wal_archive_timeline, restore_wal_archive_object_backup,
    select_wal_archive_timeline, write_wal_archive_timeline, write_wal_archive_with_timestamps,
    write_wal_control_file, write_wal_segment, WalArchiveManifest, WalArchiveObjectBackup,
    WalArchiveRecordTimestamp, WalArchiveRetentionPlan, WalArchiveTimeline,
    WalArchiveTimelineBranch, WalArchiveTimelinePrunePlan, WalArchiveTimelineRegistry,
    WalArchiveTimelineSelection, WalBuffer, WalControlFile, WalGroupCommitStats, WalRecord,
};

mod rel_exec_helpers;
pub(crate) use rel_exec_helpers::*;
mod write_path;
pub(crate) use write_path::*;
mod mvcc_read_model;
pub use mvcc_read_model::*;
mod relational_model;
pub use relational_model::*;
mod mvcc_read_exec;
pub(crate) use mvcc_read_exec::*;
mod engine_state;
pub use engine_state::*;
mod resident_storage;
pub(crate) use resident_storage::*;
mod resident_route;
pub(crate) use resident_route::*;
mod engine_catalog;
mod engine_introspection;
mod engine_mvcc_dispatch;
mod engine_residency;
mod engine_resident_probe;
mod engine_retained_read;
mod engine_select_bind;
mod engine_select_exec;
mod engine_wal_archive;

#[derive(Debug, Default)]
pub struct KvStateMachine {
    pub applied: Vec<Vec<u8>>,
    pub kv: BTreeMap<String, String>,
}

impl ReplicatedStateMachine for KvStateMachine {
    fn apply(&mut self, entry: &LogEntry) -> Result<(), EngineError> {
        self.applied.push(entry.payload.clone());
        if let Ok(s) = std::str::from_utf8(&entry.payload) {
            if let Ok(cmd) = parse_command(s) {
                match cmd {
                    Command::SetKv { key, value } => {
                        self.kv.insert(key, value);
                    }
                    Command::DeleteKv { key } => {
                        self.kv.remove(&key);
                    }
                    Command::Begin
                    | Command::Commit { .. }
                    | Command::Rollback { .. }
                    | Command::Flush
                    | Command::ResetAll
                    | Command::SetRole { .. }
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
}

impl ExecuteError {
    /// Whether this error is a retryable Snapshot-Isolation serialization conflict (the caller may
    /// retry the whole statement against a fresh snapshot). Lets adapters classify the retryable
    /// class-40 case without string-matching the message.
    pub fn is_serialization_conflict(&self) -> bool {
        matches!(self, ExecuteError::Serialization(_))
    }

    /// Whether this error is the resident-route "the table's GPU residency was invalidated out from
    /// under this statement" case (write-half MVCC, Stage 4): a concurrent committer tombstoned the
    /// table's device-memory `SnapshotCell` (`publish(None)`) between this statement's resident-route
    /// plan (which saw it published) and the GPU probe (which loaded the now-`None` cell). It is NOT a
    /// genuine GPU/CUDA failure — the engine can transparently re-serve the statement from the CPU
    /// pinned-read path against the current published data generation. Matched on the precise probe
    /// message so a real device error (which carries a different message) is never masked.
    fn is_residency_invalidated(&self) -> bool {
        matches!(
            self,
            ExecuteError::Engine(EngineError::ApplyFailed(msg))
                if msg.contains(RESIDENT_DEVICE_MEMORY_MISSING)
        )
    }
}

/// The substring every GPU resident-route probe uses when a table's device-memory cell is `None`
/// (tombstoned/never-populated). Used to detect the residency-invalidated-mid-statement case so the
/// read can fall back to the CPU pinned-read path (write-half MVCC, Stage 4).
const RESIDENT_DEVICE_MEMORY_MISSING: &str = "has no retained resident device memory";

#[derive(Debug, Clone)]
struct PendingMutation {
    txn_id: u64,
    payload: Vec<u8>,
}

pub struct Engine {
    /// The commit-critical mutable substate — the replicator (commit-`Index` oracle), the WAL, the
    /// per-txn commit timestamps, and the recent-commits conflict ledger — bundled behind ONE mutex
    /// that IS the **commit_mutex** (write-half MVCC, Stage 4). The concurrent DML commit path locks
    /// it for its short critical section (validate → assign `commit_seq` → WAL fsync → publish), so
    /// commits serialize ONLY here while prepare runs off-lock and readers stay lock-free. Code that
    /// already holds `&mut self` (serialized DDL apply, recovery, checkpoint/snapshot admin) reaches
    /// it via `commit_state_mut()` (a zero-cost `Mutex::get_mut`, no actual locking).
    commit: Mutex<CommitState>,
    /// In-flight transactions' read snapshots (write-half MVCC, Stage 4), for the oldest-active GC
    /// boundary. Separate from `commit` so a transaction can register its snapshot at prepare-begin
    /// WITHOUT serializing on the commit_mutex (prepare is off-lock).
    active_snapshots: Mutex<ActiveSnapshots>,
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
}

/// The DDL-only catalog working state, serialized behind the engine's **catalog latch**
/// (`Engine::catalog_latch`) — see that field's doc. It bundles every `Engine` catalog field that is
/// mutated by DDL and is NOT interior-mutable: the working catalog maps the read path consults only
/// through the *published* snapshot (`relational_catalog`/`relational_views`/
/// `relational_materialized_views`/`relational_functions`), plus the DDL-only maps the read path never
/// consults at all. DDL serializes (one writer), so holding the latch makes a DDL's working-map
/// mutation + its published-snapshot publish atomic w.r.t. another DDL; lock-free readers and the
/// concurrent-DML path never take this latch.
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
    relational_next_oid: u32,
    relational_next_column_id: u32,
}

/// The commit-critical mutable substate bundled behind the engine's commit_mutex (write-half MVCC,
/// Stage 4). Holding the lock on this is exactly the "short commit critical section": validate the
/// write-set against the ledger, assign a `commit_seq` from the replicator, append + fsync the WAL,
/// then (back in the engine) publish and bump `committed_seq`. Code holding `&mut Engine` reaches it
/// lock-free via `Mutex::get_mut`.
struct CommitState {
    /// The commit-`Index` oracle + log: `propose` assigns the next monotonic `commit_seq` inside the
    /// critical section (Stage 0 unification — `commit_seq == commit Index`).
    repl: LocalReplicator,
    /// The write-ahead log; `append` + `flush_all` inside the critical section make the commit
    /// crash-durable before it is published (the group-commit mechanism amortizes concurrent
    /// committers' fsyncs).
    wal: WalBuffer,
    /// Per-txn commit timestamps (durable transaction identity → wall-clock micros), for
    /// PITR-by-timestamp lookups. Keyed by the façade txn_id (the durable identity), distinct from
    /// the MVCC `commit_seq`.
    wal_commit_timestamps_micros: BTreeMap<TxnId, u64>,
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

impl Engine {
    pub fn new_local() -> Self {
        Self::with_planner_config(PlannerConfig::default())
    }

    pub fn recover_from_durable_wal(records: &[WalRecord]) -> Result<Self, EngineError> {
        let engine = Self::new_local();
        for record in records {
            engine.commit_mutation(record.txn_id, record.payload.clone())?;
        }
        Ok(engine)
    }

    pub fn recover_from_durable_wal_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        let records = read_wal_segment(path)?;
        Self::recover_from_durable_wal(&records)
    }

    pub fn recover_from_durable_wal_checkpoint(
        control_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        let (_control, records) = read_wal_checkpoint(control_path)?;
        Self::recover_from_durable_wal(&records)
    }

    pub fn recover_from_durable_wal_archive(
        manifest_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        let (_manifest, records) = read_wal_archive(manifest_path)?;
        Self::recover_from_durable_wal(&records)
    }

    pub fn recover_from_durable_wal_archive_to_txn(
        manifest_path: impl AsRef<std::path::Path>,
        target_txn_id: TxnId,
    ) -> Result<Self, EngineError> {
        let (_manifest, _target, records) = read_wal_archive_to_txn(manifest_path, target_txn_id)?;
        Self::recover_from_durable_wal(&records)
    }

    pub fn recover_from_durable_wal_archive_to_timestamp_micros(
        manifest_path: impl AsRef<std::path::Path>,
        target_timestamp_micros: u64,
    ) -> Result<Self, EngineError> {
        let (_manifest, _target, records) =
            read_wal_archive_to_timestamp_micros(manifest_path, target_timestamp_micros)?;
        Self::recover_from_durable_wal(&records)
    }

    pub fn recover_from_durable_wal_checkpoint_and_archive_to_txn(
        control_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
        target_txn_id: TxnId,
    ) -> Result<Self, EngineError> {
        let (control, base_records) = read_wal_checkpoint(control_path)?;
        let (_manifest, _target, archive_records) =
            read_wal_archive_to_txn(manifest_path, target_txn_id)?;
        Self::recover_from_checkpoint_and_archive_records(&control, &base_records, &archive_records)
    }

    pub fn recover_from_durable_wal_checkpoint_and_archive_to_timestamp_micros(
        control_path: impl AsRef<std::path::Path>,
        manifest_path: impl AsRef<std::path::Path>,
        target_timestamp_micros: u64,
    ) -> Result<Self, EngineError> {
        let (control, base_records) = read_wal_checkpoint(control_path)?;
        let (_manifest, _target, archive_records) =
            read_wal_archive_to_timestamp_micros(manifest_path, target_timestamp_micros)?;
        Self::recover_from_checkpoint_and_archive_records(&control, &base_records, &archive_records)
    }

    fn recover_from_checkpoint_and_archive_records(
        control: &WalControlFile,
        base_records: &[WalRecord],
        archive_records: &[WalRecord],
    ) -> Result<Self, EngineError> {
        let boundary_index =
            Self::validate_checkpoint_archive_overlap(control, base_records, archive_records)?;

        let mut recovered_records = base_records.to_vec();
        recovered_records.extend_from_slice(&archive_records[boundary_index + 1..]);
        Self::recover_from_durable_wal(&recovered_records)
    }

    fn validate_checkpoint_archive_overlap(
        control: &WalControlFile,
        base_records: &[WalRecord],
        archive_records: &[WalRecord],
    ) -> Result<usize, EngineError> {
        let base_last_txn_id = control.checkpoint.last_durable_txn_id.ok_or_else(|| {
            EngineError::Durability(
                "base backup checkpoint has no durable transaction boundary".to_string(),
            )
        })?;
        let boundary_index = archive_records
            .iter()
            .position(|record| record.txn_id == base_last_txn_id)
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "WAL archive does not overlap base backup transaction boundary {}",
                    base_last_txn_id
                ))
            })?;
        let archive_prefix = &archive_records[..=boundary_index];
        if archive_prefix == base_records {
            return Ok(boundary_index);
        }
        if boundary_index == 0 && archive_records.first() == base_records.last() {
            return Ok(boundary_index);
        }
        Err(EngineError::Durability(format!(
            "WAL archive prefix does not match base backup checkpoint boundary {}",
            base_last_txn_id
        )))
    }

    pub fn with_planner_config(planner_cfg: PlannerConfig) -> Self {
        Self {
            commit: Mutex::new(CommitState {
                repl: LocalReplicator::leader(),
                wal: WalBuffer::default(),
                wal_commit_timestamps_micros: BTreeMap::new(),
                ledger: RecentCommitsLedger::default(),
                sm: KvStateMachine::default(),
                txn_manager: TxnManager::default(),
            }),
            active_snapshots: Mutex::new(ActiveSnapshots::default()),
            read_state: Arc::new(ReadState::new()),
            catalog_latch: Mutex::new(DdlCatalogState {
                relational_catalog: BTreeMap::new(),
                relational_views: BTreeMap::new(),
                relational_materialized_views: BTreeMap::new(),
                relational_functions: BTreeMap::new(),
                relational_sequences: BTreeMap::new(),
                relational_domains: BTreeMap::new(),
                relational_publications: BTreeMap::new(),
                relational_subscriptions: BTreeMap::new(),
                relational_roles: BTreeMap::new(),
                relational_databases: BTreeMap::new(),
                relational_tablespaces: BTreeMap::new(),
                relational_public_schema_exists: true,
                relational_public_schema_implicit: true,
                relational_schema_acl: BTreeMap::new(),
                relational_default_table_acl: BTreeMap::new(),
                relational_comments: BTreeMap::new(),
                relational_resident_cache: RelationalResidentCache::default(),
                relational_next_oid: FIRST_USER_RELATION_OID,
                relational_next_column_id: FIRST_USER_COLUMN_ID,
            }),
            metrics: RuntimeMetrics::default(),
            batcher: Mutex::new(DualTriggerBatcher::new(64, Duration::from_millis(1))),
            planner: Planner::new(planner_cfg),
            router: DeviceRouter::new(MockGpuRuntime::default()),
            cached_cuda_probe_runtime: OnceLock::new(),
        }
    }

    /// Lock the commit_mutex, recovering from poison. Continuing past a poisoned commit lock is the
    /// engine's existing wedge-don't-recover policy's counterpart here: the façade re-homes the
    /// poison-on-panic decision to the commit path (a committer that panics mid-section poisons this
    /// lock, and the façade refuses to serve), so this `into_inner` recovery is the path the façade
    /// then trips on — it never silently serves torn state.
    fn commit_state(&self) -> std::sync::MutexGuard<'_, CommitState> {
        self.commit
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `&mut`-access the commit substate WITHOUT locking — sound because `&mut self` already proves
    /// exclusive access (no other thread can hold `&self`). Used by the serialized DDL apply,
    /// recovery, and checkpoint/snapshot admin paths, which all run under the façade's exclusive
    /// (catalog-latch) write lock.
    fn commit_state_mut(&mut self) -> &mut CommitState {
        self.commit
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Lock the **catalog latch**, recovering from poison (same wedge-don't-recover policy as
    /// [`Engine::commit_state`]: a DDL that panics mid-apply poisons this latch, and the façade then
    /// refuses to serve rather than expose a torn working catalog). Lock order is fixed: a caller that
    /// also needs the commit_mutex must take `commit_state()` FIRST, then this. Lock-free readers and
    /// the concurrent-DML path never call this.
    fn ddl_catalog(&self) -> std::sync::MutexGuard<'_, DdlCatalogState> {
        self.catalog_latch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `&mut`-access the DDL catalog state WITHOUT locking — sound because `&mut self` already proves
    /// exclusive access. Used by construction / serialized DDL-via-`&mut self` / recovery / residency
    /// admin paths (mirrors [`Engine::commit_state_mut`]).
    fn ddl_catalog_mut(&mut self) -> &mut DdlCatalogState {
        self.catalog_latch
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Whether the catalog latch is currently poisoned (a DDL panicked mid-apply). The façade checks
    /// this on the DDL branch so a DDL that panicked wedges service rather than serving a torn catalog.
    pub fn is_catalog_latch_poisoned(&self) -> bool {
        self.catalog_latch.is_poisoned()
    }

    /// Whether the commit_mutex is currently poisoned (a committer panicked mid-section). The façade
    /// checks this to re-home its poison-on-panic policy onto the commit path (write-half Stage 4).
    pub fn is_commit_path_poisoned(&self) -> bool {
        self.commit.is_poisoned()
    }

    /// A value-`Arc` clone of the lock-free read-path state. The concurrent-dispatch façade pins this
    /// once (e.g. at `SharedEngine` construction) so reads + the concurrent-DML path reach `mvcc`,
    /// `committed_seq`, resident device memory, and route telemetry WITHOUT taking the engine
    /// `RwLock` (lock-free read path, write-half MVCC). Cheap (one refcount bump); the returned
    /// handle shares the *same* interior-mutable state the engine mutates under the catalog latch.
    pub fn read_state(&self) -> Arc<ReadState> {
        Arc::clone(&self.read_state)
    }

    // `&self` read-only shims over the commit substate, for the scattered leader-checks and admin
    // queries that used to read `self.repl`/`self.wal` directly. Each takes the commit lock only
    // briefly (a cheap field read) — never across a read body.
    fn repl_role(&self) -> Role {
        self.commit_state().repl.role()
    }

    /// Lock the pending-mutation batcher, recovering from poison. Held only briefly (a field read or a
    /// drain), never across a `commit_mutation` (see the `batcher` field doc).
    fn batcher(&self) -> std::sync::MutexGuard<'_, DualTriggerBatcher<PendingMutation>> {
        self.batcher
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Whether the current thread is executing a relational read from INSIDE the commit critical
    /// section (a materialized-view create/refresh applying a committed entry), where the commit_mutex
    /// is already held and the deep read executor's leader gate would self-deadlock if it re-locked it.
    /// Thread-local + RAII-scoped ([`Engine::skip_leader_check_during_internal_read`]); false for every
    /// client read, so their leader gate is unchanged.
    fn mvcc_read_skips_leader_check(&self) -> bool {
        MVCC_READ_SKIPS_LEADER_CHECK.with(|flag| flag.get())
    }

    /// Run `f` (an internal mid-commit relational read) with the deep read executor's leader re-check
    /// suppressed on this thread, restoring the prior value on the way out (RAII, panic-safe).
    fn skip_leader_check_during_internal_read<R>(&self, f: impl FnOnce(&Self) -> R) -> R {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                MVCC_READ_SKIPS_LEADER_CHECK.with(|flag| flag.set(self.0));
            }
        }
        let _restore = MVCC_READ_SKIPS_LEADER_CHECK.with(|flag| {
            let prev = flag.get();
            flag.set(true);
            Restore(prev)
        });
        f(self)
    }

    pub fn with_batching(max_items: usize, max_wait: Duration) -> Self {
        Self::with_batching_and_planner_config(max_items, max_wait, PlannerConfig::default())
    }

    pub fn with_batching_and_planner_config(
        max_items: usize,
        max_wait: Duration,
        planner_cfg: PlannerConfig,
    ) -> Self {
        let mut s = Self::with_planner_config(planner_cfg);
        s.batcher = Mutex::new(DualTriggerBatcher::new(max_items, max_wait));
        s
    }

    /// A fresh engine whose commit path is **crash-durable**: every committed mutation's WAL record
    /// is fsynced to `segment_path` (file bytes + parent-directory entry) before the commit becomes
    /// visible (`visible_up_to` is bumped). The default [`Engine::new_local`] keeps the WAL purely
    /// in-memory; this is the constructor to use when durability is required.
    pub fn with_durable_wal_segment(segment_path: impl Into<std::path::PathBuf>) -> Self {
        Self::with_durable_wal_segment_and_planner_config(segment_path, PlannerConfig::default())
    }

    pub fn with_durable_wal_segment_and_planner_config(
        segment_path: impl Into<std::path::PathBuf>,
        planner_cfg: PlannerConfig,
    ) -> Self {
        let mut engine = Self::with_planner_config(planner_cfg);
        engine.commit_state_mut().wal = WalBuffer::with_durable_segment(segment_path);
        engine
    }

    /// Open (recover) a durable database from an existing WAL `segment_path` and keep writing to it.
    ///
    /// On crash recovery this is the realistic entry point: it replays every record that was fsync-
    /// durable in the segment — reconstructing exactly the committed state, since each record is
    /// re-applied through [`Engine::commit_mutation`], which re-derives the MVCC stamp from the
    /// commit `Index` (Stage 0 stamp/boundary unification) — and then continues to append durably to
    /// the same segment. A torn or partially-written trailing record is rejected by the segment's
    /// CRC at [`read_wal_segment`] time, so a commit whose fsync did not complete is never replayed
    /// (no visible-but-not-durable state). If the segment does not exist yet, this behaves like
    /// [`Engine::with_durable_wal_segment`] (a fresh durable database).
    pub fn open_durable_wal_segment(
        segment_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, EngineError> {
        Self::open_durable_wal_segment_with_planner_config(segment_path, PlannerConfig::default())
    }

    pub fn open_durable_wal_segment_with_planner_config(
        segment_path: impl AsRef<std::path::Path>,
        planner_cfg: PlannerConfig,
    ) -> Result<Self, EngineError> {
        let segment_path = segment_path.as_ref();
        let recovered_records = if segment_path.exists() {
            read_wal_segment(segment_path)?
        } else {
            Vec::new()
        };
        let mut engine = Self::with_planner_config(planner_cfg);
        // Replay the durable prefix WITHOUT a durable backing so the replay does not rewrite the
        // segment on every record; then install the durable segment so post-recovery commits append
        // durably to the same file. The replayed records are then re-marked durable so the segment
        // (rewritten on the next commit's flush) continues to include them.
        for record in &recovered_records {
            engine.commit_mutation(record.txn_id, record.payload.clone())?;
        }
        engine.commit_state_mut().wal = WalBuffer::with_durable_segment(segment_path);
        engine
            .commit_state_mut()
            .wal
            .reinstate_durable_records(recovered_records);
        Ok(engine)
    }

    pub fn simulate_next_wal_flush_failure(&mut self) {
        self.commit_state_mut().wal.fail_next_flush();
    }

    /// Group-commit accounting for the live WAL (fsync groups, durable records, largest group).
    pub fn wal_group_commit_stats(&self) -> WalGroupCommitStats {
        self.commit_state().wal.group_commit_stats()
    }

    /// Whether the engine's commit path is crash-durable (WAL fsynced before visibility).
    pub fn wal_is_durable(&self) -> bool {
        self.commit_state().wal.is_durable()
    }

    pub fn mark_gpu_unavailable(&mut self, gpu_id: u16) {
        self.router.runtime_mut().mark_unavailable(gpu_id);
    }

    pub fn clear_gpu_unavailable(&mut self, gpu_id: u16) {
        self.router.runtime_mut().clear_unavailable(gpu_id);
    }

    pub fn mark_gpu_memory_pressured(&mut self, gpu_id: u16) {
        self.router.runtime_mut().mark_memory_pressured(gpu_id);
        self.invalidate_relational_residency_for_memory_pressure(gpu_id);
    }

    pub fn clear_gpu_memory_pressured(&mut self, gpu_id: u16) {
        self.router.runtime_mut().clear_memory_pressured(gpu_id);
    }

    pub fn set_relational_residency_budget_bytes(&mut self, gpu_id: u16, budget_bytes: u64) {
        self.ddl_catalog_mut()
            .relational_resident_cache
            .budget_bytes_by_gpu
            .insert(gpu_id, budget_bytes);
    }

    pub fn clear_relational_residency_budget_bytes(&mut self, gpu_id: u16) {
        self.ddl_catalog_mut()
            .relational_resident_cache
            .budget_bytes_by_gpu
            .remove(&gpu_id);
    }

    pub fn relational_residency_budget_bytes(&self, gpu_id: u16) -> Option<u64> {
        self.ddl_catalog()
            .relational_resident_cache
            .budget_bytes_by_gpu
            .get(&gpu_id)
            .copied()
    }

    pub fn relational_resident_bytes_for_gpu(&self, gpu_id: u16) -> u64 {
        let snapshot_bytes: u64 = self
            .read_state
            .residency
            .snapshots
            .load()
            .values()
            .filter(|snapshot| snapshot.gpu_id == gpu_id)
            .map(|snapshot| snapshot.resident_bytes)
            .sum();
        let partition_bytes: u64 = self
            .read_state
            .residency
            .partitions
            .load()
            .values()
            .flatten()
            .filter(|partition| partition.gpu_id == gpu_id)
            .map(|partition| partition.resident_bytes)
            .sum();
        snapshot_bytes.saturating_add(partition_bytes)
    }

    pub fn set_gpu_runtime_saturated(&mut self, saturated: bool) {
        self.router.runtime_mut().set_saturated(saturated);
    }

    pub fn become_follower(&mut self, term: Term) {
        self.commit_state_mut().repl.become_follower(term);
    }

    pub fn become_leader(&mut self, term: Term) {
        self.commit_state_mut().repl.become_leader(term);
    }

    pub fn become_candidate(&mut self, term: Term) {
        self.commit_state_mut().repl.become_candidate(term);
    }

    pub fn commit_mutation(
        &self,
        txn_id: u64,
        payload: Vec<u8>,
    ) -> Result<CommitToken, EngineError> {
        let timestamp_micros = self.next_commit_timestamp_micros();
        self.commit_mutation_at(txn_id, payload, timestamp_micros)
    }

    fn next_commit_timestamp_micros(&self) -> u64 {
        let wall_clock = current_timestamp_micros();
        self.commit_state()
            .wal_commit_timestamps_micros
            .values()
            .copied()
            .max()
            .map(|last| wall_clock.max(last.saturating_add(1)))
            .unwrap_or(wall_clock)
    }

    pub fn commit_mutation_at(
        &self,
        txn_id: u64,
        payload: Vec<u8>,
        timestamp_micros: u64,
    ) -> Result<CommitToken, EngineError> {
        if self.repl_role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        // Durable-commit critical section under the **commit_mutex** (A.4 unification — this method is
        // now `&self`, the SERIALIZED commit path for DDL / KV / sequence-default INSERT / replay,
        // reached through `&Engine`; the concurrent autocommit-DML path is `commit_dml_concurrent`,
        // which runs the SAME WAL-before-publish sequence under this same lock). Lock order is fixed:
        // `commit_state()` (commit_mutex) FIRST, then `ddl_catalog()` (catalog latch) acquired INSIDE.
        // `next_commit_timestamp_micros` (which itself locks the commit_mutex) is computed by the
        // caller BEFORE this, so we never re-enter the non-reentrant commit_mutex.
        let mut commit = self.commit_state();
        let token = {
            let wal_len_before = commit.wal.len();
            commit.wal.append(WalRecord {
                txn_id,
                payload: payload.clone(),
            });
            let token = match commit.repl.propose(payload) {
                Ok(token) => token,
                Err(err) => {
                    commit.wal.truncate(wal_len_before);
                    return Err(err);
                }
            };
            if let Err(err) = commit.wal.flush_all() {
                commit.repl.rollback_unapplied_from(token.index);
                commit.wal.truncate(wal_len_before);
                return Err(err);
            }
            commit
                .repl
                .wait_committed(token, Duration::from_millis(0))?;
            // `txn_id` (the façade `next_txn_id`) is the durable transaction *identity* — recorded in
            // the WAL record and keyed here for PITR lookups. Intentionally DECOUPLED from the MVCC
            // version stamp, which uses the commit `Index` (see `apply_mvcc_entry`).
            commit
                .wal_commit_timestamps_micros
                .insert(txn_id, timestamp_micros);
            token
        };

        let to_apply: Vec<LogEntry> = commit
            .repl
            .drain_committed_from(commit.repl.applied_index())
            .cloned()
            .collect();

        // Hold the catalog latch across the WHOLE apply loop AND the catalog publish (PART B), so a
        // DDL's working-map mutation + the published-snapshot push are atomic w.r.t. another DDL. Lock
        // order is fixed: commit_mutex (held in `commit`) FIRST, then this latch.
        {
            let mut catalog_guard = self.ddl_catalog();
            let cat = &mut *catalog_guard;
            for e in &to_apply {
                commit.sm.apply(e)?;
                self.apply_mvcc_entry(e, cat)?;
                commit.repl.mark_applied(e.index);
            }

            // Publish ordering (Stage 2 — blocker #1; PART B catalog↔data co-pinning). The apply loop
            // published this commit's data generation(s) and mutated the working catalog maps (for any
            // DDL entries). Order the rest so a lock-free reader gets a consistent (catalog, data) pair:
            //   1. residency tombstones, 2. catalog ring push (stamped at `token.index`), then LAST
            //   3. `committed_seq` release-store.
            // The catalog is pushed BEFORE `committed_seq` (the FLIP from the old order) so a reader
            // that loads `committed_seq = token.index` and selects `catalog_as_of(token.index)` is
            // guaranteed to find this generation — the catalog is visible no later than `committed_seq`.
            // That, with the per-boundary self-consistency of the data (MVCC versions stamp old/new
            // part-counts at the DDL's commit_seq), rules out a reader straddling a shape-changing DDL.
            self.invalidate_relational_residency_for_commit(&to_apply, txn_id, token.index);
            let prune_below = self.catalog_prune_boundary(token.index);
            self.publish_catalog_snapshot(cat, token.index, prune_below);
        }
        self.publish_committed_seq(token.index);
        self.metrics.inc_commit();
        drop(commit);

        Ok(token)
    }

    fn commit_mutation_at_with_current_apply<F>(
        &self,
        txn_id: u64,
        payload: Vec<u8>,
        timestamp_micros: u64,
        mut apply_current: F,
    ) -> Result<(CommitToken, u128), EngineError>
    where
        // `apply_current` receives the commit sequence (the replicator-assigned commit `Index`) so
        // the directly-applied current entry stamps versions with the SAME commit-seq that
        // `apply_mvcc_entry` derives from `entry.index` on replay — keeping the live COPY hot path
        // byte-identical to a WAL replay of the same record (Stage 0 stamp/boundary unification).
        F: FnMut(&Self, &mut DdlCatalogState, Index) -> Result<(), EngineError>,
    {
        if self.repl_role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        // A.4 unification: `&self`, the whole critical section under the commit_mutex (held in
        // `commit`); the catalog latch is acquired INSIDE (fixed lock order).
        let mut commit = self.commit_state();
        let token = {
            let wal_len_before = commit.wal.len();
            commit.wal.append(WalRecord {
                txn_id,
                payload: payload.clone(),
            });
            let token = match commit.repl.propose(payload) {
                Ok(token) => token,
                Err(err) => {
                    commit.wal.truncate(wal_len_before);
                    return Err(err);
                }
            };
            if let Err(err) = commit.wal.flush_all() {
                commit.repl.rollback_unapplied_from(token.index);
                commit.wal.truncate(wal_len_before);
                return Err(err);
            }
            commit
                .repl
                .wait_committed(token, Duration::from_millis(0))?;
            // `txn_id` is the durable transaction identity (decoupled from the MVCC `commit_seq`).
            commit
                .wal_commit_timestamps_micros
                .insert(txn_id, timestamp_micros);
            token
        };

        let to_apply: Vec<LogEntry> = commit
            .repl
            .drain_committed_from(commit.repl.applied_index())
            .cloned()
            .collect();

        // Hold the catalog latch across the apply loop AND the catalog publish (PART B; lock order:
        // commit_mutex FIRST, then this latch).
        let residency_invalidation_micros;
        {
            let mut catalog_guard = self.ddl_catalog();
            let cat = &mut *catalog_guard;
            for e in &to_apply {
                if e.index == token.index {
                    // The caller applies the current entry directly through Engine state. Avoid
                    // cloning and reparsing the large SQL payload through the generic KV state
                    // machine on the COPY hot path while preserving WAL/replay records. Pass the
                    // commit sequence (`e.index`) so the stamp matches `apply_mvcc_entry`'s replay
                    // stamp, plus the held catalog latch for any working-map mutation.
                    apply_current(self, cat, e.index)?;
                } else {
                    commit.sm.apply(e)?;
                    self.apply_mvcc_entry(e, cat)?;
                }
                commit.repl.mark_applied(e.index);
            }

            // Publish ordering (PART B): residency → catalog ring push → `committed_seq` LAST (mirrors
            // `commit_mutation_at`). The current-apply closure already published data + mutated the maps.
            let residency_invalidation_started = Instant::now();
            self.invalidate_relational_residency_for_commit(&to_apply, txn_id, token.index);
            residency_invalidation_micros = residency_invalidation_started.elapsed().as_micros();
            let prune_below = self.catalog_prune_boundary(token.index);
            self.publish_catalog_snapshot(cat, token.index, prune_below);
        }
        self.publish_committed_seq(token.index);
        self.metrics.inc_commit();

        Ok((token, residency_invalidation_micros))
    }

    /// Global (stop-the-world) residency invalidation: invalidate every resident
    /// table. Retained as the **conservative fallback** for commit batches whose
    /// mutated tables cannot be determined precisely ([`Engine::residency_invalidation_scope`]
    /// returns `None`). Equivalent to invalidating each resident table individually.
    fn invalidate_relational_residency(&self, txn_id: TxnId, index: Index) {
        let tables: BTreeSet<String> = self
            .read_state
            .residency
            .snapshots
            .load()
            .keys()
            .cloned()
            .chain(self.read_state.residency.partitions.load().keys().cloned())
            .collect();
        for table in &tables {
            self.invalidate_relational_residency_table(table, txn_id, index);
        }
    }

    /// Invalidate the residency of a **single** table (its snapshot, device memory,
    /// and partitions). This is the per-table unit the commit path uses so a write to
    /// one table no longer evicts every other table's residency — the former
    /// stop-the-world behavior. Summing this over all resident tables reproduces the
    /// previous global invalidation; the device-memory/partition removals here are
    /// unconditional, so in degenerate cache states it may clear a stray cross-map
    /// entry the old two-loop form left — strictly-safe extra cleanup, never stale.
    fn invalidate_relational_residency_table(&self, table: &str, txn_id: TxnId, index: Index) {
        // Stage 3 — blocker #2: the snapshot/partition flag maps are now published behind `ArcSwap`,
        // so flag the invalidation copy-on-write under the serialized catalog latch (this never runs
        // on the concurrent commit path — that uses `invalidate_relational_residency_tables_concurrent`
        // which only tombstones the device-memory cells via `&self`).
        self.read_state.residency.with_snapshots_mut(|snapshots| {
            if let Some(snapshot) = snapshots.get_mut(table) {
                if snapshot.invalidated_by_txn_id.is_none() {
                    snapshot.invalidated_by_txn_id = Some(txn_id);
                    snapshot.invalidated_at_index = Some(index);
                }
                if let Some(proof) = snapshot.device_memory_proof.as_mut() {
                    proof.retained = false;
                }
            }
        });
        self.read_state.residency.device_memory.invalidate(table);
        self.read_state.residency.with_partitions_mut(|partitions| {
            if let Some(partitions) = partitions.get_mut(table) {
                for partition in partitions.iter_mut() {
                    if partition.invalidated_by_txn_id.is_none() {
                        partition.invalidated_by_txn_id = Some(txn_id);
                        partition.invalidated_at_index = Some(index);
                    }
                    if let Some(proof) = partition.device_memory_proof.as_mut() {
                        proof.retained = false;
                    }
                }
            }
        });
        self.read_state
            .residency
            .partition_device_memory
            .invalidate_table(table);
    }

    /// Invalidate the GPU residency of the `tables` a CONCURRENT commit mutated, via `&self`
    /// (write-half MVCC, Stage 4). Publishes a `None` tombstone on each table's resident
    /// device-memory cell(s) — the authoritative gate the read-path's `plan_relational_resident_route`
    /// checks (`has_retained_device_memory`), so after this a reader takes the CPU route on the new
    /// committed data rather than a stale GPU snapshot (residency↔data consistency, design Risk #3).
    /// Called INSIDE the commit critical section, before `committed_seq` is bumped, so a reader that
    /// observes the new `committed_seq` also observes the residency tombstone.
    ///
    /// It deliberately does NOT mutate the `snapshots`/`partitions` flag maps (those are not
    /// interior-mutable and are read lock-free by `&self` readers): the cell tombstone alone forces
    /// the CPU route. The flag-based telemetry / the explicit resident-snapshot-probe API are kept
    /// current only on the serialized invalidation path (`invalidate_relational_residency_table`),
    /// which runs under the exclusive catalog latch.
    fn invalidate_relational_residency_tables_concurrent(
        &self,
        tables: &BTreeSet<String>,
        _txn_id: TxnId,
        _index: Index,
    ) {
        for table in tables {
            self.read_state.residency.device_memory.invalidate(table);
            self.read_state
                .residency
                .partition_device_memory
                .invalidate_table(table);
        }
    }

    /// The set of tables a committed batch invalidates, or `None` to fall back to a
    /// global invalidation. **Conservative by construction:** it narrows only for
    /// commands whose mutated table(s) are unambiguous (single-table DML, TRUNCATE,
    /// DROP TABLE) and treats CREATE TABLE as touching no existing residency. Any other
    /// command — or a payload that fails to decode or parse — returns `None`, so
    /// residency is never left stale. Over-invalidation is merely a performance cost;
    /// under-invalidation would serve wrong rows, so this must never narrow when unsure.
    fn residency_invalidation_scope(entries: &[LogEntry]) -> Option<BTreeSet<String>> {
        let mut tables = BTreeSet::new();
        for entry in entries {
            let text = std::str::from_utf8(&entry.payload).ok()?;
            let command = parse_command(text).ok()?;
            match command {
                Command::Insert(insert) => {
                    tables.insert(insert.table);
                }
                Command::Update(update) => {
                    tables.insert(update.table);
                }
                Command::Delete(delete) => {
                    tables.insert(delete.table);
                }
                Command::TruncateTable(truncate) => {
                    tables.insert(truncate.name);
                }
                Command::DropTable(drop) => {
                    tables.extend(drop.names);
                }
                // A brand-new table has no prior residency to invalidate.
                Command::CreateTable(_) => {}
                // Any other command (other DDL, ACL, KV, schema/db/role/...) is not yet
                // precisely scoped; invalidate everything rather than risk staleness.
                _ => return None,
            }
        }
        Some(tables)
    }

    /// Invalidate residency for a committed batch: per-table when the mutated tables
    /// can be determined, else a conservative global invalidation.
    fn invalidate_relational_residency_for_commit(
        &self,
        entries: &[LogEntry],
        txn_id: TxnId,
        index: Index,
    ) {
        match Self::residency_invalidation_scope(entries) {
            Some(tables) => {
                for table in &tables {
                    self.invalidate_relational_residency_table(table, txn_id, index);
                }
            }
            None => self.invalidate_relational_residency(txn_id, index),
        }
    }

    fn invalidate_relational_residency_for_memory_pressure(&self, gpu_id: u16) {
        // Stage 3 — blocker #2: COW the snapshot/partition flag maps under the catalog latch, then
        // tombstone the device-memory cells of every table that was pressured (the cell `invalidate`
        // is `&self`, done outside the COW closure on the collected tables).
        let pressured_snapshot_tables = self.read_state.residency.with_snapshots_mut(|snapshots| {
            let mut tables = Vec::new();
            for (table, snapshot) in snapshots.iter_mut() {
                if snapshot.gpu_id == gpu_id {
                    snapshot.invalidated_by_memory_pressure = true;
                    snapshot.memory_pressure_active = true;
                    if let Some(proof) = snapshot.device_memory_proof.as_mut() {
                        proof.retained = false;
                    }
                    tables.push(table.clone());
                }
            }
            tables
        });
        for table in &pressured_snapshot_tables {
            self.read_state.residency.device_memory.invalidate(table);
        }
        let pressured_partition_tables =
            self.read_state.residency.with_partitions_mut(|partitions| {
                let mut tables = Vec::new();
                for (table, table_partitions) in partitions.iter_mut() {
                    let mut table_pressured = false;
                    for partition in table_partitions {
                        if partition.gpu_id == gpu_id {
                            partition.invalidated_by_memory_pressure = true;
                            partition.memory_pressure_active = true;
                            table_pressured = true;
                            if let Some(proof) = partition.device_memory_proof.as_mut() {
                                proof.retained = false;
                            }
                        }
                    }
                    if table_pressured {
                        tables.push(table.clone());
                    }
                }
                tables
            });
        for table in &pressured_partition_tables {
            self.read_state
                .residency
                .partition_device_memory
                .invalidate_table(table);
        }
    }

    /// Apply one committed log entry to the engine state (`&self`). The caller holds the **catalog
    /// latch** and passes `&mut DdlCatalogState` so a DDL entry's working-map mutation can be made
    /// atomic with the subsequent catalog-snapshot publish (the caller holds the SAME guard across both
    /// — PART B). Lock order is fixed: the caller already holds the commit_mutex, then the catalog
    /// latch. The `apply_*` methods (now `&self`) freely call `self.read_state.*` and the `&self`
    /// `preflight_*` tree (which reads the published catalog snapshot, never this latch — no reentry).
    fn apply_mvcc_entry(
        &self,
        entry: &LogEntry,
        cat: &mut DdlCatalogState,
    ) -> Result<(), EngineError> {
        let Ok(text) = std::str::from_utf8(&entry.payload) else {
            return Ok(());
        };
        let Ok(cmd) = parse_command(text) else {
            return Ok(());
        };

        // Stage 0 (write-half MVCC): the version stamp is the commit sequence, which is the
        // replicator-assigned commit `Index` (== WAL append order == read boundary `visible_up_to`).
        // Deliberately NOT the façade `next_txn_id`: under the future off-lock prepare, txn_id
        // allocation order diverges from commit order, but `entry.index` is always in commit order.
        // Because recovery re-proposes WAL records in log order, each entry gets the identical
        // monotonic `entry.index` on replay, so this re-derives byte-identical `created_by`/
        // `deleted_by` stamps from log position alone — independent of the recorded façade txn_id.
        let commit_seq: TxnId = entry.index;
        let visibility = StorageVisibility {
            read_txn_id: commit_seq,
        };

        match cmd {
            Command::SetKv { key, value } => {
                // KV lives in its own partition. Read the current version under the loaded
                // generation, then publish a new KV generation with the update/insert applied.
                let existing = self
                    .read_state
                    .mvcc
                    .load_kv()
                    .get()
                    .tuple_fetch_by_key(&key, visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let tuple_id = if existing.is_none() {
                    Some(self.read_state.mvcc.reserve_tuple_id())
                } else {
                    None
                };
                self.read_state.mvcc.with_kv_mut(|store| {
                    if let Some(tuple) = existing {
                        store
                            .tuple_update(tuple.tuple_id, value, commit_seq)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))
                    } else {
                        store
                            .tuple_insert_with_id(
                                tuple_id.expect("fresh id reserved for new key"),
                                NewTuple { key, value },
                                commit_seq,
                            )
                            .map(|_| ())
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))
                    }
                })?;
            }
            Command::DeleteKv { key } => {
                let existing = self
                    .read_state
                    .mvcc
                    .load_kv()
                    .get()
                    .tuple_fetch_by_key(&key, visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                if let Some(tuple) = existing {
                    self.read_state.mvcc.with_kv_mut(|store| {
                        store
                            .tuple_delete(tuple.tuple_id, commit_seq)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))
                    })?;
                }
            }
            Command::CreateSchema(create) => self.apply_create_schema(cat, create)?,
            Command::DropSchema(drop) => self.apply_drop_schema(cat, drop)?,
            Command::CreateDatabase(create) => self.apply_create_database(cat, create)?,
            Command::DropDatabase(drop) => self.apply_drop_database(cat, drop)?,
            Command::RenameDatabase(rename) => self.apply_rename_database(cat, rename)?,
            Command::CreateTablespace(create) => self.apply_create_tablespace(cat, create)?,
            Command::DropTablespace(drop) => self.apply_drop_tablespace(cat, drop)?,
            Command::RenameTablespace(rename) => self.apply_rename_tablespace(cat, rename)?,
            Command::CreateTable(create) => self.apply_create_table(cat, create)?,
            Command::AddPrimaryKey(add) => self.apply_add_primary_key(cat, add)?,
            Command::AddUniqueConstraint(add) => self.apply_add_unique_constraint(cat, add)?,
            Command::AddCheckConstraint(add) => self.apply_add_check_constraint(cat, add)?,
            Command::AddForeignKey(add) => self.apply_add_foreign_key(cat, add, commit_seq)?,
            Command::AddColumn(add) => self.apply_add_column(cat, add, commit_seq)?,
            Command::RenameTable(rename) => self.apply_rename_table(cat, rename, commit_seq)?,
            Command::RenameColumn(rename) => self.apply_rename_column(cat, rename)?,
            Command::RenameConstraint(rename) => self.apply_rename_constraint(cat, rename)?,
            Command::DropColumn(drop) => self.apply_drop_column(cat, drop, commit_seq)?,
            Command::DropConstraint(drop) => self.apply_drop_constraint(cat, drop)?,
            Command::CreateIndex(create) => self.apply_create_index(cat, create)?,
            Command::RenameIndex(rename) => self.apply_rename_index(cat, rename)?,
            Command::CreateView(create) => self.apply_create_view(cat, create)?,
            Command::RenameView(rename) => self.apply_rename_view(cat, rename)?,
            Command::CreateMaterializedView(create) => {
                self.apply_create_materialized_view(cat, create)?
            }
            Command::RefreshMaterializedView(refresh) => {
                self.apply_refresh_materialized_view(cat, refresh)?
            }
            Command::RenameMaterializedView(rename) => {
                self.apply_rename_materialized_view(cat, rename)?
            }
            Command::CreateFunction(create) => self.apply_create_function(cat, create)?,
            Command::RenameFunction(rename) => self.apply_rename_function(cat, rename)?,
            Command::DropFunction(drop) => self.apply_drop_function(cat, drop)?,
            Command::SelectFunction(_) => {}
            Command::CreateSequence(create) => self.apply_create_sequence(cat, create)?,
            Command::CreateDomain(create) => self.apply_create_domain(cat, create)?,
            Command::SequenceNextVal(nextval) => {
                self.apply_sequence_nextval(cat, nextval)?;
            }
            Command::SequenceSetVal(setval) => {
                self.apply_sequence_setval(cat, setval)?;
            }
            Command::RenameSequence(rename) => self.apply_rename_sequence(cat, rename)?,
            Command::DropTable(drop) => self.apply_drop_table(cat, drop, commit_seq)?,
            Command::TruncateTable(truncate) => {
                self.apply_truncate_table(cat, truncate, commit_seq)?
            }
            Command::DropIndex(drop) => self.apply_drop_index(cat, drop)?,
            Command::DropView(drop) => self.apply_drop_view(cat, drop)?,
            Command::DropMaterializedView(drop) => self.apply_drop_materialized_view(cat, drop)?,
            Command::DropSequence(drop) => self.apply_drop_sequence(cat, drop)?,
            Command::DropDomain(drop) => self.apply_drop_domain(cat, drop)?,
            Command::CreatePublication(create) => self.apply_create_publication(cat, create)?,
            Command::DropPublication(drop) => self.apply_drop_publication(cat, drop)?,
            Command::CreateSubscription(create) => self.apply_create_subscription(cat, create)?,
            Command::DropSubscription(drop) => self.apply_drop_subscription(cat, drop)?,
            Command::CreateRole(create) => self.apply_create_role(cat, create)?,
            Command::DropRole(drop) => self.apply_drop_role(cat, drop)?,
            Command::RenameRole(rename) => self.apply_rename_role(cat, rename)?,
            Command::GrantTable(grant) => self.apply_grant_acl(
                cat,
                &grant.relation,
                grant.kind,
                &grant.grantee,
                &grant.privileges,
            )?,
            Command::RevokeTable(revoke) => self.apply_revoke_acl(
                cat,
                &revoke.relation,
                revoke.kind,
                &revoke.grantee,
                &revoke.privileges,
            )?,
            Command::GrantSchema(grant) => {
                self.apply_grant_schema_acl(cat, &grant.schema, &grant.grantee, &grant.privileges)?
            }
            Command::RevokeSchema(revoke) => self.apply_revoke_schema_acl(
                cat,
                &revoke.schema,
                &revoke.grantee,
                &revoke.privileges,
            )?,
            Command::GrantDatabase(grant) => self.apply_grant_database_acl(
                cat,
                &grant.database,
                &grant.grantee,
                &grant.privileges,
            )?,
            Command::RevokeDatabase(revoke) => self.apply_revoke_database_acl(
                cat,
                &revoke.database,
                &revoke.grantee,
                &revoke.privileges,
            )?,
            Command::GrantTablespace(grant) => self.apply_grant_tablespace_acl(
                cat,
                &grant.tablespace,
                &grant.grantee,
                &grant.privileges,
            )?,
            Command::RevokeTablespace(revoke) => self.apply_revoke_tablespace_acl(
                cat,
                &revoke.tablespace,
                &revoke.grantee,
                &revoke.privileges,
            )?,
            Command::GrantFunction(grant) => self.apply_grant_function_acl(
                cat,
                &grant.function,
                &grant.grantee,
                &grant.privileges,
            )?,
            Command::RevokeFunction(revoke) => self.apply_revoke_function_acl(
                cat,
                &revoke.function,
                &revoke.grantee,
                &revoke.privileges,
            )?,
            Command::GrantDefaultTablePrivileges(grant) => {
                self.apply_grant_default_table_privileges(cat, &grant.grantee, &grant.privileges)?
            }
            Command::RevokeDefaultTablePrivileges(revoke) => self
                .apply_revoke_default_table_privileges(cat, &revoke.grantee, &revoke.privileges)?,
            Command::AlterColumnDefault(alter) => self.apply_alter_column_default(cat, alter)?,
            Command::CommentOn(comment) => self.apply_comment_on(cat, comment)?,
            Command::Insert(insert) => self.apply_insert(cat, insert, commit_seq)?,
            Command::Delete(delete) => self.apply_delete(cat, delete, commit_seq)?,
            Command::Update(update) => self.apply_update(cat, update, commit_seq)?,
            _ => {}
        }

        Ok(())
    }

    fn apply_create_view(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateView,
    ) -> Result<(), EngineError> {
        if cat.relational_catalog.contains_key(&create.name)
            || cat.relational_materialized_views.contains_key(&create.name)
            || cat.relational_sequences.contains_key(&create.name)
            || (!create.or_replace && cat.relational_views.contains_key(&create.name))
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                create.name
            )));
        }
        if cat
            .relational_materialized_views
            .contains_key(&create.query.table)
        {
            return Err(EngineError::ApplyFailed(
                "views over materialized views are unsupported".to_string(),
            ));
        }
        if create.or_replace && self.relational_view_has_dependents(&create.name) {
            return Err(EngineError::ApplyFailed(
                "cannot replace view because another view depends on it".to_string(),
            ));
        }
        if cat.relational_views.contains_key(&create.query.table) {
            if self.relational_view_depends_on(&create.query.table, &create.name) {
                return Err(EngineError::ApplyFailed(
                    "view dependency cycle is unsupported".to_string(),
                ));
            }
        } else if !cat.relational_catalog.contains_key(&create.query.table) {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" does not exist",
                create.query.table
            )));
        }
        let (oid, acl) = if let Some(existing) = cat.relational_views.get(&create.name) {
            (existing.oid, existing.acl.clone())
        } else {
            let oid = cat.relational_next_oid;
            cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
                EngineError::ApplyFailed("relational view OID allocation exhausted".to_string())
            })?;
            (oid, BTreeMap::new())
        };
        cat.relational_views.insert(
            create.name.clone(),
            RelationalView {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name: create.name,
                oid,
                query: create.query,
                definition: create.definition,
                acl,
            },
        );
        Ok(())
    }

    fn relational_view_depends_on(&self, view: &str, target: &str) -> bool {
        let mut seen = BTreeSet::new();
        self.relational_view_depends_on_inner(view, target, &mut seen)
    }

    fn relational_view_depends_on_inner(
        &self,
        view: &str,
        target: &str,
        seen: &mut BTreeSet<String>,
    ) -> bool {
        let cat = self.catalog_snapshot();
        if view == target {
            return true;
        }
        if !seen.insert(view.to_string()) {
            return false;
        }
        let Some(view) = cat.relational_views.get(view) else {
            return false;
        };
        self.relational_view_depends_on_inner(&view.query.table, target, seen)
    }

    fn relational_view_has_dependents(&self, view: &str) -> bool {
        let cat = self.catalog_snapshot();
        cat.relational_views.iter().any(|(candidate, _)| {
            candidate != view && self.relational_view_depends_on(candidate, view)
        })
    }

    fn apply_create_materialized_view(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateMaterializedView,
    ) -> Result<(), EngineError> {
        self.preflight_create_materialized_view(&create)?;
        // This SELECT runs INSIDE the commit critical section (the commit_mutex is held); suppress the
        // deep read executor's leader re-check on this thread so it does not self-deadlock re-locking it.
        let result = self
            .skip_leader_check_during_internal_read(|engine| {
                engine.execute_relational_select(&create.query)
            })
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed(
                "relational materialized view OID allocation exhausted".to_string(),
            )
        })?;
        let mut columns = Vec::with_capacity(result.columns.len());
        let mut next_column_id = cat.relational_next_column_id;
        for (idx, column) in result.columns.into_iter().enumerate() {
            let attnum = i16::try_from(idx + 1).map_err(|_| {
                EngineError::ApplyFailed(
                    "too many columns for bootstrap materialized view".to_string(),
                )
            })?;
            let id = next_column_id;
            next_column_id = next_column_id.checked_add(1).ok_or_else(|| {
                EngineError::ApplyFailed(
                    "relational materialized view column id allocation exhausted".to_string(),
                )
            })?;
            columns.push(RelationalColumn {
                id,
                table_oid: oid,
                attnum,
                name: column.name,
                ty: column.ty,
                domain: None,
                default: None,
                type_oid: column.type_oid,
                type_size: column.type_size,
            });
        }
        cat.relational_next_column_id = next_column_id;
        let rows = if create.with_data {
            result.rows
        } else {
            Vec::new()
        };
        cat.relational_materialized_views.insert(
            create.name.clone(),
            RelationalMaterializedView {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name: create.name,
                oid,
                query: create.query,
                definition: create.definition,
                columns,
                rows,
                acl: BTreeMap::new(),
            },
        );
        Ok(())
    }

    fn apply_refresh_materialized_view(
        &self,
        cat: &mut DdlCatalogState,
        refresh: RefreshMaterializedView,
    ) -> Result<(), EngineError> {
        self.preflight_refresh_materialized_view(&refresh)?;
        let existing = cat
            .relational_materialized_views
            .get(&refresh.name)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "materialized view \"{}\" does not exist",
                    refresh.name
                ))
            })?;
        let query = existing.query.clone();
        let columns = existing.columns.clone();
        // Mid-commit read (commit_mutex held): suppress the read executor's leader re-check.
        let result = self
            .skip_leader_check_during_internal_read(|engine| {
                engine.execute_relational_select(&query)
            })
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        if result.columns.len() != columns.len()
            || result
                .columns
                .iter()
                .zip(columns.iter())
                .any(|(left, right)| left.name != right.name || left.ty != right.ty)
        {
            return Err(EngineError::ApplyFailed(
                "materialized view refresh changed the result shape".to_string(),
            ));
        }
        let view = cat
            .relational_materialized_views
            .get_mut(&refresh.name)
            .expect("materialized view existence preflighted");
        view.rows = result.rows;
        Ok(())
    }

    fn apply_create_function(
        &self,
        cat: &mut DdlCatalogState,
        create: gpu_db_sql::CreateFunction,
    ) -> Result<(), EngineError> {
        if cat.relational_functions.contains_key(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "function \"{}\" already exists",
                create.name
            )));
        }
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational function OID allocation exhausted".to_string())
        })?;
        cat.relational_functions.insert(
            create.name.clone(),
            RelationalFunction {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name: create.name,
                oid,
                return_type: create.return_type,
                body: create.body,
                acl: BTreeMap::new(),
            },
        );
        Ok(())
    }

    fn apply_drop_function(
        &self,
        cat: &mut DdlCatalogState,
        drop: gpu_db_sql::DropFunction,
    ) -> Result<(), EngineError> {
        if !drop.if_exists && !cat.relational_functions.contains_key(&drop.name) {
            return Err(EngineError::ApplyFailed(format!(
                "function \"{}\" does not exist",
                drop.name
            )));
        }
        cat.relational_functions.remove(&drop.name);
        cat.relational_comments
            .remove(&RelationalCommentTarget::Function {
                function: drop.name,
            });
        Ok(())
    }

    fn apply_rename_function(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameFunction,
    ) -> Result<(), EngineError> {
        if !cat.relational_functions.contains_key(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "function \"{}\" does not exist",
                rename.old_name
            )));
        }
        if cat.relational_functions.contains_key(&rename.new_name) {
            return Err(EngineError::ApplyFailed(format!(
                "function \"{}\" already exists",
                rename.new_name
            )));
        }
        let Some(mut function) = cat.relational_functions.remove(&rename.old_name) else {
            return Ok(());
        };
        function.name = rename.new_name.clone();
        cat.relational_functions
            .insert(rename.new_name.clone(), function);
        let old_target = RelationalCommentTarget::Function {
            function: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::Function {
                    function: rename.new_name,
                },
                comment,
            );
        }
        Ok(())
    }

    fn apply_create_sequence(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateSequence,
    ) -> Result<(), EngineError> {
        if cat.relational_catalog.contains_key(&create.name)
            || cat.relational_views.contains_key(&create.name)
            || cat.relational_materialized_views.contains_key(&create.name)
            || cat.relational_sequences.contains_key(&create.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                create.name
            )));
        }
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational sequence OID allocation exhausted".to_string())
        })?;
        cat.relational_sequences.insert(
            create.name.clone(),
            RelationalSequence {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name: create.name,
                oid,
                last_value: 1,
                is_called: false,
                acl: BTreeMap::new(),
            },
        );
        Ok(())
    }

    fn create_implicit_sequence(
        &self,
        cat: &mut DdlCatalogState,
        name: &str,
    ) -> Result<(), EngineError> {
        self.apply_create_sequence(
            cat,
            CreateSequence {
                name: name.to_string(),
            },
        )
    }

    fn preflight_create_domain(&self, create: &CreateDomain) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(&create.name)
            || cat.relational_views.contains_key(&create.name)
            || cat.relational_materialized_views.contains_key(&create.name)
            || cat.relational_sequences.contains_key(&create.name)
            || cat.relational_domains.contains_key(&create.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "type \"{}\" already exists",
                create.name
            )));
        }
        Ok(())
    }

    fn apply_create_domain(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateDomain,
    ) -> Result<(), EngineError> {
        self.preflight_create_domain(&create)?;
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational domain OID allocation exhausted".to_string())
        })?;
        cat.relational_domains.insert(
            create.name.clone(),
            RelationalDomain {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name: create.name,
                oid,
                base_type: create.base_type,
            },
        );
        Ok(())
    }

    fn preflight_drop_domain(&self, drop: &DropDomain) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.domains {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "domain \"{}\" specified more than once",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_domains.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "domain \"{}\" does not exist",
                    name
                )));
            }
            if cat.relational_catalog.values().any(|table| {
                table
                    .columns
                    .iter()
                    .any(|column| column.domain.as_deref() == Some(name.as_str()))
            }) {
                return Err(EngineError::ApplyFailed(format!(
                    "cannot drop domain \"{}\" because other objects depend on it",
                    name
                )));
            }
        }
        Ok(())
    }

    fn apply_drop_domain(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropDomain,
    ) -> Result<(), EngineError> {
        self.preflight_drop_domain(&drop)?;
        for name in &drop.domains {
            cat.relational_domains.remove(name);
            cat.relational_comments
                .remove(&RelationalCommentTarget::Domain {
                    domain: name.clone(),
                });
        }
        Ok(())
    }

    fn resolve_column_domain_type(
        &self,
        column: &mut ColumnDef,
    ) -> Result<(u32, i16), EngineError> {
        let cat = self.catalog_snapshot();
        if let Some(domain_name) = column.domain.as_ref() {
            let domain = cat.relational_domains.get(domain_name).ok_or_else(|| {
                EngineError::ApplyFailed(format!("type \"{}\" does not exist", domain_name))
            })?;
            column.ty = domain.base_type;
            Ok((domain.oid, domain.base_type.type_size()))
        } else {
            Ok((column.ty.postgres_oid(), column.ty.type_size()))
        }
    }

    fn preflight_implicit_sequence_name(&self, name: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(name)
            || cat.relational_views.contains_key(name)
            || cat.relational_materialized_views.contains_key(name)
            || cat.relational_sequences.contains_key(name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{name}\" already exists"
            )));
        }
        Ok(())
    }

    fn preflight_column_default_target(&self, default: &ColumnDefault) -> Result<(), EngineError> {
        match default {
            ColumnDefault::Literal(_) => Ok(()),
            ColumnDefault::SequenceNextVal {
                sequence,
                create_if_missing: true,
            } => self.preflight_implicit_sequence_name(sequence),
            ColumnDefault::SequenceNextVal {
                sequence,
                create_if_missing: false,
            } => self.preflight_sequence_target(sequence),
        }
    }

    fn apply_sequence_nextval(
        &self,
        cat: &mut DdlCatalogState,
        nextval: SequenceNextVal,
    ) -> Result<i64, EngineError> {
        self.preflight_sequence_target(&nextval.name)?;
        let sequence = cat
            .relational_sequences
            .get_mut(&nextval.name)
            .expect("sequence target preflighted");
        let value = if sequence.is_called {
            sequence
                .last_value
                .checked_add(1)
                .ok_or_else(|| EngineError::ApplyFailed("sequence value overflow".to_string()))?
        } else {
            sequence.last_value
        };
        sequence.last_value = value;
        sequence.is_called = true;
        Ok(value)
    }

    fn evaluate_column_default(
        &self,
        cat: &mut DdlCatalogState,
        default: &ColumnDefault,
    ) -> Result<SqlValue, EngineError> {
        match default {
            ColumnDefault::Literal(value) => Ok(value.clone()),
            ColumnDefault::SequenceNextVal { sequence, .. } => {
                let value = self.apply_sequence_nextval(
                    cat,
                    SequenceNextVal {
                        name: sequence.clone(),
                    },
                )?;
                i32::try_from(value).map(SqlValue::Int4).map_err(|_| {
                    EngineError::ApplyFailed(
                        "sequence value is out of range for int4 default".to_string(),
                    )
                })
            }
        }
    }

    /// PURE column-default evaluation for `prepare_insert` (write-half MVCC, Stage 2). Identical
    /// arithmetic to [`Engine::evaluate_column_default`] / [`Engine::apply_sequence_nextval`], but
    /// `nextval` advances a per-call `seq_state` scratch (seeded lazily from the engine's sequence
    /// catalog) instead of mutating `self`. The scratch's final `(last_value, is_called)` per
    /// sequence is installed by `apply_delta`, so a prepare→apply pair advances the sequence by
    /// exactly what the old in-line apply did — while prepare stays `&self`.
    fn evaluate_column_default_pure(
        &self,
        default: &ColumnDefault,
        seq_state: &mut BTreeMap<String, (i64, bool)>,
    ) -> Result<SqlValue, EngineError> {
        match default {
            ColumnDefault::Literal(value) => Ok(value.clone()),
            ColumnDefault::SequenceNextVal { sequence, .. } => {
                self.preflight_sequence_target(sequence)?;
                let entry = seq_state.entry(sequence.clone()).or_insert_with(|| {
                    let catalog = self.catalog_snapshot();
                    let seq = catalog
                        .relational_sequences
                        .get(sequence)
                        .expect("sequence target preflighted");
                    (seq.last_value, seq.is_called)
                });
                let (last_value, is_called) = *entry;
                let value = if is_called {
                    last_value.checked_add(1).ok_or_else(|| {
                        EngineError::ApplyFailed("sequence value overflow".to_string())
                    })?
                } else {
                    last_value
                };
                *entry = (value, true);
                i32::try_from(value).map(SqlValue::Int4).map_err(|_| {
                    EngineError::ApplyFailed(
                        "sequence value is out of range for int4 default".to_string(),
                    )
                })
            }
        }
    }

    fn apply_sequence_setval(
        &self,
        cat: &mut DdlCatalogState,
        setval: SequenceSetVal,
    ) -> Result<i64, EngineError> {
        self.preflight_sequence_target(&setval.name)?;
        let sequence = cat
            .relational_sequences
            .get_mut(&setval.name)
            .expect("sequence target preflighted");
        sequence.last_value = setval.value;
        sequence.is_called = setval.is_called;
        Ok(setval.value)
    }

    fn apply_create_schema(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateSchema,
    ) -> Result<(), EngineError> {
        if create.name != PUBLIC_SCHEMA_NAME {
            return Err(EngineError::ApplyFailed(format!(
                "schema \"{}\" is not supported",
                create.name
            )));
        }
        if cat.relational_public_schema_exists
            && !create.if_not_exists
            && !cat.relational_public_schema_implicit
        {
            return Err(EngineError::ApplyFailed(format!(
                "schema \"{}\" already exists",
                create.name
            )));
        }
        cat.relational_public_schema_exists = true;
        cat.relational_public_schema_implicit = false;
        Ok(())
    }

    fn apply_drop_schema(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropSchema,
    ) -> Result<(), EngineError> {
        if drop.name != PUBLIC_SCHEMA_NAME {
            if drop.if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "schema \"{}\" does not exist",
                drop.name
            )));
        }
        if !cat.relational_public_schema_exists {
            if drop.if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "schema \"{}\" does not exist",
                drop.name
            )));
        }
        if !cat.relational_catalog.is_empty()
            || !cat.relational_views.is_empty()
            || !cat.relational_materialized_views.is_empty()
            || !cat.relational_functions.is_empty()
            || !cat.relational_sequences.is_empty()
            || !cat.relational_domains.is_empty()
            || !cat.relational_publications.is_empty()
            || !cat.relational_subscriptions.is_empty()
        {
            return Err(EngineError::ApplyFailed(format!(
                "cannot drop non-empty schema \"{}\"",
                drop.name
            )));
        }
        cat.relational_public_schema_exists = false;
        cat.relational_public_schema_implicit = false;
        cat.relational_schema_acl.clear();
        cat.relational_comments
            .remove(&RelationalCommentTarget::Schema { schema: drop.name });
        Ok(())
    }

    fn database_exists(&self, database: &str) -> bool {
        let cat = self.catalog_snapshot();
        database == "postgres" || cat.relational_databases.contains_key(database)
    }

    fn tablespace_exists(&self, tablespace: &str) -> bool {
        let cat = self.catalog_snapshot();
        matches!(tablespace, "pg_default" | "pg_global")
            || cat.relational_tablespaces.contains_key(tablespace)
    }

    fn apply_create_database(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateDatabase,
    ) -> Result<(), EngineError> {
        if self.database_exists(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "database \"{}\" already exists",
                create.name
            )));
        }
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational database OID allocation exhausted".to_string())
        })?;
        cat.relational_databases.insert(
            create.name.clone(),
            RelationalDatabase {
                name: create.name,
                oid,
                acl: BTreeMap::new(),
            },
        );
        Ok(())
    }

    fn apply_drop_database(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropDatabase,
    ) -> Result<(), EngineError> {
        let mut seen = BTreeSet::new();
        for database in &drop.names {
            if !seen.insert(database.clone()) {
                return Err(EngineError::ApplyFailed(format!(
                    "database \"{}\" specified more than once",
                    database
                )));
            }
            if database == "postgres" {
                return Err(EngineError::ApplyFailed(
                    "cannot drop bootstrap database \"postgres\"".to_string(),
                ));
            }
            if !drop.if_exists && !cat.relational_databases.contains_key(database) {
                return Err(EngineError::ApplyFailed(format!(
                    "database \"{}\" does not exist",
                    database
                )));
            }
        }
        for database in &drop.names {
            cat.relational_databases.remove(database);
            cat.relational_comments
                .remove(&RelationalCommentTarget::Database {
                    database: database.clone(),
                });
        }
        Ok(())
    }

    fn apply_rename_database(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameDatabase,
    ) -> Result<(), EngineError> {
        if rename.old_name == "postgres" {
            return Err(EngineError::ApplyFailed(
                "cannot rename bootstrap database \"postgres\"".to_string(),
            ));
        }
        if !cat.relational_databases.contains_key(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "database \"{}\" does not exist",
                rename.old_name
            )));
        }
        if self.database_exists(&rename.new_name) {
            return Err(EngineError::ApplyFailed(format!(
                "database \"{}\" already exists",
                rename.new_name
            )));
        }
        let mut database = cat
            .relational_databases
            .remove(&rename.old_name)
            .expect("database existence checked");
        database.name = rename.new_name.clone();
        cat.relational_databases
            .insert(rename.new_name.clone(), database);
        let old_target = RelationalCommentTarget::Database {
            database: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            let new_target = RelationalCommentTarget::Database {
                database: rename.new_name,
            };
            cat.relational_comments.insert(new_target, comment);
        }
        Ok(())
    }

    fn apply_create_tablespace(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateTablespace,
    ) -> Result<(), EngineError> {
        if self.tablespace_exists(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "tablespace \"{}\" already exists",
                create.name
            )));
        }
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational tablespace OID allocation exhausted".to_string())
        })?;
        cat.relational_tablespaces.insert(
            create.name.clone(),
            RelationalTablespace {
                name: create.name,
                oid,
                location: create.location,
                acl: BTreeMap::new(),
            },
        );
        Ok(())
    }

    fn apply_drop_tablespace(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropTablespace,
    ) -> Result<(), EngineError> {
        let mut seen = BTreeSet::new();
        for tablespace in &drop.names {
            if !seen.insert(tablespace.clone()) {
                return Err(EngineError::ApplyFailed(format!(
                    "tablespace \"{}\" specified more than once",
                    tablespace
                )));
            }
            if matches!(tablespace.as_str(), "pg_default" | "pg_global") {
                return Err(EngineError::ApplyFailed(format!(
                    "cannot drop bootstrap tablespace \"{}\"",
                    tablespace
                )));
            }
            if !drop.if_exists && !cat.relational_tablespaces.contains_key(tablespace) {
                return Err(EngineError::ApplyFailed(format!(
                    "tablespace \"{}\" does not exist",
                    tablespace
                )));
            }
        }
        for tablespace in &drop.names {
            cat.relational_tablespaces.remove(tablespace);
            cat.relational_comments
                .remove(&RelationalCommentTarget::Tablespace {
                    tablespace: tablespace.clone(),
                });
        }
        Ok(())
    }

    fn apply_rename_tablespace(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameTablespace,
    ) -> Result<(), EngineError> {
        if matches!(rename.old_name.as_str(), "pg_default" | "pg_global") {
            return Err(EngineError::ApplyFailed(format!(
                "cannot rename bootstrap tablespace \"{}\"",
                rename.old_name
            )));
        }
        if !cat.relational_tablespaces.contains_key(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "tablespace \"{}\" does not exist",
                rename.old_name
            )));
        }
        if self.tablespace_exists(&rename.new_name) {
            return Err(EngineError::ApplyFailed(format!(
                "tablespace \"{}\" already exists",
                rename.new_name
            )));
        }
        let mut tablespace = cat
            .relational_tablespaces
            .remove(&rename.old_name)
            .expect("tablespace existence checked");
        tablespace.name = rename.new_name.clone();
        cat.relational_tablespaces
            .insert(rename.new_name.clone(), tablespace);
        let old_target = RelationalCommentTarget::Tablespace {
            tablespace: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            let new_target = RelationalCommentTarget::Tablespace {
                tablespace: rename.new_name,
            };
            cat.relational_comments.insert(new_target, comment);
        }
        Ok(())
    }

    fn apply_create_table(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateTable,
    ) -> Result<(), EngineError> {
        if !cat.relational_public_schema_exists {
            return Err(EngineError::ApplyFailed(format!(
                "schema \"{}\" does not exist",
                PUBLIC_SCHEMA_NAME
            )));
        }
        if cat.relational_catalog.contains_key(&create.table)
            || cat.relational_views.contains_key(&create.table)
            || cat
                .relational_materialized_views
                .contains_key(&create.table)
            || cat.relational_sequences.contains_key(&create.table)
            || cat.relational_domains.contains_key(&create.table)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                create.table
            )));
        }
        let mut seen = BTreeSet::new();
        for column in &create.columns {
            if !seen.insert(column.name.clone()) {
                return Err(EngineError::ApplyFailed(format!(
                    "column \"{}\" specified more than once",
                    column.name
                )));
            }
        }
        let oid = cat.relational_next_oid;
        let next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational table OID allocation exhausted".to_string())
        })?;
        let implicit_sequences = create
            .columns
            .iter()
            .filter_map(|column| match &column.default {
                Some(ColumnDefault::SequenceNextVal {
                    sequence,
                    create_if_missing: true,
                }) => Some(sequence.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        for sequence in &implicit_sequences {
            self.preflight_implicit_sequence_name(sequence)?;
        }
        let mut columns = Vec::with_capacity(create.columns.len());
        let mut next_column_id = cat.relational_next_column_id;
        for (idx, mut column) in create.columns.into_iter().enumerate() {
            let (type_oid, type_size) = self.resolve_column_domain_type(&mut column)?;
            if let Some(default) = column.default.take() {
                // Coerce a cross-type default literal to the column type (parity with INSERT),
                // e.g. `bal NUMERIC DEFAULT 0` -> Numeric at the column scale.
                let default = coerce_column_default(default, column.ty, &column.name)?;
                self.preflight_column_default_target(&default)?;
                column.default = Some(default);
            }
            let attnum = i16::try_from(idx + 1).map_err(|_| {
                EngineError::ApplyFailed("too many columns for bootstrap catalog".to_string())
            })?;
            let id = next_column_id;
            next_column_id = next_column_id.checked_add(1).ok_or_else(|| {
                EngineError::ApplyFailed("relational column id allocation exhausted".to_string())
            })?;
            columns.push(RelationalColumn::from_def(
                id, oid, attnum, column, type_oid, type_size,
            ));
        }
        let primary_key = create.primary_key.clone();
        let unique_constraints = create.unique_constraints.clone();
        let check_constraints = create.check_constraints.clone();
        let name = create.table;
        let mut indexes = Vec::new();
        let mut checks = Vec::new();
        if let Some(primary_key) = primary_key {
            let constraint_name = primary_key.name.unwrap_or_else(|| format!("{}_pkey", name));
            indexes.push(RelationalIndex {
                name: constraint_name,
                table: name.clone(),
                column: primary_key.column,
                unique: true,
                primary_key: true,
                unique_constraint: false,
            });
        }
        for unique in unique_constraints {
            let constraint_name = unique
                .name
                .unwrap_or_else(|| format!("{}_{}_key", name, unique.column));
            if indexes.iter().any(|index| index.name == constraint_name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" already exists",
                    constraint_name
                )));
            }
            indexes.push(RelationalIndex {
                name: constraint_name,
                table: name.clone(),
                column: unique.column,
                unique: true,
                primary_key: false,
                unique_constraint: true,
            });
        }
        for check in check_constraints {
            let Some(column) = columns
                .iter()
                .find(|column| column.name == check.filter.column)
            else {
                return Err(EngineError::ApplyFailed(format!(
                    "column \"{}\" does not exist",
                    check.filter.column
                )));
            };
            if !sql_value_matches_type(&check.filter.value, column.ty) {
                return Err(EngineError::ApplyFailed(format!(
                    "invalid value for column \"{}\"",
                    check.filter.column
                )));
            }
            let constraint_name = check
                .name
                .unwrap_or_else(|| format!("{}_{}_check", name, check.filter.column));
            if indexes.iter().any(|index| index.name == constraint_name)
                || checks
                    .iter()
                    .any(|candidate: &RelationalCheckConstraint| candidate.name == constraint_name)
            {
                return Err(EngineError::ApplyFailed(format!(
                    "constraint \"{}\" already exists",
                    constraint_name
                )));
            }
            checks.push(RelationalCheckConstraint {
                name: constraint_name,
                column: check.filter.column,
                op: check.filter.op,
                value: check.filter.value,
            });
        }
        cat.relational_next_oid = next_oid;
        for sequence in &implicit_sequences {
            self.create_implicit_sequence(cat, sequence)?;
        }
        let next_oid = cat.relational_next_oid.max(next_oid);
        cat.relational_catalog.insert(
            name.clone(),
            RelationalTable {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name,
                oid,
                columns,
                indexes,
                check_constraints: checks,
                foreign_keys: Vec::new(),
                acl: cat.relational_default_table_acl.clone(),
            },
        );
        cat.relational_next_oid = next_oid;
        cat.relational_next_column_id = next_column_id;
        Ok(())
    }

    fn apply_add_primary_key(
        &self,
        cat: &mut DdlCatalogState,
        add: gpu_db_sql::AddPrimaryKey,
    ) -> Result<(), EngineError> {
        if cat
            .relational_catalog
            .values()
            .any(|table| table.indexes.iter().any(|index| index.name == add.name))
            || cat.relational_catalog.contains_key(&add.name)
            || cat.relational_views.contains_key(&add.name)
            || cat.relational_materialized_views.contains_key(&add.name)
            || cat.relational_sequences.contains_key(&add.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                add.name
            )));
        }
        if cat
            .relational_catalog
            .get(&add.table)
            .is_some_and(|table| table.indexes.iter().any(|index| index.primary_key))
        {
            return Err(EngineError::ApplyFailed(format!(
                "multiple primary keys for table \"{}\" are not allowed",
                add.table
            )));
        }
        let create = CreateIndex {
            name: add.name,
            table: add.table,
            column: add.column,
            unique: true,
        };
        self.apply_create_index_with_constraint_flags(cat, create, true, false)
    }

    fn apply_create_index(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateIndex,
    ) -> Result<(), EngineError> {
        self.apply_create_index_with_constraint_flags(cat, create, false, false)
    }

    fn apply_create_index_with_constraint_flags(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateIndex,
        primary_key: bool,
        unique_constraint: bool,
    ) -> Result<(), EngineError> {
        if cat
            .relational_catalog
            .values()
            .any(|table| table.indexes.iter().any(|index| index.name == create.name))
            || cat.relational_catalog.contains_key(&create.name)
            || cat.relational_views.contains_key(&create.name)
            || cat.relational_materialized_views.contains_key(&create.name)
            || cat.relational_sequences.contains_key(&create.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                create.name
            )));
        }
        let table = cat
            .relational_catalog
            .get(&create.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", create.table))
            })?
            .clone();
        let Some(column_idx) = table
            .columns
            .iter()
            .position(|column| column.name == create.column)
        else {
            return Err(EngineError::ApplyFailed(format!(
                "column \"{}\" does not exist",
                create.column
            )));
        };
        if create.unique {
            let visibility = StorageVisibility {
                read_txn_id: self.committed_seq() as TxnId,
            };
            let rows = self.visible_relational_rows(&table, visibility)?;
            Self::validate_unique_values(&rows, column_idx, &create.name)?;
        }
        cat.relational_catalog
            .get_mut(&create.table)
            .expect("table existence validated")
            .indexes
            .push(RelationalIndex {
                name: create.name,
                table: create.table,
                column: create.column,
                unique: create.unique,
                primary_key,
                unique_constraint,
            });
        Ok(())
    }

    fn apply_add_unique_constraint(
        &self,
        cat: &mut DdlCatalogState,
        add: AddUniqueConstraint,
    ) -> Result<(), EngineError> {
        let create = CreateIndex {
            name: add.name,
            table: add.table,
            column: add.column,
            unique: true,
        };
        self.apply_create_index_with_constraint_flags(cat, create, false, true)
    }

    fn apply_add_check_constraint(
        &self,
        cat: &mut DdlCatalogState,
        add: AddCheckConstraint,
    ) -> Result<(), EngineError> {
        self.preflight_add_check_constraint(&add)?;
        let table = cat
            .relational_catalog
            .get_mut(&add.table)
            .expect("table existence preflighted");
        table.check_constraints.push(RelationalCheckConstraint {
            name: add.name,
            column: add.filter.column,
            op: add.filter.op,
            value: add.filter.value,
        });
        Ok(())
    }

    fn apply_add_foreign_key(
        &self,
        cat: &mut DdlCatalogState,
        add: AddForeignKey,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        self.preflight_add_foreign_key(&add, txn_id)?;
        let table = cat
            .relational_catalog
            .get_mut(&add.table)
            .expect("table existence preflighted");
        table.foreign_keys.push(RelationalForeignKey {
            name: add.name,
            column: add.column,
            referenced_table: add.referenced_table,
            referenced_column: add.referenced_column,
        });
        Ok(())
    }

    fn apply_drop_constraint(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropConstraint,
    ) -> Result<(), EngineError> {
        let Some(table) = cat.relational_catalog.get_mut(&drop.table) else {
            if drop.table_if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" does not exist",
                drop.table
            )));
        };
        let old_index_len = table.indexes.len();
        table.indexes.retain(|index| {
            !(index.name == drop.name && (index.primary_key || index.unique_constraint))
        });
        let old_check_len = table.check_constraints.len();
        table
            .check_constraints
            .retain(|constraint| constraint.name != drop.name);
        let old_foreign_key_len = table.foreign_keys.len();
        table
            .foreign_keys
            .retain(|constraint| constraint.name != drop.name);
        if table.indexes.len() == old_index_len
            && table.check_constraints.len() == old_check_len
            && table.foreign_keys.len() == old_foreign_key_len
        {
            if drop.if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "constraint \"{}\" does not exist",
                drop.name
            )));
        }
        cat.relational_comments
            .remove(&RelationalCommentTarget::Index {
                index: drop.name.clone(),
            });
        cat.relational_comments
            .remove(&RelationalCommentTarget::Constraint {
                table: drop.table,
                constraint: drop.name,
            });
        Ok(())
    }

    fn apply_rename_constraint(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameConstraint,
    ) -> Result<(), EngineError> {
        if cat.relational_views.contains_key(&rename.table)
            || cat
                .relational_materialized_views
                .contains_key(&rename.table)
            || cat.relational_sequences.contains_key(&rename.table)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a table",
                rename.table
            )));
        }
        if !cat.relational_catalog.contains_key(&rename.table) {
            if rename.table_if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" does not exist",
                rename.table
            )));
        }
        if cat.relational_catalog.values().any(|candidate| {
            candidate
                .indexes
                .iter()
                .any(|index| index.name == rename.new_name)
        }) || cat.relational_catalog.contains_key(&rename.new_name)
            || cat.relational_views.contains_key(&rename.new_name)
            || cat
                .relational_materialized_views
                .contains_key(&rename.new_name)
            || cat.relational_sequences.contains_key(&rename.new_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                rename.new_name
            )));
        }
        let table = cat
            .relational_catalog
            .get_mut(&rename.table)
            .expect("table existence validated");
        if let Some(index) = table.indexes.iter_mut().find(|index| {
            index.name == rename.old_name && (index.primary_key || index.unique_constraint)
        }) {
            index.name = rename.new_name.clone();
        } else if let Some(check) = table
            .check_constraints
            .iter_mut()
            .find(|constraint| constraint.name == rename.old_name)
        {
            check.name = rename.new_name.clone();
        } else if let Some(foreign_key) = table
            .foreign_keys
            .iter_mut()
            .find(|constraint| constraint.name == rename.old_name)
        {
            foreign_key.name = rename.new_name.clone();
        } else {
            return Err(EngineError::ApplyFailed(format!(
                "constraint \"{}\" does not exist",
                rename.old_name
            )));
        };

        let old_index_target = RelationalCommentTarget::Index {
            index: rename.old_name.clone(),
        };
        if let Some(comment) = cat.relational_comments.remove(&old_index_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::Index {
                    index: rename.new_name.clone(),
                },
                comment,
            );
        }
        let old_constraint_target = RelationalCommentTarget::Constraint {
            table: rename.table.clone(),
            constraint: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_constraint_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::Constraint {
                    table: rename.table,
                    constraint: rename.new_name,
                },
                comment,
            );
        }
        Ok(())
    }

    fn visible_relational_rows(
        &self,
        table: &RelationalTable,
        visibility: StorageVisibility,
    ) -> Result<Vec<Vec<SqlValue>>, EngineError> {
        let prefix = relational_key_prefix(&table.name);
        let mut rows = Vec::new();
        // Load this table's published MVCC generation; the cursor reads its immutable rows
        // lock-free (the prefix filter is redundant now each partition is single-table, but kept
        // so the read stays correct regardless of partition contents — write-half Stage 3).
        let table_rows = self.read_state.mvcc.table_rows(&table.name);
        let mut cursor = table_rows
            .store()
            .seq_scan_open(visibility)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        while let Some(tuple) = cursor.next() {
            if tuple.key.starts_with(&prefix) {
                rows.push(
                    decode_relational_row(&tuple.value, &table.columns)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?,
                );
            }
        }
        Ok(rows)
    }

    fn validate_unique_values(
        rows: &[Vec<SqlValue>],
        column_idx: usize,
        index_name: &str,
    ) -> Result<(), EngineError> {
        let mut seen = BTreeSet::new();
        for row in rows {
            if !seen.insert(row[column_idx].clone()) {
                return Err(EngineError::ApplyFailed(format!(
                    "duplicate key value violates unique index \"{}\"",
                    index_name
                )));
            }
        }
        Ok(())
    }

    fn validate_unique_indexes_for_rows(
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(), EngineError> {
        for index in table.indexes.iter().filter(|index| index.unique) {
            let Some(column_idx) = table
                .columns
                .iter()
                .position(|column| column.name == index.column)
            else {
                continue;
            };
            Self::validate_unique_values(rows, column_idx, &index.name)?;
        }
        Ok(())
    }

    fn validate_check_constraints_for_rows(
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(), EngineError> {
        for constraint in &table.check_constraints {
            let column_idx = relational_column_index(table, &constraint.column)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            for row in rows {
                if !select_filter_matches(&row[column_idx], constraint.op, &constraint.value) {
                    return Err(EngineError::ApplyFailed(format!(
                        "new row for relation \"{}\" violates check constraint \"{}\"",
                        table.name, constraint.name
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_foreign_keys_with_table_rows(
        &self,
        changed_table: &str,
        changed_rows: &[Vec<SqlValue>],
        visibility: StorageVisibility,
    ) -> Result<(), EngineError> {
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): pin ONE catalog snapshot for the whole
        // cross-table FK scan so a concurrent DDL cannot change FK definitions mid-validation.
        let catalog = self.catalog_snapshot();
        for child_table in catalog.relational_catalog.values() {
            let child_rows = if child_table.name == changed_table {
                changed_rows.to_vec()
            } else {
                self.visible_relational_rows(child_table, visibility)?
            };
            for foreign_key in &child_table.foreign_keys {
                let Some(parent_table) = catalog
                    .relational_catalog
                    .get(&foreign_key.referenced_table)
                else {
                    continue;
                };
                let parent_rows = if parent_table.name == changed_table {
                    changed_rows.to_vec()
                } else {
                    self.visible_relational_rows(parent_table, visibility)?
                };
                Self::validate_foreign_key_rows(
                    child_table,
                    &child_rows,
                    parent_table,
                    &parent_rows,
                    foreign_key,
                )?;
            }
        }
        Ok(())
    }

    fn validate_foreign_key_rows(
        child_table: &RelationalTable,
        child_rows: &[Vec<SqlValue>],
        parent_table: &RelationalTable,
        parent_rows: &[Vec<SqlValue>],
        foreign_key: &RelationalForeignKey,
    ) -> Result<(), EngineError> {
        let child_column_idx = relational_column_index(child_table, &foreign_key.column)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let parent_column_idx =
            relational_column_index(parent_table, &foreign_key.referenced_column)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let parent_values = parent_rows
            .iter()
            .map(|row| row[parent_column_idx].clone())
            .collect::<BTreeSet<_>>();
        for row in child_rows {
            if !parent_values.contains(&row[child_column_idx]) {
                return Err(EngineError::ApplyFailed(format!(
                    "insert or update on table \"{}\" violates foreign key constraint \"{}\"",
                    child_table.name, foreign_key.name
                )));
            }
        }
        Ok(())
    }

    fn preflight_add_check_constraint(&self, add: &AddCheckConstraint) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
            EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
        })?;
        if cat.relational_catalog.values().any(|candidate| {
            candidate.indexes.iter().any(|index| index.name == add.name)
                || candidate
                    .check_constraints
                    .iter()
                    .any(|constraint| constraint.name == add.name)
                || candidate
                    .foreign_keys
                    .iter()
                    .any(|constraint| constraint.name == add.name)
        }) || cat.relational_catalog.contains_key(&add.name)
            || cat.relational_views.contains_key(&add.name)
            || cat.relational_materialized_views.contains_key(&add.name)
            || cat.relational_sequences.contains_key(&add.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "constraint \"{}\" already exists",
                add.name
            )));
        }
        let column_idx = relational_column_index(table, &add.filter.column)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        if !sql_value_matches_type(&add.filter.value, table.columns[column_idx].ty) {
            return Err(EngineError::ApplyFailed(format!(
                "invalid value for column \"{}\"",
                add.filter.column
            )));
        }
        let rows = self.visible_relational_rows(
            table,
            StorageVisibility {
                read_txn_id: self.committed_seq() as TxnId,
            },
        )?;
        for row in rows {
            if !select_filter_matches(&row[column_idx], add.filter.op, &add.filter.value) {
                return Err(EngineError::ApplyFailed(format!(
                    "check constraint \"{}\" is violated by some row",
                    add.name
                )));
            }
        }
        Ok(())
    }

    fn preflight_add_foreign_key(
        &self,
        add: &AddForeignKey,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if add.table == add.referenced_table {
            return Err(EngineError::ApplyFailed(
                "self-referential foreign keys are not supported".to_string(),
            ));
        }
        let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
            EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
        })?;
        let referenced_table = cat
            .relational_catalog
            .get(&add.referenced_table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "relation \"{}\" does not exist",
                    add.referenced_table
                ))
            })?;
        if cat.relational_catalog.values().any(|candidate| {
            candidate.indexes.iter().any(|index| index.name == add.name)
                || candidate
                    .check_constraints
                    .iter()
                    .any(|constraint| constraint.name == add.name)
                || candidate
                    .foreign_keys
                    .iter()
                    .any(|constraint| constraint.name == add.name)
        }) || cat.relational_catalog.contains_key(&add.name)
            || cat.relational_views.contains_key(&add.name)
            || cat.relational_materialized_views.contains_key(&add.name)
            || cat.relational_sequences.contains_key(&add.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "constraint \"{}\" already exists",
                add.name
            )));
        }
        let column_idx = relational_column_index(table, &add.column)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let referenced_column_idx =
            relational_column_index(referenced_table, &add.referenced_column)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        if table.columns[column_idx].ty != referenced_table.columns[referenced_column_idx].ty {
            return Err(EngineError::ApplyFailed(
                "foreign key column type does not match referenced column type".to_string(),
            ));
        }
        let has_referenced_unique_key = referenced_table.indexes.iter().any(|index| {
            index.column == add.referenced_column && (index.primary_key || index.unique_constraint)
        });
        if !has_referenced_unique_key {
            return Err(EngineError::ApplyFailed(format!(
                "there is no unique constraint matching given keys for referenced table \"{}\"",
                add.referenced_table
            )));
        }
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let child_rows = self.visible_relational_rows(table, visibility)?;
        let parent_rows = self.visible_relational_rows(referenced_table, visibility)?;
        Self::validate_foreign_key_rows(
            table,
            &child_rows,
            referenced_table,
            &parent_rows,
            &RelationalForeignKey {
                name: add.name.clone(),
                column: add.column.clone(),
                referenced_table: add.referenced_table.clone(),
                referenced_column: add.referenced_column.clone(),
            },
        )?;
        Ok(())
    }

    fn apply_drop_index(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropIndex,
    ) -> Result<(), EngineError> {
        if !drop.if_exists {
            for name in &drop.names {
                if !cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == *name))
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "index \"{}\" does not exist",
                        name
                    )));
                }
            }
        }
        let drop_names = drop.names.iter().cloned().collect::<BTreeSet<_>>();
        let mut dropped_constraints = Vec::new();
        for table in cat.relational_catalog.values_mut() {
            dropped_constraints.extend(
                table
                    .indexes
                    .iter()
                    .filter(|index| {
                        drop_names.contains(&index.name)
                            && (index.primary_key || index.unique_constraint)
                    })
                    .map(|index| (index.table.clone(), index.name.clone())),
            );
            table
                .indexes
                .retain(|index| !drop_names.contains(&index.name));
        }
        for name in &drop.names {
            cat.relational_comments
                .remove(&RelationalCommentTarget::Index {
                    index: name.clone(),
                });
        }
        for (table, constraint) in dropped_constraints {
            cat.relational_comments
                .remove(&RelationalCommentTarget::Constraint { table, constraint });
        }
        Ok(())
    }

    fn preflight_drop_index(&self, drop: &DropIndex) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if drop.if_exists {
            return Ok(());
        }
        for name in &drop.names {
            if !cat
                .relational_catalog
                .values()
                .any(|table| table.indexes.iter().any(|index| index.name == *name))
            {
                return Err(EngineError::ApplyFailed(format!(
                    "index \"{}\" does not exist",
                    name
                )));
            }
        }
        let drop_names = drop.names.iter().cloned().collect::<BTreeSet<_>>();
        if cat.relational_catalog.values().any(|table| {
            table.foreign_keys.iter().any(|constraint| {
                drop_names.contains(&table.name)
                    || drop_names.contains(&constraint.referenced_table)
            })
        }) {
            return Err(EngineError::ApplyFailed(
                "cannot drop table because a foreign key constraint depends on it".to_string(),
            ));
        }
        Ok(())
    }

    fn apply_rename_index(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameIndex,
    ) -> Result<(), EngineError> {
        if cat.relational_catalog.values().any(|table| {
            table
                .indexes
                .iter()
                .any(|index| index.name == rename.new_name)
        }) {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                rename.new_name
            )));
        }

        for table in cat.relational_catalog.values_mut() {
            let Some(index) = table
                .indexes
                .iter_mut()
                .find(|index| index.name == rename.old_name)
            else {
                continue;
            };
            if index.primary_key || index.unique_constraint {
                return Err(EngineError::ApplyFailed(format!(
                    "cannot rename constraint-backed index \"{}\" with ALTER INDEX",
                    rename.old_name
                )));
            }
            index.name = rename.new_name.clone();
            let old_target = RelationalCommentTarget::Index {
                index: rename.old_name,
            };
            if let Some(comment) = cat.relational_comments.remove(&old_target) {
                cat.relational_comments.insert(
                    RelationalCommentTarget::Index {
                        index: rename.new_name,
                    },
                    comment,
                );
            }
            return Ok(());
        }

        Err(EngineError::ApplyFailed(format!(
            "index \"{}\" does not exist",
            rename.old_name
        )))
    }

    fn apply_rename_table(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameTable,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        if cat.relational_views.contains_key(&rename.old_name)
            || cat
                .relational_materialized_views
                .contains_key(&rename.old_name)
            || cat.relational_sequences.contains_key(&rename.old_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a table",
                rename.old_name
            )));
        }
        if !cat.relational_catalog.contains_key(&rename.old_name) {
            if rename.if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" does not exist",
                rename.old_name
            )));
        }
        if cat.relational_catalog.contains_key(&rename.new_name)
            || cat.relational_views.contains_key(&rename.new_name)
            || cat
                .relational_materialized_views
                .contains_key(&rename.new_name)
            || cat.relational_sequences.contains_key(&rename.new_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                rename.new_name
            )));
        }
        if cat
            .relational_views
            .values()
            .any(|view| view.query.table == rename.old_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "cannot rename relation \"{}\" because a view depends on it",
                rename.old_name
            )));
        }
        let Some(mut table) = cat.relational_catalog.remove(&rename.old_name) else {
            return Ok(());
        };

        let old_prefix = relational_key_prefix(&rename.old_name);
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        // Read the rows to move out of the OLD partition's published generation.
        let mut moves = Vec::new();
        {
            let old_rows = self.read_state.mvcc.table_rows(&rename.old_name);
            let mut cursor = old_rows
                .store()
                .seq_scan_open(visibility)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            while let Some(tuple) = cursor.next() {
                if tuple.key.starts_with(&old_prefix) {
                    moves.push((tuple.tuple_id, tuple.value.clone()));
                }
            }
        }

        // Build the moved rows + their value-index for the NEW table (decoded against the renamed
        // table's columns), reserving fresh global tuple ids and relational row ids — identical to
        // the old in-line per-row bumps.
        table.name = rename.new_name.clone();
        for index in &mut table.indexes {
            index.table = rename.new_name.clone();
        }
        for foreign_key in &mut table.foreign_keys {
            if foreign_key.referenced_table == rename.old_name {
                foreign_key.referenced_table = rename.new_name.clone();
            }
        }
        let mut new_rows: Vec<(TupleId, String, String)> = Vec::with_capacity(moves.len());
        let mut new_value_index: BTreeMap<ColumnValueKey, Vec<String>> = BTreeMap::new();
        let old_tuple_ids: Vec<TupleId> = moves.iter().map(|(tuple_id, _)| *tuple_id).collect();
        for (_old_tuple_id, value) in &moves {
            let row_id = self.read_state.mvcc.current_row_id();
            self.read_state.mvcc.advance_row_id(1);
            let new_key = relational_row_key(&rename.new_name, row_id);
            let new_tuple_id = self.read_state.mvcc.reserve_tuple_id();
            let values = decode_relational_row(value, &table.columns)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            for (column, column_value) in table.columns.iter().zip(values.iter()) {
                new_value_index
                    .entry(ColumnValueKey {
                        column: column.name.clone(),
                        value: relational_index_value(column_value),
                    })
                    .or_default()
                    .push(new_key.clone());
            }
            new_rows.push((new_tuple_id, new_key, value.clone()));
        }

        // Publish the NEW table partition with the moved rows + rebuilt value-index.
        self.read_state
            .mvcc
            .with_table_mut(&rename.new_name, |data| {
                for (tuple_id, new_key, value) in &new_rows {
                    data.rows
                        .tuple_insert_with_id(
                            *tuple_id,
                            NewTuple {
                                key: new_key.clone(),
                                value: value.clone(),
                            },
                            txn_id,
                        )
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
                for (key, keys) in &new_value_index {
                    let mut slot = data.value_index.get(key).cloned().unwrap_or_default();
                    std::sync::Arc::make_mut(&mut slot).extend(keys.iter().cloned());
                    data.value_index.insert(key.clone(), slot);
                }
                Ok::<(), EngineError>(())
            })?;

        // Tombstone the moved rows in the OLD partition (keeping their history, exactly as the
        // old in-line `tuple_delete` did) and clear the old partition's value-index.
        self.read_state
            .mvcc
            .with_table_mut(&rename.old_name, |data| {
                for old_tuple_id in &old_tuple_ids {
                    data.rows
                        .tuple_delete(*old_tuple_id, txn_id)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
                data.value_index = imbl::OrdMap::new();
                Ok::<(), EngineError>(())
            })?;

        cat.relational_catalog
            .insert(rename.new_name.clone(), table.clone());
        for candidate in cat.relational_catalog.values_mut() {
            for foreign_key in &mut candidate.foreign_keys {
                if foreign_key.referenced_table == rename.old_name {
                    foreign_key.referenced_table = rename.new_name.clone();
                }
            }
        }

        let mut retargeted_comments = Vec::new();
        cat.relational_comments
            .retain(|target, comment| match target {
                RelationalCommentTarget::Table { table } if table == &rename.old_name => {
                    retargeted_comments.push((
                        RelationalCommentTarget::Table {
                            table: rename.new_name.clone(),
                        },
                        comment.clone(),
                    ));
                    false
                }
                RelationalCommentTarget::Column { table, attnum } if table == &rename.old_name => {
                    retargeted_comments.push((
                        RelationalCommentTarget::Column {
                            table: rename.new_name.clone(),
                            attnum: *attnum,
                        },
                        comment.clone(),
                    ));
                    false
                }
                RelationalCommentTarget::Constraint { table, constraint }
                    if table == &rename.old_name =>
                {
                    retargeted_comments.push((
                        RelationalCommentTarget::Constraint {
                            table: rename.new_name.clone(),
                            constraint: constraint.clone(),
                        },
                        comment.clone(),
                    ));
                    false
                }
                _ => true,
            });
        for (target, comment) in retargeted_comments {
            cat.relational_comments.insert(target, comment);
        }
        self.read_state
            .residency
            .with_snapshots_mut(|snapshots| snapshots.remove(&rename.old_name));
        self.read_state
            .residency
            .device_memory
            .remove(&rename.old_name);
        Ok(())
    }

    fn apply_drop_table(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropTable,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        self.preflight_drop_table(&drop)?;

        for name in &drop.names {
            let Some(table) = cat.relational_catalog.remove(name) else {
                continue;
            };
            let prefix = relational_key_prefix(&table.name);
            let visibility = StorageVisibility {
                read_txn_id: txn_id,
            };
            let mut tuple_ids = Vec::new();
            {
                let table_rows = self.read_state.mvcc.table_rows(name);
                let mut cursor = table_rows
                    .store()
                    .seq_scan_open(visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                while let Some(tuple) = cursor.next() {
                    if tuple.key.starts_with(&prefix) {
                        tuple_ids.push(tuple.tuple_id);
                    }
                }
            }
            // Tombstone the dropped table's rows in place (keeping their version history, exactly
            // as the old single-store `tuple_delete` did), publishing one new generation. The
            // partition cell + (now-stale) value-index are retained — `all_versions` still sees the
            // tombstoned versions, matching the pre-partition behavior.
            if !tuple_ids.is_empty() {
                self.read_state.mvcc.with_table_mut(name, |data| {
                    for tuple_id in &tuple_ids {
                        data.rows
                            .tuple_delete(*tuple_id, txn_id)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    Ok::<(), EngineError>(())
                })?;
            }

            let index_names = table
                .indexes
                .iter()
                .map(|index| index.name.clone())
                .collect::<BTreeSet<_>>();
            cat.relational_comments.retain(|target, _| match target {
                RelationalCommentTarget::Table { table }
                | RelationalCommentTarget::Column { table, .. }
                | RelationalCommentTarget::Constraint { table, .. } => table != name,
                RelationalCommentTarget::Index { index } => !index_names.contains(index),
                RelationalCommentTarget::Database { .. }
                | RelationalCommentTarget::Role { .. }
                | RelationalCommentTarget::Schema { .. }
                | RelationalCommentTarget::Tablespace { .. }
                | RelationalCommentTarget::View { .. }
                | RelationalCommentTarget::MaterializedView { .. }
                | RelationalCommentTarget::Extension { .. }
                | RelationalCommentTarget::Function { .. }
                | RelationalCommentTarget::Sequence { .. }
                | RelationalCommentTarget::Domain { .. }
                | RelationalCommentTarget::Publication { .. }
                | RelationalCommentTarget::Subscription { .. } => true,
            });
            self.read_state
                .residency
                .with_snapshots_mut(|snapshots| snapshots.remove(name));
            self.read_state.residency.device_memory.remove(name);
        }
        Ok(())
    }

    fn preflight_drop_table(&self, drop: &DropTable) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "table \"{}\" specified more than once",
                    name
                )));
            }
            if cat.relational_views.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a table",
                    name
                )));
            }
            if cat.relational_sequences.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a table",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_catalog.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" does not exist",
                    name
                )));
            }
        }
        Ok(())
    }

    fn apply_truncate_table(
        &self,
        cat: &mut DdlCatalogState,
        truncate: TruncateTable,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        if cat.relational_views.contains_key(&truncate.name)
            || cat
                .relational_materialized_views
                .contains_key(&truncate.name)
            || cat.relational_sequences.contains_key(&truncate.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a table",
                truncate.name
            )));
        }
        let table = cat
            .relational_catalog
            .get(&truncate.name)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", truncate.name))
            })?
            .clone();
        if cat.relational_catalog.values().any(|candidate| {
            candidate
                .foreign_keys
                .iter()
                .any(|foreign_key| foreign_key.referenced_table == table.name)
        }) {
            self.validate_foreign_keys_with_table_rows(
                &table.name,
                &[],
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?;
        }
        let restart_sequences = if truncate.restart_identity {
            table
                .columns
                .iter()
                .filter_map(|column| match &column.default {
                    Some(ColumnDefault::SequenceNextVal { sequence, .. }) => Some(sequence.clone()),
                    _ => None,
                })
                .collect::<BTreeSet<_>>()
        } else {
            BTreeSet::new()
        };
        for sequence in &restart_sequences {
            if !cat.relational_sequences.contains_key(sequence) {
                return Err(EngineError::ApplyFailed(format!(
                    "sequence \"{sequence}\" does not exist"
                )));
            }
        }

        let prefix = relational_key_prefix(&table.name);
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let mut tuple_ids = Vec::new();
        {
            let table_rows = self.read_state.mvcc.table_rows(&truncate.name);
            let mut cursor = table_rows
                .store()
                .seq_scan_open(visibility)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            while let Some(tuple) = cursor.next() {
                if tuple.key.starts_with(&prefix) {
                    tuple_ids.push(tuple.tuple_id);
                }
            }
        }
        // Tombstone all rows in place (keeping history), publishing one new generation. As before,
        // the value-index is left as-is; its now-stale entries point to tombstoned rows and are
        // filtered out by visibility + the predicate recheck.
        if !tuple_ids.is_empty() {
            self.read_state
                .mvcc
                .with_table_mut(&truncate.name, |data| {
                    for tuple_id in &tuple_ids {
                        data.rows
                            .tuple_delete(*tuple_id, txn_id)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    Ok::<(), EngineError>(())
                })?;
        }
        if truncate.restart_identity {
            for sequence in restart_sequences {
                let sequence_state = cat
                    .relational_sequences
                    .get_mut(&sequence)
                    .expect("restart identity sequence preflighted");
                sequence_state.last_value = 1;
                sequence_state.is_called = false;
            }
        }
        Ok(())
    }

    fn apply_drop_view(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropView,
    ) -> Result<(), EngineError> {
        self.preflight_drop_view(&drop)?;
        for name in &drop.names {
            if cat.relational_views.remove(name).is_none() {
                continue;
            }
            cat.relational_comments
                .remove(&RelationalCommentTarget::View { view: name.clone() });
        }
        Ok(())
    }

    fn apply_drop_materialized_view(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropMaterializedView,
    ) -> Result<(), EngineError> {
        self.preflight_drop_materialized_view(&drop)?;
        for name in &drop.names {
            if cat.relational_materialized_views.remove(name).is_none() {
                continue;
            }
            cat.relational_comments
                .remove(&RelationalCommentTarget::MaterializedView {
                    materialized_view: name.clone(),
                });
        }
        Ok(())
    }

    fn apply_drop_sequence(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropSequence,
    ) -> Result<(), EngineError> {
        self.preflight_drop_sequence(&drop)?;
        for name in &drop.names {
            if cat.relational_sequences.remove(name).is_none() {
                continue;
            }
            cat.relational_comments
                .remove(&RelationalCommentTarget::Sequence {
                    sequence: name.clone(),
                });
        }
        Ok(())
    }

    fn preflight_create_publication(&self, create: &CreatePublication) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_publications.contains_key(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "publication \"{}\" already exists",
                create.name
            )));
        }
        if let PublicationTarget::Tables(tables) = &create.target {
            let mut seen = BTreeSet::new();
            for table in tables {
                if !seen.insert(table) {
                    return Err(EngineError::ApplyFailed(format!(
                        "table \"{}\" specified more than once",
                        table
                    )));
                }
                self.preflight_table_acl_target(table)?;
            }
        }
        Ok(())
    }

    fn apply_create_publication(
        &self,
        cat: &mut DdlCatalogState,
        create: CreatePublication,
    ) -> Result<(), EngineError> {
        self.preflight_create_publication(&create)?;
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational publication OID allocation exhausted".to_string())
        })?;
        let (all_tables, tables) = match create.target {
            PublicationTarget::AllTables => (true, Vec::new()),
            PublicationTarget::Tables(tables) => (false, tables),
        };
        cat.relational_publications.insert(
            create.name.clone(),
            RelationalPublication {
                name: create.name,
                oid,
                all_tables,
                tables,
            },
        );
        Ok(())
    }

    fn role_exists(&self, role: &str) -> bool {
        let cat = self.catalog_snapshot();
        role == "postgres" || cat.relational_roles.contains_key(role)
    }

    fn apply_create_role(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateRole,
    ) -> Result<(), EngineError> {
        if create.name == "postgres" || cat.relational_roles.contains_key(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "role \"{}\" already exists",
                create.name
            )));
        }
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational OID counter overflow".to_string())
        })?;
        cat.relational_roles.insert(
            create.name.clone(),
            RelationalRole {
                name: create.name,
                oid,
                login: create.login,
            },
        );
        Ok(())
    }

    fn role_has_dependencies(&self, role: &str) -> bool {
        let cat = self.catalog_snapshot();
        cat.relational_comments
            .contains_key(&RelationalCommentTarget::Role {
                role: role.to_string(),
            })
            || cat
                .relational_catalog
                .values()
                .any(|table| table.acl.contains_key(role))
            || cat
                .relational_views
                .values()
                .any(|view| view.acl.contains_key(role))
            || cat
                .relational_materialized_views
                .values()
                .any(|view| view.acl.contains_key(role))
            || cat
                .relational_sequences
                .values()
                .any(|sequence| sequence.acl.contains_key(role))
            || cat
                .relational_databases
                .values()
                .any(|database| database.acl.contains_key(role))
            || cat
                .relational_tablespaces
                .values()
                .any(|tablespace| tablespace.acl.contains_key(role))
            || cat
                .relational_functions
                .values()
                .any(|function| function.acl.contains_key(role))
            || cat.relational_schema_acl.contains_key(role)
            || cat.relational_default_table_acl.contains_key(role)
    }

    fn apply_drop_role(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropRole,
    ) -> Result<(), EngineError> {
        let mut seen = BTreeSet::new();
        for role in &drop.names {
            if !seen.insert(role.clone()) {
                return Err(EngineError::ApplyFailed(format!(
                    "role \"{}\" specified more than once",
                    role
                )));
            }
            if role == "postgres" {
                return Err(EngineError::ApplyFailed(
                    "cannot drop bootstrap role \"postgres\"".to_string(),
                ));
            }
            if !drop.if_exists && !cat.relational_roles.contains_key(role) {
                return Err(EngineError::ApplyFailed(format!(
                    "role \"{}\" does not exist",
                    role
                )));
            }
            if cat.relational_roles.contains_key(role) && self.role_has_dependencies(role) {
                return Err(EngineError::ApplyFailed(format!(
                    "role \"{}\" cannot be dropped because dependent metadata exists",
                    role
                )));
            }
        }
        for role in drop.names {
            cat.relational_roles.remove(&role);
        }
        Ok(())
    }

    fn apply_rename_role(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameRole,
    ) -> Result<(), EngineError> {
        if rename.old_name == "postgres" {
            return Err(EngineError::ApplyFailed(
                "cannot rename bootstrap role \"postgres\"".to_string(),
            ));
        }
        if !cat.relational_roles.contains_key(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "role \"{}\" does not exist",
                rename.old_name
            )));
        }
        if self.role_exists(&rename.new_name) {
            return Err(EngineError::ApplyFailed(format!(
                "role \"{}\" already exists",
                rename.new_name
            )));
        }
        let mut role = cat
            .relational_roles
            .remove(&rename.old_name)
            .expect("role existence checked");
        role.name = rename.new_name.clone();
        cat.relational_roles.insert(rename.new_name.clone(), role);
        let old_target = RelationalCommentTarget::Role {
            role: rename.old_name.clone(),
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            let new_target = RelationalCommentTarget::Role {
                role: rename.new_name.clone(),
            };
            cat.relational_comments.insert(new_target, comment);
        }
        for table in cat.relational_catalog.values_mut() {
            if let Some(privileges) = table.acl.remove(&rename.old_name) {
                table.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for view in cat.relational_views.values_mut() {
            if let Some(privileges) = view.acl.remove(&rename.old_name) {
                view.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for view in cat.relational_materialized_views.values_mut() {
            if let Some(privileges) = view.acl.remove(&rename.old_name) {
                view.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for sequence in cat.relational_sequences.values_mut() {
            if let Some(privileges) = sequence.acl.remove(&rename.old_name) {
                sequence.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for database in cat.relational_databases.values_mut() {
            if let Some(privileges) = database.acl.remove(&rename.old_name) {
                database.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for tablespace in cat.relational_tablespaces.values_mut() {
            if let Some(privileges) = tablespace.acl.remove(&rename.old_name) {
                tablespace.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for function in cat.relational_functions.values_mut() {
            if let Some(privileges) = function.acl.remove(&rename.old_name) {
                function.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        if let Some(privileges) = cat.relational_schema_acl.remove(&rename.old_name) {
            cat.relational_schema_acl
                .insert(rename.new_name.clone(), privileges);
        }
        if let Some(privileges) = cat.relational_default_table_acl.remove(&rename.old_name) {
            cat.relational_default_table_acl
                .insert(rename.new_name, privileges);
        }
        Ok(())
    }

    fn preflight_drop_publication(&self, drop: &DropPublication) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "publication \"{}\" specified more than once",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_publications.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "publication \"{}\" does not exist",
                    name
                )));
            }
        }
        Ok(())
    }

    fn apply_drop_publication(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropPublication,
    ) -> Result<(), EngineError> {
        self.preflight_drop_publication(&drop)?;
        for name in &drop.names {
            cat.relational_publications.remove(name);
            cat.relational_comments
                .remove(&RelationalCommentTarget::Publication {
                    publication: name.clone(),
                });
        }
        Ok(())
    }

    fn preflight_create_subscription(
        &self,
        create: &CreateSubscription,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_subscriptions.contains_key(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "subscription \"{}\" already exists",
                create.name
            )));
        }
        let mut seen = BTreeSet::new();
        for publication in &create.publications {
            if !seen.insert(publication) {
                return Err(EngineError::ApplyFailed(format!(
                    "publication \"{}\" specified more than once",
                    publication
                )));
            }
            if !cat.relational_publications.contains_key(publication) {
                return Err(EngineError::ApplyFailed(format!(
                    "publication \"{}\" does not exist",
                    publication
                )));
            }
        }
        Ok(())
    }

    fn apply_create_subscription(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateSubscription,
    ) -> Result<(), EngineError> {
        self.preflight_create_subscription(&create)?;
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational subscription OID allocation exhausted".to_string())
        })?;
        cat.relational_subscriptions.insert(
            create.name.clone(),
            RelationalSubscription {
                name: create.name,
                oid,
                connection: create.connection,
                publications: create.publications,
                enabled: false,
            },
        );
        Ok(())
    }

    fn preflight_drop_subscription(&self, drop: &DropSubscription) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "subscription \"{}\" specified more than once",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_subscriptions.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "subscription \"{}\" does not exist",
                    name
                )));
            }
        }
        Ok(())
    }

    fn apply_drop_subscription(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropSubscription,
    ) -> Result<(), EngineError> {
        self.preflight_drop_subscription(&drop)?;
        for name in &drop.names {
            cat.relational_subscriptions.remove(name);
            cat.relational_comments
                .remove(&RelationalCommentTarget::Subscription {
                    subscription: name.clone(),
                });
        }
        Ok(())
    }

    fn apply_rename_sequence(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameSequence,
    ) -> Result<(), EngineError> {
        if cat.relational_catalog.contains_key(&rename.old_name)
            || cat.relational_views.contains_key(&rename.old_name)
            || cat
                .relational_materialized_views
                .contains_key(&rename.old_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a sequence",
                rename.old_name
            )));
        }
        if !cat.relational_sequences.contains_key(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "sequence \"{}\" does not exist",
                rename.old_name
            )));
        }
        if cat.relational_catalog.contains_key(&rename.new_name)
            || cat.relational_views.contains_key(&rename.new_name)
            || cat
                .relational_materialized_views
                .contains_key(&rename.new_name)
            || cat.relational_sequences.contains_key(&rename.new_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                rename.new_name
            )));
        }
        let Some(mut sequence) = cat.relational_sequences.remove(&rename.old_name) else {
            return Ok(());
        };
        sequence.name = rename.new_name.clone();
        cat.relational_sequences
            .insert(rename.new_name.clone(), sequence);

        let old_target = RelationalCommentTarget::Sequence {
            sequence: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::Sequence {
                    sequence: rename.new_name,
                },
                comment,
            );
        }
        Ok(())
    }

    fn apply_rename_materialized_view(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameMaterializedView,
    ) -> Result<(), EngineError> {
        if cat.relational_catalog.contains_key(&rename.old_name)
            || cat.relational_views.contains_key(&rename.old_name)
            || cat.relational_sequences.contains_key(&rename.old_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a materialized view",
                rename.old_name
            )));
        }
        if cat.relational_catalog.contains_key(&rename.new_name)
            || cat.relational_views.contains_key(&rename.new_name)
            || cat
                .relational_materialized_views
                .contains_key(&rename.new_name)
            || cat.relational_sequences.contains_key(&rename.new_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                rename.new_name
            )));
        }
        let Some(mut view) = cat.relational_materialized_views.remove(&rename.old_name) else {
            return Err(EngineError::ApplyFailed(format!(
                "materialized view \"{}\" does not exist",
                rename.old_name
            )));
        };
        view.name = rename.new_name.clone();
        cat.relational_materialized_views
            .insert(rename.new_name.clone(), view);

        let old_target = RelationalCommentTarget::MaterializedView {
            materialized_view: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::MaterializedView {
                    materialized_view: rename.new_name,
                },
                comment,
            );
        }
        Ok(())
    }

    fn preflight_create_sequence(&self, create: &CreateSequence) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(&create.name)
            || cat.relational_views.contains_key(&create.name)
            || cat.relational_materialized_views.contains_key(&create.name)
            || cat.relational_sequences.contains_key(&create.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                create.name
            )));
        }
        Ok(())
    }

    fn preflight_sequence_target(&self, name: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(name)
            || cat.relational_views.contains_key(name)
            || cat.relational_materialized_views.contains_key(name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{name}\" is not a sequence"
            )));
        }
        if !cat.relational_sequences.contains_key(name) {
            return Err(EngineError::ApplyFailed(format!(
                "sequence \"{name}\" does not exist"
            )));
        }
        Ok(())
    }

    fn preflight_table_acl_target(&self, table: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_views.contains_key(table)
            || cat.relational_materialized_views.contains_key(table)
            || cat.relational_sequences.contains_key(table)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{table}\" is not a table"
            )));
        }
        if !cat.relational_catalog.contains_key(table) {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{table}\" does not exist"
            )));
        }
        Ok(())
    }

    fn preflight_acl_target(
        &self,
        relation: &str,
        kind: AclRelationKind,
    ) -> Result<(), EngineError> {
        let actual = self.acl_relation_kind(relation).ok_or_else(|| {
            EngineError::ApplyFailed(format!("relation \"{relation}\" does not exist"))
        })?;
        let table_keyword_matches_relation = kind == AclRelationKind::Table
            && matches!(
                actual,
                AclRelationKind::Table | AclRelationKind::View | AclRelationKind::MaterializedView
            );
        if kind != AclRelationKind::Relation && kind != actual && !table_keyword_matches_relation {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{relation}\" is not a {}",
                acl_relation_kind_label(kind)
            )));
        }
        Ok(())
    }

    fn preflight_acl_grantee(&self, grantee: &str) -> Result<(), EngineError> {
        if self.role_exists(grantee) || grantee == "public" {
            Ok(())
        } else {
            Err(EngineError::ApplyFailed(format!(
                "role \"{grantee}\" does not exist"
            )))
        }
    }

    fn acl_relation_kind(&self, relation: &str) -> Option<AclRelationKind> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(relation) {
            Some(AclRelationKind::Table)
        } else if cat.relational_views.contains_key(relation) {
            Some(AclRelationKind::View)
        } else if cat.relational_materialized_views.contains_key(relation) {
            Some(AclRelationKind::MaterializedView)
        } else if cat.relational_sequences.contains_key(relation) {
            Some(AclRelationKind::Sequence)
        } else {
            None
        }
    }

    fn relational_acl_mut<'a>(
        cat: &'a mut DdlCatalogState,
        relation: &str,
    ) -> Option<&'a mut BTreeMap<String, BTreeSet<TablePrivilege>>> {
        if let Some(table) = cat.relational_catalog.get_mut(relation) {
            Some(&mut table.acl)
        } else if let Some(view) = cat.relational_views.get_mut(relation) {
            Some(&mut view.acl)
        } else if let Some(view) = cat.relational_materialized_views.get_mut(relation) {
            Some(&mut view.acl)
        } else if let Some(sequence) = cat.relational_sequences.get_mut(relation) {
            Some(&mut sequence.acl)
        } else {
            None
        }
    }

    fn apply_grant_acl(
        &self,
        cat: &mut DdlCatalogState,
        relation: &str,
        kind: AclRelationKind,
        grantee: &str,
        privileges: &[TablePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_acl_target(relation, kind)?;
        self.preflight_acl_grantee(grantee)?;
        let acl = Self::relational_acl_mut(cat, relation)
            .expect("relation ACL target preflighted")
            .entry(grantee.to_string())
            .or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    fn apply_revoke_acl(
        &self,
        cat: &mut DdlCatalogState,
        relation: &str,
        kind: AclRelationKind,
        grantee: &str,
        privileges: &[TablePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_acl_target(relation, kind)?;
        self.preflight_acl_grantee(grantee)?;
        let relation_acl =
            Self::relational_acl_mut(cat, relation).expect("relation ACL target preflighted");
        if let Some(acl) = relation_acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                relation_acl.remove(grantee);
            }
        }
        Ok(())
    }

    fn preflight_function_acl_target(&self, function: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_functions.contains_key(function) {
            Ok(())
        } else {
            Err(EngineError::ApplyFailed(format!(
                "function \"{function}\" does not exist"
            )))
        }
    }

    fn apply_grant_function_acl(
        &self,
        cat: &mut DdlCatalogState,
        function: &str,
        grantee: &str,
        privileges: &[FunctionPrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_function_acl_target(function)?;
        self.preflight_acl_grantee(grantee)?;
        let acl = cat
            .relational_functions
            .get_mut(function)
            .expect("function ACL target preflighted")
            .acl
            .entry(grantee.to_string())
            .or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    fn apply_revoke_function_acl(
        &self,
        cat: &mut DdlCatalogState,
        function: &str,
        grantee: &str,
        privileges: &[FunctionPrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_function_acl_target(function)?;
        self.preflight_acl_grantee(grantee)?;
        let function = cat
            .relational_functions
            .get_mut(function)
            .expect("function ACL target preflighted");
        if let Some(acl) = function.acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                function.acl.remove(grantee);
            }
        }
        Ok(())
    }

    fn preflight_schema_acl_target(&self, schema: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if schema != PUBLIC_SCHEMA_NAME || !cat.relational_public_schema_exists {
            return Err(EngineError::ApplyFailed(
                "schema does not exist".to_string(),
            ));
        }
        Ok(())
    }

    fn apply_grant_schema_acl(
        &self,
        cat: &mut DdlCatalogState,
        schema: &str,
        grantee: &str,
        privileges: &[SchemaPrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_schema_acl_target(schema)?;
        self.preflight_acl_grantee(grantee)?;
        let acl = cat
            .relational_schema_acl
            .entry(grantee.to_string())
            .or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    fn apply_revoke_schema_acl(
        &self,
        cat: &mut DdlCatalogState,
        schema: &str,
        grantee: &str,
        privileges: &[SchemaPrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_schema_acl_target(schema)?;
        self.preflight_acl_grantee(grantee)?;
        if let Some(acl) = cat.relational_schema_acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                cat.relational_schema_acl.remove(grantee);
            }
        }
        Ok(())
    }

    fn apply_grant_default_table_privileges(
        &self,
        cat: &mut DdlCatalogState,
        grantee: &str,
        privileges: &[TablePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_acl_grantee(grantee)?;
        let acl = cat
            .relational_default_table_acl
            .entry(grantee.to_string())
            .or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    fn apply_revoke_default_table_privileges(
        &self,
        cat: &mut DdlCatalogState,
        grantee: &str,
        privileges: &[TablePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_acl_grantee(grantee)?;
        if let Some(acl) = cat.relational_default_table_acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                cat.relational_default_table_acl.remove(grantee);
            }
        }
        Ok(())
    }

    fn preflight_database_acl_target(&self, database: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_databases.contains_key(database) {
            Ok(())
        } else {
            Err(EngineError::ApplyFailed(format!(
                "database \"{database}\" does not exist"
            )))
        }
    }

    fn apply_grant_database_acl(
        &self,
        cat: &mut DdlCatalogState,
        database: &str,
        grantee: &str,
        privileges: &[DatabasePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_database_acl_target(database)?;
        self.preflight_acl_grantee(grantee)?;
        let Some(database) = cat.relational_databases.get_mut(database) else {
            return Ok(());
        };
        let acl = database.acl.entry(grantee.to_string()).or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    fn apply_revoke_database_acl(
        &self,
        cat: &mut DdlCatalogState,
        database: &str,
        grantee: &str,
        privileges: &[DatabasePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_database_acl_target(database)?;
        self.preflight_acl_grantee(grantee)?;
        let Some(database) = cat.relational_databases.get_mut(database) else {
            return Ok(());
        };
        if let Some(acl) = database.acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                database.acl.remove(grantee);
            }
        }
        Ok(())
    }

    fn preflight_tablespace_acl_target(&self, tablespace: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_tablespaces.contains_key(tablespace) {
            Ok(())
        } else {
            Err(EngineError::ApplyFailed(format!(
                "tablespace \"{tablespace}\" does not exist"
            )))
        }
    }

    fn apply_grant_tablespace_acl(
        &self,
        cat: &mut DdlCatalogState,
        tablespace: &str,
        grantee: &str,
        privileges: &[TablespacePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_tablespace_acl_target(tablespace)?;
        self.preflight_acl_grantee(grantee)?;
        let Some(tablespace) = cat.relational_tablespaces.get_mut(tablespace) else {
            return Ok(());
        };
        let acl = tablespace.acl.entry(grantee.to_string()).or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    fn apply_revoke_tablespace_acl(
        &self,
        cat: &mut DdlCatalogState,
        tablespace: &str,
        grantee: &str,
        privileges: &[TablespacePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_tablespace_acl_target(tablespace)?;
        self.preflight_acl_grantee(grantee)?;
        let Some(tablespace) = cat.relational_tablespaces.get_mut(tablespace) else {
            return Ok(());
        };
        if let Some(acl) = tablespace.acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                tablespace.acl.remove(grantee);
            }
        }
        Ok(())
    }

    fn preflight_create_materialized_view(
        &self,
        create: &CreateMaterializedView,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(&create.name)
            || cat.relational_views.contains_key(&create.name)
            || cat.relational_materialized_views.contains_key(&create.name)
            || cat.relational_sequences.contains_key(&create.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                create.name
            )));
        }
        if cat.relational_views.contains_key(&create.query.table)
            || cat
                .relational_materialized_views
                .contains_key(&create.query.table)
        {
            return Err(EngineError::ApplyFailed(
                "materialized views over views are unsupported".to_string(),
            ));
        }
        if !cat.relational_catalog.contains_key(&create.query.table) {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" does not exist",
                create.query.table
            )));
        }
        Ok(())
    }

    fn preflight_refresh_materialized_view(
        &self,
        refresh: &RefreshMaterializedView,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(&refresh.name)
            || cat.relational_views.contains_key(&refresh.name)
            || cat.relational_sequences.contains_key(&refresh.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a materialized view",
                refresh.name
            )));
        }
        if !cat
            .relational_materialized_views
            .contains_key(&refresh.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "materialized view \"{}\" does not exist",
                refresh.name
            )));
        }
        Ok(())
    }

    fn preflight_drop_view(&self, drop: &DropView) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        let drop_names = drop.names.iter().cloned().collect::<BTreeSet<_>>();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "view \"{}\" specified more than once",
                    name
                )));
            }
            if cat.relational_catalog.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a view",
                    name
                )));
            }
            if cat.relational_sequences.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a view",
                    name
                )));
            }
            if cat.relational_materialized_views.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a view",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_views.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "view \"{}\" does not exist",
                    name
                )));
            }
            if cat.relational_views.iter().any(|(candidate, _)| {
                !drop_names.contains(candidate) && self.relational_view_depends_on(candidate, name)
            }) {
                return Err(EngineError::ApplyFailed(format!(
                    "cannot drop view \"{}\" because another view depends on it",
                    name
                )));
            }
        }
        Ok(())
    }

    fn preflight_drop_materialized_view(
        &self,
        drop: &DropMaterializedView,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "materialized view \"{}\" specified more than once",
                    name
                )));
            }
            if cat.relational_catalog.contains_key(name)
                || cat.relational_views.contains_key(name)
                || cat.relational_sequences.contains_key(name)
            {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a materialized view",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_materialized_views.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "materialized view \"{}\" does not exist",
                    name
                )));
            }
        }
        Ok(())
    }

    fn preflight_drop_sequence(&self, drop: &DropSequence) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "sequence \"{}\" specified more than once",
                    name
                )));
            }
            if cat.relational_catalog.contains_key(name)
                || cat.relational_views.contains_key(name)
                || cat.relational_materialized_views.contains_key(name)
            {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a sequence",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_sequences.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "sequence \"{}\" does not exist",
                    name
                )));
            }
        }
        Ok(())
    }

    fn apply_rename_view(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameView,
    ) -> Result<(), EngineError> {
        if cat.relational_catalog.contains_key(&rename.old_name)
            || cat
                .relational_materialized_views
                .contains_key(&rename.old_name)
            || cat.relational_sequences.contains_key(&rename.old_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a view",
                rename.old_name
            )));
        }
        if cat.relational_catalog.contains_key(&rename.new_name)
            || cat.relational_views.contains_key(&rename.new_name)
            || cat
                .relational_materialized_views
                .contains_key(&rename.new_name)
            || cat.relational_sequences.contains_key(&rename.new_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                rename.new_name
            )));
        }
        if self.relational_view_has_dependents(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "cannot rename view \"{}\" because another view depends on it",
                rename.old_name
            )));
        }
        let Some(mut view) = cat.relational_views.remove(&rename.old_name) else {
            return Err(EngineError::ApplyFailed(format!(
                "view \"{}\" does not exist",
                rename.old_name
            )));
        };
        view.name = rename.new_name.clone();
        cat.relational_views.insert(rename.new_name.clone(), view);

        let old_target = RelationalCommentTarget::View {
            view: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::View {
                    view: rename.new_name,
                },
                comment,
            );
        }
        Ok(())
    }

    fn apply_comment_on(
        &self,
        cat: &mut DdlCatalogState,
        comment: gpu_db_sql::CommentOn,
    ) -> Result<(), EngineError> {
        let target = match comment.target {
            CommentTarget::Database { database } => {
                if !self.database_exists(&database) {
                    return Err(EngineError::ApplyFailed(format!(
                        "database \"{}\" does not exist",
                        database
                    )));
                }
                RelationalCommentTarget::Database { database }
            }
            CommentTarget::Role { role } => {
                if !self.role_exists(&role) {
                    return Err(EngineError::ApplyFailed(format!(
                        "role \"{}\" does not exist",
                        role
                    )));
                }
                RelationalCommentTarget::Role { role }
            }
            CommentTarget::Schema { schema } => {
                if schema != "public" {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" does not exist",
                        schema
                    )));
                }
                RelationalCommentTarget::Schema { schema }
            }
            CommentTarget::Tablespace { tablespace } => {
                if !self.tablespace_exists(&tablespace) {
                    return Err(EngineError::ApplyFailed(format!(
                        "tablespace \"{}\" does not exist",
                        tablespace
                    )));
                }
                RelationalCommentTarget::Tablespace { tablespace }
            }
            CommentTarget::Table { table } => {
                if !cat.relational_catalog.contains_key(&table) {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        table
                    )));
                }
                RelationalCommentTarget::Table { table }
            }
            CommentTarget::Column { table, column } => {
                let table_ref = cat.relational_catalog.get(&table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", table))
                })?;
                let column_ref = table_ref
                    .columns
                    .iter()
                    .find(|candidate| candidate.name == column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!("column \"{}\" does not exist", column))
                    })?;
                RelationalCommentTarget::Column {
                    table,
                    attnum: column_ref.attnum,
                }
            }
            CommentTarget::Index { index } => {
                if !cat.relational_catalog.values().any(|table| {
                    table
                        .indexes
                        .iter()
                        .any(|candidate| candidate.name == index)
                }) {
                    return Err(EngineError::ApplyFailed(format!(
                        "index \"{}\" does not exist",
                        index
                    )));
                }
                RelationalCommentTarget::Index { index }
            }
            CommentTarget::View { view } => {
                if !cat.relational_views.contains_key(&view) {
                    if cat.relational_catalog.contains_key(&view)
                        || cat.relational_materialized_views.contains_key(&view)
                        || cat.relational_sequences.contains_key(&view)
                    {
                        return Err(EngineError::ApplyFailed(format!(
                            "relation \"{}\" is not a view",
                            view
                        )));
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "view \"{}\" does not exist",
                        view
                    )));
                }
                RelationalCommentTarget::View { view }
            }
            CommentTarget::MaterializedView { materialized_view } => {
                if !cat
                    .relational_materialized_views
                    .contains_key(&materialized_view)
                {
                    if cat.relational_catalog.contains_key(&materialized_view)
                        || cat.relational_views.contains_key(&materialized_view)
                        || cat.relational_sequences.contains_key(&materialized_view)
                    {
                        return Err(EngineError::ApplyFailed(format!(
                            "relation \"{}\" is not a materialized view",
                            materialized_view
                        )));
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "materialized view \"{}\" does not exist",
                        materialized_view
                    )));
                }
                RelationalCommentTarget::MaterializedView { materialized_view }
            }
            CommentTarget::Function { function } => {
                if !cat.relational_functions.contains_key(&function) {
                    return Err(EngineError::ApplyFailed(format!(
                        "function \"{}\" does not exist",
                        function
                    )));
                }
                RelationalCommentTarget::Function { function }
            }
            CommentTarget::Extension { extension } => {
                if extension != "plpgsql" {
                    return Err(EngineError::ApplyFailed(format!(
                        "extension \"{}\" does not exist",
                        extension
                    )));
                }
                RelationalCommentTarget::Extension { extension }
            }
            CommentTarget::Sequence { sequence } => {
                if !cat.relational_sequences.contains_key(&sequence) {
                    if cat.relational_catalog.contains_key(&sequence)
                        || cat.relational_views.contains_key(&sequence)
                        || cat.relational_materialized_views.contains_key(&sequence)
                    {
                        return Err(EngineError::ApplyFailed(format!(
                            "relation \"{}\" is not a sequence",
                            sequence
                        )));
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "sequence \"{}\" does not exist",
                        sequence
                    )));
                }
                RelationalCommentTarget::Sequence { sequence }
            }
            CommentTarget::Domain { domain } => {
                if !cat.relational_domains.contains_key(&domain) {
                    return Err(EngineError::ApplyFailed(format!(
                        "domain \"{}\" does not exist",
                        domain
                    )));
                }
                RelationalCommentTarget::Domain { domain }
            }
            CommentTarget::Publication { publication } => {
                if !cat.relational_publications.contains_key(&publication) {
                    return Err(EngineError::ApplyFailed(format!(
                        "publication \"{}\" does not exist",
                        publication
                    )));
                }
                RelationalCommentTarget::Publication { publication }
            }
            CommentTarget::Subscription { subscription } => {
                if !cat.relational_subscriptions.contains_key(&subscription) {
                    return Err(EngineError::ApplyFailed(format!(
                        "subscription \"{}\" does not exist",
                        subscription
                    )));
                }
                RelationalCommentTarget::Subscription { subscription }
            }
            CommentTarget::Constraint { table, constraint } => {
                let table_ref = cat.relational_catalog.get(&table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", table))
                })?;
                if !table_ref.indexes.iter().any(|candidate| {
                    candidate.name == constraint
                        && (candidate.primary_key || candidate.unique_constraint)
                }) && !table_ref
                    .check_constraints
                    .iter()
                    .any(|candidate| candidate.name == constraint)
                    && !table_ref
                        .foreign_keys
                        .iter()
                        .any(|candidate| candidate.name == constraint)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "constraint \"{}\" does not exist",
                        constraint
                    )));
                }
                RelationalCommentTarget::Constraint { table, constraint }
            }
        };
        if let Some(value) = comment.comment {
            cat.relational_comments.insert(target, value);
        } else {
            cat.relational_comments.remove(&target);
        }
        Ok(())
    }

    fn apply_alter_column_default(
        &self,
        cat: &mut DdlCatalogState,
        alter: gpu_db_sql::AlterColumnDefault,
    ) -> Result<(), EngineError> {
        // Coerce the new default to the column type (parity with INSERT/CREATE) before the
        // mutable borrow, so `ALTER ... SET DEFAULT 0` on a numeric column is accepted and
        // stored at the column scale rather than rejected as a type mismatch.
        let coerced_default = if let Some(default) = alter.default {
            let table = cat.relational_catalog.get(&alter.table).ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", alter.table))
            })?;
            let column = table
                .columns
                .iter()
                .find(|column| column.name == alter.column)
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!("column \"{}\" does not exist", alter.column))
                })?;
            let coerced = coerce_column_default(default, column.ty, &alter.column)?;
            self.preflight_column_default_target(&coerced)?;
            Some(coerced)
        } else {
            None
        };
        let table = cat
            .relational_catalog
            .get_mut(&alter.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", alter.table))
            })?;
        let column = table
            .columns
            .iter_mut()
            .find(|column| column.name == alter.column)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("column \"{}\" does not exist", alter.column))
            })?;
        column.default = coerced_default;
        Ok(())
    }

    fn apply_add_column(
        &self,
        cat: &mut DdlCatalogState,
        add: gpu_db_sql::AddColumn,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        let mut column_def = add.column;
        let (type_oid, type_size) = self.resolve_column_domain_type(&mut column_def)?;
        if cat.relational_views.contains_key(&add.table)
            || cat.relational_materialized_views.contains_key(&add.table)
            || cat.relational_sequences.contains_key(&add.table)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a table",
                add.table
            )));
        }
        let Some(default) = column_def.default.clone() else {
            return Err(EngineError::ApplyFailed(
                "ADD COLUMN requires a supported DEFAULT in the bootstrap relational subset"
                    .to_string(),
            ));
        };
        if !add_column_default_supported(&default) {
            return Err(EngineError::ApplyFailed(
                "ADD COLUMN SERIAL is unsupported in the bootstrap relational subset".to_string(),
            ));
        }
        // Coerce a cross-type default literal to the column type (parity with INSERT/CREATE),
        // storing it back so both the existing-row backfill and `from_def` use the coerced
        // value; also re-validates a nextval default against the int4 restriction.
        let default = coerce_column_default(default, column_def.ty, &column_def.name)?;
        column_def.default = Some(default.clone());
        let table = cat
            .relational_catalog
            .get(&add.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
            })?
            .clone();
        if table
            .columns
            .iter()
            .any(|column| column.name == column_def.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "column \"{}\" of relation \"{}\" already exists",
                column_def.name, add.table
            )));
        }
        self.preflight_column_default_target(&default)?;
        let row_count = self
            .visible_relational_rows(
                &table,
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?
            .len();
        let default_values = (0..row_count)
            .map(|_| self.evaluate_column_default(cat, &default))
            .collect::<Result<Vec<_>, _>>()?;
        let next_attnum = i16::try_from(table.columns.len() + 1).map_err(|_| {
            EngineError::ApplyFailed("too many columns for bootstrap catalog".to_string())
        })?;
        let column_id = cat.relational_next_column_id;
        let next_column_id = cat
            .relational_next_column_id
            .checked_add(1)
            .ok_or_else(|| {
                EngineError::ApplyFailed("relational column id allocation exhausted".to_string())
            })?;
        let new_column = RelationalColumn::from_def(
            column_id,
            table.oid,
            next_attnum,
            column_def,
            type_oid,
            type_size,
        );

        let prefix = relational_key_prefix(&add.table);
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let mut updates = Vec::new();
        let mut default_values = default_values.into_iter();
        {
            let table_rows = self.read_state.mvcc.table_rows(&add.table);
            let mut cursor = table_rows
                .store()
                .seq_scan_open(visibility)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let mut row = decode_relational_row(&tuple.value, &table.columns)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let default_value = default_values.next().ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "ADD COLUMN default rewrite row count drifted".to_string(),
                    )
                })?;
                row.push(default_value);
                updates.push((tuple.tuple_id, tuple.key.clone(), row));
            }
        }
        if default_values.next().is_some() {
            return Err(EngineError::ApplyFailed(
                "ADD COLUMN default rewrite row count drifted".to_string(),
            ));
        }

        let new_column_name = new_column.name.clone();
        self.read_state.mvcc.with_table_mut(&add.table, |data| {
            for (tuple_id, row_key, values) in &updates {
                let index_value = values.last().expect("new column default appended").clone();
                data.rows
                    .tuple_update(*tuple_id, encode_relational_row(values), txn_id)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let index_key = ColumnValueKey {
                    column: new_column_name.clone(),
                    value: relational_index_value(&index_value),
                };
                let mut slot = data
                    .value_index
                    .get(&index_key)
                    .cloned()
                    .unwrap_or_default();
                std::sync::Arc::make_mut(&mut slot).push(row_key.clone());
                data.value_index.insert(index_key, slot);
            }
            Ok::<(), EngineError>(())
        })?;
        let table_ref = cat
            .relational_catalog
            .get_mut(&add.table)
            .expect("table existence validated");
        table_ref.columns.push(new_column.clone());
        cat.relational_next_column_id = next_column_id;
        self.read_state
            .residency
            .with_snapshots_mut(|snapshots| snapshots.remove(&add.table));
        self.read_state.residency.device_memory.remove(&add.table);
        Ok(())
    }

    fn apply_rename_column(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameColumn,
    ) -> Result<(), EngineError> {
        if cat.relational_views.contains_key(&rename.table)
            || cat
                .relational_materialized_views
                .contains_key(&rename.table)
            || cat.relational_sequences.contains_key(&rename.table)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a table",
                rename.table
            )));
        }
        let table = cat.relational_catalog.get(&rename.table).ok_or_else(|| {
            EngineError::ApplyFailed(format!("relation \"{}\" does not exist", rename.table))
        })?;
        if !table
            .columns
            .iter()
            .any(|column| column.name == rename.old_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "column \"{}\" does not exist",
                rename.old_name
            )));
        }
        if table
            .columns
            .iter()
            .any(|column| column.name == rename.new_name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "column \"{}\" of relation \"{}\" already exists",
                rename.new_name, rename.table
            )));
        }

        // Rename the column within this table's per-table value-index (keys are `(column, value)`),
        // publishing one new generation. `imbl::OrdMap` has no in-place `retain`; rebuild the index,
        // re-keying the matching column's slots (carrying the same `Arc` row-key list, an O(1) move)
        // and keeping the rest by `Arc`-clone. Order is preserved (OrdMap is ordered).
        self.read_state.mvcc.with_table_mut(&rename.table, |data| {
            let mut rebuilt = imbl::OrdMap::new();
            for (key, row_keys) in data.value_index.iter() {
                let new_key = if key.column == rename.old_name {
                    ColumnValueKey {
                        column: rename.new_name.clone(),
                        value: key.value.clone(),
                    }
                } else {
                    key.clone()
                };
                rebuilt.insert(new_key, std::sync::Arc::clone(row_keys));
            }
            data.value_index = rebuilt;
        });

        let table_ref = cat
            .relational_catalog
            .get_mut(&rename.table)
            .expect("table existence validated");
        let column = table_ref
            .columns
            .iter_mut()
            .find(|column| column.name == rename.old_name)
            .expect("column existence validated");
        column.name = rename.new_name.clone();
        for index in &mut table_ref.indexes {
            if index.column == rename.old_name {
                index.column = rename.new_name.clone();
            }
        }
        for constraint in &mut table_ref.check_constraints {
            if constraint.column == rename.old_name {
                constraint.column = rename.new_name.clone();
            }
        }
        for constraint in &mut table_ref.foreign_keys {
            if constraint.column == rename.old_name {
                constraint.column = rename.new_name.clone();
            }
            if constraint.referenced_table == rename.table
                && constraint.referenced_column == rename.old_name
            {
                constraint.referenced_column = rename.new_name.clone();
            }
        }
        for candidate in cat.relational_catalog.values_mut() {
            if candidate.name == rename.table {
                continue;
            }
            for constraint in &mut candidate.foreign_keys {
                if constraint.referenced_table == rename.table
                    && constraint.referenced_column == rename.old_name
                {
                    constraint.referenced_column = rename.new_name.clone();
                }
            }
        }
        self.read_state
            .residency
            .with_snapshots_mut(|snapshots| snapshots.remove(&rename.table));
        self.read_state
            .residency
            .device_memory
            .remove(&rename.table);
        Ok(())
    }

    fn apply_drop_column(
        &self,
        cat: &mut DdlCatalogState,
        drop_column: gpu_db_sql::DropColumn,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        if cat.relational_views.contains_key(&drop_column.table)
            || cat
                .relational_materialized_views
                .contains_key(&drop_column.table)
            || cat.relational_sequences.contains_key(&drop_column.table)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a table",
                drop_column.table
            )));
        }
        let table = cat
            .relational_catalog
            .get(&drop_column.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "relation \"{}\" does not exist",
                    drop_column.table
                ))
            })?
            .clone();
        let drop_idx = table
            .columns
            .iter()
            .position(|column| column.name == drop_column.column)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "column \"{}\" does not exist",
                    drop_column.column
                ))
            })?;
        if table
            .indexes
            .iter()
            .any(|index| index.column == drop_column.column)
            || table
                .check_constraints
                .iter()
                .any(|constraint| constraint.column == drop_column.column)
            || table
                .foreign_keys
                .iter()
                .any(|constraint| constraint.column == drop_column.column)
            || cat.relational_catalog.values().any(|candidate| {
                candidate.foreign_keys.iter().any(|constraint| {
                    constraint.referenced_table == drop_column.table
                        && constraint.referenced_column == drop_column.column
                })
            })
        {
            return Err(EngineError::ApplyFailed(format!(
                "cannot drop column \"{}\" because an index or constraint depends on it",
                drop_column.column
            )));
        }

        let prefix = relational_key_prefix(&drop_column.table);
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let mut updates = Vec::new();
        {
            let table_rows = self.read_state.mvcc.table_rows(&drop_column.table);
            let mut cursor = table_rows
                .store()
                .seq_scan_open(visibility)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            while let Some(tuple) = cursor.next() {
                if !tuple.key.starts_with(&prefix) {
                    continue;
                }
                let mut row = decode_relational_row(&tuple.value, &table.columns)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                row.remove(drop_idx);
                updates.push((tuple.tuple_id, row));
            }
        }

        let dropped_column_name = drop_column.column.clone();
        self.read_state
            .mvcc
            .with_table_mut(&drop_column.table, |data| {
                for (tuple_id, values) in &updates {
                    data.rows
                        .tuple_update(*tuple_id, encode_relational_row(values), txn_id)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                }
                // `imbl::OrdMap` has no in-place `retain`; rebuild keeping every slot whose column is
                // not the dropped one (`Arc`-clone is O(1), and OrdMap preserves key order).
                data.value_index = data
                    .value_index
                    .iter()
                    .filter(|(key, _)| key.column != dropped_column_name)
                    .map(|(key, row_keys)| (key.clone(), std::sync::Arc::clone(row_keys)))
                    .collect();
                Ok::<(), EngineError>(())
            })?;

        let dropped_attnum = table.columns[drop_idx].attnum;
        let table_ref = cat
            .relational_catalog
            .get_mut(&drop_column.table)
            .expect("table existence validated");
        table_ref.columns.remove(drop_idx);
        for (idx, column) in table_ref.columns.iter_mut().enumerate() {
            column.attnum = i16::try_from(idx + 1).map_err(|_| {
                EngineError::ApplyFailed("too many columns for bootstrap catalog".to_string())
            })?;
        }

        let shifted_comments = cat
            .relational_comments
            .iter()
            .filter_map(|(target, comment)| match target {
                RelationalCommentTarget::Column { table, attnum }
                    if table == &drop_column.table && *attnum > dropped_attnum =>
                {
                    Some((
                        target.clone(),
                        RelationalCommentTarget::Column {
                            table: table.clone(),
                            attnum: *attnum - 1,
                        },
                        comment.clone(),
                    ))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        cat.relational_comments.retain(|target, _| match target {
            RelationalCommentTarget::Column { table, attnum } => {
                !(table == &drop_column.table && *attnum >= dropped_attnum)
            }
            _ => true,
        });
        for (_, new_target, comment) in shifted_comments {
            cat.relational_comments.insert(new_target, comment);
        }
        self.read_state
            .residency
            .with_snapshots_mut(|snapshots| snapshots.remove(&drop_column.table));
        self.read_state
            .residency
            .device_memory
            .remove(&drop_column.table);
        Ok(())
    }

    fn apply_insert(
        &self,
        cat: &mut DdlCatalogState,
        insert: Insert,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        self.apply_insert_with_profile(cat, insert, txn_id, None)
    }

    /// The off-lock read boundary a `prepare_*` runs against (write-half MVCC, Stage 2).
    ///
    /// Under serialization today `commit_seq` is the entry's commit `Index` and `next_row_id`
    /// is `relational_next_row_id` captured immediately before apply — so `prepare_*` reads
    /// exactly what the old direct apply read, and computes the identical row keys. When the
    /// commit lock is removed (Stage 4) this becomes a true snapshot taken at statement start.
    fn dml_read_snapshot(&self, commit_seq: TxnId) -> DmlReadSnapshot {
        DmlReadSnapshot {
            commit_seq,
            next_row_id: self.read_state.mvcc.current_row_id(),
        }
    }

    /// PURE preflight + encode for `INSERT` (write-half MVCC, Stage 2). Reads only from the
    /// `snapshot` (no `&mut self`, no engine mutation); runs the unique / check / FK preflight
    /// exactly as the old `apply_insert_with_profile`; encodes the new row versions and computes
    /// the write-set. The returned [`WriteDelta`] is what [`Engine::apply_delta`] installs.
    ///
    /// One deliberate refinement vs. the old in-line apply: the old code evaluated `nextval`
    /// column defaults (mutating the sequence) BEFORE the preflight, so a preflight FAILURE still
    /// advanced the sequence. Here the advance is deferred to `apply_delta`, so a prepare that
    /// fails preflight advances nothing — the Stage-4 abort-is-side-effect-free semantics. This is
    /// not observable on the live paths: `execute_text` / the COPY path run the same preflight
    /// BEFORE committing, so a constraint-violating INSERT never reaches apply in the first place.
    fn prepare_insert(
        &self,
        insert: &Insert,
        snapshot: DmlReadSnapshot,
        mut profile: Option<&mut RelationalCopyAdmissionProfile>,
    ) -> Result<WriteDelta, EngineError> {
        let txn_id = snapshot.commit_seq;
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): bind the target against a pinned
        // catalog snapshot, cloning it out. FK validation pins its own snapshot internally.
        let table = self
            .catalog_snapshot()
            .relational_catalog
            .get(&insert.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", insert.table))
            })?
            .clone();
        let column_indexes = if insert.columns.is_empty() {
            (0..table.columns.len()).collect::<Vec<_>>()
        } else {
            let mut indexes = Vec::with_capacity(insert.columns.len());
            for column in &insert.columns {
                let idx = table
                    .columns
                    .iter()
                    .position(|candidate| candidate.name == *column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!("column \"{}\" does not exist", column))
                    })?;
                indexes.push(idx);
            }
            indexes
        };
        let row_prepare_started = Instant::now();
        let mut new_rows = Vec::with_capacity(insert.rows.len());
        // Sequence advancement scratch: keeps `prepare_insert` pure (no `&mut self`) while
        // evaluating `nextval` column defaults. Seeded lazily from the engine's sequence catalog,
        // advanced per row in source order (matching the old in-line apply), then installed by
        // `apply_delta`.
        let mut seq_state: BTreeMap<String, (i64, bool)> = BTreeMap::new();
        for row in &insert.rows {
            if row.len() != column_indexes.len() {
                return Err(EngineError::ApplyFailed(
                    "INSERT value count must match target columns".to_string(),
                ));
            }
            let mut values = vec![None; table.columns.len()];
            for (source_idx, target_idx) in column_indexes.iter().copied().enumerate() {
                let value = row[source_idx].clone();
                let expected_ty = table.columns[target_idx].ty;
                let coerced =
                    coerce_insert_value(value, expected_ty, &table.columns[target_idx].name)?;
                values[target_idx] = Some(coerced);
            }
            for (idx, value) in values.iter_mut().enumerate() {
                if value.is_none() {
                    if let Some(default) = table.columns[idx].default.clone() {
                        *value = Some(self.evaluate_column_default_pure(&default, &mut seq_state)?);
                    }
                }
            }
            if values.iter().any(Option::is_none) {
                return Err(EngineError::ApplyFailed(
                    "INSERT must provide every column without a default in the bootstrap relational subset"
                        .to_string(),
                ));
            }
            let values = values.into_iter().map(Option::unwrap).collect::<Vec<_>>();
            new_rows.push(values);
        }
        if let Some(profile) = profile.as_mut() {
            profile.row_prepare_micros += row_prepare_started.elapsed().as_micros();
        }

        if table.indexes.iter().any(|index| index.unique) {
            let unique_preflight_started = Instant::now();
            let mut candidate_rows = self.visible_relational_rows(
                &table,
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?;
            candidate_rows.extend(new_rows.clone());
            Self::validate_unique_indexes_for_rows(&table, &candidate_rows)?;
            if let Some(profile) = profile.as_mut() {
                profile.unique_preflight_micros += unique_preflight_started.elapsed().as_micros();
            }
        }
        if !table.check_constraints.is_empty() {
            let check_preflight_started = Instant::now();
            let mut candidate_rows = self.visible_relational_rows(
                &table,
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?;
            candidate_rows.extend(new_rows.clone());
            Self::validate_check_constraints_for_rows(&table, &candidate_rows)?;
            if let Some(profile) = profile.as_mut() {
                profile.check_preflight_micros += check_preflight_started.elapsed().as_micros();
            }
        }
        if !table.foreign_keys.is_empty() {
            let foreign_key_preflight_started = Instant::now();
            let mut candidate_rows = self.visible_relational_rows(
                &table,
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?;
            candidate_rows.extend(new_rows.clone());
            self.validate_foreign_keys_with_table_rows(
                &table.name,
                &candidate_rows,
                StorageVisibility {
                    read_txn_id: txn_id,
                },
            )?;
            if let Some(profile) = profile.as_mut() {
                profile.foreign_key_preflight_micros +=
                    foreign_key_preflight_started.elapsed().as_micros();
            }
        }

        // Encode the new versions against the snapshot's `next_row_id` base (pure: does not bump
        // `relational_next_row_id` — `apply_delta` advances it by `rows_consumed`). These row keys
        // are exactly what the old `apply_insert` assigned because, under the still-serialized
        // commit, the snapshot is taken immediately before apply.
        let rows_consumed = new_rows.len() as u64;
        let mut inserted_rows = Vec::with_capacity(new_rows.len());
        for (offset, values) in new_rows.into_iter().enumerate() {
            let row_id = snapshot.next_row_id + offset as u64;
            let row_key = relational_row_key(&insert.table, row_id);
            inserted_rows.push((row_key, values));
        }
        let value_index_entries =
            relational_value_index_entries_for_rows(&table.columns, &inserted_rows);

        let mut write_set = WriteSet::default();
        for (_row_key, values) in &inserted_rows {
            // An INSERT claims a FRESH, unique row id at install time (`apply_delta` reserves the
            // tuple id + advances `relational_next_row_id` under the commit lock), so its row slot
            // can never truly collide with another writer's — the predicted `row_key` here is only
            // a snapshot-relative label and is re-derived live on install. Putting it in the
            // conflict `write_set.rows` would make two concurrent disjoint inserts whose prepare
            // windows overlap (and therefore read the SAME off-lock `next_row_id`) predict the SAME
            // key and FALSELY conflict. Inserts conflict ONLY on the unique-index slots they
            // occupy (the genuine first-committer-wins point); the row slot is intentionally NOT a
            // conflict dimension for inserts.
            write_set.add_unique_slots(&table, values);
        }

        Ok(WriteDelta {
            write_set,
            rows_consumed,
            mutation: PreparedMutation::Insert {
                table: insert.table.clone(),
                inserted_rows,
                value_index_entries,
                seq_advances: seq_state,
            },
        })
    }

    /// PURE preflight + scan for `DELETE` (write-half MVCC, Stage 2). Resolves which existing
    /// versions match (against `snapshot`), runs the inbound-FK preflight as the old
    /// `apply_delete`, and records the tombstone write-set. No engine mutation.
    fn prepare_delete(
        &self,
        delete: &Delete,
        snapshot: DmlReadSnapshot,
    ) -> Result<WriteDelta, EngineError> {
        let txn_id = snapshot.commit_seq;
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): pin ONE catalog snapshot for both the
        // target bind and the inbound-FK-dependents scan below.
        let catalog = self.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(&delete.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", delete.table))
            })?
            .clone();
        let filter_groups = bind_delete_filter_groups(&table, delete)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let prefix = relational_key_prefix(&delete.table);
        // Resolve (tuple_id, key, row) for each matching version against this table's published
        // generation: tuple_id is what apply tombstones; key/row feed the write-set entries.
        let table_rows = self.read_state.mvcc.table_rows(&delete.table);
        let mut deletes: Vec<(u64, String, Vec<SqlValue>)> = Vec::new();
        let mut cursor = table_rows
            .store()
            .seq_scan_open(visibility)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;

        while let Some(tuple) = cursor.next() {
            if !tuple.key.starts_with(&prefix) {
                continue;
            }
            let row = decode_relational_row(&tuple.value, &table.columns)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            if filter_groups.iter().any(|filters| {
                filters
                    .iter()
                    .all(|(idx, op, value)| select_filter_matches(&row[*idx], *op, value))
            }) {
                deletes.push((tuple.tuple_id, tuple.key.clone(), row));
            }
        }
        drop(cursor);

        if catalog.relational_catalog.values().any(|candidate| {
            candidate
                .foreign_keys
                .iter()
                .any(|foreign_key| foreign_key.referenced_table == table.name)
        }) {
            let deleted_ids = deletes
                .iter()
                .map(|(tuple_id, _, _)| *tuple_id)
                .collect::<BTreeSet<_>>();
            let mut candidate_rows = Vec::new();
            let mut cursor = table_rows
                .store()
                .seq_scan_open(visibility)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            while let Some(tuple) = cursor.next() {
                if tuple.key.starts_with(&prefix) && !deleted_ids.contains(&tuple.tuple_id) {
                    candidate_rows.push(
                        decode_relational_row(&tuple.value, &table.columns)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?,
                    );
                }
            }
            drop(cursor);
            self.validate_foreign_keys_with_table_rows(&table.name, &candidate_rows, visibility)?;
        }

        let mut write_set = WriteSet::default();
        let mut tuple_ids = Vec::with_capacity(deletes.len());
        for (tuple_id, key, row) in &deletes {
            tuple_ids.push(*tuple_id);
            write_set.rows.push(RowWriteKey {
                table: delete.table.clone(),
                row_key: key.clone(),
            });
            // A delete releases the row's unique-index slots; record them as written so a
            // concurrent insert reusing the value conflicts (Stage 4 first-committer-wins).
            write_set.add_unique_slots(&table, row);
        }

        Ok(WriteDelta {
            write_set,
            rows_consumed: 0,
            mutation: PreparedMutation::Delete {
                table: delete.table.clone(),
                tuple_ids,
            },
        })
    }

    /// PURE preflight + scan + encode for `UPDATE` (write-half MVCC, Stage 2). Resolves the
    /// matching versions, applies the assignments to encode the new row images, runs the unique /
    /// check / FK preflight as the old `apply_update`, and records the write-set (old slot
    /// tombstoned + new version + unique slots). No engine mutation.
    fn prepare_update(
        &self,
        update: &Update,
        snapshot: DmlReadSnapshot,
    ) -> Result<WriteDelta, EngineError> {
        let txn_id = snapshot.commit_seq;
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): pin ONE catalog snapshot for the
        // target bind and the inbound-FK-dependents scan below.
        let catalog = self.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(&update.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", update.table))
            })?
            .clone();
        let assignments = bind_update_assignments(&table, update)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let filter_groups = bind_delete_filter_groups(
            &table,
            &Delete {
                table: update.table.clone(),
                filter: update.filter.clone(),
                filters: update.filters.clone(),
                filter_groups: update.filter_groups.clone(),
            },
        )
        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let prefix = relational_key_prefix(&update.table);
        let mut updates = Vec::new();
        let mut candidate_rows = Vec::new();
        // Unique slots the OLD images RELEASE (prereq #2, Stage-4 audit). An UPDATE that changes a
        // unique column frees its old `(table, column, value)` slot; record those freed slots in the
        // write-set so a CONCURRENT insert/update reusing the freed value conflicts under
        // first-committer-wins — matching the DELETE path, which already records the released slots.
        // This is the conservative choice: it never admits a phantom unique duplicate across a
        // concurrent free+reuse (a slot-release left unrecorded could). A no-op-on-the-unique-column
        // UPDATE records the same slot as both released (old) and claimed (new) — harmless (the
        // write-set dedups to one slot), so an idempotent rewrite does not self-conflict.
        let mut released_unique_slots: Vec<UniqueIndexSlotKey> = Vec::new();
        let table_rows = self.read_state.mvcc.table_rows(&update.table);
        let mut cursor = table_rows
            .store()
            .seq_scan_open(visibility)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;

        while let Some(tuple) = cursor.next() {
            if !tuple.key.starts_with(&prefix) {
                continue;
            }
            let mut row = decode_relational_row(&tuple.value, &table.columns)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            if filter_groups.iter().any(|filters| {
                filters
                    .iter()
                    .all(|(idx, op, value)| select_filter_matches(&row[*idx], *op, value))
            }) {
                // Capture the old image's unique slots BEFORE the assignments overwrite them.
                let mut old_slots = WriteSet::default();
                old_slots.add_unique_slots(&table, &row);
                released_unique_slots.append(&mut old_slots.unique_slots);
                for (idx, value) in &assignments {
                    row[*idx] = value.clone();
                }
                updates.push((tuple.tuple_id, tuple.key.clone(), row));
            } else {
                candidate_rows.push(row);
            }
        }
        drop(cursor);

        if table.indexes.iter().any(|index| index.unique)
            || !table.check_constraints.is_empty()
            || !table.foreign_keys.is_empty()
            || catalog.relational_catalog.values().any(|candidate| {
                candidate
                    .foreign_keys
                    .iter()
                    .any(|foreign_key| foreign_key.referenced_table == table.name)
            })
        {
            candidate_rows.extend(updates.iter().map(|(_, _, row)| row.clone()));
        }
        if table.indexes.iter().any(|index| index.unique) {
            Self::validate_unique_indexes_for_rows(&table, &candidate_rows)?;
        }
        if !table.check_constraints.is_empty() {
            Self::validate_check_constraints_for_rows(&table, &candidate_rows)?;
        }
        if !table.foreign_keys.is_empty()
            || catalog.relational_catalog.values().any(|candidate| {
                candidate
                    .foreign_keys
                    .iter()
                    .any(|foreign_key| foreign_key.referenced_table == table.name)
            })
        {
            self.validate_foreign_keys_with_table_rows(&table.name, &candidate_rows, visibility)?;
        }

        let updated_rows: Vec<(String, Vec<SqlValue>)> = updates
            .iter()
            .map(|(_, key, row)| (key.clone(), row.clone()))
            .collect();
        let value_index_entries =
            relational_value_index_entries_for_rows(&table.columns, &updated_rows);

        let mut write_set = WriteSet::default();
        for (_, key, row) in &updates {
            // An UPDATE tombstones the old version and installs a new one at the SAME row key,
            // so the row slot is written once.
            write_set.rows.push(RowWriteKey {
                table: update.table.clone(),
                row_key: key.clone(),
            });
            // The new image's unique-index slots are claimed by this txn.
            write_set.add_unique_slots(&table, row);
        }
        // The old images' RELEASED unique slots are also conflict points (prereq #2). Dedup so a
        // value carried unchanged through the UPDATE (same slot released and re-claimed) is recorded
        // once and never self-conflicts.
        write_set.unique_slots.append(&mut released_unique_slots);
        write_set.unique_slots.sort();
        write_set.unique_slots.dedup();

        // `updates` is already `(tuple_id, row_key, new_values)` — exactly the install shape.
        Ok(WriteDelta {
            write_set,
            rows_consumed: 0,
            mutation: PreparedMutation::Update {
                table: update.table.clone(),
                installs: updates,
                value_index_entries,
            },
        })
    }

    /// Install a prepared [`WriteDelta`]'s data, stamping new versions with `commit_seq` (Stage 0
    /// stamp/boundary unification: `commit_seq == commit Index`). `&self` (write-half Stage 4): it
    /// mutates exactly the structures the write-set names — the mutated table's `TableVersionData`
    /// (row chains + value-index, via the now-`&self` COW `with_table_mut`) and the atomic
    /// relational row-id / tuple-id allocators — then **publishes** one new generation for that
    /// table. The CALLER provides serialization (the commit critical section under the commit_mutex,
    /// or a serialized DDL apply under the catalog latch), so concurrent committers never interleave.
    ///
    /// `nextval` sequence advancement is NOT applied here (sequences are not interior-mutable): a
    /// delta carrying `seq_advances` MUST go through the serialized [`Engine::apply_delta_serialized`]
    /// (`&mut self`), which applies the advancement first. `apply_delta` asserts the delta is
    /// sequence-free.
    fn apply_delta(
        &self,
        delta: WriteDelta,
        commit_seq: TxnId,
        mut profile: Option<&mut RelationalCopyAdmissionProfile>,
    ) -> Result<(), EngineError> {
        match delta.mutation {
            PreparedMutation::Insert {
                table,
                inserted_rows,
                value_index_entries,
                seq_advances,
            } => {
                debug_assert!(
                    seq_advances.is_empty(),
                    "apply_delta (&self) cannot install nextval sequence advances; route through \
                     apply_delta_serialized"
                );
                // Reserve the globally-unique tuple ids up front (the old in-line bump consumed one
                // per row from the single shared allocator; `next_tuple_id` is now shared across all
                // partitions so ids are identical). Advance the relational row-id allocator by the
                // same count `prepare_insert` already computed its row keys from.
                let tuple_ids: Vec<TupleId> = (0..inserted_rows.len())
                    .map(|_| self.read_state.mvcc.reserve_tuple_id())
                    .collect();
                self.read_state
                    .mvcc
                    .advance_row_id(inserted_rows.len() as u64);
                let mvcc_insert_started = Instant::now();
                let insert_result = self.read_state.mvcc.with_table_mut(&table, |data| {
                    for (tuple_id, (row_key, values)) in tuple_ids.iter().zip(inserted_rows.iter())
                    {
                        data.rows
                            .tuple_insert_reserved_key_with_id(
                                *tuple_id,
                                NewTuple {
                                    key: row_key.clone(),
                                    value: encode_relational_row(values),
                                },
                                commit_seq,
                            )
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    // Append the value-index entries within the SAME published generation, so a
                    // reader that loads it sees rows + value-index mutually consistent. `Arc::make_mut`
                    // copies a slot's row-key list ONLY if a live snapshot still shares it (COW),
                    // keeping the per-commit cost O(k·log n) for the k touched slots.
                    for (key, mut row_keys) in value_index_entries {
                        let mut slot = data.value_index.get(&key).cloned().unwrap_or_default();
                        std::sync::Arc::make_mut(&mut slot).append(&mut row_keys);
                        data.value_index.insert(key, slot);
                    }
                    Ok::<(), EngineError>(())
                });
                insert_result?;
                if let Some(profile) = profile.as_mut() {
                    // The row inserts and the value-index append now happen inside one published
                    // mutation (`with_table_mut`); attribute the whole window to the insert timer.
                    profile.mvcc_insert_micros += mvcc_insert_started.elapsed().as_micros();
                }
                debug_assert_eq!(inserted_rows.len() as u64, delta.rows_consumed);
            }
            PreparedMutation::Update {
                table,
                installs,
                value_index_entries,
            } => {
                self.read_state.mvcc.with_table_mut(&table, |data| {
                    for (tuple_id, _row_key, values) in &installs {
                        data.rows
                            .tuple_update(*tuple_id, encode_relational_row(values), commit_seq)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    for (key, mut row_keys) in value_index_entries {
                        let mut slot = data.value_index.get(&key).cloned().unwrap_or_default();
                        std::sync::Arc::make_mut(&mut slot).append(&mut row_keys);
                        data.value_index.insert(key, slot);
                    }
                    Ok::<(), EngineError>(())
                })?;
            }
            PreparedMutation::Delete { table, tuple_ids } => {
                self.read_state.mvcc.with_table_mut(&table, |data| {
                    for tuple_id in tuple_ids {
                        data.rows
                            .tuple_delete(tuple_id, commit_seq)
                            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    }
                    Ok::<(), EngineError>(())
                })?;
            }
        }
        Ok(())
    }

    /// `&mut self` install for the SERIALIZED commit path (DDL / COPY / replay): apply any `nextval`
    /// sequence advancement (needs `&mut self` — sequences are not interior-mutable) and then install
    /// the rest of the delta via the `&self` [`Engine::apply_delta`]. Behaviorally identical to the
    /// pre-Stage-4 `apply_delta` (sequence advance first, then rows + value-index), so the live
    /// serialized apply stays byte-identical to a WAL replay.
    fn apply_delta_serialized(
        &self,
        cat: &mut DdlCatalogState,
        mut delta: WriteDelta,
        commit_seq: TxnId,
        profile: Option<&mut RelationalCopyAdmissionProfile>,
    ) -> Result<(), EngineError> {
        if let PreparedMutation::Insert { seq_advances, .. } = &mut delta.mutation {
            // Install the sequence advancement `prepare_insert` computed for `nextval` column
            // defaults (before the row inserts, matching the old apply). Idempotent assignment of the
            // final `(last_value, is_called)`.
            let advances = std::mem::take(seq_advances);
            for (sequence, (last_value, is_called)) in advances {
                if let Some(seq) = cat.relational_sequences.get_mut(&sequence) {
                    seq.last_value = last_value;
                    seq.is_called = is_called;
                }
            }
        }
        self.apply_delta(delta, commit_seq, profile)
    }

    fn apply_insert_with_profile(
        &self,
        cat: &mut DdlCatalogState,
        insert: Insert,
        txn_id: TxnId,
        mut profile: Option<&mut RelationalCopyAdmissionProfile>,
    ) -> Result<(), EngineError> {
        // Stage 2 split: PURE prepare (preflight + encode + write-set) then a `&mut self` install,
        // both under the existing commit lock so the result is byte-identical to the old direct
        // apply. `txn_id` is the commit-seq (== `entry.index`), used as BOTH the read boundary and
        // the version stamp exactly as before. The snapshot is taken immediately before prepare, so
        // `next_row_id` and the read visibility match what the in-line apply used.
        let snapshot = self.dml_read_snapshot(txn_id);
        let delta = self.prepare_insert(&insert, snapshot, profile.as_deref_mut())?;
        self.apply_delta_serialized(cat, delta, txn_id, profile)
    }

    fn apply_delete(
        &self,
        cat: &mut DdlCatalogState,
        delete: Delete,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        // Stage 2 split: PURE prepare (resolve matches + FK preflight + write-set) then a
        // `&mut self` tombstone install. `txn_id` is the commit-seq used as both the read boundary
        // and the version stamp, identical to the old direct apply (still under the commit lock).
        let snapshot = self.dml_read_snapshot(txn_id);
        let delta = self.prepare_delete(&delete, snapshot)?;
        self.apply_delta_serialized(cat, delta, txn_id, None)
    }

    fn apply_update(
        &self,
        cat: &mut DdlCatalogState,
        update: Update,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        // Stage 2 split: PURE prepare (resolve matches + encode new images + preflight +
        // write-set) then a `&mut self` version-rewrite install. `txn_id` is the commit-seq used as
        // both the read boundary and the version stamp, identical to the old direct apply.
        let snapshot = self.dml_read_snapshot(txn_id);
        let delta = self.prepare_update(&update, snapshot)?;
        self.apply_delta_serialized(cat, delta, txn_id, None)
    }

    fn preflight_unique_index_constraints(
        &self,
        cmd: &Command,
        txn_id: TxnId,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        match cmd {
            Command::CreateSchema(create) => {
                if create.name != PUBLIC_SCHEMA_NAME {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" is not supported",
                        create.name
                    )));
                }
                if cat.relational_public_schema_exists
                    && !create.if_not_exists
                    && !cat.relational_public_schema_implicit
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" already exists",
                        create.name
                    )));
                }
            }
            Command::DropSchema(drop) => {
                if drop.name != PUBLIC_SCHEMA_NAME {
                    if !drop.if_exists {
                        return Err(EngineError::ApplyFailed(format!(
                            "schema \"{}\" does not exist",
                            drop.name
                        )));
                    }
                    return Ok(());
                }
                if !cat.relational_public_schema_exists && !drop.if_exists {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" does not exist",
                        drop.name
                    )));
                }
                if cat.relational_public_schema_exists
                    && (!cat.relational_catalog.is_empty()
                        || !cat.relational_views.is_empty()
                        || !cat.relational_materialized_views.is_empty()
                        || !cat.relational_functions.is_empty()
                        || !cat.relational_sequences.is_empty()
                        || !cat.relational_domains.is_empty()
                        || !cat.relational_publications.is_empty()
                        || !cat.relational_subscriptions.is_empty())
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot drop non-empty schema \"{}\"",
                        drop.name
                    )));
                }
            }
            Command::CreateDatabase(create) if self.database_exists(&create.name) => {
                return Err(EngineError::ApplyFailed(format!(
                    "database \"{}\" already exists",
                    create.name
                )));
            }
            Command::CreateDatabase(_) => {}
            Command::DropDatabase(drop) => {
                let mut seen = BTreeSet::new();
                for database in &drop.names {
                    if !seen.insert(database.clone()) {
                        return Err(EngineError::ApplyFailed(format!(
                            "database \"{}\" specified more than once",
                            database
                        )));
                    }
                    if database == "postgres" {
                        return Err(EngineError::ApplyFailed(
                            "cannot drop bootstrap database \"postgres\"".to_string(),
                        ));
                    }
                    if !drop.if_exists && !cat.relational_databases.contains_key(database) {
                        return Err(EngineError::ApplyFailed(format!(
                            "database \"{}\" does not exist",
                            database
                        )));
                    }
                }
            }
            Command::RenameDatabase(rename) => {
                if rename.old_name == "postgres" {
                    return Err(EngineError::ApplyFailed(
                        "cannot rename bootstrap database \"postgres\"".to_string(),
                    ));
                }
                if !cat.relational_databases.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "database \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if self.database_exists(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "database \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::CreateTablespace(create) if self.tablespace_exists(&create.name) => {
                return Err(EngineError::ApplyFailed(format!(
                    "tablespace \"{}\" already exists",
                    create.name
                )));
            }
            Command::CreateTablespace(_) => {}
            Command::DropTablespace(drop) => {
                let mut seen = BTreeSet::new();
                for tablespace in &drop.names {
                    if !seen.insert(tablespace.clone()) {
                        return Err(EngineError::ApplyFailed(format!(
                            "tablespace \"{}\" specified more than once",
                            tablespace
                        )));
                    }
                    if matches!(tablespace.as_str(), "pg_default" | "pg_global") {
                        return Err(EngineError::ApplyFailed(format!(
                            "cannot drop bootstrap tablespace \"{}\"",
                            tablespace
                        )));
                    }
                    if !drop.if_exists && !cat.relational_tablespaces.contains_key(tablespace) {
                        return Err(EngineError::ApplyFailed(format!(
                            "tablespace \"{}\" does not exist",
                            tablespace
                        )));
                    }
                }
            }
            Command::RenameTablespace(rename) => {
                if matches!(rename.old_name.as_str(), "pg_default" | "pg_global") {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename bootstrap tablespace \"{}\"",
                        rename.old_name
                    )));
                }
                if !cat.relational_tablespaces.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "tablespace \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if self.tablespace_exists(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "tablespace \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::CreateTable(create) => {
                if !cat.relational_public_schema_exists {
                    return Err(EngineError::ApplyFailed(format!(
                        "schema \"{}\" does not exist",
                        PUBLIC_SCHEMA_NAME
                    )));
                }
                let mut implicit_sequences = BTreeSet::new();
                for column in &create.columns {
                    if let Some(domain_name) = column.domain.as_ref() {
                        if !cat.relational_domains.contains_key(domain_name) {
                            return Err(EngineError::ApplyFailed(format!(
                                "type \"{}\" does not exist",
                                domain_name
                            )));
                        }
                    }
                }
                for default in sequence_defaults(&create.columns) {
                    if let ColumnDefault::SequenceNextVal {
                        sequence,
                        create_if_missing: true,
                    } = default
                    {
                        if !implicit_sequences.insert(sequence.clone()) {
                            return Err(EngineError::ApplyFailed(format!(
                                "relation \"{sequence}\" already exists"
                            )));
                        }
                    }
                    self.preflight_column_default_target(default)?;
                }
            }
            Command::CreateIndex(create) if create.unique => {
                if cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == create.name))
                    || cat.relational_catalog.contains_key(&create.name)
                    || cat.relational_views.contains_key(&create.name)
                    || cat.relational_materialized_views.contains_key(&create.name)
                    || cat.relational_sequences.contains_key(&create.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        create.name
                    )));
                }
                let table = cat.relational_catalog.get(&create.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        create.table
                    ))
                })?;
                let column_idx = table
                    .columns
                    .iter()
                    .position(|column| column.name == create.column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "column \"{}\" does not exist",
                            create.column
                        ))
                    })?;
                let rows = self.visible_relational_rows(
                    table,
                    StorageVisibility {
                        read_txn_id: self.committed_seq() as TxnId,
                    },
                )?;
                Self::validate_unique_values(&rows, column_idx, &create.name)?;
            }
            Command::AddPrimaryKey(add) => {
                if cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == add.name))
                    || cat.relational_catalog.contains_key(&add.name)
                    || cat.relational_views.contains_key(&add.name)
                    || cat.relational_materialized_views.contains_key(&add.name)
                    || cat.relational_sequences.contains_key(&add.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        add.name
                    )));
                }
                let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
                })?;
                if table.indexes.iter().any(|index| index.primary_key) {
                    return Err(EngineError::ApplyFailed(format!(
                        "multiple primary keys for table \"{}\" are not allowed",
                        add.table
                    )));
                }
                let column_idx = table
                    .columns
                    .iter()
                    .position(|column| column.name == add.column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "column \"{}\" does not exist",
                            add.column
                        ))
                    })?;
                let rows = self.visible_relational_rows(
                    table,
                    StorageVisibility {
                        read_txn_id: self.committed_seq() as TxnId,
                    },
                )?;
                Self::validate_unique_values(&rows, column_idx, &add.name)?;
            }
            Command::AddUniqueConstraint(add) => {
                if cat
                    .relational_catalog
                    .values()
                    .any(|table| table.indexes.iter().any(|index| index.name == add.name))
                    || cat.relational_catalog.contains_key(&add.name)
                    || cat.relational_views.contains_key(&add.name)
                    || cat.relational_materialized_views.contains_key(&add.name)
                    || cat.relational_sequences.contains_key(&add.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        add.name
                    )));
                }
                let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
                })?;
                let column_idx = table
                    .columns
                    .iter()
                    .position(|column| column.name == add.column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "column \"{}\" does not exist",
                            add.column
                        ))
                    })?;
                let rows = self.visible_relational_rows(
                    table,
                    StorageVisibility {
                        read_txn_id: self.committed_seq() as TxnId,
                    },
                )?;
                Self::validate_unique_values(&rows, column_idx, &add.name)?;
            }
            Command::AddCheckConstraint(add) => self.preflight_add_check_constraint(add)?,
            Command::AddForeignKey(add) => self.preflight_add_foreign_key(add, txn_id)?,
            Command::AddColumn(add) => {
                if cat.relational_views.contains_key(&add.table)
                    || cat.relational_materialized_views.contains_key(&add.table)
                    || cat.relational_sequences.contains_key(&add.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        add.table
                    )));
                }
                let Some(default) = add.column.default.as_ref() else {
                    return Err(EngineError::ApplyFailed(
                        "ADD COLUMN requires a supported DEFAULT in the bootstrap relational subset"
                            .to_string(),
                    ));
                };
                if !add_column_default_supported(default) {
                    return Err(EngineError::ApplyFailed(
                        "ADD COLUMN SERIAL is unsupported in the bootstrap relational subset"
                            .to_string(),
                    ));
                }
                // Validate the default is coercible to the column type (parity with apply);
                // this concurrent-DDL preflight only checks — apply coerces and stores.
                coerce_column_default(default.clone(), add.column.ty, &add.column.name)?;
                let table = cat.relational_catalog.get(&add.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", add.table))
                })?;
                if table
                    .columns
                    .iter()
                    .any(|column| column.name == add.column.name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" of relation \"{}\" already exists",
                        add.column.name, add.table
                    )));
                }
                self.preflight_column_default_target(default)?;
            }
            Command::RenameTable(rename) => {
                if cat.relational_views.contains_key(&rename.old_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.old_name)
                    || cat.relational_sequences.contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        rename.old_name
                    )));
                }
                if !cat.relational_catalog.contains_key(&rename.old_name) {
                    if rename.if_exists {
                        return Ok(());
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                if cat
                    .relational_views
                    .values()
                    .any(|view| view.query.table == rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename relation \"{}\" because a view depends on it",
                        rename.old_name
                    )));
                }
            }
            Command::RenameColumn(rename) => {
                if cat.relational_views.contains_key(&rename.table)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.table)
                    || cat.relational_sequences.contains_key(&rename.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        rename.table
                    )));
                }
                let table = cat.relational_catalog.get(&rename.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        rename.table
                    ))
                })?;
                if !table
                    .columns
                    .iter()
                    .any(|column| column.name == rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if table
                    .columns
                    .iter()
                    .any(|column| column.name == rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" of relation \"{}\" already exists",
                        rename.new_name, rename.table
                    )));
                }
            }
            Command::RenameConstraint(rename) => {
                if cat.relational_views.contains_key(&rename.table)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.table)
                    || cat.relational_sequences.contains_key(&rename.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        rename.table
                    )));
                }
                let Some(table) = cat.relational_catalog.get(&rename.table) else {
                    if rename.table_if_exists {
                        return Ok(());
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        rename.table
                    )));
                };
                if cat.relational_catalog.values().any(|candidate| {
                    candidate
                        .indexes
                        .iter()
                        .any(|index| index.name == rename.new_name)
                        || candidate
                            .check_constraints
                            .iter()
                            .any(|constraint| constraint.name == rename.new_name)
                        || candidate
                            .foreign_keys
                            .iter()
                            .any(|constraint| constraint.name == rename.new_name)
                }) || cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                if !table.indexes.iter().any(|index| {
                    index.name == rename.old_name && (index.primary_key || index.unique_constraint)
                }) && !table
                    .check_constraints
                    .iter()
                    .any(|constraint| constraint.name == rename.old_name)
                    && !table
                        .foreign_keys
                        .iter()
                        .any(|constraint| constraint.name == rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "constraint \"{}\" does not exist",
                        rename.old_name
                    )));
                }
            }
            Command::RenameIndex(rename) => {
                if cat.relational_catalog.values().any(|table| {
                    table
                        .indexes
                        .iter()
                        .any(|index| index.name == rename.new_name)
                }) || cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                let Some(index) = cat
                    .relational_catalog
                    .values()
                    .flat_map(|table| table.indexes.iter())
                    .find(|index| index.name == rename.old_name)
                else {
                    return Err(EngineError::ApplyFailed(format!(
                        "index \"{}\" does not exist",
                        rename.old_name
                    )));
                };
                if index.primary_key || index.unique_constraint {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename constraint-backed index \"{}\" with ALTER INDEX",
                        rename.old_name
                    )));
                }
            }
            Command::CreateView(create) => {
                if cat.relational_catalog.contains_key(&create.name)
                    || cat.relational_materialized_views.contains_key(&create.name)
                    || cat.relational_sequences.contains_key(&create.name)
                    || (!create.or_replace && cat.relational_views.contains_key(&create.name))
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        create.name
                    )));
                }
                if cat
                    .relational_materialized_views
                    .contains_key(&create.query.table)
                {
                    return Err(EngineError::ApplyFailed(
                        "views over materialized views are unsupported".to_string(),
                    ));
                }
                if create.or_replace && self.relational_view_has_dependents(&create.name) {
                    return Err(EngineError::ApplyFailed(
                        "cannot replace view because another view depends on it".to_string(),
                    ));
                }
                if cat.relational_views.contains_key(&create.query.table) {
                    if self.relational_view_depends_on(&create.query.table, &create.name) {
                        return Err(EngineError::ApplyFailed(
                            "view dependency cycle is unsupported".to_string(),
                        ));
                    }
                } else if !cat.relational_catalog.contains_key(&create.query.table) {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        create.query.table
                    )));
                }
            }
            Command::RenameView(rename) => {
                if cat.relational_catalog.contains_key(&rename.old_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.old_name)
                    || cat.relational_sequences.contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a view",
                        rename.old_name
                    )));
                }
                if !cat.relational_views.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "view \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
                if self.relational_view_has_dependents(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot rename view \"{}\" because another view depends on it",
                        rename.old_name
                    )));
                }
            }
            Command::CreateMaterializedView(create) => {
                self.preflight_create_materialized_view(create)?
            }
            Command::RefreshMaterializedView(refresh) => {
                self.preflight_refresh_materialized_view(refresh)?
            }
            Command::RenameMaterializedView(rename) => {
                if cat.relational_catalog.contains_key(&rename.old_name)
                    || cat.relational_views.contains_key(&rename.old_name)
                    || cat.relational_sequences.contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a materialized view",
                        rename.old_name
                    )));
                }
                if !cat
                    .relational_materialized_views
                    .contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "materialized view \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::CreateFunction(create)
                if cat.relational_functions.contains_key(&create.name) =>
            {
                return Err(EngineError::ApplyFailed(format!(
                    "function \"{}\" already exists",
                    create.name
                )));
            }
            Command::RenameFunction(rename) => {
                if !cat.relational_functions.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "function \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_functions.contains_key(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "function \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::DropFunction(drop)
                if !drop.if_exists && !cat.relational_functions.contains_key(&drop.name) =>
            {
                return Err(EngineError::ApplyFailed(format!(
                    "function \"{}\" does not exist",
                    drop.name
                )));
            }
            Command::CreateFunction(_) | Command::DropFunction(_) => {}
            Command::CommentOn(comment) => {
                if let CommentTarget::Function { function } = &comment.target {
                    if !cat.relational_functions.contains_key(function) {
                        return Err(EngineError::ApplyFailed(format!(
                            "function \"{}\" does not exist",
                            function
                        )));
                    }
                }
            }
            Command::CreateSequence(create) => self.preflight_create_sequence(create)?,
            Command::CreateDomain(create) => self.preflight_create_domain(create)?,
            Command::SequenceNextVal(nextval) => self.preflight_sequence_target(&nextval.name)?,
            Command::SequenceSetVal(setval) => self.preflight_sequence_target(&setval.name)?,
            Command::RenameSequence(rename) => {
                if cat.relational_catalog.contains_key(&rename.old_name)
                    || cat.relational_views.contains_key(&rename.old_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.old_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a sequence",
                        rename.old_name
                    )));
                }
                if !cat.relational_sequences.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "sequence \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if cat.relational_catalog.contains_key(&rename.new_name)
                    || cat.relational_views.contains_key(&rename.new_name)
                    || cat
                        .relational_materialized_views
                        .contains_key(&rename.new_name)
                    || cat.relational_sequences.contains_key(&rename.new_name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::DropColumn(drop) => {
                if cat.relational_views.contains_key(&drop.table)
                    || cat.relational_materialized_views.contains_key(&drop.table)
                    || cat.relational_sequences.contains_key(&drop.table)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a table",
                        drop.table
                    )));
                }
                let table = cat.relational_catalog.get(&drop.table).ok_or_else(|| {
                    EngineError::ApplyFailed(format!("relation \"{}\" does not exist", drop.table))
                })?;
                if !table
                    .columns
                    .iter()
                    .any(|column| column.name == drop.column)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "column \"{}\" does not exist",
                        drop.column
                    )));
                }
                if table
                    .indexes
                    .iter()
                    .any(|index| index.column == drop.column)
                    || table
                        .check_constraints
                        .iter()
                        .any(|constraint| constraint.column == drop.column)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "cannot drop column \"{}\" because an index or constraint depends on it",
                        drop.column
                    )));
                }
            }
            Command::DropConstraint(drop) => {
                let Some(table) = cat.relational_catalog.get(&drop.table) else {
                    if drop.table_if_exists {
                        return Ok(());
                    }
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        drop.table
                    )));
                };
                if !table.indexes.iter().any(|index| {
                    index.name == drop.name && (index.primary_key || index.unique_constraint)
                }) && !table
                    .check_constraints
                    .iter()
                    .any(|constraint| constraint.name == drop.name)
                    && !table
                        .foreign_keys
                        .iter()
                        .any(|constraint| constraint.name == drop.name)
                    && !drop.if_exists
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "constraint \"{}\" does not exist",
                        drop.name
                    )));
                }
            }
            Command::DropTable(drop) => self.preflight_drop_table(drop)?,
            Command::DropIndex(drop) => self.preflight_drop_index(drop)?,
            Command::DropView(drop) => self.preflight_drop_view(drop)?,
            Command::DropMaterializedView(drop) => self.preflight_drop_materialized_view(drop)?,
            Command::DropSequence(drop) => self.preflight_drop_sequence(drop)?,
            Command::DropDomain(drop) => self.preflight_drop_domain(drop)?,
            Command::GrantTable(grant) => {
                self.preflight_acl_target(&grant.relation, grant.kind)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeTable(revoke) => {
                self.preflight_acl_target(&revoke.relation, revoke.kind)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantSchema(grant) => {
                self.preflight_schema_acl_target(&grant.schema)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeSchema(revoke) => {
                self.preflight_schema_acl_target(&revoke.schema)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantDatabase(grant) => {
                self.preflight_database_acl_target(&grant.database)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeDatabase(revoke) => {
                self.preflight_database_acl_target(&revoke.database)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantTablespace(grant) => {
                self.preflight_tablespace_acl_target(&grant.tablespace)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeTablespace(revoke) => {
                self.preflight_tablespace_acl_target(&revoke.tablespace)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::GrantFunction(grant) => {
                self.preflight_function_acl_target(&grant.function)?;
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeFunction(revoke) => {
                self.preflight_function_acl_target(&revoke.function)?;
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::CreatePublication(create) => self.preflight_create_publication(create)?,
            Command::DropPublication(drop) => self.preflight_drop_publication(drop)?,
            Command::CreateSubscription(create) => self.preflight_create_subscription(create)?,
            Command::DropSubscription(drop) => self.preflight_drop_subscription(drop)?,
            Command::CreateRole(create) if self.role_exists(&create.name) => {
                return Err(EngineError::ApplyFailed(format!(
                    "role \"{}\" already exists",
                    create.name
                )));
            }
            Command::CreateRole(_) => {}
            Command::DropRole(drop) => {
                let mut seen = BTreeSet::new();
                for role in &drop.names {
                    if !seen.insert(role.clone()) {
                        return Err(EngineError::ApplyFailed(format!(
                            "role \"{}\" specified more than once",
                            role
                        )));
                    }
                    if role == "postgres" {
                        return Err(EngineError::ApplyFailed(
                            "cannot drop bootstrap role \"postgres\"".to_string(),
                        ));
                    }
                    if !drop.if_exists && !cat.relational_roles.contains_key(role) {
                        return Err(EngineError::ApplyFailed(format!(
                            "role \"{}\" does not exist",
                            role
                        )));
                    }
                    if cat.relational_roles.contains_key(role) && self.role_has_dependencies(role) {
                        return Err(EngineError::ApplyFailed(format!(
                            "role \"{}\" cannot be dropped because dependent metadata exists",
                            role
                        )));
                    }
                }
            }
            Command::RenameRole(rename) => {
                if rename.old_name == "postgres" {
                    return Err(EngineError::ApplyFailed(
                        "cannot rename bootstrap role \"postgres\"".to_string(),
                    ));
                }
                if !cat.relational_roles.contains_key(&rename.old_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "role \"{}\" does not exist",
                        rename.old_name
                    )));
                }
                if self.role_exists(&rename.new_name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "role \"{}\" already exists",
                        rename.new_name
                    )));
                }
            }
            Command::GrantDefaultTablePrivileges(grant) => {
                self.preflight_acl_grantee(&grant.grantee)?;
            }
            Command::RevokeDefaultTablePrivileges(revoke) => {
                self.preflight_acl_grantee(&revoke.grantee)?;
            }
            Command::Insert(insert) => {
                // Lock-free concurrent-DML preflight (Stage 2 — blocker #1): pin ONE catalog snapshot
                // for the whole arm (its `table` borrow + the inbound-FK-dependents scan).
                let catalog = self.catalog_snapshot();
                let table = catalog
                    .relational_catalog
                    .get(&insert.table)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "relation \"{}\" does not exist",
                            insert.table
                        ))
                    })?;
                let column_indexes = if insert.columns.is_empty() {
                    (0..table.columns.len()).collect::<Vec<_>>()
                } else {
                    let mut indexes = Vec::with_capacity(insert.columns.len());
                    for column in &insert.columns {
                        let idx = table
                            .columns
                            .iter()
                            .position(|candidate| candidate.name == *column)
                            .ok_or_else(|| {
                                EngineError::ApplyFailed(format!(
                                    "column \"{}\" does not exist",
                                    column
                                ))
                            })?;
                        indexes.push(idx);
                    }
                    indexes
                };
                let mut simulated_sequences = cat.relational_sequences.clone();
                let mut new_rows = Vec::with_capacity(insert.rows.len());
                for row in &insert.rows {
                    if row.len() != column_indexes.len() {
                        return Err(EngineError::ApplyFailed(
                            "INSERT value count must match target columns".to_string(),
                        ));
                    }
                    let mut values = vec![None; table.columns.len()];
                    for (source_idx, target_idx) in column_indexes.iter().copied().enumerate() {
                        let value = row[source_idx].clone();
                        let expected_ty = table.columns[target_idx].ty;
                        let coerced = coerce_insert_value(
                            value,
                            expected_ty,
                            &table.columns[target_idx].name,
                        )?;
                        values[target_idx] = Some(coerced);
                    }
                    for (idx, value) in values.iter_mut().enumerate() {
                        if value.is_none() {
                            if let Some(default) = table.columns[idx].default.clone() {
                                *value = Some(match default {
                                    ColumnDefault::Literal(value) => value,
                                    ColumnDefault::SequenceNextVal { sequence, .. } => {
                                        self.preflight_sequence_target(&sequence)?;
                                        let sequence_state = simulated_sequences
                                            .get_mut(&sequence)
                                            .expect("sequence target preflighted");
                                        let value = if sequence_state.is_called {
                                            sequence_state.last_value.checked_add(1).ok_or_else(
                                                || {
                                                    EngineError::ApplyFailed(
                                                        "sequence value overflow".to_string(),
                                                    )
                                                },
                                            )?
                                        } else {
                                            sequence_state.last_value
                                        };
                                        sequence_state.last_value = value;
                                        sequence_state.is_called = true;
                                        SqlValue::Int4(i32::try_from(value).map_err(|_| {
                                            EngineError::ApplyFailed(
                                                "sequence value is out of range for int4 default"
                                                    .to_string(),
                                            )
                                        })?)
                                    }
                                });
                            }
                        }
                    }
                    if values.iter().any(Option::is_none) {
                        return Err(EngineError::ApplyFailed(
                            "INSERT must provide every column without a default in the bootstrap relational subset"
                                .to_string(),
                        ));
                    }
                    new_rows.push(values.into_iter().map(Option::unwrap).collect());
                }
                if !table.indexes.iter().any(|index| index.unique)
                    && table.check_constraints.is_empty()
                    && table.foreign_keys.is_empty()
                    && !catalog.relational_catalog.values().any(|candidate| {
                        candidate
                            .foreign_keys
                            .iter()
                            .any(|foreign_key| foreign_key.referenced_table == table.name)
                    })
                {
                    return Ok(());
                }
                let mut candidate_rows = self.visible_relational_rows(
                    table,
                    StorageVisibility {
                        read_txn_id: txn_id,
                    },
                )?;
                candidate_rows.extend(new_rows);
                Self::validate_unique_indexes_for_rows(table, &candidate_rows)?;
                Self::validate_check_constraints_for_rows(table, &candidate_rows)?;
                self.validate_foreign_keys_with_table_rows(
                    &table.name,
                    &candidate_rows,
                    StorageVisibility {
                        read_txn_id: txn_id,
                    },
                )?;
            }
            Command::Update(update) => {
                // Lock-free concurrent-DML preflight (Stage 2 — blocker #1): pin ONE catalog snapshot
                // for the whole arm.
                let catalog = self.catalog_snapshot();
                let table = catalog
                    .relational_catalog
                    .get(&update.table)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "relation \"{}\" does not exist",
                            update.table
                        ))
                    })?;
                if !table.indexes.iter().any(|index| index.unique)
                    && table.check_constraints.is_empty()
                    && table.foreign_keys.is_empty()
                    && !catalog.relational_catalog.values().any(|candidate| {
                        candidate
                            .foreign_keys
                            .iter()
                            .any(|foreign_key| foreign_key.referenced_table == table.name)
                    })
                {
                    return Ok(());
                }
                let assignments = bind_update_assignments(table, update)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let filter_groups = bind_delete_filter_groups(
                    table,
                    &Delete {
                        table: update.table.clone(),
                        filter: update.filter.clone(),
                        filters: update.filters.clone(),
                        filter_groups: update.filter_groups.clone(),
                    },
                )
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let visibility = StorageVisibility {
                    read_txn_id: txn_id,
                };
                let prefix = relational_key_prefix(&update.table);
                let mut candidate_rows = Vec::new();
                let table_rows = self.read_state.mvcc.table_rows(&update.table);
                let mut cursor = table_rows
                    .store()
                    .seq_scan_open(visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                while let Some(tuple) = cursor.next() {
                    if !tuple.key.starts_with(&prefix) {
                        continue;
                    }
                    let mut row = decode_relational_row(&tuple.value, &table.columns)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    if filter_groups.iter().any(|filters| {
                        filters
                            .iter()
                            .all(|(idx, op, value)| select_filter_matches(&row[*idx], *op, value))
                    }) {
                        for (idx, value) in &assignments {
                            row[*idx] = value.clone();
                        }
                    }
                    candidate_rows.push(row);
                }
                Self::validate_unique_indexes_for_rows(table, &candidate_rows)?;
                Self::validate_check_constraints_for_rows(table, &candidate_rows)?;
                self.validate_foreign_keys_with_table_rows(
                    &table.name,
                    &candidate_rows,
                    visibility,
                )?;
            }
            Command::Delete(delete) => {
                // Lock-free concurrent-DML preflight (Stage 2 — blocker #1): pin ONE catalog snapshot
                // for the whole arm.
                let catalog = self.catalog_snapshot();
                let table = catalog
                    .relational_catalog
                    .get(&delete.table)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "relation \"{}\" does not exist",
                            delete.table
                        ))
                    })?;
                if !catalog.relational_catalog.values().any(|candidate| {
                    candidate
                        .foreign_keys
                        .iter()
                        .any(|foreign_key| foreign_key.referenced_table == table.name)
                }) {
                    return Ok(());
                }
                let filter_groups = bind_delete_filter_groups(table, delete)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let visibility = StorageVisibility {
                    read_txn_id: txn_id,
                };
                let prefix = relational_key_prefix(&delete.table);
                let mut candidate_rows = Vec::new();
                let table_rows = self.read_state.mvcc.table_rows(&delete.table);
                let mut cursor = table_rows
                    .store()
                    .seq_scan_open(visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                while let Some(tuple) = cursor.next() {
                    if !tuple.key.starts_with(&prefix) {
                        continue;
                    }
                    let row = decode_relational_row(&tuple.value, &table.columns)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    if !filter_groups.iter().any(|filters| {
                        filters
                            .iter()
                            .all(|(idx, op, value)| select_filter_matches(&row[*idx], *op, value))
                    }) {
                        candidate_rows.push(row);
                    }
                }
                self.validate_foreign_keys_with_table_rows(
                    &table.name,
                    &candidate_rows,
                    visibility,
                )?;
            }
            _ => {}
        }
        Ok(())
    }

    fn command_requires_immediate_unique_index_commit(&self, cmd: &Command) -> bool {
        let cat = self.catalog_snapshot();
        match cmd {
            Command::AddPrimaryKey(_) => true,
            Command::AddUniqueConstraint(_) => true,
            Command::AddCheckConstraint(_) => true,
            Command::AddForeignKey(_) => true,
            Command::DropConstraint(_) => true,
            Command::CreateIndex(create) => create.unique,
            Command::Insert(insert) => {
                cat.relational_catalog
                    .get(&insert.table)
                    .is_some_and(|table| {
                        table.indexes.iter().any(|index| index.unique)
                            || !table.check_constraints.is_empty()
                            || !table.foreign_keys.is_empty()
                    })
            }
            Command::Update(update) => {
                cat.relational_catalog
                    .get(&update.table)
                    .is_some_and(|table| {
                        table.indexes.iter().any(|index| index.unique)
                            || !table.check_constraints.is_empty()
                            || !table.foreign_keys.is_empty()
                            || cat.relational_catalog.values().any(|candidate| {
                                candidate
                                    .foreign_keys
                                    .iter()
                                    .any(|foreign_key| foreign_key.referenced_table == table.name)
                            })
                    })
            }
            Command::Delete(delete) => {
                cat.relational_catalog
                    .get(&delete.table)
                    .is_some_and(|table| {
                        cat.relational_catalog.values().any(|candidate| {
                            candidate
                                .foreign_keys
                                .iter()
                                .any(|foreign_key| foreign_key.referenced_table == table.name)
                        })
                    })
            }
            _ => false,
        }
    }

    pub fn enqueue_set_text(
        &mut self,
        txn_id: u64,
        text: &str,
        now: Instant,
    ) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;
        match cmd {
            Command::SetKv { .. }
            | Command::DeleteKv { .. }
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
            | Command::CreateSequence(_)
            | Command::CreateDomain(_)
            | Command::SequenceNextVal(_)
            | Command::SequenceSetVal(_)
            | Command::RenameSequence(_)
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
            | Command::CreatePublication(_)
            | Command::DropPublication(_)
            | Command::CreateSubscription(_)
            | Command::DropSubscription(_)
            | Command::CreateRole(_)
            | Command::DropRole(_)
            | Command::RenameRole(_)
            | Command::GrantDefaultTablePrivileges(_)
            | Command::RevokeDefaultTablePrivileges(_)
            | Command::AlterColumnDefault(_)
            | Command::CommentOn(_)
            | Command::Insert(_)
            | Command::Delete(_)
            | Command::Update(_) => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.preflight_unique_index_constraints(&cmd, txn_id)?;
                if self.command_requires_immediate_unique_index_commit(&cmd) {
                    self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                    self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                    return Ok(());
                }

                match self.route_command(&cmd) {
                    RouteDecision::Gpu(_) => {
                        let queue_cap = self.batcher().max_items();
                        let pending = self.batcher().len();
                        if pending >= queue_cap {
                            self.metrics.inc_fallback(FallbackReason::GpuQueueSaturated);
                            return Err(ExecuteError::Engine(
                                EngineError::MutationQueueOverloaded {
                                    pending,
                                    cap: queue_cap,
                                },
                            ));
                        }

                        let maybe_batch = self.batcher().enqueue(
                            PendingMutation {
                                txn_id,
                                payload: text.as_bytes().to_vec(),
                            },
                            now,
                        );
                        if let Some(batch) = maybe_batch {
                            self.metrics.observe_pending_batch_len(batch.items.len());
                            self.apply_batch(batch.reason, batch.items.into_iter(), now)?;
                        } else {
                            self.metrics.observe_pending_batch_len(self.batcher().len());
                        }
                    }
                    RouteDecision::CpuFallback { reason, .. } => {
                        self.metrics.inc_gpu_fallback(reason);
                        self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                    }
                    RouteDecision::Cpu => {
                        self.commit_mutation(txn_id, text.as_bytes().to_vec())?;
                    }
                }
            }
            Command::Flush => {
                self.flush_admin()?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::ResetAll | Command::SetRole { .. } => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::CreateExtension(create) => {
                validate_bootstrap_create_extension(&create).map_err(ExecuteError::Engine)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::DropExtension(drop) => {
                validate_bootstrap_drop_extension(&drop).map_err(ExecuteError::Engine)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Begin => {
                self.commit_state_mut().txn_manager.begin_with_id(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                self.commit_state_mut().txn_manager.commit(txn_id)?;
                if chain {
                    self.commit_state_mut().txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                self.commit_state_mut().txn_manager.rollback(txn_id)?;
                if chain {
                    self.commit_state_mut().txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { key } => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.commit_state_mut().sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
            }
            Command::Select(_) | Command::SelectFunction(_) | Command::SequenceCurrVal(_) => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
        }
        Ok(())
    }

    pub fn tick_batching(&self, now: Instant) -> Result<(), EngineError> {
        if self.repl_role() != Role::Leader {
            if self.has_pending_batch() {
                return Err(EngineError::NotLeader);
            }
            return Ok(());
        }

        let due = self.batcher().maybe_flush_due_to_time(now);
        if let Some(batch) = due {
            self.apply_batch(batch.reason, batch.items.into_iter(), now)?;
        }
        Ok(())
    }

    pub fn flush_admin(&self) -> Result<(), EngineError> {
        if self.repl_role() != Role::Leader {
            return Err(EngineError::NotLeader);
        }

        let admin = self.batcher().flush_admin();
        if let Some(batch) = admin {
            self.apply_batch(batch.reason, batch.items.into_iter(), Instant::now())?;
        }
        Ok(())
    }

    fn apply_batch<I>(
        &self,
        reason: FlushReason,
        items: I,
        flushed_at: Instant,
    ) -> Result<(), EngineError>
    where
        I: Iterator<Item = BatchItem<PendingMutation>>,
    {
        let metric_reason = match reason {
            FlushReason::Count => BatchFlushReason::Count,
            FlushReason::Time => BatchFlushReason::Time,
            FlushReason::Admin => BatchFlushReason::Admin,
        };

        let mut remaining = items.peekable();
        while let Some(p) = remaining.next() {
            let wait = flushed_at
                .saturating_duration_since(p.enqueued_at)
                .as_millis() as u64;
            let txn_id = p.item.txn_id;
            let payload = p.item.payload.clone();

            // In no-GPU bootstrap mode, batched mutations represent the simulated
            // GPU-eligible write path. Track transfer and kernel timing envelopes
            // so telemetry contracts are stable before CUDA is wired in.
            self.metrics.observe_h2d_bytes(payload.len() as u64);
            let simulated_kernel_ms = ((payload.len() as u64) / 1024).max(1);
            self.metrics.observe_kernel_exec_ms(simulated_kernel_ms);
            let simulated_occupancy = Self::simulate_kernel_occupancy_permyriad(payload.len());
            self.metrics
                .observe_kernel_occupancy_permyriad(simulated_occupancy);

            if let Err(err) = self.commit_mutation(txn_id, payload) {
                let tail: Vec<_> = std::iter::once(p).chain(remaining).collect();
                self.batcher().requeue_front(tail);
                self.metrics.observe_pending_batch_len(self.batcher().len());
                return Err(err);
            }

            self.metrics.observe_batch_wait_ms(wait);
        }

        self.metrics.observe_pending_batch_len(self.batcher().len());
        self.metrics.inc_batch_flush(metric_reason);
        Ok(())
    }

    pub fn plan_text(&self, text: &str) -> Result<ExecutionPlan, ParseError> {
        let cmd = parse_command(text)?;
        Ok(self.planner.plan_command(&cmd))
    }

    /// Whether `text` is a DML statement (`INSERT`/`UPDATE`/`DELETE` on an existing base table whose
    /// columns carry no `nextval` sequence default) that the **concurrent** commit path can execute
    /// via off-lock prepare + the short commit critical section (write-half MVCC, Stage 4). Anything
    /// else — DDL, KV, sequence-default INSERTs, transaction control, parse errors, unknown tables —
    /// returns `false` and the caller routes it through the SERIALIZED `execute_text` under the
    /// catalog latch. Conservative by construction: it never returns `true` for a statement the
    /// concurrent path can't faithfully execute (a wrong "yes" only ever means a serialized fallback,
    /// never a wrong result — but here a wrong "yes" would mis-route, so the checks are exact).
    pub fn is_concurrent_dml(&self, text: &str) -> bool {
        let Ok(cmd) = parse_command(text) else {
            return false;
        };
        let table_name = match &cmd {
            Command::Insert(insert) => &insert.table,
            Command::Update(update) => &update.table,
            Command::Delete(delete) => &delete.table,
            _ => return false,
        };
        // Lock-free concurrent-DML classify (Stage 2 — blocker #1): probe the pinned catalog snapshot.
        let catalog = self.catalog_snapshot();
        let Some(table) = catalog.relational_catalog.get(table_name) else {
            return false;
        };
        // INSERTs that evaluate a `nextval` column default mutate sequence state, which is not
        // interior-mutable; route those through the serialized path (which applies the advance under
        // `&mut self`). UPDATE/DELETE never touch sequences, so they are always eligible.
        if matches!(cmd, Command::Insert(_)) {
            let touches_sequence_default = table.columns.iter().any(|column| {
                matches!(column.default, Some(ColumnDefault::SequenceNextVal { .. }))
            });
            if touches_sequence_default {
                return false;
            }
        }
        true
    }

    /// Register a transaction's read snapshot (its `read_snapshot` `commit_seq`) for the
    /// oldest-active GC/ledger-prune boundary, returning a guard that deregisters on drop (write-half
    /// MVCC, Stage 4). Done off-lock at prepare-begin so taking a snapshot never serializes on the
    /// commit_mutex.
    fn register_active_snapshot(&self, snapshot: Index) -> ActiveSnapshotGuard<'_> {
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .register(snapshot);
        ActiveSnapshotGuard {
            engine: self,
            snapshot,
        }
    }

    fn deregister_active_snapshot(&self, snapshot: Index) {
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .deregister(snapshot);
    }

    /// Execute one autocommit DML statement (`INSERT`/`UPDATE`/`DELETE`) on the CONCURRENT commit
    /// path under Snapshot Isolation (write-half MVCC, Stage 4 — the concurrency flip):
    ///
    /// 1. **Begin (off-lock):** pin a read snapshot `S = committed_seq` and register it.
    /// 2. **Prepare (off-lock, no commit_mutex):** parse, constraint-preflight against `S`, and
    ///    compute the conflict write-set (`prepare_*` at `S`). Many writers run this concurrently,
    ///    and concurrently with lock-free readers.
    /// 3. **Commit (short critical section under the commit_mutex):** validate the write-set against
    ///    the recent-commits ledger (overlap since `S` ⇒ retryable [`ExecuteError::Serialization`],
    ///    first-committer-wins) → assign `commit_seq` (the commit `Index`) → WAL append + group-commit
    ///    fsync → install the delta RE-RESOLVED at `commit_seq` (so the live apply is byte-identical
    ///    to a WAL replay) + publish the table generation → record the write-set in the ledger → bump
    ///    `committed_seq` LAST (release-store: the publish point).
    /// 4. **Abort/retry:** a conflict (or any prepare error) publishes nothing and is returned; a
    ///    serialization conflict is retryable with a fresh snapshot.
    ///
    /// `&self`: the whole path runs without an engine write lock, so writers overlap on prepare and
    /// serialize only briefly on the commit_mutex, and a writer never blocks a reader.
    pub fn execute_dml_concurrent(&self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        self.execute_dml_concurrent_instrumented(txn_id, text, || {})
    }

    /// [`Engine::execute_dml_concurrent`] with a hook invoked AFTER the off-lock snapshot capture +
    /// prepare but BEFORE the commit critical section. The concurrency-correctness suite uses this to
    /// rendezvous two writers at a barrier between snapshot and commit, deterministically forcing the
    /// SI write-write conflict window (both read the same snapshot, then both try to commit) — the
    /// lost-update exit criterion. The production entry point passes an empty hook, so this is a
    /// zero-overhead extraction of the real path, not a separate code path.
    pub fn execute_dml_concurrent_instrumented(
        &self,
        txn_id: u64,
        text: &str,
        on_prepared: impl FnOnce(),
    ) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        // (1) Begin: pin + register the read snapshot for the off-lock prepare.
        let read_snapshot = self.committed_seq();
        let _snapshot_guard = self.register_active_snapshot(read_snapshot);

        // (2) Prepare OFF-LOCK at the read snapshot: validate constraints + compute the conflict
        // write-set. (The delta itself is recomputed at commit_seq under the lock so the live apply
        // matches a WAL replay; this off-lock pass is the expensive validation + the write-set.)
        self.preflight_unique_index_constraints(&cmd, txn_id)
            .map_err(ExecuteError::Engine)?;
        let snapshot = self.dml_read_snapshot(read_snapshot);
        let prepared = self.prepare_dml(&cmd, snapshot)?;
        let residency_tables = Self::dml_mutated_tables(&cmd);

        // The snapshot is now pinned and prepare is done; the commit critical section has not started.
        // (Tests barrier here to align two writers' snapshots before their commits race.)
        on_prepared();

        // (3) Commit critical section under the commit_mutex.
        self.commit_dml_concurrent(
            txn_id,
            &cmd,
            text,
            prepared.write_set,
            read_snapshot,
            residency_tables,
        )
    }

    /// Off-lock prepare dispatch: run the pure `prepare_*` for a DML command against `snapshot`.
    fn prepare_dml(
        &self,
        cmd: &Command,
        snapshot: DmlReadSnapshot,
    ) -> Result<WriteDelta, ExecuteError> {
        let delta = match cmd {
            Command::Insert(insert) => self.prepare_insert(insert, snapshot, None),
            Command::Update(update) => self.prepare_update(update, snapshot),
            Command::Delete(delete) => self.prepare_delete(delete, snapshot),
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "execute_dml_concurrent received a non-DML command".to_string(),
                )))
            }
        }?;
        Ok(delta)
    }

    /// The set of tables a DML command mutates (for per-table residency invalidation on commit).
    fn dml_mutated_tables(cmd: &Command) -> BTreeSet<String> {
        let mut tables = BTreeSet::new();
        match cmd {
            Command::Insert(insert) => {
                tables.insert(insert.table.clone());
            }
            Command::Update(update) => {
                tables.insert(update.table.clone());
            }
            Command::Delete(delete) => {
                tables.insert(delete.table.clone());
            }
            _ => {}
        }
        tables
    }

    /// The short commit critical section (write-half MVCC, Stage 4). Holds the commit_mutex for:
    /// SI conflict validation → RE-RESOLVE/re-validate the delta at the peeked `commit_seq` → WAL
    /// append+fsync (`commit_seq` assignment) → delta install + table publish → ledger record →
    /// `committed_seq` release-store. Returns the retryable [`ExecuteError::Serialization`] on a
    /// first-committer-wins conflict OR on a re-resolve/constraint failure under a legal concurrent
    /// interleaving (a phantom absorbed since the snapshot, or a read-only FK parent a concurrent
    /// committer deleted) — in BOTH cases nothing was proposed/published/made durable and no
    /// commit-seq hole is left. The re-resolve runs the same `prepare_*` the serialized path uses
    /// (and `apply_delta` installs it), so the live apply is byte-identical to a WAL replay of the
    /// recorded SQL (the kill-mid-commit-under-concurrency invariant). The re-resolve happens BEFORE
    /// the WAL append/`propose`, so an abort never has anything durable to roll back.
    fn commit_dml_concurrent(
        &self,
        txn_id: u64,
        cmd: &Command,
        text: &str,
        write_set: WriteSet,
        read_snapshot: Index,
        residency_tables: BTreeSet<String>,
    ) -> Result<(), ExecuteError> {
        let timestamp_micros = self.next_commit_timestamp_micros();
        let payload = text.as_bytes().to_vec();

        // === enter the commit critical section ===
        let mut commit = self.commit_state();

        // (3a) Validate the prepared write-set against commits since the read snapshot. Any overlap
        // means a concurrent transaction committed a write to one of our keys after we snapshotted —
        // first-committer-wins aborts us (retryable). Nothing has been proposed/written yet, so the
        // abort is side-effect-free.
        if commit.ledger.conflicts(&write_set, read_snapshot) {
            return Err(ExecuteError::Serialization(format!(
                "write-write conflict on a key committed after read snapshot {read_snapshot}"
            )));
        }

        // (3b) PEEK the commit_seq this txn WILL be assigned, then RE-RESOLVE + re-validate the delta
        // at it — BEFORE anything durable (WAL/propose) happens. We hold the commit_mutex, so no other
        // committer can `propose` between this peek and ours, and our own delta is not installed yet;
        // therefore re-resolving at `committed_seq = next_index` now sees EXACTLY the state it would
        // see after `propose` but before `apply` (the highest existing version stamp is < commit_seq,
        // so resolving at `commit_seq` admits all currently-committed versions and none of our own).
        //
        // The re-prepare re-runs the FULL unique/CHECK/FK preflight + UPDATE/DELETE predicate
        // resolution against `commit_seq`. The off-lock prepare validated only against an OLDER
        // snapshot and the SI conflict check (3a) only covers keys in our WRITE-set; a phantom
        // committed in (snapshot, commit_seq] — e.g. an FK PARENT we merely READ then a concurrent
        // DELETE removed, or a row a re-resolve now absorbs into a unique/CHECK violation — can break
        // a constraint at commit_seq even though (3a) passed. That is a LEGAL concurrent interleaving,
        // not an invariant violation, so it is a RETRYABLE serialization abort: because we have not
        // proposed or appended to the WAL yet, the abort leaves NOTHING durable and NO commit-seq hole
        // (we never consumed the index). The caller retries against a fresh snapshot.
        let commit_seq = commit.repl.peek_next_index();
        let install_snapshot = self.dml_read_snapshot(commit_seq);
        let delta = self.prepare_dml(cmd, install_snapshot).map_err(|err| {
            // A re-prepare failure under a legal concurrent interleaving (phantom absorbed by the
            // re-resolve, or a read-only FK parent deleted by a concurrent committer). Surface as a
            // retryable Serialization abort rather than a panic — nothing was made durable.
            match err {
                ExecuteError::Serialization(_) => err,
                other => ExecuteError::Serialization(format!(
                    "re-resolve at commit_seq {commit_seq} failed on a concurrent interleaving \
                     (retryable): {other}"
                )),
            }
        })?;

        // (3c) Only NOW assign commit_seq for real (WAL append + propose + group-commit fsync). The
        // `propose` MUST return the index we peeked, since we hold the commit_mutex (single proposer).
        // The WAL-before-visibility invariant: the fsync completes before we publish or bump
        // committed_seq.
        let wal_len_before = commit.wal.len();
        commit.wal.append(WalRecord {
            txn_id,
            payload: payload.clone(),
        });
        let token = match commit.repl.propose(payload) {
            Ok(token) => token,
            Err(err) => {
                commit.wal.truncate(wal_len_before);
                return Err(ExecuteError::Engine(err));
            }
        };
        debug_assert_eq!(
            token.index, commit_seq,
            "commit_mutex is the single proposer: the proposed index must equal the peeked one"
        );
        let commit_seq = token.index;
        if let Err(err) = commit.wal.flush_all() {
            commit.repl.rollback_unapplied_from(commit_seq);
            commit.wal.truncate(wal_len_before);
            return Err(ExecuteError::Engine(err));
        }
        commit
            .repl
            .wait_committed(token, Duration::from_millis(0))?;
        commit
            .wal_commit_timestamps_micros
            .insert(txn_id, timestamp_micros);

        // (3d) Install the already-validated delta + publish the table generation. The delta was
        // re-resolved at exactly this `commit_seq` above, so this is a PURE install (reserve fresh
        // tuple ids, advance the row-id allocator, mutate the per-table version chains + value index
        // under the commit_mutex). The MvccData publish + the atomic allocators are `&self`; the
        // commit_mutex (held here) serializes installs so row-id assignment + the per-table publish
        // are atomic w.r.t. other committers. A failure HERE is unreachable on any legal interleaving
        // (the validation already succeeded at this seq and we hold the lock) AND the WAL record is
        // already durable, so it would be a true unrecoverable invariant violation — we PANIC, which
        // poisons the commit_mutex; the façade's re-homed poison-on-panic policy then refuses further
        // service rather than serve state inconsistent with the durable WAL (a restart replays the
        // WAL, the source of truth). Mark the entry applied so the replicator's applied_index tracks
        // the directly-applied commit (no later re-drain / re-apply).
        self.apply_delta(delta, commit_seq, None).unwrap_or_else(|err| {
            panic!(
                "commit-path invariant violation: apply at commit_seq {commit_seq} failed after the \
                 WAL was made durable, although re-validation at this seq succeeded: {err}"
            )
        });
        commit.repl.mark_applied(commit_seq);

        // (3e) Record OUR write-set in the ledger for future conflict detection, then prune entries
        // below the oldest active snapshot (the safe GC/ledger boundary).
        commit.ledger.record(&write_set, commit_seq);
        let prune_boundary = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .oldest()
            .map(|oldest| oldest.saturating_sub(1))
            .unwrap_or(commit_seq);
        commit.ledger.prune_below(prune_boundary);

        // Residency invalidation for the mutated tables, INSIDE the critical section so it is atomic
        // with the data publish (residency↔data consistency, design Risk #3): a reader that observes
        // the new committed_seq also sees the table's GPU residency invalidated.
        self.invalidate_relational_residency_tables_concurrent(
            &residency_tables,
            txn_id,
            commit_seq,
        );

        // (3f) Publish point: bump committed_seq LAST (release-store). Strictly after the WAL fsync
        // and the data/value-index publish, so an acquire-load by a reader observes a fully durable,
        // fully published commit.
        self.publish_committed_seq(commit_seq);
        self.metrics.inc_commit();
        drop(commit);
        // === leave the commit critical section ===
        Ok(())
    }

    pub fn execute_text(&self, txn_id: u64, text: &str) -> Result<(), ExecuteError> {
        let timestamp_micros = self.next_commit_timestamp_micros();
        self.execute_text_at_timestamp_micros(txn_id, text, timestamp_micros)
    }

    pub fn execute_text_at_timestamp_micros(
        &self,
        txn_id: u64,
        text: &str,
        timestamp_micros: u64,
    ) -> Result<(), ExecuteError> {
        let cmd = parse_command(text)?;

        match cmd {
            Command::SetKv { .. }
            | Command::DeleteKv { .. }
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
            | Command::CreateSequence(_)
            | Command::CreateDomain(_)
            | Command::SequenceNextVal(_)
            | Command::SequenceSetVal(_)
            | Command::RenameSequence(_)
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
            | Command::CreatePublication(_)
            | Command::DropPublication(_)
            | Command::CreateSubscription(_)
            | Command::DropSubscription(_)
            | Command::CreateRole(_)
            | Command::DropRole(_)
            | Command::RenameRole(_)
            | Command::GrantDefaultTablePrivileges(_)
            | Command::RevokeDefaultTablePrivileges(_)
            | Command::AlterColumnDefault(_)
            | Command::CommentOn(_)
            | Command::Insert(_)
            | Command::Delete(_)
            | Command::Update(_) => {
                self.preflight_unique_index_constraints(&cmd, txn_id)?;
                match self.route_command(&cmd) {
                    RouteDecision::Gpu(_) | RouteDecision::Cpu => {
                        self.commit_mutation_at(
                            txn_id,
                            text.as_bytes().to_vec(),
                            timestamp_micros,
                        )?;
                    }
                    RouteDecision::CpuFallback { reason, .. } => {
                        self.metrics.inc_gpu_fallback(reason);
                        self.commit_mutation_at(
                            txn_id,
                            text.as_bytes().to_vec(),
                            timestamp_micros,
                        )?;
                    }
                }
            }
            Command::Flush => {
                self.flush_admin()?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::ResetAll | Command::SetRole { .. } => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::CreateExtension(create) => {
                validate_bootstrap_create_extension(&create).map_err(ExecuteError::Engine)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::DropExtension(drop) => {
                validate_bootstrap_drop_extension(&drop).map_err(ExecuteError::Engine)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Begin => {
                self.commit_state().txn_manager.begin_with_id(txn_id)?;
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Commit { chain } => {
                self.commit_state().txn_manager.commit(txn_id)?;
                if chain {
                    self.commit_state().txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::Rollback { chain } => {
                self.commit_state().txn_manager.rollback(txn_id)?;
                if chain {
                    self.commit_state().txn_manager.begin()?;
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
            Command::GetKv { key } => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.commit_state().sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
            }
            Command::Select(_) | Command::SelectFunction(_) | Command::SequenceCurrVal(_) => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
            }
        }

        Ok(())
    }

    pub fn execute_read_text(&mut self, text: &str) -> Result<Option<String>, ExecuteError> {
        let cmd = parse_command(text)?;

        match cmd {
            Command::GetKv { key } => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }

                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.commit_state_mut().sm.kv.get(&key).map(|v| v.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
                Ok(self.get(&key))
            }
            Command::Begin => Err(ExecuteError::NonReadCommand("BEGIN")),
            Command::Commit { .. } => Err(ExecuteError::NonReadCommand("COMMIT")),
            Command::Rollback { .. } => Err(ExecuteError::NonReadCommand("ROLLBACK")),
            Command::Flush => Err(ExecuteError::NonReadCommand("FLUSH")),
            Command::ResetAll => Err(ExecuteError::NonReadCommand("RESET ALL")),
            Command::SetRole { .. } => Err(ExecuteError::NonReadCommand("SET ROLE")),
            Command::SetKv { .. } => Err(ExecuteError::NonReadCommand("SET")),
            Command::DeleteKv { .. } => Err(ExecuteError::NonReadCommand("DEL/DELETE")),
            Command::CreateSchema(_) => Err(ExecuteError::NonReadCommand("CREATE SCHEMA")),
            Command::DropSchema(_) => Err(ExecuteError::NonReadCommand("DROP SCHEMA")),
            Command::CreateDatabase(_) => Err(ExecuteError::NonReadCommand("CREATE DATABASE")),
            Command::DropDatabase(_) => Err(ExecuteError::NonReadCommand("DROP DATABASE")),
            Command::RenameDatabase(_) => Err(ExecuteError::NonReadCommand("ALTER DATABASE")),
            Command::CreateTablespace(_) => Err(ExecuteError::NonReadCommand("CREATE TABLESPACE")),
            Command::DropTablespace(_) => Err(ExecuteError::NonReadCommand("DROP TABLESPACE")),
            Command::RenameTablespace(_) => Err(ExecuteError::NonReadCommand("ALTER TABLESPACE")),
            Command::CreateTable(_) => Err(ExecuteError::NonReadCommand("CREATE TABLE")),
            Command::AddPrimaryKey(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddUniqueConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddCheckConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddForeignKey(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::AddColumn(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::RenameTable(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::RenameColumn(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::RenameConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::DropColumn(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::DropConstraint(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::CreateIndex(_) => Err(ExecuteError::NonReadCommand("CREATE INDEX")),
            Command::RenameIndex(_) => Err(ExecuteError::NonReadCommand("ALTER INDEX")),
            Command::CreateView(_) => Err(ExecuteError::NonReadCommand("CREATE VIEW")),
            Command::RenameView(_) => Err(ExecuteError::NonReadCommand("ALTER VIEW")),
            Command::CreateMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("CREATE MATERIALIZED VIEW"))
            }
            Command::CreateExtension(_) => Err(ExecuteError::NonReadCommand("CREATE EXTENSION")),
            Command::DropExtension(_) => Err(ExecuteError::NonReadCommand("DROP EXTENSION")),
            Command::RefreshMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("REFRESH MATERIALIZED VIEW"))
            }
            Command::RenameMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("ALTER MATERIALIZED VIEW"))
            }
            Command::CreateFunction(_) => Err(ExecuteError::NonReadCommand("CREATE FUNCTION")),
            Command::RenameFunction(_) => Err(ExecuteError::NonReadCommand("ALTER FUNCTION")),
            Command::DropFunction(_) => Err(ExecuteError::NonReadCommand("DROP FUNCTION")),
            Command::CreateSequence(_) => Err(ExecuteError::NonReadCommand("CREATE SEQUENCE")),
            Command::CreateDomain(_) => Err(ExecuteError::NonReadCommand("CREATE DOMAIN")),
            Command::SequenceNextVal(_) => Err(ExecuteError::NonReadCommand("SELECT nextval")),
            Command::SequenceSetVal(_) => Err(ExecuteError::NonReadCommand("SELECT setval")),
            Command::RenameSequence(_) => Err(ExecuteError::NonReadCommand("ALTER SEQUENCE")),
            Command::DropTable(_) => Err(ExecuteError::NonReadCommand("DROP TABLE")),
            Command::TruncateTable(_) => Err(ExecuteError::NonReadCommand("TRUNCATE TABLE")),
            Command::DropIndex(_) => Err(ExecuteError::NonReadCommand("DROP INDEX")),
            Command::DropView(_) => Err(ExecuteError::NonReadCommand("DROP VIEW")),
            Command::DropMaterializedView(_) => {
                Err(ExecuteError::NonReadCommand("DROP MATERIALIZED VIEW"))
            }
            Command::DropSequence(_) => Err(ExecuteError::NonReadCommand("DROP SEQUENCE")),
            Command::DropDomain(_) => Err(ExecuteError::NonReadCommand("DROP DOMAIN")),
            Command::GrantTable(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeTable(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantSchema(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeSchema(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantDatabase(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeDatabase(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantTablespace(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeTablespace(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::GrantFunction(_) => Err(ExecuteError::NonReadCommand("GRANT")),
            Command::RevokeFunction(_) => Err(ExecuteError::NonReadCommand("REVOKE")),
            Command::CreatePublication(_) => {
                Err(ExecuteError::NonReadCommand("CREATE PUBLICATION"))
            }
            Command::DropPublication(_) => Err(ExecuteError::NonReadCommand("DROP PUBLICATION")),
            Command::CreateSubscription(_) => {
                Err(ExecuteError::NonReadCommand("CREATE SUBSCRIPTION"))
            }
            Command::DropSubscription(_) => Err(ExecuteError::NonReadCommand("DROP SUBSCRIPTION")),
            Command::CreateRole(_) => Err(ExecuteError::NonReadCommand("CREATE ROLE")),
            Command::DropRole(_) => Err(ExecuteError::NonReadCommand("DROP ROLE")),
            Command::RenameRole(_) => Err(ExecuteError::NonReadCommand("ALTER ROLE")),
            Command::GrantDefaultTablePrivileges(_) => {
                Err(ExecuteError::NonReadCommand("ALTER DEFAULT PRIVILEGES"))
            }
            Command::RevokeDefaultTablePrivileges(_) => {
                Err(ExecuteError::NonReadCommand("ALTER DEFAULT PRIVILEGES"))
            }
            Command::AlterColumnDefault(_) => Err(ExecuteError::NonReadCommand("ALTER TABLE")),
            Command::CommentOn(_) => Err(ExecuteError::NonReadCommand("COMMENT")),
            Command::Insert(_) => Err(ExecuteError::NonReadCommand("INSERT")),
            Command::Delete(_) => Err(ExecuteError::NonReadCommand("DELETE")),
            Command::Update(_) => Err(ExecuteError::NonReadCommand("UPDATE")),
            Command::Select(_) | Command::SelectFunction(_) | Command::SequenceCurrVal(_) => {
                Err(ExecuteError::NonReadCommand("SELECT"))
            }
        }
    }

    /// Pin one statement-stable relational read snapshot at the ALREADY-CHOSEN boundary `s` (PART B
    /// catalog↔data co-pinning). The boundary `s` was loaded ONCE per statement (by
    /// [`Engine::bind_relational_select_for_execution`], which also selected the catalog as-of `s`), so
    /// the data this pins and the catalog the statement bound against are the SAME generation — a
    /// concurrent shape-changing DDL can never split the reader's (catalog, data) pair. Pins ONE
    /// generation of `table` (its rows + value-index together) at `s`.
    fn pin_relational_read_at(&self, table: &str, s: Index) -> RelationalReadPin {
        RelationalReadPin {
            visibility: StorageVisibility { read_txn_id: s },
            table_rows: self.read_state.mvcc.table_rows(table),
        }
    }
}

#[cfg(test)]
mod tests;
