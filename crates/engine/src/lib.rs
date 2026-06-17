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
mod engine_ddl_acl;
mod engine_ddl_alter;
mod engine_ddl_pubsub_role;
mod engine_ddl_table;
mod engine_dml_concurrent;
mod engine_dml_prepare;
mod engine_introspection;
mod engine_mvcc_dispatch;
mod engine_residency;
mod engine_resident_probe;
mod engine_retained_read;
mod engine_select_bind;
mod engine_select_exec;
mod engine_wal_archive;
mod engine_write_apply;

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
}

#[cfg(test)]
mod tests;
