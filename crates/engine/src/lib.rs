use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;
use gpu_db_batching::{BatchItem, DualTriggerBatcher, FlushReason};
use gpu_db_execution::{
    CudaDeviceMemoryChunk, CudaDeviceMemoryProof, CudaDriverRuntime, CudaI32BatchProjectionColumns,
    CudaI32Comparison, CudaI32EqualAnyProjectSubmission, CudaI32IndexProbeDenseSubmission,
    CudaI32Stats, CudaMvccRowBatch, CudaOwnedDeviceMemoryChunk, CudaResidentDeviceMemory,
    CudaResidentDeviceMemoryReadView, DeviceRouter, DeviceTarget, ExprStep, FilterOperator,
    LimitOperator, MockGpuRuntime, Operator, PlannedOp, ProjectOperator, ResidentElemType,
    RouteDecision, ScanOperator, SortOperator, WriteLocateShard,
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
    DropTable, DropTablespace, DropView, FunctionPrivilege, GroupedAggKind, GroupedAggregate,
    Insert, ParseError, PublicationTarget, RefreshMaterializedView, RenameColumn, RenameConstraint,
    RenameDatabase, RenameFunction, RenameIndex, RenameMaterializedView, RenameRole,
    RenameSequence, RenameTable, RenameTablespace, RenameView, SchemaPrivilege, Select,
    SelectFilterOp, SelectFunction, SelectProjection, SequenceNextVal, SequenceSetVal, SqlType,
    SqlValue, TablePrivilege, TablespacePrivilege, TruncateTable, Update,
    NUMERIC_DEFAULT_PRECISION,
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
    read_wal_segment, recover_wal_segment, register_wal_archive_timeline,
    restore_wal_archive_object_backup, select_wal_archive_timeline, write_wal_archive_timeline,
    write_wal_archive_with_timestamps, write_wal_control_file, write_wal_segment,
    WalArchiveManifest, WalArchiveObjectBackup, WalArchiveRecordTimestamp, WalArchiveRetentionPlan,
    WalArchiveTimeline, WalArchiveTimelineBranch, WalArchiveTimelinePrunePlan,
    WalArchiveTimelineRegistry, WalArchiveTimelineSelection, WalBuffer, WalCheckpointMeta,
    WalControlFile, WalGroupCommitStats, WalRecord,
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
mod engine_commit;
mod engine_ddl_acl;
mod engine_ddl_alter;
mod engine_ddl_objects;
mod engine_ddl_pubsub_role;
mod engine_ddl_table;
mod engine_dml_concurrent;
mod engine_dml_prepare;
pub(crate) use engine_dml_prepare::InsertPrepareValidation;
mod engine_expr;
mod engine_introspection;
mod engine_lifecycle;
mod engine_mvcc_dispatch;
mod engine_residency;
mod engine_resident_probe;
mod engine_retained_read;
mod engine_select_bind;
mod engine_select_exec;
mod engine_sql_pg;
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
                    || (msg.contains(RESIDENT_SHARD_PREFIX)
                        && (msg.contains(RESIDENT_SHARD_INVALID)
                            || msg.contains(RESIDENT_SHARD_MEMORY_MISSING)))
        )
    }
}

/// The substring every GPU resident-route probe uses when a table's device-memory cell is `None`
/// (tombstoned/never-populated). Used to detect the residency-invalidated-mid-statement case so the
/// read can fall back to the CPU pinned-read path (write-half MVCC, Stage 4).
const RESIDENT_DEVICE_MEMORY_MISSING: &str = "has no retained resident device memory";

/// W0c (audit B2): the SHARDED unified-source errors (`build_sharded_unified_exec_source`'s
/// `source_for`: "resident shard {id} is invalid" / "resident shard {id} has no retained device
/// memory") are the same residency-invalidated-mid-statement case in per-shard form — the route
/// plan accepted an earlier generation and a concurrent commit flagged the shards before the
/// executor's own load. W0 made that window COMMON under OLTP write load (every concurrent
/// invalidation now flags descriptors), and without these matches a racing reader got a hard
/// client error where the transparent CPU pinned-read fallback is the correct behavior. Matched
/// as (prefix AND suffix) so a genuine device/CUDA error is never masked.
const RESIDENT_SHARD_PREFIX: &str = "resident shard ";
const RESIDENT_SHARD_INVALID: &str = " is invalid";
const RESIDENT_SHARD_MEMORY_MISSING: &str = " has no retained device memory";

#[derive(Debug, Clone)]
struct PendingMutation {
    txn_id: u64,
    payload: Vec<u8>,
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
}

impl Default for GroupFlushState {
    fn default() -> Self {
        Self {
            coord: Mutex::new(GroupFlushCoord::default()),
            cv: std::sync::Condvar::new(),
            durable_records: std::sync::atomic::AtomicUsize::new(0),
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
    /// it for its short critical section (validate → assign `commit_seq` → WAL fsync → publish), so
    /// commits serialize ONLY here while prepare runs off-lock and readers stay lock-free. Code that
    /// already holds `&mut self` (serialized DDL apply, recovery, checkpoint/snapshot admin) reaches
    /// it via `commit_state_mut()` (a zero-cost `Mutex::get_mut`, no actual locking).
    commit: Mutex<CommitState>,
    /// In-flight transactions' read snapshots (write-half MVCC, Stage 4), for the oldest-active GC
    /// boundary. Separate from `commit` so a transaction can register its snapshot at prepare-begin
    /// WITHOUT serializing on the commit_mutex (prepare is off-lock).
    active_snapshots: Mutex<ActiveSnapshots>,
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
    /// STRATA S-B: when true, a committing mutation auto-admits its tables to GPU residency after the
    /// commit publishes (best-effort, never fails the commit). Default off. Interior-mutable (`&self`).
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
    /// >=b4096, audit SHIP) — set false to A/B against the atomic kernel. Only the unique index route honors
    /// it — the non-unique scan always keeps the atomic kernel. Interior-mutable.
    dense_index_probe_enabled: AtomicBool,
    /// Billions-of-rows scaling (segmented layout, S-d1): when true, a table is admitted as a SEGMENTED
    /// shard list (sealed shards + one bounded open shard) routed through the sharded resident read path,
    /// instead of one capacity-padded unified buffer that caps at ~536M rows and re-admits O(table). DEFAULT
    /// OFF — production stays on the single buffer until seal/rollover (S-d2) + per-shard bloom index (S-d3)
    /// make the shard path strictly better at scale; this flag is the A/B lever to validate it. Interior-mutable.
    shard_residency_enabled: AtomicBool,
    /// SV4b: route a single-entry DELETE commit through the GPU-native tombstone (locate + stamp `deleted_by`
    /// in place, O(rows)) instead of the O(table) invalidate + re-admit. DEFAULT OFF, nested under the shard
    /// path (a DELETE re-admits exactly as before until this flips) — the independent A/B lever for the
    /// incremental-DELETE win. Interior-mutable (read on the commit path).
    resident_delete_tombstone_enabled: AtomicBool,
    /// SV5: route a single-entry UPDATE commit through the GPU-native tombstone-old + append-new (O(rows))
    /// instead of the O(table) invalidate + re-admit. DEFAULT OFF, nested under the shard path. Interior-mutable.
    resident_update_tombstone_enabled: AtomicBool,
    /// Sub-slice 3b: route a shard-resident int4 UNIQUE-key equality POINT lookup through the CROSS-SHARD PK
    /// INDEX (cached hash+bloom `locate`) so the sharded read gathers ONLY the located shard(s) instead of
    /// every zone-map-non-excluded shard. DEFAULT OFF, nested under `shard_residency_enabled` (the sharded read
    /// falls back to the existing zone-map scan + recompaction until this flips — byte-identical). The A/B lever
    /// for the membership-pruning win that lets the shard path stay O(1) even when zone-maps degrade under
    /// UPDATE key-scatter (scalability-ledger #4/#8). Interior-mutable (the read path reads it).
    shard_index_probe_enabled: AtomicBool,
    /// lpb-for-shards wiring: admit a shard-resident int4 point-lookup BATCH into the facade point-lookup
    /// batcher and serve it via the batched cross-shard gather (`submit_sharded_point_lookups_batched`)
    /// instead of degrading to per-query single-flight. DEFAULT OFF (nested under shard residency): OFF =>
    /// `submit_sharded_point_lookups_batched` returns `None` and the batcher keeps its existing behavior
    /// (byte-identical). The A/B lever that LANDS the ~310x batched throughput on real workloads. Interior-mutable.
    shard_batched_point_read_enabled: AtomicBool,
    /// S-d2c: target row count per shard. When the open shard reaches it, an append SEALS the open shard
    /// (immutable) and ROLLS OVER to a fresh open shard (O(rows), never the O(table) re-admit), so a table
    /// grows as bounded shards to billions of rows. Caps the admit headroom + sizes a rollover shard.
    /// Default 4M (seals in ~3ms, ~250 shards/1B per the admit-scaling measurement); settable small in
    /// tests. Interior-mutable.
    /// PHASE C slice 1: DELETE/UPDATE prepare resolves matches via the per-table equality value
    /// index (O(matches)). Kill switch -> the O(table) seq_scan (the differential oracle).
    dml_value_index_resolve_enabled: std::sync::atomic::AtomicBool,
    /// RETIREMENT A2: single-Eq DML resolves via the DEVICE (locate -> row-identity region ->
    /// derived key), not the host value index — the dependency A4 deletes. Kill switch -> the
    /// value-index resolve (slice 1), then the scan.
    dml_device_resolve_enabled: std::sync::atomic::AtomicBool,
    dml_device_validate_enabled: std::sync::atomic::AtomicBool,
    host_install_elision_enabled: std::sync::atomic::AtomicBool,
    /// TYPE-COVERAGE track 1: UNIQUE-INDEXED (PK'd) i32-section tables may ELIDE — the
    /// core-banking table shape. Requires the resolve+validate ladders ON (eligibility checks
    /// them). DEFAULT ON (the 2026-07-03 flip). Kill switch -> unique tables never enter
    /// elision (already-elided tables rehydrate through the ladder seams as usual).
    constrained_elision_enabled: std::sync::atomic::AtomicBool,
    /// TYPE-COVERAGE track 2 slice 2: sharded admission includes i64-SECTION columns
    /// (Int8/Timestamp) alongside the i32 sections — the first non-i32 shard section.
    /// DEFAULT ON (the 2026-07-03 flip). Kill switch -> int8-bearing tables admit
    /// single-buffer (the pre-slice layout).
    shard_int8_section_enabled: std::sync::atomic::AtomicBool,
    /// M1 (charter ruling 2026-07-03): the write path's PK locate (A2 resolve, A3 validators)
    /// probes the per-shard DEVICE hash index via a kernel instead of the host `shard_pk_index`
    /// hash cache — key->slot ADDRESSING is device work. Default OFF until the SLO gate + audit;
    /// the host cache stays as the flag-off oracle until M3 deletes it. Kill switch -> host probe.
    device_write_locate_enabled: std::sync::atomic::AtomicBool,
    /// M1 design B (wave-time batched validation): eligible INSERTs' PK-unique check is DEFERRED
    /// from the off-lock prepare to the wave sequencer, which batches the whole wave's PK needles
    /// into ONE device locate (the amortization win: launch cost is flat vs batch size). Default
    /// OFF; requires device_write_locate_enabled. Kill switch -> per-item off-lock validation.
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

impl CommitState {
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

