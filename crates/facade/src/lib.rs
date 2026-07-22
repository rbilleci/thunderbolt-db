//! Protocol-neutral engine façade (prototype→production plan, Phase 0 / P0-M1).
//!
//! This crate is the single seam through which any client protocol reaches the
//! `Engine`. It exists to keep the engine **protocol-neutral**: the façade speaks
//! only engine-native concepts — a [`SharedSession`], a SQL command string, neutral
//! typed result rows ([`DbValue`] / [`ColumnMeta`]), and neutral errors
//! ([`DbError`]). It never exposes wire type OIDs, `SQLSTATE` codes, the
//! PostgreSQL extended-query message lifecycle, or `pg_catalog` shapes — those
//! belong in a per-protocol *adapter* (see [`pg_adapter`] for the first one).
//!
//! Design principle (plan §5.0): pgwire is the first adapter over this façade,
//! not fused into the engine. A MySQL or HTTP/WebSocket adapter would map the
//! same neutral types to a different wire representation without the engine
//! learning that those protocols exist.
//!
//! ## Known transitional shape
//!
//! - The neutral value vocabulary ([`DbValue`], [`LogicalType`]) mirrors the
//!   SQL vocabulary's `SqlValue`/`SqlType` (now in the lower `gpu_db_sql` crate)
//!   and is converted at the boundary so those engine-facing types do not leak.
//!   The `engine -> protocol` dependency has been inverted: both the engine and
//!   this façade depend on `gpu_db_sql`, so the parsed types are identity-equal
//!   across the boundary (roadmap §9.2).
//! - Result-bearing DML preserves affected-row counts and `RETURNING`; generic commands whose
//!   current engine strategy has no count continue to return `None`.
//! - Transaction control (`BEGIN`/`COMMIT`/`ROLLBACK`) owns one engine transaction identity and one
//!   generation-owned read snapshot for the session lifetime. SELECT consumes that retained catalog,
//!   MVCC, and GPU-residency generation. DML stages into a transaction-private GPU generation;
//!   COMMIT publishes its resolved row mutations through one durable record and one commit index.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use gpu_db_engine::{
    CopyMutationRequest, CopyTargetProof, Engine, ExecuteError, MutationRequest, RelationalColumn,
    TransactionAdmissionResult, TransactionCopyTargetOrigin,
};
use gpu_db_sql::{
    parse_command, Command, CopyFromStdin, CopyToStdout, Decimal128, ParseError, ParsedCommand,
    Select, SelectProjection, SqlType, SqlValue, TransactionCharacteristics, TransactionIsolation,
};

#[cfg(test)]
mod mutation_admission_tests;
pub mod pg_adapter;
mod point_lookup_batcher;
mod prepared;

pub use point_lookup_batcher::{PointLookupBatcher, PointLookupBatcherActivitySnapshot};
pub use prepared::{BoundPreparedStatement, PreparedStatement};

/// Neutral logical column type. Carries no wire OID; adapters derive the wire
/// type from this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalType {
    Int4,
    Int8,
    Numeric,
    Bool,
    Text,
    Date,
    Timestamp,
    Uuid,
    Int2,
}

/// Neutral value vocabulary. Owned by the façade so the protocol crate's
/// `SqlValue` does not cross the boundary. `Numeric` carries the engine's
/// fixed-point [`Decimal128`]; the wire adapter renders it to text (the binary
/// numeric wire codec is a later milestone).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbValue {
    /// SQL `NULL` — the typeless absence of a value. The wire adapter maps it to the
    /// protocol's `-1` field length via [`pg_adapter::db_value_text_opt`]. Mirrors
    /// [`gpu_db_sql::SqlValue::Null`].
    Null,
    Int4(i32),
    Int8(i64),
    Numeric(Decimal128),
    Bool(bool),
    Text(String),
    /// A `date` as i32 days since 2000-01-01 (PostgreSQL's date epoch).
    Date(i32),
    /// A `timestamp` as i64 microseconds since 2000-01-01 00:00:00.
    Timestamp(i64),
    /// A `uuid` as its 16 raw bytes.
    Uuid([u8; 16]),
    /// A `smallint` (int2) as i16.
    Int2(i16),
}

/// Neutral column metadata: a name and a logical type. No OID, no typmod.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnMeta {
    pub name: String,
    pub logical_type: LogicalType,
}

/// Neutral COPY input metadata. Numeric typmod is retained because COPY text decoding must round
/// at the table's declared scale before mutation admission; non-numeric columns carry `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyColumnMeta {
    pub name: String,
    pub logical_type: LogicalType,
    pub numeric_typmod: Option<(u8, u8)>,
}

/// Opaque, protocol-neutral COPY FROM target retained from description through final admission.
///
/// Wire adapters may inspect only the input columns.  The normalized SQL target and exact engine
/// relation proof remain facade-owned so CopyDone cannot be redirected to a DROP/recreated table.
#[derive(Debug, Clone)]
pub struct CopyTarget {
    copy: CopyFromStdin,
    columns: Vec<CopyColumnMeta>,
    proof: CopyTargetProof,
    engine_identity: Arc<()>,
    transaction_identity: Option<u64>,
}

impl CopyTarget {
    pub fn columns(&self) -> &[CopyColumnMeta] {
        &self.columns
    }
}

impl PartialEq for CopyTarget {
    fn eq(&self, other: &Self) -> bool {
        self.copy == other.copy
            && self.columns == other.columns
            && self.proof == other.proof
            && Arc::ptr_eq(&self.engine_identity, &other.engine_identity)
            && self.transaction_identity == other.transaction_identity
    }
}

impl Eq for CopyTarget {}

/// Neutral command classification. Adapters format the protocol-specific
/// completion tag (e.g. the PostgreSQL `INSERT 0 N`) from this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandTag {
    Begin,
    Commit,
    Rollback,
    CreateTable,
    CreateIndex,
    Insert,
    Update,
    Delete,
    Truncate,
    Copy,
    Other(String),
}

/// Neutral query result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryOutcome {
    Rows {
        columns: Vec<ColumnMeta>,
        rows: Vec<Vec<DbValue>>,
    },
    Returning {
        tag: CommandTag,
        columns: Vec<ColumnMeta>,
        rows: Vec<Vec<DbValue>>,
        rows_affected: u64,
    },
    Command {
        tag: CommandTag,
        rows_affected: Option<u64>,
    },
    /// COPY FROM STDIN was validated and needs typed wire rows. The neutral columns are in COPY
    /// target order and carry no PostgreSQL format or framing state.
    CopyIn { target: CopyTarget },
    /// The statement was empty (e.g. `""`, `";"`, only whitespace/comments). Wire
    /// adapters must reply with their empty-query signal (PostgreSQL's
    /// `EmptyQueryResponse`), not a syntax error.
    Empty,
}

/// Neutral error category. Adapters map this to their protocol's error code
/// taxonomy (`SQLSTATE`, MySQL error numbers, HTTP status, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Syntax,
    Unsupported,
    UndefinedRelation,
    UndefinedColumn,
    DuplicateColumn,
    IndeterminateDatatype,
    DatatypeMismatch,
    InvalidRequest,
    /// The engine refused a bounded pre-effect admission because its foreground resource or
    /// queue-time credits were unavailable. PostgreSQL adapters map this to class 53.
    ResourceExhausted,
    /// The protocol owner cancelled active effect-free work before admission or before exposing
    /// its result. A successful mutation is never rewritten into this category after publication.
    Cancelled,
    InFailedTransaction,
    UniqueViolation,
    Engine,
    Internal,
    /// A retryable Snapshot-Isolation write-write serialization conflict (write-half MVCC, Stage 4):
    /// a concurrent transaction committed a write to a key in this transaction's write-set after it
    /// took its read snapshot, so first-committer-wins aborted this one. The transaction made no
    /// durable or visible change; the client may retry it. Adapters map this to a retryable class-40
    /// code (PostgreSQL `40001` serialization_failure).
    Serialization,
}

/// Neutral error. Carries a category and a human message — never a wire code.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{category:?}: {message}")]
pub struct DbError {
    pub category: ErrorCategory,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionTransactionStatus {
    Idle,
    InTransaction,
    FailedTransaction,
}
/// A `Send + Sync` engine wrapper for concurrent dispatch (lock-free read path, write-half MVCC —
/// the destination milestone): one engine shared across a server's worker pool behind a plain
/// `Arc<Engine>`, with **NO façade lock**. The engine is now fully interior-mutable for the
/// concurrency-relevant paths (`&self` reads, `&self` concurrent-DML commit under the engine's own
/// `commit_mutex`, `&self` DDL under the engine's own `catalog_latch`), so reads run truly lock-free
/// and a writer never blocks a reader. The server holds an `Arc<SharedEngine>` and reaches the engine
/// only through this neutral type, never naming `Engine` directly (keeps the §5.0 protocol-neutral
/// boundary intact).
pub struct SharedEngine {
    engine: Arc<Engine>,
    identity: Arc<()>,
    next_txn_id: AtomicU64,
}

/// One protocol-neutral request admitted through [`SharedEngine::submit`].
///
/// The variants select representation and read scheduling only; they are not independent
/// execution authorities. Any mutation or transaction control carried by `Text` or `Prepared`,
/// plus disconnect rollback carried by `CloseSession`, reaches the same engine transaction
/// admission boundary below `submit`.
pub enum SubmissionRequest<'a> {
    /// Parse and execute one SQL command.
    Text(&'a str),
    /// Execute one already-bound, opaque prepared AST without reparsing SQL text.
    Prepared(&'a BoundPreparedStatement),
    /// Parse once, batch an eligible autocommit point read, and otherwise execute immediately.
    BatchedText {
        sql: &'a str,
        batcher: &'a PointLookupBatcher,
    },
    /// Validate a COPY FROM STDIN target and return its neutral typed input columns.
    CopyFromStart(&'a CopyFromStdin),
    /// Admit decoded COPY FROM STDIN rows through the same transaction boundary as ordinary DML.
    CopyFrom {
        target: &'a CopyTarget,
        rows: Vec<Vec<DbValue>>,
    },
    /// Read a COPY TO STDOUT table from the session's exact GPU-native snapshot.
    CopyTo(&'a CopyToStdout),
    /// Execute an extended COPY TO through its retained bound prepared owner while preserving
    /// COPY-specific column validation from the original statement.
    PreparedCopyTo {
        copy: &'a CopyToStdout,
        bound: &'a BoundPreparedStatement,
    },
    /// Deterministic test seam for rendezvousing after concurrent DML preparation.
    #[doc(hidden)]
    InstrumentedDml {
        sql: &'a str,
        on_prepared: Box<dyn FnOnce() + Send + 'static>,
    },
    /// Deterministic test seam for rendezvousing after a SELECT pins its snapshot.
    #[doc(hidden)]
    InstrumentedSelect {
        sql: &'a str,
        on_pinned: Box<dyn FnOnce() + 'a>,
    },
    /// End a client session, atomically rolling back any still-active transaction.
    CloseSession,
}

/// Result of [`SharedEngine::submit`]. Mutations, transaction control, prepared statements, and
/// session close always resolve immediately. Only an eligible read-only `BatchedText` request can
/// return a receiver.
pub enum SubmissionDispatch {
    Immediate(Result<QueryOutcome, DbError>),
    Batched(tokio::sync::oneshot::Receiver<Result<QueryOutcome, DbError>>),
}

impl SubmissionDispatch {
    /// Extract a synchronously resolved result. `Text`, `Prepared`, and `CloseSession` always use
    /// this arm; callers that submit `BatchedText` must match both variants instead.
    pub fn into_immediate(self) -> Result<QueryOutcome, DbError> {
        match self {
            Self::Immediate(result) => result,
            Self::Batched(_) => Err(DbError {
                category: ErrorCategory::Internal,
                message: "batched read submission cannot be consumed as an immediate result"
                    .to_string(),
            }),
        }
    }
}

/// Per-client transaction ownership for the lock-free shared-engine façade. The value is cheap,
/// connection-local, and `Send`; the engine keeps the authoritative transaction state and the
/// keyed snapshot hold. It deliberately contains no relational data or CPU execution state.
#[derive(Debug)]
pub struct SharedSession {
    engine_identity: Arc<()>,
    active_txn_id: Option<u64>,
    transaction_characteristics: Option<TransactionCharacteristics>,
    transaction_has_statement: bool,
    transaction_failed: bool,
}

impl SharedSession {
    pub fn in_transaction(&self) -> bool {
        self.active_txn_id.is_some()
    }

    pub fn transaction_status(&self) -> SessionTransactionStatus {
        match (self.active_txn_id, self.transaction_failed) {
            (Some(_), true) => SessionTransactionStatus::FailedTransaction,
            (Some(_), false) => SessionTransactionStatus::InTransaction,
            (None, _) => SessionTransactionStatus::Idle,
        }
    }

    pub fn mark_transaction_failed(&mut self) {
        self.transaction_failed |= self.active_txn_id.is_some();
    }

    /// Active catalog-description owner. Failed transactions cannot acquire a new statement
    /// snapshot; protocol adapters may emit an already-cached rowless description, while every
    /// facade description/revalidation request still enforces failed-transaction precedence.
    pub(crate) fn description_txn_id(&self) -> Option<u64> {
        (!self.transaction_failed)
            .then_some(self.active_txn_id)
            .flatten()
    }
}

/// Protocol-neutral non-vacuity counters for mixed GPU-native workloads. These are monotonic
/// process-local observations, intended for operational diagnostics and benchmark gates: result
/// equality alone cannot prove whether a resident read silently fell back or whether a write
/// published device authority. `resident_shards` is scoped to the table passed to
/// [`SharedEngine::gpu_native_activity_snapshot`]; every other field is engine-global.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuNativeActivitySnapshot {
    pub valid_resident_tables: usize,
    pub resident_shards: usize,
    pub dense_index_probe_batches: u64,
    pub sharded_point_batches: u64,
    pub sharded_gpu_probe_batches: u64,
    pub sharded_binary_route_batches: u64,
    pub open_shard_append_commits: u64,
    pub device_authoritative_commits: u64,
}

impl SharedEngine {
    /// Construct a shared façade over a local single-node engine.
    pub fn new() -> Self {
        Self::from_engine(Engine::new_local())
    }

    /// Wrap an already-built engine — e.g. one pre-seeded and warmed to GPU residency before
    /// serving, so the served read path takes the resident route (used by the GPU-retained
    /// benchmark).
    pub fn from_engine(engine: Engine) -> Self {
        Self::from_engine_arc(Arc::new(engine))
    }

    /// Internal constructor for tests that need read-only instrumentation from the same engine
    /// while driving all SQL through the canonical facade boundary.
    pub(crate) fn from_engine_arc(engine: Arc<Engine>) -> Self {
        let next_txn_id = engine.next_durable_transaction_id_floor();
        Self {
            engine,
            identity: Arc::new(()),
            next_txn_id: AtomicU64::new(next_txn_id),
        }
    }

    /// Open a connection/request session for transaction ownership. Session identity is local to
    /// the façade; an engine transaction identity is allocated only when the client sends BEGIN.
    pub fn open_session(&self) -> SharedSession {
        SharedSession {
            engine_identity: Arc::clone(&self.identity),
            active_txn_id: None,
            transaction_characteristics: None,
            transaction_has_statement: false,
            transaction_failed: false,
        }
    }

    /// Submit every statement/session-ending action through the facade's sole public execution
    /// boundary. Request variants preserve typed prepared execution and optional point-read
    /// batching without creating another mutation or transaction authority.
    pub fn submit(
        &self,
        session: &mut SharedSession,
        request: SubmissionRequest<'_>,
    ) -> SubmissionDispatch {
        if let Err(error) = self.ensure_session_owner(session) {
            return SubmissionDispatch::Immediate(Err(error));
        }
        match request {
            SubmissionRequest::Text(sql) => {
                SubmissionDispatch::Immediate(submit_text_inner(self, session, sql))
            }
            SubmissionRequest::Prepared(bound) => {
                SubmissionDispatch::Immediate(prepared::submit_prepared_inner(self, session, bound))
            }
            SubmissionRequest::BatchedText { sql, batcher } => {
                submit_batched_text_inner(self, session, batcher, sql)
            }
            SubmissionRequest::CopyFromStart(copy) => {
                SubmissionDispatch::Immediate(submit_copy_from_start(self, session, copy))
            }
            SubmissionRequest::CopyFrom { target, rows } => {
                SubmissionDispatch::Immediate(submit_copy_from(self, session, target, rows))
            }
            SubmissionRequest::CopyTo(copy) => {
                SubmissionDispatch::Immediate(submit_copy_to(self, session, copy))
            }
            SubmissionRequest::PreparedCopyTo { copy, bound } => {
                SubmissionDispatch::Immediate(submit_prepared_copy_to(self, session, copy, bound))
            }
            SubmissionRequest::InstrumentedDml { sql, on_prepared } => {
                SubmissionDispatch::Immediate(
                    submit_instrumented_dml(self, session, sql, on_prepared)
                        .map(|()| QueryOutcome::Empty),
                )
            }
            SubmissionRequest::InstrumentedSelect { sql, on_pinned } => {
                SubmissionDispatch::Immediate(submit_instrumented_select(
                    self, session, sql, on_pinned,
                ))
            }
            SubmissionRequest::CloseSession => {
                SubmissionDispatch::Immediate(self.close_session_inner(session))
            }
        }
    }

    pub(crate) fn ensure_session_owner(&self, session: &SharedSession) -> Result<(), DbError> {
        if Arc::ptr_eq(&self.identity, &session.engine_identity) {
            Ok(())
        } else {
            Err(DbError {
                category: ErrorCategory::InvalidRequest,
                message: "session belongs to a different SharedEngine".to_string(),
            })
        }
    }

    /// Roll back an active session during canonical `CloseSession` submission.
    fn close_session_inner(&self, session: &mut SharedSession) -> Result<QueryOutcome, DbError> {
        if let Some(txn_id) = session.active_txn_id {
            let rollback =
                ParsedCommand::parse("ROLLBACK").expect("static rollback command must parse");
            let result = self
                .engine
                .submit_transaction(txn_id, rollback)
                .map_err(map_execute_error)?;
            let TransactionAdmissionResult::Transaction(None) = result else {
                return Err(invalid_mutation_result("session close ROLLBACK"));
            };
            session.active_txn_id = None;
            session.transaction_characteristics = None;
            session.transaction_has_statement = false;
        }
        session.transaction_failed = false;
        Ok(QueryOutcome::Empty)
    }

    fn take_txn_id(&self) -> u64 {
        self.next_txn_id.fetch_add(1, Ordering::Relaxed)
    }

    /// A shared façade over a **crash-durable** engine: opens (recovering) or creates the WAL
    /// segment at `segment_path`, and fsyncs every commit's WAL record before it becomes
    /// visible. The durability counterpart to [`SharedEngine::new`], whose WAL is in-memory only.
    pub fn new_durable(segment_path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let segment_path = segment_path.as_ref();
        let engine = Engine::open_durable_wal_segment_auto(segment_path).map_err(|err| {
            format!(
                "failed to open durable WAL segment {}: {err}",
                segment_path.display()
            )
        })?;
        Ok(Self::from_engine(engine))
    }

    /// Construct the shared façade honoring the first-class durability deployment config
    /// (write-path assessment D4): when `GPU_DB_WAL_SEGMENT` is set, the engine is opened
    /// crash-durable at that path (recover-or-create); unset keeps the in-memory WAL of
    /// [`SharedEngine::new`]. Servers use this so durability is a deployment setting, not a
    /// code change.
    pub fn new_from_env() -> Result<Self, String> {
        match durable_wal_segment_from_env(std::env::var_os("GPU_DB_WAL_SEGMENT").as_deref()) {
            Some(segment_path) => Self::new_durable(segment_path),
            None => Ok(Self::new()),
        }
    }

    /// Whether this façade's engine fsyncs commits (a durable WAL segment is installed).
    pub fn is_durable(&self) -> bool {
        self.engine.wal_is_durable()
    }

    /// Read-only activity snapshot for proving that a mixed workload used the GPU-native read and
    /// write paths. It deliberately exposes neutral counters rather than the underlying `Engine`,
    /// preserving the facade as the protocol boundary. Only `resident_shards` is scoped to `table`;
    /// all activity counters and `valid_resident_tables` describe this engine instance globally.
    pub fn gpu_native_activity_snapshot(&self, table: &str) -> GpuNativeActivitySnapshot {
        let status = self.engine.status_snapshot();
        GpuNativeActivitySnapshot {
            valid_resident_tables: status.relational_residency.valid_snapshot_count(),
            resident_shards: self.engine.resident_shard_count(table),
            dense_index_probe_batches: self.engine.dense_index_probe_hits(),
            sharded_point_batches: self.engine.sharded_point_batch_hits(),
            sharded_gpu_probe_batches: self.engine.sharded_point_gpu_probe_hits(),
            sharded_binary_route_batches: self.engine.sharded_point_binary_route_hits(),
            open_shard_append_commits: self.engine.open_shard_append_hits(),
            device_authoritative_commits: self.engine.device_authoritative_commits(),
        }
    }

    /// Borrow the shared engine (no lock — the engine is interior-mutable). The point-lookup batcher
    /// uses this to drive a whole batch's prepare→submit→complete over ONE pinned generation; the
    /// "one read-lock per batch" invariant is now "one `committed_seq` pin per batch", enforced inside
    /// the engine's `&self` job APIs. Mapped to `Result` only to preserve the batcher's existing call
    /// shape (it can no longer fail here — a panicked committer is surfaced by the per-statement
    /// `is_commit_path_poisoned()` checks, not by a poisoned façade lock).
    pub(crate) fn read_engine(&self) -> Result<&Engine, ()> {
        Ok(&self.engine)
    }

    /// Revalidate an already-described COPY target without executing or admitting a mutation.
    /// Extended-protocol Describe/Execute use this before emitting cached COPY metadata; final
    /// CopyDone still repeats the same proof under the engine's transaction/commit lock.
    pub fn revalidate_copy_target(
        &self,
        session: &SharedSession,
        target: &CopyTarget,
    ) -> Result<(), DbError> {
        self.ensure_session_owner(session)?;
        if session.transaction_failed {
            return Err(in_failed_transaction_error());
        }
        self.ensure_copy_target_owner(session, target)?;
        let current = match session.active_txn_id {
            Some(txn_id) => self
                .engine
                .relational_copy_target_in_transaction(txn_id, &target.copy.table),
            None => self.engine.relational_copy_target(&target.copy.table),
        }
        .map_err(map_execute_error)?
        .1;
        if current != target.proof {
            return Err(stale_copy_target_error(&target.copy.table));
        }
        Ok(())
    }

    fn ensure_copy_target_owner(
        &self,
        session: &SharedSession,
        target: &CopyTarget,
    ) -> Result<(), DbError> {
        if !Arc::ptr_eq(&self.identity, &target.engine_identity) {
            return Err(DbError {
                category: ErrorCategory::InvalidRequest,
                message: "COPY target belongs to a different SharedEngine".to_string(),
            });
        }
        if target
            .transaction_identity
            .is_some_and(|required| session.active_txn_id != Some(required))
        {
            return Err(DbError {
                category: ErrorCategory::Serialization,
                message:
                    "COPY target belongs to a different transaction/catalog generation context"
                        .to_string(),
            });
        }
        Ok(())
    }
}

fn submit_copy_from_start(
    shared: &SharedEngine,
    session: &mut SharedSession,
    copy: &CopyFromStdin,
) -> Result<QueryOutcome, DbError> {
    submit_copy_from_start_with_hook(shared, session, copy, || {})
}

fn submit_copy_from_start_with_hook(
    shared: &SharedEngine,
    session: &mut SharedSession,
    copy: &CopyFromStdin,
    on_target_resolved: impl FnOnce(),
) -> Result<QueryOutcome, DbError> {
    if session.transaction_failed {
        return Err(in_failed_transaction_error());
    }
    let was_active = session.in_transaction();
    let result = (|| {
        let (table_columns, proof, origin) = match session.active_txn_id {
            Some(txn_id) => shared
                .engine
                .relational_copy_target_in_transaction_with_origin(txn_id, &copy.table),
            None => shared
                .engine
                .relational_copy_target(&copy.table)
                .map(|(columns, proof)| {
                    (columns, proof, TransactionCopyTargetOrigin::SnapshotBase)
                }),
        }
        .map_err(map_execute_error)?;
        on_target_resolved();
        let (column_names, columns) = match &copy.columns {
            None => (
                table_columns
                    .iter()
                    .map(|column| column.name.clone())
                    .collect(),
                table_columns.iter().map(map_copy_column).collect(),
            ),
            Some(requested) => {
                let mut seen = std::collections::BTreeSet::new();
                let mut columns = Vec::with_capacity(requested.len());
                for name in requested {
                    if !seen.insert(name) {
                        return Err(DbError {
                            category: ErrorCategory::DuplicateColumn,
                            message: format!("COPY column \"{name}\" was specified more than once"),
                        });
                    }
                    let column = table_columns
                        .iter()
                        .find(|column| column.name == *name)
                        .ok_or_else(|| DbError {
                            category: ErrorCategory::UndefinedColumn,
                            message: format!("column \"{name}\" does not exist"),
                        })?;
                    columns.push(map_copy_column(column));
                }
                (requested.clone(), columns)
            }
        };
        let mut normalized_copy = copy.clone();
        normalized_copy.columns = Some(column_names);
        // A target equal to the currently published relation is portable across session
        // transaction cycles: final admission still revalidates it against the destination
        // transaction catalog. A private CREATE/shape or an older snapshot-only relation has no
        // published equivalent and must remain bound to the exact transaction that described it,
        // closing rollback/recreate ABA without breaking Parse/Sync/Execute lifecycle semantics.
        let transaction_identity = session.active_txn_id.and_then(|txn_id| {
            if origin == TransactionCopyTargetOrigin::TransactionOverlay {
                return Some(txn_id);
            }
            let published_matches = shared
                .engine
                .relational_copy_target(&normalized_copy.table)
                .is_ok_and(|(_columns, published)| published == proof);
            (!published_matches).then_some(txn_id)
        });
        Ok(QueryOutcome::CopyIn {
            target: CopyTarget {
                copy: normalized_copy,
                columns,
                proof,
                engine_identity: Arc::clone(&shared.identity),
                transaction_identity,
            },
        })
    })();
    if result.is_err() && was_active {
        session.mark_transaction_failed();
    }
    result
}

fn submit_copy_from(
    shared: &SharedEngine,
    session: &mut SharedSession,
    target: &CopyTarget,
    rows: Vec<Vec<DbValue>>,
) -> Result<QueryOutcome, DbError> {
    if session.transaction_failed {
        return Err(in_failed_transaction_error());
    }
    let was_active = session.in_transaction();
    if let Err(error) = shared.ensure_copy_target_owner(session, target) {
        if was_active {
            session.mark_transaction_failed();
        }
        return Err(error);
    }
    let txn_id = session
        .active_txn_id
        .unwrap_or_else(|| shared.take_txn_id());
    let rows = rows
        .iter()
        .map(|row| row.iter().map(map_db_value_to_sql).collect())
        .collect();
    let result = shared
        .engine
        .submit_transaction(
            txn_id,
            CopyMutationRequest::new(target.copy.clone(), rows, target.proof.clone()),
        )
        .map_err(map_execute_error)
        .and_then(|admitted| match admitted {
            TransactionAdmissionResult::Dml(result) if result.returning.is_none() => {
                Ok(QueryOutcome::Command {
                    tag: CommandTag::Copy,
                    rows_affected: Some(result.rows_affected),
                })
            }
            _ => Err(invalid_mutation_result("COPY FROM STDIN")),
        });
    if result.is_err() && was_active {
        session.mark_transaction_failed();
    }
    result
}

fn submit_copy_to(
    shared: &SharedEngine,
    session: &mut SharedSession,
    copy: &CopyToStdout,
) -> Result<QueryOutcome, DbError> {
    if session.transaction_failed {
        return Err(in_failed_transaction_error());
    }
    if let Err(error) = validate_copy_to_columns(copy) {
        if session.in_transaction() {
            session.mark_transaction_failed();
        }
        return Err(error);
    }
    let projection = match &copy.columns {
        Some(columns) => columns.join(", "),
        None => "*".to_string(),
    };
    let source = format!("SELECT {projection} FROM {}", copy.table);
    let parsed = ParsedCommand::parse(&source).map_err(map_parse_error)?;
    let outcome = submit_parsed(shared, session, parsed)?;
    match outcome {
        QueryOutcome::Rows { .. } => Ok(outcome),
        _ => Err(DbError {
            category: ErrorCategory::Internal,
            message: "COPY TO STDOUT did not produce a relational row result".to_string(),
        }),
    }
}

fn submit_prepared_copy_to(
    shared: &SharedEngine,
    session: &mut SharedSession,
    copy: &CopyToStdout,
    bound: &BoundPreparedStatement,
) -> Result<QueryOutcome, DbError> {
    if session.transaction_failed {
        return Err(in_failed_transaction_error());
    }
    if let Err(error) = validate_copy_to_columns(copy) {
        if session.in_transaction() {
            session.mark_transaction_failed();
        }
        return Err(error);
    }
    if !bound.is_exact_copy_to_select(copy) {
        let error = DbError {
            category: ErrorCategory::InvalidRequest,
            message: "bound prepared owner does not match the exact COPY TO projection".to_string(),
        };
        if session.in_transaction() {
            session.mark_transaction_failed();
        }
        return Err(error);
    }
    let outcome = prepared::submit_prepared_inner(shared, session, bound)?;
    match outcome {
        QueryOutcome::Rows { .. } => Ok(outcome),
        _ => Err(DbError {
            category: ErrorCategory::Internal,
            message: "prepared COPY TO STDOUT did not produce a relational row result".to_string(),
        }),
    }
}

fn validate_copy_to_columns(copy: &CopyToStdout) -> Result<(), DbError> {
    let Some(columns) = &copy.columns else {
        return Ok(());
    };
    let mut seen = std::collections::BTreeSet::new();
    for column in columns {
        if !seen.insert(column) {
            return Err(DbError {
                category: ErrorCategory::DuplicateColumn,
                message: format!("COPY column \"{column}\" was specified more than once"),
            });
        }
    }
    Ok(())
}

impl Default for SharedEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Pure decode of the `GPU_DB_WAL_SEGMENT` value (`None` ⇒ unset): the durable segment path, or
/// `None` for the in-memory default. An empty / whitespace-only value counts as unset. Kept
/// separate from the `std::env` read so it is unit-testable without mutating process-global env
/// in parallel tests (same pattern as the server's `parse_batching_flag`).
pub fn durable_wal_segment_from_env(value: Option<&std::ffi::OsStr>) -> Option<std::path::PathBuf> {
    let value = value?;
    if value.to_str().is_some_and(|s| s.trim().is_empty()) {
        return None;
    }
    Some(std::path::PathBuf::from(value))
}

/// Execute one already-parsed autocommit command after the public session-owned submission boundary
/// has excluded transaction control. There is no facade lock; every command runs on `&Engine`:
///
/// - **Reads** (`SELECT`) run truly lock-free against a single pinned `committed_seq` + the catalog
///   selected as-of that boundary + the per-table data generation (catalog↔data co-pinned), so a
///   concurrent writer or DDL never blocks or splits them.
/// - **Mutations** cross `Engine::submit_transaction` once. The engine
///   privately selects concurrent DML, explicit overlay, or serialized generic preparation before
///   any sequence/WAL claim; those strategies are not separate facade routes.
/// - **Legacy non-relational reads** (`GET`, the exact session-cleanup advisory-unlock call,
///   `currval`) use a separate unsequenced compatibility read boundary and cannot trigger
///   representation repair.
/// - **DDL / sequence-default INSERT / everything else** still serializes inside the engine's
///   catalog/commit ownership, not under a facade write lock, so it does not exclude readers.
///
/// Transaction-control statements are rejected defensively because only the session executor owns
/// their lifecycle. The per-statement txn id is a durable identity, not a transaction handle.
///
/// **Poison-on-panic (homed to the engine's own locks).** A writer that panics mid-commit poisons the
/// engine's `commit_mutex`; a DDL that panics mid-apply poisons the engine's `catalog_latch`.
/// Either way every subsequent statement fails loud with [`ErrorCategory::Internal`] rather than serve
/// possibly-torn state — the engine deliberately wedges (the WAL is the durable source of truth; a
/// restart replays it). A retryable SI serialization conflict is NOT a poison: it maps to
/// [`ErrorCategory::Serialization`] (class-40) and the client retries.
fn submit_autocommit_parsed_with_catalog(
    shared: &SharedEngine,
    parsed: ParsedCommand,
    expected_catalog_version: Option<u64>,
) -> Result<QueryOutcome, DbError> {
    let engine: &Engine = &shared.engine;
    match parsed.command() {
        Command::Select(select) => {
            // A writer that panicked mid-commit poisons the commit path; refuse to serve a read
            // against possibly-torn published state (re-homed poison policy).
            if engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let result = engine
                .execute_relational_select(select)
                .map_err(map_execute_error)?;
            Ok(map_relational_result(result))
        }
        Command::SelectLiteral(literal) => {
            if engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let result = engine
                .execute_relational_literal(literal)
                .map_err(map_execute_error)?;
            Ok(map_relational_result(result))
        }
        Command::SelectFunction(call) if is_effect_free_session_function(call) => {
            let tag = command_tag(parsed.command());
            engine
                .execute_parsed_compatibility_read(parsed)
                .map_err(map_execute_error)?;
            Ok(QueryOutcome::Command {
                tag,
                rows_affected: None,
            })
        }
        Command::SelectFunction(call) => {
            if engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let result = engine
                .execute_relational_function(call)
                .map_err(map_execute_error)?;
            Ok(map_relational_result(result))
        }
        Command::Begin { .. } | Command::Commit { .. } | Command::Rollback { .. } => {
            Err(stateless_transaction_control_error())
        }
        Command::GetKv { .. } | Command::SequenceCurrVal(_) => {
            let tag = command_tag(parsed.command());
            engine
                .execute_parsed_compatibility_read(parsed)
                .map_err(map_execute_error)?;
            Ok(QueryOutcome::Command {
                tag,
                rows_affected: None,
            })
        }
        other => {
            let tag = command_tag(other);
            let txn_id = shared.take_txn_id();
            if engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let mut request = MutationRequest::new(parsed);
            if let Some(version) = expected_catalog_version {
                request = request.with_expected_catalog_version(version);
            }
            match engine
                .submit_transaction(txn_id, request)
                .map_err(map_execute_error)?
            {
                TransactionAdmissionResult::Dml(result) => Ok(map_dml_result(tag, result)),
                TransactionAdmissionResult::Command => Ok(QueryOutcome::Command {
                    tag,
                    rows_affected: None,
                }),
                TransactionAdmissionResult::Transaction(_) => {
                    Err(invalid_mutation_result("statement"))
                }
                TransactionAdmissionResult::Predeclared(_) => {
                    Err(invalid_mutation_result("single statement"))
                }
            }
        }
    }
}

/// Execute through the concurrent shared engine while preserving one transaction owner per client
/// session. Transaction control drives the engine's keyed snapshot lifecycle; SELECT executes on
/// that retained generation, while DML stages into a transaction-private GPU generation and COMMIT
/// publishes the resolved mutations atomically.
fn submit_text_inner(
    shared: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    let parsed = match ParsedCommand::parse_allowing_catalog(sql) {
        Ok(parsed) => parsed,
        Err(ParseError::Empty) => return Ok(QueryOutcome::Empty),
        // The typed parser intentionally rejects joins, expressions, and other richer SELECT
        // syntax. The engine's libpg_query lowering is the one general GPU relational path for
        // those statements; an actual syntax error or non-SELECT still fails pre-effect there.
        Err(_) if gpu_db_sql::is_select_statement(sql) => {
            return submit_general_select_text_inner(shared, session, sql);
        }
        Err(error) => {
            session.mark_transaction_failed();
            return Err(map_parse_error(error));
        }
    };
    submit_parsed(shared, session, parsed)
}

fn submit_general_select_text_inner(
    shared: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    if session.transaction_failed {
        return Err(in_failed_transaction_error());
    }
    let was_active = session.in_transaction();
    if was_active {
        session.transaction_has_statement = true;
    }
    let result = match session.active_txn_id {
        Some(txn_id) => shared
            .engine
            .execute_resident_expr_select_sql_in_transaction(txn_id, sql),
        None => shared.engine.execute_resident_expr_select_sql(sql),
    }
    .map(map_relational_result)
    .map_err(map_execute_error);
    if result.is_err() && was_active {
        session.mark_transaction_failed();
    }
    result
}

/// Catalog SELECTs must cross the libpg_query binder even when the bounded typed parser can
/// represent their surface shape. That binder owns schema qualification, empty-source binding,
/// aggregate/grouping validation, and catalog presentation; letting a typed parse bypass it would
/// make correctness depend on which parser happened to accept the SQL first.
fn select_requires_general_catalog_binding(select: &Select) -> bool {
    !select.public_only
        && (select.table.starts_with("pg_catalog.")
            || select.table.starts_with("information_schema.")
            || select.table.starts_with("pg_"))
}

fn select_has_pg_dump_sequence_state_shape(select: &Select) -> bool {
    select.public_only
        && select.projection
            == SelectProjection::Columns(vec!["last_value".to_string(), "is_called".to_string()])
        && select.group_by.is_none()
        && select.having_groups.is_empty()
        && select.filter_groups.is_empty()
        && select.order_by.is_empty()
        && select.limit.is_none()
        && select.offset.is_none()
}

fn select_requires_general_engine_binding(select: &Select) -> bool {
    select_requires_general_catalog_binding(select)
        || select_has_pg_dump_sequence_state_shape(select)
}

fn submit_parsed(
    shared: &SharedEngine,
    session: &mut SharedSession,
    parsed: ParsedCommand,
) -> Result<QueryOutcome, DbError> {
    submit_parsed_with_catalog(shared, session, parsed, None)
}

pub(crate) fn submit_parsed_with_catalog(
    shared: &SharedEngine,
    session: &mut SharedSession,
    parsed: ParsedCommand,
    expected_catalog_version: Option<u64>,
) -> Result<QueryOutcome, DbError> {
    submit_parsed_with_catalog_mode(shared, session, parsed, expected_catalog_version, false)
}

fn submit_bound_prepared_with_catalog(
    shared: &SharedEngine,
    session: &mut SharedSession,
    parsed: ParsedCommand,
    expected_catalog_version: Option<u64>,
) -> Result<QueryOutcome, DbError> {
    submit_parsed_with_catalog_mode(shared, session, parsed, expected_catalog_version, true)
}

fn submit_parsed_with_catalog_mode(
    shared: &SharedEngine,
    session: &mut SharedSession,
    parsed: ParsedCommand,
    expected_catalog_version: Option<u64>,
    bound_prepared_ast: bool,
) -> Result<QueryOutcome, DbError> {
    if session.transaction_failed {
        let rollback = match parsed.command() {
            Command::Rollback { chain } | Command::Commit { chain } => Some(*chain),
            _ => None,
        };
        let Some(chain) = rollback else {
            return Err(in_failed_transaction_error());
        };
        let rollback = ParsedCommand::parse(if chain {
            "ROLLBACK AND CHAIN"
        } else {
            "ROLLBACK"
        })
        .expect("static rollback command must parse");
        submit_parsed_inner(shared, session, rollback, None)?;
        return Ok(QueryOutcome::Command {
            tag: CommandTag::Rollback,
            rows_affected: None,
        });
    }
    if !bound_prepared_ast {
        if let Command::Select(select) = parsed.command() {
            if select_requires_general_engine_binding(select) {
                return submit_general_select_text_inner(shared, session, parsed.source());
            }
        }
    }
    if session.in_transaction()
        && !matches!(
            parsed.command(),
            Command::Begin { .. }
                | Command::Commit { .. }
                | Command::Rollback { .. }
                | Command::ResetAll
                | Command::SessionControl { .. }
                | Command::ShowTransactionIsolation
        )
    {
        session.transaction_has_statement = true;
    }
    let was_active = session.in_transaction();
    let result = submit_parsed_inner(shared, session, parsed, expected_catalog_version);
    if result.is_err() && was_active {
        session.mark_transaction_failed();
    }
    result
}

fn submit_parsed_inner(
    shared: &SharedEngine,
    session: &mut SharedSession,
    parsed: ParsedCommand,
    expected_catalog_version: Option<u64>,
) -> Result<QueryOutcome, DbError> {
    match parsed.command() {
        Command::PreparedCatalog(program) => {
            if shared.engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let result = match session.active_txn_id {
                Some(txn_id) => shared
                    .engine
                    .execute_prepared_catalog_program_in_transaction(txn_id, program),
                None => shared.engine.execute_prepared_catalog_program(program),
            }
            .map_err(map_execute_error)?;
            Ok(map_relational_result(result))
        }
        Command::SessionControl {
            transaction,
            access_share_relations,
        } => {
            if let Some(characteristics) = transaction {
                reset_empty_transaction_characteristics(
                    shared,
                    session,
                    parsed.clone(),
                    *characteristics,
                )?;
            }
            if !access_share_relations.is_empty() {
                let Some(txn_id) = session.active_txn_id else {
                    return Err(DbError {
                        category: ErrorCategory::InvalidRequest,
                        message: "LOCK TABLE can only be used in transaction blocks".to_string(),
                    });
                };
                shared
                    .engine
                    .validate_access_share_relations_in_transaction(txn_id, access_share_relations)
                    .map_err(map_execute_error)?;
            }
            Ok(QueryOutcome::Command {
                tag: CommandTag::Other(if access_share_relations.is_empty() {
                    "SET".to_string()
                } else {
                    "LOCK TABLE".to_string()
                }),
                rows_affected: None,
            })
        }
        Command::ResetAll => Ok(QueryOutcome::Command {
            // The SQL parser deliberately normalizes the bounded session-cleanup family
            // (RESET/DISCARD/DEALLOCATE/CLOSE/UNLISTEN) to one effect-free command. It belongs
            // to the session control plane and must neither enter transaction catalog staging
            // nor claim an autocommit transaction/WAL position.
            tag: command_tag(parsed.command()),
            rows_affected: None,
        }),
        Command::ShowTransactionIsolation => {
            let isolation = session
                .transaction_characteristics
                .unwrap_or_default()
                .isolation;
            let value = match isolation {
                TransactionIsolation::ReadUncommitted | TransactionIsolation::ReadCommitted => {
                    "read committed"
                }
                TransactionIsolation::RepeatableRead => "repeatable read",
                TransactionIsolation::Serializable => "serializable",
            };
            Ok(QueryOutcome::Rows {
                columns: vec![ColumnMeta {
                    name: "transaction_isolation".to_string(),
                    logical_type: LogicalType::Text,
                }],
                rows: vec![vec![DbValue::Text(value.to_string())]],
            })
        }
        Command::Begin { characteristics } => {
            if session.active_txn_id.is_none() {
                let mut accepted = *characteristics;
                if accepted.isolation == TransactionIsolation::ReadUncommitted {
                    accepted.isolation = TransactionIsolation::ReadCommitted;
                }
                if shared.engine.is_commit_path_poisoned() {
                    return Err(poisoned_engine_error());
                }
                let txn_id = shared.take_txn_id();
                let result = shared
                    .engine
                    .submit_transaction(txn_id, parsed)
                    .map_err(map_execute_error)?;
                debug_assert!(matches!(
                    result,
                    TransactionAdmissionResult::Transaction(Some(id)) if id == txn_id
                ));
                session.active_txn_id = Some(txn_id);
                session.transaction_characteristics = Some(accepted);
                session.transaction_has_statement = false;
                session.transaction_failed = false;
            }
            Ok(QueryOutcome::Command {
                tag: CommandTag::Begin,
                rows_affected: None,
            })
        }
        Command::Commit { .. } => {
            if let Some(txn_id) = session.active_txn_id {
                if shared.engine.is_commit_path_poisoned() {
                    return Err(poisoned_engine_error());
                }
                let result = shared
                    .engine
                    .submit_transaction(txn_id, parsed)
                    .map_err(map_execute_error)?;
                let TransactionAdmissionResult::Transaction(successor) = result else {
                    return Err(invalid_mutation_result("COMMIT"));
                };
                if let Some(next_txn_id) = successor {
                    shared
                        .next_txn_id
                        .fetch_max(next_txn_id.saturating_add(1), Ordering::Relaxed);
                }
                session.active_txn_id = successor;
                session.transaction_characteristics =
                    successor.and(session.transaction_characteristics);
                session.transaction_has_statement = false;
                session.transaction_failed = false;
            }
            Ok(QueryOutcome::Command {
                tag: CommandTag::Commit,
                rows_affected: None,
            })
        }
        Command::Rollback { .. } => {
            if let Some(txn_id) = session.active_txn_id {
                if shared.engine.is_commit_path_poisoned() {
                    return Err(poisoned_engine_error());
                }
                let result = shared
                    .engine
                    .submit_transaction(txn_id, parsed)
                    .map_err(map_execute_error)?;
                let TransactionAdmissionResult::Transaction(successor) = result else {
                    return Err(invalid_mutation_result("ROLLBACK"));
                };
                if let Some(next_txn_id) = successor {
                    shared
                        .next_txn_id
                        .fetch_max(next_txn_id.saturating_add(1), Ordering::Relaxed);
                }
                session.active_txn_id = successor;
                session.transaction_characteristics =
                    successor.and(session.transaction_characteristics);
                session.transaction_has_statement = false;
                session.transaction_failed = false;
            }
            Ok(QueryOutcome::Command {
                tag: CommandTag::Rollback,
                rows_affected: None,
            })
        }
        Command::Select(select) if session.active_txn_id.is_some() => {
            if shared.engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let txn_id = session
                .active_txn_id
                .expect("guarded by transaction-active match arm");
            let result = shared
                .engine
                .execute_relational_select_in_transaction(txn_id, select)
                .map_err(map_execute_error)?;
            Ok(map_relational_result(result))
        }
        Command::SelectLiteral(literal) if session.active_txn_id.is_some() => {
            // The scalar has no catalog or table dependency, so every isolation level observes the
            // same immutable value. Session transaction ownership and failed-state gating still
            // happen above this arm; execution shares the GPU relational result path.
            if shared.engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let result = shared
                .engine
                .execute_relational_literal(literal)
                .map_err(map_execute_error)?;
            Ok(map_relational_result(result))
        }
        Command::SelectFunction(call)
            if session.active_txn_id.is_some() && is_effect_free_session_function(call) =>
        {
            if shared.engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let tag = command_tag(parsed.command());
            shared
                .engine
                .execute_parsed_compatibility_read(parsed)
                .map_err(map_execute_error)?;
            Ok(QueryOutcome::Command {
                tag,
                rows_affected: None,
            })
        }
        Command::SelectFunction(call) if session.active_txn_id.is_some() => {
            if shared.engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let txn_id = session
                .active_txn_id
                .expect("guarded by transaction-active match arm");
            let result = shared
                .engine
                .execute_relational_function_in_transaction(txn_id, call)
                .map_err(map_execute_error)?;
            Ok(map_relational_result(result))
        }
        Command::GetKv { .. } | Command::SequenceCurrVal(_) if session.active_txn_id.is_some() => {
            // Compatibility reads remain unsequenced inside an explicit/implicit block just as
            // they are in autocommit. In particular asyncpg's pool reset begins with
            // pg_advisory_unlock_all(); routing it through mutation admission would both violate
            // the read-only ownership rule and fail the enclosing implicit Query transaction.
            if shared.engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let tag = command_tag(parsed.command());
            shared
                .engine
                .execute_parsed_compatibility_read(parsed)
                .map_err(map_execute_error)?;
            Ok(QueryOutcome::Command {
                tag,
                rows_affected: None,
            })
        }
        Command::Insert(_) | Command::Update(_) | Command::Delete(_)
            if session.active_txn_id.is_some() =>
        {
            if shared.engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let txn_id = session
                .active_txn_id
                .expect("guarded by transaction-active match arm");
            let tag = command_tag(parsed.command());
            let mut request = MutationRequest::new(parsed);
            if let Some(version) = expected_catalog_version {
                request = request.with_expected_catalog_version(version);
            }
            let admitted = shared
                .engine
                .submit_transaction(txn_id, request)
                .map_err(map_execute_error)?;
            let TransactionAdmissionResult::Dml(result) = admitted else {
                return Err(invalid_mutation_result("transaction DML"));
            };
            Ok(map_dml_result(tag, result))
        }
        _ if session.active_txn_id.is_some() => {
            if shared.engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let txn_id = session
                .active_txn_id
                .expect("guarded by transaction-active match arm");
            let tag = command_tag(parsed.command());
            let mut request = MutationRequest::new(parsed);
            if let Some(version) = expected_catalog_version {
                request = request.with_expected_catalog_version(version);
            }
            match shared
                .engine
                .submit_transaction(txn_id, request)
                .map_err(map_execute_error)?
            {
                TransactionAdmissionResult::Command => Ok(QueryOutcome::Command {
                    tag,
                    rows_affected: None,
                }),
                _ => Err(invalid_mutation_result("transaction catalog command")),
            }
        }
        _ => submit_autocommit_parsed_with_catalog(shared, parsed, expected_catalog_version),
    }
}

/// PostgreSQL allows `SET TRANSACTION` immediately after `BEGIN`, before the first query. Replace
/// only the characteristics on that still-empty engine snapshot; the transaction identity and
/// retained catalog/data generation remain unchanged, and rejection leaves the transaction active
/// for the caller's normal failed-transaction handling.
fn reset_empty_transaction_characteristics(
    shared: &SharedEngine,
    session: &mut SharedSession,
    parsed: ParsedCommand,
    mut characteristics: TransactionCharacteristics,
) -> Result<(), DbError> {
    let Some(current_txn_id) = session.active_txn_id else {
        return Err(DbError {
            category: ErrorCategory::InvalidRequest,
            message: "SET TRANSACTION requires an active transaction".to_string(),
        });
    };
    if session.transaction_has_statement {
        return Err(DbError {
            category: ErrorCategory::InvalidRequest,
            message: "SET TRANSACTION must precede the first transaction statement".to_string(),
        });
    }
    let result = shared
        .engine
        .submit_transaction(current_txn_id, MutationRequest::new(parsed))
        .map_err(map_execute_error)?;
    let TransactionAdmissionResult::Command = result else {
        return Err(invalid_mutation_result("SET TRANSACTION"));
    };
    if characteristics.isolation == TransactionIsolation::ReadUncommitted {
        characteristics.isolation = TransactionIsolation::ReadCommitted;
    }
    session.transaction_characteristics = Some(characteristics);
    Ok(())
}

/// Test-support: run a concurrent DML statement through the shared engine with a hook invoked
/// between the off-lock snapshot capture+prepare and the commit critical section (write-half MVCC).
/// The concurrency-correctness suite uses this to rendezvous two writers at a barrier in that window,
/// deterministically forcing the SI write-write conflict (both snapshot, then both commit). Runs on
/// the shared `&Engine` with no façade lock (the concurrent-DML path), so the hook runs fully
/// concurrently. The canonical facade allocator supplies its durable identity.
#[doc(hidden)]
fn submit_instrumented_dml(
    shared: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
    on_prepared: impl FnOnce() + Send + 'static,
) -> Result<(), DbError> {
    require_idle_instrumented_session(session, "DML")?;
    let engine: &Engine = &shared.engine;
    if engine.is_commit_path_poisoned() {
        return Err(poisoned_engine_error());
    }
    let parsed = ParsedCommand::parse(sql).map_err(map_parse_error)?;
    if !matches!(
        parsed.command(),
        Command::Insert(_) | Command::Update(_) | Command::Delete(_)
    ) {
        return Err(DbError {
            category: ErrorCategory::InvalidRequest,
            message: "instrumented DML test submission requires INSERT, UPDATE, or DELETE"
                .to_string(),
        });
    }
    let txn_id = shared.take_txn_id();
    let request = MutationRequest::new(parsed).with_prepared_hook(on_prepared);
    match engine
        .submit_transaction(txn_id, request)
        .map_err(map_execute_error)?
    {
        TransactionAdmissionResult::Dml(_) => Ok(()),
        _ => Err(invalid_mutation_result("instrumented DML")),
    }
}

/// Test-support: run a `SELECT` through the shared engine with a hook invoked in the window BETWEEN
/// the read's catalog bind and its data pin (PART B catalog↔data co-pinning). The
/// concurrency-correctness suite uses this to park a reader there while a writer commits a
/// shape-changing DDL, deterministically straddling the co-pinned region: with co-pinning the read's
/// (catalog, data) pair stays consistent, without it the decode mismatches the catalog shape. Runs on
/// the shared `&Engine` with no façade lock, so the writer's DDL commits fully concurrently with the
/// parked reader (the property only the lock-free path can exhibit).
#[doc(hidden)]
fn submit_instrumented_select(
    shared: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
    on_pinned: impl FnOnce(),
) -> Result<QueryOutcome, DbError> {
    require_idle_instrumented_session(session, "SELECT")?;
    let engine: &Engine = &shared.engine;
    if engine.is_commit_path_poisoned() {
        return Err(poisoned_engine_error());
    }
    let select = match parse_command(sql) {
        Ok(Command::Select(select)) => select,
        Ok(_) => {
            return Err(DbError {
                category: ErrorCategory::Unsupported,
                message: "execute_select_with_pinned_hook requires a SELECT".to_string(),
            })
        }
        Err(err) => return Err(map_parse_error(err)),
    };
    let result = engine
        .execute_relational_select_instrumented(&select, on_pinned)
        .map_err(map_execute_error)?;
    let columns = result.columns.iter().map(map_column).collect();
    let rows = result
        .rows
        .iter()
        .map(|row| row.iter().cloned().map(map_value).collect())
        .collect();
    Ok(QueryOutcome::Rows { columns, rows })
}

fn require_idle_instrumented_session(
    session: &mut SharedSession,
    kind: &str,
) -> Result<(), DbError> {
    match session.transaction_status() {
        SessionTransactionStatus::Idle => Ok(()),
        SessionTransactionStatus::FailedTransaction => Err(in_failed_transaction_error()),
        SessionTransactionStatus::InTransaction => {
            session.mark_transaction_failed();
            Err(DbError {
                category: ErrorCategory::Unsupported,
                message: format!("instrumented {kind} test submission requires an idle session"),
            })
        }
    }
}

/// Session-owned classify-or-fallback entry used by the canonical server. It parses exactly once,
/// keeps transaction control on the session boundary, and otherwise shares the point-read batcher.
fn submit_batched_text_inner(
    shared: &SharedEngine,
    session: &mut SharedSession,
    batcher: &PointLookupBatcher,
    sql: &str,
) -> SubmissionDispatch {
    if !batcher.is_bound_to(shared) {
        return SubmissionDispatch::Immediate(Err(DbError {
            category: ErrorCategory::InvalidRequest,
            message: "point-lookup batcher belongs to a different SharedEngine".to_string(),
        }));
    }
    let parsed = match ParsedCommand::parse_allowing_catalog(sql) {
        Ok(parsed) => parsed,
        Err(ParseError::Empty) => return SubmissionDispatch::Immediate(Ok(QueryOutcome::Empty)),
        Err(_) if gpu_db_sql::is_select_statement(sql) => {
            return SubmissionDispatch::Immediate(submit_general_select_text_inner(
                shared, session, sql,
            ));
        }
        Err(error) => {
            session.mark_transaction_failed();
            return SubmissionDispatch::Immediate(Err(map_parse_error(error)));
        }
    };
    if session.in_transaction()
        || matches!(
            parsed.command(),
            Command::Begin { .. } | Command::Commit { .. } | Command::Rollback { .. }
        )
    {
        return SubmissionDispatch::Immediate(submit_parsed(shared, session, parsed));
    }
    if let Command::Select(select) = parsed.command() {
        if select_requires_general_engine_binding(select) {
            return SubmissionDispatch::Immediate(submit_parsed(shared, session, parsed));
        }
    }
    // Classify the already-parsed owner under a read lock. Non-batchable commands consume that
    // same owner on the per-query path; no fallback reparses SQL text.
    match classify_batchable_point_lookup(shared, &parsed) {
        Some((select, needle)) => SubmissionDispatch::Batched(batcher.enqueue(select, needle)),
        // The fallback remains session-owned even while idle. Session metadata/control commands
        // (for example SHOW transaction isolation) are deliberately not autocommit mutations and
        // must retain the same SharedSession semantics as ordinary Text submission.
        None => SubmissionDispatch::Immediate(submit_parsed(shared, session, parsed)),
    }
}

/// Decide whether an already-parsed command is a batchable single-predicate int4 equality
/// point-lookup against a resident, valid-generation table, and if so return its `Select` + int4
/// needle. Conservative: it returns `Some` only when the engine's own resident-route
/// planner accepts the statement AND reports one of the batchable **all-int4** projection
/// shapes — `int4_equality_projection` (single column) or `int4_equality_multi_column_projection`
/// (multiple int4 columns). The mixed int4+text shape (`int4_equality_mixed_column_projection`)
/// is deliberately NOT batched: the resident `equal_any` kernel materializes int4 columns only,
/// and the general (text-capable) batch executor is not CUDA-context-safe on the batcher's
/// coalescer thread (a pre-existing latent constraint), so mixed point lookups take the unchanged
/// per-query path. The needle is extracted from the equality filter with the SAME
/// `filter_groups`/`filters`/`filter` precedence the engine uses to bind the job.
fn classify_batchable_point_lookup(
    shared: &SharedEngine,
    parsed: &ParsedCommand,
) -> Option<(Select, i32)> {
    let Command::Select(select) = parsed.command() else {
        return None;
    };
    let select = select.clone();
    // A read lock just for the planning probe; released before the batcher takes its own
    // (single) read lock for the batch. The planner is `&self`. An accepted single-predicate
    // all-int4-equality projection — single-column or multi-column — is batchable; anything else
    // (including the mixed int4/text projection — see the doc comment) returns `None` and the
    // caller takes the unchanged per-query path.
    {
        let engine = shared.read_engine().ok()?;
        let decision = engine.plan_relational_resident_route(&select);
        // Single-buffer resident int4 point lookups are always batchable (via the retained template).
        // lpb-for-shards: a SHARD-resident int4 point lookup (`sharded_int4_equality_[multi_column_]projection`)
        // is ALSO admitted when `shard_batched_point_read_enabled` is ON — the batcher serves it via the
        // batched cross-shard gather (`submit_sharded_point_lookups_batched`). Mixed int4+text is excluded on
        // both (text kernel is not coalescer-thread-safe). Flag OFF => sharded shapes take the per-query path
        // (unchanged / byte-identical).
        let single_buffer_batchable = matches!(
            decision.query_shape.as_str(),
            "int4_equality_projection" | "int4_equality_multi_column_projection"
        );
        let sharded_batchable = engine.shard_batched_point_read_enabled()
            && matches!(
                decision.query_shape.as_str(),
                "sharded_int4_equality_projection"
                    | "sharded_int4_equality_multi_column_projection"
            );
        if !decision.accepted || !(single_buffer_batchable || sharded_batchable) {
            return None;
        }
    }
    let needle = select_int4_equality_needle(&select)?;
    Some((select, needle))
}

/// Extract the int4 needle from a single-equality-predicate `Select`, mirroring the
/// engine's `filter_groups` → `filters` → `filter` precedence. Returns `None` if the
/// predicate is not exactly one int4 equality (the planner should already have rejected
/// such shapes, so this is a belt-and-suspenders guard).
fn select_int4_equality_needle(select: &Select) -> Option<i32> {
    let filter = if !select.filter_groups.is_empty() {
        if select.filter_groups.len() != 1 || select.filter_groups[0].len() != 1 {
            return None;
        }
        &select.filter_groups[0][0]
    } else if !select.filters.is_empty() {
        if select.filters.len() != 1 {
            return None;
        }
        &select.filters[0]
    } else {
        select.filter.as_ref()?
    };
    if filter.op != gpu_db_sql::SelectFilterOp::Eq {
        return None;
    }
    match filter.value {
        SqlValue::Int4(needle) => Some(needle),
        _ => None,
    }
}

/// The engine lock was poisoned by a panicked writer: refuse to serve possibly-torn state.
fn poisoned_engine_error() -> DbError {
    DbError {
        category: ErrorCategory::Internal,
        message: "engine unavailable: a prior statement panicked mid-execution, so engine \
                  state may be inconsistent — restart required (the WAL is the durable source \
                  of truth)"
            .to_string(),
    }
}

fn invalid_mutation_result(command: &str) -> DbError {
    DbError {
        category: ErrorCategory::Internal,
        message: format!("engine mutation admission returned the wrong result kind for {command}"),
    }
}

fn stale_copy_target_error(table: &str) -> DbError {
    DbError {
        category: ErrorCategory::Serialization,
        message: format!("COPY target relation \"{table}\" changed after COPY began; restart COPY"),
    }
}

fn stateless_transaction_control_error() -> DbError {
    DbError {
        category: ErrorCategory::Unsupported,
        message: "transaction control requires a session-owned facade boundary".to_string(),
    }
}

fn in_failed_transaction_error() -> DbError {
    DbError {
        category: ErrorCategory::InFailedTransaction,
        message: "current transaction is aborted, commands ignored until end of transaction block"
            .to_string(),
    }
}

fn map_parse_error(err: ParseError) -> DbError {
    let category = match &err {
        ParseError::Unsupported(source) if gpu_db_sql::is_copy_statement(source) => {
            ErrorCategory::Unsupported
        }
        _ => ErrorCategory::Syntax,
    };
    DbError {
        category,
        message: err.to_string(),
    }
}

fn map_execute_error(err: ExecuteError) -> DbError {
    // `Serialization` maps to the retryable class-40 category (write-half MVCC, Stage 4 — the
    // engine now exposes a typed `ExecuteError::Serialization` for SI write-write conflicts, so no
    // message string-sniffing). The remaining `Engine`/`Txn`/`Storage` cases collapse to `Engine`
    // (→ SQLSTATE XX000); finer categorization of those waits on typed engine errors (Phase 3).
    let category = if err.is_unique_violation() {
        ErrorCategory::UniqueViolation
    } else {
        match &err {
            ExecuteError::Parse(_) => ErrorCategory::Syntax,
            ExecuteError::NonReadCommand(_) | ExecuteError::Unsupported(_) => {
                ErrorCategory::Unsupported
            }
            ExecuteError::UndefinedRelation(_) => ErrorCategory::UndefinedRelation,
            ExecuteError::UndefinedColumn(_) => ErrorCategory::UndefinedColumn,
            ExecuteError::IndeterminateParameterType(_) => ErrorCategory::IndeterminateDatatype,
            ExecuteError::DatatypeMismatch(_) => ErrorCategory::DatatypeMismatch,
            ExecuteError::InvalidRequest(_) => ErrorCategory::InvalidRequest,
            ExecuteError::ResourceExhausted(_) => ErrorCategory::ResourceExhausted,
            ExecuteError::Serialization(_) => ErrorCategory::Serialization,
            _ => ErrorCategory::Engine,
        }
    };
    DbError {
        category,
        message: err.to_string(),
    }
}

fn map_logical_type(ty: SqlType) -> LogicalType {
    match ty {
        SqlType::Int4 => LogicalType::Int4,
        SqlType::Int8 => LogicalType::Int8,
        SqlType::Numeric { .. } => LogicalType::Numeric,
        SqlType::Bool => LogicalType::Bool,
        SqlType::Text => LogicalType::Text,
        SqlType::Date => LogicalType::Date,
        SqlType::Timestamp => LogicalType::Timestamp,
        SqlType::Uuid => LogicalType::Uuid,
        SqlType::Int2 => LogicalType::Int2,
    }
}

fn map_column(column: &RelationalColumn) -> ColumnMeta {
    ColumnMeta {
        name: column.name.clone(),
        logical_type: map_logical_type(column.ty),
    }
}

fn map_copy_column(column: &gpu_db_sql::CopyColumn) -> CopyColumnMeta {
    CopyColumnMeta {
        name: column.name.clone(),
        logical_type: map_logical_type(column.ty),
        numeric_typmod: match column.ty {
            SqlType::Numeric { precision, scale } => Some((precision, scale)),
            _ => None,
        },
    }
}

fn map_relational_result(result: gpu_db_engine::RelationalSelectResult) -> QueryOutcome {
    let columns = result.columns.iter().map(map_column).collect();
    let rows = result
        .rows
        .iter()
        .map(|row| row.iter().cloned().map(map_value).collect())
        .collect();
    QueryOutcome::Rows { columns, rows }
}

fn map_dml_result(tag: CommandTag, result: gpu_db_engine::DmlExecutionResult) -> QueryOutcome {
    match result.returning {
        Some(returning) => QueryOutcome::Returning {
            tag,
            columns: returning.columns.iter().map(map_column).collect(),
            rows: returning
                .rows
                .iter()
                .map(|row| row.iter().cloned().map(map_value).collect())
                .collect(),
            rows_affected: result.rows_affected,
        },
        None => QueryOutcome::Command {
            tag,
            rows_affected: Some(result.rows_affected),
        },
    }
}

fn map_value(value: SqlValue) -> DbValue {
    match value {
        SqlValue::Null => DbValue::Null,
        SqlValue::Int4(value) => DbValue::Int4(value),
        SqlValue::Int8(value) => DbValue::Int8(value),
        SqlValue::Numeric(value) => DbValue::Numeric(value),
        SqlValue::Bool(value) => DbValue::Bool(value),
        SqlValue::Text(value) => DbValue::Text(value),
        SqlValue::Date(value) => DbValue::Date(value),
        SqlValue::Timestamp(value) => DbValue::Timestamp(value),
        SqlValue::Uuid(value) => DbValue::Uuid(value),
        SqlValue::Int2(value) => DbValue::Int2(value),
        SqlValue::Parameter { .. } => {
            unreachable!("facade results never contain unbound prepared parameters")
        }
    }
}

fn map_db_value_to_sql(value: &DbValue) -> SqlValue {
    match value {
        DbValue::Null => SqlValue::Null,
        DbValue::Int4(value) => SqlValue::Int4(*value),
        DbValue::Int8(value) => SqlValue::Int8(*value),
        DbValue::Numeric(value) => SqlValue::Numeric(*value),
        DbValue::Bool(value) => SqlValue::Bool(*value),
        DbValue::Text(value) => SqlValue::Text(value.clone()),
        DbValue::Date(value) => SqlValue::Date(*value),
        DbValue::Timestamp(value) => SqlValue::Timestamp(*value),
        DbValue::Uuid(value) => SqlValue::Uuid(*value),
        DbValue::Int2(value) => SqlValue::Int2(*value),
    }
}

fn map_db_value(value: &DbValue) -> SqlValue {
    match value {
        DbValue::Null => SqlValue::Null,
        DbValue::Int2(value) => SqlValue::Int2(*value),
        DbValue::Int4(value) => SqlValue::Int4(*value),
        DbValue::Int8(value) => SqlValue::Int8(*value),
        DbValue::Numeric(value) => SqlValue::Numeric(*value),
        DbValue::Bool(value) => SqlValue::Bool(*value),
        DbValue::Text(value) => SqlValue::Text(value.clone()),
        DbValue::Date(value) => SqlValue::Date(*value),
        DbValue::Timestamp(value) => SqlValue::Timestamp(*value),
        DbValue::Uuid(value) => SqlValue::Uuid(*value),
    }
}

fn command_tag(command: &Command) -> CommandTag {
    match command {
        Command::CreateTable(_) => CommandTag::CreateTable,
        Command::CreateIndex(_) => CommandTag::CreateIndex,
        Command::Insert(_) => CommandTag::Insert,
        Command::Update(_) => CommandTag::Update,
        Command::Delete(_) => CommandTag::Delete,
        Command::TruncateTable(_) => CommandTag::Truncate,
        _ => CommandTag::Other("OK".to_string()),
    }
}

fn is_effect_free_session_function(call: &gpu_db_sql::SelectFunction) -> bool {
    call.name == "pg_advisory_unlock_all"
}

#[cfg(test)]
mod tests;
