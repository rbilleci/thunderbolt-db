//! Protocol-neutral engine façade (prototype→production plan, Phase 0 / P0-M1).
//!
//! This crate is the single seam through which any client protocol reaches the
//! `Engine`. It exists to keep the engine **protocol-neutral**: the façade speaks
//! only engine-native concepts — a [`SessionId`], a SQL command string, neutral
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
//! - `rows_affected` is `None` for DML: the engine's `execute_text` does not yet
//!   return an affected-row count. Surfacing that is a tracked follow-up.
//! - Transaction control (`BEGIN`/`COMMIT`/`ROLLBACK`) owns one engine transaction identity and one
//!   generation-owned read snapshot for the session lifetime. SELECT consumes that retained catalog,
//!   MVCC, and GPU-residency generation. DML stages into a transaction-private GPU generation;
//!   COMMIT publishes its resolved row mutations through one durable record and one commit index.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use gpu_db_engine::{Engine, ExecuteError, RelationalColumn};
use gpu_db_sql::{parse_command, Command, Decimal128, ParseError, Select, SqlType, SqlValue};

pub mod pg_adapter;
mod point_lookup_batcher;

pub use point_lookup_batcher::{PointLookupBatcher, PointLookupBatcherActivitySnapshot};

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
    Other(String),
}

/// Neutral query result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryOutcome {
    Rows {
        columns: Vec<ColumnMeta>,
        rows: Vec<Vec<DbValue>>,
    },
    Command {
        tag: CommandTag,
        rows_affected: Option<u64>,
    },
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

/// Engine-owned session identity, decoupled from any transport/connection.
/// Long-lived protocols (pgwire, MySQL, WebSocket) and request-scoped protocols
/// (HTTP with a session token) both address a session by this id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

struct SessionState {
    active_txn_id: Option<u64>,
}

/// The protocol-neutral entry point to the engine.
pub struct EngineFacade {
    engine: Engine,
    sessions: BTreeMap<u64, SessionState>,
    next_session_id: u64,
    next_txn_id: u64,
}

impl EngineFacade {
    /// Construct a façade over a local single-node engine.
    pub fn new() -> Self {
        Self {
            engine: Engine::new_local(),
            sessions: BTreeMap::new(),
            next_session_id: 1,
            next_txn_id: 1,
        }
    }

    /// Open a session and return its id. Sessions are not tied to a connection.
    pub fn open_session(&mut self) -> SessionId {
        let id = self.next_session_id;
        self.next_session_id += 1;
        self.sessions.insert(
            id,
            SessionState {
                active_txn_id: None,
            },
        );
        SessionId(id)
    }

    /// Close a session. Unknown ids are ignored.
    pub fn close_session(&mut self, session: SessionId) {
        if let Some(state) = self.sessions.remove(&session.0) {
            if let Some(txn_id) = state.active_txn_id {
                // Connection/session teardown is an abort boundary. The transaction-control path
                // releases both TxnManager state and the transaction-held active snapshot.
                let _ = self.engine.execute_text(txn_id, "ROLLBACK");
            }
        }
    }

    /// Whether the session currently believes it is inside a transaction.
    pub fn session_in_transaction(&self, session: SessionId) -> bool {
        self.sessions
            .get(&session.0)
            .is_some_and(|state| state.active_txn_id.is_some())
    }

    // A monotonic durable identity. Explicit transaction control keeps one id from BEGIN through
    // COMMIT/ROLLBACK; its SELECT and DML use that held identity, while autocommit statements each
    // receive a fresh identity.
    fn take_txn_id(&mut self) -> u64 {
        let id = self.next_txn_id;
        self.next_txn_id += 1;
        id
    }

    /// Execute one SQL command on behalf of a session and return a neutral
    /// outcome. This is the single execution boundary every protocol adapter
    /// drives.
    pub fn execute(&mut self, session: SessionId, sql: &str) -> Result<QueryOutcome, DbError> {
        if !self.sessions.contains_key(&session.0) {
            return Err(DbError {
                category: ErrorCategory::Internal,
                message: format!("unknown session {}", session.0),
            });
        }

        let active_txn_id = self
            .sessions
            .get(&session.0)
            .and_then(|state| state.active_txn_id);
        match parse_command(sql) {
            Ok(Command::Begin) => {
                // Preserve the façade's existing idempotent BEGIN-in-BEGIN behavior (the neutral
                // result type has no warning channel), while avoiding a duplicate registry entry.
                if active_txn_id.is_none() {
                    let txn_id = self.take_txn_id();
                    self.engine
                        .execute_text(txn_id, sql)
                        .map_err(map_execute_error)?;
                    self.set_active_txn_id(session, Some(txn_id));
                }
                return Ok(QueryOutcome::Command {
                    tag: CommandTag::Begin,
                    rows_affected: None,
                });
            }
            Ok(Command::Commit { chain }) => {
                if let Some(txn_id) = active_txn_id {
                    let successor = self
                        .engine
                        .commit_explicit_transaction(txn_id, chain)
                        .map_err(map_execute_error)?;
                    if let Some(next_txn_id) = successor {
                        self.next_txn_id = self.next_txn_id.max(next_txn_id.saturating_add(1));
                    }
                    self.set_active_txn_id(session, successor);
                }
                return Ok(QueryOutcome::Command {
                    tag: CommandTag::Commit,
                    rows_affected: None,
                });
            }
            Ok(Command::Rollback { chain }) => {
                if let Some(txn_id) = active_txn_id {
                    let successor = self
                        .engine
                        .rollback_explicit_transaction(txn_id, chain)
                        .map_err(map_execute_error)?;
                    if let Some(next_txn_id) = successor {
                        self.next_txn_id = self.next_txn_id.max(next_txn_id.saturating_add(1));
                    }
                    self.set_active_txn_id(session, successor);
                }
                return Ok(QueryOutcome::Command {
                    tag: CommandTag::Rollback,
                    rows_affected: None,
                });
            }
            _ => {}
        }

        let statement_txn_id = self.take_txn_id();
        execute_on_engine_with_transaction(&mut self.engine, statement_txn_id, active_txn_id, sql)
    }

    fn set_active_txn_id(&mut self, session: SessionId, txn_id: Option<u64>) {
        if let Some(state) = self.sessions.get_mut(&session.0) {
            state.active_txn_id = txn_id;
        }
    }
}

impl Default for EngineFacade {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute one SQL command against a borrowed engine and return a neutral
/// outcome.
///
/// This is the stateless core of [`EngineFacade::execute`], exposed so a serving
/// path that already owns its `Engine` (the benchmark endpoint today, the
/// production server next) can route a statement through the neutral boundary
/// without surrendering ownership of the engine. Transaction-control statements
/// produce the corresponding [`CommandTag`] but do not update any session state —
/// the owning caller decides whether to track that.
pub fn execute_on_engine(
    engine: &mut Engine,
    txn_id: u64,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    execute_on_engine_with_transaction(engine, txn_id, None, sql)
}

fn execute_on_engine_with_transaction(
    engine: &mut Engine,
    statement_txn_id: u64,
    read_txn_id: Option<u64>,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    let command = match parse_command(sql) {
        Ok(command) => command,
        // An empty statement is not an error in the wire protocol — surface it as
        // a distinct neutral outcome so adapters emit EmptyQueryResponse.
        Err(ParseError::Empty) => return Ok(QueryOutcome::Empty),
        Err(err) => return Err(map_parse_error(err)),
    };
    match command {
        Command::Select(select) => {
            let result = match read_txn_id {
                Some(txn_id) => engine.execute_relational_select_in_transaction(txn_id, &select),
                None => engine.execute_relational_select(&select),
            }
            .map_err(map_execute_error)?;
            let columns = result.columns.iter().map(map_column).collect();
            let rows = result
                .rows
                .iter()
                .map(|row| row.iter().cloned().map(map_value).collect())
                .collect();
            Ok(QueryOutcome::Rows { columns, rows })
        }
        Command::Begin => Ok(QueryOutcome::Command {
            tag: CommandTag::Begin,
            rows_affected: None,
        }),
        Command::Commit { .. } => Ok(QueryOutcome::Command {
            tag: CommandTag::Commit,
            rows_affected: None,
        }),
        Command::Rollback { .. } => Ok(QueryOutcome::Command {
            tag: CommandTag::Rollback,
            rows_affected: None,
        }),
        other => {
            let tag = command_tag(&other);
            if let Some(txn_id) = read_txn_id {
                if matches!(
                    other,
                    Command::Insert(_) | Command::Update(_) | Command::Delete(_)
                ) {
                    engine
                        .execute_dml_in_transaction(txn_id, sql)
                        .map_err(map_execute_error)?;
                } else {
                    return Err(DbError {
                        category: ErrorCategory::Unsupported,
                        message: "command is not supported inside an active transaction; it was not executed"
                            .to_string(),
                    });
                }
            } else {
                engine
                    .execute_text(statement_txn_id, sql)
                    .map_err(map_execute_error)?;
            }
            Ok(QueryOutcome::Command {
                tag,
                rows_affected: None,
            })
        }
    }
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
    next_txn_id: AtomicU64,
}

/// Per-client transaction ownership for the lock-free shared-engine façade. The value is cheap,
/// connection-local, and `Send`; the engine keeps the authoritative transaction state and the
/// keyed snapshot hold. It deliberately contains no relational data or CPU execution state.
#[derive(Debug, Default)]
pub struct SharedSession {
    active_txn_id: Option<u64>,
}

impl SharedSession {
    pub fn in_transaction(&self) -> bool {
        self.active_txn_id.is_some()
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
        let next_txn_id = engine.next_durable_transaction_id_floor();
        Self {
            engine: Arc::new(engine),
            next_txn_id: AtomicU64::new(next_txn_id),
        }
    }

    /// Open a connection/request session for transaction ownership. Session identity is local to
    /// the façade; an engine transaction identity is allocated only when the client sends BEGIN.
    pub fn open_session(&self) -> SharedSession {
        SharedSession::default()
    }

    /// Close a shared session, rolling back any still-active transaction so its engine state and
    /// transaction-held snapshot cannot outlive a disconnected client.
    pub fn close_session(&self, session: &mut SharedSession) {
        if let Some(txn_id) = session.active_txn_id.take() {
            let _ = self.engine.execute_text(txn_id, "ROLLBACK");
        }
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

/// Execute one SQL statement against a shared engine (lock-free read path, write-half MVCC — the
/// destination milestone). There is NO façade lock; every statement runs on the shared `&Engine`:
///
/// - **Reads** (`SELECT`) run truly lock-free against a single pinned `committed_seq` + the catalog
///   selected as-of that boundary + the per-table data generation (catalog↔data co-pinned), so a
///   concurrent writer or DDL never blocks or splits them.
/// - **Concurrent DML** (`INSERT`/`UPDATE`/`DELETE` on a base table without `nextval` defaults) goes
///   through the engine's `&self` `execute_dml_concurrent` — off-lock prepare + a short internal
///   `commit_mutex` critical section. Concurrent writers OVERLAP each other and never block readers.
/// - **DDL / sequence-default INSERT / KV / everything else** goes through the engine's `&self`
///   `execute_text`, which serializes under the engine's own **catalog latch** (+ commit_mutex) — DDL
///   is serialized inside the engine, NOT by a façade write lock, so it no longer excludes readers.
///
/// Transaction-control statements run no engine call and produce only the tag. The per-statement txn
/// id is allocated internally (a durable-identity placeholder, not a transaction handle).
///
/// **Poison-on-panic (homed to the engine's own locks).** A writer that panics mid-commit poisons the
/// engine's `commit_mutex` (checked via [`Engine::is_commit_path_poisoned`]); a DDL that panics
/// mid-apply poisons the engine's `catalog_latch` (checked via [`Engine::is_catalog_latch_poisoned`]).
/// Either way every subsequent statement fails loud with [`ErrorCategory::Internal`] rather than serve
/// possibly-torn state — the engine deliberately wedges (the WAL is the durable source of truth; a
/// restart replays it). A retryable SI serialization conflict is NOT a poison: it maps to
/// [`ErrorCategory::Serialization`] (class-40) and the client retries.
pub fn execute_on_shared_engine(shared: &SharedEngine, sql: &str) -> Result<QueryOutcome, DbError> {
    let command = match parse_command(sql) {
        Ok(command) => command,
        Err(ParseError::Empty) => return Ok(QueryOutcome::Empty),
        Err(err) => return Err(map_parse_error(err)),
    };
    let engine: &Engine = &shared.engine;
    match command {
        Command::Select(select) => {
            // A writer that panicked mid-commit poisons the commit path; refuse to serve a read
            // against possibly-torn published state (re-homed poison policy).
            if engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let result = engine
                .execute_relational_select(&select)
                .map_err(map_execute_error)?;
            let columns = result.columns.iter().map(map_column).collect();
            let rows = result
                .rows
                .iter()
                .map(|row| row.iter().cloned().map(map_value).collect())
                .collect();
            Ok(QueryOutcome::Rows { columns, rows })
        }
        Command::Begin => Ok(QueryOutcome::Command {
            tag: CommandTag::Begin,
            rows_affected: None,
        }),
        Command::Commit { .. } => Ok(QueryOutcome::Command {
            tag: CommandTag::Commit,
            rows_affected: None,
        }),
        Command::Rollback { .. } => Ok(QueryOutcome::Command {
            tag: CommandTag::Rollback,
            rows_affected: None,
        }),
        other => {
            let tag = command_tag(&other);
            let txn_id = shared.take_txn_id();
            // Probe whether this statement is concurrent-eligible DML; if so, the engine's `&self`
            // concurrent-DML path (off-lock prepare + short commit_mutex) overlaps other writers and
            // never blocks readers.
            if engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            if engine.is_concurrent_dml(sql) {
                engine
                    .execute_dml_concurrent(txn_id, sql)
                    .map_err(map_execute_error)?;
                return Ok(QueryOutcome::Command {
                    tag,
                    rows_affected: None,
                });
            }
            // DDL / sequence-default INSERT / KV / other: the engine's catalog latch is the serializer.
            // A DDL that panicked mid-apply poisons the catalog latch — refuse rather than serve a torn
            // catalog (the catalog-latch counterpart of the commit-path poison check).
            if engine.is_catalog_latch_poisoned() {
                return Err(poisoned_engine_error());
            }
            engine
                .execute_text(txn_id, sql)
                .map_err(map_execute_error)?;
            Ok(QueryOutcome::Command {
                tag,
                rows_affected: None,
            })
        }
    }
}

/// Whether a SQL statement is transaction control and therefore must run through a connection's
/// [`SharedSession`] even while that session is currently idle. Servers use this to keep BEGIN off
/// the stateless/batched dispatch path without duplicating SQL prefix heuristics.
pub fn is_transaction_control(sql: &str) -> bool {
    matches!(
        parse_command(sql),
        Ok(Command::Begin | Command::Commit { .. } | Command::Rollback { .. })
    )
}

/// Execute through the concurrent shared engine while preserving one transaction owner per client
/// session. Transaction control drives the engine's keyed snapshot lifecycle; SELECT executes on
/// that retained generation, while DML stages into a transaction-private GPU generation and COMMIT
/// publishes the resolved mutations atomically.
pub fn execute_on_shared_engine_session(
    shared: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    match parse_command(sql) {
        Ok(Command::Begin) => {
            if session.active_txn_id.is_none() {
                if shared.engine.is_commit_path_poisoned() {
                    return Err(poisoned_engine_error());
                }
                let txn_id = shared.take_txn_id();
                shared
                    .engine
                    .execute_text(txn_id, sql)
                    .map_err(map_execute_error)?;
                session.active_txn_id = Some(txn_id);
            }
            Ok(QueryOutcome::Command {
                tag: CommandTag::Begin,
                rows_affected: None,
            })
        }
        Ok(Command::Commit { chain }) => {
            if let Some(txn_id) = session.active_txn_id {
                if shared.engine.is_commit_path_poisoned() {
                    return Err(poisoned_engine_error());
                }
                let successor = shared
                    .engine
                    .commit_explicit_transaction(txn_id, chain)
                    .map_err(map_execute_error)?;
                if let Some(next_txn_id) = successor {
                    shared
                        .next_txn_id
                        .fetch_max(next_txn_id.saturating_add(1), Ordering::Relaxed);
                }
                session.active_txn_id = successor;
            }
            Ok(QueryOutcome::Command {
                tag: CommandTag::Commit,
                rows_affected: None,
            })
        }
        Ok(Command::Rollback { chain }) => {
            if let Some(txn_id) = session.active_txn_id {
                if shared.engine.is_commit_path_poisoned() {
                    return Err(poisoned_engine_error());
                }
                let successor = shared
                    .engine
                    .rollback_explicit_transaction(txn_id, chain)
                    .map_err(map_execute_error)?;
                if let Some(next_txn_id) = successor {
                    shared
                        .next_txn_id
                        .fetch_max(next_txn_id.saturating_add(1), Ordering::Relaxed);
                }
                session.active_txn_id = successor;
            }
            Ok(QueryOutcome::Command {
                tag: CommandTag::Rollback,
                rows_affected: None,
            })
        }
        Ok(Command::Select(select)) if session.active_txn_id.is_some() => {
            if shared.engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let txn_id = session
                .active_txn_id
                .expect("guarded by transaction-active match arm");
            let result = shared
                .engine
                .execute_relational_select_in_transaction(txn_id, &select)
                .map_err(map_execute_error)?;
            let columns = result.columns.iter().map(map_column).collect();
            let rows = result
                .rows
                .iter()
                .map(|row| row.iter().cloned().map(map_value).collect())
                .collect();
            Ok(QueryOutcome::Rows { columns, rows })
        }
        Ok(Command::Insert(_) | Command::Update(_) | Command::Delete(_))
            if session.active_txn_id.is_some() =>
        {
            if shared.engine.is_commit_path_poisoned() {
                return Err(poisoned_engine_error());
            }
            let txn_id = session
                .active_txn_id
                .expect("guarded by transaction-active match arm");
            let command = parse_command(sql).expect("matched parsed DML");
            let tag = command_tag(&command);
            shared
                .engine
                .execute_dml_in_transaction(txn_id, sql)
                .map_err(map_execute_error)?;
            Ok(QueryOutcome::Command {
                tag,
                rows_affected: None,
            })
        }
        Ok(_) if session.active_txn_id.is_some() => Err(DbError {
            category: ErrorCategory::Unsupported,
            message: "command is not supported inside an active transaction; it was not executed"
                .to_string(),
        }),
        _ => execute_on_shared_engine(shared, sql),
    }
}

/// Test-support: run a concurrent DML statement through the shared engine with a hook invoked
/// between the off-lock snapshot capture+prepare and the commit critical section (write-half MVCC).
/// The concurrency-correctness suite uses this to rendezvous two writers at a barrier in that window,
/// deterministically forcing the SI write-write conflict (both snapshot, then both commit). Runs on
/// the shared `&Engine` with no façade lock (the concurrent-DML path), so the hook runs fully
/// concurrently. The supplied `txn_id` is the durable identity (the caller picks a unique one).
#[doc(hidden)]
pub fn execute_concurrent_dml_with_prepared_hook(
    shared: &SharedEngine,
    txn_id: u64,
    sql: &str,
    on_prepared: impl FnOnce(),
) -> Result<(), DbError> {
    let engine: &Engine = &shared.engine;
    if engine.is_commit_path_poisoned() {
        return Err(poisoned_engine_error());
    }
    engine
        .execute_dml_concurrent_instrumented(txn_id, sql, on_prepared)
        .map_err(map_execute_error)
}

/// Test-support: run a `SELECT` through the shared engine with a hook invoked in the window BETWEEN
/// the read's catalog bind and its data pin (PART B catalog↔data co-pinning). The
/// concurrency-correctness suite uses this to park a reader there while a writer commits a
/// shape-changing DDL, deterministically straddling the co-pinned region: with co-pinning the read's
/// (catalog, data) pair stays consistent, without it the decode mismatches the catalog shape. Runs on
/// the shared `&Engine` with no façade lock, so the writer's DDL commits fully concurrently with the
/// parked reader (the property only the lock-free path can exhibit).
#[doc(hidden)]
pub fn execute_select_with_pinned_hook(
    shared: &SharedEngine,
    sql: &str,
    on_pinned: impl FnOnce(),
) -> Result<QueryOutcome, DbError> {
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

/// Classify-or-fallback entry (Thread-3 Stage 1/Stage 4, strictly additive). If `sql` is a
/// batchable single-predicate int4-equality projection point-lookup (single-column,
/// multi-column all-int4, or mixed int4/text) on a resident, valid-generation
/// table, it is enqueued on the `batcher` (the caller `await`s the returned `oneshot`,
/// holding no semaphore permit while parked). EVERYTHING else — a non-SELECT, a SELECT
/// of any other shape, a non-resident/stale table, a parse-empty statement — falls
/// through to [`execute_on_shared_engine`] UNCHANGED. Misclassification only ever costs
/// a slow path, never a wrong result: a query that slipped through as "batchable" but is
/// actually unbatchable is rejected by the engine's job preparation and the waiter gets
/// that error (it is never silently mis-executed).
///
/// Returns either the resolved outcome or, when batched, the `oneshot::Receiver` to
/// `await`. Kept as a two-variant return (rather than `async` here) so the façade does
/// not pull in a tokio runtime feature; the async ingress drives the await.
pub fn execute_on_shared_engine_batched(
    shared: &SharedEngine,
    batcher: &PointLookupBatcher,
    sql: &str,
) -> BatchedDispatch {
    // Parse + classify under a read lock; non-batchable (including parse errors and
    // non-SELECTs) falls straight through to the unchanged per-query path.
    match classify_batchable_point_lookup(shared, sql) {
        Some((select, needle)) => BatchedDispatch::Batched(batcher.enqueue(select, needle)),
        None => BatchedDispatch::Immediate(execute_on_shared_engine(shared, sql)),
    }
}

/// Outcome of [`execute_on_shared_engine_batched`]: either an already-resolved result
/// (the statement took the unchanged per-query path) or a `oneshot` the caller awaits
/// (the statement was handed to the batcher).
pub enum BatchedDispatch {
    Immediate(Result<QueryOutcome, DbError>),
    Batched(tokio::sync::oneshot::Receiver<Result<QueryOutcome, DbError>>),
}

/// Decide whether `sql` is a batchable single-predicate int4 equality point-lookup against
/// a resident, valid-generation table, and if so return the parsed `Select` + its int4
/// needle. Conservative: it returns `Some` only when the engine's own resident-route
/// planner accepts the statement AND reports one of the batchable **all-int4** projection
/// shapes — `int4_equality_projection` (single column) or `int4_equality_multi_column_projection`
/// (multiple int4 columns). The mixed int4+text shape (`int4_equality_mixed_column_projection`)
/// is deliberately NOT batched: the resident `equal_any` kernel materializes int4 columns only,
/// and the general (text-capable) batch executor is not CUDA-context-safe on the batcher's
/// coalescer thread (a pre-existing latent constraint), so mixed point lookups take the unchanged
/// per-query path. The needle is extracted from the equality filter with the SAME
/// `filter_groups`/`filters`/`filter` precedence the engine uses to bind the job.
fn classify_batchable_point_lookup(shared: &SharedEngine, sql: &str) -> Option<(Select, i32)> {
    let Ok(Command::Select(select)) = parse_command(sql) else {
        return None;
    };
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

fn map_parse_error(err: ParseError) -> DbError {
    DbError {
        category: ErrorCategory::Syntax,
        message: err.to_string(),
    }
}

fn map_execute_error(err: ExecuteError) -> DbError {
    // `Serialization` maps to the retryable class-40 category (write-half MVCC, Stage 4 — the
    // engine now exposes a typed `ExecuteError::Serialization` for SI write-write conflicts, so no
    // message string-sniffing). The remaining `Engine`/`Txn`/`Storage` cases collapse to `Engine`
    // (→ SQLSTATE XX000); finer categorization of those waits on typed engine errors (Phase 3).
    let category = match &err {
        ExecuteError::Parse(_) => ErrorCategory::Syntax,
        ExecuteError::NonReadCommand(_) => ErrorCategory::Unsupported,
        ExecuteError::Serialization(_) => ErrorCategory::Serialization,
        _ => ErrorCategory::Engine,
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
    }
}

fn command_tag(command: &Command) -> CommandTag {
    match command {
        Command::CreateTable(_) => CommandTag::CreateTable,
        Command::CreateIndex(_) => CommandTag::CreateIndex,
        Command::Insert(_) => CommandTag::Insert,
        Command::Update(_) => CommandTag::Update,
        Command::Delete(_) => CommandTag::Delete,
        _ => CommandTag::Other("OK".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_wal_segment_env_decode_is_unset_for_missing_or_blank() {
        // D4 config decode (pure — no process-global env mutation in parallel tests): unset and
        // blank values keep the in-memory default; anything else is the durable segment path.
        assert_eq!(durable_wal_segment_from_env(None), None);
        assert_eq!(
            durable_wal_segment_from_env(Some(std::ffi::OsStr::new(""))),
            None
        );
        assert_eq!(
            durable_wal_segment_from_env(Some(std::ffi::OsStr::new("   "))),
            None
        );
        assert_eq!(
            durable_wal_segment_from_env(Some(std::ffi::OsStr::new("/var/lib/gpu-db/wal.segment"))),
            Some(std::path::PathBuf::from("/var/lib/gpu-db/wal.segment"))
        );
    }

    #[test]
    fn new_durable_shared_engine_fsyncs_commits_and_recovers_after_restart() {
        // D4: the served engine, when configured durable, survives a "crash" (drop + reopen at
        // the same segment path) with its committed writes intact.
        let dir = std::env::temp_dir().join(format!(
            "gpu-db-facade-durable-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let segment_path = dir.join("served.wal");

        let engine = SharedEngine::new_durable(&segment_path).unwrap();
        assert!(engine.is_durable());
        execute_on_shared_engine(&engine, "CREATE TABLE t (id INT)").unwrap();
        execute_on_shared_engine(&engine, "INSERT INTO t (id) VALUES (7)").unwrap();
        drop(engine);

        let recovered = SharedEngine::new_durable(&segment_path).unwrap();
        let outcome = execute_on_shared_engine(&recovered, "SELECT id FROM t").unwrap();
        let QueryOutcome::Rows { rows, .. } = outcome else {
            panic!("expected rows after durable recovery");
        };
        assert_eq!(rows, vec![vec![DbValue::Int4(7)]]);
        // The in-memory default remains non-durable (the assessment's D4 gap, now a setting).
        assert!(!SharedEngine::new().is_durable());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn null_maps_through_the_value_model_to_a_wire_null() {
        // M3 slice 1: SqlValue::Null → DbValue::Null → the wire boundary returns `None`
        // (the protocol's `-1` DataRow field length), while every typed value still
        // renders to `Some(text)`. This closes the engine→wire NULL path.
        assert_eq!(map_value(SqlValue::Null), DbValue::Null);
        assert_eq!(pg_adapter::db_value_text_opt(&DbValue::Null), None);
        assert_eq!(
            pg_adapter::db_value_text_opt(&DbValue::Int4(42)),
            Some("42".to_string())
        );
        assert_eq!(
            pg_adapter::db_value_text_opt(&DbValue::Text("x".to_string())),
            Some("x".to_string())
        );
        // A non-null value's text encoding is unchanged by the new boundary.
        assert_eq!(pg_adapter::db_value_text(&DbValue::Bool(true)), "t");
    }

    #[test]
    fn relational_lifecycle_round_trips_through_facade() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade
            .execute(session, "CREATE TABLE accounts (id INT, name TEXT)")
            .unwrap();
        facade
            .execute(
                session,
                "INSERT INTO accounts (id, name) VALUES (1, 'alice')",
            )
            .unwrap();
        facade
            .execute(session, "INSERT INTO accounts (id, name) VALUES (2, 'bob')")
            .unwrap();

        let outcome = facade
            .execute(session, "SELECT id, name FROM accounts WHERE id = 1")
            .unwrap();
        match outcome {
            QueryOutcome::Rows { columns, rows } => {
                let column_shape: Vec<(&str, LogicalType)> = columns
                    .iter()
                    .map(|column| (column.name.as_str(), column.logical_type))
                    .collect();
                assert_eq!(
                    column_shape,
                    vec![("id", LogicalType::Int4), ("name", LogicalType::Text)]
                );
                assert_eq!(
                    rows,
                    vec![vec![DbValue::Int4(1), DbValue::Text("alice".to_string())]]
                );
            }
            other => panic!("expected rows, got {other:?}"),
        }
    }

    #[test]
    fn count_aggregate_round_trips_as_a_neutral_integer() {
        // Finding (recorded in the P0-M1 milestone report): this engine returns
        // `COUNT(*)` as `Int4`, not `Int8`. The façade faithfully surfaces what the
        // engine produces; it does not invent wider aggregate typing. Richer
        // aggregate result types (int8/numeric for COUNT/SUM) are part of the
        // Phase 3 type-system work, at which point this test's expectation widens.
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade.execute(session, "CREATE TABLE t (a INT)").unwrap();
        facade
            .execute(session, "INSERT INTO t (a) VALUES (1)")
            .unwrap();
        facade
            .execute(session, "INSERT INTO t (a) VALUES (2)")
            .unwrap();

        let outcome = facade.execute(session, "SELECT COUNT(*) FROM t").unwrap();
        match outcome {
            QueryOutcome::Rows { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].len(), 1);
                let count = match &rows[0][0] {
                    DbValue::Int4(value) => i64::from(*value),
                    DbValue::Int8(value) => *value,
                    other => panic!("expected an integer count, got {other:?}"),
                };
                assert_eq!(count, 2);
            }
            other => panic!("expected rows, got {other:?}"),
        }
    }

    #[test]
    fn select_from_unknown_table_returns_neutral_error() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        let error = facade
            .execute(session, "SELECT id FROM missing_table")
            .unwrap_err();
        assert!(!error.message.is_empty());
        // The neutral error maps to a SQLSTATE only at the adapter; no SQLSTATE
        // ever appears in the façade type itself.
        let sqlstate = pg_adapter::error_sqlstate(error.category);
        assert_eq!(sqlstate.len(), 5);
    }

    #[test]
    fn command_tags_are_neutral_until_adapter_formats_them() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        let outcome = facade.execute(session, "CREATE TABLE t (a INT)").unwrap();
        assert_eq!(
            outcome,
            QueryOutcome::Command {
                tag: CommandTag::CreateTable,
                rows_affected: None
            }
        );
        assert_eq!(pg_adapter::command_complete_tag(&outcome), "CREATE TABLE");
    }

    #[test]
    fn sessions_track_transaction_state_independent_of_connection() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        assert!(!facade.session_in_transaction(session));
        facade.execute(session, "BEGIN").unwrap();
        assert!(facade.session_in_transaction(session));
        assert_eq!(facade.engine.active_txn_count(), 1);
        facade.execute(session, "COMMIT").unwrap();
        assert!(!facade.session_in_transaction(session));
        assert_eq!(facade.engine.active_txn_count(), 0);
    }

    #[test]
    fn session_close_rolls_back_engine_transaction_context() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade.execute(session, "BEGIN").unwrap();
        assert_eq!(facade.engine.active_txn_count(), 1);

        facade.close_session(session);

        assert!(!facade.session_in_transaction(session));
        assert_eq!(facade.engine.active_txn_count(), 0);
    }

    #[test]
    fn session_and_chain_transfers_to_a_known_engine_transaction() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade.execute(session, "BEGIN").unwrap();

        facade.execute(session, "COMMIT AND CHAIN").unwrap();
        assert!(facade.session_in_transaction(session));
        assert_eq!(facade.engine.active_txn_count(), 1);

        facade.execute(session, "ROLLBACK").unwrap();
        assert!(!facade.session_in_transaction(session));
        assert_eq!(facade.engine.active_txn_count(), 0);
    }

    #[test]
    fn active_transaction_rejects_commands_it_cannot_stage() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade.execute(session, "BEGIN").unwrap();

        let error = facade
            .execute(session, "CREATE TABLE must_not_autocommit (id INT)")
            .unwrap_err();
        assert_eq!(error.category, ErrorCategory::Unsupported);
        assert!(facade.session_in_transaction(session));

        facade.execute(session, "ROLLBACK").unwrap();
        facade
            .execute(session, "CREATE TABLE must_not_autocommit (id INT)")
            .unwrap();
    }

    #[test]
    fn shared_active_transaction_rejects_commands_it_cannot_stage() {
        let shared = SharedEngine::new();
        let mut session = shared.open_session();
        execute_on_shared_engine_session(&shared, &mut session, "BEGIN").unwrap();

        let error = execute_on_shared_engine_session(
            &shared,
            &mut session,
            "CREATE TABLE must_not_autocommit (id INT)",
        )
        .unwrap_err();
        assert_eq!(error.category, ErrorCategory::Unsupported);
        assert!(session.in_transaction());

        execute_on_shared_engine_session(&shared, &mut session, "ROLLBACK").unwrap();
        execute_on_shared_engine(&shared, "CREATE TABLE must_not_autocommit (id INT)").unwrap();
    }

    #[test]
    fn shared_session_owns_engine_transaction_and_aborts_on_close() {
        let shared = SharedEngine::new();
        let mut session = shared.open_session();
        assert!(!session.in_transaction());

        execute_on_shared_engine_session(&shared, &mut session, "BEGIN").unwrap();
        assert!(session.in_transaction());
        assert_eq!(shared.read_engine().unwrap().active_txn_count(), 1);

        execute_on_shared_engine_session(&shared, &mut session, "COMMIT AND CHAIN").unwrap();
        assert!(session.in_transaction());
        assert_eq!(shared.read_engine().unwrap().active_txn_count(), 1);

        shared.close_session(&mut session);
        assert!(!session.in_transaction());
        assert_eq!(shared.read_engine().unwrap().active_txn_count(), 0);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn sequential_session_select_uses_its_retained_gpu_generation() {
        let mut facade = EngineFacade::new();
        facade.engine.set_shard_residency_enabled(true);
        facade.engine.set_auto_admit_on_commit(true);
        let reader = facade.open_session();
        let writer = facade.open_session();
        facade
            .execute(reader, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        facade
            .execute(reader, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
            .unwrap();
        facade.execute(reader, "BEGIN").unwrap();
        facade
            .execute(writer, "UPDATE accounts SET balance = 200 WHERE id = 1")
            .unwrap();

        let outcome = facade
            .execute(reader, "SELECT balance FROM accounts WHERE id = 1")
            .unwrap();
        let QueryOutcome::Rows { rows, .. } = outcome else {
            panic!("expected rows")
        };
        assert_eq!(rows, vec![vec![DbValue::Int4(100)]]);
        facade.execute(reader, "ROLLBACK").unwrap();
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shared_session_select_uses_its_retained_gpu_generation() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine
            .execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        engine
            .execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
            .unwrap();
        let shared = SharedEngine::from_engine(engine);
        let mut reader = shared.open_session();
        execute_on_shared_engine_session(&shared, &mut reader, "BEGIN").unwrap();
        execute_on_shared_engine(&shared, "UPDATE accounts SET balance = 200 WHERE id = 1")
            .unwrap();

        let outcome = execute_on_shared_engine_session(
            &shared,
            &mut reader,
            "SELECT balance FROM accounts WHERE id = 1",
        )
        .unwrap();
        let QueryOutcome::Rows { rows, .. } = outcome else {
            panic!("expected rows")
        };
        assert_eq!(rows, vec![vec![DbValue::Int4(100)]]);
        execute_on_shared_engine_session(&shared, &mut reader, "ROLLBACK").unwrap();
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn sequential_session_dml_is_private_until_atomic_commit() {
        let mut facade = EngineFacade::new();
        facade.engine.set_shard_residency_enabled(true);
        facade.engine.set_auto_admit_on_commit(true);
        let writer = facade.open_session();
        let observer = facade.open_session();
        facade
            .execute(
                writer,
                "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)",
            )
            .unwrap();
        facade
            .execute(writer, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
            .unwrap();
        facade.execute(writer, "BEGIN").unwrap();
        let wal_before = facade.engine.durable_wal_records().len();
        facade
            .execute(writer, "UPDATE accounts SET balance = 200 WHERE id = 1")
            .unwrap();

        let QueryOutcome::Rows { rows, .. } = facade
            .execute(writer, "SELECT balance FROM accounts WHERE id = 1")
            .unwrap()
        else {
            panic!("expected writer rows")
        };
        assert_eq!(rows, vec![vec![DbValue::Int4(200)]]);
        let QueryOutcome::Rows { rows, .. } = facade
            .execute(observer, "SELECT balance FROM accounts WHERE id = 1")
            .unwrap()
        else {
            panic!("expected observer rows")
        };
        assert_eq!(rows, vec![vec![DbValue::Int4(100)]]);
        assert_eq!(facade.engine.durable_wal_records().len(), wal_before);

        facade.execute(writer, "COMMIT").unwrap();
        assert_eq!(facade.engine.durable_wal_records().len(), wal_before + 1);
        let QueryOutcome::Rows { rows, .. } = facade
            .execute(observer, "SELECT balance FROM accounts WHERE id = 1")
            .unwrap()
        else {
            panic!("expected committed rows")
        };
        assert_eq!(rows, vec![vec![DbValue::Int4(200)]]);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn shared_session_dml_is_private_until_atomic_commit() {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine
            .execute_text(1, "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT)")
            .unwrap();
        engine
            .execute_text(2, "INSERT INTO accounts (id, balance) VALUES (1, 100)")
            .unwrap();
        let shared = SharedEngine::from_engine(engine);
        let mut writer = shared.open_session();
        execute_on_shared_engine_session(&shared, &mut writer, "BEGIN").unwrap();
        execute_on_shared_engine_session(
            &shared,
            &mut writer,
            "UPDATE accounts SET balance = 200 WHERE id = 1",
        )
        .unwrap();

        let QueryOutcome::Rows { rows, .. } = execute_on_shared_engine_session(
            &shared,
            &mut writer,
            "SELECT balance FROM accounts WHERE id = 1",
        )
        .unwrap() else {
            panic!("expected writer rows")
        };
        assert_eq!(rows, vec![vec![DbValue::Int4(200)]]);
        let QueryOutcome::Rows { rows, .. } =
            execute_on_shared_engine(&shared, "SELECT balance FROM accounts WHERE id = 1").unwrap()
        else {
            panic!("expected observer rows")
        };
        assert_eq!(rows, vec![vec![DbValue::Int4(100)]]);

        execute_on_shared_engine_session(&shared, &mut writer, "COMMIT").unwrap();
        let QueryOutcome::Rows { rows, .. } =
            execute_on_shared_engine(&shared, "SELECT balance FROM accounts WHERE id = 1").unwrap()
        else {
            panic!("expected committed rows")
        };
        assert_eq!(rows, vec![vec![DbValue::Int4(200)]]);
    }

    #[test]
    fn empty_statement_is_neutral_empty_not_a_syntax_error() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        for sql in ["", "   ", ";", ";;;"] {
            assert_eq!(
                facade.execute(session, sql),
                Ok(QueryOutcome::Empty),
                "empty statement {sql:?} should be QueryOutcome::Empty"
            );
        }
    }

    #[test]
    fn unknown_session_is_rejected() {
        let mut facade = EngineFacade::new();
        let error = facade.execute(SessionId(999), "SELECT 1").unwrap_err();
        assert_eq!(error.category, ErrorCategory::Internal);
    }

    fn shared_count(shared: &SharedEngine) -> i64 {
        match execute_on_shared_engine(shared, "SELECT COUNT(*) FROM t").unwrap() {
            QueryOutcome::Rows { rows, .. } => match &rows[0][0] {
                DbValue::Int4(value) => i64::from(*value),
                DbValue::Int8(value) => *value,
                other => panic!("expected integer count, got {other:?}"),
            },
            other => panic!("expected rows, got {other:?}"),
        }
    }

    #[test]
    fn shared_engine_round_trips_write_then_read() {
        // The read/write-lock split (P1-M4): writes go through the write lock, reads the
        // read lock, both via the single shared entry point.
        let shared = SharedEngine::new();
        execute_on_shared_engine(&shared, "CREATE TABLE t (a INT)").unwrap();
        execute_on_shared_engine(&shared, "INSERT INTO t (a) VALUES (1)").unwrap();
        execute_on_shared_engine(&shared, "INSERT INTO t (a) VALUES (2)").unwrap();
        assert_eq!(shared_count(&shared), 2);
    }

    #[test]
    fn shared_engine_serves_concurrent_readers() {
        // Many threads read one shared engine concurrently (read lock) and all see the
        // correct committed state — the concurrent-dispatch property the server relies on.
        use std::sync::Arc;
        use std::thread;

        let shared = Arc::new(SharedEngine::new());
        execute_on_shared_engine(&shared, "CREATE TABLE t (a INT)").unwrap();
        for i in 0..50 {
            execute_on_shared_engine(&shared, &format!("INSERT INTO t (a) VALUES ({i})")).unwrap();
        }
        let mut handles = Vec::new();
        for _ in 0..8 {
            let shared = Arc::clone(&shared);
            handles.push(thread::spawn(move || {
                for _ in 0..25 {
                    assert_eq!(shared_count(&shared), 50);
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }

    // ---- Phase-3 M1: typed storable columns (NUMERIC / BIGINT / BOOL) ----

    /// Run a SELECT and return its rows, panicking on anything else.
    fn select_rows(facade: &mut EngineFacade, session: SessionId, sql: &str) -> Vec<Vec<DbValue>> {
        match facade.execute(session, sql).unwrap() {
            QueryOutcome::Rows { rows, .. } => rows,
            other => panic!("expected rows from {sql:?}, got {other:?}"),
        }
    }

    #[test]
    fn typed_columns_round_trip_create_insert_select_and_text_wire() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade
            .execute(
                session,
                "CREATE TABLE acct (bal NUMERIC(12,2), n BIGINT, ok BOOL)",
            )
            .unwrap();
        facade
            .execute(
                session,
                "INSERT INTO acct (bal, n, ok) VALUES (1234.5, 9000000000, TRUE)",
            )
            .unwrap();

        let outcome = facade
            .execute(session, "SELECT bal, n, ok FROM acct")
            .unwrap();
        let QueryOutcome::Rows { columns, rows } = outcome else {
            panic!("expected rows");
        };
        // Column logical types map to numeric / int8 / bool.
        assert_eq!(
            columns.iter().map(|c| c.logical_type).collect::<Vec<_>>(),
            vec![LogicalType::Numeric, LogicalType::Int8, LogicalType::Bool]
        );
        // Stored values round-trip; the numeric rescales to the column scale (2).
        assert_eq!(
            rows,
            vec![vec![
                DbValue::Numeric(Decimal128::new(123450, 2)),
                DbValue::Int8(9_000_000_000),
                DbValue::Bool(true),
            ]]
        );
        // Text wire encoding: money with two decimals, plain bigint, `t` for true.
        let wire: Vec<String> = rows[0].iter().map(pg_adapter::db_value_text).collect();
        assert_eq!(wire, vec!["1234.50", "9000000000", "t"]);
    }

    #[test]
    fn numeric_equality_is_scale_insensitive() {
        // A stored `1.0` (declared NUMERIC(12,2), so persisted as 1.00) must match a
        // `WHERE bal = 1.00` literal — canonical-by-construction value-index equality.
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade
            .execute(session, "CREATE TABLE m (bal NUMERIC(12,2))")
            .unwrap();
        facade
            .execute(session, "INSERT INTO m (bal) VALUES (1.0)")
            .unwrap();

        let matched = select_rows(&mut facade, session, "SELECT bal FROM m WHERE bal = 1.00");
        assert_eq!(
            matched,
            vec![vec![DbValue::Numeric(Decimal128::new(100, 2))]]
        );
        // And the equality fast-path renders the stored money form on the wire.
        assert_eq!(pg_adapter::db_value_text(&matched[0][0]), "1.00");

        // A different value does not match.
        let unmatched = select_rows(&mut facade, session, "SELECT bal FROM m WHERE bal = 2.00");
        assert!(unmatched.is_empty());
    }

    #[test]
    fn numeric_foreign_key_equality_links_parent_and_child() {
        // A NUMERIC equality check across a declared FK exercises the value-index equality
        // path on Decimal128 keys end to end (42.0 stored under NUMERIC(12,2) == 42.00).
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade
            .execute(
                session,
                "CREATE TABLE parent (id NUMERIC(12,2) PRIMARY KEY)",
            )
            .unwrap();
        facade
            .execute(session, "INSERT INTO parent (id) VALUES (42.00)")
            .unwrap();
        facade
            .execute(session, "CREATE TABLE child (pid NUMERIC(12,2))")
            .unwrap();
        facade
            .execute(
                session,
                "ALTER TABLE ONLY public.child ADD CONSTRAINT child_pid_fk FOREIGN KEY (pid) REFERENCES public.parent(id)",
            )
            .unwrap();
        // The FK check resolves the parent row by NUMERIC equality (42.0 == 42.00).
        facade
            .execute(session, "INSERT INTO child (pid) VALUES (42.0)")
            .unwrap();
        // A missing parent key is rejected by the FK.
        let err = facade
            .execute(session, "INSERT INTO child (pid) VALUES (7.00)")
            .unwrap_err();
        assert!(!err.message.is_empty());
    }

    #[test]
    fn numeric_overflow_beyond_precision_is_a_clean_error() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade
            .execute(session, "CREATE TABLE small (bal NUMERIC(4,2))")
            .unwrap();
        // 999.99 needs 5 significant digits but the column allows 4 -> numeric field overflow.
        let err = facade
            .execute(session, "INSERT INTO small (bal) VALUES (999.99)")
            .unwrap_err();
        assert!(
            err.message.contains("numeric field overflow"),
            "unexpected error: {}",
            err.message
        );
    }

    #[test]
    fn count_star_returns_a_neutral_int8() {
        // Phase-3 widening: COUNT(*) is now Int8 (the expectation the P0-M1 test anticipated).
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade.execute(session, "CREATE TABLE t (a INT)").unwrap();
        facade
            .execute(session, "INSERT INTO t (a) VALUES (1)")
            .unwrap();
        facade
            .execute(session, "INSERT INTO t (a) VALUES (2)")
            .unwrap();
        let rows = select_rows(&mut facade, session, "SELECT COUNT(*) FROM t");
        assert_eq!(rows, vec![vec![DbValue::Int8(2)]]);
    }

    #[test]
    fn bigint_and_bool_round_trip_on_gpu_native_facade() {
        let mut facade = EngineFacade::new();
        let session = facade.open_session();
        facade
            .execute(session, "CREATE TABLE flags (n BIGINT, ok BOOL)")
            .unwrap();
        facade
            .execute(
                session,
                "INSERT INTO flags (n, ok) VALUES (10000000000, TRUE)",
            )
            .unwrap();
        facade
            .execute(
                session,
                "INSERT INTO flags (n, ok) VALUES (20000000000, FALSE)",
            )
            .unwrap();
        // The production facade requires a supported GPU route. Predicate-family coverage lives
        // in the engine's actual-GPU expression suite; this facade test owns neutral type mapping
        // and the false wire representation without depending on the removed CPU query fallback.
        let rows = select_rows(&mut facade, session, "SELECT n, ok FROM flags");
        assert_eq!(
            rows,
            vec![
                vec![DbValue::Int8(10_000_000_000), DbValue::Bool(true)],
                vec![DbValue::Int8(20_000_000_000), DbValue::Bool(false)],
            ]
        );
        assert_eq!(pg_adapter::db_value_text(&rows[1][1]), "f");
    }
}
