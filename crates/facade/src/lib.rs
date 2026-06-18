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
//! - Transaction control (`BEGIN`/`COMMIT`/`ROLLBACK`) updates session state but
//!   does not yet drive real MVCC isolation — that is Phase 1 (P1-M3).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use gpu_db_engine::{Engine, ExecuteError, RelationalColumn};
use gpu_db_sql::{parse_command, Command, Decimal128, ParseError, Select, SqlType, SqlValue};

pub mod pg_adapter;
mod point_lookup_batcher;

pub use point_lookup_batcher::PointLookupBatcher;

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
}

/// Neutral value vocabulary. Owned by the façade so the protocol crate's
/// `SqlValue` does not cross the boundary. `Numeric` carries the engine's
/// fixed-point [`Decimal128`]; the wire adapter renders it to text (the binary
/// numeric wire codec is a later milestone).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbValue {
    Int4(i32),
    Int8(i64),
    Numeric(Decimal128),
    Bool(bool),
    Text(String),
    /// A `date` as i32 days since 2000-01-01 (PostgreSQL's date epoch).
    Date(i32),
    /// A `timestamp` as i64 microseconds since 2000-01-01 00:00:00.
    Timestamp(i64),
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
    in_transaction: bool,
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
                in_transaction: false,
            },
        );
        SessionId(id)
    }

    /// Close a session. Unknown ids are ignored.
    pub fn close_session(&mut self, session: SessionId) {
        self.sessions.remove(&session.0);
    }

    /// Whether the session currently believes it is inside a transaction.
    pub fn session_in_transaction(&self, session: SessionId) -> bool {
        self.sessions
            .get(&session.0)
            .map(|state| state.in_transaction)
            .unwrap_or(false)
    }

    // A per-statement monotonic id handed to the engine's `execute_text`. It is a
    // placeholder, NOT a transaction handle: `BEGIN/INSERT/INSERT/COMMIT` get
    // several unrelated ids today (real MVCC transaction identity arrives in
    // P1-M3). Allocated per `execute` call; only the write path actually uses it.
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

        let txn_id = self.take_txn_id();
        let outcome = execute_on_engine(&mut self.engine, txn_id, sql)?;
        if let QueryOutcome::Command { tag, .. } = &outcome {
            match tag {
                CommandTag::Begin => self.set_in_transaction(session, true),
                CommandTag::Commit | CommandTag::Rollback => {
                    self.set_in_transaction(session, false)
                }
                _ => {}
            }
        }
        Ok(outcome)
    }

    fn set_in_transaction(&mut self, session: SessionId, value: bool) {
        if let Some(state) = self.sessions.get_mut(&session.0) {
            state.in_transaction = value;
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
    let command = match parse_command(sql) {
        Ok(command) => command,
        // An empty statement is not an error in the wire protocol — surface it as
        // a distinct neutral outcome so adapters emit EmptyQueryResponse.
        Err(ParseError::Empty) => return Ok(QueryOutcome::Empty),
        Err(err) => return Err(map_parse_error(err)),
    };
    match command {
        Command::Select(select) => {
            let result = engine
                .execute_relational_select(&select)
                .map_err(map_execute_error)?;
            let columns = result.columns.iter().map(map_column).collect();
            let rows = result
                .rows
                .into_iter()
                .map(|row| row.into_iter().map(map_value).collect())
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

impl SharedEngine {
    /// Construct a shared façade over a local single-node engine.
    pub fn new() -> Self {
        Self::from_engine(Engine::new_local())
    }

    /// Wrap an already-built engine — e.g. one pre-seeded and warmed to GPU residency before
    /// serving, so the served read path takes the resident route (used by the GPU-retained
    /// benchmark).
    pub fn from_engine(engine: Engine) -> Self {
        Self {
            engine: Arc::new(engine),
            next_txn_id: AtomicU64::new(1),
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
                .into_iter()
                .map(|row| row.into_iter().map(map_value).collect())
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
            let txn_id = shared.next_txn_id.fetch_add(1, Ordering::Relaxed);
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
        .into_iter()
        .map(|row| row.into_iter().map(map_value).collect())
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
/// planner accepts the statement AND reports one of the batchable projection shapes —
/// `int4_equality_projection` (single column), `int4_equality_multi_column_projection`
/// (multiple int4 columns), or `int4_equality_mixed_column_projection` (int4 + text)
/// (Stage 4 widened this from the single-column shape only). The needle is extracted from
/// the equality filter with the SAME `filter_groups`/`filters`/`filter` precedence the
/// engine uses to bind the job, so the batcher's needle-dedup key matches the engine's
/// bound needle exactly.
fn classify_batchable_point_lookup(shared: &SharedEngine, sql: &str) -> Option<(Select, i32)> {
    let Ok(Command::Select(select)) = parse_command(sql) else {
        return None;
    };
    // A read lock just for the planning probe; released before the batcher takes its own
    // (single) read lock for the batch. The planner is `&self`. An accepted single-predicate
    // int4-equality projection — single-column, multi-column (all-int4), or mixed int4/text —
    // is batchable (Thread-3 Stage 4 widened this from single-column only); anything else
    // returns `None` and the caller takes the unchanged per-query path. The multi-predicate
    // mixed shape also reports `int4_equality_mixed_column_projection`, but the engine batch
    // API accepts only ONE predicate, so the single-needle guard below
    // (`select_int4_equality_needle`) rejects it and it falls through to the per-query path.
    {
        let engine = shared.read_engine().ok()?;
        let decision = engine.plan_relational_resident_route(&select);
        if !decision.accepted
            || !matches!(
                decision.query_shape.as_str(),
                "int4_equality_projection"
                    | "int4_equality_multi_column_projection"
                    | "int4_equality_mixed_column_projection"
            )
        {
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
        SqlValue::Int4(value) => DbValue::Int4(value),
        SqlValue::Int8(value) => DbValue::Int8(value),
        SqlValue::Numeric(value) => DbValue::Numeric(value),
        SqlValue::Bool(value) => DbValue::Bool(value),
        SqlValue::Text(value) => DbValue::Text(value),
        SqlValue::Date(value) => DbValue::Date(value),
        SqlValue::Timestamp(value) => DbValue::Timestamp(value),
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
        facade.execute(session, "COMMIT").unwrap();
        assert!(!facade.session_in_transaction(session));
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
    fn numeric_equality_is_scale_insensitive_on_the_cpu_path() {
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
    fn bigint_and_bool_round_trip_with_filters() {
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
        // BIGINT range filter on the CPU path.
        let big = select_rows(
            &mut facade,
            session,
            "SELECT n FROM flags WHERE n > 15000000000",
        );
        assert_eq!(big, vec![vec![DbValue::Int8(20_000_000_000)]]);
        // BOOL equality filter, and the `f` text wire form.
        let off = select_rows(
            &mut facade,
            session,
            "SELECT ok FROM flags WHERE ok = FALSE",
        );
        assert_eq!(off, vec![vec![DbValue::Bool(false)]]);
        assert_eq!(pg_adapter::db_value_text(&off[0][0]), "f");
    }
}
