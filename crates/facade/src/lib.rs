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
//! - The neutral value vocabulary ([`DbValue`], [`LogicalType`]) currently mirrors
//!   the protocol crate's `SqlValue`/`SqlType` and is converted at the boundary so
//!   those protocol types do not leak. The next milestone moves the neutral
//!   vocabulary into `gpu_db_types` and inverts the `engine -> protocol`
//!   dependency that exists today.
//! - `rows_affected` is `None` for DML: the engine's `execute_text` does not yet
//!   return an affected-row count. Surfacing that is a tracked follow-up.
//! - Transaction control (`BEGIN`/`COMMIT`/`ROLLBACK`) updates session state but
//!   does not yet drive real MVCC isolation — that is Phase 1 (P1-M3).

use std::collections::BTreeMap;

use gpu_db_engine::{Engine, ExecuteError, RelationalColumn};
use gpu_db_protocol::{parse_command, Command, ParseError, SqlType, SqlValue};

pub mod pg_adapter;

/// Neutral logical column type. Carries no wire OID; adapters derive the wire
/// type from this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalType {
    Int4,
    Int8,
    Numeric,
    Text,
}

/// Neutral value vocabulary. Owned by the façade so the protocol crate's
/// `SqlValue` does not cross the boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbValue {
    Int4(i32),
    Int8(i64),
    Numeric(String),
    Text(String),
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
}

/// Neutral error category. Adapters map this to their protocol's error code
/// taxonomy (`SQLSTATE`, MySQL error numbers, HTTP status, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Syntax,
    Unsupported,
    Engine,
    Internal,
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
    match parse_command(sql).map_err(map_parse_error)? {
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

fn map_parse_error(err: ParseError) -> DbError {
    DbError {
        category: ErrorCategory::Syntax,
        message: err.to_string(),
    }
}

fn map_execute_error(err: ExecuteError) -> DbError {
    let category = match &err {
        ExecuteError::Parse(_) => ErrorCategory::Syntax,
        ExecuteError::NonReadCommand(_) => ErrorCategory::Unsupported,
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
        SqlType::Text => LogicalType::Text,
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
        SqlValue::Text(value) => DbValue::Text(value),
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
    fn unknown_session_is_rejected() {
        let mut facade = EngineFacade::new();
        let error = facade.execute(SessionId(999), "SELECT 1").unwrap_err();
        assert_eq!(error.category, ErrorCategory::Internal);
    }
}
