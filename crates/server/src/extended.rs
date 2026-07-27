//! One PostgreSQL extended-query lifecycle shared by every engine-backed server ingress.

use std::collections::HashMap;
use std::io;

use crate::copy::{classify_copy_statement, CopyClassification, CopyStatement};
use crate::sql_cursor::{classify_sql_cursor_statement, SqlCursorAction};
use crate::sql_prepared::{
    classify_extended_sql_execute, decode_sql_execute_literal, deferred_sql_prepare_analysis_query,
    fill_unused_parameter_holes, ExtendedSqlExecuteArgument, SqlPreparedPlan,
};
use gpu_db_facade::{
    pg_adapter, BoundPreparedStatement, ColumnMeta, CommandTag, CopyTarget, DbError, DbValue,
    ErrorCategory, LogicalType, PreparedStatement, QueryOutcome, SessionTransactionStatus,
    SharedEngine, SharedSession, SubmissionRequest,
};
use gpu_db_protocol::backend::{BackendColumn, BackendError, BackendWriter};
use gpu_db_protocol::{
    canonicalize_sql_for_exact_match, parse_frontend_message, CopyToStdout, DescribeTarget,
    FrontendMessage,
};

#[derive(Debug, Clone)]
struct Statement {
    prepared: PreparedStatement,
    sql_execute_bind_arguments: Option<Vec<SqlExecuteBindArgument>>,
    deferred_execution_error: Option<DbError>,
    cursor_declaration: Option<ExtendedCursorDeclaration>,
    copy: Option<CopyStatement>,
    copy_target: Option<CopyTarget>,
    parameter_oids: Vec<u32>,
    columns: Vec<ColumnMeta>,
}

#[derive(Debug, Clone)]
struct Portal {
    statement_name: String,
    transaction_exit: bool,
    bound: BoundPreparedStatement,
    deferred_execution_error: Option<DbError>,
    cursor_declaration: Option<ExtendedCursorDeclaration>,
    copy: Option<CopyStatement>,
    copy_target: Option<CopyTarget>,
    columns: Vec<ColumnMeta>,
    result_formats: Vec<i16>,
    outcome: Option<Result<QueryOutcome, DbError>>,
    row_offset: usize,
    command_completed: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ExtendedCursorDeclaration {
    name: String,
    transaction_bound: bool,
}

#[derive(Debug, Clone)]
pub(crate) enum ExecutionRequest {
    Query(Box<BoundPreparedStatement>),
    Copy {
        statement: CopyStatement,
        target: Option<CopyTarget>,
        bound: Box<BoundPreparedStatement>,
    },
}

impl ExecutionRequest {
    #[cfg(test)]
    pub(crate) fn query_bound(&self) -> &BoundPreparedStatement {
        match self {
            Self::Query(bound) => bound,
            Self::Copy { .. } => panic!("test expected a regular query execution request"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PrepareRequest {
    pub statement_name: String,
    pub parsed: PreparedStatement,
    pub(crate) copy: Option<CopyStatement>,
    pub(crate) sql_execute_bind_arguments: Option<Vec<SqlExecuteBindArgument>>,
    pub(crate) outer_parameter_types: Option<Vec<LogicalType>>,
    pub(crate) deferred_execution_error: Option<DbError>,
    pub(crate) cursor_declaration: Option<ExtendedCursorDeclaration>,
    #[cfg(test)]
    pub query: String,
    pub parameter_type_hints: Vec<Option<LogicalType>>,
}

pub(crate) struct PrepareAnalysis {
    prepared: PreparedStatement,
    parameter_types: Vec<LogicalType>,
    copy_target: Option<CopyTarget>,
}

#[derive(Debug, Clone)]
pub(crate) enum SqlExecuteBindArgument {
    OuterParameter(usize),
    Literal(DbValue),
}

#[derive(Debug, Clone)]
pub(crate) struct BindRequest {
    portal_name: String,
    statement_name: String,
    statement: Statement,
    parameter_format_codes: Vec<i16>,
    parameters: Vec<Option<Vec<u8>>>,
    result_format_codes: Vec<i16>,
}

pub(crate) struct BindCompletion {
    portal_name: String,
    portal: Portal,
}

#[derive(Clone)]
pub(crate) enum DescriptionOwner {
    Cached,
    Copy(CopyTarget),
    Statement(Box<PreparedStatement>),
    Portal(Box<BoundPreparedStatement>),
}

pub(crate) enum Dispatch {
    SimpleQuery(String),
    Prepare(Box<PrepareRequest>),
    Bind(Box<BindRequest>),
    Describe {
        target: DescribeTarget,
        name: String,
    },
    Execute {
        portal_name: String,
        max_rows: u32,
    },
    Sync,
    Terminate,
    Response(Result<Vec<u8>, ExtendedError>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SkippingFrameAction {
    Parse,
    Discard,
    RollbackAndSync,
    Sync,
    Terminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MalformedFrameAction {
    SkipUntilSync,
    ErrorAndReady(TransactionAction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransactionAction {
    None,
    BeginImplicit,
    CommitImplicit,
    RollbackImplicit,
}

impl TransactionAction {
    pub(crate) fn sql(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::BeginImplicit => Some("BEGIN"),
            Self::CommitImplicit => Some("COMMIT"),
            Self::RollbackImplicit => Some("ROLLBACK"),
        }
    }

    pub(crate) fn failure_cleanup(self) -> Self {
        match self {
            Self::CommitImplicit => Self::RollbackImplicit,
            Self::None | Self::BeginImplicit | Self::RollbackImplicit => Self::None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExtendedError {
    pub code: &'static str,
    pub message: String,
}

impl ExtendedError {
    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub(crate) fn encode(&self) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        BackendWriter::new(&mut buf)
            .error_response(&BackendError::new(self.code, self.message.clone()))?;
        Ok(buf)
    }

    fn in_failed_transaction() -> Self {
        Self::new(
            "25P02",
            "current transaction is aborted, commands ignored until end of transaction block",
        )
    }
}

impl From<DbError> for ExtendedError {
    fn from(error: DbError) -> Self {
        Self::new(pg_adapter::error_sqlstate(error.category), error.message)
    }
}

impl From<gpu_db_protocol::FrontendMessageError> for ExtendedError {
    fn from(error: gpu_db_protocol::FrontendMessageError) -> Self {
        Self::new("08P01", error.to_string())
    }
}

#[derive(Debug, Default)]
pub(crate) struct ExtendedSession {
    statements: HashMap<String, Statement>,
    sql_statements: HashMap<String, SqlPreparedPlan>,
    sql_cursors: HashMap<String, SqlCursor>,
    portals: HashMap<String, Portal>,
    skip_until_sync: bool,
    implicit_transaction: bool,
}

#[derive(Debug, Clone)]
struct SqlCursor {
    columns: Vec<ColumnMeta>,
    rows: Vec<Vec<DbValue>>,
    position: usize,
    transaction_bound: bool,
}

impl ExtendedSession {
    pub(crate) fn ensure_sql_cursor_name_available(&self, name: &str) -> Result<(), ExtendedError> {
        if self.sql_cursors.contains_key(name) {
            return Err(ExtendedError::new("42P03", "cursor already exists"));
        }
        Ok(())
    }

    pub(crate) fn install_sql_cursor(
        &mut self,
        name: String,
        outcome: QueryOutcome,
        transaction_bound: bool,
    ) -> Result<(), ExtendedError> {
        self.ensure_sql_cursor_name_available(&name)?;
        let QueryOutcome::Rows { columns, rows } = outcome else {
            return Err(ExtendedError::new(
                "0A000",
                "cursor declarations require a row-producing SELECT",
            ));
        };
        self.sql_cursors.insert(
            name,
            SqlCursor {
                columns,
                rows,
                position: 0,
                transaction_bound,
            },
        );
        Ok(())
    }

    pub(crate) fn fetch_sql_cursor(
        &mut self,
        name: &str,
        count: Option<usize>,
    ) -> Result<QueryOutcome, ExtendedError> {
        let cursor = self
            .sql_cursors
            .get_mut(name)
            .ok_or_else(|| ExtendedError::new("34000", "cursor does not exist"))?;
        let start = cursor.position;
        let end = count
            .map(|count| start.saturating_add(count).min(cursor.rows.len()))
            .unwrap_or(cursor.rows.len());
        cursor.position = end;
        Ok(QueryOutcome::Returning {
            tag: CommandTag::Other("FETCH".to_string()),
            columns: cursor.columns.clone(),
            rows: cursor.rows[start..end].to_vec(),
            rows_affected: (end - start) as u64,
        })
    }

    pub(crate) fn move_sql_cursor(
        &mut self,
        name: &str,
        count: Option<usize>,
    ) -> Result<QueryOutcome, ExtendedError> {
        let cursor = self
            .sql_cursors
            .get_mut(name)
            .ok_or_else(|| ExtendedError::new("34000", "cursor does not exist"))?;
        let start = cursor.position;
        let end = count
            .map(|count| start.saturating_add(count).min(cursor.rows.len()))
            .unwrap_or(cursor.rows.len());
        cursor.position = end;
        Ok(QueryOutcome::Command {
            tag: CommandTag::Other(format!("MOVE {}", end - start)),
            rows_affected: None,
        })
    }

    pub(crate) fn close_sql_cursor(
        &mut self,
        target: crate::sql_cursor::SqlCursorCloseTarget,
    ) -> Result<(), ExtendedError> {
        match target {
            crate::sql_cursor::SqlCursorCloseTarget::All => self.sql_cursors.clear(),
            crate::sql_cursor::SqlCursorCloseTarget::Named(name) => {
                if self.sql_cursors.remove(&name).is_none() {
                    return Err(ExtendedError::new("34000", "cursor does not exist"));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn analyze_sql_prepare(
        engine: &SharedEngine,
        session: &SharedSession,
        query: &str,
        parameter_hints: &[Option<LogicalType>],
    ) -> Result<PreparedStatement, DbError> {
        engine.prepare_statement(session, query, parameter_hints)
    }

    pub(crate) fn analyze_sql_prepare_plan(
        engine: &SharedEngine,
        session: &SharedSession,
        query: &str,
        parameter_hints: &[Option<LogicalType>],
    ) -> Result<SqlPreparedPlan, DbError> {
        match Self::analyze_sql_prepare(engine, session, query, parameter_hints) {
            Ok(prepared) => Ok(SqlPreparedPlan::ready(prepared)),
            Err(error) => {
                let Some(analysis_query) = deferred_sql_prepare_analysis_query(query, &error)
                else {
                    return Err(error);
                };
                let prepared =
                    Self::analyze_sql_prepare(engine, session, &analysis_query, parameter_hints)?;
                Ok(SqlPreparedPlan {
                    prepared,
                    deferred_execution_error: Some(error),
                })
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn install_sql_prepared(
        &mut self,
        name: String,
        prepared: PreparedStatement,
    ) -> Result<(), ExtendedError> {
        self.install_sql_prepared_plan(name, SqlPreparedPlan::ready(prepared))
    }

    pub(crate) fn install_sql_prepared_plan(
        &mut self,
        name: String,
        plan: SqlPreparedPlan,
    ) -> Result<(), ExtendedError> {
        if self.sql_statements.contains_key(&name) {
            return Err(ExtendedError::new(
                "42P05",
                "prepared statement already exists",
            ));
        }
        self.sql_statements.insert(name, plan);
        Ok(())
    }

    pub(crate) fn has_sql_prepared(&self, name: &str) -> bool {
        self.sql_statements.contains_key(name)
    }

    pub(crate) fn sql_prepared_plan(&self, name: &str) -> Result<SqlPreparedPlan, ExtendedError> {
        self.sql_statements.get(name).cloned().ok_or_else(|| {
            ExtendedError::new(
                "26000",
                format!("prepared statement \"{name}\" does not exist"),
            )
        })
    }

    pub(crate) fn deallocate_sql_prepared(
        &mut self,
        target: crate::sql_prepared::SqlDeallocateTarget,
    ) -> Result<(), ExtendedError> {
        match target {
            crate::sql_prepared::SqlDeallocateTarget::All => self.sql_statements.clear(),
            crate::sql_prepared::SqlDeallocateTarget::Named(name) => {
                if self.sql_statements.remove(&name).is_none() {
                    return Err(ExtendedError::new(
                        "26000",
                        format!("prepared statement \"{name}\" does not exist"),
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn has_implicit_transaction(&self) -> bool {
        self.implicit_transaction
    }

    /// Canonical message dispatch for all server ingresses. Socket I/O and blocking-engine
    /// scheduling stay in the ingress; lifecycle decisions and response encoding live only here.
    /// The async ingress still calls response encoding on its runtime task until SCALE-001 adds
    /// bounded streaming/offload.
    pub(crate) fn dispatch(
        &mut self,
        message: FrontendMessage,
        transaction_status: SessionTransactionStatus,
    ) -> Dispatch {
        match message {
            FrontendMessage::SimpleQuery(sql) => Dispatch::SimpleQuery(sql),
            FrontendMessage::Execute {
                portal_name,
                max_rows,
            } => self.prepare_execute_dispatch(portal_name, max_rows, transaction_status),
            FrontendMessage::Sync => Dispatch::Sync,
            FrontendMessage::Terminate => Dispatch::Terminate,
            FrontendMessage::Parse {
                statement_name,
                query,
                parameter_type_oids,
            } => match self.prepare_request(
                statement_name,
                query,
                &parameter_type_oids,
                transaction_status,
            ) {
                Ok(request) => Dispatch::Prepare(Box::new(request)),
                Err(error) => Dispatch::Response(Err(error)),
            },
            FrontendMessage::Bind {
                portal_name,
                statement_name,
                parameter_format_codes,
                parameters,
                result_format_codes,
            } => match self.prepare_bind_request(
                portal_name,
                statement_name,
                parameter_format_codes,
                parameters,
                result_format_codes,
                transaction_status,
            ) {
                Ok(request) => Dispatch::Bind(Box::new(request)),
                Err(error) => Dispatch::Response(Err(error)),
            },
            FrontendMessage::Describe { target, name } => Dispatch::Describe { target, name },
            FrontendMessage::Close { target, name } => {
                Dispatch::Response(self.close(target, &name))
            }
            FrontendMessage::Flush => Dispatch::Response(Ok(Vec::new())),
            FrontendMessage::PasswordMessage(_) => Dispatch::Response(Err(ExtendedError::new(
                "08P01",
                "password messages are not supported after startup",
            ))),
            FrontendMessage::SaslInitialResponse { .. } | FrontendMessage::SaslResponse(_) => {
                Dispatch::Response(Err(ExtendedError::new(
                    "08P01",
                    "SASL authentication is only supported during startup",
                )))
            }
            _ => Dispatch::Response(Err(ExtendedError::new(
                "0A000",
                "frontend message is not supported by the engine-backed server",
            ))),
        }
    }

    fn prepare_execute_dispatch(
        &self,
        portal_name: String,
        max_rows: u32,
        transaction_status: SessionTransactionStatus,
    ) -> Dispatch {
        let Some(portal) = self.portals.get(&portal_name) else {
            return Dispatch::Response(Err(ExtendedError::new(
                "34000",
                format!("portal \"{portal_name}\" does not exist"),
            )));
        };
        if transaction_status == SessionTransactionStatus::FailedTransaction
            && !portal.transaction_exit
            && (!portal.bound.is_empty() || portal.copy.is_some())
        {
            return Dispatch::Response(Err(ExtendedError::in_failed_transaction()));
        }
        Dispatch::Execute {
            portal_name,
            max_rows,
        }
    }

    /// Classify a frame before decoding its payload. Once an extended-cycle error occurs,
    /// PostgreSQL discards every queued message through the next Sync; malformed discarded
    /// messages must not manufacture extra ErrorResponses. The terminating Sync frame is decoded
    /// after it ends the skip state, so a malformed Sync adds a protocol ErrorResponse before Ready.
    pub(crate) fn skipping_frame_action(&self, tag: u8) -> SkippingFrameAction {
        if !self.skip_until_sync {
            return SkippingFrameAction::Parse;
        }
        match tag {
            b'S' if self.implicit_transaction => SkippingFrameAction::RollbackAndSync,
            b'S' => SkippingFrameAction::Sync,
            b'X' => SkippingFrameAction::Terminate,
            _ => SkippingFrameAction::Discard,
        }
    }

    /// PostgreSQL treats Query and Sync as non-extended messages: a malformed frame gets one
    /// ErrorResponse followed immediately by ReadyForQuery. Malformed extended messages enter
    /// ignore-till-Sync instead. A synthetic implicit cycle must roll back before that Ready.
    pub(crate) fn malformed_frame_action(&self, tag: u8) -> MalformedFrameAction {
        if matches!(tag, b'Q' | b'S') {
            MalformedFrameAction::ErrorAndReady(if self.implicit_transaction {
                TransactionAction::RollbackImplicit
            } else {
                TransactionAction::None
            })
        } else {
            MalformedFrameAction::SkipUntilSync
        }
    }

    pub(crate) fn before_dispatch_transaction_action(
        &self,
        message: &FrontendMessage,
        status: SessionTransactionStatus,
    ) -> TransactionAction {
        if !self.implicit_transaction
            && status == SessionTransactionStatus::Idle
            && matches!(
                message,
                FrontendMessage::Parse { .. }
                    | FrontendMessage::Bind { .. }
                    | FrontendMessage::Describe { .. }
                    | FrontendMessage::Execute { .. }
                    | FrontendMessage::Close { .. }
            )
        {
            TransactionAction::BeginImplicit
        } else {
            TransactionAction::None
        }
    }

    pub(crate) fn sync_transaction_action(&self) -> TransactionAction {
        if self.implicit_transaction {
            TransactionAction::CommitImplicit
        } else {
            TransactionAction::None
        }
    }

    pub(crate) fn simple_query_completion_action(
        &mut self,
        outcome: &Result<QueryOutcome, DbError>,
    ) -> TransactionAction {
        match outcome {
            Ok(QueryOutcome::Command {
                tag: CommandTag::Commit | CommandTag::Rollback,
                ..
            }) => {
                self.implicit_transaction = false;
                self.portals.clear();
                TransactionAction::None
            }
            Ok(QueryOutcome::Command {
                tag: CommandTag::Begin,
                ..
            }) => {
                self.implicit_transaction = false;
                TransactionAction::None
            }
            Ok(_) if self.implicit_transaction => TransactionAction::CommitImplicit,
            Err(_) if self.implicit_transaction => TransactionAction::RollbackImplicit,
            Ok(_) | Err(_) => TransactionAction::None,
        }
    }

    pub(crate) fn complete_transaction_action(
        &mut self,
        action: TransactionAction,
        succeeded: bool,
    ) {
        if !succeeded {
            return;
        }
        match action {
            TransactionAction::None => {}
            TransactionAction::BeginImplicit => self.implicit_transaction = true,
            TransactionAction::CommitImplicit | TransactionAction::RollbackImplicit => {
                self.implicit_transaction = false;
                self.portals.clear();
            }
        }
    }

    pub(crate) fn fail(&mut self) {
        self.skip_until_sync = true;
    }

    pub(crate) fn sync(&mut self, transaction_open: bool) {
        self.skip_until_sync = false;
        self.finish_transaction_boundary(transaction_open);
    }

    pub(crate) fn complete_skipped_sync_frame(
        &mut self,
        frame: &[u8],
        transaction_open: bool,
    ) -> Result<(), ExtendedError> {
        if self.implicit_transaction {
            self.implicit_transaction = false;
            self.portals.clear();
        }
        self.sync(transaction_open);
        match parse_frontend_message(frame) {
            Ok(FrontendMessage::Sync) => Ok(()),
            Ok(_) => Err(ExtendedError::new(
                "08P01",
                "skipped Sync frame decoded as a different frontend message",
            )),
            Err(error) => Err(ExtendedError::new("08P01", error.to_string())),
        }
    }

    pub(crate) fn finish_transaction_boundary(&mut self, transaction_open: bool) {
        if !transaction_open {
            self.portals.clear();
            self.sql_cursors
                .retain(|_, cursor| !cursor.transaction_bound);
        }
    }

    pub(crate) fn clear_unnamed_for_simple_query(&mut self) {
        self.statements.remove("");
        self.portals.remove("");
    }

    #[cfg(test)]
    pub(crate) fn parse(
        &mut self,
        statement_name: String,
        query: &str,
        parameter_type_oids: &[u32],
        prepare: impl FnOnce(&str, &[Option<LogicalType>]) -> Result<PreparedStatement, DbError>,
    ) -> Result<Vec<u8>, ExtendedError> {
        let request = self.prepare_request(
            statement_name,
            query.to_string(),
            parameter_type_oids,
            SessionTransactionStatus::Idle,
        )?;
        let prepared =
            prepare(&request.query, &request.parameter_type_hints).and_then(|prepared| {
                let parameter_types = prepared
                    .parameter_types()
                    .ok_or_else(|| DbError {
                        category: ErrorCategory::Internal,
                        message: "test Parse analysis lost its parameter types".to_string(),
                    })?
                    .to_vec();
                Ok(PrepareAnalysis {
                    prepared,
                    parameter_types,
                    copy_target: None,
                })
            });
        self.complete_parse(request, prepared)
    }

    fn prepare_request(
        &mut self,
        statement_name: String,
        query: String,
        parameter_type_oids: &[u32],
        transaction_status: SessionTransactionStatus,
    ) -> Result<PrepareRequest, ExtendedError> {
        if statement_name.is_empty() {
            self.statements.remove("");
        }
        // Parse syntax before the failed-transaction gate. PostgreSQL still reports malformed SQL
        // in an aborted transaction, and an empty Parse is permitted there.
        let copy = match classify_copy_statement(&query) {
            CopyClassification::NotCopy => None,
            CopyClassification::Supported(copy) => Some(copy),
            CopyClassification::Unsupported => {
                return Err(ExtendedError::new(
                    "0A000",
                    "COPY statement is not supported by the canonical product server",
                ));
            }
        };
        if copy.is_some() && !parameter_type_oids.is_empty() {
            return Err(ExtendedError::new(
                "08P01",
                "COPY Parse message has too many parameter type OIDs",
            ));
        }
        let mut cursor_declaration = None;
        let analysis_sql = match &copy {
            Some(CopyStatement::From(_)) => String::new(),
            Some(CopyStatement::To(copy)) => copy_to_validation_sql(copy),
            None => match classify_sql_cursor_statement(&query)? {
                Some(SqlCursorAction::Declare {
                    name,
                    query: cursor_query,
                }) => {
                    let transaction_bound = !self.implicit_transaction
                        && transaction_status != SessionTransactionStatus::Idle;
                    cursor_declaration = Some(ExtendedCursorDeclaration {
                        name,
                        transaction_bound,
                    });
                    cursor_query
                }
                Some(
                    SqlCursorAction::Fetch { .. }
                    | SqlCursorAction::Move { .. }
                    | SqlCursorAction::Close(_),
                )
                | None => query.clone(),
            },
        };
        // The lexical gates deliberately return early for ordinary SQL. Preserve the historic
        // extended-Parse diagnostic precedence by proving quote/comment/dollar lexical validity
        // before semantic parameter-OID mapping. COPY owns its syntax proof above; SQL EXECUTE
        // and ordinary shape/name/plan parsing intentionally retain their old post-OID order.
        if copy.is_none() {
            canonicalize_sql_for_exact_match(&analysis_sql)
                .map_err(|error| ExtendedError::new("42601", error.to_string()))?;
        }
        let parameter_type_hints = parameter_type_oids
            .iter()
            .map(|oid| {
                if *oid == 0 {
                    Ok(None)
                } else {
                    pg_adapter::logical_type_from_oid(*oid)
                        .map(Some)
                        .map_err(|error| ExtendedError::new("0A000", error.to_string()))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let parameter_type_hints = fill_unused_parameter_holes(&analysis_sql, parameter_type_hints);
        let extended_sql_execute = classify_extended_sql_execute(&analysis_sql)?;
        let (parsed, sql_execute_bind_arguments, outer_parameter_types, deferred_execution_error) =
            if let Some(extended_sql_execute) = extended_sql_execute {
                let plan = self.sql_prepared_plan(&extended_sql_execute.name)?;
                let (sql_execute_bind_arguments, outer_parameter_types) =
                    sql_execute_bind_arguments(
                        &plan.prepared,
                        &extended_sql_execute.arguments,
                        &parameter_type_hints,
                    )?;
                (
                    plan.prepared,
                    Some(sql_execute_bind_arguments),
                    Some(outer_parameter_types),
                    plan.deferred_execution_error,
                )
            } else {
                let parsed = PreparedStatement::parse(&analysis_sql)
                    .map_err(extended_prepared_parse_error)?;
                if cursor_declaration.is_some() && parsed.parameter_count() > 0 {
                    return Err(ExtendedError::new(
                        "0A000",
                        "parameterized cursor declarations are not supported",
                    ));
                }
                (parsed, None, None, None)
            };
        if transaction_status == SessionTransactionStatus::FailedTransaction
            && (copy.is_some() || !parsed.is_empty())
            && !parsed.is_transaction_exit()
        {
            return Err(ExtendedError::in_failed_transaction());
        }
        Ok(PrepareRequest {
            statement_name,
            parsed,
            copy,
            sql_execute_bind_arguments,
            outer_parameter_types,
            deferred_execution_error,
            cursor_declaration,
            #[cfg(test)]
            query,
            parameter_type_hints,
        })
    }

    /// Perform effect-free catalog analysis for one already syntax-classified Parse request.
    /// COPY FROM obtains an opaque target proof here; COPY TO describes its synthesized SELECT
    /// template.  Neither path starts wire COPY mode or admits a mutation.
    pub(crate) fn analyze_prepare(
        engine: &SharedEngine,
        session: &mut SharedSession,
        request: &PrepareRequest,
    ) -> Result<PrepareAnalysis, DbError> {
        let prepared = if request.sql_execute_bind_arguments.is_some() {
            engine
                .revalidate_prepared_description(session, &request.parsed)
                .map_err(extended_relational_error)?;
            request.parsed.clone()
        } else {
            engine
                .describe_prepared_statement(
                    session,
                    request.parsed.clone(),
                    &request.parameter_type_hints,
                )
                .map_err(extended_relational_error)?
        };
        let parameter_types = if let Some(outer_parameter_types) = &request.outer_parameter_types {
            outer_parameter_types.clone()
        } else {
            prepared
                .parameter_types()
                .ok_or_else(|| DbError {
                    category: ErrorCategory::Internal,
                    message: "prepared Parse analysis lost its parameter types".to_string(),
                })?
                .to_vec()
        };
        let copy_target = match &request.copy {
            Some(CopyStatement::From(copy)) => {
                let outcome = engine
                    .submit(session, SubmissionRequest::CopyFromStart(copy))
                    .into_immediate()?;
                let QueryOutcome::CopyIn { target } = outcome else {
                    return Err(DbError {
                        category: ErrorCategory::Internal,
                        message: "COPY FROM Parse analysis returned the wrong facade outcome"
                            .to_string(),
                    });
                };
                Some(target)
            }
            Some(CopyStatement::To(_)) | None => None,
        };
        Ok(PrepareAnalysis {
            prepared,
            parameter_types,
            copy_target,
        })
    }

    pub(crate) fn complete_parse(
        &mut self,
        request: PrepareRequest,
        analysis: Result<PrepareAnalysis, DbError>,
    ) -> Result<Vec<u8>, ExtendedError> {
        let PrepareRequest {
            statement_name,
            parsed: _,
            copy,
            sql_execute_bind_arguments,
            outer_parameter_types: _,
            deferred_execution_error,
            cursor_declaration,
            #[cfg(test)]
                query: _,
            parameter_type_hints: _,
        } = request;
        let PrepareAnalysis {
            prepared,
            parameter_types,
            copy_target,
        } = analysis.map_err(ExtendedError::from)?;
        let parameter_oids = parameter_types
            .iter()
            .copied()
            .map(pg_adapter::logical_type_oid)
            .collect();
        let described_columns = prepared
            .result_columns()
            .ok_or_else(|| ExtendedError::new("XX000", "prepared statement was not described"))?
            .to_vec();
        let columns = if copy.is_some() || cursor_declaration.is_some() {
            Vec::new()
        } else {
            described_columns
        };
        if matches!(copy, Some(CopyStatement::From(_))) && copy_target.is_none() {
            return Err(ExtendedError::new(
                "XX000",
                "COPY FROM Parse analysis did not retain a target proof",
            ));
        }
        // A named duplicate is checked only after syntax parsing and catalog analysis have
        // succeeded, matching PostgreSQL's store-prepared-plan point.
        if !statement_name.is_empty() && self.statements.contains_key(&statement_name) {
            return Err(ExtendedError::new(
                "42P05",
                format!("prepared statement \"{statement_name}\" already exists"),
            ));
        }
        self.statements.insert(
            statement_name,
            Statement {
                prepared,
                sql_execute_bind_arguments,
                deferred_execution_error,
                cursor_declaration,
                copy,
                copy_target,
                parameter_oids,
                columns,
            },
        );
        let mut buf = Vec::new();
        BackendWriter::new(&mut buf).parse_complete()?;
        Ok(buf)
    }

    fn prepare_bind_request(
        &mut self,
        portal_name: String,
        statement_name: String,
        parameter_format_codes: Vec<i16>,
        parameters: Vec<Option<Vec<u8>>>,
        result_format_codes: Vec<i16>,
        transaction_status: SessionTransactionStatus,
    ) -> Result<BindRequest, ExtendedError> {
        let statement = self
            .statements
            .get(&statement_name)
            .cloned()
            .ok_or_else(|| {
                ExtendedError::new(
                    "26000",
                    format!("prepared statement \"{statement_name}\" does not exist"),
                )
            })?;
        validate_bind_shape(&statement, &parameter_format_codes, &parameters)?;
        if transaction_status == SessionTransactionStatus::FailedTransaction
            && (statement.copy.is_some()
                || !statement.prepared.is_transaction_exit()
                || !parameters.is_empty())
        {
            return Err(ExtendedError::in_failed_transaction());
        }
        // CreatePortal follows statement lookup, arity/format-count checks, and the aborted-state
        // gate. Only from this point may a replacement destroy the unnamed portal.
        if portal_name.is_empty() {
            self.portals.remove("");
        } else if self.portals.contains_key(&portal_name) {
            return Err(ExtendedError::new(
                "42P03",
                format!("portal \"{portal_name}\" already exists"),
            ));
        }
        Ok(BindRequest {
            portal_name,
            statement_name,
            statement,
            parameter_format_codes,
            parameters,
            result_format_codes,
        })
    }

    pub(crate) fn bind_request(request: BindRequest) -> Result<BindCompletion, ExtendedError> {
        let BindRequest {
            portal_name,
            statement_name,
            statement,
            parameter_format_codes,
            parameters,
            result_format_codes,
        } = request;
        let transaction_exit = statement.prepared.is_transaction_exit();
        let deferred_execution_error = statement.deferred_execution_error.clone();
        let cursor_declaration = statement.cursor_declaration.clone();
        let copy = statement.copy.clone();
        let copy_target = statement.copy_target.clone();
        let params = parameters
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let format = format_at(&parameter_format_codes, index);
                pg_adapter::decode_parameter(
                    statement.parameter_oids[index],
                    format,
                    value.as_deref(),
                )
                .map_err(|error| {
                    parameter_codec_error(error, statement.parameter_oids[index], value.as_deref())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Result formats are applied after CreatePortal and parameter decoding. Their arity must
        // not outrank failed state, a named-portal duplicate, or a parameter codec error.
        if !statement.columns.is_empty() {
            validate_format_count("result", result_format_codes.len(), statement.columns.len())?;
        }
        validate_result_formats(&statement.columns, &result_format_codes)?;
        let bound_parameters = match &statement.sql_execute_bind_arguments {
            Some(arguments) => arguments
                .iter()
                .map(|argument| match argument {
                    SqlExecuteBindArgument::OuterParameter(index) => {
                        params.get(index.saturating_sub(1)).cloned().ok_or_else(|| {
                            ExtendedError::new(
                                "XX000",
                                "extended SQL EXECUTE lost a validated outer parameter",
                            )
                        })
                    }
                    SqlExecuteBindArgument::Literal(value) => Ok(value.clone()),
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => params,
        };
        let bound = statement
            .prepared
            .bind_values(&bound_parameters)
            .map_err(ExtendedError::from)?;
        Ok(BindCompletion {
            portal_name,
            portal: Portal {
                statement_name,
                transaction_exit,
                bound,
                deferred_execution_error,
                cursor_declaration,
                copy,
                copy_target,
                columns: statement.columns,
                result_formats: result_format_codes,
                outcome: None,
                row_offset: 0,
                command_completed: false,
            },
        })
    }

    pub(crate) fn complete_bind(
        &mut self,
        completion: Result<BindCompletion, ExtendedError>,
    ) -> Result<Vec<u8>, ExtendedError> {
        let BindCompletion {
            portal_name,
            portal,
        } = completion?;
        if portal_name.is_empty() {
            self.portals.remove("");
        } else if self.portals.contains_key(&portal_name) {
            return Err(ExtendedError::new(
                "42P03",
                format!("portal \"{portal_name}\" already exists"),
            ));
        }
        self.portals.insert(portal_name, portal);
        let mut buf = Vec::new();
        BackendWriter::new(&mut buf).bind_complete()?;
        Ok(buf)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind(
        &mut self,
        portal_name: String,
        statement_name: &str,
        parameter_format_codes: &[i16],
        parameters: &[Option<Vec<u8>>],
        result_format_codes: &[i16],
    ) -> Result<Vec<u8>, ExtendedError> {
        let request = self.prepare_bind_request(
            portal_name,
            statement_name.to_string(),
            parameter_format_codes.to_vec(),
            parameters.to_vec(),
            result_format_codes.to_vec(),
            SessionTransactionStatus::Idle,
        )?;
        let completion = Self::bind_request(request);
        self.complete_bind(completion)
    }

    pub(crate) fn describe(
        &self,
        target: DescribeTarget,
        name: &str,
        transaction_status: SessionTransactionStatus,
    ) -> Result<Vec<u8>, ExtendedError> {
        let mut buf = Vec::new();
        let mut writer = BackendWriter::new(&mut buf);
        match target {
            DescribeTarget::Statement => {
                let statement = self.statements.get(name).ok_or_else(|| {
                    ExtendedError::new(
                        "26000",
                        format!("prepared statement \"{name}\" does not exist"),
                    )
                })?;
                if transaction_status == SessionTransactionStatus::FailedTransaction
                    && !statement.columns.is_empty()
                {
                    return Err(ExtendedError::in_failed_transaction());
                }
                writer.parameter_description(&statement.parameter_oids)?;
                write_description(&mut writer, &statement.columns, &[])?;
            }
            DescribeTarget::Portal => {
                let portal = self.portals.get(name).ok_or_else(|| {
                    ExtendedError::new("34000", format!("portal \"{name}\" does not exist"))
                })?;
                if transaction_status == SessionTransactionStatus::FailedTransaction
                    && !portal.columns.is_empty()
                {
                    return Err(ExtendedError::in_failed_transaction());
                }
                write_description(&mut writer, &portal.columns, &portal.result_formats)?;
            }
        }
        Ok(buf)
    }

    pub(crate) fn describe_revalidated(
        &self,
        engine: &SharedEngine,
        session: &SharedSession,
        target: DescribeTarget,
        name: &str,
        transaction_status: SessionTransactionStatus,
    ) -> Result<Vec<u8>, ExtendedError> {
        let owner = self.description_owner(target, name, transaction_status)?;
        Self::revalidate_description_owner(engine, session, owner)?;
        self.describe(target, name, transaction_status)
    }

    pub(crate) fn description_owner(
        &self,
        target: DescribeTarget,
        name: &str,
        transaction_status: SessionTransactionStatus,
    ) -> Result<DescriptionOwner, ExtendedError> {
        match target {
            DescribeTarget::Statement => {
                let statement = self.statements.get(name).ok_or_else(|| {
                    ExtendedError::new(
                        "26000",
                        format!("prepared statement \"{name}\" does not exist"),
                    )
                })?;
                if transaction_status == SessionTransactionStatus::FailedTransaction
                    && !statement.columns.is_empty()
                {
                    return Err(ExtendedError::in_failed_transaction());
                }
                if transaction_status == SessionTransactionStatus::FailedTransaction {
                    return Ok(DescriptionOwner::Cached);
                }
                if matches!(statement.copy, Some(CopyStatement::From(_))) {
                    return statement
                        .copy_target
                        .clone()
                        .map(DescriptionOwner::Copy)
                        .ok_or_else(|| {
                            ExtendedError::new("XX000", "COPY FROM statement lost its target proof")
                        });
                }
                Ok(DescriptionOwner::Statement(Box::new(
                    statement.prepared.clone(),
                )))
            }
            DescribeTarget::Portal => {
                let portal = self.portals.get(name).ok_or_else(|| {
                    ExtendedError::new("34000", format!("portal \"{name}\" does not exist"))
                })?;
                if transaction_status == SessionTransactionStatus::FailedTransaction
                    && !portal.columns.is_empty()
                {
                    return Err(ExtendedError::in_failed_transaction());
                }
                if transaction_status == SessionTransactionStatus::FailedTransaction {
                    return Ok(DescriptionOwner::Cached);
                }
                if matches!(portal.copy, Some(CopyStatement::From(_))) {
                    return portal
                        .copy_target
                        .clone()
                        .map(DescriptionOwner::Copy)
                        .ok_or_else(|| {
                            ExtendedError::new("XX000", "COPY FROM portal lost its target proof")
                        });
                }
                Ok(DescriptionOwner::Portal(Box::new(portal.bound.clone())))
            }
        }
    }

    pub(crate) fn revalidate_description_owner(
        engine: &SharedEngine,
        session: &SharedSession,
        owner: DescriptionOwner,
    ) -> Result<(), ExtendedError> {
        match owner {
            DescriptionOwner::Cached => Ok(()),
            DescriptionOwner::Copy(target) => engine
                .revalidate_copy_target(session, &target)
                .map_err(ExtendedError::from),
            DescriptionOwner::Statement(prepared) => engine
                .revalidate_prepared_description(session, &prepared)
                .map_err(ExtendedError::from),
            DescriptionOwner::Portal(bound) => engine
                .revalidate_bound_description(session, &bound)
                .map_err(ExtendedError::from),
        }
    }

    pub(crate) fn close(
        &mut self,
        target: DescribeTarget,
        name: &str,
    ) -> Result<Vec<u8>, ExtendedError> {
        match target {
            DescribeTarget::Statement => {
                self.statements.remove(name);
                self.portals
                    .retain(|_, portal| portal.statement_name != name);
            }
            DescribeTarget::Portal => {
                self.portals.remove(name);
            }
        }
        let mut buf = Vec::new();
        BackendWriter::new(&mut buf).close_complete()?;
        Ok(buf)
    }

    pub(crate) fn execution_request(
        &self,
        portal_name: &str,
    ) -> Result<Option<ExecutionRequest>, ExtendedError> {
        let portal = self.portals.get(portal_name).ok_or_else(|| {
            ExtendedError::new("34000", format!("portal \"{portal_name}\" does not exist"))
        })?;
        if portal.command_completed && portal.copy.is_some() {
            return Err(ExtendedError::new(
                "55000",
                format!("portal \"{portal_name}\" cannot be run"),
            ));
        }
        if portal.command_completed {
            return Ok(None);
        }
        if let Some(cursor) = &portal.cursor_declaration {
            self.ensure_sql_cursor_name_available(&cursor.name)?;
        }
        if let Some(error) = &portal.deferred_execution_error {
            return Err(error.clone().into());
        }
        Ok(match &portal.copy {
            Some(copy) => Some(ExecutionRequest::Copy {
                statement: copy.clone(),
                target: portal.copy_target.clone(),
                bound: Box::new(portal.bound.clone()),
            }),
            None if portal.outcome.is_none() => {
                Some(ExecutionRequest::Query(Box::new(portal.bound.clone())))
            }
            None => None,
        })
    }

    pub(crate) fn complete_copy_execute(&mut self, portal_name: &str) -> Result<(), ExtendedError> {
        let portal = self.portals.get_mut(portal_name).ok_or_else(|| {
            ExtendedError::new("34000", format!("portal \"{portal_name}\" does not exist"))
        })?;
        if portal.copy.is_none() {
            return Err(ExtendedError::new(
                "XX000",
                "non-COPY portal entered COPY completion",
            ));
        }
        portal.command_completed = true;
        Ok(())
    }

    pub(crate) fn portal_ends_transaction(&self, portal_name: &str) -> bool {
        self.portals
            .get(portal_name)
            .is_some_and(|portal| portal.transaction_exit)
    }

    /// Whether a cached portal result can still be replaced by a cancellation error before any
    /// bytes are published to the client. Rows and empty outcomes are effect-free. Any error may
    /// represent an indeterminate post-durable failure, while a Command or RETURNING outcome may
    /// represent an already-published mutation; neither may be falsely relabelled.
    pub(crate) fn portal_outcome_can_be_cancelled(
        &self,
        portal_name: &str,
    ) -> Result<bool, ExtendedError> {
        let portal = self.portals.get(portal_name).ok_or_else(|| {
            ExtendedError::new("34000", format!("portal \"{portal_name}\" does not exist"))
        })?;
        Ok(matches!(
            portal.outcome.as_ref(),
            Some(Ok(QueryOutcome::Rows { .. } | QueryOutcome::Empty))
        ))
    }

    pub(crate) fn set_execution_outcome(
        &mut self,
        portal_name: &str,
        outcome: Result<QueryOutcome, DbError>,
    ) -> Result<(), ExtendedError> {
        let cursor_declaration = self
            .portals
            .get(portal_name)
            .ok_or_else(|| {
                ExtendedError::new("34000", format!("portal \"{portal_name}\" does not exist"))
            })?
            .cursor_declaration
            .clone();
        let outcome = match (cursor_declaration, outcome) {
            (Some(cursor), Ok(outcome)) => {
                self.install_sql_cursor(cursor.name, outcome, cursor.transaction_bound)?;
                Ok(QueryOutcome::Command {
                    tag: CommandTag::Other("DECLARE CURSOR".to_string()),
                    rows_affected: None,
                })
            }
            (_, outcome) => outcome,
        };
        if matches!(
            &outcome,
            Ok(QueryOutcome::Command {
                tag: CommandTag::Begin | CommandTag::Commit | CommandTag::Rollback,
                ..
            })
        ) {
            self.implicit_transaction = false;
        }
        let portal = self.portals.get_mut(portal_name).ok_or_else(|| {
            ExtendedError::new("34000", format!("portal \"{portal_name}\" does not exist"))
        })?;
        if portal.copy.is_some() {
            return Err(ExtendedError::new(
                "XX000",
                "COPY portal must be encoded by the COPY wire owner",
            ));
        }
        if portal.outcome.is_none() {
            portal.outcome = Some(outcome);
        }
        Ok(())
    }

    pub(crate) fn encode_execute(
        &mut self,
        portal_name: &str,
        max_rows: u32,
    ) -> Result<Vec<u8>, ExtendedError> {
        let portal = self.portals.get_mut(portal_name).ok_or_else(|| {
            ExtendedError::new("34000", format!("portal \"{portal_name}\" does not exist"))
        })?;
        let outcome = portal
            .outcome
            .as_ref()
            .ok_or_else(|| ExtendedError::new("XX000", "portal has not been executed"))?;
        let outcome = outcome
            .as_ref()
            .map_err(|error| ExtendedError::from(error.clone()))?;
        if matches!(outcome, QueryOutcome::CopyIn { .. }) {
            return Err(ExtendedError::new(
                "XX000",
                "COPY start outcome reached the ordinary portal encoder",
            ));
        }
        if portal.command_completed && matches!(outcome, QueryOutcome::Command { .. }) {
            return Err(ExtendedError::new(
                "55000",
                format!("portal \"{portal_name}\" cannot be run"),
            ));
        }
        let mut buf = Vec::new();
        let mut writer = BackendWriter::new(&mut buf);
        let rows = match outcome {
            QueryOutcome::Rows { rows, .. } | QueryOutcome::Returning { rows, .. } => Some(rows),
            _ => None,
        };
        let mut rows_this_execute = None;
        if let Some(rows) = rows {
            let remaining = rows.len().saturating_sub(portal.row_offset);
            let take = if max_rows == 0 {
                remaining
            } else {
                remaining.min(max_rows as usize)
            };
            for row in &rows[portal.row_offset..portal.row_offset + take] {
                let values = row
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        pg_adapter::encode_result_value(
                            value,
                            format_at(&portal.result_formats, index),
                        )
                        .map_err(|error| codec_error(error, "result"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                write_data_row(&mut writer, &values)?;
            }
            portal.row_offset += take;
            rows_this_execute = Some(take);
            // PostgreSQL suspends when Execute consumes exactly max_rows because the executor has
            // not probed EOF yet. A subsequent Execute observes zero remaining rows and completes
            // with a zero-row tag.
            if max_rows > 0 && take == max_rows as usize {
                writer.portal_suspended()?;
                return Ok(buf);
            }
        }
        match outcome {
            QueryOutcome::Empty => writer.empty_query_response()?,
            _ => {
                writer.command_complete(&portal_command_complete_tag(outcome, rows_this_execute))?
            }
        }
        if matches!(outcome, QueryOutcome::Command { .. }) {
            portal.command_completed = true;
        }
        Ok(buf)
    }
}

fn portal_command_complete_tag(outcome: &QueryOutcome, rows_this_execute: Option<usize>) -> String {
    match (outcome, rows_this_execute) {
        (QueryOutcome::Rows { .. }, Some(rows)) => format!("SELECT {rows}"),
        (
            QueryOutcome::Returning {
                tag,
                columns,
                rows: _,
                rows_affected: _,
            },
            Some(rows),
        ) => pg_adapter::command_complete_tag(&QueryOutcome::Returning {
            tag: tag.clone(),
            columns: columns.clone(),
            rows: Vec::new(),
            rows_affected: rows as u64,
        }),
        _ => pg_adapter::command_complete_tag(outcome),
    }
}

fn validate_format_count(
    kind: &str,
    format_count: usize,
    value_count: usize,
) -> Result<(), ExtendedError> {
    if matches!(format_count, 0 | 1) || format_count == value_count {
        Ok(())
    } else {
        Err(ExtendedError::new(
            "08P01",
            format!("Bind {kind} format count does not match its value count"),
        ))
    }
}

fn validate_bind_shape(
    statement: &Statement,
    parameter_formats: &[i16],
    parameters: &[Option<Vec<u8>>],
) -> Result<(), ExtendedError> {
    if parameters.len() != statement.parameter_oids.len() {
        return Err(ExtendedError::new(
            "08P01",
            "bind message has wrong number of parameters",
        ));
    }
    validate_format_count("parameter", parameter_formats.len(), parameters.len())
}

fn sql_execute_bind_arguments(
    target: &PreparedStatement,
    arguments: &[ExtendedSqlExecuteArgument],
    supplied: &[Option<LogicalType>],
) -> Result<(Vec<SqlExecuteBindArgument>, Vec<LogicalType>), ExtendedError> {
    let target_parameter_types = target
        .parameter_types()
        .ok_or_else(|| ExtendedError::new("XX000", "SQL prepared statement was not described"))?;
    if target_parameter_types.len() != arguments.len() {
        return Err(ExtendedError::new(
            "08P01",
            "bound parameter count does not match prepared statement",
        ));
    }
    let parameter_count = arguments
        .iter()
        .filter_map(|argument| match argument {
            ExtendedSqlExecuteArgument::OuterParameter { index, .. } => Some(*index),
            ExtendedSqlExecuteArgument::Literal(_) => None,
        })
        .max()
        .unwrap_or_default()
        .max(supplied.len());
    let mut outer_parameter_types = vec![None; parameter_count];
    for (slot, hint) in outer_parameter_types
        .iter_mut()
        .zip(supplied.iter().copied())
    {
        *slot = hint;
    }
    let mut bind_arguments = Vec::with_capacity(arguments.len());
    for (target_type, argument) in target_parameter_types.iter().copied().zip(arguments) {
        match argument {
            ExtendedSqlExecuteArgument::OuterParameter {
                index: outer_index,
                explicit_type,
            } => {
                if explicit_type.is_some_and(|explicit_type| explicit_type != target_type) {
                    return Err(ExtendedError::new(
                        "0A000",
                        format!(
                            "SQL EXECUTE argument cast {explicit_type:?} does not match prepared \
                             parameter type {target_type:?}"
                        ),
                    ));
                }
                let Some(slot) = outer_parameter_types.get_mut(outer_index.saturating_sub(1))
                else {
                    return Err(ExtendedError::new(
                        "XX000",
                        "extended SQL EXECUTE lost a validated outer parameter",
                    ));
                };
                match *slot {
                    Some(existing) if existing != target_type => {
                        return Err(ExtendedError::new(
                            "42P08",
                            "inconsistent parameter types for SQL EXECUTE placeholder",
                        ));
                    }
                    _ => *slot = Some(target_type),
                }
                bind_arguments.push(SqlExecuteBindArgument::OuterParameter(*outer_index));
            }
            ExtendedSqlExecuteArgument::Literal(value) => {
                let literal = decode_sql_execute_literal(target_type, value)
                    .map_err(sql_execute_literal_error)?;
                bind_arguments.push(SqlExecuteBindArgument::Literal(literal));
            }
        }
    }
    let outer_parameter_types = outer_parameter_types
        .into_iter()
        .enumerate()
        .map(|(index, logical_type)| {
            logical_type.ok_or_else(|| {
                ExtendedError::new(
                    "42P18",
                    format!("could not determine data type of parameter ${}", index + 1),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((bind_arguments, outer_parameter_types))
}

fn sql_execute_literal_error(error: DbError) -> ExtendedError {
    if error.category == ErrorCategory::InvalidRequest {
        return ExtendedError::new("22P02", error.message);
    }
    ExtendedError::from(error)
}

fn extended_prepared_parse_error(error: DbError) -> ExtendedError {
    let error = extended_relational_error(error);
    if error.message == "invalid SQL parameter reference" {
        return ExtendedError::new("42P02", "there is no parameter $0");
    }
    ExtendedError::from(error)
}

/// Parse and catalog-description failures share one extended-query compatibility boundary.  The
/// simple-query route keeps the facade's generic diagnostic; an extended relational SELECT must
/// retain the frozen protocol diagnostic before a statement or portal can be installed.
fn extended_relational_error(error: DbError) -> DbError {
    if error.message.starts_with("invalid relational SQL syntax;")
        || error.message == "query shape is not supported by the compatibility stub"
    {
        DbError {
            category: ErrorCategory::Unsupported,
            message: "extended query protocol only supports relational SELECT".to_string(),
        }
    } else {
        error
    }
}

fn validate_result_formats(columns: &[ColumnMeta], formats: &[i16]) -> Result<(), ExtendedError> {
    for (index, _column) in columns.iter().enumerate() {
        let format = format_at(formats, index);
        if !matches!(format, 0 | 1) {
            return Err(ExtendedError::new(
                "22023",
                format!("unsupported result format code {format}"),
            ));
        }
    }
    Ok(())
}

fn format_at(formats: &[i16], index: usize) -> i16 {
    match formats {
        [] => 0,
        [format] => *format,
        _ => formats[index],
    }
}

fn codec_error(error: pg_adapter::PgValueCodecError, kind: &str) -> ExtendedError {
    let code = match &error {
        pg_adapter::PgValueCodecError::InvalidValue { format: 1, .. } => "22P03",
        pg_adapter::PgValueCodecError::InvalidValue { .. } => "22P02",
        pg_adapter::PgValueCodecError::NumericValueOutOfRange { .. } => "22003",
        pg_adapter::PgValueCodecError::UnsupportedFormat(_) => "22023",
        _ => "0A000",
    };
    ExtendedError::new(code, format!("invalid {kind}: {error}"))
}

fn parameter_codec_error(
    error: pg_adapter::PgValueCodecError,
    oid: u32,
    value: Option<&[u8]>,
) -> ExtendedError {
    if matches!(
        &error,
        pg_adapter::PgValueCodecError::InvalidValue { format: 0, .. }
    ) {
        let value = value.map_or_else(String::new, |raw| String::from_utf8_lossy(raw).into_owned());
        return ExtendedError::new(
            "22P02",
            format!("invalid input syntax for parameter type oid {oid}: {value:?}"),
        );
    }
    codec_error(error, "parameter")
}

fn backend_columns(columns: &[ColumnMeta]) -> Vec<BackendColumn> {
    columns
        .iter()
        .map(|column| {
            BackendColumn::new(
                column.name.clone(),
                pg_adapter::logical_type_oid(column.logical_type),
                pg_adapter::logical_type_size(column.logical_type),
            )
            .with_type_modifier(pg_adapter::logical_type_typmod(
                column.logical_type,
                column.numeric_typmod,
            ))
        })
        .collect()
}

fn copy_to_validation_sql(copy: &CopyToStdout) -> String {
    let projection = copy
        .columns
        .as_ref()
        .map_or_else(|| "*".to_string(), |columns| columns.join(", "));
    format!("SELECT {projection} FROM {}", copy.table)
}

fn write_description<W: io::Write + ?Sized>(
    writer: &mut BackendWriter<'_, W>,
    columns: &[ColumnMeta],
    formats: &[i16],
) -> io::Result<()> {
    if columns.is_empty() {
        writer.no_data()
    } else {
        writer.row_description_with_formats(&backend_columns(columns), formats)
    }
}

fn write_data_row<W: io::Write + ?Sized>(
    writer: &mut BackendWriter<'_, W>,
    values: &[Option<Vec<u8>>],
) -> io::Result<()> {
    let count = i16::try_from(values.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many row values"))?;
    let mut payload = Vec::new();
    payload.extend_from_slice(&count.to_be_bytes());
    for value in values {
        match value {
            Some(value) => {
                let len = i32::try_from(value.len()).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "row value is too large")
                })?;
                payload.extend_from_slice(&len.to_be_bytes());
                payload.extend_from_slice(value);
            }
            None => payload.extend_from_slice(&(-1_i32).to_be_bytes()),
        }
    }
    writer.message(b'D', &payload)
}

impl From<io::Error> for ExtendedError {
    fn from(error: io::Error) -> Self {
        Self::new("XX000", error.to_string())
    }
}

#[cfg(test)]
mod tests;
