//! One PostgreSQL extended-query lifecycle shared by every engine-backed server ingress.

use std::collections::HashMap;
use std::io;

use gpu_db_facade::{
    pg_adapter, BoundPreparedStatement, ColumnMeta, CommandTag, DbError, LogicalType,
    PreparedStatement, QueryOutcome, SessionTransactionStatus, SharedEngine, SharedSession,
};
use gpu_db_protocol::backend::{BackendColumn, BackendError, BackendWriter};
use gpu_db_protocol::{parse_frontend_message, DescribeTarget, FrontendMessage};

#[derive(Debug, Clone)]
struct Statement {
    prepared: PreparedStatement,
    parameter_oids: Vec<u32>,
    columns: Vec<ColumnMeta>,
}

#[derive(Debug, Clone)]
struct Portal {
    statement_name: String,
    transaction_exit: bool,
    bound: BoundPreparedStatement,
    columns: Vec<ColumnMeta>,
    result_formats: Vec<i16>,
    outcome: Option<Result<QueryOutcome, DbError>>,
    row_offset: usize,
    command_completed: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ExecutionRequest {
    pub bound: BoundPreparedStatement,
}

#[derive(Debug, Clone)]
pub(crate) struct PrepareRequest {
    pub statement_name: String,
    pub parsed: PreparedStatement,
    #[cfg(test)]
    pub query: String,
    pub parameter_type_hints: Vec<Option<LogicalType>>,
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
    fn new(code: &'static str, message: impl Into<String>) -> Self {
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
    portals: HashMap<String, Portal>,
    skip_until_sync: bool,
    implicit_transaction: bool,
}

impl ExtendedSession {
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
            && !portal.bound.is_empty()
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
        let prepared = prepare(&request.query, &request.parameter_type_hints);
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
        let parsed = PreparedStatement::parse(&query).map_err(ExtendedError::from)?;
        if transaction_status == SessionTransactionStatus::FailedTransaction
            && !parsed.is_empty()
            && !parsed.is_transaction_exit()
        {
            return Err(ExtendedError::in_failed_transaction());
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
        Ok(PrepareRequest {
            statement_name,
            parsed,
            #[cfg(test)]
            query,
            parameter_type_hints,
        })
    }

    pub(crate) fn complete_parse(
        &mut self,
        request: PrepareRequest,
        prepared: Result<PreparedStatement, DbError>,
    ) -> Result<Vec<u8>, ExtendedError> {
        let PrepareRequest {
            statement_name,
            parsed: _,
            #[cfg(test)]
                query: _,
            parameter_type_hints: _,
        } = request;
        let prepared = prepared.map_err(ExtendedError::from)?;
        let parameter_types = prepared
            .parameter_types()
            .ok_or_else(|| ExtendedError::new("XX000", "prepared statement was not described"))?;
        let parameter_oids = parameter_types
            .iter()
            .copied()
            .map(pg_adapter::logical_type_oid)
            .collect();
        let columns = prepared
            .result_columns()
            .ok_or_else(|| ExtendedError::new("XX000", "prepared statement was not described"))?
            .to_vec();
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
            && (!statement.prepared.is_transaction_exit() || !parameters.is_empty())
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
                .map_err(|error| codec_error(error, "parameter"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Result formats are applied after CreatePortal and parameter decoding. Their arity must
        // not outrank failed state, a named-portal duplicate, or a parameter codec error.
        if !statement.columns.is_empty() {
            validate_format_count("result", result_format_codes.len(), statement.columns.len())?;
        }
        validate_result_formats(&statement.columns, &result_format_codes)?;
        let bound = statement
            .prepared
            .bind_values(&params)
            .map_err(ExtendedError::from)?;
        Ok(BindCompletion {
            portal_name,
            portal: Portal {
                statement_name,
                transaction_exit,
                bound,
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
        Ok(portal.outcome.is_none().then(|| ExecutionRequest {
            bound: portal.bound.clone(),
        }))
    }

    pub(crate) fn portal_ends_transaction(&self, portal_name: &str) -> bool {
        self.portals
            .get(portal_name)
            .is_some_and(|portal| portal.transaction_exit)
    }

    pub(crate) fn set_execution_outcome(
        &mut self,
        portal_name: &str,
        outcome: Result<QueryOutcome, DbError>,
    ) -> Result<(), ExtendedError> {
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
            format!(
                "Bind supplies {} parameters, but prepared statement requires {}",
                parameters.len(),
                statement.parameter_oids.len()
            ),
        ));
    }
    validate_format_count("parameter", parameter_formats.len(), parameters.len())
}

fn validate_result_formats(columns: &[ColumnMeta], formats: &[i16]) -> Result<(), ExtendedError> {
    for (index, column) in columns.iter().enumerate() {
        let format = format_at(formats, index);
        if !matches!(format, 0 | 1) {
            return Err(ExtendedError::new(
                "22023",
                format!("unsupported result format code {format}"),
            ));
        }
        if format == 1
            && !matches!(
                column.logical_type,
                LogicalType::Int2
                    | LogicalType::Int4
                    | LogicalType::Int8
                    | LogicalType::Text
                    | LogicalType::Uuid
            )
        {
            return Err(ExtendedError::new(
                "0A000",
                format!(
                    "binary result encoding is not implemented for {:?}",
                    column.logical_type
                ),
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
        pg_adapter::PgValueCodecError::UnsupportedFormat(_) => "22023",
        _ => "0A000",
    };
    ExtendedError::new(code, format!("invalid {kind}: {error}"))
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
        })
        .collect()
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
