//! Canonical COPY wire state over the protocol-neutral facade.
//!
//! This module owns only pgwire framing, text/CSV decoding, and connection-local COPY state. It
//! never owns relational rows: COPY FROM completion submits typed values through
//! `SharedEngine::submit`, and COPY TO reads the session's exact engine snapshot through that same
//! facade. WAL, global order, apply, and publication remain engine-owned.

use std::io::{self, Write};
use std::net::TcpStream;
use std::sync::Arc;

use gpu_db_facade::{
    pg_adapter, BoundPreparedStatement, CommandTag, CopyColumnMeta, CopyTarget, DbError, DbValue,
    ErrorCategory, LogicalType, QueryOutcome, SessionTransactionStatus, SharedEngine,
    SharedSession, SubmissionRequest,
};
use gpu_db_protocol::backend::{BackendError, BackendWriter};
use gpu_db_protocol::{
    is_copy_statement, parse_copy_from_stdin, parse_copy_row, parse_copy_to_stdout_table,
    parse_frontend_message, CopyColumn, CopyFormat, CopyFromStdin, CopyOptions, CopyParseError,
    CopyToStdout, FrontendMessage, SqlType, SqlValue,
};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream as TokioTcpStream;

use super::extended::ExtendedSession;
use super::{
    complete_simple_query_action_async, complete_simple_query_action_blocking, encode_ready,
    shared_session_transaction_status,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CopyStatement {
    From(CopyFromStdin),
    To(CopyToStdout),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CopyClassification {
    NotCopy,
    Supported(CopyStatement),
    Unsupported,
}

pub(crate) fn classify_copy_statement(sql: &str) -> CopyClassification {
    if let Some(copy) = parse_copy_from_stdin(sql) {
        CopyClassification::Supported(CopyStatement::From(copy))
    } else if let Some(copy) = parse_copy_to_stdout_table(sql) {
        CopyClassification::Supported(CopyStatement::To(copy))
    } else if is_copy_statement(sql) {
        CopyClassification::Unsupported
    } else {
        CopyClassification::NotCopy
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CopyWireError {
    pub code: &'static str,
    pub message: String,
}

impl CopyWireError {
    pub(crate) fn unsupported() -> Self {
        Self {
            code: "0A000",
            message: "COPY statement is not supported by the canonical product server".to_string(),
        }
    }

    pub(crate) fn protocol(message: impl Into<String>) -> Self {
        Self {
            code: "08P01",
            message: message.into(),
        }
    }

    pub(crate) fn client_aborted() -> Self {
        Self {
            code: "57014",
            message: "COPY from stdin was aborted by the client".to_string(),
        }
    }
}

impl From<DbError> for CopyWireError {
    fn from(error: DbError) -> Self {
        Self {
            code: pg_adapter::error_sqlstate(error.category),
            message: error.message,
        }
    }
}

impl From<CopyParseError> for CopyWireError {
    fn from(error: CopyParseError) -> Self {
        Self {
            code: error.postgres_code(),
            message: error.postgres_message().to_string(),
        }
    }
}

pub(crate) struct CopyInState {
    copy: CopyFromStdin,
    target: CopyTarget,
    columns: Vec<CopyColumn>,
    pending_bytes: Vec<u8>,
    pending_rows: Vec<Vec<DbValue>>,
    seen_terminator: bool,
    ready_after_done: bool,
}

impl CopyInState {
    pub(crate) fn ready_after_done(&self) -> bool {
        self.ready_after_done
    }
}

pub(crate) fn begin_copy_from(
    engine: &SharedEngine,
    session: &mut SharedSession,
    copy: CopyFromStdin,
    ready_after_done: bool,
) -> Result<(CopyInState, Vec<u8>), CopyWireError> {
    let outcome = engine
        .submit(session, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .map_err(CopyWireError::from)?;
    let QueryOutcome::CopyIn { target } = outcome else {
        return Err(CopyWireError {
            code: "XX000",
            message: "COPY FROM description returned the wrong facade outcome".to_string(),
        });
    };
    begin_prepared_copy_from(engine, session, copy, target, ready_after_done)
}

/// Enter COPY mode from an extended-protocol target already analyzed during Parse.  Revalidation
/// happens before CopyInResponse, while final CopyDone repeats the proof under mutation admission.
pub(crate) fn begin_prepared_copy_from(
    engine: &SharedEngine,
    session: &mut SharedSession,
    mut copy: CopyFromStdin,
    target: CopyTarget,
    ready_after_done: bool,
) -> Result<(CopyInState, Vec<u8>), CopyWireError> {
    engine
        .revalidate_copy_target(session, &target)
        .map_err(CopyWireError::from)?;
    let columns = target
        .columns()
        .iter()
        .map(copy_column)
        .collect::<Result<Vec<_>, _>>()?;
    // Keep the exact validated target order with the buffered rows. This also makes COPY without
    // an explicit column list correct for a table created privately in the same transaction.
    copy.columns = Some(columns.iter().map(|column| column.name.clone()).collect());
    let mut response = Vec::new();
    BackendWriter::new(&mut response).copy_in_response(columns.len())?;
    Ok((
        CopyInState {
            copy,
            target,
            columns,
            pending_bytes: Vec::new(),
            pending_rows: Vec::new(),
            seen_terminator: false,
            ready_after_done,
        },
        response,
    ))
}

pub(crate) fn append_copy_data(state: &mut CopyInState, bytes: &[u8]) -> Result<(), CopyWireError> {
    state.pending_bytes.extend_from_slice(bytes);
    while let Some(newline) = state.pending_bytes.iter().position(|byte| *byte == b'\n') {
        let mut line = state.pending_bytes.drain(..=newline).collect::<Vec<_>>();
        line.pop();
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        let line = std::str::from_utf8(&line).map_err(|_| CopyParseError::InvalidUtf8)?;
        if state.copy.options.format == CopyFormat::Text && line == r"\." {
            state.seen_terminator = true;
            continue;
        }
        if state.seen_terminator {
            return Err(CopyWireError {
                code: "22P04",
                message: "COPY data follows the end-of-data marker".to_string(),
            });
        }
        if state.copy.options.header {
            state.copy.options.header = false;
            continue;
        }
        let columns = state
            .copy
            .columns
            .as_deref()
            .expect("COPY start always fixes target columns");
        let row = parse_copy_row(&state.columns, columns, state.copy.options, line)?;
        state
            .pending_rows
            .push(row.into_iter().map(db_value).collect());
    }
    Ok(())
}

pub(crate) fn finish_copy_from(
    engine: &SharedEngine,
    session: &mut SharedSession,
    state: CopyInState,
) -> Result<QueryOutcome, CopyWireError> {
    if !state.pending_bytes.is_empty() {
        std::str::from_utf8(&state.pending_bytes).map_err(|_| CopyParseError::InvalidUtf8)?;
        return Err(CopyWireError {
            code: "22P04",
            message: "COPY data ended before row terminator".to_string(),
        });
    }
    engine
        .submit(
            session,
            SubmissionRequest::CopyFrom {
                target: &state.target,
                rows: state.pending_rows,
            },
        )
        .into_immediate()
        .map_err(CopyWireError::from)
}

pub(crate) fn execute_copy_to(
    engine: &SharedEngine,
    session: &mut SharedSession,
    copy: &CopyToStdout,
) -> Result<Vec<u8>, CopyWireError> {
    let outcome = engine
        .submit(session, SubmissionRequest::CopyTo(copy))
        .into_immediate()
        .map_err(CopyWireError::from)?;
    encode_copy_to(outcome, copy.options)
}

/// Execute extended-protocol COPY TO through the exact bound prepared owner produced at Bind.
/// The prepared facade path revalidates catalog dependencies and result columns before this
/// function emits CopyOutResponse, so a stale portal cannot silently adopt a new table shape.
pub(crate) fn execute_prepared_copy_to(
    engine: &SharedEngine,
    session: &mut SharedSession,
    bound: &BoundPreparedStatement,
    copy: &CopyToStdout,
) -> Result<Vec<u8>, CopyWireError> {
    let outcome = engine
        .submit(session, SubmissionRequest::PreparedCopyTo { copy, bound })
        .into_immediate()
        .map_err(CopyWireError::from)?;
    encode_copy_to(outcome, copy.options)
}

fn encode_copy_to(outcome: QueryOutcome, options: CopyOptions) -> Result<Vec<u8>, CopyWireError> {
    let QueryOutcome::Rows { columns, rows } = outcome else {
        return Err(CopyWireError {
            code: "XX000",
            message: "COPY TO returned the wrong facade outcome".to_string(),
        });
    };
    let mut response = Vec::new();
    let mut writer = BackendWriter::new(&mut response);
    writer.copy_out_response(columns.len())?;
    if options.header {
        let mut payload = String::new();
        for (index, column) in columns.iter().enumerate() {
            if index > 0 {
                payload.push(options.delimiter);
            }
            payload.push_str(&copy_csv_text(
                &column.name,
                options.delimiter,
                options.quote,
                options.escape,
                false,
            ));
        }
        payload.push('\n');
        writer.copy_data(payload.as_bytes())?;
    }
    for row in &rows {
        let mut payload = String::new();
        for (index, value) in row.iter().enumerate() {
            if index > 0 {
                payload.push(match options.format {
                    CopyFormat::Text => '\t',
                    CopyFormat::Csv => options.delimiter,
                });
            }
            match options.format {
                CopyFormat::Text => payload.push_str(&copy_text_value(value)),
                CopyFormat::Csv => payload.push_str(&copy_csv_value(value, options)),
            }
        }
        payload.push('\n');
        writer.copy_data(payload.as_bytes())?;
    }
    writer.copy_done()?;
    writer.command_complete(&format!("COPY {}", rows.len()))?;
    Ok(response)
}

pub(crate) fn encode_copy_completion(outcome: &QueryOutcome) -> Result<Vec<u8>, CopyWireError> {
    let mut response = Vec::new();
    BackendWriter::new(&mut response)
        .command_complete(&pg_adapter::command_complete_tag(outcome))?;
    Ok(response)
}

pub(crate) fn encode_copy_error(error: &CopyWireError) -> Vec<u8> {
    let mut response = Vec::new();
    let _ = BackendWriter::new(&mut response)
        .error_response(&BackendError::new(error.code, error.message.clone()));
    response
}

fn copy_column(column: &CopyColumnMeta) -> Result<CopyColumn, CopyWireError> {
    let ty = match column.logical_type {
        LogicalType::Int2 => SqlType::Int2,
        LogicalType::Int4 => SqlType::Int4,
        LogicalType::Int8 => SqlType::Int8,
        LogicalType::Numeric => {
            let Some((precision, scale)) = column.numeric_typmod else {
                return Err(CopyWireError {
                    code: "XX000",
                    message: "numeric COPY column is missing its neutral typmod".to_string(),
                });
            };
            SqlType::Numeric { precision, scale }
        }
        LogicalType::Bool => SqlType::Bool,
        LogicalType::Text => SqlType::Text,
        LogicalType::Date => SqlType::Date,
        LogicalType::Timestamp => SqlType::Timestamp,
        LogicalType::Uuid => SqlType::Uuid,
    };
    Ok(CopyColumn {
        name: column.name.clone(),
        ty,
    })
}

fn db_value(value: SqlValue) -> DbValue {
    match value {
        SqlValue::Null => DbValue::Null,
        SqlValue::Int2(value) => DbValue::Int2(value),
        SqlValue::Int4(value) => DbValue::Int4(value),
        SqlValue::Int8(value) => DbValue::Int8(value),
        SqlValue::Numeric(value) => DbValue::Numeric(value),
        SqlValue::Bool(value) => DbValue::Bool(value),
        SqlValue::Text(value) => DbValue::Text(value),
        SqlValue::Date(value) => DbValue::Date(value),
        SqlValue::Timestamp(value) => DbValue::Timestamp(value),
        SqlValue::Uuid(value) => DbValue::Uuid(value),
        SqlValue::Parameter { .. } => unreachable!("COPY row decoding never creates parameters"),
    }
}

fn copy_text_value(value: &DbValue) -> String {
    let Some(text) = pg_adapter::db_value_text_opt(value) else {
        return r"\N".to_string();
    };
    text.replace('\\', r"\\")
        .replace('\t', r"\t")
        .replace('\n', r"\n")
        .replace('\r', r"\r")
}

fn copy_csv_value(value: &DbValue, options: CopyOptions) -> String {
    let Some(text) = pg_adapter::db_value_text_opt(value) else {
        return String::new();
    };
    copy_csv_text(
        &text,
        options.delimiter,
        options.quote,
        options.escape,
        text.is_empty(),
    )
}

fn copy_csv_text(
    text: &str,
    delimiter: char,
    quote: char,
    escape: char,
    force_quote: bool,
) -> String {
    if force_quote || text.contains([delimiter, quote, escape, '\n', '\r']) {
        let mut escaped = String::with_capacity(text.len());
        for ch in text.chars() {
            if ch == quote || ch == escape {
                escaped.push(escape);
            }
            escaped.push(ch);
        }
        format!("{quote}{escaped}{quote}")
    } else {
        text.to_string()
    }
}

impl From<io::Error> for CopyWireError {
    fn from(error: io::Error) -> Self {
        Self {
            code: "XX000",
            message: error.to_string(),
        }
    }
}

pub(crate) fn copy_lifecycle_success() -> QueryOutcome {
    QueryOutcome::Command {
        tag: CommandTag::Copy,
        rows_affected: None,
    }
}

pub(crate) fn copy_lifecycle_error(error: &CopyWireError) -> DbError {
    DbError {
        category: ErrorCategory::Engine,
        message: error.message.clone(),
    }
}

pub(crate) fn handle_copy_frame_blocking(
    stream: &mut TcpStream,
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    copy_in: &mut Option<CopyInState>,
    frame: &[u8],
) -> Result<bool, String> {
    let message = match parse_frontend_message(frame) {
        Ok(message) => message,
        Err(error) => {
            let state = copy_in.take().expect("COPY frame requires active state");
            write_copy_terminal_blocking(
                stream,
                engine,
                session,
                extended,
                state,
                Err(CopyWireError::protocol(error.to_string())),
            )?;
            return Ok(true);
        }
    };
    match message {
        FrontendMessage::CopyData(bytes) => {
            let result = append_copy_data(
                copy_in.as_mut().expect("COPY data requires active state"),
                &bytes,
            );
            if let Err(error) = result {
                let state = copy_in.take().expect("COPY parse error retains state");
                write_copy_terminal_blocking(stream, engine, session, extended, state, Err(error))?;
            }
        }
        FrontendMessage::CopyDone => {
            let state = copy_in.take().expect("COPY done requires active state");
            let ready_after_done = state.ready_after_done();
            let result = finish_copy_from(engine, session, state);
            write_copy_terminal_result_blocking(
                stream,
                engine,
                session,
                extended,
                ready_after_done,
                result,
            )?;
        }
        FrontendMessage::CopyFail(_) => {
            let state = copy_in.take().expect("COPY fail requires active state");
            write_copy_terminal_blocking(
                stream,
                engine,
                session,
                extended,
                state,
                Err(CopyWireError::client_aborted()),
            )?;
        }
        FrontendMessage::Flush => stream.flush().map_err(|error| error.to_string())?,
        // Extended clients pipeline the Execute-cycle Sync before COPY data. Ignore it here; the
        // completion Sync follows CopyDone/CopyFail and owns ReadyForQuery.
        FrontendMessage::Sync => {}
        FrontendMessage::Terminate => {
            copy_in.take();
            return Ok(false);
        }
        _ => {
            let state = copy_in.take().expect("COPY protocol error retains state");
            write_copy_terminal_blocking(
                stream,
                engine,
                session,
                extended,
                state,
                Err(CopyWireError::protocol(
                    "only CopyData, CopyDone, CopyFail, Flush, Sync, or Terminate is allowed during COPY FROM STDIN",
                )),
            )?;
        }
    }
    Ok(true)
}

fn write_copy_terminal_blocking(
    stream: &mut TcpStream,
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    state: CopyInState,
    result: Result<QueryOutcome, CopyWireError>,
) -> Result<(), String> {
    let ready_after_done = state.ready_after_done();
    write_copy_terminal_result_blocking(stream, engine, session, extended, ready_after_done, result)
}

fn write_copy_terminal_result_blocking(
    stream: &mut TcpStream,
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    ready_after_done: bool,
    result: Result<QueryOutcome, CopyWireError>,
) -> Result<(), String> {
    let lifecycle = result
        .as_ref()
        .map(|_| copy_lifecycle_success())
        .map_err(copy_lifecycle_error);
    if result.is_err() {
        session.mark_transaction_failed();
    }
    let mut response = match result {
        Ok(outcome) => encode_copy_completion(&outcome).map_err(|error| error.message)?,
        Err(error) => encode_copy_error(&error),
    };
    if ready_after_done {
        complete_simple_query_action_blocking(engine, session, extended, &lifecycle, &mut response)
            .map_err(|error| error.to_string())?;
        let status = session.transaction_status();
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
    } else if lifecycle.is_err() {
        extended.fail();
    }
    stream
        .write_all(&response)
        .map_err(|error| error.to_string())
}

pub(crate) fn mark_shared_session_failed(session: &Arc<std::sync::Mutex<SharedSession>>) {
    session
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .mark_transaction_failed();
}

pub(crate) async fn begin_copy_from_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    copy: CopyFromStdin,
    ready_after_done: bool,
) -> Result<Result<(CopyInState, Vec<u8>), CopyWireError>, String> {
    let _permit = executor
        .acquire()
        .await
        .map_err(|error| error.to_string())?;
    tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        begin_copy_from(&engine, &mut session, copy, ready_after_done)
    })
    .await
    .map_err(|error| error.to_string())
}

pub(crate) async fn begin_prepared_copy_from_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    copy: CopyFromStdin,
    target: CopyTarget,
    ready_after_done: bool,
) -> Result<Result<(CopyInState, Vec<u8>), CopyWireError>, String> {
    let _permit = executor
        .acquire()
        .await
        .map_err(|error| error.to_string())?;
    tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        begin_prepared_copy_from(&engine, &mut session, copy, target, ready_after_done)
    })
    .await
    .map_err(|error| error.to_string())
}

pub(crate) async fn execute_copy_to_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    copy: CopyToStdout,
) -> Result<Result<Vec<u8>, CopyWireError>, String> {
    let _permit = executor
        .acquire()
        .await
        .map_err(|error| error.to_string())?;
    tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        execute_copy_to(&engine, &mut session, &copy)
    })
    .await
    .map_err(|error| error.to_string())
}

pub(crate) async fn execute_prepared_copy_to_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    bound: BoundPreparedStatement,
    copy: CopyToStdout,
) -> Result<Result<Vec<u8>, CopyWireError>, String> {
    let _permit = executor
        .acquire()
        .await
        .map_err(|error| error.to_string())?;
    tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        execute_prepared_copy_to(&engine, &mut session, &bound, &copy)
    })
    .await
    .map_err(|error| error.to_string())
}

async fn finish_copy_from_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    state: CopyInState,
) -> Result<Result<QueryOutcome, CopyWireError>, String> {
    let _permit = executor
        .acquire()
        .await
        .map_err(|error| error.to_string())?;
    tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        finish_copy_from(&engine, &mut session, state)
    })
    .await
    .map_err(|error| error.to_string())
}

async fn append_copy_data_async(
    executor: &Arc<tokio::sync::Semaphore>,
    mut state: CopyInState,
    bytes: Vec<u8>,
) -> Result<(CopyInState, Result<(), CopyWireError>), String> {
    let _permit = executor
        .acquire()
        .await
        .map_err(|error| error.to_string())?;
    tokio::task::spawn_blocking(move || {
        let result = append_copy_data(&mut state, &bytes);
        (state, result)
    })
    .await
    .map_err(|error| error.to_string())
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_copy_frame_async(
    stream: &mut TokioTcpStream,
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    extended: &mut ExtendedSession,
    copy_in: &mut Option<CopyInState>,
    frame: Vec<u8>,
) -> Result<bool, String> {
    let message = {
        let _permit = executor
            .acquire()
            .await
            .map_err(|error| error.to_string())?;
        tokio::task::spawn_blocking(move || parse_frontend_message(&frame))
            .await
            .map_err(|error| error.to_string())?
    };
    let message = match message {
        Ok(message) => message,
        Err(error) => {
            let state = copy_in.take().expect("COPY frame requires active state");
            write_copy_terminal_async(
                stream,
                engine,
                session,
                executor,
                extended,
                state,
                Err(CopyWireError::protocol(error.to_string())),
            )
            .await?;
            return Ok(true);
        }
    };
    match message {
        FrontendMessage::CopyData(bytes) => {
            let state = copy_in.take().expect("COPY data requires active state");
            let (state, result) = append_copy_data_async(executor, state, bytes).await?;
            match result {
                Ok(()) => *copy_in = Some(state),
                Err(error) => {
                    write_copy_terminal_async(
                        stream,
                        engine,
                        session,
                        executor,
                        extended,
                        state,
                        Err(error),
                    )
                    .await?;
                }
            }
        }
        FrontendMessage::CopyDone => {
            let state = copy_in.take().expect("COPY done requires active state");
            let ready_after_done = state.ready_after_done();
            let result =
                finish_copy_from_async(Arc::clone(&engine), Arc::clone(&session), executor, state)
                    .await?;
            write_copy_terminal_result_async(
                stream,
                engine,
                session,
                executor,
                extended,
                ready_after_done,
                result,
            )
            .await?;
        }
        FrontendMessage::CopyFail(_) => {
            let state = copy_in.take().expect("COPY fail requires active state");
            write_copy_terminal_async(
                stream,
                engine,
                session,
                executor,
                extended,
                state,
                Err(CopyWireError::client_aborted()),
            )
            .await?;
        }
        FrontendMessage::Flush => stream.flush().await.map_err(|error| error.to_string())?,
        // Extended clients pipeline the Execute-cycle Sync before COPY data. Ignore it here; the
        // completion Sync follows CopyDone/CopyFail and owns ReadyForQuery.
        FrontendMessage::Sync => {}
        FrontendMessage::Terminate => {
            copy_in.take();
            return Ok(false);
        }
        _ => {
            let state = copy_in.take().expect("COPY protocol error retains state");
            write_copy_terminal_async(
                stream,
                engine,
                session,
                executor,
                extended,
                state,
                Err(CopyWireError::protocol(
                    "only CopyData, CopyDone, CopyFail, Flush, Sync, or Terminate is allowed during COPY FROM STDIN",
                )),
            )
            .await?;
        }
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
async fn write_copy_terminal_async(
    stream: &mut TokioTcpStream,
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    extended: &mut ExtendedSession,
    state: CopyInState,
    result: Result<QueryOutcome, CopyWireError>,
) -> Result<(), String> {
    let ready_after_done = state.ready_after_done();
    write_copy_terminal_result_async(
        stream,
        engine,
        session,
        executor,
        extended,
        ready_after_done,
        result,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn write_copy_terminal_result_async(
    stream: &mut TokioTcpStream,
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    extended: &mut ExtendedSession,
    ready_after_done: bool,
    result: Result<QueryOutcome, CopyWireError>,
) -> Result<(), String> {
    let lifecycle = result
        .as_ref()
        .map(|_| copy_lifecycle_success())
        .map_err(copy_lifecycle_error);
    if result.is_err() {
        mark_shared_session_failed(&session);
    }
    let mut response = match result {
        Ok(outcome) => encode_copy_completion(&outcome).map_err(|error| error.message)?,
        Err(error) => encode_copy_error(&error),
    };
    if ready_after_done {
        complete_simple_query_action_async(
            engine,
            Arc::clone(&session),
            executor,
            extended,
            &lifecycle,
            &mut response,
        )
        .await?;
        let status = shared_session_transaction_status(&session);
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
    } else if lifecycle.is_err() {
        extended.fail();
    }
    stream
        .write_all(&response)
        .await
        .map_err(|error| error.to_string())
}
