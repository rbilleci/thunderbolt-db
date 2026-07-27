//! Canonical COPY wire state over the protocol-neutral facade.
//!
//! This module owns only pgwire framing, text/CSV decoding, and connection-local COPY state. It
//! never owns relational rows: COPY FROM completion submits typed values through
//! `SharedEngine::submit`, and COPY TO reads the session's exact engine snapshot through that same
//! facade. WAL, global order, apply, and publication remain engine-owned.

use std::io;
use std::sync::Arc;

use gpu_db_facade::{
    pg_adapter, BoundPreparedStatement, CommandTag, CopyColumnMeta, CopyTarget, DbError, DbValue,
    ErrorCategory, LogicalType, QueryOutcome, SessionTransactionStatus, SharedEngine,
    SharedSession, SubmissionRequest,
};
use gpu_db_protocol::backend::{BackendError, BackendWriter};
use gpu_db_protocol::{
    is_copy_statement, parse_copy_from_stdin, parse_copy_row, parse_copy_to_stdout_table,
    parse_frontend_message, sql_may_start_with_any_keyword, CopyColumn, CopyFormat, CopyFromStdin,
    CopyOptions, CopyParseError, CopyToStdout, FrontendMessage, SqlType, SqlValue,
};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream as TokioTcpStream;

use crate::cancellation::{cancel_effect_free_success, ActiveRequest};
use crate::transport::ReadWrite;

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
    if !compat_classifier_gate(sql, &["COPY"]) {
        return CopyClassification::NotCopy;
    }
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

fn compat_classifier_gate(sql: &str, candidates: &[&str]) -> bool {
    let admitted = sql_may_start_with_any_keyword(sql, candidates);
    #[cfg(feature = "probe-timing")]
    crate::insert_probe::record_compat_classifier_gate(sql.len() as u64, admitted);
    admitted
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

    pub(crate) fn query_cancelled() -> Self {
        Self {
            code: "57014",
            message: "canceling statement due to user request".to_string(),
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
        // PostgreSQL archive COPY members end with `\.\n\n\n`; pg_restore forwards that member
        // through CopyData and then sends CopyDone. The two separator lines are archive framing,
        // not rows. Preserve the strict post-marker rejection for every non-empty line.
        if state.seen_terminator {
            if line.is_empty() {
                continue;
            }
            return Err(CopyWireError {
                code: "22P04",
                message: "COPY data follows the end-of-data marker".to_string(),
            });
        }
        if line == r"\." {
            state.seen_terminator = true;
            continue;
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

/// Validate the buffered payload, then run the supplied guard at the last pre-facade point. The
/// async path also checks before entering this validation, so an already-observed cancellation
/// wins with `57014`; the final guard closes a race during validation without ever rewriting a
/// success or error returned after the facade has been crossed.
fn finish_copy_from_guarded(
    engine: &SharedEngine,
    session: &mut SharedSession,
    state: CopyInState,
    before_submit: impl FnOnce() -> Result<(), CopyWireError>,
) -> Result<QueryOutcome, CopyWireError> {
    if !state.pending_bytes.is_empty() {
        std::str::from_utf8(&state.pending_bytes).map_err(|_| CopyParseError::InvalidUtf8)?;
        return Err(CopyWireError {
            code: "22P04",
            message: "COPY data ended before row terminator".to_string(),
        });
    }
    before_submit()?;
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
    stream: &mut dyn ReadWrite,
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    copy_in: &mut Option<CopyInState>,
    active: &ActiveRequest,
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
                active,
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
                write_copy_terminal_blocking(
                    stream,
                    engine,
                    session,
                    extended,
                    active,
                    state,
                    Err(error),
                )?;
            }
        }
        FrontendMessage::CopyDone => {
            let state = copy_in.take().expect("COPY done requires active state");
            let ready_after_done = state.ready_after_done();
            let result = finish_copy_from_guarded(engine, session, state, || {
                if active.is_cancelled() {
                    Err(CopyWireError::query_cancelled())
                } else {
                    Ok(())
                }
            });
            write_copy_terminal_result_blocking(
                stream,
                engine,
                session,
                extended,
                active,
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
                active,
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
                active,
                state,
                Err(CopyWireError::protocol(
                    "only CopyData, CopyDone, CopyFail, Flush, Sync, or Terminate is allowed during COPY FROM STDIN",
                )),
            )?;
        }
    }
    Ok(true)
}

pub(crate) fn cancel_copy_blocking(
    stream: &mut dyn ReadWrite,
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    copy_in: &mut Option<CopyInState>,
    active: &ActiveRequest,
) -> Result<(), String> {
    let state = copy_in
        .take()
        .expect("COPY cancellation requires active state");
    write_copy_terminal_blocking(
        stream,
        engine,
        session,
        extended,
        active,
        state,
        Err(CopyWireError::query_cancelled()),
    )
}

fn write_copy_terminal_blocking(
    stream: &mut dyn ReadWrite,
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    active: &ActiveRequest,
    state: CopyInState,
    result: Result<QueryOutcome, CopyWireError>,
) -> Result<(), String> {
    let ready_after_done = state.ready_after_done();
    write_copy_terminal_result_blocking(
        stream,
        engine,
        session,
        extended,
        active,
        ready_after_done,
        result,
    )
}

fn write_copy_terminal_result_blocking(
    stream: &mut dyn ReadWrite,
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    active: &ActiveRequest,
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
        complete_simple_query_action_blocking(
            engine,
            session,
            extended,
            active,
            &lifecycle,
            &mut response,
        )
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
    active: &ActiveRequest,
) -> Result<Result<(CopyInState, Vec<u8>), CopyWireError>, String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok(Err(CopyWireError::query_cancelled()));
    };
    let cancellation = active.token();
    let result = tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cancellation.is_cancelled() {
            return Err(CopyWireError::query_cancelled());
        }
        begin_copy_from(&engine, &mut session, copy, ready_after_done)
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok(cancel_effect_free_success(
        active,
        result,
        CopyWireError::query_cancelled,
    ))
}

pub(crate) async fn begin_prepared_copy_from_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    copy: CopyFromStdin,
    target: CopyTarget,
    ready_after_done: bool,
    active: &ActiveRequest,
) -> Result<Result<(CopyInState, Vec<u8>), CopyWireError>, String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok(Err(CopyWireError::query_cancelled()));
    };
    let cancellation = active.token();
    let result = tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cancellation.is_cancelled() {
            return Err(CopyWireError::query_cancelled());
        }
        begin_prepared_copy_from(&engine, &mut session, copy, target, ready_after_done)
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok(cancel_effect_free_success(
        active,
        result,
        CopyWireError::query_cancelled,
    ))
}

pub(crate) async fn execute_copy_to_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    copy: CopyToStdout,
    active: &ActiveRequest,
) -> Result<Result<Vec<u8>, CopyWireError>, String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok(Err(CopyWireError::query_cancelled()));
    };
    let cancellation = active.token();
    let result = tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cancellation.is_cancelled() {
            return Err(CopyWireError::query_cancelled());
        }
        execute_copy_to(&engine, &mut session, &copy)
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok(cancel_effect_free_success(
        active,
        result,
        CopyWireError::query_cancelled,
    ))
}

pub(crate) async fn execute_prepared_copy_to_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    bound: BoundPreparedStatement,
    copy: CopyToStdout,
    active: &ActiveRequest,
) -> Result<Result<Vec<u8>, CopyWireError>, String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok(Err(CopyWireError::query_cancelled()));
    };
    let cancellation = active.token();
    let result = tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cancellation.is_cancelled() {
            return Err(CopyWireError::query_cancelled());
        }
        execute_prepared_copy_to(&engine, &mut session, &bound, &copy)
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok(cancel_effect_free_success(
        active,
        result,
        CopyWireError::query_cancelled,
    ))
}

async fn finish_copy_from_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    state: CopyInState,
    active: &ActiveRequest,
) -> Result<Result<QueryOutcome, CopyWireError>, String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok(Err(CopyWireError::query_cancelled()));
    };
    let cancellation = active.token();
    tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cancellation.is_cancelled() {
            return Err(CopyWireError::query_cancelled());
        }
        finish_copy_from_guarded(&engine, &mut session, state, || {
            if cancellation.is_cancelled() {
                Err(CopyWireError::query_cancelled())
            } else {
                Ok(())
            }
        })
    })
    .await
    .map_err(|error| error.to_string())
}

async fn append_copy_data_async(
    executor: &Arc<tokio::sync::Semaphore>,
    mut state: CopyInState,
    bytes: Vec<u8>,
    active: &ActiveRequest,
) -> Result<(CopyInState, Result<(), CopyWireError>), String> {
    let Some(_permit) = active
        .acquire_permit(executor)
        .await
        .map_err(|error| error.to_string())?
    else {
        return Ok((state, Err(CopyWireError::query_cancelled())));
    };
    let cancellation = active.token();
    let (state, result) = tokio::task::spawn_blocking(move || {
        let result = if cancellation.is_cancelled() {
            Err(CopyWireError::query_cancelled())
        } else {
            append_copy_data(&mut state, &bytes)
        };
        (state, result)
    })
    .await
    .map_err(|error| error.to_string())?;
    Ok((
        state,
        cancel_effect_free_success(active, result, CopyWireError::query_cancelled),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_copy_frame_async(
    stream: &mut TokioTcpStream,
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    extended: &mut ExtendedSession,
    copy_in: &mut Option<CopyInState>,
    active: &ActiveRequest,
    frame: Vec<u8>,
) -> Result<bool, String> {
    let message = {
        let Some(_permit) = active
            .acquire_permit(executor)
            .await
            .map_err(|error| error.to_string())?
        else {
            cancel_copy_async(stream, engine, session, executor, extended, copy_in, active).await?;
            return Ok(true);
        };
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
                active,
                state,
                Err(CopyWireError::protocol(error.to_string())),
            )
            .await?;
            return Ok(true);
        }
    };
    if active.is_cancelled() {
        cancel_copy_async(stream, engine, session, executor, extended, copy_in, active).await?;
        return Ok(true);
    }
    match message {
        FrontendMessage::CopyData(bytes) => {
            let state = copy_in.take().expect("COPY data requires active state");
            let (state, result) = append_copy_data_async(executor, state, bytes, active).await?;
            match result {
                Ok(()) => *copy_in = Some(state),
                Err(error) => {
                    write_copy_terminal_async(
                        stream,
                        engine,
                        session,
                        executor,
                        extended,
                        active,
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
            let result = finish_copy_from_async(
                Arc::clone(&engine),
                Arc::clone(&session),
                executor,
                state,
                active,
            )
            .await?;
            write_copy_terminal_result_async(
                stream,
                engine,
                session,
                executor,
                extended,
                active,
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
                active,
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
                active,
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
pub(crate) async fn cancel_copy_async(
    stream: &mut TokioTcpStream,
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    extended: &mut ExtendedSession,
    copy_in: &mut Option<CopyInState>,
    active: &ActiveRequest,
) -> Result<(), String> {
    let state = copy_in
        .take()
        .expect("COPY cancellation requires active state");
    write_copy_terminal_async(
        stream,
        engine,
        session,
        executor,
        extended,
        active,
        state,
        Err(CopyWireError::query_cancelled()),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn write_copy_terminal_async(
    stream: &mut TokioTcpStream,
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    extended: &mut ExtendedSession,
    active: &ActiveRequest,
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
        active,
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
    active: &ActiveRequest,
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
            active,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancellation::CancellationRegistry;
    use std::sync::Mutex;

    #[test]
    fn lexical_gate_preserves_supported_copy_forms_and_rejects_keyword_extensions() {
        for statement in [
            "-- leading COPY\ncopy accounts FROM STDIN",
            "/* outer /* inner */ */ CoPy accounts TO STDOUT WITH CSV",
            "COPY\u{2003}\u{202f}accounts FROM STDIN",
        ] {
            assert!(matches!(
                classify_copy_statement(statement),
                CopyClassification::Supported(_)
            ));
        }
        for statement in [
            "COPYfoo accounts",
            "COPY_ accounts",
            "COPY$1 accounts",
            "COPYé accounts",
        ] {
            assert_eq!(
                classify_copy_statement(statement),
                CopyClassification::NotCopy
            );
        }
    }

    #[test]
    fn copy_accepts_archive_separator_lines_after_the_client_terminator() {
        let engine = SharedEngine::new();
        let mut session = engine.open_session();
        engine
            .submit(
                &mut session,
                SubmissionRequest::Text("CREATE TABLE archive_copy (id int4, name text)"),
            )
            .into_immediate()
            .unwrap();
        let copy = parse_copy_from_stdin("COPY archive_copy FROM STDIN").unwrap();
        let mut state = begin_copy_from(&engine, &mut session, copy, false)
            .unwrap()
            .0;

        append_copy_data(&mut state, b"1\tAda\n\\.\n\n\n").unwrap();
        assert_eq!(state.pending_rows.len(), 1);
        let error = append_copy_data(&mut state, b"2\tGrace\n").unwrap_err();
        assert_eq!(error.code, "22P04");
    }

    #[test]
    fn csv_copy_accepts_psql16_client_terminator_but_preserves_quoted_data() {
        let engine = SharedEngine::new();
        let mut session = engine.open_session();
        engine
            .submit(
                &mut session,
                SubmissionRequest::Text("CREATE TABLE csv_marker (id int4, name text)"),
            )
            .into_immediate()
            .unwrap();
        let copy = parse_copy_from_stdin("COPY csv_marker FROM STDIN WITH CSV").unwrap();
        let mut state = begin_copy_from(&engine, &mut session, copy, false)
            .unwrap()
            .0;

        append_copy_data(&mut state, b"1,\"\\.\"\n\\.\n").unwrap();
        assert_eq!(
            state.pending_rows,
            vec![vec![DbValue::Int4(1), DbValue::Text(r"\.".to_string())]]
        );
        assert!(state.seen_terminator);
    }

    #[tokio::test]
    async fn copy_done_cancelled_while_queued_never_crosses_the_facade() {
        let engine = Arc::new(SharedEngine::new());
        let session = Arc::new(Mutex::new(engine.open_session()));
        {
            let mut session = session
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            engine
                .submit(
                    &mut session,
                    SubmissionRequest::Text(
                        "CREATE TABLE queued_copy_finish (id int4 PRIMARY KEY, name text)",
                    ),
                )
                .into_immediate()
                .unwrap();
        }
        let copy = parse_copy_from_stdin("COPY queued_copy_finish FROM STDIN WITH CSV").unwrap();
        let mut state = {
            let mut session = session
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            begin_copy_from(&engine, &mut session, copy, false)
                .unwrap()
                .0
        };
        append_copy_data(&mut state, b"1,queued\n").unwrap();

        let executor = Arc::new(tokio::sync::Semaphore::new(0));
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();
        let active = connection.begin_request().unwrap();
        let finish = finish_copy_from_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            &executor,
            state,
            &active,
        );
        let cancel = async {
            tokio::task::yield_now().await;
            assert!(registry.cancel(
                connection.backend_key().process_id(),
                &connection.backend_key().secret_key_bytes(),
            ));
        };
        let (finish, ()) = tokio::join!(finish, cancel);
        let finish = finish.unwrap();
        let Err(error) = finish else {
            panic!("queued cancelled COPY unexpectedly published")
        };
        assert_eq!(error.code, "57014");
        drop(active);

        // Reusing the exact primary key proves the buffered row never reached facade CopyFrom.
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        engine
            .submit(
                &mut session,
                SubmissionRequest::Text("INSERT INTO queued_copy_finish VALUES (1, 'recovered')"),
            )
            .into_immediate()
            .unwrap();
    }

    #[test]
    fn copy_final_guard_blocks_a_buffered_row_at_the_last_pre_facade_point() {
        let engine = SharedEngine::new();
        let mut session = engine.open_session();
        engine
            .submit(
                &mut session,
                SubmissionRequest::Text(
                    "CREATE TABLE guarded_copy_finish (id int4 PRIMARY KEY, name text)",
                ),
            )
            .into_immediate()
            .unwrap();
        let copy = parse_copy_from_stdin("COPY guarded_copy_finish FROM STDIN WITH CSV").unwrap();
        let mut state = begin_copy_from(&engine, &mut session, copy, false)
            .unwrap()
            .0;
        append_copy_data(&mut state, b"1,guarded\n").unwrap();
        let result = finish_copy_from_guarded(&engine, &mut session, state, || {
            Err(CopyWireError::query_cancelled())
        });
        let Err(error) = result else {
            panic!("final cancellation guard unexpectedly admitted COPY")
        };
        assert_eq!(error.code, "57014");

        engine
            .submit(
                &mut session,
                SubmissionRequest::Text("INSERT INTO guarded_copy_finish VALUES (1, 'recovered')"),
            )
            .into_immediate()
            .unwrap();
    }

    #[tokio::test]
    async fn every_async_copy_entry_cancels_at_a_zero_permit_boundary() {
        let target_engine = Arc::new(SharedEngine::new());
        let source_engine = SharedEngine::new();
        let mut source_session = source_engine.open_session();
        source_engine
            .submit(
                &mut source_session,
                SubmissionRequest::Text(
                    "CREATE TABLE queued_copy_source (id int4 PRIMARY KEY, name text)",
                ),
            )
            .into_immediate()
            .unwrap();
        let source_from =
            parse_copy_from_stdin("COPY queued_copy_source FROM STDIN WITH CSV").unwrap();
        let source_target = match source_engine
            .submit(
                &mut source_session,
                SubmissionRequest::CopyFromStart(&source_from),
            )
            .into_immediate()
            .unwrap()
        {
            QueryOutcome::CopyIn { target } => target,
            _ => panic!("COPY start returned the wrong outcome"),
        };
        let source_to =
            parse_copy_to_stdout_table("COPY queued_copy_source TO STDOUT WITH CSV").unwrap();
        let source_bound = source_engine
            .prepare_statement(
                &source_session,
                "SELECT id, name FROM queued_copy_source",
                &[],
            )
            .unwrap()
            .bind_values(&[])
            .unwrap();
        let missing_from =
            parse_copy_from_stdin("COPY queued_copy_missing FROM STDIN WITH CSV").unwrap();
        let missing_to =
            parse_copy_to_stdout_table("COPY queued_copy_missing TO STDOUT WITH CSV").unwrap();

        // If any helper crossed its facade call, these exact inputs would return a non-57014
        // unknown-relation or cross-engine owner error. Zero permits plus exact cancellation must
        // instead stop all four helpers before that call.
        let executor = Arc::new(tokio::sync::Semaphore::new(0));
        let registry = Arc::new(CancellationRegistry::new());
        let connection = registry.register();

        let active = connection.begin_request().unwrap();
        let result = begin_copy_from_async(
            Arc::clone(&target_engine),
            Arc::new(Mutex::new(target_engine.open_session())),
            &executor,
            missing_from.clone(),
            true,
            &active,
        );
        let cancel = async {
            tokio::task::yield_now().await;
            assert!(registry.cancel(
                connection.backend_key().process_id(),
                &connection.backend_key().secret_key_bytes(),
            ));
        };
        let (result, ()) = tokio::join!(result, cancel);
        let Err(error) = result.unwrap() else {
            panic!("queued simple COPY FROM unexpectedly crossed the facade")
        };
        assert_eq!(error.code, "57014");
        drop(active);

        let active = connection.begin_request().unwrap();
        let result = begin_prepared_copy_from_async(
            Arc::clone(&target_engine),
            Arc::new(Mutex::new(target_engine.open_session())),
            &executor,
            source_from.clone(),
            source_target.clone(),
            false,
            &active,
        );
        let cancel = async {
            tokio::task::yield_now().await;
            assert!(registry.cancel(
                connection.backend_key().process_id(),
                &connection.backend_key().secret_key_bytes(),
            ));
        };
        let (result, ()) = tokio::join!(result, cancel);
        let Err(error) = result.unwrap() else {
            panic!("queued prepared COPY FROM unexpectedly crossed the facade")
        };
        assert_eq!(error.code, "57014");
        drop(active);

        let active = connection.begin_request().unwrap();
        let result = execute_copy_to_async(
            Arc::clone(&target_engine),
            Arc::new(Mutex::new(target_engine.open_session())),
            &executor,
            missing_to.clone(),
            &active,
        );
        let cancel = async {
            tokio::task::yield_now().await;
            assert!(registry.cancel(
                connection.backend_key().process_id(),
                &connection.backend_key().secret_key_bytes(),
            ));
        };
        let (result, ()) = tokio::join!(result, cancel);
        let Err(error) = result.unwrap() else {
            panic!("queued simple COPY TO unexpectedly crossed the facade")
        };
        assert_eq!(error.code, "57014");
        drop(active);

        let active = connection.begin_request().unwrap();
        let result = execute_prepared_copy_to_async(
            Arc::clone(&target_engine),
            Arc::new(Mutex::new(target_engine.open_session())),
            &executor,
            source_bound.clone(),
            source_to.clone(),
            &active,
        );
        let cancel = async {
            tokio::task::yield_now().await;
            assert!(registry.cancel(
                connection.backend_key().process_id(),
                &connection.backend_key().secret_key_bytes(),
            ));
        };
        let (result, ()) = tokio::join!(result, cancel);
        let Err(error) = result.unwrap() else {
            panic!("queued prepared COPY TO unexpectedly crossed the facade")
        };
        assert_eq!(error.code, "57014");
        assert_eq!(executor.available_permits(), 0);
        let Err(error) = cancel_effect_free_success(
            &active,
            Err::<(), _>(CopyWireError {
                code: "XX000",
                message: "original COPY facade error".to_string(),
            }),
            CopyWireError::query_cancelled,
        ) else {
            panic!("COPY error unexpectedly became success")
        };
        assert_eq!(error.code, "XX000");
        assert_eq!(error.message, "original COPY facade error");
        drop(active);

        // Prove the sentinels are non-vacuous: without cancellation, crossing each facade owner
        // really does produce its original non-cancellation error.
        let mut probe = target_engine.open_session();
        let Err(error) = begin_copy_from(&target_engine, &mut probe, missing_from, true) else {
            panic!("missing COPY FROM target unexpectedly resolved")
        };
        assert_ne!(error.code, "57014");
        let mut probe = target_engine.open_session();
        let Err(error) = begin_prepared_copy_from(
            &target_engine,
            &mut probe,
            source_from,
            source_target,
            false,
        ) else {
            panic!("cross-engine COPY FROM target unexpectedly resolved")
        };
        assert_ne!(error.code, "57014");
        let mut probe = target_engine.open_session();
        let Err(error) = execute_copy_to(&target_engine, &mut probe, &missing_to) else {
            panic!("missing COPY TO target unexpectedly resolved")
        };
        assert_ne!(error.code, "57014");
        let mut probe = target_engine.open_session();
        let Err(error) =
            execute_prepared_copy_to(&target_engine, &mut probe, &source_bound, &source_to)
        else {
            panic!("cross-engine prepared COPY TO unexpectedly resolved")
        };
        assert_ne!(error.code, "57014");
    }
}
