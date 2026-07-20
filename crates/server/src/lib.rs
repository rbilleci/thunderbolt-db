//! Engine-backed pgwire server over the protocol-neutral façade (P0-M3).
//!
//! This is the Phase 0 unification target: a server that speaks the PostgreSQL
//! wire protocol *and* executes against the real `Engine`, reaching it only
//! through `gpu_db_facade` (the neutral boundary) — never calling the engine
//! directly. Wire framing/parsing/encoding is reused from `gpu_db_protocol`;
//! execution and result/error shaping go through the façade and its `pg_adapter`.
//!
//! ## Scope (honest boundaries)
//!
//! - **Simple query plus engine-backed extended query.** Parse/Bind/Describe/Execute/Close share
//!   one connection-local lifecycle and execute prepared R1/W1 commands through the same façade
//!   session as simple Query. COPY, auth (SCRAM/TLS), and the catalog/introspection surface remain
//!   outside this canonical server until their compatibility migration slices.
//!   A multi-statement simple `Query` uses the shared SQL splitter, executes statements in order,
//!   and emits one final ReadyForQuery. Each idle segment runs in one implicit transaction; exact
//!   BEGIN characteristics promote that segment, while COMMIT/ROLLBACK divide it. A failing
//!   implicit statement rolls back every staged segment predecessor and suppresses every successor.
//!   Empty messages return EmptyQueryResponse.
//! - **Concurrent dispatch (P1-M4).** `serve` shares one engine across a worker pool
//!   (`Arc<SharedEngine>`, thread-per-connection on the existing blocking sockets) and
//!   dispatches each statement through `SharedEngine::submit`: read-only statements
//!   pin immutable snapshots and run concurrently. Writers prepare off-lock, then the canonical
//!   commit owner serializes WAL/apply/publication; no facade read/write lock exists.
//!   `serve_sequential` changes only connection acceptance policy for the A/B baseline and uses
//!   the same connection handler, `SharedEngine`, session, and submission boundary.
//! - **Async ingress (P1-M5).** `serve_async` is a `tokio` acceptor that spawns a
//!   lightweight task per connection (not an OS thread), so idle connections are cheap and
//!   the server scales to far more concurrent connections than thread-per-connection can.
//!   Statement execution is dispatched to the blocking engine via `spawn_blocking`, gated by
//!   a semaphore (bounded executor) so the runtime is never blocked. Measured: peak OS-thread
//!   count stays ~constant as connections grow, vs thread-per-conn's linear growth. Trades a
//!   little throughput for connection *scale*; thread-per-conn `serve` stays the
//!   higher-throughput choice at a few hundred connections.

use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener as TokioTcpListener, TcpStream as TokioTcpStream};

use gpu_db_facade::{
    pg_adapter, BoundPreparedStatement, CommandTag, DbError, PointLookupBatcher, PreparedStatement,
    QueryOutcome, SessionTransactionStatus, SharedEngine, SharedSession, SubmissionDispatch,
    SubmissionRequest,
};
use gpu_db_protocol::backend::{BackendColumn, BackendError, BackendWriter};
use gpu_db_protocol::{
    parse_command, parse_frontend_message, parse_startup_packet, split_simple_query, Command,
    StartupPacket, TransactionStatus as WireTransactionStatus,
};

mod extended;
use extended::{
    Dispatch as ExtendedDispatch, ExtendedSession, MalformedFrameAction, SkippingFrameAction,
    TransactionAction,
};

/// Maximum accepted pgwire frame length (DoS guard): a malicious/huge length prefix would
/// otherwise `resize` a buffer to that size before reading a byte — reachable pre-auth, and
/// more exposed now that async ingress holds many untrusted connections. 64 MiB is far above
/// any reasonable simple-query statement (bulk payloads belong in COPY, out of scope here).
const MAX_FRAME_LEN: usize = 64 * 1024 * 1024;

/// Serve connections **concurrently** on `listener`: one engine shared across a
/// thread-per-connection worker pool (`Arc<SharedEngine>`), each statement dispatched
/// through `SharedEngine::submit` — read-only statements pin a snapshot and run
/// concurrently; writes prepare off-lock and serialize only at the canonical commit owner.
/// Blocks until the listener stops.
/// This is the P1-M4 concurrent dispatch (the first production caller of the `&self` engine
/// read path); `serve_sequential` keeps one-at-a-time acceptance as the A/B baseline while sharing
/// this exact handler and submission path.
pub fn serve(listener: TcpListener) -> io::Result<()> {
    serve_with_engine(listener, shared_engine_from_env()?)
}

/// Build the default served engine honoring the first-class durability config (write-path
/// assessment D4): `GPU_DB_WAL_SEGMENT=<path>` serves a crash-durable engine that recovers any
/// existing segment at that path and fsyncs every commit before visibility; unset serves the
/// in-memory-WAL engine (no crash durability — the pre-existing default, kept for benchmarks).
fn shared_engine_from_env() -> io::Result<Arc<SharedEngine>> {
    let engine = SharedEngine::new_from_env().map_err(io::Error::other)?;
    if engine.is_durable() {
        eprintln!(
            "gpu-db-engine-server: durable WAL enabled (GPU_DB_WAL_SEGMENT={})",
            std::env::var("GPU_DB_WAL_SEGMENT").unwrap_or_default()
        );
    }
    Ok(Arc::new(engine))
}

/// `serve` over a caller-provided shared engine — e.g. one pre-warmed to GPU residency
/// before serving (the GPU-retained benchmark).
pub fn serve_with_engine(listener: TcpListener, engine: Arc<SharedEngine>) -> io::Result<()> {
    for stream in listener.incoming() {
        let stream = stream?;
        // Disable Nagle: pgwire responses are several small frames (RowDescription, DataRow,
        // CommandComplete, ReadyForQuery), and Nagle + delayed-ACK otherwise stalls each
        // reply ~40ms on loopback. Surfaced by the P1-M4 load harness.
        let _ = stream.set_nodelay(true);
        let engine = Arc::clone(&engine);
        thread::spawn(move || {
            let mut stream = stream;
            if let Err(err) = handle_connection(&mut stream, &engine) {
                eprintln!("gpu-db-engine-server connection error: {err}");
            }
        });
    }
    Ok(())
}

/// Serve connections sequentially on `listener` using the same shared facade and connection
/// handler. Only connection acceptance is serial; this is the P1-M4 A/B baseline.
pub fn serve_sequential(listener: TcpListener) -> io::Result<()> {
    let engine = shared_engine_from_env()?;
    for stream in listener.incoming() {
        let mut stream = stream?;
        let _ = stream.set_nodelay(true);
        if let Err(err) = handle_connection(&mut stream, &engine) {
            eprintln!("gpu-db-engine-server connection error: {err}");
        }
    }
    Ok(())
}

/// Drive the startup handshake. Returns `Ok(false)` if the client disconnected before sending a
/// real startup message.
fn complete_startup(stream: &mut TcpStream) -> Result<bool, String> {
    let mut frame = match read_startup_frame(stream).map_err(|err| err.to_string())? {
        Some(frame) => frame,
        None => return Ok(false),
    };
    loop {
        match parse_startup_packet(&frame).map_err(|err| err.to_string())? {
            StartupPacket::SslRequest | StartupPacket::GssEncRequest => {
                stream.write_all(b"N").map_err(|err| err.to_string())?;
                frame = match read_startup_frame(stream).map_err(|err| err.to_string())? {
                    Some(frame) => frame,
                    None => return Ok(false),
                };
            }
            StartupPacket::CancelRequest { .. } => return Ok(false),
            StartupPacket::Startup { .. } => break,
        }
    }

    let mut writer = BackendWriter::new(&mut *stream);
    writer.authentication_ok().map_err(|err| err.to_string())?;
    writer
        .parameter_status("server_version", "16.0-gpu-db-engine-facade")
        .map_err(|err| err.to_string())?;
    writer
        .parameter_status("client_encoding", "UTF8")
        .map_err(|err| err.to_string())?;
    writer
        .parameter_status("DateStyle", "ISO, MDY")
        .map_err(|err| err.to_string())?;
    writer
        .parameter_status("integer_datetimes", "on")
        .map_err(|err| err.to_string())?;
    writer
        .ready_for_query(false)
        .map_err(|err| err.to_string())?;
    Ok(true)
}

/// Handle one connection against the shared engine: startup handshake, then a simple-query loop
/// dispatched through the session-aware shared façade (one transaction owner per connection).
pub fn handle_connection(stream: &mut TcpStream, engine: &SharedEngine) -> Result<(), String> {
    if !complete_startup(stream)? {
        return Ok(());
    }
    let mut session = engine.open_session();
    let result = run_shared_query_loop(stream, engine, &mut session);
    let _ = engine.submit(&mut session, SubmissionRequest::CloseSession);
    result
}

fn submit_text(
    engine: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    engine
        .submit(session, SubmissionRequest::Text(sql))
        .into_immediate()
}

fn submit_prepared(
    engine: &SharedEngine,
    session: &mut SharedSession,
    bound: &BoundPreparedStatement,
) -> Result<QueryOutcome, DbError> {
    engine
        .submit(session, SubmissionRequest::Prepared(bound))
        .into_immediate()
}

fn outcome_is_transaction_control(outcome: &Result<QueryOutcome, DbError>) -> bool {
    matches!(
        outcome,
        Ok(QueryOutcome::Command {
            tag: CommandTag::Begin | CommandTag::Commit | CommandTag::Rollback,
            ..
        })
    )
}

fn preflight_simple_query<S: AsRef<str>>(statements: &[S]) -> Result<(), DbError> {
    if statements.len() <= 1 {
        return Ok(());
    }
    // PostgreSQL performs lexical/syntactic analysis of the complete simple Query before running
    // any statement. Parse and zero-arity Bind every executable span up front so a later syntax or
    // unbound-parameter error cannot follow an already-published explicit COMMIT. Catalog lookup,
    // constraints, and other semantic work remain in execution order through the facade.
    for statement in statements {
        PreparedStatement::parse(statement.as_ref())?.bind_values(&[])?;
    }
    Ok(())
}

fn simple_query_segment_begin<'a, S: AsRef<str>>(
    remaining_statements: &'a [S],
    status: SessionTransactionStatus,
    extended: &ExtendedSession,
) -> Option<&'a str> {
    if remaining_statements.len() <= 1
        || status != SessionTransactionStatus::Idle
        || extended.has_implicit_transaction()
    {
        return None;
    }

    // PostgreSQL starts an implicit block for an idle multi-statement Query, but a later BEGIN
    // converts that same segment into a regular transaction retroactively. Start the segment with
    // that BEGIN's exact characteristics instead of a default BEGIN that would make the facade
    // treat the client's statement as a nested no-op. Stop at COMMIT/ROLLBACK: any BEGIN after that
    // boundary belongs to the next segment and will be selected when that segment becomes idle.
    for statement in remaining_statements {
        let statement = statement.as_ref();
        match parse_command(statement) {
            Ok(Command::Begin { .. }) => return Some(statement),
            Ok(Command::Commit { .. } | Command::Rollback { .. }) => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }
    Some("BEGIN")
}

/// Execute one PostgreSQL simple-query message. Each idle multi-statement segment gets one implicit
/// transaction. Explicit BEGIN/COMMIT/ROLLBACK statements can divide the message, so a suffix after
/// COMMIT starts a fresh implicit segment and cannot publish a prefix before a later suffix error.
/// Every statement still crosses `SharedEngine::submit`; this owner adds only pgwire framing and
/// session transaction lifecycle around the canonical facade boundary.
fn execute_simple_query_blocking(
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    sql: &str,
) -> io::Result<Vec<u8>> {
    extended.clear_unnamed_for_simple_query();
    let statements = split_simple_query(sql);
    if statements.is_empty() {
        let outcome = Ok(QueryOutcome::Empty);
        let mut response = encode_outcome_messages(outcome.clone())?;
        complete_simple_query_action_blocking(engine, session, extended, &outcome, &mut response)?;
        let status = session.transaction_status();
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status)?);
        return Ok(response);
    }
    if let Err(error) = preflight_simple_query(&statements) {
        session.mark_transaction_failed();
        let outcome = Err(error);
        let mut response = encode_outcome_messages(outcome.clone())?;
        complete_simple_query_action_blocking(engine, session, extended, &outcome, &mut response)?;
        let status = session.transaction_status();
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status)?);
        return Ok(response);
    }

    let mut response = Vec::new();
    let mut last_outcome = Ok(QueryOutcome::Empty);
    for (index, statement) in statements.iter().enumerate() {
        if let Some(begin) =
            simple_query_segment_begin(&statements[index..], session.transaction_status(), extended)
        {
            if let Err(error) = submit_text(engine, session, begin) {
                response.extend_from_slice(&encode_outcome_messages(Err(error))?);
                break;
            }
            extended.complete_transaction_action(TransactionAction::BeginImplicit, true);
        }
        let outcome = submit_text(engine, session, statement);
        response.extend_from_slice(&encode_outcome_messages(outcome.clone())?);
        let failed = outcome.is_err();
        if failed || outcome_is_transaction_control(&outcome) {
            complete_simple_query_action_blocking(
                engine,
                session,
                extended,
                &outcome,
                &mut response,
            )?;
        }
        last_outcome = outcome;
        if failed {
            break;
        }
    }
    if extended.has_implicit_transaction() {
        complete_simple_query_action_blocking(
            engine,
            session,
            extended,
            &last_outcome,
            &mut response,
        )?;
    }
    let status = session.transaction_status();
    extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
    response.extend_from_slice(&encode_ready(status)?);
    Ok(response)
}

fn complete_simple_query_action_blocking(
    engine: &SharedEngine,
    session: &mut SharedSession,
    extended: &mut ExtendedSession,
    outcome: &Result<QueryOutcome, DbError>,
    response: &mut Vec<u8>,
) -> io::Result<()> {
    let action = extended.simple_query_completion_action(outcome);
    let Some(sql) = action.sql() else {
        extended.complete_transaction_action(action, true);
        return Ok(());
    };
    match submit_text(engine, session, sql) {
        Ok(_) => extended.complete_transaction_action(action, true),
        Err(error) => {
            response.extend_from_slice(&encode_outcome_messages(Err(error))?);
            if let Some(cleanup) = action.failure_cleanup().sql() {
                if submit_text(engine, session, cleanup).is_ok() {
                    extended.complete_transaction_action(action.failure_cleanup(), true);
                }
            }
        }
    }
    Ok(())
}

fn run_shared_query_loop(
    stream: &mut TcpStream,
    engine: &SharedEngine,
    session: &mut SharedSession,
) -> Result<(), String> {
    let mut extended = ExtendedSession::default();
    while let Some(frame) = read_tagged_frame(stream).map_err(|err| err.to_string())? {
        match extended.skipping_frame_action(frame[0]) {
            SkippingFrameAction::Parse => {}
            SkippingFrameAction::Discard => continue,
            SkippingFrameAction::Terminate => break,
            SkippingFrameAction::RollbackAndSync => {
                let _ = submit_text(engine, session, "ROLLBACK");
                let status = session.transaction_status();
                let sync_error = extended
                    .complete_skipped_sync_frame(&frame, status != SessionTransactionStatus::Idle)
                    .err();
                stream
                    .write_all(
                        &encode_optional_error_and_ready(sync_error, status)
                            .map_err(|err| err.to_string())?,
                    )
                    .map_err(|err| err.to_string())?;
                continue;
            }
            SkippingFrameAction::Sync => {
                let status = session.transaction_status();
                let sync_error = extended
                    .complete_skipped_sync_frame(&frame, status != SessionTransactionStatus::Idle)
                    .err();
                stream
                    .write_all(
                        &encode_optional_error_and_ready(sync_error, status)
                            .map_err(|err| err.to_string())?,
                    )
                    .map_err(|err| err.to_string())?;
                continue;
            }
        }
        let message = match parse_frontend_message(&frame) {
            Ok(message) => message,
            Err(error) => {
                session.mark_transaction_failed();
                match extended.malformed_frame_action(frame[0]) {
                    MalformedFrameAction::SkipUntilSync => {
                        extended.fail();
                        stream
                            .write_all(&encode_frontend_message_error(error))
                            .map_err(|err| err.to_string())?;
                    }
                    MalformedFrameAction::ErrorAndReady(action) => {
                        if let Some(sql) = action.sql() {
                            let _ = submit_text(engine, session, sql);
                        }
                        extended.complete_transaction_action(action, true);
                        let status = session.transaction_status();
                        extended
                            .finish_transaction_boundary(status != SessionTransactionStatus::Idle);
                        stream
                            .write_all(
                                &encode_optional_error_and_ready(Some(error.into()), status)
                                    .map_err(|err| err.to_string())?,
                            )
                            .map_err(|err| err.to_string())?;
                    }
                }
                continue;
            }
        };
        let transaction_action =
            extended.before_dispatch_transaction_action(&message, session.transaction_status());
        if let Some(sql) = transaction_action.sql() {
            if let Err(error) = submit_text(engine, session, sql) {
                extended.complete_transaction_action(transaction_action, false);
                extended.fail();
                stream
                    .write_all(&encode_extended_error(error.into()))
                    .map_err(|err| err.to_string())?;
                continue;
            }
            extended.complete_transaction_action(transaction_action, true);
        }
        let response = match extended.dispatch(message, session.transaction_status()) {
            ExtendedDispatch::SimpleQuery(sql) => {
                execute_simple_query_blocking(engine, session, &mut extended, &sql)
                    .map_err(encode_io_error)
            }
            ExtendedDispatch::Prepare(request) => {
                let prepared = engine.describe_prepared_statement(
                    session,
                    request.parsed.clone(),
                    &request.parameter_type_hints,
                );
                extended
                    .complete_parse(*request, prepared)
                    .map_err(encode_extended_error)
            }
            ExtendedDispatch::Bind(request) => {
                let completion = ExtendedSession::bind_request(*request);
                extended
                    .complete_bind(completion)
                    .map_err(encode_extended_error)
            }
            ExtendedDispatch::Describe { target, name } => extended
                .describe_revalidated(engine, session, target, &name, session.transaction_status())
                .map_err(encode_extended_error),
            ExtendedDispatch::Terminate => break,
            ExtendedDispatch::Sync => {
                let action = extended.sync_transaction_action();
                let commit_error = action
                    .sql()
                    .and_then(|sql| submit_text(engine, session, sql).err());
                if commit_error.is_some() {
                    if let Some(cleanup) = action.failure_cleanup().sql() {
                        let _ = submit_text(engine, session, cleanup);
                    }
                }
                extended.complete_transaction_action(action, true);
                let status = session.transaction_status();
                extended.sync(status != SessionTransactionStatus::Idle);
                match commit_error {
                    Some(error) => encode_outcome(Err(error), status).map_err(encode_io_error),
                    None => encode_ready(status).map_err(encode_io_error),
                }
            }
            ExtendedDispatch::Execute {
                portal_name,
                max_rows,
            } => {
                let transaction_ended = extended.portal_ends_transaction(&portal_name);
                let result = (|| {
                    if let Some(request) = extended.execution_request(&portal_name)? {
                        let outcome = submit_prepared(engine, session, &request.bound);
                        extended.set_execution_outcome(&portal_name, outcome)?;
                    }
                    extended.encode_execute(&portal_name, max_rows)
                })();
                if transaction_ended && result.is_ok() {
                    extended.finish_transaction_boundary(false);
                }
                result.map_err(encode_extended_error)
            }
            ExtendedDispatch::Response(result) => result.map_err(encode_extended_error),
        };
        match response {
            Ok(buf) => stream.write_all(&buf).map_err(|err| err.to_string())?,
            Err(buf) => {
                session.mark_transaction_failed();
                extended.fail();
                stream.write_all(&buf).map_err(|err| err.to_string())?;
            }
        }
    }
    Ok(())
}

/// Encode a neutral façade outcome into pgwire backend message bytes (EmptyQueryResponse /
/// RowDescription+DataRow+CommandComplete / ErrorResponse, then ReadyForQuery). All
/// PostgreSQL-specific encoding lives in `gpu_db_facade::pg_adapter`. Built into a buffer so
/// it is shared by the sync (`write_outcome`) and async (`serve_async`) write paths.
fn encode_outcome(
    outcome: Result<QueryOutcome, DbError>,
    transaction_status: SessionTransactionStatus,
) -> io::Result<Vec<u8>> {
    let mut buf = encode_outcome_messages(outcome)?;
    buf.extend_from_slice(&encode_ready(transaction_status)?);
    Ok(buf)
}

/// Encode one statement's simple-query messages without ReadyForQuery. A multi-statement Query
/// concatenates these fragments and emits exactly one ReadyForQuery after its final transaction
/// action, matching PostgreSQL's message boundary.
fn encode_outcome_messages(outcome: Result<QueryOutcome, DbError>) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut writer = BackendWriter::new(&mut buf);
        match outcome {
            // An empty statement gets EmptyQueryResponse (not CommandComplete), per
            // the wire protocol.
            Ok(QueryOutcome::Empty) => {
                writer.empty_query_response()?;
            }
            Ok(outcome) => {
                let tag = pg_adapter::command_complete_tag(&outcome);
                let returned_rows = match &outcome {
                    QueryOutcome::Rows { columns, rows }
                    | QueryOutcome::Returning { columns, rows, .. } => Some((columns, rows)),
                    _ => None,
                };
                if let Some((columns, rows)) = returned_rows {
                    let backend_columns: Vec<BackendColumn> = columns
                        .iter()
                        .map(|column| {
                            BackendColumn::new(
                                column.name.clone(),
                                pg_adapter::logical_type_oid(column.logical_type),
                                pg_adapter::logical_type_size(column.logical_type),
                            )
                        })
                        .collect();
                    writer.row_description(&backend_columns)?;
                    for row in rows {
                        let values: Vec<Option<String>> =
                            row.iter().map(pg_adapter::db_value_text_opt).collect();
                        writer.data_row(&values)?;
                    }
                }
                writer.command_complete(&tag)?;
            }
            Err(error) => {
                writer.error_response(&BackendError::new(
                    pg_adapter::error_sqlstate(error.category).to_string(),
                    error.message,
                ))?;
            }
        }
    }
    Ok(buf)
}

fn encode_ready(transaction_status: SessionTransactionStatus) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    BackendWriter::new(&mut buf)
        .ready_for_query_status(wire_transaction_status(transaction_status))?;
    Ok(buf)
}

fn encode_optional_error_and_ready(
    error: Option<extended::ExtendedError>,
    transaction_status: SessionTransactionStatus,
) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    if let Some(error) = error {
        buf.extend_from_slice(&error.encode()?);
    }
    buf.extend_from_slice(&encode_ready(transaction_status)?);
    Ok(buf)
}

fn wire_transaction_status(status: SessionTransactionStatus) -> WireTransactionStatus {
    match status {
        SessionTransactionStatus::Idle => WireTransactionStatus::Idle,
        SessionTransactionStatus::InTransaction => WireTransactionStatus::InTransaction,
        SessionTransactionStatus::FailedTransaction => WireTransactionStatus::FailedTransaction,
    }
}

fn encode_extended_error(error: extended::ExtendedError) -> Vec<u8> {
    error.encode().unwrap_or_default()
}

fn encode_io_error(error: io::Error) -> Vec<u8> {
    encode_extended_error(extended::ExtendedError {
        code: "XX000",
        message: error.to_string(),
    })
}

fn encode_frontend_message_error(error: gpu_db_protocol::FrontendMessageError) -> Vec<u8> {
    encode_extended_error(extended::ExtendedError {
        code: "08P01",
        message: error.to_string(),
    })
}

/// Read one untagged startup-style frame (4-byte length prefix, no type byte).
fn read_startup_frame(stream: &mut TcpStream) -> io::Result<Option<Vec<u8>>> {
    let mut len = [0_u8; 4];
    match stream.read_exact(&mut len) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }
    let frame_len = u32::from_be_bytes(len) as usize;
    if frame_len < 4 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "startup frame length is shorter than length field",
        ));
    }
    if frame_len > MAX_FRAME_LEN {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "startup frame length exceeds maximum",
        ));
    }
    let mut frame = len.to_vec();
    frame.resize(frame_len, 0);
    stream.read_exact(&mut frame[4..])?;
    Ok(Some(frame))
}

/// Read one tagged frame (1-byte type + 4-byte length prefix + payload).
fn read_tagged_frame(stream: &mut TcpStream) -> io::Result<Option<Vec<u8>>> {
    let mut tag = [0_u8; 1];
    match stream.read_exact(&mut tag) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }
    let mut len = [0_u8; 4];
    stream.read_exact(&mut len)?;
    let frame_len = u32::from_be_bytes(len) as usize;
    if frame_len < 4 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "frontend frame length is shorter than length field",
        ));
    }
    if frame_len > MAX_FRAME_LEN {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "frontend frame length exceeds maximum",
        ));
    }
    let mut frame = vec![tag[0]];
    frame.extend_from_slice(&len);
    frame.resize(frame_len + 1, 0);
    stream.read_exact(&mut frame[5..])?;
    Ok(Some(frame))
}

// ---------------------------------------------------------------------------
// Async ingress (P1-M5): tokio acceptor + per-connection tasks + bounded executor
// ---------------------------------------------------------------------------

/// Default cap on concurrent blocking engine executions (the "bounded executor"). Idle
/// connections hold no permit, so this bounds *in-flight engine work*, not connection count.
const DEFAULT_MAX_CONCURRENT_EXECUTIONS: usize = 256;

/// Async-ingress server (P1-M5): a `tokio` acceptor spawns a lightweight task per connection
/// (not an OS thread), so idle connections are cheap parked tasks and the server scales to
/// far more concurrent connections than thread-per-connection can. Each statement runs on
/// the blocking engine via `spawn_blocking`, gated by a semaphore so at most
/// `DEFAULT_MAX_CONCURRENT_EXECUTIONS` engine calls run at once (bounded executor — the
/// blocking engine never stalls the async runtime). Reads run concurrently / writes serialize
/// via the shared engine's `RwLock`. Must be run inside a tokio runtime.
pub async fn serve_async(listener: TokioTcpListener) -> io::Result<()> {
    serve_async_with_permits(listener, DEFAULT_MAX_CONCURRENT_EXECUTIONS).await
}

/// `serve_async` with an explicit bound on concurrent blocking executions (for benchmarks).
pub async fn serve_async_with_permits(
    listener: TokioTcpListener,
    max_concurrent_executions: usize,
) -> io::Result<()> {
    serve_async_with_engine(
        listener,
        shared_engine_from_env()?,
        max_concurrent_executions,
    )
    .await
}

/// `serve_async` over a caller-provided shared engine — e.g. one pre-warmed to GPU residency
/// before serving (the GPU-retained benchmark).
///
/// **Point-lookup batching (Thread-3, default ON).** By default batchable point-lookups are
/// routed through a shared [`PointLookupBatcher`] (one coalescer thread, one GPU submission per
/// batch) and the connection task `await`s a `oneshot` while holding no semaphore permit; every
/// other statement keeps the unchanged per-query `spawn_blocking` path. Batching is `>=` the
/// per-query path at every concurrency (≈parity at c1 via the adaptive `max_wait`, 3.7-4.7× at
/// high concurrency) and produces byte-identical results, so it is on unless explicitly disabled.
/// Set `GPU_DB_BATCHING=0` (or `false`/`off`/`no`) to fall back to the per-query path.
/// `serve_async_with_engine_batching` sets the arm explicitly for benchmarks.
pub async fn serve_async_with_engine(
    listener: TokioTcpListener,
    engine: Arc<SharedEngine>,
    max_concurrent_executions: usize,
) -> io::Result<()> {
    serve_async_with_engine_batching(
        listener,
        engine,
        max_concurrent_executions,
        batching_enabled_from_env(),
    )
    .await
}

/// Read the `GPU_DB_BATCHING` flag from the environment. **Default ON** (Thread-3 default-on):
/// when the variable is unset, or set to anything other than an explicit off value, batching is
/// enabled. The escape hatch `GPU_DB_BATCHING=0`/`false`/`off`/`no` (case-insensitive) disables
/// it and restores the unchanged per-query path.
fn batching_enabled_from_env() -> bool {
    parse_batching_flag(std::env::var("GPU_DB_BATCHING").ok().as_deref())
}

/// Pure decode of the `GPU_DB_BATCHING` value (`None` ⇒ unset). **Default ON**: `None`, an empty
/// value, or any value other than an explicit off token enables batching; only `0`/`false`/`off`/
/// `no` (case-insensitive, surrounding whitespace ignored) disables it. Kept separate from the
/// `std::env` read so it is unit-testable without mutating process-global env in parallel tests.
fn parse_batching_flag(value: Option<&str>) -> bool {
    match value {
        // Only an explicit off token disables batching; unset and everything else stays default-on.
        Some(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        None => true,
    }
}

/// `serve_async_with_engine` with the point-lookup batching A/B arm chosen explicitly (instead
/// of via the `GPU_DB_BATCHING` env flag) — for benchmarks that drive ON vs OFF directly. When
/// `batching` is true a single [`PointLookupBatcher`] is shared across all connections for the
/// lifetime of this server; dropping it (on shutdown) drains its queue so no waiter is stranded.
pub async fn serve_async_with_engine_batching(
    listener: TokioTcpListener,
    engine: Arc<SharedEngine>,
    max_concurrent_executions: usize,
    batching: bool,
) -> io::Result<()> {
    let executor = Arc::new(tokio::sync::Semaphore::new(
        max_concurrent_executions.max(1),
    ));
    // One coalescer for the whole server when batching is on; `None` keeps the unchanged path.
    let batcher = batching.then(|| Arc::new(PointLookupBatcher::new(Arc::clone(&engine))));
    loop {
        let (stream, _addr) = listener.accept().await?;
        let _ = stream.set_nodelay(true);
        let engine = Arc::clone(&engine);
        let executor = Arc::clone(&executor);
        let batcher = batcher.clone();
        tokio::spawn(async move {
            if let Err(err) =
                handle_connection_async(stream, &engine, &executor, batcher.as_ref()).await
            {
                eprintln!("gpu-db-engine-server async connection error: {err}");
            }
        });
    }
}

async fn handle_connection_async(
    mut stream: TokioTcpStream,
    engine: &Arc<SharedEngine>,
    executor: &Arc<tokio::sync::Semaphore>,
    batcher: Option<&Arc<PointLookupBatcher>>,
) -> Result<(), String> {
    if !complete_startup_async(&mut stream).await? {
        return Ok(());
    }
    let session = Arc::new(std::sync::Mutex::new(engine.open_session()));
    let result = run_async_query_loop(&mut stream, engine, executor, batcher, &session).await;
    let mut session = session
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _ = engine.submit(&mut session, SubmissionRequest::CloseSession);
    result
}

async fn complete_startup_async(stream: &mut TokioTcpStream) -> Result<bool, String> {
    let mut frame = match read_startup_frame_async(stream).await? {
        Some(frame) => frame,
        None => return Ok(false),
    };
    loop {
        match parse_startup_packet(&frame).map_err(|err| err.to_string())? {
            StartupPacket::SslRequest | StartupPacket::GssEncRequest => {
                stream
                    .write_all(b"N")
                    .await
                    .map_err(|err| err.to_string())?;
                frame = match read_startup_frame_async(stream).await? {
                    Some(frame) => frame,
                    None => return Ok(false),
                };
            }
            StartupPacket::CancelRequest { .. } => return Ok(false),
            StartupPacket::Startup { .. } => break,
        }
    }
    let handshake = encode_startup_handshake().map_err(|err| err.to_string())?;
    stream
        .write_all(&handshake)
        .await
        .map_err(|err| err.to_string())?;
    Ok(true)
}

/// Build the startup-OK handshake (AuthenticationOk, ParameterStatus×4, ReadyForQuery).
fn encode_startup_handshake() -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut writer = BackendWriter::new(&mut buf);
        writer.authentication_ok()?;
        writer.parameter_status("server_version", "16.0-gpu-db-engine-facade")?;
        writer.parameter_status("client_encoding", "UTF8")?;
        writer.parameter_status("DateStyle", "ISO, MDY")?;
        writer.parameter_status("integer_datetimes", "on")?;
        writer.ready_for_query(false)?;
    }
    Ok(buf)
}

async fn run_async_query_loop(
    stream: &mut TokioTcpStream,
    engine: &Arc<SharedEngine>,
    executor: &Arc<tokio::sync::Semaphore>,
    batcher: Option<&Arc<PointLookupBatcher>>,
    session: &Arc<std::sync::Mutex<SharedSession>>,
) -> Result<(), String> {
    let mut extended = ExtendedSession::default();
    while let Some(frame) = read_tagged_frame_async(stream).await? {
        let frame_tag = frame[0];
        match extended.skipping_frame_action(frame_tag) {
            SkippingFrameAction::Parse => {}
            SkippingFrameAction::Discard => continue,
            SkippingFrameAction::Terminate => break,
            SkippingFrameAction::RollbackAndSync => {
                let _ = execute_shared_session_blocking(
                    Arc::clone(engine),
                    Arc::clone(session),
                    executor,
                    "ROLLBACK".to_string(),
                )
                .await?;
                let status = shared_session_transaction_status(session);
                let sync_error = extended
                    .complete_skipped_sync_frame(&frame, status != SessionTransactionStatus::Idle)
                    .err();
                let buf = encode_optional_error_and_ready(sync_error, status)
                    .map_err(|err| err.to_string())?;
                stream
                    .write_all(&buf)
                    .await
                    .map_err(|err| err.to_string())?;
                continue;
            }
            SkippingFrameAction::Sync => {
                let status = shared_session_transaction_status(session);
                let sync_error = extended
                    .complete_skipped_sync_frame(&frame, status != SessionTransactionStatus::Idle)
                    .err();
                let buf = encode_optional_error_and_ready(sync_error, status)
                    .map_err(|err| err.to_string())?;
                stream
                    .write_all(&buf)
                    .await
                    .map_err(|err| err.to_string())?;
                continue;
            }
        }
        let message = {
            let _permit = executor.acquire().await.map_err(|err| err.to_string())?;
            tokio::task::spawn_blocking(move || parse_frontend_message(&frame))
                .await
                .map_err(|err| err.to_string())?
        };
        let message = match message {
            Ok(message) => message,
            Err(error) => {
                session
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .mark_transaction_failed();
                match extended.malformed_frame_action(frame_tag) {
                    MalformedFrameAction::SkipUntilSync => {
                        extended.fail();
                        stream
                            .write_all(&encode_frontend_message_error(error))
                            .await
                            .map_err(|err| err.to_string())?;
                    }
                    MalformedFrameAction::ErrorAndReady(action) => {
                        if let Some(sql) = action.sql() {
                            let _ = execute_shared_session_blocking(
                                Arc::clone(engine),
                                Arc::clone(session),
                                executor,
                                sql.to_string(),
                            )
                            .await?;
                        }
                        extended.complete_transaction_action(action, true);
                        let status = shared_session_transaction_status(session);
                        extended
                            .finish_transaction_boundary(status != SessionTransactionStatus::Idle);
                        stream
                            .write_all(
                                &encode_optional_error_and_ready(Some(error.into()), status)
                                    .map_err(|err| err.to_string())?,
                            )
                            .await
                            .map_err(|err| err.to_string())?;
                    }
                }
                continue;
            }
        };
        let transaction_action = extended.before_dispatch_transaction_action(
            &message,
            shared_session_transaction_status(session),
        );
        if let Some(sql) = transaction_action.sql() {
            let begin = execute_shared_session_blocking(
                Arc::clone(engine),
                Arc::clone(session),
                executor,
                sql.to_string(),
            )
            .await?;
            if let Err(error) = begin {
                extended.complete_transaction_action(transaction_action, false);
                extended.fail();
                stream
                    .write_all(&encode_extended_error(error.into()))
                    .await
                    .map_err(|err| err.to_string())?;
                continue;
            }
            extended.complete_transaction_action(transaction_action, true);
        }
        let response = match extended.dispatch(message, shared_session_transaction_status(session))
        {
            ExtendedDispatch::SimpleQuery(sql) => execute_simple_query_async(
                Arc::clone(engine),
                Arc::clone(session),
                executor,
                batcher.cloned(),
                &mut extended,
                sql,
            )
            .await
            .map_err(|error| encode_io_error(io::Error::other(error))),
            ExtendedDispatch::Prepare(request) => {
                let _permit = executor.acquire().await.map_err(|err| err.to_string())?;
                let engine = Arc::clone(engine);
                let session = Arc::clone(session);
                let parsed = request.parsed.clone();
                let hints = request.parameter_type_hints.clone();
                let prepared = tokio::task::spawn_blocking(move || {
                    let session = session
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    engine.describe_prepared_statement(&session, parsed, &hints)
                })
                .await
                .map_err(|err| err.to_string())?;
                extended
                    .complete_parse(*request, prepared)
                    .map_err(encode_extended_error)
            }
            ExtendedDispatch::Bind(request) => {
                let _permit = executor.acquire().await.map_err(|err| err.to_string())?;
                let completion =
                    tokio::task::spawn_blocking(move || ExtendedSession::bind_request(*request))
                        .await
                        .map_err(|err| err.to_string())?;
                extended
                    .complete_bind(completion)
                    .map_err(encode_extended_error)
            }
            ExtendedDispatch::Describe { target, name } => {
                let status = shared_session_transaction_status(session);
                match extended.description_owner(target, &name, status) {
                    Err(error) => Err(encode_extended_error(error)),
                    Ok(owner) => {
                        let _permit = executor.acquire().await.map_err(|err| err.to_string())?;
                        let engine_for_validation = Arc::clone(engine);
                        let session_for_validation = Arc::clone(session);
                        let validation = tokio::task::spawn_blocking(move || {
                            let session = session_for_validation
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                            ExtendedSession::revalidate_description_owner(
                                &engine_for_validation,
                                &session,
                                owner,
                            )
                        })
                        .await
                        .map_err(|err| err.to_string())?;
                        match validation {
                            Ok(()) => extended
                                .describe(target, &name, status)
                                .map_err(encode_extended_error),
                            Err(error) => Err(encode_extended_error(error)),
                        }
                    }
                }
            }
            ExtendedDispatch::Terminate => break,
            ExtendedDispatch::Sync => {
                let action = extended.sync_transaction_action();
                let commit_error = if let Some(sql) = action.sql() {
                    execute_shared_session_blocking(
                        Arc::clone(engine),
                        Arc::clone(session),
                        executor,
                        sql.to_string(),
                    )
                    .await?
                    .err()
                } else {
                    None
                };
                if commit_error.is_some() {
                    if let Some(cleanup) = action.failure_cleanup().sql() {
                        let _ = execute_shared_session_blocking(
                            Arc::clone(engine),
                            Arc::clone(session),
                            executor,
                            cleanup.to_string(),
                        )
                        .await?;
                    }
                }
                extended.complete_transaction_action(action, true);
                let status = shared_session_transaction_status(session);
                extended.sync(status != SessionTransactionStatus::Idle);
                match commit_error {
                    Some(error) => encode_outcome(Err(error), status).map_err(encode_io_error),
                    None => encode_ready(status).map_err(encode_io_error),
                }
            }
            ExtendedDispatch::Execute {
                portal_name,
                max_rows,
            } => {
                let transaction_ended = extended.portal_ends_transaction(&portal_name);
                let result = match extended.execution_request(&portal_name) {
                    Err(error) => Err(encode_extended_error(error)),
                    Ok(Some(request)) => {
                        let _permit = executor.acquire().await.map_err(|err| err.to_string())?;
                        let engine = Arc::clone(engine);
                        let session = Arc::clone(session);
                        let outcome = tokio::task::spawn_blocking(move || {
                            let mut session = session
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                            submit_prepared(&engine, &mut session, &request.bound)
                        })
                        .await
                        .map_err(|err| err.to_string())?;
                        extended
                            .set_execution_outcome(&portal_name, outcome)
                            .and_then(|()| extended.encode_execute(&portal_name, max_rows))
                            .map_err(encode_extended_error)
                    }
                    Ok(None) => extended
                        .encode_execute(&portal_name, max_rows)
                        .map_err(encode_extended_error),
                };
                if transaction_ended && result.is_ok() {
                    extended.finish_transaction_boundary(false);
                }
                result
            }
            ExtendedDispatch::Response(result) => result.map_err(encode_extended_error),
        };
        match response {
            Ok(buf) => {
                if !buf.is_empty() {
                    stream
                        .write_all(&buf)
                        .await
                        .map_err(|err| err.to_string())?;
                }
            }
            Err(buf) => {
                session
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .mark_transaction_failed();
                extended.fail();
                stream
                    .write_all(&buf)
                    .await
                    .map_err(|err| err.to_string())?;
            }
        }
    }
    Ok(())
}

fn shared_session_transaction_status(
    session: &Arc<std::sync::Mutex<SharedSession>>,
) -> SessionTransactionStatus {
    session
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .transaction_status()
}

async fn execute_simple_query_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    batcher: Option<Arc<PointLookupBatcher>>,
    extended: &mut ExtendedSession,
    sql: String,
) -> Result<Vec<u8>, String> {
    extended.clear_unnamed_for_simple_query();
    let statements = split_simple_query(&sql)
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if statements.is_empty() {
        let outcome = Ok(QueryOutcome::Empty);
        let mut response =
            encode_outcome_messages(outcome.clone()).map_err(|error| error.to_string())?;
        complete_simple_query_action_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            executor,
            extended,
            &outcome,
            &mut response,
        )
        .await?;
        let status = shared_session_transaction_status(&session);
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
        return Ok(response);
    }
    if let Err(error) = preflight_simple_query(&statements) {
        session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .mark_transaction_failed();
        let outcome = Err(error);
        let mut response =
            encode_outcome_messages(outcome.clone()).map_err(|error| error.to_string())?;
        complete_simple_query_action_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            executor,
            extended,
            &outcome,
            &mut response,
        )
        .await?;
        let status = shared_session_transaction_status(&session);
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
        return Ok(response);
    }
    let single_statement = statements.len() == 1;
    let single_can_batch = single_statement
        && !extended.has_implicit_transaction()
        && shared_session_transaction_status(&session) == SessionTransactionStatus::Idle;

    let mut response = Vec::new();
    let mut last_outcome = Ok(QueryOutcome::Empty);
    for (index, statement) in statements.iter().enumerate() {
        if let Some(begin) = simple_query_segment_begin(
            &statements[index..],
            shared_session_transaction_status(&session),
            extended,
        ) {
            let begin = execute_shared_session_blocking(
                Arc::clone(&engine),
                Arc::clone(&session),
                executor,
                begin.to_string(),
            )
            .await?;
            if let Err(error) = begin {
                response.extend_from_slice(
                    &encode_outcome_messages(Err(error)).map_err(|error| error.to_string())?,
                );
                break;
            }
            extended.complete_transaction_action(TransactionAction::BeginImplicit, true);
        }
        let outcome = if single_can_batch {
            match &batcher {
                Some(batcher) => {
                    execute_batchable_or_fallback(
                        Arc::clone(&engine),
                        executor,
                        Arc::clone(batcher),
                        Arc::clone(&session),
                        statement.clone(),
                    )
                    .await?
                }
                None => {
                    execute_shared_session_blocking(
                        Arc::clone(&engine),
                        Arc::clone(&session),
                        executor,
                        statement.clone(),
                    )
                    .await?
                }
            }
        } else {
            execute_shared_session_blocking(
                Arc::clone(&engine),
                Arc::clone(&session),
                executor,
                statement.clone(),
            )
            .await?
        };
        response.extend_from_slice(
            &encode_outcome_messages(outcome.clone()).map_err(|error| error.to_string())?,
        );
        let failed = outcome.is_err();
        if failed || outcome_is_transaction_control(&outcome) {
            complete_simple_query_action_async(
                Arc::clone(&engine),
                Arc::clone(&session),
                executor,
                extended,
                &outcome,
                &mut response,
            )
            .await?;
        }
        last_outcome = outcome;
        if failed {
            break;
        }
    }
    if extended.has_implicit_transaction() {
        complete_simple_query_action_async(
            engine,
            Arc::clone(&session),
            executor,
            extended,
            &last_outcome,
            &mut response,
        )
        .await?;
    }
    let status = shared_session_transaction_status(&session);
    extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
    response.extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
    Ok(response)
}

async fn complete_simple_query_action_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    extended: &mut ExtendedSession,
    outcome: &Result<QueryOutcome, DbError>,
    response: &mut Vec<u8>,
) -> Result<(), String> {
    let action = extended.simple_query_completion_action(outcome);
    let Some(sql) = action.sql() else {
        extended.complete_transaction_action(action, true);
        return Ok(());
    };
    match execute_shared_session_blocking(
        Arc::clone(&engine),
        Arc::clone(&session),
        executor,
        sql.to_string(),
    )
    .await?
    {
        Ok(_) => extended.complete_transaction_action(action, true),
        Err(error) => {
            response.extend_from_slice(
                &encode_outcome_messages(Err(error)).map_err(|error| error.to_string())?,
            );
            if let Some(cleanup) = action.failure_cleanup().sql() {
                if execute_shared_session_blocking(engine, session, executor, cleanup.to_string())
                    .await?
                    .is_ok()
                {
                    extended.complete_transaction_action(action.failure_cleanup(), true);
                }
            }
        }
    }
    Ok(())
}

async fn execute_shared_session_blocking(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    sql: String,
) -> Result<Result<QueryOutcome, DbError>, String> {
    let _permit = executor.acquire().await.map_err(|err| err.to_string())?;
    tokio::task::spawn_blocking(move || {
        let mut session = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        submit_text(&engine, &mut session, &sql)
    })
    .await
    .map_err(|err| err.to_string())
}

/// Batching-ON dispatch for one SimpleQuery (Thread-3 Stage 1). Classification + the unchanged
/// per-query fallback run inside `spawn_blocking` under a permit (so the runtime never blocks on
/// the engine lock or a slow query). A batchable point-lookup returns a `oneshot::Receiver`,
/// whose permit is then released and the connection task `await`s the receiver with NO permit
/// held — so parked lookups don't consume the bounded-executor budget while they coalesce. A
/// dropped/closed coalescer makes the receiver resolve to a `RecvError`, surfaced as a neutral
/// engine error rather than a hang.
async fn execute_batchable_or_fallback(
    engine: Arc<SharedEngine>,
    executor: &Arc<tokio::sync::Semaphore>,
    batcher: Arc<PointLookupBatcher>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    sql: String,
) -> Result<Result<QueryOutcome, DbError>, String> {
    // Phase 1 (permit held): classify against the engine snapshot and either resolve the
    // unchanged path or enqueue on the batcher. Both are short or self-bounded.
    let dispatch = {
        let _permit = executor.acquire().await.map_err(|err| err.to_string())?;
        tokio::task::spawn_blocking(move || {
            let mut session = session
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // The session-owned batched boundary returns `Immediate(result)` for everything
            // non-batchable or transaction-bound, or `Batched(receiver)` for an eligible read.
            match engine.submit(
                &mut session,
                SubmissionRequest::BatchedText {
                    sql: &sql,
                    batcher: &batcher,
                },
            ) {
                SubmissionDispatch::Immediate(result) => DispatchOut::Immediate(result),
                SubmissionDispatch::Batched(receiver) => DispatchOut::Batched(receiver),
            }
        })
        .await
        .map_err(|err| err.to_string())?
        // permit drops here
    };
    // Phase 2 (no permit): if batched, park on the oneshot off the bounded executor.
    match dispatch {
        DispatchOut::Immediate(result) => Ok(result),
        DispatchOut::Batched(receiver) => match receiver.await {
            Ok(result) => Ok(result),
            // The coalescer dropped the sender (shutdown / unexpected): a neutral error, never
            // a hung connection.
            Err(_) => Ok(Err(DbError {
                category: gpu_db_facade::ErrorCategory::Internal,
                message: "batched point-lookup did not produce a response (coalescer unavailable)"
                    .to_string(),
            })),
        },
    }
}

/// Internal owned form of `SubmissionDispatch` so it can cross `spawn_blocking`.
enum DispatchOut {
    Immediate(Result<QueryOutcome, DbError>),
    Batched(tokio::sync::oneshot::Receiver<Result<QueryOutcome, DbError>>),
}

/// Async read of one untagged startup frame (4-byte length prefix, no type byte).
async fn read_startup_frame_async(stream: &mut TokioTcpStream) -> Result<Option<Vec<u8>>, String> {
    let mut len = [0_u8; 4];
    match stream.read_exact(&mut len).await {
        Ok(_) => {}
        Err(err) if err.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err.to_string()),
    }
    let frame_len = u32::from_be_bytes(len) as usize;
    if frame_len < 4 {
        return Err("startup frame length is shorter than length field".to_string());
    }
    if frame_len > MAX_FRAME_LEN {
        return Err("startup frame length exceeds maximum".to_string());
    }
    let mut frame = len.to_vec();
    frame.resize(frame_len, 0);
    stream
        .read_exact(&mut frame[4..])
        .await
        .map_err(|err| err.to_string())?;
    Ok(Some(frame))
}

/// Async read of one tagged frame (1-byte type + 4-byte length prefix + payload).
async fn read_tagged_frame_async(stream: &mut TokioTcpStream) -> Result<Option<Vec<u8>>, String> {
    let mut tag = [0_u8; 1];
    match stream.read_exact(&mut tag).await {
        Ok(_) => {}
        Err(err) if err.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err.to_string()),
    }
    let mut len = [0_u8; 4];
    stream
        .read_exact(&mut len)
        .await
        .map_err(|err| err.to_string())?;
    let frame_len = u32::from_be_bytes(len) as usize;
    if frame_len < 4 {
        return Err("frontend frame length is shorter than length field".to_string());
    }
    if frame_len > MAX_FRAME_LEN {
        return Err("frontend frame length exceeds maximum".to_string());
    }
    let mut frame = vec![tag[0]];
    frame.extend_from_slice(&len);
    frame.resize(frame_len + 1, 0);
    stream
        .read_exact(&mut frame[5..])
        .await
        .map_err(|err| err.to_string())?;
    Ok(Some(frame))
}

#[cfg(test)]
mod tests;
