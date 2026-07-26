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
//!   session as simple Query. COPY FROM/TO uses the same session and facade mutation/read boundary.
//!   The product listener offers an explicit local-development trust profile or a fail-closed
//!   production TLS + SCRAM-SHA-256 profile; transport authentication wraps this same dispatcher.
//!   The catalog/introspection surface remains subject to its compatibility migration slice.
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

use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener as TokioTcpListener, TcpStream as TokioTcpStream};

use gpu_db_facade::{
    CommandTag, DbError, ErrorCategory, PointLookupBatcher, PreparedStatement, QueryOutcome,
    SessionTransactionStatus, SharedEngine, SharedSession, SubmissionRequest,
};
use gpu_db_protocol::{
    parse_command, parse_frontend_message, parse_startup_packet, split_simple_query, Command,
    StartupPacket,
};

mod async_submit;
use async_submit::{
    analyze_prepare_cancellable, bind_cancellable,
    complete_simple_query_action as complete_simple_query_action_async,
    execute_batchable_or_fallback,
    execute_prepared_cancellable as execute_prepared_shared_session_blocking_cancellable,
    execute_text as execute_shared_session_blocking,
    execute_text_cancellable as execute_shared_session_blocking_cancellable,
    revalidate_description_cancellable,
};
mod cancellation;
use cancellation::{
    cancel_effect_free_success, ActiveRequest, CancellationRegistry, ConnectionCancellation,
};
mod copy;
use copy::{
    begin_copy_from, begin_copy_from_async, begin_prepared_copy_from,
    begin_prepared_copy_from_async, cancel_copy_async, cancel_copy_blocking,
    classify_copy_statement, copy_lifecycle_error, copy_lifecycle_success, encode_copy_error,
    execute_copy_to, execute_copy_to_async, execute_prepared_copy_to,
    execute_prepared_copy_to_async, handle_copy_frame_async, handle_copy_frame_blocking,
    mark_shared_session_failed, CopyClassification, CopyInState, CopyStatement, CopyWireError,
};
mod extended;
use extended::{
    Dispatch as ExtendedDispatch, ExecutionRequest, ExtendedSession, MalformedFrameAction,
    SkippingFrameAction, TransactionAction,
};
mod security;
use security::complete_local_startup;
pub use security::ServerConfig;
mod sql_prepared;
use sql_prepared::classify_sql_prepared_statement;
mod sql_cursor;
use sql_cursor::classify_sql_cursor_statement;
mod sql_session;
use sql_session::{
    classify_prepared_action, execute_cursor_action_async as execute_sql_cursor_action_async,
    execute_cursor_action_blocking as execute_sql_cursor_action_blocking,
    execute_prepared_action_async as execute_sql_prepared_action_async,
    execute_prepared_action_blocking as execute_sql_prepared_action_blocking,
    submit_prepared_cancellable,
};
mod transport;
use transport::{
    read_startup_frame_async, read_tagged_frame, read_tagged_frame_async,
    read_tagged_frame_polling_cancel, read_tagged_frame_polling_cancel_async, PolledTaggedFrame,
    ReadWrite,
};
mod wire_response;
use wire_response::{
    cancellation_checked_outcome, cancellation_error, encode_cancellable_outcome_messages,
    encode_execute_cancellable, encode_extended_error, encode_frontend_message_error,
    encode_io_error, encode_optional_error_and_ready, encode_outcome, encode_outcome_messages,
    encode_ready, encode_startup_handshake, encode_startup_statuses_and_ready,
};

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

/// Parse-time validated product configuration owns the listener security posture. Runtime TLS
/// material and the SCRAM verifier are loaded before bind, then every accepted transport enters
/// the same facade-backed connection/session handler used by local and test ingresses.
pub fn serve_configured(config: ServerConfig) -> io::Result<()> {
    let (listen, security) = config.into_runtime()?;
    let listener = TcpListener::bind(&listen)?;
    let engine = shared_engine_from_env()?;
    let security = Arc::new(security);
    let cancellations = Arc::new(CancellationRegistry::new());
    eprintln!("gpu-db-engine-server (facade-backed) listening on {listen}");
    for stream in listener.incoming() {
        let stream = stream?;
        let _ = stream.set_nodelay(true);
        let timeout_control = stream.try_clone()?;
        let engine = Arc::clone(&engine);
        let security = Arc::clone(&security);
        let cancellations = Arc::clone(&cancellations);
        thread::spawn(
            move || match security.accept_blocking(stream, &cancellations) {
                Ok(Some((mut stream, cancellation))) => {
                    if let Err(error) = handle_ready_connection(
                        stream.as_mut(),
                        &engine,
                        &cancellation,
                        Some(&timeout_control),
                    ) {
                        eprintln!("gpu-db-engine-server connection error: {error}");
                    }
                }
                Ok(None) => {}
                Err(error) => eprintln!("gpu-db-engine-server startup error: {error}"),
            },
        );
    }
    Ok(())
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
    let cancellations = Arc::new(CancellationRegistry::new());
    for stream in listener.incoming() {
        let stream = stream?;
        // Disable Nagle: pgwire responses are several small frames (RowDescription, DataRow,
        // CommandComplete, ReadyForQuery), and Nagle + delayed-ACK otherwise stalls each
        // reply ~40ms on loopback. Surfaced by the P1-M4 load harness.
        let _ = stream.set_nodelay(true);
        let engine = Arc::clone(&engine);
        let cancellations = Arc::clone(&cancellations);
        thread::spawn(move || {
            let mut stream = stream;
            if let Err(err) = handle_connection_registered(&mut stream, &engine, &cancellations) {
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
    let cancellations = Arc::new(CancellationRegistry::new());
    for stream in listener.incoming() {
        let mut stream = stream?;
        let _ = stream.set_nodelay(true);
        if let Err(err) = handle_connection_registered(&mut stream, &engine, &cancellations) {
            eprintln!("gpu-db-engine-server connection error: {err}");
        }
    }
    Ok(())
}

/// Drive the startup handshake. Returns `Ok(false)` if the client disconnected before sending a
/// real startup message.
fn complete_startup(
    stream: &mut TcpStream,
    cancellations: &Arc<CancellationRegistry>,
) -> Result<Option<ConnectionCancellation>, String> {
    complete_local_startup(stream, cancellations)
}

/// Handle one connection against the shared engine: startup handshake, then a simple-query loop
/// dispatched through the session-aware shared façade (one transaction owner per connection).
pub fn handle_connection(stream: &mut TcpStream, engine: &SharedEngine) -> Result<(), String> {
    let cancellations = Arc::new(CancellationRegistry::new());
    handle_connection_registered(stream, engine, &cancellations)
}

fn handle_connection_registered(
    stream: &mut TcpStream,
    engine: &SharedEngine,
    cancellations: &Arc<CancellationRegistry>,
) -> Result<(), String> {
    let Some(cancellation) = complete_startup(stream, cancellations)? else {
        return Ok(());
    };
    let timeout_control = stream.try_clone().map_err(|error| error.to_string())?;
    handle_ready_connection(stream, engine, &cancellation, Some(&timeout_control))
}

fn handle_ready_connection(
    stream: &mut dyn ReadWrite,
    engine: &SharedEngine,
    cancellation: &ConnectionCancellation,
    timeout_control: Option<&TcpStream>,
) -> Result<(), String> {
    let mut session = engine.open_session();
    let result = run_shared_query_loop(stream, engine, &mut session, cancellation, timeout_control);
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

fn submit_text_cancellable(
    engine: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
    active: &ActiveRequest,
) -> Result<QueryOutcome, DbError> {
    if active.is_cancelled() {
        return Err(cancellation_error());
    }
    cancellation_checked_outcome(active, submit_text(engine, session, sql))
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

fn preflight_simple_query<S: AsRef<str>>(
    statements: &[S],
    copy_statements: &[CopyClassification],
) -> Result<(), DbError> {
    if statements.len() <= 1 {
        return Ok(());
    }
    debug_assert_eq!(statements.len(), copy_statements.len());
    // PostgreSQL performs lexical/syntactic analysis of the complete simple Query before running
    // any statement. Parse and zero-arity Bind every executable span up front so a later syntax or
    // unbound-parameter error cannot follow an already-published explicit COMMIT. Catalog lookup,
    // constraints, and other semantic work remain in execution order through the facade. COPY has
    // its own parser; a supported classification is its syntax proof, while unsupported COPY is a
    // semantic decision deliberately deferred until every ordinary span has been checked.
    for (statement, copy) in statements.iter().zip(copy_statements) {
        if matches!(copy, CopyClassification::NotCopy)
            && classify_sql_cursor_statement(statement.as_ref())?.is_none()
            && classify_sql_prepared_statement(statement.as_ref())?.is_none()
        {
            PreparedStatement::parse(statement.as_ref())?.bind_values(&[])?;
        }
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
    copy_in: &mut Option<CopyInState>,
    sql: &str,
    active: &ActiveRequest,
) -> io::Result<Vec<u8>> {
    extended.clear_unnamed_for_simple_query();
    if active.is_cancelled() {
        let outcome = Err(cancellation_error());
        let mut response = encode_outcome_messages(outcome.clone())?;
        complete_simple_query_action_blocking(
            engine,
            session,
            extended,
            active,
            &outcome,
            &mut response,
        )?;
        let status = session.transaction_status();
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status)?);
        return Ok(response);
    }
    let statements = split_simple_query(sql);
    if statements.is_empty() {
        let outcome = cancellation_checked_outcome(active, Ok(QueryOutcome::Empty));
        let mut response = encode_outcome_messages(outcome.clone())?;
        complete_simple_query_action_blocking(
            engine,
            session,
            extended,
            active,
            &outcome,
            &mut response,
        )?;
        let status = session.transaction_status();
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status)?);
        return Ok(response);
    }
    let copy_statements = statements
        .iter()
        .map(|statement| classify_copy_statement(statement))
        .collect::<Vec<_>>();
    if let Err(error) = preflight_simple_query(&statements, &copy_statements) {
        session.mark_transaction_failed();
        let outcome = Err(error);
        let mut response = encode_outcome_messages(outcome.clone())?;
        complete_simple_query_action_blocking(
            engine,
            session,
            extended,
            active,
            &outcome,
            &mut response,
        )?;
        let status = session.transaction_status();
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status)?);
        return Ok(response);
    }
    if statements.len() != 1
        && copy_statements
            .iter()
            .any(|copy| !matches!(copy, CopyClassification::NotCopy))
    {
        let error = DbError {
            category: ErrorCategory::Unsupported,
            message: "COPY must be the only statement in a simple Query message".to_string(),
        };
        session.mark_transaction_failed();
        let outcome = Err(error);
        let mut response = encode_outcome_messages(outcome.clone())?;
        complete_simple_query_action_blocking(
            engine,
            session,
            extended,
            active,
            &outcome,
            &mut response,
        )?;
        let status = session.transaction_status();
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status)?);
        return Ok(response);
    }
    if let Some(copy) = copy_statements.into_iter().next() {
        match copy {
            CopyClassification::NotCopy => {}
            CopyClassification::Unsupported => {
                let error = DbError {
                    category: ErrorCategory::Unsupported,
                    message: CopyWireError::unsupported().message,
                };
                session.mark_transaction_failed();
                let outcome = Err(error);
                let mut response = encode_outcome_messages(outcome.clone())?;
                complete_simple_query_action_blocking(
                    engine,
                    session,
                    extended,
                    active,
                    &outcome,
                    &mut response,
                )?;
                let status = session.transaction_status();
                extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
                response.extend_from_slice(&encode_ready(status)?);
                return Ok(response);
            }
            CopyClassification::Supported(CopyStatement::To(copy)) => {
                let result = if active.is_cancelled() {
                    Err(CopyWireError::query_cancelled())
                } else {
                    let result = execute_copy_to(engine, session, &copy);
                    cancel_effect_free_success(active, result, CopyWireError::query_cancelled)
                };
                let lifecycle = result
                    .as_ref()
                    .map(|_| copy_lifecycle_success())
                    .map_err(copy_lifecycle_error);
                if result.is_err() {
                    session.mark_transaction_failed();
                }
                let mut response = match result {
                    Ok(response) => response,
                    Err(error) => encode_copy_error(&error),
                };
                complete_simple_query_action_blocking(
                    engine,
                    session,
                    extended,
                    active,
                    &lifecycle,
                    &mut response,
                )?;
                let status = session.transaction_status();
                extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
                response.extend_from_slice(&encode_ready(status)?);
                return Ok(response);
            }
            CopyClassification::Supported(CopyStatement::From(copy)) => {
                let result = if active.is_cancelled() {
                    Err(CopyWireError::query_cancelled())
                } else {
                    let result = begin_copy_from(engine, session, copy, true);
                    cancel_effect_free_success(active, result, CopyWireError::query_cancelled)
                };
                return match result {
                    Ok((state, response)) => {
                        *copy_in = Some(state);
                        Ok(response)
                    }
                    Err(error) => {
                        session.mark_transaction_failed();
                        let lifecycle = Err(copy_lifecycle_error(&error));
                        let mut response = encode_copy_error(&error);
                        complete_simple_query_action_blocking(
                            engine,
                            session,
                            extended,
                            active,
                            &lifecycle,
                            &mut response,
                        )?;
                        let status = session.transaction_status();
                        extended
                            .finish_transaction_boundary(status != SessionTransactionStatus::Idle);
                        response.extend_from_slice(&encode_ready(status)?);
                        Ok(response)
                    }
                };
            }
        }
    }
    let mut response = Vec::new();
    let mut last_outcome = Ok(QueryOutcome::Empty);
    for (index, statement) in statements.iter().enumerate() {
        if let Some(begin) =
            simple_query_segment_begin(&statements[index..], session.transaction_status(), extended)
        {
            if let Err(error) = submit_text_cancellable(engine, session, begin, active) {
                response.extend_from_slice(&encode_outcome_messages(Err(error))?);
                break;
            }
            extended.complete_transaction_action(TransactionAction::BeginImplicit, true);
        }
        let mut outcome = match classify_sql_cursor_statement(statement) {
            Ok(Some(action)) => {
                execute_sql_cursor_action_blocking(engine, session, extended, action, active)
            }
            Err(error) => Err(error),
            Ok(None) => {
                match classify_prepared_action(extended, statement, session.transaction_status()) {
                    Ok(Some(action)) => execute_sql_prepared_action_blocking(
                        engine, session, extended, action, active,
                    ),
                    Ok(None) => submit_text_cancellable(engine, session, statement, active),
                    Err(error) => Err(error),
                }
            }
        };
        response.extend_from_slice(&encode_cancellable_outcome_messages(active, &mut outcome)?);
        let failed = outcome.is_err();
        if failed || outcome_is_transaction_control(&outcome) {
            complete_simple_query_action_blocking(
                engine,
                session,
                extended,
                active,
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
            active,
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
    active: &ActiveRequest,
    outcome: &Result<QueryOutcome, DbError>,
    response: &mut Vec<u8>,
) -> io::Result<()> {
    if sql_session::outcome_error_poison_transaction(outcome) {
        session.mark_transaction_failed();
    }
    let action = extended.simple_query_completion_action(outcome);
    let Some(sql) = action.sql() else {
        extended.complete_transaction_action(action, true);
        return Ok(());
    };
    let completion = if matches!(action, TransactionAction::CommitImplicit) {
        submit_text_cancellable(engine, session, sql, active)
    } else {
        submit_text(engine, session, sql)
    };
    match completion {
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

fn cancelled_simple_copy_needs_drain(
    copy_in: &Option<CopyInState>,
    consumed_tag: Option<u8>,
) -> bool {
    copy_in.as_ref().is_some_and(CopyInState::ready_after_done)
        && !matches!(consumed_tag, Some(b'c' | b'f'))
}

fn drain_cancelled_simple_copy_frame(draining: &mut bool, tag: u8) -> bool {
    if !*draining {
        return false;
    }
    match tag {
        b'd' | b'H' => true,
        b'c' | b'f' => {
            *draining = false;
            true
        }
        _ => {
            *draining = false;
            false
        }
    }
}

fn run_shared_query_loop(
    stream: &mut dyn ReadWrite,
    engine: &SharedEngine,
    session: &mut SharedSession,
    cancellation: &ConnectionCancellation,
    timeout_control: Option<&TcpStream>,
) -> Result<(), String> {
    let mut extended = ExtendedSession::default();
    let mut copy_in = None;
    let mut copy_request: Option<ActiveRequest> = None;
    let mut draining_cancelled_simple_copy = false;
    loop {
        let polled = if copy_in.is_some() {
            let active = copy_request
                .as_ref()
                .expect("COPY state retains its active request");
            match timeout_control {
                Some(timeout_control) => {
                    read_tagged_frame_polling_cancel(stream, timeout_control, || {
                        active.is_cancelled()
                    })
                    .map_err(|error| error.to_string())?
                }
                None if active.is_cancelled() => PolledTaggedFrame::Cancelled,
                None => PolledTaggedFrame::Frame(
                    read_tagged_frame(stream).map_err(|error| error.to_string())?,
                ),
            }
        } else {
            PolledTaggedFrame::Frame(read_tagged_frame(stream).map_err(|error| error.to_string())?)
        };
        let frame = match polled {
            PolledTaggedFrame::Frame(Some(frame)) => frame,
            PolledTaggedFrame::Frame(None) => break,
            PolledTaggedFrame::Cancelled => {
                draining_cancelled_simple_copy = cancelled_simple_copy_needs_drain(&copy_in, None);
                let active = copy_request
                    .as_ref()
                    .expect("COPY cancellation retains its active request");
                cancel_copy_blocking(stream, engine, session, &mut extended, &mut copy_in, active)?;
                copy_request.take();
                continue;
            }
        };
        if drain_cancelled_simple_copy_frame(&mut draining_cancelled_simple_copy, frame[0]) {
            continue;
        }
        if copy_in.is_some() {
            let active = copy_request
                .take()
                .expect("COPY frame retains its active request");
            if active.is_cancelled() {
                draining_cancelled_simple_copy =
                    cancelled_simple_copy_needs_drain(&copy_in, Some(frame[0]));
                cancel_copy_blocking(
                    stream,
                    engine,
                    session,
                    &mut extended,
                    &mut copy_in,
                    &active,
                )?;
            } else {
                let simple_copy = copy_in.as_ref().is_some_and(CopyInState::ready_after_done);
                if !handle_copy_frame_blocking(
                    stream,
                    engine,
                    session,
                    &mut extended,
                    &mut copy_in,
                    &active,
                    &frame,
                )? {
                    break;
                }
                if simple_copy && copy_in.is_none() && !matches!(frame[0], b'c' | b'f') {
                    draining_cancelled_simple_copy = true;
                }
            }
            if copy_in.is_some() {
                copy_request = Some(active);
            }
            continue;
        }
        let active = cancellation
            .begin_request()
            .map_err(|error| error.to_string())?;
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
        if active.is_cancelled()
            && !matches!(
                &message,
                gpu_db_protocol::FrontendMessage::SimpleQuery(_)
                    | gpu_db_protocol::FrontendMessage::Sync
                    | gpu_db_protocol::FrontendMessage::Terminate
            )
        {
            session.mark_transaction_failed();
            extended.fail();
            stream
                .write_all(&encode_extended_error(extended::ExtendedError::new(
                    "57014",
                    "canceling statement due to user request",
                )))
                .map_err(|error| error.to_string())?;
            continue;
        }
        let transaction_action =
            extended.before_dispatch_transaction_action(&message, session.transaction_status());
        if let Some(sql) = transaction_action.sql() {
            if let Err(error) = submit_text_cancellable(engine, session, sql, &active) {
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
            ExtendedDispatch::SimpleQuery(sql) => execute_simple_query_blocking(
                engine,
                session,
                &mut extended,
                &mut copy_in,
                &sql,
                &active,
            )
            .map_err(encode_io_error),
            ExtendedDispatch::Prepare(request) => {
                let prepared = ExtendedSession::analyze_prepare(engine, session, &request);
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
                    .and_then(|sql| submit_text_cancellable(engine, session, sql, &active).err());
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
                        match request {
                            ExecutionRequest::Query(bound) => {
                                let outcome =
                                    submit_prepared_cancellable(engine, session, &bound, &active);
                                extended.set_execution_outcome(&portal_name, outcome)?;
                            }
                            ExecutionRequest::Copy {
                                statement: CopyStatement::To(copy),
                                target: _,
                                bound,
                            } => {
                                if active.is_cancelled() {
                                    return Err(extended::ExtendedError::new(
                                        "57014",
                                        "canceling statement due to user request",
                                    ));
                                }
                                let response =
                                    execute_prepared_copy_to(engine, session, &bound, &copy)
                                        .map_err(|error| {
                                            extended::ExtendedError::new(error.code, error.message)
                                        })?;
                                if active.is_cancelled() {
                                    return Err(extended::ExtendedError::new(
                                        "57014",
                                        "canceling statement due to user request",
                                    ));
                                }
                                extended.complete_copy_execute(&portal_name)?;
                                return Ok(response);
                            }
                            ExecutionRequest::Copy {
                                statement: CopyStatement::From(copy),
                                target,
                                bound: _,
                            } => {
                                let target = target.ok_or_else(|| {
                                    extended::ExtendedError::new(
                                        "XX000",
                                        "COPY FROM portal lost its target proof",
                                    )
                                })?;
                                if active.is_cancelled() {
                                    return Err(extended::ExtendedError::new(
                                        "57014",
                                        "canceling statement due to user request",
                                    ));
                                }
                                let (state, response) =
                                    begin_prepared_copy_from(engine, session, copy, target, false)
                                        .map_err(|error| {
                                            extended::ExtendedError::new(error.code, error.message)
                                        })?;
                                if active.is_cancelled() {
                                    return Err(extended::ExtendedError::new(
                                        "57014",
                                        "canceling statement due to user request",
                                    ));
                                }
                                extended.complete_copy_execute(&portal_name)?;
                                copy_in = Some(state);
                                return Ok(response);
                            }
                        }
                    }
                    encode_execute_cancellable(&mut extended, &portal_name, max_rows, &active)
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
        if copy_in.is_some() {
            copy_request = Some(active);
        }
    }
    Ok(())
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
    let cancellations = Arc::new(CancellationRegistry::new());
    // One coalescer for the whole server when batching is on; `None` keeps the unchanged path.
    let batcher = batching.then(|| Arc::new(PointLookupBatcher::new(Arc::clone(&engine))));
    loop {
        let (stream, _addr) = listener.accept().await?;
        let _ = stream.set_nodelay(true);
        let engine = Arc::clone(&engine);
        let executor = Arc::clone(&executor);
        let batcher = batcher.clone();
        let cancellations = Arc::clone(&cancellations);
        tokio::spawn(async move {
            if let Err(err) = handle_connection_async(
                stream,
                &engine,
                &executor,
                batcher.as_ref(),
                &cancellations,
            )
            .await
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
    cancellations: &Arc<CancellationRegistry>,
) -> Result<(), String> {
    let Some(cancellation) = complete_startup_async(&mut stream, cancellations).await? else {
        return Ok(());
    };
    let session = Arc::new(std::sync::Mutex::new(engine.open_session()));
    let result = run_async_query_loop(
        &mut stream,
        engine,
        executor,
        batcher,
        &session,
        &cancellation,
    )
    .await;
    let mut session = session
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _ = engine.submit(&mut session, SubmissionRequest::CloseSession);
    result
}

async fn complete_startup_async(
    stream: &mut TokioTcpStream,
    cancellations: &Arc<CancellationRegistry>,
) -> Result<Option<ConnectionCancellation>, String> {
    let mut frame = match read_startup_frame_async(stream).await? {
        Some(frame) => frame,
        None => return Ok(None),
    };
    loop {
        if security::is_malformed_cancel_request(&frame) {
            return Ok(None);
        }
        match parse_startup_packet(&frame).map_err(|err| err.to_string())? {
            StartupPacket::SslRequest | StartupPacket::GssEncRequest => {
                stream
                    .write_all(b"N")
                    .await
                    .map_err(|err| err.to_string())?;
                frame = match read_startup_frame_async(stream).await? {
                    Some(frame) => frame,
                    None => return Ok(None),
                };
            }
            StartupPacket::CancelRequest {
                process_id,
                secret_key,
            } => {
                cancellations.cancel(process_id, &secret_key);
                return Ok(None);
            }
            StartupPacket::Startup { .. } => break,
        }
    }
    let cancellation = cancellations.register();
    let handshake =
        encode_startup_handshake(cancellation.backend_key()).map_err(|err| err.to_string())?;
    stream
        .write_all(&handshake)
        .await
        .map_err(|err| err.to_string())?;
    Ok(Some(cancellation))
}

async fn run_async_query_loop(
    stream: &mut TokioTcpStream,
    engine: &Arc<SharedEngine>,
    executor: &Arc<tokio::sync::Semaphore>,
    batcher: Option<&Arc<PointLookupBatcher>>,
    session: &Arc<std::sync::Mutex<SharedSession>>,
    cancellation: &ConnectionCancellation,
) -> Result<(), String> {
    let mut extended = ExtendedSession::default();
    let mut copy_in = None;
    let mut copy_request: Option<ActiveRequest> = None;
    let mut draining_cancelled_simple_copy = false;
    loop {
        let polled = if copy_in.is_some() {
            let active = copy_request
                .as_ref()
                .expect("COPY state retains its active request");
            read_tagged_frame_polling_cancel_async(stream, active.cancelled()).await?
        } else {
            PolledTaggedFrame::Frame(read_tagged_frame_async(stream).await?)
        };
        let frame = match polled {
            PolledTaggedFrame::Frame(Some(frame)) => frame,
            PolledTaggedFrame::Frame(None) => break,
            PolledTaggedFrame::Cancelled => {
                draining_cancelled_simple_copy = cancelled_simple_copy_needs_drain(&copy_in, None);
                let active = copy_request
                    .as_ref()
                    .expect("COPY cancellation retains its active request");
                cancel_copy_async(
                    stream,
                    Arc::clone(engine),
                    Arc::clone(session),
                    executor,
                    &mut extended,
                    &mut copy_in,
                    active,
                )
                .await?;
                copy_request.take();
                continue;
            }
        };
        if drain_cancelled_simple_copy_frame(&mut draining_cancelled_simple_copy, frame[0]) {
            continue;
        }
        if copy_in.is_some() {
            let active = copy_request
                .take()
                .expect("COPY frame retains its active request");
            if active.is_cancelled() {
                draining_cancelled_simple_copy =
                    cancelled_simple_copy_needs_drain(&copy_in, Some(frame[0]));
                cancel_copy_async(
                    stream,
                    Arc::clone(engine),
                    Arc::clone(session),
                    executor,
                    &mut extended,
                    &mut copy_in,
                    &active,
                )
                .await?;
            } else {
                let simple_copy = copy_in.as_ref().is_some_and(CopyInState::ready_after_done);
                let frame_tag = frame[0];
                if !handle_copy_frame_async(
                    stream,
                    Arc::clone(engine),
                    Arc::clone(session),
                    executor,
                    &mut extended,
                    &mut copy_in,
                    &active,
                    frame,
                )
                .await?
                {
                    break;
                }
                if simple_copy && copy_in.is_none() && !matches!(frame_tag, b'c' | b'f') {
                    draining_cancelled_simple_copy = true;
                }
            }
            if copy_in.is_some() {
                copy_request = Some(active);
            }
            continue;
        }
        let active = cancellation
            .begin_request()
            .map_err(|error| error.to_string())?;
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
        if active.is_cancelled()
            && !matches!(
                &message,
                gpu_db_protocol::FrontendMessage::SimpleQuery(_)
                    | gpu_db_protocol::FrontendMessage::Sync
                    | gpu_db_protocol::FrontendMessage::Terminate
            )
        {
            session
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .mark_transaction_failed();
            extended.fail();
            stream
                .write_all(&encode_extended_error(extended::ExtendedError::new(
                    "57014",
                    "canceling statement due to user request",
                )))
                .await
                .map_err(|error| error.to_string())?;
            continue;
        }
        let transaction_action = extended.before_dispatch_transaction_action(
            &message,
            shared_session_transaction_status(session),
        );
        if let Some(sql) = transaction_action.sql() {
            let begin = execute_shared_session_blocking_cancellable(
                Arc::clone(engine),
                Arc::clone(session),
                executor,
                sql.to_string(),
                &active,
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
                &mut copy_in,
                sql,
                &active,
            )
            .await
            .map_err(|error| encode_io_error(io::Error::other(error))),
            ExtendedDispatch::Prepare(request) => {
                let prepared = analyze_prepare_cancellable(
                    Arc::clone(engine),
                    Arc::clone(session),
                    executor,
                    (*request).clone(),
                    &active,
                )
                .await?;
                let prepared = cancel_effect_free_success(&active, prepared, cancellation_error);
                extended
                    .complete_parse(*request, prepared)
                    .map_err(encode_extended_error)
            }
            ExtendedDispatch::Bind(request) => {
                let completion = bind_cancellable(executor, *request, &active).await?;
                let completion = cancel_effect_free_success(
                    &active,
                    completion,
                    wire_response::cancelled_extended_error,
                );
                extended
                    .complete_bind(completion)
                    .map_err(encode_extended_error)
            }
            ExtendedDispatch::Describe { target, name } => {
                let status = shared_session_transaction_status(session);
                match extended.description_owner(target, &name, status) {
                    Err(error) => Err(encode_extended_error(error)),
                    Ok(owner) => {
                        let validation = revalidate_description_cancellable(
                            Arc::clone(engine),
                            Arc::clone(session),
                            executor,
                            owner,
                            &active,
                        )
                        .await?;
                        match cancel_effect_free_success(
                            &active,
                            validation,
                            wire_response::cancelled_extended_error,
                        ) {
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
                    execute_shared_session_blocking_cancellable(
                        Arc::clone(engine),
                        Arc::clone(session),
                        executor,
                        sql.to_string(),
                        &active,
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
                    Ok(Some(ExecutionRequest::Query(bound))) => {
                        let outcome = execute_prepared_shared_session_blocking_cancellable(
                            Arc::clone(engine),
                            Arc::clone(session),
                            executor,
                            *bound,
                            &active,
                        )
                        .await?;
                        extended
                            .set_execution_outcome(&portal_name, outcome)
                            .and_then(|()| {
                                encode_execute_cancellable(
                                    &mut extended,
                                    &portal_name,
                                    max_rows,
                                    &active,
                                )
                            })
                            .map_err(encode_extended_error)
                    }
                    Ok(Some(ExecutionRequest::Copy {
                        statement: CopyStatement::To(copy),
                        target: _,
                        bound,
                    })) => {
                        if active.is_cancelled() {
                            Err(encode_extended_error(extended::ExtendedError::new(
                                "57014",
                                "canceling statement due to user request",
                            )))
                        } else {
                            let result = execute_prepared_copy_to_async(
                                Arc::clone(engine),
                                Arc::clone(session),
                                executor,
                                *bound,
                                copy,
                                &active,
                            )
                            .await?;
                            if active.is_cancelled() && result.is_ok() {
                                Err(encode_extended_error(extended::ExtendedError::new(
                                    "57014",
                                    "canceling statement due to user request",
                                )))
                            } else {
                                match result {
                                    Ok(response) => extended
                                        .complete_copy_execute(&portal_name)
                                        .map(|()| response)
                                        .map_err(encode_extended_error),
                                    Err(error) => Err(encode_extended_error(
                                        extended::ExtendedError::new(error.code, error.message),
                                    )),
                                }
                            }
                        }
                    }
                    Ok(Some(ExecutionRequest::Copy {
                        statement: CopyStatement::From(copy),
                        target,
                        bound: _,
                    })) => match target {
                        None => Err(encode_extended_error(extended::ExtendedError::new(
                            "XX000",
                            "COPY FROM portal lost its target proof",
                        ))),
                        Some(target) => {
                            if active.is_cancelled() {
                                Err(encode_extended_error(extended::ExtendedError::new(
                                    "57014",
                                    "canceling statement due to user request",
                                )))
                            } else {
                                let result = begin_prepared_copy_from_async(
                                    Arc::clone(engine),
                                    Arc::clone(session),
                                    executor,
                                    copy,
                                    target,
                                    false,
                                    &active,
                                )
                                .await?;
                                if active.is_cancelled() && result.is_ok() {
                                    Err(encode_extended_error(extended::ExtendedError::new(
                                        "57014",
                                        "canceling statement due to user request",
                                    )))
                                } else {
                                    match result {
                                        Ok((state, response)) => extended
                                            .complete_copy_execute(&portal_name)
                                            .map(|()| {
                                                copy_in = Some(state);
                                                response
                                            })
                                            .map_err(encode_extended_error),
                                        Err(error) => Err(encode_extended_error(
                                            extended::ExtendedError::new(error.code, error.message),
                                        )),
                                    }
                                }
                            }
                        }
                    },
                    Ok(None) => {
                        encode_execute_cancellable(&mut extended, &portal_name, max_rows, &active)
                            .map_err(encode_extended_error)
                    }
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
        if copy_in.is_some() {
            copy_request = Some(active);
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

#[allow(clippy::too_many_arguments)]
async fn execute_simple_query_async(
    engine: Arc<SharedEngine>,
    session: Arc<std::sync::Mutex<SharedSession>>,
    executor: &Arc<tokio::sync::Semaphore>,
    batcher: Option<Arc<PointLookupBatcher>>,
    extended: &mut ExtendedSession,
    copy_in: &mut Option<CopyInState>,
    sql: String,
    active: &ActiveRequest,
) -> Result<Vec<u8>, String> {
    extended.clear_unnamed_for_simple_query();
    if active.is_cancelled() {
        let outcome = Err(cancellation_error());
        let mut response =
            encode_outcome_messages(outcome.clone()).map_err(|error| error.to_string())?;
        complete_simple_query_action_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            executor,
            extended,
            active,
            &outcome,
            &mut response,
        )
        .await?;
        let status = shared_session_transaction_status(&session);
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
        return Ok(response);
    }
    let statements = split_simple_query(&sql)
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if statements.is_empty() {
        let outcome = cancellation_checked_outcome(active, Ok(QueryOutcome::Empty));
        let mut response =
            encode_outcome_messages(outcome.clone()).map_err(|error| error.to_string())?;
        complete_simple_query_action_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            executor,
            extended,
            active,
            &outcome,
            &mut response,
        )
        .await?;
        let status = shared_session_transaction_status(&session);
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
        return Ok(response);
    }
    let copy_statements = statements
        .iter()
        .map(|statement| classify_copy_statement(statement))
        .collect::<Vec<_>>();
    if let Err(error) = preflight_simple_query(&statements, &copy_statements) {
        mark_shared_session_failed(&session);
        let outcome = Err(error);
        let mut response =
            encode_outcome_messages(outcome.clone()).map_err(|error| error.to_string())?;
        complete_simple_query_action_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            executor,
            extended,
            active,
            &outcome,
            &mut response,
        )
        .await?;
        let status = shared_session_transaction_status(&session);
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
        return Ok(response);
    }
    if statements.len() != 1
        && copy_statements
            .iter()
            .any(|copy| !matches!(copy, CopyClassification::NotCopy))
    {
        let outcome = Err(DbError {
            category: ErrorCategory::Unsupported,
            message: "COPY must be the only statement in a simple Query message".to_string(),
        });
        mark_shared_session_failed(&session);
        let mut response =
            encode_outcome_messages(outcome.clone()).map_err(|error| error.to_string())?;
        complete_simple_query_action_async(
            Arc::clone(&engine),
            Arc::clone(&session),
            executor,
            extended,
            active,
            &outcome,
            &mut response,
        )
        .await?;
        let status = shared_session_transaction_status(&session);
        extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
        response.extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
        return Ok(response);
    }
    if let Some(copy) = copy_statements.into_iter().next() {
        match copy {
            CopyClassification::NotCopy => {}
            CopyClassification::Unsupported => {
                let outcome = Err(DbError {
                    category: ErrorCategory::Unsupported,
                    message: CopyWireError::unsupported().message,
                });
                mark_shared_session_failed(&session);
                let mut response =
                    encode_outcome_messages(outcome.clone()).map_err(|error| error.to_string())?;
                complete_simple_query_action_async(
                    Arc::clone(&engine),
                    Arc::clone(&session),
                    executor,
                    extended,
                    active,
                    &outcome,
                    &mut response,
                )
                .await?;
                let status = shared_session_transaction_status(&session);
                extended.finish_transaction_boundary(status != SessionTransactionStatus::Idle);
                response
                    .extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
                return Ok(response);
            }
            CopyClassification::Supported(CopyStatement::To(copy)) => {
                let result = if active.is_cancelled() {
                    Err(CopyWireError::query_cancelled())
                } else {
                    let result = execute_copy_to_async(
                        Arc::clone(&engine),
                        Arc::clone(&session),
                        executor,
                        copy,
                        active,
                    )
                    .await?;
                    cancel_effect_free_success(active, result, CopyWireError::query_cancelled)
                };
                let lifecycle = result
                    .as_ref()
                    .map(|_| copy_lifecycle_success())
                    .map_err(copy_lifecycle_error);
                if result.is_err() {
                    mark_shared_session_failed(&session);
                }
                let mut response = match result {
                    Ok(response) => response,
                    Err(error) => encode_copy_error(&error),
                };
                complete_simple_query_action_async(
                    Arc::clone(&engine),
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
                response
                    .extend_from_slice(&encode_ready(status).map_err(|error| error.to_string())?);
                return Ok(response);
            }
            CopyClassification::Supported(CopyStatement::From(copy)) => {
                let result = if active.is_cancelled() {
                    Err(CopyWireError::query_cancelled())
                } else {
                    let result = begin_copy_from_async(
                        Arc::clone(&engine),
                        Arc::clone(&session),
                        executor,
                        copy,
                        true,
                        active,
                    )
                    .await?;
                    cancel_effect_free_success(active, result, CopyWireError::query_cancelled)
                };
                return match result {
                    Ok((state, response)) => {
                        *copy_in = Some(state);
                        Ok(response)
                    }
                    Err(error) => {
                        mark_shared_session_failed(&session);
                        let lifecycle = Err(copy_lifecycle_error(&error));
                        let mut response = encode_copy_error(&error);
                        complete_simple_query_action_async(
                            Arc::clone(&engine),
                            Arc::clone(&session),
                            executor,
                            extended,
                            active,
                            &lifecycle,
                            &mut response,
                        )
                        .await?;
                        let status = shared_session_transaction_status(&session);
                        extended
                            .finish_transaction_boundary(status != SessionTransactionStatus::Idle);
                        response.extend_from_slice(
                            &encode_ready(status).map_err(|error| error.to_string())?,
                        );
                        Ok(response)
                    }
                };
            }
        }
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
            let begin = if active.is_cancelled() {
                Err(cancellation_error())
            } else {
                execute_shared_session_blocking_cancellable(
                    Arc::clone(&engine),
                    Arc::clone(&session),
                    executor,
                    begin.to_string(),
                    active,
                )
                .await?
            };
            let begin = cancellation_checked_outcome(active, begin);
            if let Err(error) = begin {
                response.extend_from_slice(
                    &encode_outcome_messages(Err(error)).map_err(|error| error.to_string())?,
                );
                break;
            }
            extended.complete_transaction_action(TransactionAction::BeginImplicit, true);
        }
        let sql_cursor = classify_sql_cursor_statement(statement);
        let outcome = match sql_cursor {
            Err(error) => Err(error),
            Ok(Some(action)) => {
                execute_sql_cursor_action_async(
                    Arc::clone(&engine),
                    Arc::clone(&session),
                    executor,
                    extended,
                    action,
                    active,
                )
                .await?
            }
            Ok(None) => match classify_prepared_action(
                extended,
                statement,
                shared_session_transaction_status(&session),
            ) {
                Err(error) => Err(error),
                Ok(Some(action)) => {
                    execute_sql_prepared_action_async(
                        Arc::clone(&engine),
                        Arc::clone(&session),
                        executor,
                        extended,
                        action,
                        active,
                    )
                    .await?
                }
                Ok(None) if active.is_cancelled() => Err(cancellation_error()),
                Ok(None) if single_can_batch => match &batcher {
                    Some(batcher) => {
                        execute_batchable_or_fallback(
                            Arc::clone(&engine),
                            executor,
                            Arc::clone(batcher),
                            Arc::clone(&session),
                            statement.clone(),
                            active,
                        )
                        .await?
                    }
                    None => {
                        execute_shared_session_blocking_cancellable(
                            Arc::clone(&engine),
                            Arc::clone(&session),
                            executor,
                            statement.clone(),
                            active,
                        )
                        .await?
                    }
                },
                Ok(None) => {
                    execute_shared_session_blocking_cancellable(
                        Arc::clone(&engine),
                        Arc::clone(&session),
                        executor,
                        statement.clone(),
                        active,
                    )
                    .await?
                }
            },
        };
        let mut outcome = cancellation_checked_outcome(active, outcome);
        response.extend_from_slice(
            &encode_cancellable_outcome_messages(active, &mut outcome)
                .map_err(|error| error.to_string())?,
        );
        let failed = outcome.is_err();
        if failed || outcome_is_transaction_control(&outcome) {
            complete_simple_query_action_async(
                Arc::clone(&engine),
                Arc::clone(&session),
                executor,
                extended,
                active,
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
            active,
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

#[cfg(test)]
mod tests;
