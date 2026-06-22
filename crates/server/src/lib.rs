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
//! - **Simple query protocol only, one statement per message.** Extended-protocol
//!   (Parse/Bind/Execute), COPY, auth (SCRAM/TLS), and the catalog/introspection
//!   surface are out of scope here — those live in the legacy full-compatibility
//!   `gpu-db-server` and are migrated behind the façade in later milestones.
//!   **Multi-statement simple queries are not yet supported:** a single `Query`
//!   message containing `;`-separated statements (e.g. `SELECT 1; SELECT 2`,
//!   common in migration scripts) is rejected with one error rather than executed
//!   statement-by-statement. The wire stays in sync (one message in, one error +
//!   ReadyForQuery out); splitting on top-level `;` is a tracked follow-up.
//!   Empty statements correctly return `EmptyQueryResponse`.
//! - **Concurrent dispatch (P1-M4).** `serve` shares one engine across a worker pool
//!   (`Arc<SharedEngine>`, thread-per-connection on the existing blocking sockets) and
//!   dispatches each statement through `execute_on_shared_engine`: read-only statements
//!   take a read lock and run **concurrently**; writes take a write lock and **serialize**
//!   ("N concurrent readers, one serialized writer"). This is the first production caller
//!   of the `&self` engine read path (P1-M3) + the GPU shared-context substrate (P2-M1).
//!   `serve_sequential` retains the prior one-connection-at-a-time loop as the A/B baseline.
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
    execute_on_shared_engine, execute_on_shared_engine_batched, pg_adapter, BatchedDispatch,
    DbError, EngineFacade, PointLookupBatcher, QueryOutcome, SharedEngine,
};
use gpu_db_protocol::backend::{BackendColumn, BackendError, BackendWriter};
use gpu_db_protocol::{
    parse_frontend_message, parse_startup_packet, FrontendMessage, StartupPacket,
};

/// Maximum accepted pgwire frame length (DoS guard): a malicious/huge length prefix would
/// otherwise `resize` a buffer to that size before reading a byte — reachable pre-auth, and
/// more exposed now that async ingress holds many untrusted connections. 64 MiB is far above
/// any reasonable simple-query statement (bulk payloads belong in COPY, out of scope here).
const MAX_FRAME_LEN: usize = 64 * 1024 * 1024;

/// Serve connections **concurrently** on `listener`: one engine shared across a
/// thread-per-connection worker pool (`Arc<SharedEngine>`), each statement dispatched
/// through `execute_on_shared_engine` — read-only statements take a read lock and run
/// concurrently; writes take a write lock and serialize. Blocks until the listener stops.
/// This is the P1-M4 concurrent dispatch (the first production caller of the `&self` engine
/// read path); `serve_sequential` is the prior one-at-a-time loop, kept as the A/B baseline.
pub fn serve(listener: TcpListener) -> io::Result<()> {
    serve_with_engine(listener, Arc::new(SharedEngine::new()))
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
            if let Err(err) = handle_connection_shared(&mut stream, &engine) {
                eprintln!("gpu-db-engine-server connection error: {err}");
            }
        });
    }
    Ok(())
}

/// Serve connections sequentially on `listener` (one connection at a time, one owned
/// façade). Retained as the A/B baseline for the P1-M4 concurrency benchmark.
pub fn serve_sequential(listener: TcpListener) -> io::Result<()> {
    let mut facade = EngineFacade::new();
    for stream in listener.incoming() {
        let mut stream = stream?;
        let _ = stream.set_nodelay(true);
        if let Err(err) = handle_connection(&mut stream, &mut facade) {
            eprintln!("gpu-db-engine-server connection error: {err}");
        }
    }
    Ok(())
}

/// Handle one connection against the shared engine: startup handshake, then a simple-query
/// loop dispatched through `execute_on_shared_engine` (an implicit session per connection).
fn handle_connection_shared(stream: &mut TcpStream, engine: &SharedEngine) -> Result<(), String> {
    if !complete_startup(stream)? {
        return Ok(());
    }
    run_shared_query_loop(stream, engine)
}

fn run_shared_query_loop(stream: &mut TcpStream, engine: &SharedEngine) -> Result<(), String> {
    while let Some(frame) = read_tagged_frame(stream).map_err(|err| err.to_string())? {
        match parse_frontend_message(&frame).map_err(|err| err.to_string())? {
            FrontendMessage::SimpleQuery(sql) => {
                let outcome = execute_on_shared_engine(engine, &sql);
                write_outcome(stream, outcome).map_err(|err| err.to_string())?;
            }
            FrontendMessage::Terminate => break,
            FrontendMessage::Sync => {
                let mut writer = BackendWriter::new(&mut *stream);
                writer
                    .ready_for_query(false)
                    .map_err(|err| err.to_string())?;
            }
            _ => {
                let mut writer = BackendWriter::new(&mut *stream);
                writer
                    .error_response(&BackendError::new(
                        "0A000",
                        "only the simple query protocol is supported by the engine-backed \
                         facade server",
                    ))
                    .map_err(|err| err.to_string())?;
                writer
                    .ready_for_query(false)
                    .map_err(|err| err.to_string())?;
            }
        }
    }
    Ok(())
}

/// Handle one client connection: startup handshake, then a simple-query loop
/// routed through the façade. Each connection runs on its own façade session.
pub fn handle_connection(stream: &mut TcpStream, facade: &mut EngineFacade) -> Result<(), String> {
    if !complete_startup(stream)? {
        return Ok(());
    }

    let session = facade.open_session();
    let result = run_simple_query_loop(stream, facade, session);
    facade.close_session(session);
    result
}

/// Drive the startup handshake. Returns `Ok(false)` if the client disconnected
/// before sending a real startup message.
fn complete_startup(stream: &mut TcpStream) -> Result<bool, String> {
    let mut frame = match read_startup_frame(stream).map_err(|err| err.to_string())? {
        Some(frame) => frame,
        None => return Ok(false),
    };
    loop {
        match parse_startup_packet(&frame).map_err(|err| err.to_string())? {
            StartupPacket::SslRequest | StartupPacket::GssEncRequest => {
                // No TLS/GSS on this server: decline and read the next frame.
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

fn run_simple_query_loop(
    stream: &mut TcpStream,
    facade: &mut EngineFacade,
    session: gpu_db_facade::SessionId,
) -> Result<(), String> {
    while let Some(frame) = read_tagged_frame(stream).map_err(|err| err.to_string())? {
        match parse_frontend_message(&frame).map_err(|err| err.to_string())? {
            FrontendMessage::SimpleQuery(sql) => {
                let outcome = facade.execute(session, &sql);
                write_outcome(stream, outcome).map_err(|err| err.to_string())?;
            }
            FrontendMessage::Terminate => break,
            FrontendMessage::Sync => {
                let mut writer = BackendWriter::new(&mut *stream);
                writer
                    .ready_for_query(false)
                    .map_err(|err| err.to_string())?;
            }
            _ => {
                let mut writer = BackendWriter::new(&mut *stream);
                writer
                    .error_response(&BackendError::new(
                        "0A000",
                        "only the simple query protocol is supported by the engine-backed \
                         facade server",
                    ))
                    .map_err(|err| err.to_string())?;
                writer
                    .ready_for_query(false)
                    .map_err(|err| err.to_string())?;
            }
        }
    }
    Ok(())
}

/// Encode a neutral façade outcome into pgwire backend message bytes (EmptyQueryResponse /
/// RowDescription+DataRow+CommandComplete / ErrorResponse, then ReadyForQuery). All
/// PostgreSQL-specific encoding lives in `gpu_db_facade::pg_adapter`. Built into a buffer so
/// it is shared by the sync (`write_outcome`) and async (`serve_async`) write paths.
fn encode_outcome(outcome: Result<QueryOutcome, DbError>) -> io::Result<Vec<u8>> {
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
                if let QueryOutcome::Rows { columns, rows } = &outcome {
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
                        let values: Vec<Option<String>> = row
                            .iter()
                            .map(pg_adapter::db_value_text_opt)
                            .collect();
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
        writer.ready_for_query(false)?;
    }
    Ok(buf)
}

/// Translate a neutral façade outcome into pgwire backend messages on the sync stream.
fn write_outcome(stream: &mut TcpStream, outcome: Result<QueryOutcome, DbError>) -> io::Result<()> {
    stream.write_all(&encode_outcome(outcome)?)
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
        Arc::new(SharedEngine::new()),
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
    run_async_query_loop(&mut stream, engine, executor, batcher).await
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
) -> Result<(), String> {
    while let Some(frame) = read_tagged_frame_async(stream).await? {
        match parse_frontend_message(&frame).map_err(|err| err.to_string())? {
            FrontendMessage::SimpleQuery(sql) => {
                let outcome = match batcher {
                    // Batching ON: classify-or-fallback. Classification + the unchanged
                    // per-query path still run under a permit on the blocking pool; a
                    // batchable point-lookup instead parks on a `oneshot` with NO permit
                    // held and NO `spawn_blocking` (the coalescer thread does the GPU work),
                    // so many parked lookups coalesce into one submission.
                    Some(batcher) => {
                        execute_batchable_or_fallback(
                            Arc::clone(engine),
                            executor,
                            Arc::clone(batcher),
                            sql,
                        )
                        .await?
                    }
                    // Batching OFF (GPU_DB_BATCHING=0 escape hatch): the original path, unchanged.
                    None => {
                        let _permit = executor.acquire().await.map_err(|err| err.to_string())?;
                        let engine = Arc::clone(engine);
                        tokio::task::spawn_blocking(move || execute_on_shared_engine(&engine, &sql))
                            .await
                            .map_err(|err| err.to_string())?
                    }
                };
                let buf = encode_outcome(outcome).map_err(|err| err.to_string())?;
                stream
                    .write_all(&buf)
                    .await
                    .map_err(|err| err.to_string())?;
            }
            FrontendMessage::Terminate => break,
            FrontendMessage::Sync => {
                let mut buf = Vec::new();
                BackendWriter::new(&mut buf)
                    .ready_for_query(false)
                    .map_err(|err| err.to_string())?;
                stream
                    .write_all(&buf)
                    .await
                    .map_err(|err| err.to_string())?;
            }
            _ => {
                let mut buf = Vec::new();
                {
                    let mut writer = BackendWriter::new(&mut buf);
                    writer
                        .error_response(&BackendError::new(
                            "0A000",
                            "only the simple query protocol is supported by the engine-backed \
                             facade server",
                        ))
                        .map_err(|err| err.to_string())?;
                    writer
                        .ready_for_query(false)
                        .map_err(|err| err.to_string())?;
                }
                stream
                    .write_all(&buf)
                    .await
                    .map_err(|err| err.to_string())?;
            }
        }
    }
    Ok(())
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
    sql: String,
) -> Result<Result<QueryOutcome, DbError>, String> {
    // Phase 1 (permit held): classify under the engine read lock and either resolve the
    // unchanged path or enqueue on the batcher. Both are short or self-bounded.
    let dispatch = {
        let _permit = executor.acquire().await.map_err(|err| err.to_string())?;
        tokio::task::spawn_blocking(move || {
            // `execute_on_shared_engine_batched` returns `Immediate(result)` for everything
            // non-batchable (running the unchanged per-query path here) or `Batched(receiver)`.
            match execute_on_shared_engine_batched(&engine, &batcher, &sql) {
                BatchedDispatch::Immediate(result) => DispatchOut::Immediate(result),
                BatchedDispatch::Batched(receiver) => DispatchOut::Batched(receiver),
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

/// Internal owned form of `BatchedDispatch` so it can cross the `spawn_blocking` boundary.
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
mod tests {
    use super::parse_batching_flag;

    /// Thread-3 default-on: an unset `GPU_DB_BATCHING` enables batching. A default-started
    /// async server (no env) therefore constructs a `PointLookupBatcher` and routes batchable
    /// point-lookups through it. (Decoded via the pure helper so the test does not mutate
    /// process-global env, which is racy across the parallel test runner.)
    #[test]
    fn batching_defaults_on_when_unset() {
        assert!(parse_batching_flag(None));
    }

    /// The disable escape hatch: `0`/`false`/`off`/`no` (case-insensitive, whitespace-tolerant)
    /// turns batching off and restores the per-query path.
    #[test]
    fn explicit_off_values_disable_batching() {
        for off in ["0", "false", "off", "no", "FALSE", "Off", "  no  "] {
            assert!(
                !parse_batching_flag(Some(off)),
                "{off:?} should disable batching"
            );
        }
    }

    /// Everything that is not an explicit off token keeps the default-on behavior — including
    /// the historical truthy values and any unrecognized/empty value (fail safe = on).
    #[test]
    fn truthy_and_unrecognized_values_keep_batching_on() {
        for on in [
            "1", "true", "on", "yes", "TRUE", "On", "", "enabled", "garbage",
        ] {
            assert!(
                parse_batching_flag(Some(on)),
                "{on:?} should keep batching on"
            );
        }
    }
}
