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
//! - **Single-threaded, one connection at a time.** As of P1-M3 step 3c the `Engine`
//!   is `Send + Sync` and its relational read path is `&self` (concurrent reads are
//!   supported and tested at the engine level), but this server still keeps the engine
//!   on the serve thread and handles connections sequentially. Sharing the engine
//!   (`Arc<Engine>`) across IO workers to dispatch reads concurrently — the actual
//!   latency win — is the next step (P1-M3 step 4); the concurrent `&self` read path
//!   has no production caller yet.

use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};

use gpu_db_facade::{pg_adapter, DbError, EngineFacade, QueryOutcome};
use gpu_db_protocol::backend::{BackendColumn, BackendError, BackendWriter};
use gpu_db_protocol::{
    parse_frontend_message, parse_startup_packet, FrontendMessage, StartupPacket,
};

/// Serve connections sequentially on `listener`, executing every statement
/// through a single engine-backed façade. Blocks until the listener stops.
pub fn serve(listener: TcpListener) -> io::Result<()> {
    let mut facade = EngineFacade::new();
    for stream in listener.incoming() {
        let mut stream = stream?;
        if let Err(err) = handle_connection(&mut stream, &mut facade) {
            eprintln!("gpu-db-engine-server connection error: {err}");
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

/// Translate a neutral façade outcome into pgwire backend messages. All
/// PostgreSQL-specific encoding lives in `gpu_db_facade::pg_adapter`.
fn write_outcome(stream: &mut TcpStream, outcome: Result<QueryOutcome, DbError>) -> io::Result<()> {
    let mut writer = BackendWriter::new(&mut *stream);
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
                        .map(|value| Some(pg_adapter::db_value_text(value)))
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
    writer.ready_for_query(false)
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
    let mut frame = vec![tag[0]];
    frame.extend_from_slice(&len);
    frame.resize(frame_len + 1, 0);
    stream.read_exact(&mut frame[5..])?;
    Ok(Some(frame))
}
