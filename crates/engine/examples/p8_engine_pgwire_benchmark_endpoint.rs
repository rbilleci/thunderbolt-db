use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use gpu_db_engine::{Engine, RelationalResidencyWarmupPolicy};
use gpu_db_protocol::backend::{BackendColumn, BackendWriter};
use gpu_db_protocol::{
    parse_command, parse_copy_from_stdin, parse_copy_row, parse_frontend_message,
    parse_startup_packet, Command, CopyFromStdin, FrontendMessage, SqlValue, StartupPacket,
};

const SSL_REQUEST_CODE: u32 = 80877103;

struct PendingCopy {
    copy: CopyFromStdin,
    columns: Vec<gpu_db_protocol::CopyColumn>,
    rows: Vec<Vec<SqlValue>>,
    pending_text: String,
}

impl PendingCopy {
    fn push_bytes(&mut self, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
        self.pending_text.push_str(std::str::from_utf8(bytes)?);
        while let Some(newline) = self.pending_text.find('\n') {
            let mut line = self.pending_text[..newline].to_string();
            if line.ends_with('\r') {
                line.pop();
            }
            self.pending_text.drain(..=newline);
            self.push_line(&line)?;
        }
        Ok(())
    }

    fn finish_pending_text(&mut self) -> Result<(), Box<dyn Error>> {
        if self.pending_text.is_empty() {
            return Ok(());
        }
        let line = std::mem::take(&mut self.pending_text);
        self.push_line(line.trim_end_matches('\r'))?;
        Ok(())
    }

    fn push_line(&mut self, line: &str) -> Result<(), Box<dyn Error>> {
        if line.is_empty() || line == r"\." {
            return Ok(());
        }
        let row = parse_copy_row(
            &self.columns,
            self.copy.columns.as_deref().unwrap(),
            self.copy.options,
            line,
        )?;
        self.rows.push(row);
        Ok(())
    }
}

struct EndpointState {
    engine: Engine,
    next_txn_id: u64,
    facts: File,
}

impl EndpointState {
    fn new(facts_path: &str) -> Result<Self, Box<dyn Error>> {
        let facts = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(facts_path)?;
        Ok(Self {
            engine: Engine::new_local(),
            next_txn_id: 1,
            facts,
        })
    }

    fn take_txn_id(&mut self) -> u64 {
        let txn_id = self.next_txn_id;
        self.next_txn_id += 1;
        txn_id
    }

    fn fact(&mut self, key: &str, value: impl std::fmt::Display) -> io::Result<()> {
        writeln!(self.facts, "{key}={value}")?;
        self.facts.flush()
    }

    fn handle_startup(
        &mut self,
        frame: &[u8],
        output: &mut dyn Write,
    ) -> Result<(), Box<dyn Error>> {
        match parse_startup_packet(frame)? {
            StartupPacket::Startup { .. } => {
                let mut writer = BackendWriter::new(output);
                writer.authentication_ok()?;
                writer.parameter_status("server_version", "16.0-gpu-db-engine-p8")?;
                writer.parameter_status("client_encoding", "UTF8")?;
                writer.parameter_status("DateStyle", "ISO, MDY")?;
                writer.parameter_status("integer_datetimes", "on")?;
                writer.ready_for_query(false)?;
                self.fact("startup_packet_parser_reused", true)?;
                self.fact("backend_startup_messages_written", true)?;
            }
            other => return Err(format!("unexpected startup packet: {other:?}").into()),
        }
        Ok(())
    }

    fn handle_simple_query(
        &mut self,
        sql: &str,
        output: &mut dyn Write,
    ) -> Result<(), Box<dyn Error>> {
        match parse_command(sql)? {
            Command::CreateTable(_) => {
                let txn_id = self.take_txn_id();
                self.engine.execute_text(txn_id, sql)?;
                let mut writer = BackendWriter::new(output);
                writer.command_complete("CREATE TABLE")?;
                writer.ready_for_query(false)?;
                self.fact("create_table_into_engine_wal_mvcc", true)?;
            }
            Command::Select(select) => {
                let before = self.engine.metrics().snapshot();
                let result = self.engine.execute_relational_select(&select)?;
                let after = self.engine.metrics().snapshot();
                let decision = self
                    .engine
                    .status_snapshot()
                    .relational_residency
                    .latest_route_decision(&select.table)
                    .cloned();
                let columns = result
                    .columns
                    .iter()
                    .map(|column| {
                        BackendColumn::new(&column.name, column.type_oid, column.type_size)
                    })
                    .collect::<Vec<_>>();
                let rows = result
                    .rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|value| Some(sql_value_text(value)))
                            .collect()
                    })
                    .collect::<Vec<Vec<_>>>();
                let mut writer = BackendWriter::new(output);
                writer.select_rows(&columns, &rows, true)?;
                writer.ready_for_query(false)?;
                if let Some(decision) = decision {
                    let h2d_delta = after.h2d_bytes_total.saturating_sub(before.h2d_bytes_total);
                    let zero_h2d = decision.last_execution_h2d_bytes == Some(0)
                        && h2d_delta == 0
                        && decision.h2d_bytes_if_resident == 0;
                    self.fact("client_visible_select_sql", sql)?;
                    self.fact(
                        "client_visible_select_retained_route_accepted",
                        decision.accepted,
                    )?;
                    self.fact(
                        "client_visible_select_retained_route_shape",
                        &decision.query_shape,
                    )?;
                    self.fact("client_visible_select_retained_route_zero_h2d", zero_h2d)?;
                    self.fact("client_visible_select_retained_route_h2d_delta", h2d_delta)?;
                    self.fact(
                        "client_visible_select_retained_route_d2h_delta",
                        after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total),
                    )?;
                    self.fact(
                        "client_visible_select_retained_route_kernel_delta",
                        after
                            .kernel_exec_samples
                            .saturating_sub(before.kernel_exec_samples),
                    )?;
                    self.fact(
                        "client_visible_select_retained_match_index_compaction",
                        matches!(
                            decision.query_shape.as_str(),
                            "int4_equality_projection"
                                | "int4_equality_multi_column_projection"
                                | "int4_composite_equality_multi_column_projection"
                                | "int4_equality_mixed_column_projection"
                        ),
                    )?;
                    self.fact("client_visible_select_rows", rows.len())?;
                }
            }
            other => {
                return Err(format!("unsupported benchmark simple query command: {other:?}").into())
            }
        }
        Ok(())
    }

    fn start_copy(
        &mut self,
        sql: &str,
        output: &mut dyn Write,
    ) -> Result<PendingCopy, Box<dyn Error>> {
        let copy = parse_copy_from_stdin(sql).ok_or("expected COPY FROM STDIN")?;
        let columns = self.engine.relational_copy_columns(&copy.table)?;
        BackendWriter::new(output).copy_in_response(columns.len())?;
        self.fact("copy_parser_in_protocol_lib", true)?;
        self.fact("backend_copy_in_response_written", true)?;
        Ok(PendingCopy {
            copy,
            columns,
            rows: Vec::new(),
            pending_text: String::new(),
        })
    }

    fn finish_copy(
        &mut self,
        mut pending: PendingCopy,
        output: &mut dyn Write,
    ) -> Result<(), Box<dyn Error>> {
        pending.finish_pending_text()?;
        let txn_id = self.take_txn_id();
        let copied =
            self.engine
                .execute_relational_copy_rows(txn_id, &pending.copy, pending.rows)?;
        let warmup =
            self.engine
                .warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
                    tables: vec![pending.copy.table.clone()],
                    refresh_invalidated: true,
                    ..RelationalResidencyWarmupPolicy::default()
                });
        let snapshot = self
            .engine
            .relational_residency_snapshot(&pending.copy.table)
            .ok_or("resident warmup did not install a snapshot")?;
        let mut writer = BackendWriter::new(output);
        writer.command_complete(&format!("COPY {copied}"))?;
        writer.ready_for_query(false)?;
        self.fact("copy_rows_committed_to_engine_wal_mvcc", true)?;
        self.fact("copy_rows_decoded_by_protocol", copied)?;
        self.fact("resident_admission_from_sql_visible_rows", true)?;
        self.fact("sql_visible_resident_warmup_entries", warmup.entries.len())?;
        self.fact("sql_visible_resident_row_count", snapshot.row_count)?;
        self.fact(
            "sql_visible_resident_device_memory_retained",
            snapshot.device_memory_proof.is_some(),
        )?;
        Ok(())
    }
}

enum EngineCommand {
    Startup(Vec<u8>),
    SimpleQuery(String),
    StartCopy(String),
    FinishCopy(PendingCopy),
}

enum EngineResponse {
    Bytes(Vec<u8>),
    CopyStarted {
        bytes: Vec<u8>,
        pending: PendingCopy,
    },
}

struct EngineRequest {
    command: EngineCommand,
    response_tx: mpsc::Sender<Result<EngineResponse, String>>,
}

fn request_engine(
    request_tx: &mpsc::Sender<EngineRequest>,
    command: EngineCommand,
) -> Result<EngineResponse, String> {
    let (response_tx, response_rx) = mpsc::channel();
    request_tx
        .send(EngineRequest {
            command,
            response_tx,
        })
        .map_err(|err| format!("engine scheduler request failed: {err}"))?;
    response_rx
        .recv()
        .map_err(|err| format!("engine scheduler response failed: {err}"))?
}

fn write_engine_response(stream: &mut TcpStream, response: EngineResponse) -> Result<(), String> {
    match response {
        EngineResponse::Bytes(bytes) => stream.write_all(&bytes).map_err(|err| err.to_string()),
        EngineResponse::CopyStarted { bytes, .. } => {
            stream.write_all(&bytes).map_err(|err| err.to_string())
        }
    }
}

fn sql_value_text(value: &SqlValue) -> String {
    match value {
        SqlValue::Int4(value) => value.to_string(),
        SqlValue::Int8(value) => value.to_string(),
        SqlValue::Numeric(value) | SqlValue::Text(value) => value.clone(),
    }
}

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

fn startup_code(frame: &[u8]) -> Option<u32> {
    if frame.len() < 8 {
        return None;
    }
    Some(u32::from_be_bytes(frame[4..8].try_into().ok()?))
}

fn handle_client_io(
    mut stream: TcpStream,
    request_tx: mpsc::Sender<EngineRequest>,
) -> Result<bool, String> {
    let mut startup = match read_startup_frame(&mut stream).map_err(|err| err.to_string())? {
        Some(frame) => frame,
        None => return Ok(false),
    };
    while startup_code(&startup) == Some(SSL_REQUEST_CODE) {
        stream.write_all(b"N").map_err(|err| err.to_string())?;
        startup = match read_startup_frame(&mut stream).map_err(|err| err.to_string())? {
            Some(frame) => frame,
            None => return Ok(false),
        };
    }
    write_engine_response(
        &mut stream,
        request_engine(&request_tx, EngineCommand::Startup(startup))?,
    )?;

    let mut pending_copy: Option<PendingCopy> = None;
    while let Some(frame) = read_tagged_frame(&mut stream).map_err(|err| err.to_string())? {
        match parse_frontend_message(&frame).map_err(|err| err.to_string())? {
            FrontendMessage::SimpleQuery(sql) if parse_copy_from_stdin(&sql).is_some() => {
                match request_engine(&request_tx, EngineCommand::StartCopy(sql))? {
                    EngineResponse::CopyStarted { bytes, pending } => {
                        stream.write_all(&bytes).map_err(|err| err.to_string())?;
                        pending_copy = Some(pending);
                    }
                    EngineResponse::Bytes(_) => {
                        return Err("engine scheduler returned bytes for COPY start".to_string())
                    }
                }
            }
            FrontendMessage::SimpleQuery(sql) => {
                write_engine_response(
                    &mut stream,
                    request_engine(&request_tx, EngineCommand::SimpleQuery(sql))?,
                )?;
            }
            FrontendMessage::CopyData(bytes) => {
                let pending = pending_copy
                    .as_mut()
                    .ok_or("COPY data arrived without pending COPY stream")?;
                pending.push_bytes(&bytes).map_err(|err| err.to_string())?;
            }
            FrontendMessage::CopyDone => {
                let pending = pending_copy
                    .take()
                    .ok_or("COPY done arrived without pending COPY stream")?;
                write_engine_response(
                    &mut stream,
                    request_engine(&request_tx, EngineCommand::FinishCopy(pending))?,
                )?;
            }
            FrontendMessage::Terminate => break,
            other => {
                return Err(format!("unsupported benchmark frontend message: {other:?}").into())
            }
        }
    }
    Ok(true)
}

fn main() -> Result<(), Box<dyn Error>> {
    let listen = std::env::var("GPU_DB_P8_ENGINE_PGWIRE_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:55437".to_string());
    let facts_path = std::env::var("GPU_DB_P8_ENGINE_PGWIRE_FACTS")
        .unwrap_or_else(|_| "target/p8-engine-pgwire-endpoint/facts.txt".to_string());
    let max_sessions = std::env::var("GPU_DB_P8_ENGINE_PGWIRE_MAX_SESSIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);

    let listener = TcpListener::bind(&listen)?;
    let (request_tx, request_rx) = mpsc::channel::<EngineRequest>();
    let (completed_tx, completed_rx) = mpsc::channel::<()>();
    let accept_request_tx = request_tx.clone();
    let accept_completed_tx = completed_tx.clone();
    let accept_handle = thread::spawn(move || -> Result<(), String> {
        for stream in listener.incoming().take(max_sessions) {
            let stream = stream.map_err(|err| err.to_string())?;
            let client_request_tx = accept_request_tx.clone();
            let client_completed_tx = accept_completed_tx.clone();
            thread::spawn(move || {
                let completed = handle_client_io(stream, client_request_tx).unwrap_or(false);
                if completed {
                    let _ = client_completed_tx.send(());
                }
            });
        }
        Ok(())
    });

    let mut state = EndpointState::new(&facts_path)?;
    state.fact("engine_backed_pgwire_tcp_endpoint", true)?;
    state.fact("listen", &listen)?;
    state.fact("protocol_parser_reused", true)?;
    state.fact("backend_writer_api_available", true)?;
    state.fact("owner_thread_engine_scheduler", true)?;
    state.fact("client_io_workers_engine_owned_state", false)?;
    state.fact(
        "crate_direction",
        "gpu_db_engine_depends_on_gpu_db_protocol",
    )?;

    let mut completed = 0;
    while completed < max_sessions {
        while completed_rx.try_recv().is_ok() {
            completed += 1;
        }
        match request_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(request) => {
                let result = (|| -> Result<EngineResponse, Box<dyn Error>> {
                    let mut output = Vec::new();
                    match request.command {
                        EngineCommand::Startup(frame) => {
                            state.handle_startup(&frame, &mut output)?;
                            Ok(EngineResponse::Bytes(output))
                        }
                        EngineCommand::SimpleQuery(sql) => {
                            state.handle_simple_query(&sql, &mut output)?;
                            Ok(EngineResponse::Bytes(output))
                        }
                        EngineCommand::StartCopy(sql) => {
                            let pending = state.start_copy(&sql, &mut output)?;
                            Ok(EngineResponse::CopyStarted {
                                bytes: output,
                                pending,
                            })
                        }
                        EngineCommand::FinishCopy(pending) => {
                            state.finish_copy(pending, &mut output)?;
                            Ok(EngineResponse::Bytes(output))
                        }
                    }
                })()
                .map_err(|err| err.to_string());
                let _ = request.response_tx.send(result);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = accept_handle.join();
    state.fact("completed_client_sessions", completed)?;
    Ok(())
}
