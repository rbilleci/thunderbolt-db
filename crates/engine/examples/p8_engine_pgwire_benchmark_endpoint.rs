use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

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
        stream: &mut TcpStream,
    ) -> Result<(), Box<dyn Error>> {
        match parse_startup_packet(frame)? {
            StartupPacket::Startup { .. } => {
                let mut writer = BackendWriter::new(stream);
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
        stream: &mut TcpStream,
    ) -> Result<(), Box<dyn Error>> {
        match parse_command(sql)? {
            Command::CreateTable(_) => {
                let txn_id = self.take_txn_id();
                self.engine.execute_text(txn_id, sql)?;
                let mut writer = BackendWriter::new(stream);
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
                let mut writer = BackendWriter::new(stream);
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
        stream: &mut TcpStream,
    ) -> Result<PendingCopy, Box<dyn Error>> {
        let copy = parse_copy_from_stdin(sql).ok_or("expected COPY FROM STDIN")?;
        let columns = self.engine.relational_copy_columns(&copy.table)?;
        BackendWriter::new(stream).copy_in_response(columns.len())?;
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
        stream: &mut TcpStream,
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
        let mut writer = BackendWriter::new(stream);
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

fn handle_client(
    mut stream: TcpStream,
    state: Arc<Mutex<EndpointState>>,
) -> Result<bool, Box<dyn Error>> {
    let mut startup = match read_startup_frame(&mut stream)? {
        Some(frame) => frame,
        None => return Ok(false),
    };
    while startup_code(&startup) == Some(SSL_REQUEST_CODE) {
        stream.write_all(b"N")?;
        startup = match read_startup_frame(&mut stream)? {
            Some(frame) => frame,
            None => return Ok(false),
        };
    }
    state
        .lock()
        .unwrap()
        .handle_startup(&startup, &mut stream)?;

    let mut pending_copy: Option<PendingCopy> = None;
    while let Some(frame) = read_tagged_frame(&mut stream)? {
        match parse_frontend_message(&frame)? {
            FrontendMessage::SimpleQuery(sql) if parse_copy_from_stdin(&sql).is_some() => {
                pending_copy = Some(state.lock().unwrap().start_copy(&sql, &mut stream)?);
            }
            FrontendMessage::SimpleQuery(sql) => {
                state
                    .lock()
                    .unwrap()
                    .handle_simple_query(&sql, &mut stream)?;
            }
            FrontendMessage::CopyData(bytes) => {
                let pending = pending_copy
                    .as_mut()
                    .ok_or("COPY data arrived without pending COPY stream")?;
                pending.push_bytes(&bytes)?;
            }
            FrontendMessage::CopyDone => {
                let pending = pending_copy
                    .take()
                    .ok_or("COPY done arrived without pending COPY stream")?;
                state.lock().unwrap().finish_copy(pending, &mut stream)?;
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
    let state = Arc::new(Mutex::new(EndpointState::new(&facts_path)?));
    {
        let mut state = state.lock().unwrap();
        state.fact("engine_backed_pgwire_tcp_endpoint", true)?;
        state.fact("listen", &listen)?;
        state.fact("protocol_parser_reused", true)?;
        state.fact("backend_writer_api_available", true)?;
        state.fact(
            "crate_direction",
            "gpu_db_engine_depends_on_gpu_db_protocol",
        )?;
    }

    let mut completed = 0;
    for stream in listener.incoming() {
        let stream = stream?;
        if handle_client(stream, Arc::clone(&state))? {
            completed += 1;
            if completed >= max_sessions {
                break;
            }
        }
    }
    state
        .lock()
        .unwrap()
        .fact("completed_client_sessions", completed)?;
    Ok(())
}
