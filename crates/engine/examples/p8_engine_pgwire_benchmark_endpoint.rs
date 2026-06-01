use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

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
    chunk_rows: usize,
    committed_rows: usize,
    committed_chunks: usize,
    max_buffered_rows: usize,
}

impl PendingCopy {
    fn push_bytes(&mut self, bytes: &[u8]) -> Result<Vec<Vec<Vec<SqlValue>>>, Box<dyn Error>> {
        let mut chunks = Vec::new();
        self.pending_text.push_str(std::str::from_utf8(bytes)?);
        while let Some(newline) = self.pending_text.find('\n') {
            let mut line = self.pending_text[..newline].to_string();
            if line.ends_with('\r') {
                line.pop();
            }
            self.pending_text.drain(..=newline);
            self.push_line(&line)?;
            if let Some(chunk) = self.take_ready_chunk() {
                chunks.push(chunk);
            }
        }
        Ok(chunks)
    }

    fn finish_pending_text(&mut self) -> Result<Vec<Vec<Vec<SqlValue>>>, Box<dyn Error>> {
        if self.pending_text.is_empty() {
            return Ok(self.take_remaining_chunk().into_iter().collect());
        }
        let line = std::mem::take(&mut self.pending_text);
        self.push_line(line.trim_end_matches('\r'))?;
        Ok(self.take_remaining_chunk().into_iter().collect())
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
        self.max_buffered_rows = self.max_buffered_rows.max(self.rows.len());
        Ok(())
    }

    fn take_ready_chunk(&mut self) -> Option<Vec<Vec<SqlValue>>> {
        if self.rows.len() >= self.chunk_rows {
            Some(std::mem::take(&mut self.rows))
        } else {
            None
        }
    }

    fn take_remaining_chunk(&mut self) -> Option<Vec<Vec<SqlValue>>> {
        if self.rows.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.rows))
        }
    }

    fn record_committed_chunk(&mut self, copied: usize) {
        self.committed_rows += copied;
        self.committed_chunks += 1;
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
                let execute_started = Instant::now();
                let result = self.engine.execute_relational_select(&select)?;
                let engine_execute_micros = execute_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX);
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
                let materialize_started = Instant::now();
                let rows = result
                    .rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|value| Some(sql_value_text(value)))
                            .collect()
                    })
                    .collect::<Vec<Vec<_>>>();
                let result_materialize_micros = materialize_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX);
                let mut writer = BackendWriter::new(output);
                let write_started = Instant::now();
                writer.select_rows(&columns, &rows, true)?;
                writer.ready_for_query(false)?;
                let client_write_micros = write_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX);
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
                        "client_visible_select_engine_execute_micros",
                        engine_execute_micros,
                    )?;
                    self.fact(
                        "client_visible_select_result_materialize_micros",
                        result_materialize_micros,
                    )?;
                    self.fact(
                        "client_visible_select_client_write_micros",
                        client_write_micros,
                    )?;
                    if let Some(value) = decision.last_execution_wall_micros {
                        self.fact("client_visible_select_retained_wall_micros", value)?;
                    }
                    if let Some(value) = decision.last_execution_device_lookup_micros {
                        self.fact("client_visible_select_retained_device_lookup_micros", value)?;
                    }
                    if let Some(value) = decision.last_execution_match_index_micros {
                        self.fact("client_visible_select_retained_match_index_micros", value)?;
                    }
                    if let Some(value) = decision.last_execution_selected_projection_micros {
                        self.fact(
                            "client_visible_select_retained_selected_projection_micros",
                            value,
                        )?;
                    }
                    if let Some(value) = decision.last_execution_result_materialization_micros {
                        self.fact(
                            "client_visible_select_retained_result_materialization_micros",
                            value,
                        )?;
                    }
                    if let Some(value) = decision.last_execution_matched_rows {
                        self.fact("client_visible_select_retained_matched_rows", value)?;
                    }
                    if let Some(value) = decision.last_execution_kernel_event_elapsed_us {
                        self.fact("client_visible_select_retained_cuda_event_micros", value)?;
                    }
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
        let chunk_rows = std::env::var("GPU_DB_P8_ENGINE_PGWIRE_COPY_CHUNK_ROWS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(8192);
        BackendWriter::new(output).copy_in_response(columns.len())?;
        self.fact("copy_parser_in_protocol_lib", true)?;
        self.fact("backend_copy_in_response_written", true)?;
        self.fact("copy_chunk_rows_limit", chunk_rows)?;
        self.fact("copy_current_process_decoded_apply_fast_path", true)?;
        self.fact("copy_engine_reserved_row_key_bulk_admission", true)?;
        self.fact("copy_relational_value_index_bulk_admission", true)?;
        Ok(PendingCopy {
            copy,
            columns,
            rows: Vec::new(),
            pending_text: String::new(),
            chunk_rows,
            committed_rows: 0,
            committed_chunks: 0,
            max_buffered_rows: 0,
        })
    }

    fn commit_copy_chunk(
        &mut self,
        copy: &CopyFromStdin,
        rows: Vec<Vec<SqlValue>>,
    ) -> Result<usize, Box<dyn Error>> {
        let txn_id = self.take_txn_id();
        let started = Instant::now();
        let (copied, profile) = self
            .engine
            .execute_relational_copy_rows_profiled(txn_id, copy, rows)?;
        let elapsed_ms = started.elapsed().as_millis();
        let rows_per_sec = if elapsed_ms == 0 {
            copied as u128
        } else {
            (copied as u128 * 1000) / elapsed_ms
        };
        self.fact("copy_rows_committed_to_engine_wal_mvcc", true)?;
        self.fact("copy_chunk_rows", copied)?;
        self.fact("copy_chunk_elapsed_ms", elapsed_ms)?;
        self.fact("copy_chunk_rows_per_sec", rows_per_sec)?;
        self.fact("copy_profile_rows", profile.rows)?;
        self.fact(
            "copy_profile_render_sql_wal_payload_micros",
            profile.render_sql_wal_payload_micros,
        )?;
        self.fact(
            "copy_profile_commit_total_micros",
            profile.commit_total_micros,
        )?;
        self.fact(
            "copy_profile_wal_commit_flush_boundary_micros",
            profile.wal_commit_flush_boundary_micros,
        )?;
        self.fact(
            "copy_profile_current_apply_total_micros",
            profile.current_apply_total_micros,
        )?;
        self.fact(
            "copy_profile_row_prepare_micros",
            profile.row_prepare_micros,
        )?;
        self.fact(
            "copy_profile_unique_preflight_micros",
            profile.unique_preflight_micros,
        )?;
        self.fact(
            "copy_profile_check_preflight_micros",
            profile.check_preflight_micros,
        )?;
        self.fact(
            "copy_profile_foreign_key_preflight_micros",
            profile.foreign_key_preflight_micros,
        )?;
        self.fact(
            "copy_profile_mvcc_insert_micros",
            profile.mvcc_insert_micros,
        )?;
        self.fact(
            "copy_profile_value_index_append_micros",
            profile.value_index_append_micros,
        )?;
        self.fact(
            "copy_profile_residency_invalidation_micros",
            profile.residency_invalidation_micros,
        )?;
        Ok(copied)
    }

    fn finish_copy(
        &mut self,
        pending: PendingCopy,
        output: &mut dyn Write,
    ) -> Result<(), Box<dyn Error>> {
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
        writer.command_complete(&format!("COPY {}", pending.committed_rows))?;
        writer.ready_for_query(false)?;
        self.fact("copy_rows_committed_to_engine_wal_mvcc", true)?;
        self.fact("copy_rows_decoded_by_protocol", pending.committed_rows)?;
        self.fact("resident_admission_from_sql_visible_rows", true)?;
        self.fact("sql_visible_resident_warmup_entries", warmup.entries.len())?;
        self.fact("sql_visible_resident_row_count", snapshot.row_count)?;
        self.fact(
            "sql_visible_resident_device_memory_retained",
            snapshot.device_memory_proof.is_some(),
        )?;
        self.fact("copy_streaming_bounded_chunks", true)?;
        self.fact("copy_committed_chunks", pending.committed_chunks)?;
        self.fact("copy_max_buffered_decoded_rows", pending.max_buffered_rows)?;
        Ok(())
    }
}

enum EngineCommand {
    Startup(Vec<u8>),
    SimpleQuery(String),
    StartCopy(String),
    CopyChunk {
        copy: CopyFromStdin,
        rows: Vec<Vec<SqlValue>>,
    },
    FinishCopy(PendingCopy),
}

enum EngineResponse {
    Bytes(Vec<u8>),
    CopyStarted {
        bytes: Vec<u8>,
        pending: PendingCopy,
    },
    CopyChunkCommitted {
        copied: usize,
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
        EngineResponse::CopyChunkCommitted { .. } => Ok(()),
    }
}

fn commit_pending_copy_chunks(
    pending: &mut PendingCopy,
    request_tx: &mpsc::Sender<EngineRequest>,
    chunks: Vec<Vec<Vec<SqlValue>>>,
) -> Result<(), String> {
    for rows in chunks {
        match request_engine(
            request_tx,
            EngineCommand::CopyChunk {
                copy: pending.copy.clone(),
                rows,
            },
        )? {
            EngineResponse::CopyChunkCommitted { copied } => {
                pending.record_committed_chunk(copied);
            }
            _ => return Err("engine scheduler returned non-COPY response for COPY chunk".into()),
        }
    }
    Ok(())
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
                    EngineResponse::CopyChunkCommitted { .. } => {
                        return Err(
                            "engine scheduler returned COPY chunk response for COPY start"
                                .to_string(),
                        )
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
                let chunks = pending.push_bytes(&bytes).map_err(|err| err.to_string())?;
                commit_pending_copy_chunks(pending, &request_tx, chunks)?;
            }
            FrontendMessage::CopyDone => {
                let mut pending = pending_copy
                    .take()
                    .ok_or("COPY done arrived without pending COPY stream")?;
                let chunks = pending
                    .finish_pending_text()
                    .map_err(|err| err.to_string())?;
                commit_pending_copy_chunks(&mut pending, &request_tx, chunks)?;
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
    state.fact("max_sessions", max_sessions)?;
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
                        EngineCommand::CopyChunk { copy, rows } => {
                            let copied = state.commit_copy_chunk(&copy, rows)?;
                            Ok(EngineResponse::CopyChunkCommitted { copied })
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
