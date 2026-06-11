use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use gpu_db_engine::{Engine, RelationalResidencyWarmupPolicy};
use gpu_db_protocol::backend::{BackendColumn, BackendWriter};
use gpu_db_protocol::{
    parse_command, parse_copy_from_stdin, parse_copy_row, parse_frontend_message,
    parse_startup_packet, Command, CopyFromStdin, FrontendMessage, Select, SelectFilterOp,
    SelectProjection, SqlValue, StartupPacket,
};

const SSL_REQUEST_CODE: u32 = 80877103;

struct RetainedReadResponseCache {
    enabled: bool,
    generation: u64,
    entries: HashMap<String, (u64, Vec<u8>)>,
    hits: u64,
    misses: u64,
    invalidations: u64,
}

impl RetainedReadResponseCache {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            generation: 0,
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
            invalidations: 0,
        }
    }

    fn get(&mut self, sql: &str) -> Option<Vec<u8>> {
        if !self.enabled {
            return None;
        }
        match self.entries.get(sql) {
            Some((generation, bytes)) if *generation == self.generation => {
                self.hits = self.hits.saturating_add(1);
                Some(bytes.clone())
            }
            _ => {
                self.misses = self.misses.saturating_add(1);
                None
            }
        }
    }

    fn insert(&mut self, sql: String, bytes: Vec<u8>) {
        if self.enabled {
            self.entries.insert(sql, (self.generation, bytes));
        }
    }

    fn invalidate(&mut self) {
        if self.enabled {
            self.generation = self.generation.saturating_add(1);
            self.invalidations = self.invalidations.saturating_add(1);
            self.entries.clear();
        }
    }
}

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

#[derive(Clone, Copy, Eq, PartialEq)]
enum SelectFactDetail {
    Full,
    PhaseOnly,
}

impl SelectFactDetail {
    fn from_env() -> Self {
        match std::env::var("GPU_DB_P8_ENGINE_PGWIRE_SELECT_FACT_DETAIL") {
            Ok(value) if matches!(value.as_str(), "full" | "FULL" | "1" | "true" | "TRUE") => {
                Self::Full
            }
            _ => Self::PhaseOnly,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::PhaseOnly => "phase_only",
        }
    }
}

struct EndpointState {
    engine: Engine,
    next_txn_id: u64,
    facts: BufWriter<File>,
    select_fact_detail: SelectFactDetail,
}

fn retained_match_index_compaction(query_shape: &str) -> bool {
    matches!(
        query_shape,
        "int4_equality_projection"
            | "int4_equality_multi_column_projection"
            | "int4_composite_equality_multi_column_projection"
            | "int4_equality_mixed_column_projection"
    )
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
            facts: BufWriter::new(facts),
            select_fact_detail: SelectFactDetail::from_env(),
        })
    }

    fn take_txn_id(&mut self) -> u64 {
        let txn_id = self.next_txn_id;
        self.next_txn_id += 1;
        txn_id
    }

    fn fact(&mut self, key: &str, value: impl std::fmt::Display) -> io::Result<()> {
        writeln!(self.facts, "{key}={value}")
    }

    fn flush_facts(&mut self) -> io::Result<()> {
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
        scheduler_queue_wait_micros: u64,
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
                let mut writer = BackendWriter::new(output);
                let write_started = Instant::now();
                let result_materialize_micros =
                    write_select_result_rows(&mut writer, &columns, &result.rows)?;
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
                    if self.select_fact_detail == SelectFactDetail::Full {
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
                        self.fact(
                            "client_visible_select_scheduler_queue_wait_micros",
                            scheduler_queue_wait_micros,
                        )?;
                        if let Some(value) = decision.last_execution_wall_micros {
                            self.fact("client_visible_select_retained_wall_micros", value)?;
                        }
                        if let Some(value) = decision.last_execution_device_lookup_micros {
                            self.fact(
                                "client_visible_select_retained_device_lookup_micros",
                                value,
                            )?;
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
                            retained_match_index_compaction(&decision.query_shape),
                        )?;
                        self.fact("client_visible_select_rows", result.rows.len())?;
                    }
                    self.fact(
                        "select_phase_json",
                        SelectPhaseFact {
                            sql,
                            query_shape: &decision.query_shape,
                            scheduler_queue_wait_micros,
                            engine_execute_micros,
                            result_materialize_micros,
                            client_write_micros,
                            retained_wall_micros: decision.last_execution_wall_micros,
                            retained_device_lookup_micros: decision
                                .last_execution_device_lookup_micros,
                            retained_match_index_micros: decision.last_execution_match_index_micros,
                            retained_selected_projection_micros: decision
                                .last_execution_selected_projection_micros,
                            retained_result_materialization_micros: decision
                                .last_execution_result_materialization_micros,
                            retained_cuda_event_micros: decision
                                .last_execution_kernel_event_elapsed_us,
                            retained_matched_rows: decision
                                .last_execution_matched_rows
                                .map(|value| value.try_into().unwrap_or(u64::MAX)),
                            h2d_delta,
                            d2h_delta: after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total),
                            kernel_delta: after
                                .kernel_exec_samples
                                .saturating_sub(before.kernel_exec_samples),
                            result_rows: result.rows.len(),
                            microbatch_kind: "none",
                            microbatch_size: 1,
                            microbatch_unique_selects: 1,
                        },
                    )?;
                }
            }
            other => {
                return Err(format!("unsupported benchmark simple query command: {other:?}").into())
            }
        }
        Ok(())
    }

    fn handle_multi_literal_select_batch(
        &mut self,
        items: &[(String, Select)],
        scheduler_queue_wait_micros: u64,
    ) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let mut unique_items = Vec::new();
        let mut unique_by_needle = HashMap::new();
        let mut item_to_unique = Vec::with_capacity(items.len());
        for (sql, select) in items {
            let needle = retained_select_literal_needle(select)
                .ok_or("literal SELECT batch requires one int4 equality needle")?;
            let unique_idx = if let Some(idx) = unique_by_needle.get(&needle) {
                *idx
            } else {
                let idx = unique_items.len();
                unique_by_needle.insert(needle, idx);
                unique_items.push((sql.clone(), select.clone()));
                idx
            };
            item_to_unique.push(unique_idx);
        }
        let selects = unique_items
            .iter()
            .map(|(_sql, select)| select.clone())
            .collect::<Vec<_>>();
        let before = self.engine.metrics().snapshot();
        let execute_started = Instant::now();
        let results = self
            .engine
            .execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
                &selects,
            )?;
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
            .latest_route_decision(&selects[0].table)
            .cloned();
        let batch_len = u64::try_from(items.len()).unwrap_or(u64::MAX).max(1);
        let h2d_delta = after.h2d_bytes_total.saturating_sub(before.h2d_bytes_total) / batch_len;
        let d2h_delta = after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total) / batch_len;
        let kernel_delta = after
            .kernel_exec_samples
            .saturating_sub(before.kernel_exec_samples)
            / batch_len;
        let mut outputs = Vec::with_capacity(items.len());
        for ((sql, _select), unique_idx) in items.iter().zip(item_to_unique) {
            let result = &results[unique_idx];
            let columns = result
                .columns
                .iter()
                .map(|column| BackendColumn::new(&column.name, column.type_oid, column.type_size))
                .collect::<Vec<_>>();
            let mut output = Vec::new();
            let mut writer = BackendWriter::new(&mut output);
            let write_started = Instant::now();
            let result_materialize_micros =
                write_select_result_rows(&mut writer, &columns, &result.rows)?;
            writer.ready_for_query(false)?;
            let client_write_micros = write_started
                .elapsed()
                .as_micros()
                .try_into()
                .unwrap_or(u64::MAX);
            if let Some(decision) = &decision {
                self.fact(
                    "select_phase_json",
                    SelectPhaseFact {
                        sql,
                        query_shape: &decision.query_shape,
                        scheduler_queue_wait_micros,
                        engine_execute_micros,
                        result_materialize_micros,
                        client_write_micros,
                        retained_wall_micros: decision.last_execution_wall_micros,
                        retained_device_lookup_micros: decision.last_execution_device_lookup_micros,
                        retained_match_index_micros: decision.last_execution_match_index_micros,
                        retained_selected_projection_micros: decision
                            .last_execution_selected_projection_micros,
                        retained_result_materialization_micros: decision
                            .last_execution_result_materialization_micros,
                        retained_cuda_event_micros: decision.last_execution_kernel_event_elapsed_us,
                        retained_matched_rows: decision
                            .last_execution_matched_rows
                            .map(|value| value.try_into().unwrap_or(u64::MAX)),
                        h2d_delta,
                        d2h_delta,
                        kernel_delta,
                        result_rows: result.rows.len(),
                        microbatch_kind: "multi_literal_gpu",
                        microbatch_size: u64::try_from(items.len()).unwrap_or(u64::MAX),
                        microbatch_unique_selects: u64::try_from(selects.len()).unwrap_or(u64::MAX),
                    },
                )?;
            }
            outputs.push(output);
        }
        Ok(outputs)
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
        let rows_per_sec = (copied as u128 * 1000)
            .checked_div(elapsed_ms)
            .unwrap_or(copied as u128);
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
    enqueued_at: Instant,
    response_tx: mpsc::Sender<Result<EngineResponse, String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RetainedSelectLiteralBatchKey {
    table: String,
    projection_columns: Vec<String>,
    filter_column: String,
}

fn retained_select_microbatch_key(command: &EngineCommand) -> Option<&str> {
    let EngineCommand::SimpleQuery(sql) = command else {
        return None;
    };
    match parse_command(sql) {
        Ok(Command::Select(_)) => Some(sql),
        _ => None,
    }
}

fn retained_select_literal_needle(select: &Select) -> Option<i32> {
    if select.filter_groups.len() > 1 {
        return None;
    }
    let filters = if !select.filter_groups.is_empty() {
        select.filter_groups[0].clone()
    } else if !select.filters.is_empty() {
        select.filters.clone()
    } else if let Some(filter) = select.filter.clone() {
        vec![filter]
    } else {
        return None;
    };
    if filters.len() != 1 || filters[0].op != SelectFilterOp::Eq {
        return None;
    }
    let SqlValue::Int4(needle) = filters[0].value else {
        return None;
    };
    Some(needle)
}

fn retained_select_literal_batch_candidate(
    command: &EngineCommand,
) -> Option<(RetainedSelectLiteralBatchKey, i32, Select, String)> {
    let EngineCommand::SimpleQuery(sql) = command else {
        return None;
    };
    let Ok(Command::Select(select)) = parse_command(sql) else {
        return None;
    };
    if select.distinct
        || select.group_by.is_some()
        || !select.having_groups.is_empty()
        || select.order_by.is_some()
        || select.limit.is_some()
        || select.offset.is_some()
        || select.filter_groups.len() > 1
    {
        return None;
    }
    let SelectProjection::Columns(projection_columns) = &select.projection else {
        return None;
    };
    if projection_columns.is_empty() {
        return None;
    }
    let filters = if !select.filter_groups.is_empty() {
        select.filter_groups[0].clone()
    } else if !select.filters.is_empty() {
        select.filters.clone()
    } else if let Some(filter) = select.filter.clone() {
        vec![filter]
    } else {
        return None;
    };
    if filters.len() != 1 || filters[0].op != SelectFilterOp::Eq {
        return None;
    }
    let needle = retained_select_literal_needle(&select)?;
    Some((
        RetainedSelectLiteralBatchKey {
            table: select.table.clone(),
            projection_columns: projection_columns.clone(),
            filter_column: filters[0].column.clone(),
        },
        needle,
        select,
        sql.clone(),
    ))
}

fn request_engine(
    request_tx: &mpsc::Sender<EngineRequest>,
    command: EngineCommand,
) -> Result<EngineResponse, String> {
    let (response_tx, response_rx) = mpsc::channel();
    request_tx
        .send(EngineRequest {
            command,
            enqueued_at: Instant::now(),
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

fn write_select_result_rows<W: Write + ?Sized>(
    writer: &mut BackendWriter<'_, W>,
    columns: &[BackendColumn],
    rows: &[Vec<SqlValue>],
) -> Result<u64, Box<dyn Error>> {
    writer.row_description(columns)?;
    let materialize_started = Instant::now();
    for row in rows {
        let values = row
            .iter()
            .map(|value| Some(sql_value_text(value)))
            .collect::<Vec<_>>();
        writer.data_row(&values)?;
    }
    let result_materialize_micros = materialize_started
        .elapsed()
        .as_micros()
        .try_into()
        .unwrap_or(u64::MAX);
    writer.command_complete(&format!("SELECT {}", rows.len()))?;
    Ok(result_materialize_micros)
}

struct SelectPhaseFact<'a> {
    sql: &'a str,
    query_shape: &'a str,
    scheduler_queue_wait_micros: u64,
    engine_execute_micros: u64,
    result_materialize_micros: u64,
    client_write_micros: u64,
    retained_wall_micros: Option<u64>,
    retained_device_lookup_micros: Option<u64>,
    retained_match_index_micros: Option<u64>,
    retained_selected_projection_micros: Option<u64>,
    retained_result_materialization_micros: Option<u64>,
    retained_cuda_event_micros: Option<u64>,
    retained_matched_rows: Option<u64>,
    h2d_delta: u64,
    d2h_delta: u64,
    kernel_delta: u64,
    result_rows: usize,
    microbatch_kind: &'a str,
    microbatch_size: u64,
    microbatch_unique_selects: u64,
}

impl std::fmt::Display for SelectPhaseFact<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{{\"sql\":\"{}\",\"query_shape\":\"{}\",\"scheduler_queue_wait_micros\":{},\"engine_execute_micros\":{},\"result_materialize_micros\":{},\"client_write_micros\":{},\"retained_wall_micros\":{},\"retained_device_lookup_micros\":{},\"retained_match_index_micros\":{},\"retained_selected_projection_micros\":{},\"retained_result_materialization_micros\":{},\"retained_cuda_event_micros\":{},\"retained_matched_rows\":{},\"h2d_delta\":{},\"d2h_delta\":{},\"kernel_delta\":{},\"result_rows\":{},\"microbatch_kind\":\"{}\",\"microbatch_size\":{},\"microbatch_unique_selects\":{}}}",
            json_escape(self.sql),
            json_escape(self.query_shape),
            self.scheduler_queue_wait_micros,
            self.engine_execute_micros,
            self.result_materialize_micros,
            self.client_write_micros,
            json_optional_u64(self.retained_wall_micros),
            json_optional_u64(self.retained_device_lookup_micros),
            json_optional_u64(self.retained_match_index_micros),
            json_optional_u64(self.retained_selected_projection_micros),
            json_optional_u64(self.retained_result_materialization_micros),
            json_optional_u64(self.retained_cuda_event_micros),
            json_optional_u64(self.retained_matched_rows),
            self.h2d_delta,
            self.d2h_delta,
            self.kernel_delta,
            self.result_rows,
            json_escape(self.microbatch_kind),
            self.microbatch_size,
            self.microbatch_unique_selects
        )
    }
}

fn json_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_escape(value: &str) -> String {
    value
        .chars()
        .flat_map(|ch| match ch {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect::<Vec<_>>(),
            '\t' => "\\t".chars().collect::<Vec<_>>(),
            _ => vec![ch],
        })
        .collect()
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
    retained_read_response_cache: Arc<Mutex<RetainedReadResponseCache>>,
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
                retained_read_response_cache
                    .lock()
                    .map_err(|_| "retained read response cache lock poisoned".to_string())?
                    .invalidate();
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
                match parse_command(&sql).map_err(|err| err.to_string())? {
                    Command::Select(_) => {
                        let cached = retained_read_response_cache
                            .lock()
                            .map_err(|_| "retained read response cache lock poisoned".to_string())?
                            .get(&sql);
                        if let Some(bytes) = cached {
                            stream.write_all(&bytes).map_err(|err| err.to_string())?;
                            continue;
                        }
                        let response =
                            request_engine(&request_tx, EngineCommand::SimpleQuery(sql.clone()))?;
                        if let EngineResponse::Bytes(bytes) = &response {
                            retained_read_response_cache
                                .lock()
                                .map_err(|_| {
                                    "retained read response cache lock poisoned".to_string()
                                })?
                                .insert(sql, bytes.clone());
                        }
                        write_engine_response(&mut stream, response)?;
                    }
                    _ => {
                        retained_read_response_cache
                            .lock()
                            .map_err(|_| "retained read response cache lock poisoned".to_string())?
                            .invalidate();
                        write_engine_response(
                            &mut stream,
                            request_engine(&request_tx, EngineCommand::SimpleQuery(sql))?,
                        )?;
                    }
                }
            }
            FrontendMessage::CopyData(bytes) => {
                let pending = pending_copy
                    .as_mut()
                    .ok_or("COPY data arrived without pending COPY stream")?;
                let chunks = pending.push_bytes(&bytes).map_err(|err| err.to_string())?;
                commit_pending_copy_chunks(pending, &request_tx, chunks)?;
            }
            FrontendMessage::CopyDone => {
                retained_read_response_cache
                    .lock()
                    .map_err(|_| "retained read response cache lock poisoned".to_string())?
                    .invalidate();
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
            other => return Err(format!("unsupported benchmark frontend message: {other:?}")),
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
    let retained_read_response_cache_enabled =
        std::env::var("GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RESPONSE_CACHE")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
    let gpu_microbatch_max = std::env::var("GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_MAX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(64);

    let listener = TcpListener::bind(&listen)?;
    let (request_tx, request_rx) = mpsc::channel::<EngineRequest>();
    let (completed_tx, completed_rx) = mpsc::channel::<()>();
    let retained_read_response_cache = Arc::new(Mutex::new(RetainedReadResponseCache::new(
        retained_read_response_cache_enabled,
    )));
    let accept_request_tx = request_tx.clone();
    let accept_completed_tx = completed_tx.clone();
    let accept_retained_read_response_cache = Arc::clone(&retained_read_response_cache);
    let accept_handle = thread::spawn(move || -> Result<(), String> {
        for stream in listener.incoming().take(max_sessions) {
            let stream = stream.map_err(|err| err.to_string())?;
            let client_request_tx = accept_request_tx.clone();
            let client_completed_tx = accept_completed_tx.clone();
            let client_retained_read_response_cache =
                Arc::clone(&accept_retained_read_response_cache);
            thread::spawn(move || {
                let completed = handle_client_io(
                    stream,
                    client_request_tx,
                    client_retained_read_response_cache,
                )
                .unwrap_or(false);
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
    state.fact(
        "owner_thread_fact_writer",
        "bufwriter_request_boundary_flush",
    )?;
    state.fact("select_fact_detail", state.select_fact_detail.as_str())?;
    state.fact("client_io_workers_engine_owned_state", false)?;
    state.fact(
        "retained_read_response_cache_enabled",
        retained_read_response_cache_enabled,
    )?;
    state.fact("owner_thread_gpu_microbatch_max", gpu_microbatch_max)?;
    state.fact("owner_thread_gpu_microbatch_exact_select", true)?;
    state.fact("owner_thread_gpu_microbatch_multi_literal_select", true)?;
    state.fact("max_sessions", max_sessions)?;
    state.fact(
        "crate_direction",
        "gpu_db_engine_depends_on_gpu_db_protocol",
    )?;

    let mut completed = 0;
    let mut deferred_requests = VecDeque::new();
    let mut gpu_microbatch_batches = 0_u64;
    let mut gpu_microbatch_coalesced_requests = 0_u64;
    let mut gpu_literal_microbatch_batches = 0_u64;
    let mut gpu_literal_microbatch_coalesced_requests = 0_u64;
    while completed < max_sessions {
        while completed_rx.try_recv().is_ok() {
            completed += 1;
        }
        let next_request = if let Some(request) = deferred_requests.pop_front() {
            Ok(request)
        } else {
            request_rx.recv_timeout(Duration::from_millis(50))
        };
        match next_request {
            Ok(request) => {
                let mut batch = vec![request];
                let mut literal_microbatch = false;
                if gpu_microbatch_max > 1 {
                    if let Some((batch_key, _first_needle, _select, _sql)) =
                        retained_select_literal_batch_candidate(&batch[0].command)
                    {
                        while batch.len() < gpu_microbatch_max {
                            match request_rx.try_recv() {
                                Ok(next) => {
                                    if let Some((next_key, _next_needle, _select, _sql)) =
                                        retained_select_literal_batch_candidate(&next.command)
                                    {
                                        if next_key == batch_key {
                                            batch.push(next);
                                        } else {
                                            deferred_requests.push_back(next);
                                            break;
                                        }
                                    } else {
                                        deferred_requests.push_back(next);
                                        break;
                                    }
                                }
                                Err(mpsc::TryRecvError::Empty) => break,
                                Err(mpsc::TryRecvError::Disconnected) => break,
                            }
                        }
                        let exact_key =
                            retained_select_microbatch_key(&batch[0].command).unwrap_or_default();
                        literal_microbatch = batch.len() > 1
                            && !batch.iter().all(|request| {
                                retained_select_microbatch_key(&request.command) == Some(exact_key)
                            });
                    } else if let Some(batch_key) =
                        retained_select_microbatch_key(&batch[0].command).map(str::to_string)
                    {
                        while batch.len() < gpu_microbatch_max {
                            match request_rx.try_recv() {
                                Ok(next) => {
                                    if retained_select_microbatch_key(&next.command)
                                        == Some(batch_key.as_str())
                                    {
                                        batch.push(next);
                                    } else {
                                        deferred_requests.push_back(next);
                                        break;
                                    }
                                }
                                Err(mpsc::TryRecvError::Empty) => break,
                                Err(mpsc::TryRecvError::Disconnected) => break,
                            }
                        }
                    }
                }
                if batch.len() > 1 {
                    if literal_microbatch {
                        gpu_literal_microbatch_batches =
                            gpu_literal_microbatch_batches.saturating_add(1);
                        gpu_literal_microbatch_coalesced_requests =
                            gpu_literal_microbatch_coalesced_requests
                                .saturating_add(u64::try_from(batch.len() - 1).unwrap_or(u64::MAX));
                    } else {
                        gpu_microbatch_batches = gpu_microbatch_batches.saturating_add(1);
                        gpu_microbatch_coalesced_requests = gpu_microbatch_coalesced_requests
                            .saturating_add(u64::try_from(batch.len() - 1).unwrap_or(u64::MAX));
                    }
                }
                let scheduler_queue_wait_micros = batch[0]
                    .enqueued_at
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX);
                if batch.len() > 1 {
                    if literal_microbatch {
                        let result = (|| -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
                            let items = batch
                                .iter()
                                .map(|request| {
                                    retained_select_literal_batch_candidate(&request.command)
                                        .map(|(_key, _needle, select, sql)| (sql, select))
                                        .ok_or("only compatible int4 equality SELECT can be literal-microbatched")
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let outputs = state.handle_multi_literal_select_batch(
                                &items,
                                scheduler_queue_wait_micros,
                            )?;
                            state.flush_facts()?;
                            Ok(outputs)
                        })()
                        .map_err(|err| err.to_string());
                        match result {
                            Ok(outputs) => {
                                for (request, output) in batch.into_iter().zip(outputs) {
                                    let _ =
                                        request.response_tx.send(Ok(EngineResponse::Bytes(output)));
                                }
                            }
                            Err(err) => {
                                for request in batch {
                                    let _ = request.response_tx.send(Err(err.clone()));
                                }
                            }
                        }
                    } else {
                        let result = (|| -> Result<Vec<u8>, Box<dyn Error>> {
                            let EngineCommand::SimpleQuery(sql) = &batch[0].command else {
                                return Err("only simple SELECT can be microbatched".into());
                            };
                            let mut output = Vec::new();
                            state.handle_simple_query(
                                sql,
                                &mut output,
                                scheduler_queue_wait_micros,
                            )?;
                            state.flush_facts()?;
                            Ok(output)
                        })()
                        .map_err(|err| err.to_string());
                        for request in batch {
                            let response = result
                                .as_ref()
                                .map(|bytes| EngineResponse::Bytes(bytes.clone()))
                                .map_err(|err| err.clone());
                            let _ = request.response_tx.send(response);
                        }
                    }
                } else {
                    let request = batch.pop().expect("single request batch is non-empty");
                    let result = (|| -> Result<EngineResponse, Box<dyn Error>> {
                        let mut output = Vec::new();
                        match request.command {
                            EngineCommand::Startup(frame) => {
                                state.handle_startup(&frame, &mut output)?;
                                Ok(EngineResponse::Bytes(output))
                            }
                            EngineCommand::SimpleQuery(sql) => {
                                state.handle_simple_query(
                                    &sql,
                                    &mut output,
                                    scheduler_queue_wait_micros,
                                )?;
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
                    .and_then(|response| {
                        state.flush_facts()?;
                        Ok(response)
                    })
                    .map_err(|err| err.to_string());
                    let _ = request.response_tx.send(result);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = accept_handle.join();
    state.fact("completed_client_sessions", completed)?;
    let cache = retained_read_response_cache
        .lock()
        .map_err(|_| "retained read response cache lock poisoned")?;
    state.fact(
        "owner_thread_gpu_microbatch_batches",
        gpu_microbatch_batches,
    )?;
    state.fact(
        "owner_thread_gpu_microbatch_coalesced_requests",
        gpu_microbatch_coalesced_requests,
    )?;
    state.fact(
        "owner_thread_gpu_literal_microbatch_batches",
        gpu_literal_microbatch_batches,
    )?;
    state.fact(
        "owner_thread_gpu_literal_microbatch_coalesced_requests",
        gpu_literal_microbatch_coalesced_requests,
    )?;
    state.fact("retained_read_response_cache_hits", cache.hits)?;
    state.fact("retained_read_response_cache_misses", cache.misses)?;
    state.fact(
        "retained_read_response_cache_invalidations",
        cache.invalidations,
    )?;
    state.flush_facts()?;
    Ok(())
}
