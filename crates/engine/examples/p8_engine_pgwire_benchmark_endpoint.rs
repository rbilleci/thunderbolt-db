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
    retained_read_job_submission_batches: u64,
    retained_read_jobs_submitted: u64,
    retained_read_job_submit_wall_micros_total: u64,
    retained_read_job_complete_wall_micros_total: u64,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum RouteLaneScanPolicy {
    Fixed,
    Adaptive,
}

impl RouteLaneScanPolicy {
    fn from_env() -> Self {
        match std::env::var("GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_POLICY") {
            Ok(value) if matches!(value.as_str(), "fixed" | "FIXED") => Self::Fixed,
            _ => Self::Adaptive,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Adaptive => "adaptive",
        }
    }
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
            retained_read_job_submission_batches: 0,
            retained_read_jobs_submitted: 0,
            retained_read_job_submit_wall_micros_total: 0,
            retained_read_job_complete_wall_micros_total: 0,
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
                            retained_read_job_route_id: None,
                            retained_snapshot_generation: decision.snapshot_generation,
                            retained_read_submit_micros: None,
                            retained_read_complete_micros: None,
                            h2d_delta,
                            d2h_delta: after.d2h_bytes_total.saturating_sub(before.d2h_bytes_total),
                            kernel_delta: after
                                .kernel_exec_samples
                                .saturating_sub(before.kernel_exec_samples),
                            result_rows: result.rows.len(),
                            microbatch_kind: "none",
                            microbatch_size: 1,
                            microbatch_unique_selects: 1,
                            microbatch_admission_wait_micros: 0,
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
        microbatch_admission_wait_micros: u64,
        microbatch_kind: &'static str,
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
        let read_jobs = selects
            .iter()
            .map(|select| self.engine.prepare_relational_retained_read_job(select))
            .collect::<Result<Vec<_>, _>>()?;
        let before = self.engine.metrics().snapshot();
        let execute_started = Instant::now();
        let submission = self
            .engine
            .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&read_jobs)?;
        let retained_read_submit_micros = submission.submit_wall_micros;
        let complete_started = Instant::now();
        let results = Engine::complete_relational_retained_read_submission(submission);
        let retained_read_complete_micros = complete_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        self.retained_read_job_submission_batches =
            self.retained_read_job_submission_batches.saturating_add(1);
        self.retained_read_jobs_submitted = self
            .retained_read_jobs_submitted
            .saturating_add(u64::try_from(read_jobs.len()).unwrap_or(u64::MAX));
        self.retained_read_job_submit_wall_micros_total = self
            .retained_read_job_submit_wall_micros_total
            .saturating_add(retained_read_submit_micros);
        self.retained_read_job_complete_wall_micros_total = self
            .retained_read_job_complete_wall_micros_total
            .saturating_add(retained_read_complete_micros);
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
        let snapshot_handle = self
            .engine
            .relational_retained_snapshot_handle(&selects[0].table);
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
                        retained_read_job_route_id: Some(&read_jobs[unique_idx].route_id),
                        retained_snapshot_generation: snapshot_handle
                            .as_ref()
                            .map(|handle| handle.generation)
                            .or(decision.snapshot_generation),
                        retained_read_submit_micros: Some(retained_read_submit_micros),
                        retained_read_complete_micros: Some(retained_read_complete_micros),
                        h2d_delta,
                        d2h_delta,
                        kernel_delta,
                        result_rows: result.rows.len(),
                        microbatch_kind,
                        microbatch_size: u64::try_from(items.len()).unwrap_or(u64::MAX),
                        microbatch_unique_selects: u64::try_from(selects.len()).unwrap_or(u64::MAX),
                        microbatch_admission_wait_micros,
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
            "sql_visible_resident_snapshot_generation",
            snapshot.generation,
        )?;
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
    batch_candidate: Option<RetainedSelectBatchCandidate>,
    enqueued_at: Instant,
    response_tx: mpsc::Sender<Result<EngineResponse, String>>,
}

#[derive(Clone)]
struct EngineRequestSender {
    throughput_tx: mpsc::Sender<EngineRequest>,
    latency_tx: mpsc::Sender<EngineRequest>,
    latency_lane_enabled: bool,
}

#[derive(Debug, Clone)]
enum RetainedSelectBatchCandidate {
    Literal {
        batch_key: RetainedSelectLiteralBatchKey,
        exact_key: String,
        select: Select,
        sql: String,
    },
    Exact {
        exact_key: String,
    },
}

impl RetainedSelectBatchCandidate {
    fn exact_key(&self) -> &str {
        match self {
            Self::Literal { exact_key, .. } | Self::Exact { exact_key } => exact_key,
        }
    }

    fn route_key(&self) -> String {
        match self {
            Self::Literal { batch_key, .. } => format!(
                "literal:{}:{}:{}",
                batch_key.table,
                batch_key.projection_columns.join(","),
                batch_key.filter_column
            ),
            Self::Exact { exact_key } => format!("exact:{exact_key}"),
        }
    }

    fn projected_payload_weight(&self) -> usize {
        match self {
            Self::Literal { batch_key, .. } => batch_key
                .projection_columns
                .iter()
                .map(|column| {
                    if column.ends_with("_info") || column.ends_with("_data") {
                        4
                    } else {
                        1
                    }
                })
                .sum::<usize>()
                .max(1),
            Self::Exact { .. } => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RetainedSelectLiteralBatchKey {
    table: String,
    projection_columns: Vec<String>,
    filter_column: String,
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
    sql: &str,
    select: Select,
) -> Option<RetainedSelectBatchCandidate> {
    let exact_key = sql.to_string();
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
    let _needle = retained_select_literal_needle(&select)?;
    Some(RetainedSelectBatchCandidate::Literal {
        batch_key: RetainedSelectLiteralBatchKey {
            table: select.table.clone(),
            projection_columns: projection_columns.clone(),
            filter_column: filters[0].column.clone(),
        },
        exact_key,
        select,
        sql: sql.to_string(),
    })
}

fn retained_select_batch_candidate(
    command: &EngineCommand,
) -> Option<RetainedSelectBatchCandidate> {
    let EngineCommand::SimpleQuery(sql) = command else {
        return None;
    };
    let Ok(Command::Select(select)) = parse_command(sql) else {
        return None;
    };
    retained_select_literal_batch_candidate(sql, select).or_else(|| {
        Some(RetainedSelectBatchCandidate::Exact {
            exact_key: sql.clone(),
        })
    })
}

fn recv_next_batch_candidate(
    request_rx: &mpsc::Receiver<EngineRequest>,
    batch_started: Instant,
    admission_window: Duration,
) -> Result<(Option<EngineRequest>, Duration), mpsc::TryRecvError> {
    match request_rx.try_recv() {
        Ok(request) => return Ok((Some(request), Duration::ZERO)),
        Err(mpsc::TryRecvError::Disconnected) => return Err(mpsc::TryRecvError::Disconnected),
        Err(mpsc::TryRecvError::Empty) => {}
    }
    if admission_window.is_zero() {
        return Ok((None, Duration::ZERO));
    }
    let elapsed = batch_started.elapsed();
    if elapsed >= admission_window {
        return Ok((None, Duration::ZERO));
    }
    let wait_started = Instant::now();
    let remaining = admission_window.saturating_sub(elapsed);
    match request_rx.recv_timeout(remaining) {
        Ok(request) => Ok((Some(request), wait_started.elapsed())),
        Err(mpsc::RecvTimeoutError::Timeout) => Ok((None, wait_started.elapsed().min(remaining))),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(mpsc::TryRecvError::Disconnected),
    }
}

fn push_ready_lane(
    lanes: &mut HashMap<String, VecDeque<EngineRequest>>,
    lane_order: &mut VecDeque<String>,
    request: EngineRequest,
) -> bool {
    let Some(candidate) = request.batch_candidate.as_ref() else {
        return false;
    };
    let route_key = candidate.route_key();
    if !lanes.contains_key(&route_key) {
        lane_order.push_back(route_key.clone());
    }
    lanes.entry(route_key).or_default().push_back(request);
    true
}

fn remove_ready_lane_order(lane_order: &mut VecDeque<String>, route_key: &str) {
    if let Some(index) = lane_order.iter().position(|key| key == route_key) {
        lane_order.remove(index);
    }
}

fn pop_ready_lane(
    lanes: &mut HashMap<String, VecDeque<EngineRequest>>,
    lane_order: &mut VecDeque<String>,
) -> Option<EngineRequest> {
    while let Some(route_key) = lane_order.pop_front() {
        let Some(queue) = lanes.get_mut(&route_key) else {
            continue;
        };
        let request = queue.pop_front();
        if queue.is_empty() {
            lanes.remove(&route_key);
        } else {
            lane_order.push_back(route_key);
        }
        if request.is_some() {
            return request;
        }
    }
    None
}

fn pop_matching_ready_lane(
    lanes: &mut HashMap<String, VecDeque<EngineRequest>>,
    lane_order: &mut VecDeque<String>,
    route_key: &str,
) -> Option<EngineRequest> {
    let queue = lanes.get_mut(route_key)?;
    let request = queue.pop_front();
    if queue.is_empty() {
        lanes.remove(route_key);
        remove_ready_lane_order(lane_order, route_key);
    }
    request
}

fn route_lane_scan_limit_for_batch(
    policy: RouteLaneScanPolicy,
    configured_limit: usize,
    microbatch_max: usize,
    batch_len: usize,
    payload_aware: bool,
    projected_payload_weight: usize,
    matching_lane_depth: usize,
) -> usize {
    match policy {
        RouteLaneScanPolicy::Fixed => configured_limit,
        RouteLaneScanPolicy::Adaptive => {
            let mut limit = configured_limit;
            if payload_aware {
                if projected_payload_weight >= 5 {
                    limit = limit.saturating_div(2).max(8).min(configured_limit);
                }
                if matching_lane_depth > 0 && batch_len.saturating_add(matching_lane_depth) >= 8 {
                    limit = limit.saturating_div(2).max(8).min(configured_limit);
                }
            }
            let half_full = microbatch_max.saturating_add(1) / 2;
            let quarter_full = microbatch_max.saturating_add(3) / 4;
            if batch_len >= half_full {
                limit.saturating_div(4).max(4).min(configured_limit)
            } else if batch_len >= quarter_full {
                limit.saturating_div(2).max(8).min(configured_limit)
            } else {
                limit
            }
        }
    }
}

fn request_engine(
    request_sender: &EngineRequestSender,
    command: EngineCommand,
) -> Result<EngineResponse, String> {
    let (response_tx, response_rx) = mpsc::channel();
    let batch_candidate = retained_select_batch_candidate(&command);
    let use_latency_lane = request_sender.latency_lane_enabled
        && matches!(
            batch_candidate,
            Some(RetainedSelectBatchCandidate::Literal { .. })
        );
    let request = EngineRequest {
        command,
        batch_candidate,
        enqueued_at: Instant::now(),
        response_tx,
    };
    let tx = if use_latency_lane {
        &request_sender.latency_tx
    } else {
        &request_sender.throughput_tx
    };
    tx.send(request)
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
    request_sender: &EngineRequestSender,
    chunks: Vec<Vec<Vec<SqlValue>>>,
) -> Result<(), String> {
    for rows in chunks {
        match request_engine(
            request_sender,
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
    retained_read_job_route_id: Option<&'a str>,
    retained_snapshot_generation: Option<u64>,
    retained_read_submit_micros: Option<u64>,
    retained_read_complete_micros: Option<u64>,
    h2d_delta: u64,
    d2h_delta: u64,
    kernel_delta: u64,
    result_rows: usize,
    microbatch_kind: &'a str,
    microbatch_size: u64,
    microbatch_unique_selects: u64,
    microbatch_admission_wait_micros: u64,
}

impl std::fmt::Display for SelectPhaseFact<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{{\"sql\":\"{}\",\"query_shape\":\"{}\",\"scheduler_queue_wait_micros\":{},\"engine_execute_micros\":{},\"result_materialize_micros\":{},\"client_write_micros\":{},\"retained_wall_micros\":{},\"retained_device_lookup_micros\":{},\"retained_match_index_micros\":{},\"retained_selected_projection_micros\":{},\"retained_result_materialization_micros\":{},\"retained_cuda_event_micros\":{},\"retained_matched_rows\":{},\"retained_read_job_route_id\":{},\"retained_snapshot_generation\":{},\"retained_read_submit_micros\":{},\"retained_read_complete_micros\":{},\"h2d_delta\":{},\"d2h_delta\":{},\"kernel_delta\":{},\"result_rows\":{},\"microbatch_kind\":\"{}\",\"microbatch_size\":{},\"microbatch_unique_selects\":{},\"microbatch_admission_wait_micros\":{}}}",
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
            json_optional_str(self.retained_read_job_route_id),
            json_optional_u64(self.retained_snapshot_generation),
            json_optional_u64(self.retained_read_submit_micros),
            json_optional_u64(self.retained_read_complete_micros),
            self.h2d_delta,
            self.d2h_delta,
            self.kernel_delta,
            self.result_rows,
            json_escape(self.microbatch_kind),
            self.microbatch_size,
            self.microbatch_unique_selects,
            self.microbatch_admission_wait_micros
        )
    }
}

fn json_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_optional_str(value: Option<&str>) -> String {
    value
        .map(|value| format!("\"{}\"", json_escape(value)))
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
    request_sender: EngineRequestSender,
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
        request_engine(&request_sender, EngineCommand::Startup(startup))?,
    )?;

    let mut pending_copy: Option<PendingCopy> = None;
    while let Some(frame) = read_tagged_frame(&mut stream).map_err(|err| err.to_string())? {
        match parse_frontend_message(&frame).map_err(|err| err.to_string())? {
            FrontendMessage::SimpleQuery(sql) if parse_copy_from_stdin(&sql).is_some() => {
                retained_read_response_cache
                    .lock()
                    .map_err(|_| "retained read response cache lock poisoned".to_string())?
                    .invalidate();
                match request_engine(&request_sender, EngineCommand::StartCopy(sql))? {
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
                        let response = request_engine(
                            &request_sender,
                            EngineCommand::SimpleQuery(sql.clone()),
                        )?;
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
                            request_engine(&request_sender, EngineCommand::SimpleQuery(sql))?,
                        )?;
                    }
                }
            }
            FrontendMessage::CopyData(bytes) => {
                let pending = pending_copy
                    .as_mut()
                    .ok_or("COPY data arrived without pending COPY stream")?;
                let chunks = pending.push_bytes(&bytes).map_err(|err| err.to_string())?;
                commit_pending_copy_chunks(pending, &request_sender, chunks)?;
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
                commit_pending_copy_chunks(&mut pending, &request_sender, chunks)?;
                write_engine_response(
                    &mut stream,
                    request_engine(&request_sender, EngineCommand::FinishCopy(pending))?,
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
    let gpu_microbatch_admission_window_micros =
        std::env::var("GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ADMISSION_WINDOW_MICROS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
    let gpu_microbatch_admission_window =
        Duration::from_micros(gpu_microbatch_admission_window_micros);
    let gpu_microbatch_ready_scan_limit =
        std::env::var("GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_READY_SCAN_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1);
    let gpu_microbatch_route_lane_scan_limit =
        std::env::var("GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_SCAN_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(32);
    let gpu_microbatch_route_lane_payload_aware =
        std::env::var("GPU_DB_P8_ENGINE_PGWIRE_GPU_MICROBATCH_ROUTE_LANE_PAYLOAD_AWARE")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
    let requested_gpu_latency_lane_enabled =
        std::env::var("GPU_DB_P8_ENGINE_PGWIRE_GPU_LATENCY_LANE_RETAINED_LITERAL")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
    let gpu_latency_lane_enabled = requested_gpu_latency_lane_enabled && max_sessions >= 128;
    let prepared_retained_routes_enabled =
        std::env::var("GPU_DB_P8_ENGINE_PGWIRE_PREPARED_RETAINED_ROUTES")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(true);
    let requested_gpu_microbatch_route_lane_scan_policy = RouteLaneScanPolicy::from_env();
    let gpu_microbatch_route_lane_scan_policy = if requested_gpu_microbatch_route_lane_scan_policy
        == RouteLaneScanPolicy::Adaptive
        && max_sessions < 128
    {
        RouteLaneScanPolicy::Fixed
    } else {
        requested_gpu_microbatch_route_lane_scan_policy
    };

    let listener = TcpListener::bind(&listen)?;
    let (request_tx, request_rx) = mpsc::channel::<EngineRequest>();
    let (latency_request_tx, latency_request_rx) = mpsc::channel::<EngineRequest>();
    let (completed_tx, completed_rx) = mpsc::channel::<()>();
    let retained_read_response_cache = Arc::new(Mutex::new(RetainedReadResponseCache::new(
        retained_read_response_cache_enabled,
    )));
    let accept_request_sender = EngineRequestSender {
        throughput_tx: request_tx.clone(),
        latency_tx: latency_request_tx.clone(),
        latency_lane_enabled: gpu_latency_lane_enabled,
    };
    let accept_completed_tx = completed_tx.clone();
    let accept_retained_read_response_cache = Arc::clone(&retained_read_response_cache);
    let accept_handle = thread::spawn(move || -> Result<(), String> {
        for stream in listener.incoming().take(max_sessions) {
            let stream = stream.map_err(|err| err.to_string())?;
            let client_request_sender = accept_request_sender.clone();
            let client_completed_tx = accept_completed_tx.clone();
            let client_retained_read_response_cache =
                Arc::clone(&accept_retained_read_response_cache);
            thread::spawn(move || {
                let completed = handle_client_io(
                    stream,
                    client_request_sender,
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
    state.fact(
        "owner_thread_gpu_microbatch_admission_window_micros",
        gpu_microbatch_admission_window_micros,
    )?;
    state.fact(
        "owner_thread_gpu_microbatch_ready_scan_limit",
        gpu_microbatch_ready_scan_limit,
    )?;
    state.fact(
        "owner_thread_gpu_microbatch_route_lane_scan_limit",
        gpu_microbatch_route_lane_scan_limit,
    )?;
    state.fact(
        "owner_thread_gpu_microbatch_route_lane_scan_requested_policy",
        requested_gpu_microbatch_route_lane_scan_policy.as_str(),
    )?;
    state.fact(
        "owner_thread_gpu_microbatch_route_lane_scan_effective_policy",
        gpu_microbatch_route_lane_scan_policy.as_str(),
    )?;
    state.fact(
        "owner_thread_gpu_microbatch_route_lane_payload_aware",
        gpu_microbatch_route_lane_payload_aware,
    )?;
    state.fact("owner_thread_gpu_microbatch_preclassified_requests", true)?;
    state.fact("owner_thread_gpu_microbatch_route_lanes", true)?;
    state.fact(
        "owner_thread_gpu_latency_lane_retained_literal_requested",
        requested_gpu_latency_lane_enabled,
    )?;
    state.fact(
        "owner_thread_gpu_latency_lane_retained_literal_effective",
        gpu_latency_lane_enabled,
    )?;
    state.fact(
        "owner_thread_gpu_prepared_retained_routes",
        prepared_retained_routes_enabled,
    )?;
    state.fact("owner_thread_gpu_microbatch_exact_select", true)?;
    state.fact("owner_thread_gpu_microbatch_multi_literal_select", true)?;
    state.fact("max_sessions", max_sessions)?;
    state.fact(
        "crate_direction",
        "gpu_db_engine_depends_on_gpu_db_protocol",
    )?;

    let mut completed = 0;
    let mut deferred_requests = VecDeque::new();
    let mut ready_lanes: HashMap<String, VecDeque<EngineRequest>> = HashMap::new();
    let mut ready_lane_order = VecDeque::new();
    let mut gpu_microbatch_batches = 0_u64;
    let mut gpu_microbatch_coalesced_requests = 0_u64;
    let mut gpu_literal_microbatch_batches = 0_u64;
    let mut gpu_literal_microbatch_coalesced_requests = 0_u64;
    let mut gpu_route_lane_scan_batches = 0_u64;
    let mut gpu_route_lane_scan_budget = 0_u64;
    let mut gpu_route_lane_scanned_ready = 0_u64;
    let mut gpu_route_lane_projected_payload_weight = 0_u64;
    let mut gpu_latency_lane_requests = 0_u64;
    let mut gpu_prepared_retained_route_requests = 0_u64;
    while completed < max_sessions {
        while completed_rx.try_recv().is_ok() {
            completed += 1;
        }
        let next_request = match latency_request_rx.try_recv() {
            Ok(request) => Ok((request, true)),
            Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {
                if let Some(request) = pop_ready_lane(&mut ready_lanes, &mut ready_lane_order) {
                    Ok((request, false))
                } else if let Some(request) = deferred_requests.pop_front() {
                    Ok((request, false))
                } else {
                    request_rx
                        .recv_timeout(Duration::from_millis(1))
                        .map(|request| (request, false))
                }
            }
        };
        match next_request {
            Ok((request, latency_lane_request)) => {
                if latency_lane_request {
                    gpu_latency_lane_requests = gpu_latency_lane_requests.saturating_add(1);
                }
                let batch_request_rx = if latency_lane_request {
                    &latency_request_rx
                } else {
                    &request_rx
                };
                let mut batch = vec![request];
                let mut literal_microbatch = false;
                let mut microbatch_admission_wait_micros = 0_u64;
                if gpu_microbatch_max > 1 {
                    let batch_started = Instant::now();
                    let mut scanned_ready = 0_usize;
                    if let Some(RetainedSelectBatchCandidate::Literal {
                        batch_key,
                        exact_key,
                        ..
                    }) = batch[0].batch_candidate.clone()
                    {
                        let route_key = batch[0]
                            .batch_candidate
                            .as_ref()
                            .expect("literal candidate exists")
                            .route_key();
                        let projected_payload_weight = batch[0]
                            .batch_candidate
                            .as_ref()
                            .expect("literal candidate exists")
                            .projected_payload_weight();
                        while batch.len() < gpu_microbatch_max {
                            if let Some(next) = pop_matching_ready_lane(
                                &mut ready_lanes,
                                &mut ready_lane_order,
                                &route_key,
                            ) {
                                batch.push(next);
                                continue;
                            }
                            if !deferred_requests.is_empty() {
                                break;
                            }
                            match recv_next_batch_candidate(
                                batch_request_rx,
                                batch_started,
                                gpu_microbatch_admission_window,
                            ) {
                                Ok((Some(next), waited)) => {
                                    microbatch_admission_wait_micros =
                                        microbatch_admission_wait_micros.saturating_add(
                                            waited.as_micros().try_into().unwrap_or(u64::MAX),
                                        );
                                    scanned_ready = scanned_ready.saturating_add(1);
                                    match &next.batch_candidate {
                                        Some(RetainedSelectBatchCandidate::Literal {
                                            batch_key: next_key,
                                            ..
                                        }) if *next_key == batch_key => {
                                            batch.push(next);
                                        }
                                        Some(_) => {
                                            push_ready_lane(
                                                &mut ready_lanes,
                                                &mut ready_lane_order,
                                                next,
                                            );
                                            let scan_limit = route_lane_scan_limit_for_batch(
                                                gpu_microbatch_route_lane_scan_policy,
                                                gpu_microbatch_route_lane_scan_limit,
                                                gpu_microbatch_max,
                                                batch.len(),
                                                gpu_microbatch_route_lane_payload_aware,
                                                projected_payload_weight,
                                                ready_lanes
                                                    .get(&route_key)
                                                    .map(VecDeque::len)
                                                    .unwrap_or(0),
                                            );
                                            if scanned_ready >= scan_limit {
                                                break;
                                            }
                                        }
                                        _ => {
                                            deferred_requests.push_back(next);
                                            break;
                                        }
                                    }
                                }
                                Ok((None, waited)) => {
                                    microbatch_admission_wait_micros =
                                        microbatch_admission_wait_micros.saturating_add(
                                            waited.as_micros().try_into().unwrap_or(u64::MAX),
                                        );
                                    break;
                                }
                                Err(
                                    mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected,
                                ) => break,
                            }
                        }
                        literal_microbatch = batch.len() > 1
                            && !batch.iter().all(|request| {
                                request
                                    .batch_candidate
                                    .as_ref()
                                    .is_some_and(|candidate| candidate.exact_key() == exact_key)
                            });
                    } else if let Some(batch_key) = batch[0]
                        .batch_candidate
                        .as_ref()
                        .map(|candidate| candidate.exact_key().to_string())
                    {
                        let route_key = format!("exact:{batch_key}");
                        let projected_payload_weight = batch[0]
                            .batch_candidate
                            .as_ref()
                            .map(RetainedSelectBatchCandidate::projected_payload_weight)
                            .unwrap_or(1);
                        while batch.len() < gpu_microbatch_max {
                            if let Some(next) = pop_matching_ready_lane(
                                &mut ready_lanes,
                                &mut ready_lane_order,
                                &route_key,
                            ) {
                                batch.push(next);
                                continue;
                            }
                            if !deferred_requests.is_empty() {
                                break;
                            }
                            match recv_next_batch_candidate(
                                batch_request_rx,
                                batch_started,
                                gpu_microbatch_admission_window,
                            ) {
                                Ok((Some(next), waited)) => {
                                    microbatch_admission_wait_micros =
                                        microbatch_admission_wait_micros.saturating_add(
                                            waited.as_micros().try_into().unwrap_or(u64::MAX),
                                        );
                                    scanned_ready = scanned_ready.saturating_add(1);
                                    if next
                                        .batch_candidate
                                        .as_ref()
                                        .is_some_and(|candidate| candidate.exact_key() == batch_key)
                                    {
                                        batch.push(next);
                                    } else if next.batch_candidate.is_some() {
                                        push_ready_lane(
                                            &mut ready_lanes,
                                            &mut ready_lane_order,
                                            next,
                                        );
                                        let scan_limit = route_lane_scan_limit_for_batch(
                                            gpu_microbatch_route_lane_scan_policy,
                                            gpu_microbatch_route_lane_scan_limit,
                                            gpu_microbatch_max,
                                            batch.len(),
                                            gpu_microbatch_route_lane_payload_aware,
                                            projected_payload_weight,
                                            ready_lanes
                                                .get(&route_key)
                                                .map(VecDeque::len)
                                                .unwrap_or(0),
                                        );
                                        if scanned_ready >= scan_limit {
                                            break;
                                        }
                                    } else {
                                        deferred_requests.push_back(next);
                                        break;
                                    }
                                }
                                Ok((None, waited)) => {
                                    microbatch_admission_wait_micros =
                                        microbatch_admission_wait_micros.saturating_add(
                                            waited.as_micros().try_into().unwrap_or(u64::MAX),
                                        );
                                    break;
                                }
                                Err(
                                    mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected,
                                ) => break,
                            }
                        }
                    }
                    if scanned_ready > 0 {
                        gpu_route_lane_scan_batches = gpu_route_lane_scan_batches.saturating_add(1);
                        gpu_route_lane_scanned_ready = gpu_route_lane_scanned_ready
                            .saturating_add(u64::try_from(scanned_ready).unwrap_or(u64::MAX));
                        let scan_limit = route_lane_scan_limit_for_batch(
                            gpu_microbatch_route_lane_scan_policy,
                            gpu_microbatch_route_lane_scan_limit,
                            gpu_microbatch_max,
                            batch.len(),
                            gpu_microbatch_route_lane_payload_aware,
                            batch[0]
                                .batch_candidate
                                .as_ref()
                                .map(RetainedSelectBatchCandidate::projected_payload_weight)
                                .unwrap_or(1),
                            0,
                        );
                        gpu_route_lane_scan_budget = gpu_route_lane_scan_budget
                            .saturating_add(u64::try_from(scan_limit).unwrap_or(u64::MAX));
                        gpu_route_lane_projected_payload_weight =
                            gpu_route_lane_projected_payload_weight.saturating_add(
                                batch[0]
                                    .batch_candidate
                                    .as_ref()
                                    .map(RetainedSelectBatchCandidate::projected_payload_weight)
                                    .and_then(|weight| u64::try_from(weight).ok())
                                    .unwrap_or(1),
                            );
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
                                    match &request.batch_candidate {
                                        Some(RetainedSelectBatchCandidate::Literal {
                                            select,
                                            sql,
                                            ..
                                        }) => Ok((sql.clone(), select.clone())),
                                        _ => Err("only compatible int4 equality SELECT can be literal-microbatched"),
                                    }
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let outputs = state.handle_multi_literal_select_batch(
                                &items,
                                scheduler_queue_wait_micros,
                                microbatch_admission_wait_micros,
                                "multi_literal_gpu",
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
                } else if prepared_retained_routes_enabled
                    && matches!(
                        batch[0].batch_candidate,
                        Some(RetainedSelectBatchCandidate::Literal { .. })
                    )
                {
                    let request = batch.pop().expect("single request batch is non-empty");
                    let result = (|| -> Result<Vec<u8>, Box<dyn Error>> {
                        let Some(RetainedSelectBatchCandidate::Literal { select, sql, .. }) =
                            &request.batch_candidate
                        else {
                            return Err("prepared retained route requires literal SELECT".into());
                        };
                        let items = [(sql.clone(), select.clone())];
                        let outputs = state.handle_multi_literal_select_batch(
                            &items,
                            scheduler_queue_wait_micros,
                            microbatch_admission_wait_micros,
                            "prepared_literal_gpu",
                        )?;
                        state.flush_facts()?;
                        outputs
                            .into_iter()
                            .next()
                            .ok_or_else(|| "prepared retained route returned no output".into())
                    })()
                    .map_err(|err| err.to_string());
                    if result.is_ok() {
                        gpu_prepared_retained_route_requests =
                            gpu_prepared_retained_route_requests.saturating_add(1);
                    }
                    let _ = request.response_tx.send(result.map(EngineResponse::Bytes));
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
    state.fact(
        "owner_thread_gpu_route_lane_scan_batches",
        gpu_route_lane_scan_batches,
    )?;
    state.fact(
        "owner_thread_gpu_route_lane_scan_budget",
        gpu_route_lane_scan_budget,
    )?;
    state.fact(
        "owner_thread_gpu_route_lane_scanned_ready",
        gpu_route_lane_scanned_ready,
    )?;
    state.fact(
        "owner_thread_gpu_route_lane_projected_payload_weight",
        gpu_route_lane_projected_payload_weight,
    )?;
    state.fact(
        "owner_thread_gpu_latency_lane_requests",
        gpu_latency_lane_requests,
    )?;
    state.fact(
        "owner_thread_gpu_prepared_retained_route_requests",
        gpu_prepared_retained_route_requests,
    )?;
    state.fact(
        "owner_thread_retained_read_job_submission_batches",
        state.retained_read_job_submission_batches,
    )?;
    state.fact(
        "owner_thread_retained_read_jobs_submitted",
        state.retained_read_jobs_submitted,
    )?;
    state.fact(
        "owner_thread_retained_read_job_submit_wall_micros_total",
        state.retained_read_job_submit_wall_micros_total,
    )?;
    state.fact(
        "owner_thread_retained_read_job_complete_wall_micros_total",
        state.retained_read_job_complete_wall_micros_total,
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
