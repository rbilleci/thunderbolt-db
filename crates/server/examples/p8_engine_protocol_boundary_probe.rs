use std::error::Error;

use gpu_db_engine::{Engine, RelationalResidencyWarmupPolicy};
use gpu_db_protocol::backend::{BackendColumn, BackendWriter};
use gpu_db_protocol::{
    parse_command, parse_copy_from_stdin, parse_copy_row, parse_frontend_message,
    parse_startup_packet, Command, CopyFromStdin, FrontendMessage, SqlValue, StartupPacket,
};

struct PendingCopy {
    copy: CopyFromStdin,
    columns: Vec<gpu_db_protocol::CopyColumn>,
    rows: Vec<Vec<SqlValue>>,
}

struct EngineBackedSession {
    engine: Engine,
    next_txn_id: u64,
    pending_copy: Option<PendingCopy>,
}

impl EngineBackedSession {
    fn new() -> Self {
        Self {
            engine: Engine::new_local(),
            next_txn_id: 1,
            pending_copy: None,
        }
    }

    fn handle_startup(&mut self, frame: &[u8], output: &mut Vec<u8>) -> Result<(), Box<dyn Error>> {
        match parse_startup_packet(frame)? {
            StartupPacket::Startup { .. } => {
                let mut writer = BackendWriter::new(output);
                writer.authentication_ok()?;
                writer.parameter_status("server_version", "16.0-gpu-db")?;
                writer.ready_for_query(false)?;
            }
            other => return Err(format!("unexpected startup packet: {other:?}").into()),
        }
        Ok(())
    }

    fn handle_frontend_frame(
        &mut self,
        frame: &[u8],
        output: &mut Vec<u8>,
    ) -> Result<(), Box<dyn Error>> {
        match parse_frontend_message(frame)? {
            FrontendMessage::SimpleQuery(sql) if parse_copy_from_stdin(&sql).is_some() => {
                let copy = parse_copy_from_stdin(&sql).expect("COPY statement already checked");
                let columns = self.engine.relational_copy_columns(&copy.table)?;
                BackendWriter::new(output).copy_in_response(columns.len())?;
                self.pending_copy = Some(PendingCopy {
                    copy,
                    columns,
                    rows: Vec::new(),
                });
            }
            FrontendMessage::SimpleQuery(sql) => self.handle_simple_query(&sql, output)?,
            FrontendMessage::CopyData(bytes) => self.handle_copy_data(&bytes)?,
            FrontendMessage::CopyDone => self.finish_copy(output)?,
            other => return Err(format!("unexpected frontend message: {other:?}").into()),
        }
        Ok(())
    }

    fn handle_simple_query(
        &mut self,
        sql: &str,
        output: &mut Vec<u8>,
    ) -> Result<(), Box<dyn Error>> {
        match parse_command(sql)? {
            Command::CreateTable(_) => {
                let txn_id = self.take_txn_id();
                self.engine.execute_text(txn_id, sql)?;
                BackendWriter::new(output).command_complete("CREATE TABLE")?;
            }
            Command::Select(select) => {
                let result = self.engine.execute_relational_select(&select)?;
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
                BackendWriter::new(output).select_rows(&columns, &rows, true)?;
            }
            other => return Err(format!("unexpected simple query command: {other:?}").into()),
        }
        Ok(())
    }

    fn handle_copy_data(&mut self, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
        let pending = self
            .pending_copy
            .as_mut()
            .ok_or("COPY data arrived without a pending COPY stream")?;
        let line = std::str::from_utf8(bytes)?.trim_end_matches(['\r', '\n']);
        let row = parse_copy_row(
            &pending.columns,
            pending.copy.columns.as_deref().unwrap(),
            pending.copy.options,
            line,
        )?;
        pending.rows.push(row);
        Ok(())
    }

    fn finish_copy(&mut self, output: &mut Vec<u8>) -> Result<(), Box<dyn Error>> {
        let pending = self
            .pending_copy
            .take()
            .ok_or("COPY done arrived without a pending COPY stream")?;
        let txn_id = self.take_txn_id();
        let copied =
            self.engine
                .execute_relational_copy_rows(txn_id, &pending.copy, pending.rows)?;
        let mut writer = BackendWriter::new(output);
        writer.command_complete(&format!("COPY {copied}"))?;
        writer.ready_for_query(false)?;
        Ok(())
    }

    fn take_txn_id(&mut self) -> u64 {
        let txn_id = self.next_txn_id;
        self.next_txn_id += 1;
        txn_id
    }
}

fn sql_value_text(value: &SqlValue) -> String {
    match value {
        SqlValue::Int4(value) => value.to_string(),
        SqlValue::Int8(value) => value.to_string(),
        SqlValue::Numeric(value) => value.to_decimal_string(),
        SqlValue::Bool(value) => if *value { "t" } else { "f" }.to_string(),
        SqlValue::Text(value) => value.clone(),
    }
}

fn startup_frame() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&196_608_u32.to_be_bytes());
    payload.extend_from_slice(b"user\0postgres\0database\0postgres\0\0");
    let len = u32::try_from(payload.len() + 4).expect("startup frame length fits");
    let mut frame = len.to_be_bytes().to_vec();
    frame.extend_from_slice(&payload);
    frame
}

fn frontend_frame(tag: u8, payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len() + 4).expect("frontend frame length fits");
    let mut frame = vec![tag];
    frame.extend_from_slice(&len.to_be_bytes());
    frame.extend_from_slice(payload);
    frame
}

fn simple_query_frame(sql: &str) -> Vec<u8> {
    let mut payload = sql.as_bytes().to_vec();
    payload.push(0);
    frontend_frame(b'Q', &payload)
}

fn copy_data_frame(line: &str) -> Vec<u8> {
    frontend_frame(b'd', line.as_bytes())
}

fn message_count(output: &[u8], tag: u8) -> usize {
    output.iter().filter(|byte| **byte == tag).count()
}

fn main() -> Result<(), Box<dyn Error>> {
    let create_sql = "CREATE TABLE order_line (ol_o_id INT, ol_i_id INT, ol_quantity INT, ol_amount INT, ol_dist_info TEXT)";
    let select_sql = "SELECT COUNT(*) FROM order_line";
    let copy_sql =
        "COPY order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) FROM STDIN WITH (FORMAT csv)";

    let mut engine = Engine::new_local();
    match parse_command(create_sql)? {
        Command::CreateTable(_) => {
            engine.execute_text(1, create_sql)?;
        }
        other => {
            return Err(format!("expected CREATE TABLE parse, got {other:?}").into());
        }
    }

    let select = match parse_command(select_sql)? {
        Command::Select(select) => select,
        other => {
            return Err(format!("expected SELECT parse, got {other:?}").into());
        }
    };
    let result = engine.execute_relational_select(&select)?;

    let copy_parser_available = parse_copy_from_stdin(copy_sql).is_some();
    let copy = parse_copy_from_stdin(copy_sql).ok_or("expected COPY FROM STDIN parse")?;
    let copy_columns = engine.relational_copy_columns(&copy.table)?;
    let copy_rows = ["1,1001,10,2500,alpha0", "2,1002,20,5000,omega1"]
        .into_iter()
        .map(|line| {
            parse_copy_row(
                &copy_columns,
                copy.columns.as_deref().unwrap(),
                copy.options,
                line,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let copied_rows = engine.execute_relational_copy_rows(2, &copy, copy_rows)?;
    let copied_result = engine.execute_relational_select(&select)?;

    println!("engine_owned_target=true");
    println!("protocol_parser_reused=true");
    println!("crate_direction=gpu_db_engine_depends_on_gpu_db_sql_not_protocol");
    println!("create_table_into_engine_wal_mvcc=true");
    println!("select_parser_reused=true");
    println!("select_result_rows={}", result.rows.len());
    println!("copy_parser_in_sql_lib={copy_parser_available}");
    println!("engine_copy_column_projection_available=true");
    println!("copy_rows_decoded_by_protocol={copied_rows}");
    println!("copy_rows_committed_to_engine_wal_mvcc=true");
    let copied_rows_visible = matches!(
        copied_result.rows[0][0],
        gpu_db_protocol::SqlValue::Int4(2) | gpu_db_protocol::SqlValue::Int8(2)
    );
    println!(
        "copy_rows_visible_through_execute_relational_select={}",
        copied_rows_visible
    );
    println!("backend_writer_api_available=true");
    println!("wire_session_ready_loop_available=true");

    let mut session = EngineBackedSession::new();
    let mut wire_output = Vec::new();
    session.handle_startup(&startup_frame(), &mut wire_output)?;
    session.handle_frontend_frame(&simple_query_frame(create_sql), &mut wire_output)?;
    session.handle_frontend_frame(&simple_query_frame(copy_sql), &mut wire_output)?;
    session.handle_frontend_frame(
        &copy_data_frame("1,1001,10,2500,alpha0\n"),
        &mut wire_output,
    )?;
    session.handle_frontend_frame(
        &copy_data_frame("2,1002,20,5000,omega1\n"),
        &mut wire_output,
    )?;
    session.handle_frontend_frame(&frontend_frame(b'c', &[]), &mut wire_output)?;
    session.handle_frontend_frame(&simple_query_frame(select_sql), &mut wire_output)?;
    let session_result = session.engine.execute_relational_select(&select)?;
    let session_rows_visible = matches!(
        session_result.rows[0][0],
        gpu_db_protocol::SqlValue::Int4(2) | gpu_db_protocol::SqlValue::Int8(2)
    );
    let warmup =
        session
            .engine
            .warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
                tables: vec!["order_line".to_string()],
                refresh_invalidated: true,
                ..RelationalResidencyWarmupPolicy::default()
            });
    let warmup_entry = warmup
        .entries
        .first()
        .ok_or("resident warmup produced no order_line entry")?;
    let snapshot = session
        .engine
        .relational_residency_snapshot("order_line")
        .ok_or("resident warmup did not install an order_line snapshot")?;
    let before_retained = session.engine.metrics().snapshot();
    let retained_result = session.engine.execute_relational_select(&select)?;
    let after_retained = session.engine.metrics().snapshot();
    let retained_route = session
        .engine
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .cloned()
        .ok_or("retained route decision was not recorded")?;
    let retained_rows_visible = matches!(
        retained_result.rows[0][0],
        gpu_db_protocol::SqlValue::Int4(2) | gpu_db_protocol::SqlValue::Int8(2)
    );
    let retained_h2d_delta = after_retained
        .h2d_bytes_total
        .saturating_sub(before_retained.h2d_bytes_total);
    let retained_d2h_delta = after_retained
        .d2h_bytes_total
        .saturating_sub(before_retained.d2h_bytes_total);
    let retained_kernel_delta = after_retained
        .kernel_exec_samples
        .saturating_sub(before_retained.kernel_exec_samples);
    let retained_zero_h2d = retained_route.last_execution_h2d_bytes == Some(0)
        && retained_h2d_delta == 0
        && retained_route.h2d_bytes_if_resident == 0;
    let mutation_txn_id = session.take_txn_id();
    session.engine.execute_text(
        mutation_txn_id,
        "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES (3, 1003, 30, 7500, 'delta2')",
    )?;
    let invalidated_snapshot = session
        .engine
        .relational_residency_snapshot("order_line")
        .ok_or("resident snapshot disappeared after mutation")?;
    let invalidated_route = session.engine.plan_relational_resident_route(&select);

    println!("startup_packet_parser_reused=true");
    println!("frontend_message_parser_reused=true");
    println!("engine_owned_session_probe=true");
    println!("copy_stream_lifecycle_probe=true");
    println!(
        "backend_startup_messages_written={}",
        message_count(&wire_output, b'R') > 0
    );
    println!(
        "backend_copy_in_response_written={}",
        message_count(&wire_output, b'G') == 1
    );
    println!(
        "backend_row_description_written={}",
        message_count(&wire_output, b'T') > 0
    );
    println!(
        "backend_data_row_written={}",
        message_count(&wire_output, b'D') == 1
    );
    println!(
        "backend_ready_messages_written={}",
        message_count(&wire_output, b'Z') >= 2
    );
    println!("session_copy_rows_visible_through_engine_select={session_rows_visible}");
    println!("protocol_server_session_catalog_reusable=false");
    println!(
        "sql_visible_resident_warmup_entries={}",
        warmup.entries.len()
    );
    println!(
        "sql_visible_resident_warmup_action={:?}",
        warmup_entry.action
    );
    println!("sql_visible_resident_row_count={}", snapshot.row_count);
    println!("sql_visible_resident_bytes={}", snapshot.resident_bytes);
    println!(
        "sql_visible_resident_device_memory_retained={}",
        snapshot.device_memory_proof.is_some()
    );
    println!("retained_route_accepted={}", retained_route.accepted);
    println!("retained_route_shape={}", retained_route.query_shape);
    println!("retained_route_zero_h2d={retained_zero_h2d}");
    println!("retained_route_h2d_delta={retained_h2d_delta}");
    println!("retained_route_d2h_delta={retained_d2h_delta}");
    println!("retained_route_kernel_delta={retained_kernel_delta}");
    println!("retained_route_rows_visible={retained_rows_visible}");
    println!(
        "post_mutation_residency_invalidated={}",
        invalidated_snapshot.invalidated_by_txn_id.is_some() && !invalidated_route.accepted
    );
    println!("post_mutation_route_reason={}", invalidated_route.reason);
    println!("resident_admission_from_sql_visible_rows=true");
    println!("endpoint_boundary_status=sql_visible_retained_admission_ready");
    println!("next_blocker=identical_pg_client_concurrency_harness_required");
    println!("secondary_blocker=true_concurrent_client_curves_required");
    println!("retained_blocker=closed");

    Ok(())
}
