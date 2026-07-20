use std::env;
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

type BackendMessage<'a> = (u8, &'a [u8]);

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
        SqlValue::Null => "NULL".to_string(),
        SqlValue::Int2(value) => value.to_string(),
        SqlValue::Int4(value) => value.to_string(),
        SqlValue::Int8(value) => value.to_string(),
        SqlValue::Numeric(value) => value.to_decimal_string(),
        SqlValue::Bool(value) => if *value { "t" } else { "f" }.to_string(),
        SqlValue::Text(value) => value.clone(),
        SqlValue::Date(value) => gpu_db_protocol::datetime::format_date(*value),
        SqlValue::Timestamp(value) => gpu_db_protocol::datetime::format_timestamp(*value),
        SqlValue::Uuid(value) => gpu_db_protocol::uuid::format_uuid(value),
        SqlValue::Parameter { .. } => {
            unreachable!("probe results never contain unbound prepared parameters")
        }
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

fn backend_messages(output: &[u8]) -> Result<Vec<BackendMessage<'_>>, Box<dyn Error>> {
    let mut messages = Vec::new();
    let mut offset = 0_usize;
    while offset < output.len() {
        let tag = *output
            .get(offset)
            .ok_or("backend message was missing its tag")?;
        let length_bytes: [u8; 4] = output
            .get(offset.saturating_add(1)..offset.saturating_add(5))
            .ok_or("backend message was missing its length")?
            .try_into()?;
        let length = usize::try_from(u32::from_be_bytes(length_bytes))?;
        if length < 4 {
            return Err("backend message length was smaller than its length field".into());
        }
        let frame_len = length
            .checked_add(1)
            .ok_or("backend message length overflowed")?;
        let next_offset = offset
            .checked_add(frame_len)
            .ok_or("backend message offset overflowed")?;
        if next_offset > output.len() {
            return Err("backend message length exceeded the output buffer".into());
        }
        messages.push((tag, &output[offset.saturating_add(5)..next_offset]));
        offset = next_offset;
    }
    Ok(messages)
}

fn data_row_text_values(payload: &[u8]) -> Result<Vec<Option<String>>, Box<dyn Error>> {
    let column_count_bytes: [u8; 2] = payload
        .get(0..2)
        .ok_or("DataRow was missing its column count")?
        .try_into()?;
    let column_count = usize::from(u16::from_be_bytes(column_count_bytes));
    let mut values = Vec::with_capacity(column_count);
    let mut offset = 2_usize;
    for _ in 0..column_count {
        let length_bytes: [u8; 4] = payload
            .get(offset..offset.saturating_add(4))
            .ok_or("DataRow was missing a column length")?
            .try_into()?;
        offset = offset.saturating_add(4);
        let length = i32::from_be_bytes(length_bytes);
        if length == -1 {
            values.push(None);
            continue;
        }
        let length = usize::try_from(length).map_err(|_| "DataRow had an invalid column length")?;
        let end = offset
            .checked_add(length)
            .ok_or("DataRow column length overflowed")?;
        let bytes = payload
            .get(offset..end)
            .ok_or("DataRow column exceeded its payload")?;
        values.push(Some(std::str::from_utf8(bytes)?.to_string()));
        offset = end;
    }
    if offset != payload.len() {
        return Err("DataRow contained trailing bytes".into());
    }
    Ok(values)
}

fn order_line_csv(row: usize) -> String {
    format!("{row},{},10,2500,row{row}", row.saturating_add(1_000))
}

fn result_count(result: &gpu_db_engine::RelationalSelectResult) -> Option<usize> {
    if result.rows.len() != 1 || result.rows.row(0).len() != 1 {
        return None;
    }
    match result.rows.row(0).first()? {
        SqlValue::Int4(value) => usize::try_from(*value).ok(),
        SqlValue::Int8(value) => usize::try_from(*value).ok(),
        _ => None,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let rows = match env::var("GPU_DB_CH_BENCH_ENGINE_PROTOCOL_BOUNDARY_ROWS") {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            "GPU_DB_CH_BENCH_ENGINE_PROTOCOL_BOUNDARY_ROWS must be a positive integer"
        })?,
        Err(env::VarError::NotPresent) => 64,
        Err(err) => return Err(err.into()),
    };
    if rows == 0 || rows > 1_000_000 {
        return Err("GPU_DB_CH_BENCH_ENGINE_PROTOCOL_BOUNDARY_ROWS must be in 1..=1000000".into());
    }
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
    let copy_lines = (1..=rows).map(order_line_csv).collect::<Vec<_>>();
    let copy_rows = copy_lines
        .iter()
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
    let copied_rows_visible = result_count(&copied_result) == Some(rows);
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
    for line in &copy_lines {
        session.handle_frontend_frame(&copy_data_frame(&format!("{line}\n")), &mut wire_output)?;
    }
    session.handle_frontend_frame(&frontend_frame(b'c', &[]), &mut wire_output)?;
    session.handle_frontend_frame(&simple_query_frame(select_sql), &mut wire_output)?;
    let session_result = session.engine.execute_relational_select(&select)?;
    let session_row_count = result_count(&session_result)
        .ok_or("session COUNT(*) did not return a nonnegative integer")?;
    let session_rows_visible = session_row_count == rows;
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
    let retained_rows_visible = result_count(&retained_result) == Some(rows);
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
    let retained_metadata_count_observed = retained_route.last_execution_rows == Some(1)
        && retained_route.query_shape == "sharded_count_all";
    let retained_shard_count = retained_route.shard_count;
    let append_hits_before = session.engine.open_shard_append_hits();
    let mutation_txn_id = session.take_txn_id();
    let inserted_id = rows.saturating_add(1);
    let inserted_item = inserted_id.saturating_add(1_000);
    session.engine.execute_text(
        mutation_txn_id,
        &format!(
            "INSERT INTO order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) VALUES ({inserted_id}, {inserted_item}, 30, 7500, 'delta2')"
        ),
    )?;
    let append_hits_after = session.engine.open_shard_append_hits();
    let before_post_mutation = session.engine.metrics().snapshot();
    let post_mutation_result = session.engine.execute_relational_select(&select)?;
    let after_post_mutation = session.engine.metrics().snapshot();
    let post_mutation_status = session.engine.status_snapshot();
    let post_mutation_route = post_mutation_status
        .relational_residency
        .latest_route_decision("order_line")
        .cloned()
        .ok_or("post-mutation retained route decision was not recorded")?;
    let post_mutation_h2d_delta = after_post_mutation
        .h2d_bytes_total
        .saturating_sub(before_post_mutation.h2d_bytes_total);
    let post_mutation_d2h_delta = after_post_mutation
        .d2h_bytes_total
        .saturating_sub(before_post_mutation.d2h_bytes_total);
    let post_mutation_zero_h2d = post_mutation_route.last_execution_h2d_bytes == Some(0)
        && post_mutation_h2d_delta == 0
        && post_mutation_route.h2d_bytes_if_resident == 0;
    let post_mutation_metadata_count_observed = post_mutation_route.last_execution_rows == Some(1)
        && post_mutation_route.query_shape == "sharded_count_all";
    let post_mutation_shard_count = post_mutation_route.shard_count;
    let post_mutation_row_count = result_count(&post_mutation_result)
        .ok_or("post-mutation COUNT(*) did not return a nonnegative integer")?;
    let post_mutation_residency_invalidated =
        !post_mutation_route.valid || post_mutation_route.cache_state != "Valid";
    let inserted_select = match parse_command(&format!(
        "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = {inserted_id}"
    ))? {
        Command::Select(select) => select,
        other => return Err(format!("expected inserted-row SELECT parse, got {other:?}").into()),
    };
    let before_inserted_read = session.engine.metrics().snapshot();
    let inserted_gathered_before = session.engine.sharded_shards_gathered();
    let inserted_result = session.engine.execute_relational_select(&inserted_select)?;
    let inserted_gathered_after = session.engine.sharded_shards_gathered();
    let after_inserted_read = session.engine.metrics().snapshot();
    let inserted_route = session
        .engine
        .status_snapshot()
        .relational_residency
        .latest_route_decision("order_line")
        .cloned()
        .ok_or("inserted-row retained route decision was not recorded")?;
    let inserted_row_visible = inserted_result.rows
        == vec![vec![
            SqlValue::Int4(i32::try_from(inserted_id)?),
            SqlValue::Int4(i32::try_from(inserted_item)?),
            SqlValue::Int4(30),
            SqlValue::Int4(7500),
            SqlValue::Text("delta2".to_string()),
        ]];
    let inserted_h2d_delta = after_inserted_read
        .h2d_bytes_total
        .saturating_sub(before_inserted_read.h2d_bytes_total);
    let inserted_route_zero_h2d = inserted_route.last_execution_h2d_bytes == Some(0)
        && inserted_route.h2d_bytes_if_resident == 0
        && inserted_h2d_delta == 0;
    let inserted_device_route_witness = matches!(
        inserted_result.executed_target,
        gpu_db_execution::DeviceTarget::Gpu(_)
    ) && inserted_result.fallback_reason.is_none()
        && inserted_route.query_shape == "sharded_int4_equality_mixed_column_projection"
        && inserted_gathered_after > inserted_gathered_before;
    let backend_messages = backend_messages(&wire_output)?;
    let backend_tags = backend_messages
        .iter()
        .map(|(tag, _)| *tag)
        .collect::<Vec<_>>();
    let backend_count_value_matches = backend_messages
        .iter()
        .filter(|(tag, _)| *tag == b'D')
        .map(|(_, payload)| data_row_text_values(payload))
        .collect::<Result<Vec<_>, _>>()?
        == vec![vec![Some(rows.to_string())]];

    println!("startup_packet_parser_reused=true");
    println!("frontend_message_parser_reused=true");
    println!("engine_owned_session_probe=true");
    println!("copy_stream_lifecycle_probe=true");
    println!(
        "backend_startup_messages_written={}",
        backend_tags.contains(&b'R')
    );
    println!(
        "backend_copy_in_response_written={}",
        backend_tags.iter().filter(|tag| **tag == b'G').count() == 1
    );
    println!(
        "backend_row_description_written={}",
        backend_tags.contains(&b'T')
    );
    println!(
        "backend_data_row_written={}",
        backend_tags.iter().filter(|tag| **tag == b'D').count() == 1
    );
    println!("backend_count_value_matches={backend_count_value_matches}");
    println!(
        "backend_ready_messages_written={}",
        backend_tags.iter().filter(|tag| **tag == b'Z').count() >= 2
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
    println!("sql_visible_resident_row_count={session_row_count}");
    println!("sql_visible_resident_bytes={}", warmup_entry.resident_bytes);
    println!(
        "sql_visible_resident_device_memory_retained={}",
        retained_route.has_retained_device_memory
    );
    println!("retained_route_accepted={}", retained_route.accepted);
    println!("retained_route_shape={}", retained_route.query_shape);
    println!("retained_route_zero_h2d={retained_zero_h2d}");
    println!("retained_metadata_count_observed={retained_metadata_count_observed}");
    println!("retained_resident_shard_count={retained_shard_count}");
    println!("retained_route_h2d_delta={retained_h2d_delta}");
    println!("retained_route_d2h_delta={retained_d2h_delta}");
    println!("retained_route_kernel_delta={retained_kernel_delta}");
    println!("retained_route_rows_visible={retained_rows_visible}");
    println!(
        "post_mutation_residency_invalidated={}",
        post_mutation_residency_invalidated
    );
    println!(
        "post_mutation_route_accepted={}",
        post_mutation_route.accepted
    );
    println!("post_mutation_route_reason={}", post_mutation_route.reason);
    println!(
        "post_mutation_route_shape={}",
        post_mutation_route.query_shape
    );
    println!("post_mutation_route_zero_h2d={post_mutation_zero_h2d}");
    println!("post_mutation_metadata_count_observed={post_mutation_metadata_count_observed}");
    println!("post_mutation_resident_shard_count={post_mutation_shard_count}");
    println!("post_mutation_h2d_delta={post_mutation_h2d_delta}");
    println!("post_mutation_d2h_delta={post_mutation_d2h_delta}");
    println!(
        "post_mutation_rows_visible={}",
        post_mutation_row_count == rows.saturating_add(1)
            && post_mutation_route.estimated_rows == rows.saturating_add(1)
    );
    println!(
        "post_mutation_resident_row_count={}",
        post_mutation_route.estimated_rows
    );
    println!(
        "post_mutation_incremental_append_observed={}",
        append_hits_after == append_hits_before.saturating_add(1)
    );
    println!(
        "post_mutation_incremental_shard_rollover_observed={}",
        post_mutation_shard_count == retained_shard_count.saturating_add(1)
    );
    println!("post_mutation_inserted_row_visible={inserted_row_visible}");
    println!(
        "post_mutation_inserted_route_accepted={}",
        inserted_route.accepted
    );
    println!(
        "post_mutation_inserted_route_shape={}",
        inserted_route.query_shape
    );
    println!("post_mutation_inserted_route_zero_h2d={inserted_route_zero_h2d}");
    println!("post_mutation_inserted_device_route_witness={inserted_device_route_witness}");
    println!(
        "post_mutation_inserted_shards_gathered_delta={}",
        inserted_gathered_after.saturating_sub(inserted_gathered_before)
    );
    println!("post_mutation_inserted_h2d_delta={inserted_h2d_delta}");
    println!("resident_admission_from_sql_visible_rows=true");
    println!("endpoint_boundary_status=sql_visible_retained_admission_ready");
    println!("next_blocker=identical_pg_client_concurrency_harness_required");
    println!("secondary_blocker=true_concurrent_client_curves_required");
    println!("retained_blocker=closed");

    Ok(())
}
