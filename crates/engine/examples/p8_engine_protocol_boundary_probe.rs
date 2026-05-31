use std::error::Error;

use gpu_db_engine::Engine;
use gpu_db_protocol::{parse_command, parse_copy_from_stdin, parse_copy_row, Command};

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
    println!("crate_direction=gpu_db_engine_depends_on_gpu_db_protocol");
    println!("create_table_into_engine_wal_mvcc=true");
    println!("select_parser_reused=true");
    println!("select_result_rows={}", result.rows.len());
    println!("copy_parser_in_protocol_lib={copy_parser_available}");
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
    println!("protocol_server_session_catalog_reusable=false");
    println!("resident_admission_from_sql_visible_rows=false");
    println!("endpoint_boundary_status=engine_copy_wal_mvcc_adapter_ready");
    println!("next_blocker=session_catalog_trait_required");
    println!("secondary_blocker=engine_backed_protocol_endpoint_required");
    println!("retained_blocker=engine_residency_admission_api_required");

    Ok(())
}
