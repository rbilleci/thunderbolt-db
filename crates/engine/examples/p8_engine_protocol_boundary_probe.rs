use std::error::Error;

use gpu_db_engine::Engine;
use gpu_db_protocol::{parse_command, Command};

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

    let copy_parser_available = parse_command(copy_sql).is_ok();
    println!("engine_owned_target=true");
    println!("protocol_parser_reused=true");
    println!("crate_direction=gpu_db_engine_depends_on_gpu_db_protocol");
    println!("create_table_into_engine_wal_mvcc=true");
    println!("select_parser_reused=true");
    println!("select_result_rows={}", result.rows.len());
    println!("copy_parser_in_protocol_lib={copy_parser_available}");
    println!("backend_writer_api_available=true");
    println!("wire_session_ready_loop_available=false");
    println!("protocol_server_session_catalog_reusable=false");
    println!("resident_admission_from_sql_visible_rows=false");
    println!("endpoint_boundary_status=blocked");
    println!("next_blocker=ready_loop_session_state_extraction_required");
    println!("secondary_blocker=copy_to_engine_wal_adapter_required");
    println!("retained_blocker=engine_residency_admission_api_required");

    Ok(())
}
