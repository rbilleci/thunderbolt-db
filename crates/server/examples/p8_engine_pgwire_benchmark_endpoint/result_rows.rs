use std::error::Error;
use std::io::Write;
use std::time::Instant;

use gpu_db_protocol::backend::{BackendColumn, BackendWriter};
use gpu_db_protocol::SqlValue;

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
    }
}

pub(in super::super) fn write_select_result_rows<W: Write + ?Sized>(
    writer: &mut BackendWriter<'_, W>,
    columns: &[BackendColumn],
    rows: &gpu_db_engine::RowBlock,
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
