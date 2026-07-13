use super::{Column, ErrorField, ReadWrite};
use gpu_db_protocol::backend::{BackendColumn, BackendError, BackendWriter};
use std::io;

pub(super) fn write_authentication_ok(stream: &mut dyn ReadWrite) -> io::Result<()> {
    BackendWriter::new(stream).authentication_ok()
}

pub(super) fn write_authentication_sasl(
    stream: &mut dyn ReadWrite,
    mechanisms: &[&str],
) -> io::Result<()> {
    BackendWriter::new(stream).authentication_sasl(mechanisms)
}

pub(super) fn write_authentication_sasl_continue(
    stream: &mut dyn ReadWrite,
    data: &[u8],
) -> io::Result<()> {
    BackendWriter::new(stream).authentication_sasl_continue(data)
}

pub(super) fn write_authentication_sasl_final(
    stream: &mut dyn ReadWrite,
    data: &[u8],
) -> io::Result<()> {
    BackendWriter::new(stream).authentication_sasl_final(data)
}

pub(super) fn write_backend_key_data(
    stream: &mut dyn ReadWrite,
    process_id: i32,
    secret_key: i32,
) -> io::Result<()> {
    BackendWriter::new(stream).backend_key_data(process_id, secret_key)
}

pub(super) fn write_parameter_status(
    stream: &mut dyn ReadWrite,
    key: &str,
    value: &str,
) -> io::Result<()> {
    BackendWriter::new(stream).parameter_status(key, value)
}

pub(super) fn write_ready_for_query(
    stream: &mut dyn ReadWrite,
    in_transaction: bool,
) -> io::Result<()> {
    BackendWriter::new(stream).ready_for_query(in_transaction)
}

pub(super) fn write_empty_query_response(stream: &mut dyn ReadWrite) -> io::Result<()> {
    BackendWriter::new(stream).empty_query_response()
}

pub(super) fn write_command_complete(stream: &mut dyn ReadWrite, tag: &str) -> io::Result<()> {
    BackendWriter::new(stream).command_complete(tag)
}

pub(super) fn write_parse_complete(stream: &mut dyn ReadWrite) -> io::Result<()> {
    BackendWriter::new(stream).parse_complete()
}

pub(super) fn write_bind_complete(stream: &mut dyn ReadWrite) -> io::Result<()> {
    BackendWriter::new(stream).bind_complete()
}

pub(super) fn write_close_complete(stream: &mut dyn ReadWrite) -> io::Result<()> {
    BackendWriter::new(stream).close_complete()
}

pub(super) fn write_portal_suspended(stream: &mut dyn ReadWrite) -> io::Result<()> {
    BackendWriter::new(stream).portal_suspended()
}

pub(super) fn write_no_data(stream: &mut dyn ReadWrite) -> io::Result<()> {
    BackendWriter::new(stream).no_data()
}

pub(super) fn write_copy_out_response(
    stream: &mut dyn ReadWrite,
    column_count: usize,
) -> io::Result<()> {
    BackendWriter::new(stream).copy_out_response(column_count)
}

pub(super) fn write_copy_in_response(
    stream: &mut dyn ReadWrite,
    column_count: usize,
) -> io::Result<()> {
    BackendWriter::new(stream).copy_in_response(column_count)
}

pub(super) fn write_copy_data(stream: &mut dyn ReadWrite, bytes: &[u8]) -> io::Result<()> {
    BackendWriter::new(stream).copy_data(bytes)
}

pub(super) fn write_copy_done(stream: &mut dyn ReadWrite) -> io::Result<()> {
    BackendWriter::new(stream).copy_done()
}

pub(super) fn write_parameter_description(
    stream: &mut dyn ReadWrite,
    type_oids: &[u32],
) -> io::Result<()> {
    BackendWriter::new(stream).parameter_description(type_oids)
}

pub(super) fn write_single_row(
    stream: &mut dyn ReadWrite,
    columns: &[Column],
    rows: &[Vec<Option<String>>],
) -> io::Result<()> {
    write_select_rows(stream, columns, rows, true)
}

pub(super) fn write_select_rows(
    stream: &mut dyn ReadWrite,
    columns: &[Column],
    rows: &[Vec<Option<String>>],
    include_row_description: bool,
) -> io::Result<()> {
    write_rows_with_tag(
        stream,
        columns,
        rows,
        include_row_description,
        &format!("SELECT {}", rows.len()),
    )
}

pub(super) fn write_rows_with_tag(
    stream: &mut dyn ReadWrite,
    columns: &[Column],
    rows: &[Vec<Option<String>>],
    include_row_description: bool,
    tag: &str,
) -> io::Result<()> {
    BackendWriter::new(stream).rows_with_tag(
        &backend_columns(columns),
        rows,
        include_row_description,
        tag,
    )
}

pub(super) fn write_row_description(
    stream: &mut dyn ReadWrite,
    columns: &[Column],
) -> io::Result<()> {
    BackendWriter::new(stream).row_description(&backend_columns(columns))
}

pub(super) fn write_row_description_with_formats(
    stream: &mut dyn ReadWrite,
    columns: &[Column],
    result_format_codes: &[i16],
) -> io::Result<()> {
    BackendWriter::new(stream)
        .row_description_with_formats(&backend_columns(columns), result_format_codes)
}

pub(super) fn write_data_row_with_formats(
    stream: &mut dyn ReadWrite,
    columns: &[Column],
    values: &[Option<String>],
    result_format_codes: &[i16],
) -> io::Result<()> {
    BackendWriter::new(stream).data_row_with_formats(
        &backend_columns(columns),
        values,
        result_format_codes,
    )
}

pub(super) fn write_error(stream: &mut dyn ReadWrite, error: &ErrorField) -> io::Result<()> {
    BackendWriter::new(stream).error_response(&backend_error(error))
}

pub(super) fn backend_columns(columns: &[Column]) -> Vec<BackendColumn> {
    columns
        .iter()
        .map(|column| BackendColumn::new(column.name.clone(), column.oid, column.type_size))
        .collect()
}

pub(super) fn backend_error(error: &ErrorField) -> BackendError {
    BackendError {
        code: error.code.to_string(),
        message: error.message.to_string(),
        position: error.position.map(str::to_string),
    }
}
