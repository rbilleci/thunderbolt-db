// Legacy COPY state and execution ownership. This is not a product execution path.

use super::{
    copy_parse_error_field, evaluate_column_default, format_sql_value, validate_check_constraints,
    validate_foreign_keys, validate_unique_indexes, write_command_complete, write_copy_data,
    write_copy_done, write_copy_in_response, write_copy_out_response, write_error, CopyInState,
    ErrorField, ReadWrite, Session,
};
use gpu_db_protocol::{parse_copy_row, CopyColumn, CopyFormat, CopyOptions, SqlValue};
use std::io;

fn copy_text_value(value: &SqlValue) -> String {
    format_sql_value(value)
        .replace('\\', r"\\")
        .replace('\t', r"\t")
        .replace('\n', r"\n")
        .replace('\r', r"\r")
}

fn copy_csv_value(value: &SqlValue, delimiter: char, quote: char, escape: char) -> String {
    let text = format_sql_value(value);
    if text.contains([delimiter, quote, escape, '\n', '\r']) {
        let mut escaped = String::with_capacity(text.len());
        for ch in text.chars() {
            if ch == quote || ch == escape {
                escaped.push(escape);
            }
            escaped.push(ch);
        }
        format!("{quote}{escaped}{quote}")
    } else {
        text
    }
}

pub(super) fn execute_copy_to_stdout(
    stream: &mut dyn ReadWrite,
    session: &Session,
    table_name: &str,
    options: CopyOptions,
) -> io::Result<()> {
    let Some(table) = session.tables.get(table_name) else {
        return write_error(
            stream,
            &ErrorField {
                code: "42P01",
                message: "relation does not exist",
                position: None,
            },
        );
    };
    write_copy_out_response(stream, table.columns.len())?;
    if options.header {
        let mut payload = String::new();
        for (idx, column) in table.columns.iter().enumerate() {
            if idx > 0 {
                payload.push(options.delimiter);
            }
            payload.push_str(&copy_csv_value(
                &SqlValue::Text(column.def.name.clone()),
                options.delimiter,
                options.quote,
                options.escape,
            ));
        }
        payload.push('\n');
        write_copy_data(stream, payload.as_bytes())?;
    }
    for row in &table.rows {
        let mut payload = String::new();
        for (idx, value) in row.iter().enumerate() {
            if idx > 0 {
                payload.push(match options.format {
                    CopyFormat::Text => '\t',
                    CopyFormat::Csv => options.delimiter,
                });
            }
            match options.format {
                CopyFormat::Text => payload.push_str(&copy_text_value(value)),
                CopyFormat::Csv => payload.push_str(&copy_csv_value(
                    value,
                    options.delimiter,
                    options.quote,
                    options.escape,
                )),
            }
        }
        payload.push('\n');
        write_copy_data(stream, payload.as_bytes())?;
    }
    write_copy_done(stream)?;
    write_command_complete(stream, &format!("COPY {}", table.rows.len()))
}

pub(super) fn begin_copy_from_stdin(
    stream: &mut dyn ReadWrite,
    session: &mut Session,
    table_name: &str,
    requested_columns: Option<Vec<String>>,
    options: CopyOptions,
    ready_after_done: bool,
) -> io::Result<()> {
    let Some(table) = session.tables.get(table_name) else {
        return write_error(
            stream,
            &ErrorField {
                code: "42P01",
                message: "relation does not exist",
                position: None,
            },
        );
    };
    let columns = requested_columns.unwrap_or_else(|| {
        table
            .columns
            .iter()
            .map(|column| column.def.name.clone())
            .collect()
    });
    if columns.len() != table.columns.len() {
        return write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "COPY FROM STDIN must provide every column",
                position: None,
            },
        );
    }
    for column in &columns {
        if !table
            .columns
            .iter()
            .any(|candidate| candidate.def.name == *column)
        {
            return write_error(
                stream,
                &ErrorField {
                    code: "42703",
                    message: "column does not exist",
                    position: None,
                },
            );
        }
    }
    session.copy_in = Some(CopyInState {
        table: table_name.to_string(),
        columns,
        format: options.format,
        header: options.header,
        delimiter: options.delimiter,
        quote: options.quote,
        escape: options.escape,
        pending_text: String::new(),
        pending_rows: Vec::new(),
        seen_terminator: false,
        ready_after_done,
    });
    write_copy_in_response(stream, table.columns.len())
}

pub(super) fn handle_copy_data(session: &mut Session, bytes: &[u8]) -> Option<ErrorField> {
    let copy = session.copy_in.as_mut()?;
    let table = session.tables.get(&copy.table)?;
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => {
            return Some(ErrorField {
                code: "22021",
                message: "COPY data contains invalid UTF-8",
                position: None,
            });
        }
    };
    copy.pending_text.push_str(text);
    while let Some(newline) = copy.pending_text.find('\n') {
        let mut line = copy.pending_text[..newline].to_string();
        if line.ends_with('\r') {
            line.pop();
        }
        copy.pending_text.drain(..=newline);
        if line == r"\." {
            copy.seen_terminator = true;
            continue;
        }
        if copy.seen_terminator && line.is_empty() {
            continue;
        }
        if copy.header {
            copy.header = false;
            continue;
        }
        let table_columns = table
            .columns
            .iter()
            .map(|column| CopyColumn {
                name: column.def.name.clone(),
                ty: column.def.ty,
            })
            .collect::<Vec<_>>();
        if let Some(error) = parse_copy_row(
            &table_columns,
            &copy.columns,
            CopyOptions {
                format: copy.format,
                header: false,
                delimiter: copy.delimiter,
                quote: copy.quote,
                escape: copy.escape,
            },
            &line,
        )
        .map_err(copy_parse_error_field)
        .map(|row| copy.pending_rows.push(row))
        .err()
        {
            return Some(error);
        }
    }
    None
}

pub(super) fn apply_copy_in_rows(session: &mut Session, copy: CopyInState) -> Option<ErrorField> {
    if !copy.pending_text.is_empty() {
        return Some(ErrorField {
            code: "22P04",
            message: "COPY data ended before row terminator",
            position: None,
        });
    }
    let catalog_indexes = session.indexes.clone();
    let table = session.tables.get(&copy.table)?.clone();
    let mut indexes = Vec::with_capacity(copy.columns.len());
    for column in &copy.columns {
        indexes.push(
            table
                .columns
                .iter()
                .position(|candidate| candidate.def.name == *column)?,
        );
    }
    let mut new_rows = Vec::with_capacity(copy.pending_rows.len());
    for row in copy.pending_rows {
        let mut projected = vec![None; table.columns.len()];
        for (source_idx, target_idx) in indexes.iter().copied().enumerate() {
            projected[target_idx] = Some(row[source_idx].clone());
        }
        for (idx, value) in projected.iter_mut().enumerate() {
            if value.is_none() {
                if let Some(default) = table.columns[idx].def.default.clone() {
                    match evaluate_column_default(session, &default) {
                        Ok(default_value) => *value = Some(default_value),
                        Err(error) => return Some(error),
                    }
                }
            }
        }
        if projected.iter().any(Option::is_none) {
            return Some(ErrorField {
                code: "0A000",
                message: "COPY must provide every column without a default",
                position: None,
            });
        }
        new_rows.push(projected.into_iter().map(Option::unwrap).collect());
    }
    let mut candidate_table = table.clone();
    candidate_table.rows.extend(new_rows.clone());
    if let Err(error) = validate_unique_indexes(&candidate_table, &catalog_indexes) {
        return Some(error);
    }
    if let Err(error) = validate_check_constraints(&candidate_table) {
        return Some(error);
    }
    let old_table = session.tables.insert(copy.table.clone(), candidate_table)?;
    if let Err(error) = validate_foreign_keys(session) {
        session.tables.insert(copy.table.clone(), old_table);
        return Some(error);
    }
    session.mark_table_dirty(copy.table);
    session.persist_catalog_snapshot();
    None
}
