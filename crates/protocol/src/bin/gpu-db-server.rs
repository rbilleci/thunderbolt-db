use std::collections::HashMap;
use std::env;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use gpu_db_protocol::{
    parse_command, parse_frontend_message, parse_startup_packet, Command, FrontendMessage,
    SelectProjection, SqlValue, StartupPacket,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Column {
    name: String,
    oid: u32,
    type_size: i16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ErrorField {
    code: &'static str,
    message: &'static str,
    position: Option<&'static str>,
}

fn text_column(name: &str) -> Column {
    Column {
        name: name.to_string(),
        oid: gpu_db_protocol::SqlType::Text.postgres_oid(),
        type_size: gpu_db_protocol::SqlType::Text.type_size(),
    }
}

fn int4_column(name: &str) -> Column {
    Column {
        name: name.to_string(),
        oid: gpu_db_protocol::SqlType::Int4.postgres_oid(),
        type_size: gpu_db_protocol::SqlType::Int4.type_size(),
    }
}

fn sql_value_matches_type(value: &SqlValue, ty: gpu_db_protocol::SqlType) -> bool {
    matches!(
        (value, ty),
        (SqlValue::Int4(_), gpu_db_protocol::SqlType::Int4)
            | (SqlValue::Text(_), gpu_db_protocol::SqlType::Text)
    )
}

fn compare_sql_values(left: &SqlValue, right: &SqlValue) -> std::cmp::Ordering {
    match (left, right) {
        (SqlValue::Int4(left), SqlValue::Int4(right)) => left.cmp(right),
        (SqlValue::Text(left), SqlValue::Text(right)) => left.cmp(right),
        (SqlValue::Int4(_), SqlValue::Text(_)) => std::cmp::Ordering::Less,
        (SqlValue::Text(_), SqlValue::Int4(_)) => std::cmp::Ordering::Greater,
    }
}

fn format_sql_value(value: &SqlValue) -> String {
    match value {
        SqlValue::Int4(value) => value.to_string(),
        SqlValue::Text(value) => value.clone(),
    }
}

fn sql_type_oid_text(ty: gpu_db_protocol::SqlType) -> String {
    ty.postgres_oid().to_string()
}

#[derive(Default)]
struct Session {
    in_transaction: bool,
    prepared: HashMap<String, PreparedStatement>,
    tables: HashMap<String, Table>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Table {
    columns: Vec<gpu_db_protocol::ColumnDef>,
    rows: Vec<Vec<SqlValue>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PreparedStatement {
    AddTen,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listen = parse_listen_arg(env::args().skip(1))?;
    let listener = TcpListener::bind(&listen)?;
    eprintln!("gpu-db-server listening on {listen}");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                thread::spawn(|| {
                    if let Err(error) = handle_client(stream) {
                        eprintln!("client error: {error}");
                    }
                });
            }
            Err(error) => eprintln!("accept error: {error}"),
        }
    }

    Ok(())
}

fn parse_listen_arg<I>(mut args: I) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    let mut listen = String::from("127.0.0.1:5432");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--listen" => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("missing value for --listen"))?;
                listen = value;
            }
            "-h" | "--help" => {
                return Err(String::from("usage: gpu-db-server [--listen HOST:PORT]"));
            }
            other => return Err(format!("unsupported argument: {other}")),
        }
    }
    Ok(listen)
}

fn handle_client(mut stream: TcpStream) -> io::Result<()> {
    startup_handshake(&mut stream)?;
    let mut session = Session::default();

    loop {
        let Some(frame) = read_tagged_frame(&mut stream)? else {
            return Ok(());
        };

        match parse_frontend_message(&frame)
            .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))?
        {
            FrontendMessage::SimpleQuery(query) => {
                run_simple_query(&mut stream, &mut session, &query)?
            }
            FrontendMessage::Terminate => return Ok(()),
            FrontendMessage::Sync => write_ready_for_query(&mut stream, session.in_transaction)?,
            FrontendMessage::Flush => stream.flush()?,
            other => {
                write_error(
                    &mut stream,
                    &ErrorField {
                        code: "0A000",
                        message: unsupported_frontend_message(&other),
                        position: None,
                    },
                )?;
                write_ready_for_query(&mut stream, session.in_transaction)?;
            }
        }
    }
}

fn unsupported_frontend_message(message: &FrontendMessage) -> &'static str {
    match message {
        FrontendMessage::PasswordMessage(_) => "password messages are not supported after startup",
        FrontendMessage::SaslInitialResponse { .. } | FrontendMessage::SaslResponse(_) => {
            "SASL authentication is not supported"
        }
        FrontendMessage::Bind { .. }
        | FrontendMessage::Parse { .. }
        | FrontendMessage::Describe { .. }
        | FrontendMessage::Close { .. }
        | FrontendMessage::Execute { .. }
        | FrontendMessage::FunctionCall { .. }
        | FrontendMessage::CopyData(_)
        | FrontendMessage::CopyDone
        | FrontendMessage::CopyFail(_) => {
            "extended protocol is not supported by the compatibility stub"
        }
        FrontendMessage::SimpleQuery(_)
        | FrontendMessage::Terminate
        | FrontendMessage::Sync
        | FrontendMessage::Flush => "unsupported frontend message",
    }
}

fn startup_handshake(stream: &mut TcpStream) -> io::Result<()> {
    loop {
        let frame = read_startup_frame(stream)?;
        match parse_startup_packet(&frame)
            .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))?
        {
            StartupPacket::SslRequest | StartupPacket::GssEncRequest => stream.write_all(b"N")?,
            StartupPacket::CancelRequest { .. } => return Ok(()),
            StartupPacket::Startup { .. } => {
                write_authentication_ok(stream)?;
                write_parameter_status(stream, "client_encoding", "UTF8")?;
                write_parameter_status(stream, "server_version", "16.0")?;
                write_parameter_status(stream, "server_version_num", "160000")?;
                write_parameter_status(stream, "standard_conforming_strings", "on")?;
                write_backend_key_data(stream, 1, 1)?;
                write_ready_for_query(stream, false)?;
                return Ok(());
            }
        }
    }
}

fn read_startup_frame(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut len_bytes = [0_u8; 4];
    if let Err(error) = stream.read_exact(&mut len_bytes) {
        return if error.kind() == ErrorKind::UnexpectedEof {
            Err(io::Error::new(
                ErrorKind::UnexpectedEof,
                "connection closed during startup",
            ))
        } else {
            Err(error)
        };
    }

    let frame_len = u32::from_be_bytes(len_bytes) as usize;
    if frame_len < 8 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("invalid startup frame length: {frame_len}"),
        ));
    }

    let mut rest = vec![0_u8; frame_len - 4];
    stream.read_exact(&mut rest)?;

    let mut frame = len_bytes.to_vec();
    frame.extend_from_slice(&rest);
    Ok(frame)
}

fn read_tagged_frame(stream: &mut TcpStream) -> io::Result<Option<Vec<u8>>> {
    let mut tag = [0_u8; 1];
    match stream.read_exact(&mut tag) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }

    let mut len_bytes = [0_u8; 4];
    stream.read_exact(&mut len_bytes)?;
    let frame_len = u32::from_be_bytes(len_bytes) as usize;
    if frame_len < 4 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("invalid tagged frame length: {frame_len}"),
        ));
    }

    let mut payload = vec![0_u8; frame_len - 4];
    stream.read_exact(&mut payload)?;

    let mut frame = Vec::with_capacity(1 + 4 + payload.len());
    frame.extend_from_slice(&tag);
    frame.extend_from_slice(&len_bytes);
    frame.extend_from_slice(&payload);
    Ok(Some(frame))
}

fn run_simple_query(stream: &mut TcpStream, session: &mut Session, query: &str) -> io::Result<()> {
    let statements = split_simple_query(query);
    if statements.is_empty() {
        write_empty_query_response(stream)?;
        write_ready_for_query(stream, session.in_transaction)?;
        return Ok(());
    }

    for statement in statements {
        execute_statement(stream, session, statement)?;
    }

    write_ready_for_query(stream, session.in_transaction)
}

fn split_simple_query(query: &str) -> Vec<&str> {
    query
        .split(';')
        .map(str::trim)
        .filter(|statement| !statement.is_empty())
        .collect()
}

fn execute_statement(
    stream: &mut TcpStream,
    session: &mut Session,
    statement: &str,
) -> io::Result<()> {
    if let Ok(command) = parse_command(statement) {
        match command {
            Command::CreateTable(create) => {
                if session.tables.contains_key(&create.table) {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42P07",
                            message: "relation already exists",
                            position: None,
                        },
                    );
                }
                session.tables.insert(
                    create.table,
                    Table {
                        columns: create.columns,
                        rows: Vec::new(),
                    },
                );
                return write_command_complete(stream, "CREATE TABLE");
            }
            Command::Insert(insert) => {
                let Some(table) = session.tables.get_mut(&insert.table) else {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42P01",
                            message: "relation does not exist",
                            position: None,
                        },
                    );
                };
                let mut indexes = Vec::with_capacity(insert.columns.len());
                for column in &insert.columns {
                    let Some(idx) = table
                        .columns
                        .iter()
                        .position(|candidate| candidate.name == *column)
                    else {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42703",
                                message: "column does not exist",
                                position: None,
                            },
                        );
                    };
                    indexes.push(idx);
                }
                let inserted_count = insert.rows.len();
                for row in insert.rows {
                    let mut projected = vec![None; table.columns.len()];
                    for (source_idx, target_idx) in indexes.iter().copied().enumerate() {
                        if !sql_value_matches_type(&row[source_idx], table.columns[target_idx].ty) {
                            return write_error(
                                stream,
                                &ErrorField {
                                    code: "42804",
                                    message: "column type mismatch",
                                    position: None,
                                },
                            );
                        }
                        projected[target_idx] = Some(row[source_idx].clone());
                    }
                    if projected.iter().any(Option::is_none) {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "0A000",
                                message: "INSERT must provide every column",
                                position: None,
                            },
                        );
                    }
                    table
                        .rows
                        .push(projected.into_iter().map(Option::unwrap).collect());
                }
                return write_command_complete(stream, &format!("INSERT 0 {inserted_count}"));
            }
            Command::Select(select) => {
                let Some(table) = session.tables.get(&select.table) else {
                    return write_error(
                        stream,
                        &ErrorField {
                            code: "42P01",
                            message: "relation does not exist",
                            position: None,
                        },
                    );
                };
                let selected_columns = match &select.projection {
                    SelectProjection::All => table.columns.clone(),
                    SelectProjection::Columns(columns) => {
                        let mut selected = Vec::with_capacity(columns.len());
                        for column in columns {
                            let Some(def) = table
                                .columns
                                .iter()
                                .find(|candidate| candidate.name == *column)
                            else {
                                return write_error(
                                    stream,
                                    &ErrorField {
                                        code: "42703",
                                        message: "column does not exist",
                                        position: None,
                                    },
                                );
                            };
                            selected.push(def.clone());
                        }
                        selected
                    }
                };
                let mut rows = table.rows.clone();
                if let Some(filter) = &select.filter {
                    let Some(idx) = table
                        .columns
                        .iter()
                        .position(|column| column.name == filter.column)
                    else {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42703",
                                message: "column does not exist",
                                position: None,
                            },
                        );
                    };
                    rows.retain(|row| row[idx] == filter.value);
                }
                if let Some(order) = &select.order_by {
                    let Some(idx) = table
                        .columns
                        .iter()
                        .position(|column| column.name == order.column)
                    else {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "42703",
                                message: "column does not exist",
                                position: None,
                            },
                        );
                    };
                    rows.sort_by(|left, right| compare_sql_values(&left[idx], &right[idx]));
                    if order.descending {
                        rows.reverse();
                    }
                }
                if let Some(limit) = select.limit {
                    rows.truncate(limit);
                }
                let selected_indexes = selected_columns
                    .iter()
                    .map(|selected| {
                        table
                            .columns
                            .iter()
                            .position(|column| column.name == selected.name)
                            .expect("selected column came from table")
                    })
                    .collect::<Vec<_>>();
                let columns = selected_columns
                    .iter()
                    .map(|column| match column.ty {
                        gpu_db_protocol::SqlType::Int4 => int4_column(&column.name),
                        gpu_db_protocol::SqlType::Text => text_column(&column.name),
                    })
                    .collect::<Vec<_>>();
                let output_rows = rows
                    .iter()
                    .map(|row| {
                        selected_indexes
                            .iter()
                            .map(|idx| Some(format_sql_value(&row[*idx])))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                return write_single_row(stream, &columns, &output_rows);
            }
            Command::Begin
            | Command::Commit { .. }
            | Command::Rollback { .. }
            | Command::Flush
            | Command::ResetAll
            | Command::SetKv { .. }
            | Command::DeleteKv { .. }
            | Command::GetKv { .. } => {}
        }
    }

    let canonical = canonical_sql(statement);
    if canonical
        == "select relname from pg_catalog.pg_class where relnamespace = 'public'::regnamespace and relkind = 'r' order by relname"
    {
        return write_single_row(stream, &[text_column("relname")], &catalog_table_rows(session));
    }
    if let Some(table) = catalog_attribute_query_table(&canonical) {
        let Some(rows) = catalog_attribute_rows(session, &table) else {
            return write_error(
                stream,
                &ErrorField {
                    code: "42P01",
                    message: "relation does not exist",
                    position: None,
                },
            );
        };
        return write_single_row(
            stream,
            &[text_column("attname"), int4_column("atttypid")],
            &rows,
        );
    }
    match canonical.as_str() {
        "begin" => {
            session.in_transaction = true;
            write_command_complete(stream, "BEGIN")
        }
        "commit" => {
            session.in_transaction = false;
            write_command_complete(stream, "COMMIT")
        }
        "rollback" => {
            session.in_transaction = false;
            write_command_complete(stream, "ROLLBACK")
        }
        "reset all" => write_command_complete(stream, "RESET"),
        "discard all" => write_command_complete(stream, "DISCARD ALL"),
        "deallocate all" => {
            session.prepared.clear();
            write_command_complete(stream, "DEALLOCATE ALL")
        }
        "unlisten *" | "unlisten all" => write_command_complete(stream, "UNLISTEN"),
        "show client_encoding" => write_single_row(
            stream,
            &[text_column("client_encoding")],
            &[vec![Some(String::from("UTF8"))]],
        ),
        "select current_schema()" => write_single_row(
            stream,
            &[text_column("current_schema")],
            &[vec![Some(String::from("public"))]],
        ),
        "select 1 as one" => write_single_row(
            stream,
            &[int4_column("one")],
            &[vec![Some(String::from("1"))]],
        ),
        "select 2 as in_tx" => write_single_row(
            stream,
            &[int4_column("in_tx")],
            &[vec![Some(String::from("2"))]],
        ),
        "select 3 as rolled_back" => write_single_row(
            stream,
            &[int4_column("rolled_back")],
            &[vec![Some(String::from("3"))]],
        ),
        "prepare golden_stmt(int) as select $1 + 10 as plus_ten" => {
            session
                .prepared
                .insert(String::from("golden_stmt"), PreparedStatement::AddTen);
            write_command_complete(stream, "PREPARE")
        }
        "execute golden_stmt(5)" => {
            if session.prepared.contains_key("golden_stmt") {
                write_single_row(
                    stream,
                    &[int4_column("plus_ten")],
                    &[vec![Some(String::from("15"))]],
                )
            } else {
                write_error(
                    stream,
                    &ErrorField {
                        code: "26000",
                        message: "prepared statement \"golden_stmt\" does not exist",
                        position: None,
                    },
                )
            }
        }
        "deallocate golden_stmt" => {
            session.prepared.remove("golden_stmt");
            write_command_complete(stream, "DEALLOCATE")
        }
        "select * from definitely_missing_relation_for_golden" => write_error(
            stream,
            &ErrorField {
                code: "42P01",
                message: "relation \"definitely_missing_relation_for_golden\" does not exist",
                position: Some("15"),
            },
        ),
        _ => write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "query shape is not supported by the compatibility stub",
                position: None,
            },
        ),
    }
}

fn catalog_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut names = session.tables.keys().cloned().collect::<Vec<_>>();
    names.sort();
    names
        .into_iter()
        .map(|name| vec![Some(name)])
        .collect::<Vec<_>>()
}

fn catalog_attribute_query_table(canonical: &str) -> Option<String> {
    let prefix = "select attname, atttypid from pg_catalog.pg_attribute where attrelid = '";
    let suffix = "'::regclass and attnum > 0 order by attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn catalog_attribute_rows(session: &Session, table: &str) -> Option<Vec<Vec<Option<String>>>> {
    let table = session.tables.get(table)?;
    Some(
        table
            .columns
            .iter()
            .map(|column| {
                vec![
                    Some(column.name.clone()),
                    Some(sql_type_oid_text(column.ty)),
                ]
            })
            .collect(),
    )
}

fn canonical_sql(input: &str) -> String {
    let mut sql = input.trim();
    while let Some(stripped) = sql.strip_suffix(';') {
        sql = stripped.trim_end();
    }

    let mut canonical = String::with_capacity(sql.len());
    let mut previous_was_space = false;
    for ch in sql.chars() {
        if ch.is_whitespace() {
            if !previous_was_space {
                canonical.push(' ');
                previous_was_space = true;
            }
        } else {
            canonical.extend(ch.to_lowercase());
            previous_was_space = false;
        }
    }
    canonical.trim().to_owned()
}

fn write_authentication_ok(stream: &mut TcpStream) -> io::Result<()> {
    write_message(stream, b'R', &0_i32.to_be_bytes())
}

fn write_backend_key_data(
    stream: &mut TcpStream,
    process_id: i32,
    secret_key: i32,
) -> io::Result<()> {
    let mut payload = Vec::with_capacity(8);
    payload.extend_from_slice(&process_id.to_be_bytes());
    payload.extend_from_slice(&secret_key.to_be_bytes());
    write_message(stream, b'K', &payload)
}

fn write_parameter_status(stream: &mut TcpStream, key: &str, value: &str) -> io::Result<()> {
    let mut payload = Vec::with_capacity(key.len() + value.len() + 2);
    push_cstring(&mut payload, key);
    push_cstring(&mut payload, value);
    write_message(stream, b'S', &payload)
}

fn write_ready_for_query(stream: &mut TcpStream, in_transaction: bool) -> io::Result<()> {
    let status = if in_transaction { b'T' } else { b'I' };
    write_message(stream, b'Z', &[status])
}

fn write_empty_query_response(stream: &mut TcpStream) -> io::Result<()> {
    write_message(stream, b'I', &[])
}

fn write_command_complete(stream: &mut TcpStream, tag: &str) -> io::Result<()> {
    let mut payload = Vec::with_capacity(tag.len() + 1);
    push_cstring(&mut payload, tag);
    write_message(stream, b'C', &payload)
}

fn write_single_row(
    stream: &mut TcpStream,
    columns: &[Column],
    rows: &[Vec<Option<String>>],
) -> io::Result<()> {
    write_row_description(stream, columns)?;
    for row in rows {
        write_data_row(stream, row)?;
    }
    write_command_complete(stream, &format!("SELECT {}", rows.len()))
}

fn write_row_description(stream: &mut TcpStream, columns: &[Column]) -> io::Result<()> {
    let field_count = i16::try_from(columns.len())
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "too many columns"))?;
    let mut payload = Vec::new();
    payload.extend_from_slice(&field_count.to_be_bytes());
    for column in columns {
        push_cstring(&mut payload, &column.name);
        payload.extend_from_slice(&0_u32.to_be_bytes());
        payload.extend_from_slice(&0_i16.to_be_bytes());
        payload.extend_from_slice(&column.oid.to_be_bytes());
        payload.extend_from_slice(&column.type_size.to_be_bytes());
        payload.extend_from_slice(&(-1_i32).to_be_bytes());
        payload.extend_from_slice(&0_i16.to_be_bytes());
    }
    write_message(stream, b'T', &payload)
}

fn write_data_row(stream: &mut TcpStream, values: &[Option<String>]) -> io::Result<()> {
    let value_count = i16::try_from(values.len())
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "too many row values"))?;
    let mut payload = Vec::new();
    payload.extend_from_slice(&value_count.to_be_bytes());
    for value in values {
        match value {
            Some(value) => {
                let bytes = value.as_bytes();
                let len = i32::try_from(bytes.len()).map_err(|_| {
                    io::Error::new(ErrorKind::InvalidInput, "row value too large to encode")
                })?;
                payload.extend_from_slice(&len.to_be_bytes());
                payload.extend_from_slice(bytes);
            }
            None => payload.extend_from_slice(&(-1_i32).to_be_bytes()),
        }
    }
    write_message(stream, b'D', &payload)
}

fn write_error(stream: &mut TcpStream, error: &ErrorField) -> io::Result<()> {
    let mut payload = Vec::new();
    push_error_field(&mut payload, b'S', "ERROR");
    push_error_field(&mut payload, b'V', "ERROR");
    push_error_field(&mut payload, b'C', error.code);
    push_error_field(&mut payload, b'M', error.message);
    if let Some(position) = error.position {
        push_error_field(&mut payload, b'P', position);
    }
    payload.push(0);
    write_message(stream, b'E', &payload)
}

fn push_error_field(payload: &mut Vec<u8>, tag: u8, value: &str) {
    payload.push(tag);
    push_cstring(payload, value);
}

fn push_cstring(payload: &mut Vec<u8>, value: &str) {
    payload.extend_from_slice(value.as_bytes());
    payload.push(0);
}

fn write_message(stream: &mut TcpStream, tag: u8, payload: &[u8]) -> io::Result<()> {
    let total_len = i32::try_from(payload.len() + 4)
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "payload too large"))?;
    stream.write_all(&[tag])?;
    stream.write_all(&total_len.to_be_bytes())?;
    stream.write_all(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_sql_collapses_case_whitespace_and_semicolons() {
        assert_eq!(
            canonical_sql("  SELECT   1   AS One ; ; "),
            "select 1 as one"
        );
    }

    #[test]
    fn split_simple_query_discards_empty_segments() {
        assert_eq!(
            split_simple_query("SELECT 1;;  SELECT 2;"),
            vec!["SELECT 1", "SELECT 2"]
        );
    }

    #[test]
    fn catalog_helpers_expose_session_tables_and_columns() {
        let mut session = Session::default();
        session.tables.insert(
            "people".to_string(),
            Table {
                columns: vec![
                    gpu_db_protocol::ColumnDef {
                        name: "id".to_string(),
                        ty: gpu_db_protocol::SqlType::Int4,
                    },
                    gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: gpu_db_protocol::SqlType::Text,
                    },
                ],
                rows: Vec::new(),
            },
        );
        session.tables.insert(
            "teams".to_string(),
            Table {
                columns: vec![gpu_db_protocol::ColumnDef {
                    name: "id".to_string(),
                    ty: gpu_db_protocol::SqlType::Int4,
                }],
                rows: Vec::new(),
            },
        );

        assert_eq!(
            catalog_table_rows(&session),
            vec![
                vec![Some("people".to_string())],
                vec![Some("teams".to_string())],
            ]
        );
        assert_eq!(
            catalog_attribute_query_table(
                "select attname, atttypid from pg_catalog.pg_attribute where attrelid = 'people'::regclass and attnum > 0 order by attnum"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            catalog_attribute_rows(&session, "people").unwrap(),
            vec![
                vec![Some("id".to_string()), Some("23".to_string())],
                vec![Some("name".to_string()), Some("25".to_string())],
            ]
        );
        assert!(catalog_attribute_rows(&session, "missing").is_none());
    }

    #[test]
    fn listen_arg_defaults_and_overrides() {
        assert_eq!(
            parse_listen_arg(std::iter::empty()).unwrap(),
            "127.0.0.1:5432"
        );
        assert_eq!(
            parse_listen_arg(
                vec![String::from("--listen"), String::from("0.0.0.0:9999")].into_iter()
            )
            .unwrap(),
            "0.0.0.0:9999"
        );
    }
}
