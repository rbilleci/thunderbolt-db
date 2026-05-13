use std::collections::{BTreeSet, HashMap};
use std::env;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use gpu_db_protocol::{
    parse_command, parse_frontend_message, parse_startup_packet, Command, FrontendMessage,
    SelectFilterOp, SelectProjection, SqlValue, StartupPacket, SUPPORTED_SQL_TYPES,
};
use gpu_db_protocol::{DescribeTarget, SqlType};

const PUBLIC_NAMESPACE_OID: u32 = 2200;

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

fn bool_column(name: &str) -> Column {
    Column {
        name: name.to_string(),
        oid: 16,
        type_size: 1,
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

fn select_filter_matches(left: &SqlValue, op: SelectFilterOp, right: &SqlValue) -> bool {
    match op {
        SelectFilterOp::Eq => left == right,
        SelectFilterOp::Lt => compare_sql_values(left, right).is_lt(),
        SelectFilterOp::Lte => !compare_sql_values(left, right).is_gt(),
        SelectFilterOp::Gt => compare_sql_values(left, right).is_gt(),
        SelectFilterOp::Gte => !compare_sql_values(left, right).is_lt(),
    }
}

fn row_matches_select_filters(
    table: &Table,
    row: &[SqlValue],
    select: &gpu_db_protocol::Select,
) -> Result<bool, ErrorField> {
    let filter_groups = if select.filter_groups.is_empty() {
        if select.filters.is_empty() {
            select
                .filter
                .iter()
                .cloned()
                .map(|filter| vec![filter])
                .collect::<Vec<_>>()
        } else {
            vec![select.filters.clone()]
        }
    } else {
        select.filter_groups.clone()
    };

    if filter_groups.is_empty() {
        return Ok(true);
    }

    for filters in filter_groups {
        let mut group_matches = true;
        for filter in filters {
            let Some(idx) = table
                .columns
                .iter()
                .position(|column| column.def.name == filter.column)
            else {
                return Err(ErrorField {
                    code: "42703",
                    message: "column does not exist",
                    position: None,
                });
            };
            if !select_filter_matches(&row[idx], filter.op, &filter.value) {
                group_matches = false;
                break;
            }
        }
        if group_matches {
            return Ok(true);
        }
    }

    Ok(false)
}

fn execute_select_result(
    session: &Session,
    select: &gpu_db_protocol::Select,
) -> Result<SelectResult, ErrorField> {
    let Some(table) = session.tables.get(&select.table) else {
        return Err(ErrorField {
            code: "42P01",
            message: "relation does not exist",
            position: None,
        });
    };
    let selected_columns = match &select.projection {
        SelectProjection::All => table.columns.clone(),
        SelectProjection::Columns(columns) => {
            let mut selected = Vec::with_capacity(columns.len());
            for column in columns {
                let Some(def) = table
                    .columns
                    .iter()
                    .find(|candidate| candidate.def.name == *column)
                else {
                    return Err(ErrorField {
                        code: "42703",
                        message: "column does not exist",
                        position: None,
                    });
                };
                selected.push(def.clone());
            }
            selected
        }
    };
    let mut rows = Vec::new();
    for row in &table.rows {
        match row_matches_select_filters(table, row, select) {
            Ok(true) => rows.push(row.clone()),
            Ok(false) => {}
            Err(error) => return Err(error),
        }
    }
    if let Some(order) = &select.order_by {
        let Some(idx) = table
            .columns
            .iter()
            .position(|column| column.def.name == order.column)
        else {
            return Err(ErrorField {
                code: "42703",
                message: "column does not exist",
                position: None,
            });
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
                .position(|column| column.def.name == selected.def.name)
                .expect("selected column came from table")
        })
        .collect::<Vec<_>>();
    let columns = selected_columns
        .iter()
        .map(|column| match column.def.ty {
            gpu_db_protocol::SqlType::Int4 => int4_column(&column.def.name),
            gpu_db_protocol::SqlType::Text => text_column(&column.def.name),
        })
        .collect::<Vec<_>>();
    let rows = rows
        .iter()
        .map(|row| {
            selected_indexes
                .iter()
                .map(|idx| Some(format_sql_value(&row[*idx])))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(SelectResult { columns, rows })
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

struct Session {
    in_transaction: bool,
    prepared: HashMap<String, PreparedStatement>,
    portals: HashMap<String, Portal>,
    cursors: HashMap<String, Cursor>,
    tables: HashMap<String, Table>,
    next_relation_oid: u32,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            in_transaction: false,
            prepared: HashMap::new(),
            portals: HashMap::new(),
            cursors: HashMap::new(),
            tables: HashMap::new(),
            next_relation_oid: FIRST_USER_RELATION_OID,
        }
    }
}

impl Session {
    fn close_extended_target(&mut self, target: DescribeTarget, name: &str) {
        match target {
            DescribeTarget::Statement => {
                self.prepared.remove(name);
                self.portals
                    .retain(|_, portal| portal.statement_name != name);
            }
            DescribeTarget::Portal => {
                self.portals.remove(name);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Table {
    oid: u32,
    name: String,
    columns: Vec<CatalogColumn>,
    rows: Vec<Vec<SqlValue>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CatalogColumn {
    attnum: i16,
    def: gpu_db_protocol::ColumnDef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PreparedStatement {
    AddTen,
    Extended(PreparedQuery),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PreparedQuery {
    query: String,
    parameter_type_oids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Portal {
    statement_name: String,
    query: PreparedQuery,
    parameters: Vec<Option<String>>,
    described: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Cursor {
    columns: Vec<Column>,
    rows: Vec<Vec<Option<String>>>,
    position: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SelectResult {
    columns: Vec<Column>,
    rows: Vec<Vec<Option<String>>>,
}

const FIRST_USER_RELATION_OID: u32 = 16_384;

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

        let message = parse_frontend_message(&frame)
            .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))?;
        match message {
            FrontendMessage::SimpleQuery(query) => {
                run_simple_query(&mut stream, &mut session, &query)?
            }
            FrontendMessage::Parse {
                statement_name,
                query,
                parameter_type_oids,
            } => handle_parse(
                &mut stream,
                &mut session,
                statement_name,
                query,
                parameter_type_oids,
            )?,
            FrontendMessage::Bind {
                portal_name,
                statement_name,
                parameter_format_codes,
                parameters,
                result_format_codes,
            } => handle_bind(
                &mut stream,
                &mut session,
                portal_name,
                statement_name,
                parameter_format_codes,
                parameters,
                result_format_codes,
            )?,
            FrontendMessage::Describe { target, name } => {
                handle_describe(&mut stream, &mut session, target, &name)?
            }
            FrontendMessage::Execute {
                portal_name,
                max_rows,
            } => handle_execute(&mut stream, &mut session, &portal_name, max_rows)?,
            FrontendMessage::Close { target, name } => {
                handle_close(&mut stream, &mut session, target, &name)?
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
        FrontendMessage::FunctionCall { .. }
        | FrontendMessage::CopyData(_)
        | FrontendMessage::CopyDone
        | FrontendMessage::CopyFail(_) => {
            "extended protocol feature is not supported by the compatibility stub"
        }
        FrontendMessage::SimpleQuery(_)
        | FrontendMessage::Bind { .. }
        | FrontendMessage::Parse { .. }
        | FrontendMessage::Describe { .. }
        | FrontendMessage::Close { .. }
        | FrontendMessage::Execute { .. }
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
        execute_statement(stream, session, statement, true)?;
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

fn handle_parse(
    stream: &mut TcpStream,
    session: &mut Session,
    statement_name: String,
    query: String,
    parameter_type_oids: Vec<u32>,
) -> io::Result<()> {
    if parameter_type_oids
        .iter()
        .copied()
        .any(|oid| !matches!(oid, 0 | 23 | 25))
    {
        return write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "only text and int4 extended-query parameters are supported",
                position: None,
            },
        );
    }
    session.prepared.insert(
        statement_name,
        PreparedStatement::Extended(PreparedQuery {
            query,
            parameter_type_oids,
        }),
    );
    write_parse_complete(stream)
}

fn handle_bind(
    stream: &mut TcpStream,
    session: &mut Session,
    portal_name: String,
    statement_name: String,
    parameter_format_codes: Vec<i16>,
    parameters: Vec<Option<Vec<u8>>>,
    result_format_codes: Vec<i16>,
) -> io::Result<()> {
    if parameter_format_codes.iter().any(|code| *code != 0)
        || result_format_codes.iter().any(|code| *code != 0)
    {
        return write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "only text format parameters and results are supported",
                position: None,
            },
        );
    }
    let Some(PreparedStatement::Extended(query)) = session.prepared.get(&statement_name) else {
        return write_error(
            stream,
            &ErrorField {
                code: "26000",
                message: "prepared statement does not exist",
                position: None,
            },
        );
    };
    let mut decoded = Vec::with_capacity(parameters.len());
    for parameter in parameters {
        decoded.push(match parameter {
            Some(bytes) => Some(String::from_utf8(bytes).map_err(|error| {
                io::Error::new(
                    ErrorKind::InvalidData,
                    format!("invalid UTF-8 parameter: {error}"),
                )
            })?),
            None => None,
        });
    }
    session.portals.insert(
        portal_name,
        Portal {
            statement_name,
            query: query.clone(),
            parameters: decoded,
            described: false,
        },
    );
    write_bind_complete(stream)
}

fn handle_describe(
    stream: &mut TcpStream,
    session: &mut Session,
    target: DescribeTarget,
    name: &str,
) -> io::Result<()> {
    match target {
        DescribeTarget::Statement => {
            let Some(PreparedStatement::Extended(query)) = session.prepared.get(name) else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "26000",
                        message: "prepared statement does not exist",
                        position: None,
                    },
                );
            };
            write_parameter_description(stream, &query.parameter_type_oids)?;
            if let Some(columns) = describe_query_columns(session, &query.query) {
                write_row_description(stream, &columns)
            } else {
                write_no_data(stream)
            }
        }
        DescribeTarget::Portal => {
            let Some(portal) = session.portals.get(name) else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "34000",
                        message: "portal does not exist",
                        position: None,
                    },
                );
            };
            let Some(bound_query) = bind_query_parameters(&portal.query, &portal.parameters) else {
                return write_error(
                    stream,
                    &ErrorField {
                        code: "08P01",
                        message: "bound parameter count does not match prepared statement",
                        position: None,
                    },
                );
            };
            if let Some(columns) = describe_query_columns(session, &bound_query) {
                write_row_description(stream, &columns)?;
                if let Some(portal) = session.portals.get_mut(name) {
                    portal.described = true;
                }
                Ok(())
            } else {
                write_no_data(stream)
            }
        }
    }
}

fn handle_execute(
    stream: &mut TcpStream,
    session: &mut Session,
    portal_name: &str,
    max_rows: u32,
) -> io::Result<()> {
    if max_rows != 0 {
        return write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "limited portal execution is not supported",
                position: None,
            },
        );
    }
    let Some(portal) = session.portals.get(portal_name).cloned() else {
        return write_error(
            stream,
            &ErrorField {
                code: "34000",
                message: "portal does not exist",
                position: None,
            },
        );
    };
    let include_row_description = !portal.described;
    let Some(bound_query) = bind_query_parameters(&portal.query, &portal.parameters) else {
        return write_error(
            stream,
            &ErrorField {
                code: "08P01",
                message: "bound parameter count does not match prepared statement",
                position: None,
            },
        );
    };
    execute_statement(stream, session, &bound_query, include_row_description)
}

fn handle_close(
    stream: &mut TcpStream,
    session: &mut Session,
    target: DescribeTarget,
    name: &str,
) -> io::Result<()> {
    session.close_extended_target(target, name);
    write_close_complete(stream)
}

fn parse_declare_cursor(statement: &str) -> Option<(String, String)> {
    let canonical = canonical_sql(statement.trim().trim_end_matches(';').trim());
    let rest = canonical.strip_prefix("declare ")?;
    for marker in [" no scroll cursor for ", " cursor for "] {
        if let Some(idx) = rest.find(marker) {
            let name = rest[..idx].trim();
            let query_start = "declare ".len() + idx + marker.len();
            let query = canonical[query_start..].trim();
            if !name.is_empty() && !query.is_empty() {
                return Some((name.to_string(), query.to_string()));
            }
        }
    }
    None
}

fn parse_fetch_forward(statement: &str) -> Option<(String, usize)> {
    let trimmed = statement.trim().trim_end_matches(';').trim();
    let lower = trimmed.to_ascii_lowercase();
    let rest = lower.strip_prefix("fetch forward ")?;
    let from_idx = rest.find(" from ")?;
    let count = rest[..from_idx].trim().parse::<usize>().ok()?;
    let name_start = "fetch forward ".len() + from_idx + " from ".len();
    let name = trimmed[name_start..].trim();
    if name.is_empty() {
        None
    } else {
        Some((name.to_string(), count))
    }
}

fn parse_close_cursor(statement: &str) -> Option<String> {
    let trimmed = statement.trim().trim_end_matches(';').trim();
    let lower = trimmed.to_ascii_lowercase();
    let name = lower.strip_prefix("close ")?;
    if name == "all" {
        return None;
    }
    let original = &trimmed["close ".len()..];
    if original.trim().is_empty() {
        None
    } else {
        Some(original.trim().to_string())
    }
}

fn execute_declare_cursor(
    stream: &mut TcpStream,
    session: &mut Session,
    name: String,
    query: &str,
) -> io::Result<()> {
    let Ok(Command::Select(select)) = parse_command(query) else {
        return write_error(
            stream,
            &ErrorField {
                code: "0A000",
                message: "cursor declarations only support relational SELECT",
                position: None,
            },
        );
    };
    let result = match execute_select_result(session, &select) {
        Ok(result) => result,
        Err(error) => return write_error(stream, &error),
    };
    session.cursors.insert(
        name,
        Cursor {
            columns: result.columns,
            rows: result.rows,
            position: 0,
        },
    );
    write_command_complete(stream, "DECLARE CURSOR")
}

fn execute_fetch_forward(
    stream: &mut TcpStream,
    session: &mut Session,
    name: &str,
    count: usize,
) -> io::Result<()> {
    let Some(cursor) = session.cursors.get_mut(name) else {
        return write_error(
            stream,
            &ErrorField {
                code: "34000",
                message: "cursor does not exist",
                position: None,
            },
        );
    };
    let start = cursor.position;
    let end = start.saturating_add(count).min(cursor.rows.len());
    cursor.position = end;
    write_rows_with_tag(
        stream,
        &cursor.columns,
        &cursor.rows[start..end],
        true,
        &format!("FETCH {}", end - start),
    )
}

fn execute_statement(
    stream: &mut TcpStream,
    session: &mut Session,
    statement: &str,
    include_row_description: bool,
) -> io::Result<()> {
    if let Some((name, query)) = parse_declare_cursor(statement) {
        return execute_declare_cursor(stream, session, name, &query);
    }
    if let Some((name, count)) = parse_fetch_forward(statement) {
        return execute_fetch_forward(stream, session, &name, count);
    }
    if let Some(name) = parse_close_cursor(statement) {
        session.cursors.remove(&name);
        return write_command_complete(stream, "CLOSE CURSOR");
    }

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
                let oid = session.next_relation_oid;
                session.next_relation_oid = match session.next_relation_oid.checked_add(1) {
                    Some(next) => next,
                    None => {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "54000",
                                message: "relation OID allocation exhausted",
                                position: None,
                            },
                        );
                    }
                };
                let mut columns = Vec::with_capacity(create.columns.len());
                for (idx, def) in create.columns.into_iter().enumerate() {
                    let Ok(attnum) = i16::try_from(idx + 1) else {
                        return write_error(
                            stream,
                            &ErrorField {
                                code: "54000",
                                message: "too many columns for bootstrap catalog",
                                position: None,
                            },
                        );
                    };
                    columns.push(CatalogColumn { attnum, def });
                }
                let name = create.table;
                session.tables.insert(
                    name.clone(),
                    Table {
                        oid,
                        name,
                        columns,
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
                        .position(|candidate| candidate.def.name == *column)
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
                        if !sql_value_matches_type(
                            &row[source_idx],
                            table.columns[target_idx].def.ty,
                        ) {
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
                let result = match execute_select_result(session, &select) {
                    Ok(result) => result,
                    Err(error) => return write_error(stream, &error),
                };
                return write_select_rows(
                    stream,
                    &result.columns,
                    &result.rows,
                    include_row_description,
                );
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
    if let Some(rows) = psql_describe_query_type_rows(&canonical) {
        return write_single_row(stream, &[text_column("Column"), text_column("Type")], &rows);
    }
    if canonical
        == "select relname from pg_catalog.pg_class where relnamespace = 'public'::regnamespace and relkind = 'r' order by relname"
    {
        return write_single_row(
            stream,
            &[text_column("relname")],
            &catalog_table_name_rows(session),
        );
    }
    if canonical
        == "select oid, relname from pg_catalog.pg_class where relnamespace = 'public'::regnamespace and relkind = 'r' order by oid"
    {
        return write_single_row(
            stream,
            &[int4_column("oid"), text_column("relname")],
            &catalog_table_oid_rows(session),
        );
    }
    if canonical == psql_describe_tables_catalog_query()
        || canonical == psql_describe_relations_catalog_query()
    {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            &catalog_psql_describe_table_rows(session),
        );
    }
    if let Some(filter) = psql_describe_tables_catalog_query_filter(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            &catalog_psql_describe_table_rows_filtered(session, &filter),
        );
    }
    if canonical == psql_describe_tables_verbose_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Access method"),
                text_column("Size"),
                text_column("Description"),
            ],
            &catalog_psql_describe_table_verbose_rows(session),
        );
    }
    if let Some(filter) = psql_describe_tables_verbose_catalog_query_filter(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Access method"),
                text_column("Size"),
                text_column("Description"),
            ],
            &catalog_psql_describe_table_verbose_rows_filtered(session, &filter),
        );
    }
    if let Some(filter) = psql_describe_table_privileges_catalog_query_filter(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Access privileges"),
                text_column("Column privileges"),
                text_column("Policies"),
            ],
            &catalog_psql_describe_table_privilege_rows_filtered(session, &filter),
        );
    }
    if canonical == psql_describe_indexes_catalog_query()
        || psql_describe_indexes_catalog_query_schema_filter(&canonical).is_some()
    {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Table"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_describe_views_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_describe_views_verbose_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Size"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_describe_materialized_views_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_describe_materialized_views_verbose_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Access method"),
                text_column("Size"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_describe_sequences_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_describe_sequences_verbose_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Owner"),
                text_column("Persistence"),
                text_column("Size"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_describe_functions_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Result data type"),
                text_column("Argument data types"),
                text_column("Type"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_aggregates_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Result data type"),
                text_column("Argument data types"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_conversions_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Source"),
                text_column("Destination"),
                text_column("Default?"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_operators_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Left arg type"),
                text_column("Right arg type"),
                text_column("Result type"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_collations_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Provider"),
                text_column("Collate"),
                text_column("Ctype"),
                text_column("ICU Locale"),
                text_column("ICU Rules"),
                text_column("Deterministic?"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_casts_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Source type"),
                text_column("Target type"),
                text_column("Function"),
                text_column("Implicit?"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_publications_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                bool_column("All tables"),
                bool_column("Inserts"),
                bool_column("Updates"),
                bool_column("Deletes"),
                bool_column("Truncates"),
                bool_column("Via root"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_publications_verbose_catalog_query() {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("pubname"),
                text_column("owner"),
                bool_column("puballtables"),
                bool_column("pubinsert"),
                bool_column("pubupdate"),
                bool_column("pubdelete"),
                bool_column("pubtruncate"),
                bool_column("pubviaroot"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_subscriptions_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                bool_column("Enabled"),
                text_column("Publication"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_default_access_privileges_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Owner"),
                text_column("Schema"),
                text_column("Type"),
                text_column("Access privileges"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_extensions_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Version"),
                text_column("Schema"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_languages_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                bool_column("Trusted"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_domains_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Collation"),
                text_column("Nullable"),
                text_column("Default"),
                text_column("Check"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_list_domains_verbose_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Type"),
                text_column("Collation"),
                text_column("Nullable"),
                text_column("Default"),
                text_column("Check"),
                text_column("Access privileges"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical == psql_describe_roles_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("rolname"),
                bool_column("rolsuper"),
                bool_column("rolinherit"),
                bool_column("rolcreaterole"),
                bool_column("rolcreatedb"),
                bool_column("rolcanlogin"),
                int4_column("rolconnlimit"),
                text_column("rolvaliduntil"),
                bool_column("rolreplication"),
                bool_column("rolbypassrls"),
            ],
            &catalog_psql_describe_role_rows(),
        );
    }
    if canonical == psql_list_databases_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                text_column("Encoding"),
                text_column("Locale Provider"),
                text_column("Collate"),
                text_column("Ctype"),
                text_column("ICU Locale"),
                text_column("ICU Rules"),
                text_column("Access privileges"),
            ],
            &catalog_psql_list_database_rows(),
        );
    }
    if canonical == psql_list_tablespaces_catalog_query() {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                text_column("Location"),
            ],
            &catalog_psql_list_tablespace_rows(),
        );
    }
    if canonical == psql_list_access_methods_catalog_query() {
        return write_single_row(
            stream,
            &[text_column("Name"), text_column("Type")],
            &catalog_psql_list_access_method_rows(),
        );
    }
    if canonical == psql_describe_schemas_catalog_query() {
        return write_single_row(
            stream,
            &[text_column("Name"), text_column("Owner")],
            &catalog_psql_describe_schema_rows(),
        );
    }
    if psql_describe_schemas_verbose_catalog_query_public_filter(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Name"),
                text_column("Owner"),
                text_column("Access privileges"),
                text_column("Description"),
            ],
            &catalog_psql_describe_schema_verbose_rows(),
        );
    }
    if canonical == psql_describe_schema_publications_query() {
        return write_single_row(stream, &[text_column("pubname")], &catalog_empty_rows());
    }
    if canonical == pg_catalog_namespace_query() {
        return write_single_row(
            stream,
            &[int4_column("oid"), text_column("nspname")],
            &pg_catalog_namespace_rows(),
        );
    }
    if let Some(type_name) = psql_describe_type_catalog_query_type(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Description"),
            ],
            &catalog_psql_describe_type_rows(&type_name),
        );
    }
    if canonical == psql_describe_pg_catalog_types_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Description"),
            ],
            &catalog_psql_describe_type_rows_for_supported_types(),
        );
    }
    if canonical == psql_describe_pg_catalog_types_verbose_query() {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Internal name"),
                text_column("Size"),
                text_column("Elements"),
                text_column("Owner"),
                text_column("Access privileges"),
                text_column("Description"),
            ],
            &catalog_psql_describe_type_verbose_rows_for_supported_types(),
        );
    }
    if catalog_describe_relation_lookup_query_public_namespace(&canonical) {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
            ],
            &catalog_describe_relation_lookup_rows_for_public_namespace(session),
        );
    }
    if let Some(table) = catalog_describe_relation_lookup_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
            ],
            &catalog_describe_relation_lookup_rows(session, &table),
        );
    }
    if let Some(oid) = catalog_describe_relation_flags_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                int4_column("relchecks"),
                text_column("relkind"),
                text_column("relhasindex"),
                text_column("relhasrules"),
                text_column("relhastriggers"),
                text_column("relrowsecurity"),
                text_column("relforcerowsecurity"),
                text_column("relhasoids"),
                text_column("relispartition"),
                text_column("?column?"),
                int4_column("reltablespace"),
                text_column("case"),
                text_column("relpersistence"),
                text_column("relreplident"),
                text_column("amname"),
            ],
            &catalog_describe_relation_flags_rows(session, oid),
        );
    }
    if let Some(oid) = catalog_describe_verbose_attribute_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("attname"),
                text_column("format_type"),
                text_column("pg_get_expr"),
                text_column("attnotnull"),
                text_column("attcollation"),
                text_column("attidentity"),
                text_column("attgenerated"),
                text_column("attstorage"),
                text_column("attcompression"),
                int4_column("attstattarget"),
                text_column("col_description"),
            ],
            &catalog_describe_verbose_attribute_rows(session, oid),
        );
    }
    if let Some(oid) = catalog_describe_attribute_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("attname"),
                text_column("format_type"),
                text_column("pg_get_expr"),
                text_column("attnotnull"),
                text_column("attcollation"),
                text_column("attidentity"),
                text_column("attgenerated"),
            ],
            &catalog_describe_attribute_rows(session, oid),
        );
    }
    if let Some(oid) = catalog_describe_policy_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("polname"),
                text_column("polpermissive"),
                text_column("array_to_string"),
                text_column("pg_get_expr"),
                text_column("pg_get_expr"),
                text_column("cmd"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        );
    }
    if let Some(oid) = catalog_describe_statistic_ext_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("stxrelid"),
                text_column("nsp"),
                text_column("stxname"),
                text_column("columns"),
                text_column("ndist_enabled"),
                text_column("deps_enabled"),
                text_column("mcv_enabled"),
                int4_column("stxstattarget"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        );
    }
    if let Some(oid) = catalog_describe_publication_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("pubname"),
                text_column("?column?"),
                text_column("?column?"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        );
    }
    if let Some(oid) = catalog_describe_inherits_parent_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[text_column("oid")],
            &catalog_empty_rows_for_relation_oid(oid),
        );
    }
    if let Some(oid) = catalog_describe_inherits_child_query_oid(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("oid"),
                text_column("relkind"),
                text_column("inhdetachpending"),
                text_column("pg_get_expr"),
            ],
            &catalog_empty_rows_for_relation_oid(oid),
        );
    }
    if canonical == pg_catalog_tables_query() {
        return write_single_row(
            stream,
            &[
                text_column("schemaname"),
                text_column("tablename"),
                text_column("tableowner"),
            ],
            &pg_catalog_table_rows(session),
        );
    }
    if canonical == pg_catalog_indexes_query() {
        return write_single_row(
            stream,
            &[
                text_column("schemaname"),
                text_column("tablename"),
                text_column("indexname"),
                text_column("indexdef"),
            ],
            &pg_catalog_index_rows(session),
        );
    }
    if canonical == pg_catalog_class_plain_tables_query() {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
                text_column("relkind"),
                text_column("relpersistence"),
            ],
            &pg_catalog_class_plain_table_rows(session),
        );
    }
    if let Some(tables) = pg_catalog_class_plain_tables_in_query_tables(&canonical) {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("nspname"),
                text_column("relname"),
                text_column("relkind"),
                text_column("relpersistence"),
            ],
            &pg_catalog_class_plain_table_rows_for_tables(session, &tables),
        );
    }
    if canonical == information_schema_tables_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("table_type"),
            ],
            &information_schema_table_rows(session),
        );
    }
    if canonical == information_schema_base_table_discovery_query() {
        return write_single_row(
            stream,
            &[text_column("table_schema"), text_column("table_name")],
            &information_schema_base_table_discovery_rows(session),
        );
    }
    if let Some(tables) = information_schema_tables_in_query_tables(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("table_type"),
            ],
            &information_schema_table_rows_for_tables(session, &tables),
        );
    }
    if canonical == information_schema_rich_tables_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("table_type"),
                text_column("self_referencing_column_name"),
                text_column("reference_generation"),
                text_column("user_defined_type_catalog"),
                text_column("user_defined_type_schema"),
                text_column("user_defined_type_name"),
                text_column("is_insertable_into"),
                text_column("is_typed"),
                text_column("commit_action"),
            ],
            &information_schema_rich_table_rows(session),
        );
    }
    if let Some(table) = information_schema_rich_tables_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("table_type"),
                text_column("self_referencing_column_name"),
                text_column("reference_generation"),
                text_column("user_defined_type_catalog"),
                text_column("user_defined_type_schema"),
                text_column("user_defined_type_name"),
                text_column("is_insertable_into"),
                text_column("is_typed"),
                text_column("commit_action"),
            ],
            &information_schema_rich_table_rows_for_table(session, &table),
        );
    }
    if let Some(table) = information_schema_rich_tables_catalog_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("table_type"),
                text_column("self_referencing_column_name"),
                text_column("reference_generation"),
                text_column("user_defined_type_catalog"),
                text_column("user_defined_type_schema"),
                text_column("user_defined_type_name"),
                text_column("is_insertable_into"),
                text_column("is_typed"),
                text_column("commit_action"),
            ],
            &information_schema_rich_table_rows_for_table(session, &table),
        );
    }
    if let Some(table) = information_schema_columns_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_column_rows(session, &table),
        );
    }
    if canonical == information_schema_all_columns_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_all_column_rows(session),
        );
    }
    if let Some(tables) = information_schema_columns_in_query_tables(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("data_type"),
            ],
            &information_schema_column_rows_for_tables(session, &tables),
        );
    }
    if let Some(table) = information_schema_column_details_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("column_name"),
                text_column("data_type"),
                text_column("is_nullable"),
                text_column("column_default"),
            ],
            &information_schema_column_detail_rows(session, &table),
        );
    }
    if canonical == information_schema_rich_columns_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_rich_column_rows(session),
        );
    }
    if canonical == information_schema_extended_columns_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows(session),
        );
    }
    if let Some(table) = information_schema_extended_columns_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows_for_table(session, &table),
        );
    }
    if let Some(table) = information_schema_extended_columns_catalog_query_table(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows_for_table(session, &table),
        );
    }
    if let Some(tables) = information_schema_extended_columns_in_query_tables(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                int4_column("ordinal_position"),
                text_column("column_default"),
                text_column("is_nullable"),
                text_column("data_type"),
                int4_column("character_maximum_length"),
                int4_column("numeric_precision"),
                int4_column("numeric_precision_radix"),
                int4_column("numeric_scale"),
                text_column("udt_schema"),
                text_column("udt_name"),
            ],
            &information_schema_extended_column_rows_for_tables(session, &tables),
        );
    }
    if canonical == information_schema_schemata_query() {
        return write_single_row(
            stream,
            &[text_column("schema_name"), text_column("schema_owner")],
            &information_schema_schemata_rows(),
        );
    }
    if canonical == information_schema_table_constraints_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("constraint_name"),
                text_column("constraint_type"),
            ],
            &information_schema_table_constraint_rows(session),
        );
    }
    if canonical == information_schema_key_column_usage_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_schema"),
                text_column("table_name"),
                text_column("column_name"),
                text_column("constraint_name"),
                int4_column("ordinal_position"),
            ],
            &information_schema_key_column_usage_rows(session),
        );
    }
    if canonical == information_schema_views_query() {
        return write_single_row(
            stream,
            &[
                text_column("table_catalog"),
                text_column("table_schema"),
                text_column("table_name"),
                text_column("view_definition"),
                text_column("check_option"),
                text_column("is_updatable"),
                text_column("is_insertable_into"),
                text_column("is_trigger_updatable"),
                text_column("is_trigger_deletable"),
                text_column("is_trigger_insertable"),
            ],
            &information_schema_view_rows(session),
        );
    }
    if canonical == pg_catalog_views_query() {
        return write_single_row(
            stream,
            &[
                text_column("schemaname"),
                text_column("viewname"),
                text_column("viewowner"),
                text_column("definition"),
            ],
            &pg_catalog_view_rows(session),
        );
    }
    if canonical == pg_catalog_constraints_query() {
        return write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("conname"),
                text_column("contype"),
            ],
            &pg_catalog_constraint_rows(session),
        );
    }
    if canonical == pg_catalog_attrdefs_query() {
        return write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("attname"),
                text_column("default_expr"),
            ],
            &pg_catalog_attrdef_rows(session),
        );
    }
    if canonical == pg_catalog_descriptions_query() {
        return write_single_row(
            stream,
            &[
                text_column("nspname"),
                text_column("relname"),
                text_column("attname"),
                text_column("description"),
            ],
            &pg_catalog_description_rows(session),
        );
    }
    if psql_list_object_descriptions_query(&canonical) {
        return write_single_row(
            stream,
            &[
                text_column("Schema"),
                text_column("Name"),
                text_column("Object"),
                text_column("Description"),
            ],
            &catalog_empty_rows(),
        );
    }
    if canonical
        == "select oid, typname, typlen from pg_catalog.pg_type where oid in (23, 25) order by oid"
    {
        return write_single_row(
            stream,
            &[
                int4_column("oid"),
                text_column("typname"),
                int4_column("typlen"),
            ],
            &catalog_type_rows_by_oid(),
        );
    }
    if canonical
        == "select typname, oid, typlen from pg_catalog.pg_type where typname in ('int4', 'text') order by typname"
    {
        return write_single_row(
            stream,
            &[
                text_column("typname"),
                int4_column("oid"),
                int4_column("typlen"),
            ],
            &catalog_type_rows_by_name(),
        );
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
    if let Some(table) = catalog_attribute_detail_query_table(&canonical) {
        let Some(rows) = catalog_attribute_detail_rows(session, &table) else {
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
            &[
                int4_column("attnum"),
                text_column("attname"),
                int4_column("atttypid"),
                int4_column("attlen"),
            ],
            &rows,
        );
    }
    if let Some(table) = pg_catalog_class_attribute_type_query_table(&canonical) {
        let Some(rows) = pg_catalog_class_attribute_type_rows(session, &table) else {
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
            &[
                int4_column("attnum"),
                text_column("attname"),
                text_column("data_type"),
                text_column("attnotnull"),
            ],
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

fn catalog_table_name_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| vec![Some(table.name.clone())])
        .collect::<Vec<_>>()
}

fn catalog_table_oid_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut rows = session
        .tables
        .iter()
        .map(|(name, table)| (table.oid, name))
        .collect::<Vec<_>>();
    rows.sort_by_key(|(oid, _)| *oid);
    rows.into_iter()
        .map(|(oid, name)| vec![Some(oid.to_string()), Some(name.clone())])
        .collect()
}

fn psql_describe_tables_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_relations_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','v','m','s','f','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_tables_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PsqlDescribeTablesFilter {
    namespace: String,
    relname_pattern: Option<String>,
}

fn psql_describe_tables_catalog_query_filter(canonical: &str) -> Option<PsqlDescribeTablesFilter> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and ";
    let namespace_prefix = "n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 1,2";
    let rest = canonical.strip_prefix(prefix)?;
    if let Some(namespace) = rest
        .strip_prefix(namespace_prefix)
        .and_then(|rest| rest.strip_suffix(namespace_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: namespace.to_string(),
            relname_pattern: None,
        });
    }

    let relname_prefix = "c.relname operator(pg_catalog.~) '^(";
    let relname_visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1,2";
    let relname_middle = ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    if let Some(relname_pattern) = rest
        .strip_prefix(relname_prefix)
        .and_then(|rest| rest.strip_suffix(relname_visible_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some(relname_pattern.to_string()),
        });
    }
    let (relname_pattern, namespace) = rest
        .strip_prefix(relname_prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(relname_middle)?;
    Some(PsqlDescribeTablesFilter {
        namespace: namespace.to_string(),
        relname_pattern: Some(relname_pattern.to_string()),
    })
}

fn psql_describe_tables_verbose_catalog_query_filter(
    canonical: &str,
) -> Option<PsqlDescribeTablesFilter> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and ";
    let namespace_prefix = "n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 1,2";
    let relname_prefix = "c.relname operator(pg_catalog.~) '^(";
    let relname_visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1,2";
    let relname_middle = ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    let rest = canonical.strip_prefix(prefix)?;

    if let Some(namespace) = rest
        .strip_prefix(namespace_prefix)
        .and_then(|rest| rest.strip_suffix(namespace_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: namespace.to_string(),
            relname_pattern: None,
        });
    }

    if let Some(relname_pattern) = rest
        .strip_prefix(relname_prefix)
        .and_then(|rest| rest.strip_suffix(relname_visible_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some(relname_pattern.to_string()),
        });
    }

    let (relname_pattern, namespace) = rest
        .strip_prefix(relname_prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(relname_middle)?;
    Some(PsqlDescribeTablesFilter {
        namespace: namespace.to_string(),
        relname_pattern: Some(relname_pattern.to_string()),
    })
}

fn psql_describe_table_privileges_catalog_query_filter(
    canonical: &str,
) -> Option<PsqlDescribeTablesFilter> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 's' then 'sequence' when 'f' then 'foreign table' when 'p' then 'partitioned table' end as \"type\", pg_catalog.array_to_string(c.relacl, e'\\n') as \"access privileges\", pg_catalog.array_to_string(array( select attname || e':\\n ' || pg_catalog.array_to_string(attacl, e'\\n ') from pg_catalog.pg_attribute a where attrelid = c.oid and not attisdropped and attacl is not null ), e'\\n') as \"column privileges\", pg_catalog.array_to_string(array( select polname || case when not polpermissive then e' (restrictive)' else '' end || case when polcmd != '*' then e' (' || polcmd::pg_catalog.text || e'):' else e':' end || case when polqual is not null then e'\\n (u): ' || pg_catalog.pg_get_expr(polqual, polrelid) else e'' end || case when polwithcheck is not null then e'\\n (c): ' || pg_catalog.pg_get_expr(polwithcheck, polrelid) else e'' end || case when polroles <> '{0}' then e'\\n to: ' || pg_catalog.array_to_string( array( select rolname from pg_catalog.pg_roles where oid = any (polroles) order by 1 ), e', ') else e'' end from pg_catalog.pg_policy pol where polrelid = c.oid), e'\\n') as \"policies\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('r','v','m','s','f','p') and ";
    let relname_prefix = "c.relname operator(pg_catalog.~) '^(";
    let relname_visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1, 2";
    let relname_middle = ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 1, 2";
    let rest = canonical.strip_prefix(prefix)?;

    if let Some(relname_pattern) = rest
        .strip_prefix(relname_prefix)
        .and_then(|rest| rest.strip_suffix(relname_visible_suffix))
    {
        return Some(PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: Some(relname_pattern.to_string()),
        });
    }

    let (relname_pattern, namespace) = rest
        .strip_prefix(relname_prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(relname_middle)?;
    Some(PsqlDescribeTablesFilter {
        namespace: namespace.to_string(),
        relname_pattern: Some(relname_pattern.to_string()),
    })
}

fn psql_describe_indexes_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_indexes_catalog_query_schema_filter(canonical: &str) -> Option<String> {
    let prefix = "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','s','') and n.nspname operator(pg_catalog.~) '^(";
    let suffix = ")$' collate pg_catalog.default order by 1,2";
    let namespace = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    (namespace == "public").then(|| namespace.to_string())
}

fn psql_describe_views_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('v','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_views_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('v','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_materialized_views_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('m','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_materialized_views_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('m','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_sequences_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('s','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_sequences_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('s','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
}

fn psql_describe_functions_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", p.proname as \"name\", pg_catalog.pg_get_function_result(p.oid) as \"result data type\", pg_catalog.pg_get_function_arguments(p.oid) as \"argument data types\", case p.prokind when 'a' then 'agg' when 'w' then 'window' when 'p' then 'proc' else 'func' end as \"type\" from pg_catalog.pg_proc p left join pg_catalog.pg_namespace n on n.oid = p.pronamespace where pg_catalog.pg_function_is_visible(p.oid) and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' order by 1, 2, 4"
}

fn psql_list_aggregates_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", p.proname as \"name\", pg_catalog.format_type(p.prorettype, null) as \"result data type\", case when p.pronargs = 0 then cast('*' as pg_catalog.text) else pg_catalog.pg_get_function_arguments(p.oid) end as \"argument data types\", pg_catalog.obj_description(p.oid, 'pg_proc') as \"description\" from pg_catalog.pg_proc p left join pg_catalog.pg_namespace n on n.oid = p.pronamespace where p.prokind = 'a' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_function_is_visible(p.oid) order by 1, 2, 4"
}

fn psql_list_conversions_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.conname as \"name\", pg_catalog.pg_encoding_to_char(c.conforencoding) as \"source\", pg_catalog.pg_encoding_to_char(c.contoencoding) as \"destination\", case when c.condefault then 'yes' else 'no' end as \"default?\" from pg_catalog.pg_conversion c join pg_catalog.pg_namespace n on n.oid = c.connamespace where true and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_conversion_is_visible(c.oid) order by 1, 2"
}

fn psql_list_operators_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", o.oprname as \"name\", case when o.oprkind='l' then null else pg_catalog.format_type(o.oprleft, null) end as \"left arg type\", case when o.oprkind='r' then null else pg_catalog.format_type(o.oprright, null) end as \"right arg type\", pg_catalog.format_type(o.oprresult, null) as \"result type\", coalesce(pg_catalog.obj_description(o.oid, 'pg_operator'), pg_catalog.obj_description(o.oprcode, 'pg_proc')) as \"description\" from pg_catalog.pg_operator o left join pg_catalog.pg_namespace n on n.oid = o.oprnamespace where n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_operator_is_visible(o.oid) order by 1, 2, 3, 4"
}

fn psql_list_collations_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", c.collname as \"name\", case c.collprovider when 'd' then 'default' when 'c' then 'libc' when 'i' then 'icu' end as \"provider\", c.collcollate as \"collate\", c.collctype as \"ctype\", c.colliculocale as \"icu locale\", c.collicurules as \"icu rules\", case when c.collisdeterministic then 'yes' else 'no' end as \"deterministic?\" from pg_catalog.pg_collation c, pg_catalog.pg_namespace n where n.oid = c.collnamespace and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and c.collencoding in (-1, pg_catalog.pg_char_to_encoding(pg_catalog.getdatabaseencoding())) and pg_catalog.pg_collation_is_visible(c.oid) order by 1, 2"
}

fn psql_list_casts_catalog_query() -> &'static str {
    "select pg_catalog.format_type(castsource, null) as \"source type\", pg_catalog.format_type(casttarget, null) as \"target type\", case when c.castmethod = 'b' then '(binary coercible)' when c.castmethod = 'i' then '(with inout)' else p.proname end as \"function\", case when c.castcontext = 'e' then 'no' when c.castcontext = 'a' then 'in assignment' else 'yes' end as \"implicit?\" from pg_catalog.pg_cast c left join pg_catalog.pg_proc p on c.castfunc = p.oid left join pg_catalog.pg_type ts on c.castsource = ts.oid left join pg_catalog.pg_namespace ns on ns.oid = ts.typnamespace left join pg_catalog.pg_type tt on c.casttarget = tt.oid left join pg_catalog.pg_namespace nt on nt.oid = tt.typnamespace where ( (true and pg_catalog.pg_type_is_visible(ts.oid) ) or (true and pg_catalog.pg_type_is_visible(tt.oid) ) ) order by 1, 2"
}

fn psql_list_publications_catalog_query() -> &'static str {
    "select pubname as \"name\", pg_catalog.pg_get_userbyid(pubowner) as \"owner\", puballtables as \"all tables\", pubinsert as \"inserts\", pubupdate as \"updates\", pubdelete as \"deletes\", pubtruncate as \"truncates\", pubviaroot as \"via root\" from pg_catalog.pg_publication order by 1"
}

fn psql_list_publications_verbose_catalog_query() -> &'static str {
    "select oid, pubname, pg_catalog.pg_get_userbyid(pubowner) as owner, puballtables, pubinsert, pubupdate, pubdelete, pubtruncate, pubviaroot from pg_catalog.pg_publication order by 2"
}

fn psql_list_subscriptions_catalog_query() -> &'static str {
    "select subname as \"name\" , pg_catalog.pg_get_userbyid(subowner) as \"owner\" , subenabled as \"enabled\" , subpublications as \"publication\" from pg_catalog.pg_subscription where subdbid = (select oid from pg_catalog.pg_database where datname = pg_catalog.current_database())order by 1"
}

fn psql_list_default_access_privileges_catalog_query() -> &'static str {
    "select pg_catalog.pg_get_userbyid(d.defaclrole) as \"owner\", n.nspname as \"schema\", case d.defaclobjtype when 'r' then 'table' when 's' then 'sequence' when 'f' then 'function' when 't' then 'type' when 'n' then 'schema' end as \"type\", pg_catalog.array_to_string(d.defaclacl, e'\\n') as \"access privileges\" from pg_catalog.pg_default_acl d left join pg_catalog.pg_namespace n on n.oid = d.defaclnamespace order by 1, 2, 3"
}

fn psql_list_extensions_catalog_query() -> &'static str {
    "select e.extname as \"name\", e.extversion as \"version\", n.nspname as \"schema\", c.description as \"description\" from pg_catalog.pg_extension e left join pg_catalog.pg_namespace n on n.oid = e.extnamespace left join pg_catalog.pg_description c on c.objoid = e.oid and c.classoid = 'pg_catalog.pg_extension'::pg_catalog.regclass order by 1"
}

fn psql_list_languages_catalog_query() -> &'static str {
    "select l.lanname as \"name\", pg_catalog.pg_get_userbyid(l.lanowner) as \"owner\", l.lanpltrusted as \"trusted\", d.description as \"description\" from pg_catalog.pg_language l left join pg_catalog.pg_description d on d.classoid = l.tableoid and d.objoid = l.oid and d.objsubid = 0 where l.lanplcallfoid != 0 order by 1"
}

fn psql_list_domains_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", t.typname as \"name\", pg_catalog.format_type(t.typbasetype, t.typtypmod) as \"type\", (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type bt where c.oid = t.typcollation and bt.oid = t.typbasetype and t.typcollation <> bt.typcollation) as \"collation\", case when t.typnotnull then 'not null' end as \"nullable\", t.typdefault as \"default\", pg_catalog.array_to_string(array( select pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where t.oid = r.contypid ), ' ') as \"check\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where t.typtype = 'd' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_type_is_visible(t.oid) order by 1, 2"
}

fn psql_list_domains_verbose_catalog_query() -> &'static str {
    "select n.nspname as \"schema\", t.typname as \"name\", pg_catalog.format_type(t.typbasetype, t.typtypmod) as \"type\", (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type bt where c.oid = t.typcollation and bt.oid = t.typbasetype and t.typcollation <> bt.typcollation) as \"collation\", case when t.typnotnull then 'not null' end as \"nullable\", t.typdefault as \"default\", pg_catalog.array_to_string(array( select pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where t.oid = r.contypid ), ' ') as \"check\", pg_catalog.array_to_string(t.typacl, e'\\n') as \"access privileges\", d.description as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace left join pg_catalog.pg_description d on d.classoid = t.tableoid and d.objoid = t.oid and d.objsubid = 0 where t.typtype = 'd' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_type_is_visible(t.oid) order by 1, 2"
}

fn psql_describe_roles_catalog_query() -> &'static str {
    "select r.rolname, r.rolsuper, r.rolinherit, r.rolcreaterole, r.rolcreatedb, r.rolcanlogin, r.rolconnlimit, r.rolvaliduntil , r.rolreplication , r.rolbypassrls from pg_catalog.pg_roles r where r.rolname !~ '^pg_' order by 1"
}

fn catalog_psql_describe_role_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some("postgres".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("t".to_string()),
        Some("-1".to_string()),
        None,
        Some("t".to_string()),
        Some("t".to_string()),
    ]]
}

fn psql_list_databases_catalog_query() -> &'static str {
    "select d.datname as \"name\", pg_catalog.pg_get_userbyid(d.datdba) as \"owner\", pg_catalog.pg_encoding_to_char(d.encoding) as \"encoding\", case d.datlocprovider when 'c' then 'libc' when 'i' then 'icu' end as \"locale provider\", d.datcollate as \"collate\", d.datctype as \"ctype\", d.daticulocale as \"icu locale\", d.daticurules as \"icu rules\", pg_catalog.array_to_string(d.datacl, e'\\n') as \"access privileges\" from pg_catalog.pg_database d order by 1"
}

fn catalog_psql_list_database_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some("postgres".to_string()),
        Some("postgres".to_string()),
        Some("UTF8".to_string()),
        Some("libc".to_string()),
        Some("C.UTF-8".to_string()),
        Some("C.UTF-8".to_string()),
        None,
        None,
        None,
    ]]
}

fn psql_list_tablespaces_catalog_query() -> &'static str {
    "select spcname as \"name\", pg_catalog.pg_get_userbyid(spcowner) as \"owner\", pg_catalog.pg_tablespace_location(oid) as \"location\" from pg_catalog.pg_tablespace order by 1"
}

fn catalog_psql_list_tablespace_rows() -> Vec<Vec<Option<String>>> {
    vec![
        vec![
            Some("pg_default".to_string()),
            Some("postgres".to_string()),
            Some(String::new()),
        ],
        vec![
            Some("pg_global".to_string()),
            Some("postgres".to_string()),
            Some(String::new()),
        ],
    ]
}

fn psql_list_access_methods_catalog_query() -> &'static str {
    "select amname as \"name\", case amtype when 'i' then 'index' when 't' then 'table' end as \"type\" from pg_catalog.pg_am order by 1"
}

fn catalog_psql_list_access_method_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![Some("heap".to_string()), Some("Table".to_string())]]
}

fn psql_describe_schemas_catalog_query() -> &'static str {
    "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\" from pg_catalog.pg_namespace n where n.nspname !~ '^pg_' and n.nspname <> 'information_schema' order by 1"
}

fn psql_describe_schemas_verbose_catalog_query_public_filter(canonical: &str) -> bool {
    canonical
        == "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\", pg_catalog.array_to_string(n.nspacl, e'\\n') as \"access privileges\", pg_catalog.obj_description(n.oid, 'pg_namespace') as \"description\" from pg_catalog.pg_namespace n where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1"
}

fn psql_describe_schema_publications_query() -> &'static str {
    "select pubname from pg_catalog.pg_publication p join pg_catalog.pg_publication_namespace pn on p.oid = pn.pnpubid join pg_catalog.pg_namespace n on n.oid = pn.pnnspid where n.nspname = 'public' order by 1"
}

fn psql_describe_type_catalog_query_type(canonical: &str) -> Option<String> {
    let prefix = "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and (t.typname operator(pg_catalog.~) '^(";
    let suffix = ")$' collate pg_catalog.default or pg_catalog.format_type(t.oid, null) operator(pg_catalog.~) '^(";
    let final_suffix = ")$' collate pg_catalog.default) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2";
    let rest = canonical.strip_prefix(prefix)?;
    let (type_name, rest) = rest.split_once(suffix)?;
    let display_name = rest.strip_suffix(final_suffix)?;
    let matched_type = sql_type_by_catalog_or_display_name(type_name)?;
    (sql_type_by_catalog_or_display_name(display_name) == Some(matched_type))
        .then(|| type_name.to_string())
}

fn psql_describe_pg_catalog_types_query() -> &'static str {
    "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
}

fn psql_describe_pg_catalog_types_verbose_query() -> &'static str {
    "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", t.typname as \"internal name\", case when t.typrelid != 0 then cast('tuple' as pg_catalog.text) when t.typlen < 0 then cast('var' as pg_catalog.text) else cast(t.typlen as pg_catalog.text) end as \"size\", pg_catalog.array_to_string( array( select e.enumlabel from pg_catalog.pg_enum e where e.enumtypid = t.oid order by e.enumsortorder ), e'\\n' ) as \"elements\", pg_catalog.pg_get_userbyid(t.typowner) as \"owner\", pg_catalog.array_to_string(t.typacl, e'\\n') as \"access privileges\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
}

fn sql_type_by_catalog_or_display_name(name: &str) -> Option<SqlType> {
    SUPPORTED_SQL_TYPES
        .into_iter()
        .find(|ty| ty.catalog_name() == name || sql_type_display_name(*ty) == name)
}

fn catalog_psql_describe_type_rows(type_name: &str) -> Vec<Vec<Option<String>>> {
    let Some(ty) = sql_type_by_catalog_or_display_name(type_name) else {
        return Vec::new();
    };
    vec![vec![
        Some("pg_catalog".to_string()),
        Some(sql_type_display_name(ty).to_string()),
        None,
    ]]
}

fn supported_sql_types_by_display_name() -> Vec<SqlType> {
    let mut types = SUPPORTED_SQL_TYPES.to_vec();
    types.sort_by_key(|ty| sql_type_display_name(*ty));
    types
}

fn catalog_psql_describe_type_rows_for_supported_types() -> Vec<Vec<Option<String>>> {
    supported_sql_types_by_display_name()
        .into_iter()
        .map(|ty| {
            vec![
                Some("pg_catalog".to_string()),
                Some(sql_type_display_name(ty).to_string()),
                None,
            ]
        })
        .collect()
}

fn catalog_psql_describe_type_verbose_rows_for_supported_types() -> Vec<Vec<Option<String>>> {
    supported_sql_types_by_display_name()
        .into_iter()
        .map(|ty| {
            vec![
                Some("pg_catalog".to_string()),
                Some(sql_type_display_name(ty).to_string()),
                Some(ty.catalog_name().to_string()),
                Some(sql_type_psql_size(ty).to_string()),
                None,
                Some("postgres".to_string()),
                None,
                None,
            ]
        })
        .collect()
}

fn sql_type_psql_size(ty: SqlType) -> &'static str {
    match ty.type_size() {
        -1 => "var",
        4 => "4",
        _ => "",
    }
}

fn catalog_psql_describe_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_rows_filtered(
        session,
        &PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: None,
        },
    )
}

fn catalog_psql_describe_table_verbose_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_verbose_rows_filtered(
        session,
        &PsqlDescribeTablesFilter {
            namespace: "public".to_string(),
            relname_pattern: None,
        },
    )
}

fn catalog_psql_describe_table_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    if filter.namespace != "public" {
        return Vec::new();
    }
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .filter(|table| {
            filter
                .relname_pattern
                .as_deref()
                .is_none_or(|pattern| psql_relname_pattern_matches(pattern, &table.name))
        })
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("table".to_string()),
                Some("postgres".to_string()),
            ]
        })
        .collect()
}

fn catalog_psql_describe_table_verbose_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    catalog_psql_describe_table_rows_filtered(session, filter)
        .into_iter()
        .map(|mut row| {
            row.extend([
                Some("permanent".to_string()),
                Some("heap".to_string()),
                None,
                None,
            ]);
            row
        })
        .collect()
}

fn catalog_psql_describe_table_privilege_rows_filtered(
    session: &Session,
    filter: &PsqlDescribeTablesFilter,
) -> Vec<Vec<Option<String>>> {
    if filter.namespace != "public" {
        return Vec::new();
    }
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .filter(|table| {
            filter
                .relname_pattern
                .as_deref()
                .is_none_or(|pattern| psql_relname_pattern_matches(pattern, &table.name))
        })
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("table".to_string()),
                None,
                None,
                None,
            ]
        })
        .collect()
}

fn psql_relname_pattern_matches(pattern: &str, table_name: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix(".*") {
        return table_name.starts_with(prefix);
    }
    table_name == pattern
}

fn catalog_psql_describe_schema_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some("public".to_string()),
        Some("postgres".to_string()),
    ]]
}

fn catalog_psql_describe_schema_verbose_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some("public".to_string()),
        Some("postgres".to_string()),
        None,
        None,
    ]]
}

fn pg_catalog_namespace_query() -> &'static str {
    "select oid, nspname from pg_catalog.pg_namespace where nspname = 'public' order by oid"
}

fn pg_catalog_namespace_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some(PUBLIC_NAMESPACE_OID.to_string()),
        Some("public".to_string()),
    ]]
}

fn catalog_describe_relation_lookup_query_table(canonical: &str) -> Option<String> {
    let prefix = "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(";
    let visible_suffix =
        ")$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 2, 3";
    if let Some(table) = canonical
        .strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(visible_suffix))
    {
        return Some(table.to_string());
    }

    let namespace_middle =
        ")$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(";
    let namespace_suffix = ")$' collate pg_catalog.default order by 2, 3";
    let (table, namespace) = canonical
        .strip_prefix(prefix)?
        .strip_suffix(namespace_suffix)?
        .split_once(namespace_middle)?;
    (namespace == "public").then(|| table.to_string())
}

fn catalog_describe_relation_lookup_query_public_namespace(canonical: &str) -> bool {
    canonical
        == "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 2, 3"
}

fn catalog_describe_relation_lookup_rows(
    session: &Session,
    relname_pattern: &str,
) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .filter(|table| psql_relname_pattern_matches(relname_pattern, &table.name))
        .map(|table| {
            vec![
                Some(table.oid.to_string()),
                Some("public".to_string()),
                Some(table.name.clone()),
            ]
        })
        .collect()
}

fn catalog_describe_relation_lookup_rows_for_public_namespace(
    session: &Session,
) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| {
            vec![
                Some(table.oid.to_string()),
                Some("public".to_string()),
                Some(table.name.clone()),
            ]
        })
        .collect()
}

fn catalog_describe_relation_flags_query_oid(canonical: &str) -> Option<u32> {
    let plain_prefix = "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, '', c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '";
    let verbose_prefix = "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, pg_catalog.array_to_string(c.reloptions || array(select 'toast.' || x from pg_catalog.unnest(tc.reloptions) x), ', '), c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '";
    let verbose_wrapped_prefix = "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, pg_catalog.array_to_string(c.reloptions || array(select 'toast.' || x from pg_catalog.unnest(tc.reloptions) x), ', ') , c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '";
    let suffix = "'";
    canonical
        .strip_prefix(plain_prefix)
        .or_else(|| canonical.strip_prefix(verbose_prefix))
        .or_else(|| canonical.strip_prefix(verbose_wrapped_prefix))?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_relation_flags_rows(session: &Session, oid: u32) -> Vec<Vec<Option<String>>> {
    if !session.tables.values().any(|table| table.oid == oid) {
        return Vec::new();
    }

    vec![vec![
        Some("0".to_string()),
        Some("r".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some("f".to_string()),
        Some(String::new()),
        Some("0".to_string()),
        Some(String::new()),
        Some("p".to_string()),
        Some("d".to_string()),
        Some("heap".to_string()),
    ]]
}

fn catalog_describe_attribute_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), (select pg_catalog.pg_get_expr(d.adbin, d.adrelid, true) from pg_catalog.pg_attrdef d where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef), a.attnotnull, (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type t where c.oid = a.attcollation and t.oid = a.atttypid and a.attcollation <> t.typcollation) as attcollation, a.attidentity, a.attgenerated from pg_catalog.pg_attribute a where a.attrelid = '";
    let suffix = "' and a.attnum > 0 and not a.attisdropped order by a.attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_verbose_attribute_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), (select pg_catalog.pg_get_expr(d.adbin, d.adrelid, true) from pg_catalog.pg_attrdef d where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef), a.attnotnull, (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type t where c.oid = a.attcollation and t.oid = a.atttypid and a.attcollation <> t.typcollation) as attcollation, a.attidentity, a.attgenerated, a.attstorage, a.attcompression as attcompression, case when a.attstattarget=-1 then null else a.attstattarget end as attstattarget, pg_catalog.col_description(a.attrelid, a.attnum) from pg_catalog.pg_attribute a where a.attrelid = '";
    let suffix = "' and a.attnum > 0 and not a.attisdropped order by a.attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_verbose_attribute_rows(
    session: &Session,
    oid: u32,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == oid) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some(column.def.name.clone()),
                Some(sql_type_display_name(column.def.ty).to_string()),
                None,
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
                Some(sql_type_storage_code(column.def.ty).to_string()),
                Some(String::new()),
                None,
                None,
            ]
        })
        .collect()
}

fn catalog_describe_attribute_rows(session: &Session, oid: u32) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.values().find(|table| table.oid == oid) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some(column.def.name.clone()),
                Some(sql_type_display_name(column.def.ty).to_string()),
                None,
                Some("f".to_string()),
                None,
                Some(String::new()),
                Some(String::new()),
            ]
        })
        .collect()
}

fn sql_type_storage_code(ty: SqlType) -> &'static str {
    match ty {
        SqlType::Int4 => "p",
        SqlType::Text => "x",
    }
}

fn sql_type_display_name(ty: SqlType) -> &'static str {
    match ty {
        SqlType::Int4 => "integer",
        SqlType::Text => "text",
    }
}

fn sql_type_by_oid(oid: u32) -> Option<SqlType> {
    SUPPORTED_SQL_TYPES
        .into_iter()
        .find(|ty| ty.postgres_oid() == oid)
}

fn psql_describe_query_type_rows(canonical: &str) -> Option<Vec<Vec<Option<String>>>> {
    let prefix =
        "select name as \"column\", pg_catalog.format_type(tp, tpm) as \"type\" from (values ";
    let suffix = ") s(name, tp, tpm)";
    let values = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut rows = Vec::new();
    for raw_value in values.split("),(") {
        let value = raw_value
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')');
        let fields = value.split(',').map(str::trim).collect::<Vec<_>>();
        let [name, oid, _typmod] = fields.as_slice() else {
            return None;
        };
        let name = name.strip_prefix('\'')?.strip_suffix('\'')?;
        let oid = oid.strip_prefix('\'')?.strip_suffix("'::pg_catalog.oid")?;
        let oid = oid.parse::<u32>().ok()?;
        let ty = sql_type_by_oid(oid)?;
        rows.push(vec![
            Some(name.to_string()),
            Some(sql_type_display_name(ty).to_string()),
        ]);
    }
    Some(rows)
}

fn catalog_describe_policy_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select pol.polname, pol.polpermissive, case when pol.polroles = '{0}' then null else pg_catalog.array_to_string(array(select rolname from pg_catalog.pg_roles where oid = any (pol.polroles) order by 1),',') end, pg_catalog.pg_get_expr(pol.polqual, pol.polrelid), pg_catalog.pg_get_expr(pol.polwithcheck, pol.polrelid), case pol.polcmd when 'r' then 'select' when 'a' then 'insert' when 'w' then 'update' when 'd' then 'delete' end as cmd from pg_catalog.pg_policy pol where pol.polrelid = '";
    let suffix = "' order by 1";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_statistic_ext_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select oid, stxrelid::pg_catalog.regclass, stxnamespace::pg_catalog.regnamespace::pg_catalog.text as nsp, stxname, pg_catalog.pg_get_statisticsobjdef_columns(oid) as columns, 'd' = any(stxkind) as ndist_enabled, 'f' = any(stxkind) as deps_enabled, 'm' = any(stxkind) as mcv_enabled, stxstattarget from pg_catalog.pg_statistic_ext where stxrelid = '";
    let suffix = "' order by nsp, stxname";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_publication_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select pubname , null , null from pg_catalog.pg_publication p join pg_catalog.pg_publication_namespace pn on p.oid = pn.pnpubid join pg_catalog.pg_class pc on pc.relnamespace = pn.pnnspid where pc.oid ='";
    let middle = "' and pg_catalog.pg_relation_is_publishable('";
    let suffix = "') union select pubname , pg_get_expr(pr.prqual, c.oid) , (case when pr.prattrs is not null then (select string_agg(attname, ', ') from pg_catalog.generate_series(0, pg_catalog.array_upper(pr.prattrs::pg_catalog.int2[], 1)) s, pg_catalog.pg_attribute where attrelid = pr.prrelid and attnum = prattrs[s]) else null end) from pg_catalog.pg_publication p join pg_catalog.pg_publication_rel pr on p.oid = pr.prpubid join pg_catalog.pg_class c on c.oid = pr.prrelid where pr.prrelid = '";
    let suffix_tail =
        "' union select pubname , null , null from pg_catalog.pg_publication p where p.puballtables and pg_catalog.pg_relation_is_publishable('";
    let final_suffix = "') order by 1";
    let rest = canonical.strip_prefix(prefix)?;
    let (first_oid, rest) = rest.split_once(middle)?;
    let (second_oid, rest) = rest.split_once(suffix)?;
    let (third_oid, rest) = rest.split_once(suffix_tail)?;
    let final_oid = rest.strip_suffix(final_suffix)?;
    if first_oid == second_oid && second_oid == third_oid && third_oid == final_oid {
        first_oid.parse().ok()
    } else {
        None
    }
}

fn catalog_describe_inherits_parent_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select c.oid::pg_catalog.regclass from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhparent and i.inhrelid = '";
    let suffix = "' and c.relkind != 'p' and c.relkind != 'i' order by inhseqno";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_describe_inherits_child_query_oid(canonical: &str) -> Option<u32> {
    let prefix = "select c.oid::pg_catalog.regclass, c.relkind, inhdetachpending, pg_catalog.pg_get_expr(c.relpartbound, c.oid) from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhrelid and i.inhparent = '";
    let suffix = "' order by pg_catalog.pg_get_expr(c.relpartbound, c.oid) = 'default', c.oid::pg_catalog.regclass::pg_catalog.text";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn catalog_empty_rows_for_relation_oid(_oid: u32) -> Vec<Vec<Option<String>>> {
    Vec::new()
}

fn catalog_empty_rows() -> Vec<Vec<Option<String>>> {
    Vec::new()
}

fn pg_catalog_tables_query() -> &'static str {
    "select schemaname, tablename, tableowner from pg_catalog.pg_tables where schemaname = 'public' order by tablename"
}

fn pg_catalog_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("postgres".to_string()),
            ]
        })
        .collect()
}

fn pg_catalog_indexes_query() -> &'static str {
    "select schemaname, tablename, indexname, indexdef from pg_catalog.pg_indexes where schemaname = 'public' order by tablename, indexname"
}

fn pg_catalog_index_rows(_session: &Session) -> Vec<Vec<Option<String>>> {
    Vec::new()
}

fn pg_catalog_class_plain_tables_query() -> &'static str {
    "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relkind = 'r' order by c.relname"
}

fn pg_catalog_class_plain_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    pg_catalog_class_plain_table_rows_from_tables(tables)
}

fn pg_catalog_class_plain_tables_in_query_tables(canonical: &str) -> Option<Vec<String>> {
    let prefix = "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relname in (";
    let suffix = ") and c.relkind = 'r' order by c.relname";
    let list = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut tables = Vec::new();
    for raw_name in list.split(',') {
        let name = raw_name.trim().strip_prefix('\'')?.strip_suffix('\'')?;
        if name.is_empty() {
            return None;
        }
        tables.push(name.to_string());
    }
    if tables.is_empty() {
        None
    } else {
        Some(tables)
    }
}

fn pg_catalog_class_plain_table_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    let requested_tables = table_names.iter().collect::<BTreeSet<_>>();
    let mut tables = requested_tables
        .iter()
        .filter_map(|table_name| session.tables.get(table_name.as_str()))
        .collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    pg_catalog_class_plain_table_rows_from_tables(tables)
}

fn pg_catalog_class_plain_table_rows_from_tables(tables: Vec<&Table>) -> Vec<Vec<Option<String>>> {
    tables
        .into_iter()
        .map(|table| {
            vec![
                Some(table.oid.to_string()),
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("r".to_string()),
                Some("p".to_string()),
            ]
        })
        .collect()
}

fn information_schema_tables_query() -> &'static str {
    "select table_schema, table_name, table_type from information_schema.tables where table_schema = 'public' order by table_name"
}

fn information_schema_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("BASE TABLE".to_string()),
            ]
        })
        .collect()
}

fn information_schema_base_table_discovery_query() -> &'static str {
    "select table_schema, table_name from information_schema.tables where table_type = 'base table' and table_schema not in ('pg_catalog', 'information_schema') order by table_schema, table_name"
}

fn information_schema_base_table_discovery_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| vec![Some("public".to_string()), Some(table.name.clone())])
        .collect()
}

fn information_schema_tables_in_query_tables(canonical: &str) -> Option<Vec<String>> {
    let prefix = "select table_schema, table_name, table_type from information_schema.tables where table_schema = 'public' and table_name in (";
    let suffix = ") order by table_name";
    let list = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut tables = Vec::new();
    for raw_name in list.split(',') {
        let name = raw_name.trim().strip_prefix('\'')?.strip_suffix('\'')?;
        if name.is_empty() {
            return None;
        }
        tables.push(name.to_string());
    }
    if tables.is_empty() {
        None
    } else {
        Some(tables)
    }
}

fn information_schema_table_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    let requested_tables = table_names.iter().collect::<BTreeSet<_>>();
    let mut tables = requested_tables
        .iter()
        .filter_map(|table_name| session.tables.get(table_name.as_str()))
        .collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(|table| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some("BASE TABLE".to_string()),
            ]
        })
        .collect()
}

fn information_schema_rich_tables_query() -> &'static str {
    "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_schema = 'public' order by table_name"
}

fn information_schema_rich_table_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .map(information_schema_rich_table_row)
        .collect()
}

fn information_schema_rich_tables_query_table(canonical: &str) -> Option<String> {
    let prefix = "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_schema = 'public' and table_name = '";
    let suffix = "' order by table_name";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_rich_tables_catalog_query_table(canonical: &str) -> Option<String> {
    let suffix = "' order by table_name";
    let current_database_prefix = "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_catalog = current_database() and table_schema = 'public' and table_name = '";
    if let Some(table) = canonical
        .strip_prefix(current_database_prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
    {
        return Some(table.to_string());
    }
    let literal_catalog_prefix = "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_catalog = 'postgres' and table_schema = 'public' and table_name = '";
    canonical
        .strip_prefix(literal_catalog_prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_rich_table_rows_for_table(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    session
        .tables
        .get(table)
        .map(information_schema_rich_table_row)
        .into_iter()
        .collect()
}

fn information_schema_rich_table_row(table: &Table) -> Vec<Option<String>> {
    vec![
        Some("postgres".to_string()),
        Some("public".to_string()),
        Some(table.name.clone()),
        Some("BASE TABLE".to_string()),
        None,
        None,
        None,
        None,
        None,
        Some("YES".to_string()),
        Some("NO".to_string()),
        None,
    ]
}

fn information_schema_columns_query_table(canonical: &str) -> Option<String> {
    let prefix = "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_column_rows(session: &Session, table: &str) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some("public".to_string()),
                Some(table.name.clone()),
                Some(column.def.name.clone()),
                Some(column.attnum.to_string()),
                Some(sql_type_display_name(column.def.ty).to_string()),
            ]
        })
        .collect()
}

fn information_schema_all_columns_query() -> &'static str {
    "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_all_column_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().map(|column| {
                vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(column.attnum.to_string()),
                    Some(sql_type_display_name(column.def.ty).to_string()),
                ]
            })
        })
        .collect()
}

fn information_schema_columns_in_query_tables(canonical: &str) -> Option<Vec<String>> {
    let prefix = "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' and table_name in (";
    let suffix = ") order by table_name, ordinal_position";
    let list = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut tables = Vec::new();
    for raw_name in list.split(',') {
        let name = raw_name.trim().strip_prefix('\'')?.strip_suffix('\'')?;
        if name.is_empty() {
            return None;
        }
        tables.push(name.to_string());
    }
    if tables.is_empty() {
        None
    } else {
        Some(tables)
    }
}

fn information_schema_column_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    let requested_tables = table_names.iter().collect::<BTreeSet<_>>();
    let mut tables = requested_tables
        .iter()
        .filter_map(|table_name| session.tables.get(table_name.as_str()))
        .collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().map(|column| {
                vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(column.attnum.to_string()),
                    Some(sql_type_display_name(column.def.ty).to_string()),
                ]
            })
        })
        .collect()
}

fn information_schema_column_details_query_table(canonical: &str) -> Option<String> {
    let prefix = "select column_name, data_type, is_nullable, column_default from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_column_detail_rows(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    table
        .columns
        .iter()
        .map(|column| {
            vec![
                Some(column.def.name.clone()),
                Some(sql_type_display_name(column.def.ty).to_string()),
                Some("YES".to_string()),
                None,
            ]
        })
        .collect()
}

fn information_schema_rich_columns_query() -> &'static str {
    "select table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, udt_schema, udt_name from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_rich_column_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(|table| {
            table.columns.iter().map(|column| {
                vec![
                    Some("public".to_string()),
                    Some(table.name.clone()),
                    Some(column.def.name.clone()),
                    Some(column.attnum.to_string()),
                    None,
                    Some("YES".to_string()),
                    Some(sql_type_display_name(column.def.ty).to_string()),
                    Some("pg_catalog".to_string()),
                    Some(column.def.ty.catalog_name().to_string()),
                ]
            })
        })
        .collect()
}

fn information_schema_extended_columns_query() -> &'static str {
    "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_extended_columns_query_table(canonical: &str) -> Option<String> {
    let prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name = '";
    let suffix = "' order by ordinal_position";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_extended_columns_catalog_query_table(canonical: &str) -> Option<String> {
    let suffix = "' order by ordinal_position";
    let current_database_prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_catalog = current_database() and table_schema = 'public' and table_name = '";
    if let Some(table) = canonical
        .strip_prefix(current_database_prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
    {
        return Some(table.to_string());
    }
    let literal_catalog_prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_catalog = 'postgres' and table_schema = 'public' and table_name = '";
    canonical
        .strip_prefix(literal_catalog_prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn information_schema_extended_columns_in_query_tables(canonical: &str) -> Option<Vec<String>> {
    let prefix = "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name in (";
    let suffix = ") order by table_name, ordinal_position";
    let list = canonical.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let mut tables = Vec::new();
    for raw_name in list.split(',') {
        let name = raw_name.trim().strip_prefix('\'')?.strip_suffix('\'')?;
        if name.is_empty() {
            return None;
        }
        tables.push(name.to_string());
    }
    if tables.is_empty() {
        None
    } else {
        Some(tables)
    }
}

fn information_schema_extended_column_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let mut tables = session.tables.values().collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(information_schema_extended_column_rows_for_catalog_table)
        .collect()
}

fn information_schema_extended_column_rows_for_table(
    session: &Session,
    table: &str,
) -> Vec<Vec<Option<String>>> {
    let Some(table) = session.tables.get(table) else {
        return Vec::new();
    };
    information_schema_extended_column_rows_for_catalog_table(table).collect()
}

fn information_schema_extended_column_rows_for_tables(
    session: &Session,
    table_names: &[String],
) -> Vec<Vec<Option<String>>> {
    let requested_tables = table_names.iter().collect::<BTreeSet<_>>();
    let mut tables = requested_tables
        .iter()
        .filter_map(|table_name| session.tables.get(table_name.as_str()))
        .collect::<Vec<_>>();
    tables.sort_by(|left, right| left.name.cmp(&right.name));
    tables
        .into_iter()
        .flat_map(information_schema_extended_column_rows_for_catalog_table)
        .collect()
}

fn information_schema_extended_column_rows_for_catalog_table(
    table: &Table,
) -> impl Iterator<Item = Vec<Option<String>>> + '_ {
    table.columns.iter().map(|column| {
        let (numeric_precision, numeric_precision_radix, numeric_scale) =
            information_schema_numeric_metadata(column.def.ty);
        vec![
            Some("postgres".to_string()),
            Some("public".to_string()),
            Some(table.name.clone()),
            Some(column.def.name.clone()),
            Some(column.attnum.to_string()),
            None,
            Some("YES".to_string()),
            Some(sql_type_display_name(column.def.ty).to_string()),
            None,
            numeric_precision.map(|value| value.to_string()),
            numeric_precision_radix.map(|value| value.to_string()),
            numeric_scale.map(|value| value.to_string()),
            Some("pg_catalog".to_string()),
            Some(column.def.ty.catalog_name().to_string()),
        ]
    })
}

fn information_schema_numeric_metadata(ty: SqlType) -> (Option<i32>, Option<i32>, Option<i32>) {
    match ty {
        SqlType::Int4 => (Some(32), Some(2), Some(0)),
        SqlType::Text => (None, None, None),
    }
}

fn information_schema_schemata_query() -> &'static str {
    "select schema_name, schema_owner from information_schema.schemata where schema_name = 'public' order by schema_name"
}

fn information_schema_schemata_rows() -> Vec<Vec<Option<String>>> {
    vec![vec![
        Some("public".to_string()),
        Some("postgres".to_string()),
    ]]
}

fn information_schema_table_constraints_query() -> &'static str {
    "select table_schema, table_name, constraint_name, constraint_type from information_schema.table_constraints where table_schema = 'public' order by table_name, constraint_name"
}

fn information_schema_table_constraint_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let _supported_plain_table_count = session.tables.len();
    Vec::new()
}

fn information_schema_key_column_usage_query() -> &'static str {
    "select table_schema, table_name, column_name, constraint_name, ordinal_position from information_schema.key_column_usage where table_schema = 'public' order by table_name, ordinal_position"
}

fn information_schema_key_column_usage_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let _supported_plain_table_count = session.tables.len();
    Vec::new()
}

fn information_schema_views_query() -> &'static str {
    "select table_catalog, table_schema, table_name, view_definition, check_option, is_updatable, is_insertable_into, is_trigger_updatable, is_trigger_deletable, is_trigger_insertable from information_schema.views where table_schema = 'public' order by table_name"
}

fn information_schema_view_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let _supported_plain_table_count = session.tables.len();
    Vec::new()
}

fn pg_catalog_views_query() -> &'static str {
    "select schemaname, viewname, viewowner, definition from pg_catalog.pg_views where schemaname = 'public' order by viewname"
}

fn pg_catalog_view_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let _supported_plain_table_count = session.tables.len();
    Vec::new()
}

fn pg_catalog_constraints_query() -> &'static str {
    "select n.nspname, c.relname, con.conname, con.contype from pg_catalog.pg_constraint con join pg_catalog.pg_class c on c.oid = con.conrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' order by c.relname, con.conname"
}

fn pg_catalog_constraint_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let _supported_plain_table_count = session.tables.len();
    Vec::new()
}

fn pg_catalog_attrdefs_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, pg_catalog.pg_get_expr(d.adbin, d.adrelid) as default_expr from pg_catalog.pg_attrdef d join pg_catalog.pg_class c on c.oid = d.adrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace join pg_catalog.pg_attribute a on a.attrelid = d.adrelid and a.attnum = d.adnum where n.nspname = 'public' order by c.relname, a.attnum"
}

fn pg_catalog_attrdef_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let _supported_plain_table_count = session.tables.len();
    Vec::new()
}

fn pg_catalog_descriptions_query() -> &'static str {
    "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind = 'r' order by c.relname, d.objsubid"
}

fn pg_catalog_description_rows(session: &Session) -> Vec<Vec<Option<String>>> {
    let _supported_plain_table_and_column_count = session
        .tables
        .values()
        .map(|table| 1 + table.columns.len())
        .sum::<usize>();
    Vec::new()
}

fn psql_list_object_descriptions_query(canonical: &str) -> bool {
    canonical.starts_with(
        "select distinct tt.nspname as \"schema\", tt.name as \"name\", tt.object as \"object\", d.description as \"description\" from ( select pgc.oid as oid, pgc.tableoid as tableoid",
    ) && canonical.contains("cast('table constraint' as pg_catalog.text) as object")
        && canonical.contains("cast('domain constraint' as pg_catalog.text) as object")
        && canonical.contains("cast('operator class' as pg_catalog.text) as object")
        && canonical.contains("cast('operator family' as pg_catalog.text) as object")
        && canonical.contains("cast('rule' as pg_catalog.text) as object")
        && canonical.contains("cast('trigger' as pg_catalog.text) as object")
        && canonical.contains("join pg_catalog.pg_description d on (tt.oid = d.objoid and tt.tableoid = d.classoid and d.objsubid = 0)")
        && canonical.ends_with("order by 1, 2, 3")
}

fn catalog_type_rows_by_oid() -> Vec<Vec<Option<String>>> {
    let mut types = SUPPORTED_SQL_TYPES;
    types.sort_by_key(|ty| ty.postgres_oid());
    types
        .into_iter()
        .map(|ty| {
            vec![
                Some(ty.postgres_oid().to_string()),
                Some(ty.catalog_name().to_string()),
                Some(ty.type_size().to_string()),
            ]
        })
        .collect()
}

fn catalog_type_rows_by_name() -> Vec<Vec<Option<String>>> {
    let mut types = SUPPORTED_SQL_TYPES;
    types.sort_by_key(|ty| ty.catalog_name());
    types
        .into_iter()
        .map(|ty| {
            vec![
                Some(ty.catalog_name().to_string()),
                Some(ty.postgres_oid().to_string()),
                Some(ty.type_size().to_string()),
            ]
        })
        .collect()
}

fn bind_query_parameters(query: &PreparedQuery, parameters: &[Option<String>]) -> Option<String> {
    if expected_parameter_count(query) != parameters.len() {
        return None;
    }
    let mut bound = query.query.clone();
    for (idx, parameter) in parameters.iter().enumerate().rev() {
        let value = parameter.as_ref()?;
        let placeholder = format!("${}", idx + 1);
        let literal = encode_parameter_literal(
            value,
            query.parameter_type_oids.get(idx).copied().unwrap_or(0),
        )?;
        bound = bound.replace(&placeholder, &literal);
    }
    if bound.as_bytes().windows(1).any(|window| window == b"$") {
        return None;
    }
    Some(bound)
}

fn expected_parameter_count(query: &PreparedQuery) -> usize {
    std::cmp::max(
        query.parameter_type_oids.len(),
        max_placeholder_index(&query.query),
    )
}

fn max_placeholder_index(query: &str) -> usize {
    let mut max_index = 0;
    let mut chars = query.char_indices().peekable();
    while let Some((_, ch)) = chars.next() {
        if ch != '$' {
            continue;
        }

        let mut value = 0usize;
        let mut saw_digit = false;
        while let Some((_, digit)) = chars.peek().copied() {
            let Some(next) = digit.to_digit(10) else {
                break;
            };
            saw_digit = true;
            value = value.saturating_mul(10).saturating_add(next as usize);
            chars.next();
        }
        if saw_digit {
            max_index = max_index.max(value);
        }
    }
    max_index
}

fn encode_parameter_literal(value: &str, type_oid: u32) -> Option<String> {
    match type_oid {
        23 => value.parse::<i32>().ok().map(|parsed| parsed.to_string()),
        25 => Some(sql_quote_text(value)),
        0 if value.parse::<i32>().is_ok() => Some(value.to_string()),
        0 => Some(sql_quote_text(value)),
        _ => None,
    }
}

fn sql_quote_text(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn describe_query_columns(session: &Session, query: &str) -> Option<Vec<Column>> {
    let (table_name, projection) = match parse_command(query).ok() {
        Some(Command::Select(select)) => (select.table, select.projection),
        _ => describe_parameterized_select_shape(query)?,
    };
    let table = session.tables.get(&table_name)?;
    let selected_columns = match projection {
        SelectProjection::All => table.columns.clone(),
        SelectProjection::Columns(columns) => {
            let mut selected = Vec::with_capacity(columns.len());
            for column in columns {
                let column = table
                    .columns
                    .iter()
                    .find(|candidate| candidate.def.name == column)?;
                selected.push(column.clone());
            }
            selected
        }
    };
    Some(
        selected_columns
            .iter()
            .map(|column| match column.def.ty {
                SqlType::Int4 => int4_column(&column.def.name),
                SqlType::Text => text_column(&column.def.name),
            })
            .collect(),
    )
}

fn describe_parameterized_select_shape(query: &str) -> Option<(String, SelectProjection)> {
    let canonical = canonical_sql(query);
    let select_rest = canonical.strip_prefix("select ")?;
    let from_pos = select_rest.find(" from ")?;
    let projection_sql = select_rest[..from_pos].trim();
    let after_from = select_rest[from_pos + " from ".len()..].trim_start();
    let table = after_from
        .split_whitespace()
        .next()?
        .trim_matches('"')
        .to_string();
    if table.is_empty() {
        return None;
    }
    if projection_sql == "*" {
        return Some((table, SelectProjection::All));
    }
    let columns = projection_sql
        .split(',')
        .map(|column| column.trim().trim_matches('"').to_string())
        .collect::<Vec<_>>();
    if columns.is_empty() || columns.iter().any(String::is_empty) {
        return None;
    }
    Some((table, SelectProjection::Columns(columns)))
}

fn catalog_attribute_query_table(canonical: &str) -> Option<String> {
    let prefix = "select attname, atttypid from pg_catalog.pg_attribute where attrelid = '";
    let suffix = "'::regclass and attnum > 0 order by attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn catalog_attribute_detail_query_table(canonical: &str) -> Option<String> {
    let prefix =
        "select attnum, attname, atttypid, attlen from pg_catalog.pg_attribute where attrelid = '";
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
                    Some(column.def.name.clone()),
                    Some(sql_type_oid_text(column.def.ty)),
                ]
            })
            .collect(),
    )
}

fn catalog_attribute_detail_rows(
    session: &Session,
    table: &str,
) -> Option<Vec<Vec<Option<String>>>> {
    let table = session.tables.get(table)?;
    Some(
        table
            .columns
            .iter()
            .map(|column| {
                vec![
                    Some(column.attnum.to_string()),
                    Some(column.def.name.clone()),
                    Some(sql_type_oid_text(column.def.ty)),
                    Some(column.def.ty.type_size().to_string()),
                ]
            })
            .collect(),
    )
}

fn pg_catalog_class_attribute_type_query_table(canonical: &str) -> Option<String> {
    let prefix = "select a.attnum, a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type, a.attnotnull from pg_catalog.pg_attribute a join pg_catalog.pg_class c on c.oid = a.attrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relname = '";
    let suffix = "' and a.attnum > 0 and not a.attisdropped order by a.attnum";
    canonical
        .strip_prefix(prefix)?
        .strip_suffix(suffix)
        .map(str::to_string)
}

fn pg_catalog_class_attribute_type_rows(
    session: &Session,
    table: &str,
) -> Option<Vec<Vec<Option<String>>>> {
    let table = session.tables.get(table)?;
    Some(
        table
            .columns
            .iter()
            .map(|column| {
                vec![
                    Some(column.attnum.to_string()),
                    Some(column.def.name.clone()),
                    Some(sql_type_display_name(column.def.ty).to_string()),
                    Some("f".to_string()),
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

fn write_parse_complete(stream: &mut TcpStream) -> io::Result<()> {
    write_message(stream, b'1', &[])
}

fn write_bind_complete(stream: &mut TcpStream) -> io::Result<()> {
    write_message(stream, b'2', &[])
}

fn write_close_complete(stream: &mut TcpStream) -> io::Result<()> {
    write_message(stream, b'3', &[])
}

fn write_no_data(stream: &mut TcpStream) -> io::Result<()> {
    write_message(stream, b'n', &[])
}

fn write_parameter_description(stream: &mut TcpStream, type_oids: &[u32]) -> io::Result<()> {
    let parameter_count = i16::try_from(type_oids.len())
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "too many parameters"))?;
    let mut payload = Vec::with_capacity(2 + type_oids.len() * 4);
    payload.extend_from_slice(&parameter_count.to_be_bytes());
    for oid in type_oids {
        payload.extend_from_slice(&oid.to_be_bytes());
    }
    write_message(stream, b't', &payload)
}

fn write_single_row(
    stream: &mut TcpStream,
    columns: &[Column],
    rows: &[Vec<Option<String>>],
) -> io::Result<()> {
    write_select_rows(stream, columns, rows, true)
}

fn write_select_rows(
    stream: &mut TcpStream,
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

fn write_rows_with_tag(
    stream: &mut TcpStream,
    columns: &[Column],
    rows: &[Vec<Option<String>>],
    include_row_description: bool,
    tag: &str,
) -> io::Result<()> {
    if include_row_description {
        write_row_description(stream, columns)?;
    }
    for row in rows {
        write_data_row(stream, row)?;
    }
    write_command_complete(stream, tag)
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
                oid: FIRST_USER_RELATION_OID,
                name: "people".to_string(),
                columns: vec![
                    CatalogColumn {
                        attnum: 1,
                        def: gpu_db_protocol::ColumnDef {
                            name: "id".to_string(),
                            ty: gpu_db_protocol::SqlType::Int4,
                        },
                    },
                    CatalogColumn {
                        attnum: 2,
                        def: gpu_db_protocol::ColumnDef {
                            name: "name".to_string(),
                            ty: gpu_db_protocol::SqlType::Text,
                        },
                    },
                ],
                rows: Vec::new(),
            },
        );
        session.tables.insert(
            "teams".to_string(),
            Table {
                oid: FIRST_USER_RELATION_OID + 1,
                name: "teams".to_string(),
                columns: vec![CatalogColumn {
                    attnum: 1,
                    def: gpu_db_protocol::ColumnDef {
                        name: "id".to_string(),
                        ty: gpu_db_protocol::SqlType::Int4,
                    },
                }],
                rows: Vec::new(),
            },
        );

        assert_eq!(
            catalog_table_name_rows(&session),
            vec![
                vec![Some("people".to_string())],
                vec![Some("teams".to_string())],
            ]
        );
        assert_eq!(
            catalog_table_oid_rows(&session),
            vec![
                vec![
                    Some(FIRST_USER_RELATION_OID.to_string()),
                    Some("people".to_string()),
                ],
                vec![
                    Some((FIRST_USER_RELATION_OID + 1).to_string()),
                    Some("teams".to_string()),
                ],
            ]
        );
        assert_eq!(
            catalog_psql_describe_table_rows(&session),
            vec![
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("table".to_string()),
                    Some("postgres".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("table".to_string()),
                    Some("postgres".to_string()),
                ],
            ]
        );
        assert_eq!(
            psql_describe_tables_verbose_catalog_query(),
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        );
        assert_eq!(
            psql_describe_relations_catalog_query(),
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','v','m','s','f','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        );
        assert_eq!(
            catalog_psql_describe_table_verbose_rows(&session),
            vec![
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("table".to_string()),
                    Some("postgres".to_string()),
                    Some("permanent".to_string()),
                    Some("heap".to_string()),
                    None,
                    None,
                ],
                vec![
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("table".to_string()),
                    Some("postgres".to_string()),
                    Some("permanent".to_string()),
                    Some("heap".to_string()),
                    None,
                    None,
                ],
            ]
        );
        assert_eq!(
            psql_describe_tables_catalog_query_filter(
                "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1,2"
            ),
            Some(PsqlDescribeTablesFilter {
                namespace: "public".to_string(),
                relname_pattern: None,
            })
        );
        assert_eq!(
            psql_describe_tables_catalog_query_filter(
                "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and c.relname operator(pg_catalog.~) '^(people_.*)$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1,2"
            ),
            Some(PsqlDescribeTablesFilter {
                namespace: "public".to_string(),
                relname_pattern: Some("people_.*".to_string()),
            })
        );
        assert_eq!(
            psql_describe_tables_catalog_query_filter(
                "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and c.relname operator(pg_catalog.~) '^(people_.*)$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
            ),
            Some(PsqlDescribeTablesFilter {
                namespace: "public".to_string(),
                relname_pattern: Some("people_.*".to_string()),
            })
        );
        assert_eq!(
            psql_describe_tables_verbose_catalog_query_filter(
                "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and c.relname operator(pg_catalog.~) '^(people)$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
            ),
            Some(PsqlDescribeTablesFilter {
                namespace: "public".to_string(),
                relname_pattern: Some("people".to_string()),
            })
        );
        assert_eq!(
            catalog_psql_describe_table_rows_filtered(
                &session,
                &PsqlDescribeTablesFilter {
                    namespace: "public".to_string(),
                    relname_pattern: Some("peo.*".to_string()),
                },
            ),
            vec![vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("table".to_string()),
                Some("postgres".to_string()),
            ]]
        );
        assert_eq!(
            catalog_psql_describe_table_verbose_rows_filtered(
                &session,
                &PsqlDescribeTablesFilter {
                    namespace: "public".to_string(),
                    relname_pattern: Some("peo.*".to_string()),
                },
            ),
            vec![vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("table".to_string()),
                Some("postgres".to_string()),
                Some("permanent".to_string()),
                Some("heap".to_string()),
                None,
                None,
            ]]
        );
        assert_eq!(
            psql_describe_tables_verbose_catalog_query_filter(
                "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('r','p','t','s','') and c.relname operator(pg_catalog.~) '^(peo.*)$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1,2"
            ),
            Some(PsqlDescribeTablesFilter {
                namespace: "public".to_string(),
                relname_pattern: Some("peo.*".to_string()),
            })
        );
        assert_eq!(
            psql_describe_table_privileges_catalog_query_filter(
                "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 's' then 'sequence' when 'f' then 'foreign table' when 'p' then 'partitioned table' end as \"type\", pg_catalog.array_to_string(c.relacl, e'\\n') as \"access privileges\", pg_catalog.array_to_string(array( select attname || e':\\n ' || pg_catalog.array_to_string(attacl, e'\\n ') from pg_catalog.pg_attribute a where attrelid = c.oid and not attisdropped and attacl is not null ), e'\\n') as \"column privileges\", pg_catalog.array_to_string(array( select polname || case when not polpermissive then e' (restrictive)' else '' end || case when polcmd != '*' then e' (' || polcmd::pg_catalog.text || e'):' else e':' end || case when polqual is not null then e'\\n (u): ' || pg_catalog.pg_get_expr(polqual, polrelid) else e'' end || case when polwithcheck is not null then e'\\n (c): ' || pg_catalog.pg_get_expr(polwithcheck, polrelid) else e'' end || case when polroles <> '{0}' then e'\\n to: ' || pg_catalog.array_to_string( array( select rolname from pg_catalog.pg_roles where oid = any (polroles) order by 1 ), e', ') else e'' end from pg_catalog.pg_policy pol where polrelid = c.oid), e'\\n') as \"policies\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('r','v','m','s','f','p') and c.relname operator(pg_catalog.~) '^(people)$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 1, 2"
            ),
            Some(PsqlDescribeTablesFilter {
                namespace: "public".to_string(),
                relname_pattern: Some("people".to_string()),
            })
        );
        assert_eq!(
            catalog_psql_describe_table_privilege_rows_filtered(
                &session,
                &PsqlDescribeTablesFilter {
                    namespace: "public".to_string(),
                    relname_pattern: Some("peo.*".to_string()),
                },
            ),
            vec![vec![
                Some("public".to_string()),
                Some("people".to_string()),
                Some("table".to_string()),
                None,
                None,
                None,
            ]]
        );
        assert_eq!(
            psql_describe_indexes_catalog_query(),
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        );
        assert_eq!(
            psql_describe_indexes_catalog_query_schema_filter(
                "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','s','') and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1,2"
            ),
            Some("public".to_string())
        );
        assert_eq!(
            psql_describe_indexes_catalog_query_schema_filter(
                "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", c2.relname as \"table\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam left join pg_catalog.pg_index i on i.indexrelid = c.oid left join pg_catalog.pg_class c2 on i.indrelid = c2.oid where c.relkind in ('i','i','s','') and n.nspname operator(pg_catalog.~) '^(private)$' collate pg_catalog.default order by 1,2"
            ),
            None
        );
        assert_eq!(
            psql_describe_views_catalog_query(),
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('v','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        );
        assert_eq!(
            psql_describe_views_verbose_catalog_query(),
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('v','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        );
        assert_eq!(
            psql_describe_materialized_views_catalog_query(),
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('m','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        );
        assert_eq!(
            psql_describe_materialized_views_verbose_catalog_query(),
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", am.amname as \"access method\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_am am on am.oid = c.relam where c.relkind in ('m','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        );
        assert_eq!(
            psql_describe_sequences_catalog_query(),
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('s','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        );
        assert_eq!(
            psql_describe_sequences_verbose_catalog_query(),
            "select n.nspname as \"schema\", c.relname as \"name\", case c.relkind when 'r' then 'table' when 'v' then 'view' when 'm' then 'materialized view' when 'i' then 'index' when 's' then 'sequence' when 't' then 'toast table' when 'f' then 'foreign table' when 'p' then 'partitioned table' when 'i' then 'partitioned index' end as \"type\", pg_catalog.pg_get_userbyid(c.relowner) as \"owner\", case c.relpersistence when 'p' then 'permanent' when 't' then 'temporary' when 'u' then 'unlogged' end as \"persistence\", pg_catalog.pg_size_pretty(pg_catalog.pg_table_size(c.oid)) as \"size\", pg_catalog.obj_description(c.oid, 'pg_class') as \"description\" from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relkind in ('s','') and n.nspname <> 'pg_catalog' and n.nspname !~ '^pg_toast' and n.nspname <> 'information_schema' and pg_catalog.pg_table_is_visible(c.oid) order by 1,2"
        );
        assert_eq!(
            psql_describe_functions_catalog_query(),
            "select n.nspname as \"schema\", p.proname as \"name\", pg_catalog.pg_get_function_result(p.oid) as \"result data type\", pg_catalog.pg_get_function_arguments(p.oid) as \"argument data types\", case p.prokind when 'a' then 'agg' when 'w' then 'window' when 'p' then 'proc' else 'func' end as \"type\" from pg_catalog.pg_proc p left join pg_catalog.pg_namespace n on n.oid = p.pronamespace where pg_catalog.pg_function_is_visible(p.oid) and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' order by 1, 2, 4"
        );
        assert_eq!(
            psql_list_extensions_catalog_query(),
            "select e.extname as \"name\", e.extversion as \"version\", n.nspname as \"schema\", c.description as \"description\" from pg_catalog.pg_extension e left join pg_catalog.pg_namespace n on n.oid = e.extnamespace left join pg_catalog.pg_description c on c.objoid = e.oid and c.classoid = 'pg_catalog.pg_extension'::pg_catalog.regclass order by 1"
        );
        assert_eq!(
            psql_list_languages_catalog_query(),
            "select l.lanname as \"name\", pg_catalog.pg_get_userbyid(l.lanowner) as \"owner\", l.lanpltrusted as \"trusted\", d.description as \"description\" from pg_catalog.pg_language l left join pg_catalog.pg_description d on d.classoid = l.tableoid and d.objoid = l.oid and d.objsubid = 0 where l.lanplcallfoid != 0 order by 1"
        );
        assert_eq!(
            psql_describe_roles_catalog_query(),
            "select r.rolname, r.rolsuper, r.rolinherit, r.rolcreaterole, r.rolcreatedb, r.rolcanlogin, r.rolconnlimit, r.rolvaliduntil , r.rolreplication , r.rolbypassrls from pg_catalog.pg_roles r where r.rolname !~ '^pg_' order by 1"
        );
        assert_eq!(
            catalog_psql_describe_role_rows(),
            vec![vec![
                Some("postgres".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("t".to_string()),
                Some("-1".to_string()),
                None,
                Some("t".to_string()),
                Some("t".to_string()),
            ]]
        );
        assert_eq!(
            psql_list_databases_catalog_query(),
            "select d.datname as \"name\", pg_catalog.pg_get_userbyid(d.datdba) as \"owner\", pg_catalog.pg_encoding_to_char(d.encoding) as \"encoding\", case d.datlocprovider when 'c' then 'libc' when 'i' then 'icu' end as \"locale provider\", d.datcollate as \"collate\", d.datctype as \"ctype\", d.daticulocale as \"icu locale\", d.daticurules as \"icu rules\", pg_catalog.array_to_string(d.datacl, e'\\n') as \"access privileges\" from pg_catalog.pg_database d order by 1"
        );
        assert_eq!(
            catalog_psql_list_database_rows(),
            vec![vec![
                Some("postgres".to_string()),
                Some("postgres".to_string()),
                Some("UTF8".to_string()),
                Some("libc".to_string()),
                Some("C.UTF-8".to_string()),
                Some("C.UTF-8".to_string()),
                None,
                None,
                None,
            ]]
        );
        assert_eq!(
            psql_list_tablespaces_catalog_query(),
            "select spcname as \"name\", pg_catalog.pg_get_userbyid(spcowner) as \"owner\", pg_catalog.pg_tablespace_location(oid) as \"location\" from pg_catalog.pg_tablespace order by 1"
        );
        assert_eq!(
            catalog_psql_list_tablespace_rows(),
            vec![
                vec![
                    Some("pg_default".to_string()),
                    Some("postgres".to_string()),
                    Some(String::new()),
                ],
                vec![
                    Some("pg_global".to_string()),
                    Some("postgres".to_string()),
                    Some(String::new()),
                ],
            ]
        );
        assert_eq!(
            psql_list_access_methods_catalog_query(),
            "select amname as \"name\", case amtype when 'i' then 'index' when 't' then 'table' end as \"type\" from pg_catalog.pg_am order by 1"
        );
        assert_eq!(
            catalog_psql_list_access_method_rows(),
            vec![vec![Some("heap".to_string()), Some("Table".to_string())]]
        );
        assert!(catalog_empty_rows().is_empty());
        assert!(catalog_psql_describe_table_rows_filtered(
            &session,
            &PsqlDescribeTablesFilter {
                namespace: "private".to_string(),
                relname_pattern: None,
            },
        )
        .is_empty());
        assert!(psql_relname_pattern_matches("peo.*", "people"));
        assert!(!psql_relname_pattern_matches("tea.*", "people"));
        assert!(psql_relname_pattern_matches("people", "people"));
        assert!(!psql_relname_pattern_matches("people", "teams"));
        assert_eq!(
            psql_describe_schemas_catalog_query(),
            "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\" from pg_catalog.pg_namespace n where n.nspname !~ '^pg_' and n.nspname <> 'information_schema' order by 1"
        );
        assert!(psql_describe_schemas_verbose_catalog_query_public_filter(
            "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\", pg_catalog.array_to_string(n.nspacl, e'\\n') as \"access privileges\", pg_catalog.obj_description(n.oid, 'pg_namespace') as \"description\" from pg_catalog.pg_namespace n where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 1"
        ));
        assert!(!psql_describe_schemas_verbose_catalog_query_public_filter(
            "select n.nspname as \"name\", pg_catalog.pg_get_userbyid(n.nspowner) as \"owner\", pg_catalog.array_to_string(n.nspacl, e'\\n') as \"access privileges\", pg_catalog.obj_description(n.oid, 'pg_namespace') as \"description\" from pg_catalog.pg_namespace n where n.nspname operator(pg_catalog.~) '^(private)$' collate pg_catalog.default order by 1"
        ));
        assert_eq!(
            psql_describe_schema_publications_query(),
            "select pubname from pg_catalog.pg_publication p join pg_catalog.pg_publication_namespace pn on p.oid = pn.pnpubid join pg_catalog.pg_namespace n on n.oid = pn.pnnspid where n.nspname = 'public' order by 1"
        );
        assert_eq!(
            psql_list_domains_catalog_query(),
            "select n.nspname as \"schema\", t.typname as \"name\", pg_catalog.format_type(t.typbasetype, t.typtypmod) as \"type\", (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type bt where c.oid = t.typcollation and bt.oid = t.typbasetype and t.typcollation <> bt.typcollation) as \"collation\", case when t.typnotnull then 'not null' end as \"nullable\", t.typdefault as \"default\", pg_catalog.array_to_string(array( select pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where t.oid = r.contypid ), ' ') as \"check\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where t.typtype = 'd' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_type_is_visible(t.oid) order by 1, 2"
        );
        assert_eq!(
            psql_list_domains_verbose_catalog_query(),
            "select n.nspname as \"schema\", t.typname as \"name\", pg_catalog.format_type(t.typbasetype, t.typtypmod) as \"type\", (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type bt where c.oid = t.typcollation and bt.oid = t.typbasetype and t.typcollation <> bt.typcollation) as \"collation\", case when t.typnotnull then 'not null' end as \"nullable\", t.typdefault as \"default\", pg_catalog.array_to_string(array( select pg_catalog.pg_get_constraintdef(r.oid, true) from pg_catalog.pg_constraint r where t.oid = r.contypid ), ' ') as \"check\", pg_catalog.array_to_string(t.typacl, e'\\n') as \"access privileges\", d.description as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace left join pg_catalog.pg_description d on d.classoid = t.tableoid and d.objoid = t.oid and d.objsubid = 0 where t.typtype = 'd' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_type_is_visible(t.oid) order by 1, 2"
        );
        assert_eq!(
            psql_list_aggregates_catalog_query(),
            "select n.nspname as \"schema\", p.proname as \"name\", pg_catalog.format_type(p.prorettype, null) as \"result data type\", case when p.pronargs = 0 then cast('*' as pg_catalog.text) else pg_catalog.pg_get_function_arguments(p.oid) end as \"argument data types\", pg_catalog.obj_description(p.oid, 'pg_proc') as \"description\" from pg_catalog.pg_proc p left join pg_catalog.pg_namespace n on n.oid = p.pronamespace where p.prokind = 'a' and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_function_is_visible(p.oid) order by 1, 2, 4"
        );
        assert_eq!(
            psql_list_conversions_catalog_query(),
            "select n.nspname as \"schema\", c.conname as \"name\", pg_catalog.pg_encoding_to_char(c.conforencoding) as \"source\", pg_catalog.pg_encoding_to_char(c.contoencoding) as \"destination\", case when c.condefault then 'yes' else 'no' end as \"default?\" from pg_catalog.pg_conversion c join pg_catalog.pg_namespace n on n.oid = c.connamespace where true and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_conversion_is_visible(c.oid) order by 1, 2"
        );
        assert_eq!(
            psql_list_operators_catalog_query(),
            "select n.nspname as \"schema\", o.oprname as \"name\", case when o.oprkind='l' then null else pg_catalog.format_type(o.oprleft, null) end as \"left arg type\", case when o.oprkind='r' then null else pg_catalog.format_type(o.oprright, null) end as \"right arg type\", pg_catalog.format_type(o.oprresult, null) as \"result type\", coalesce(pg_catalog.obj_description(o.oid, 'pg_operator'), pg_catalog.obj_description(o.oprcode, 'pg_proc')) as \"description\" from pg_catalog.pg_operator o left join pg_catalog.pg_namespace n on n.oid = o.oprnamespace where n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and pg_catalog.pg_operator_is_visible(o.oid) order by 1, 2, 3, 4"
        );
        assert_eq!(
            psql_list_collations_catalog_query(),
            "select n.nspname as \"schema\", c.collname as \"name\", case c.collprovider when 'd' then 'default' when 'c' then 'libc' when 'i' then 'icu' end as \"provider\", c.collcollate as \"collate\", c.collctype as \"ctype\", c.colliculocale as \"icu locale\", c.collicurules as \"icu rules\", case when c.collisdeterministic then 'yes' else 'no' end as \"deterministic?\" from pg_catalog.pg_collation c, pg_catalog.pg_namespace n where n.oid = c.collnamespace and n.nspname <> 'pg_catalog' and n.nspname <> 'information_schema' and c.collencoding in (-1, pg_catalog.pg_char_to_encoding(pg_catalog.getdatabaseencoding())) and pg_catalog.pg_collation_is_visible(c.oid) order by 1, 2"
        );
        assert_eq!(
            psql_list_casts_catalog_query(),
            "select pg_catalog.format_type(castsource, null) as \"source type\", pg_catalog.format_type(casttarget, null) as \"target type\", case when c.castmethod = 'b' then '(binary coercible)' when c.castmethod = 'i' then '(with inout)' else p.proname end as \"function\", case when c.castcontext = 'e' then 'no' when c.castcontext = 'a' then 'in assignment' else 'yes' end as \"implicit?\" from pg_catalog.pg_cast c left join pg_catalog.pg_proc p on c.castfunc = p.oid left join pg_catalog.pg_type ts on c.castsource = ts.oid left join pg_catalog.pg_namespace ns on ns.oid = ts.typnamespace left join pg_catalog.pg_type tt on c.casttarget = tt.oid left join pg_catalog.pg_namespace nt on nt.oid = tt.typnamespace where ( (true and pg_catalog.pg_type_is_visible(ts.oid) ) or (true and pg_catalog.pg_type_is_visible(tt.oid) ) ) order by 1, 2"
        );
        assert_eq!(
            psql_list_publications_catalog_query(),
            "select pubname as \"name\", pg_catalog.pg_get_userbyid(pubowner) as \"owner\", puballtables as \"all tables\", pubinsert as \"inserts\", pubupdate as \"updates\", pubdelete as \"deletes\", pubtruncate as \"truncates\", pubviaroot as \"via root\" from pg_catalog.pg_publication order by 1"
        );
        assert_eq!(
            psql_list_publications_verbose_catalog_query(),
            "select oid, pubname, pg_catalog.pg_get_userbyid(pubowner) as owner, puballtables, pubinsert, pubupdate, pubdelete, pubtruncate, pubviaroot from pg_catalog.pg_publication order by 2"
        );
        assert_eq!(
            psql_list_subscriptions_catalog_query(),
            "select subname as \"name\" , pg_catalog.pg_get_userbyid(subowner) as \"owner\" , subenabled as \"enabled\" , subpublications as \"publication\" from pg_catalog.pg_subscription where subdbid = (select oid from pg_catalog.pg_database where datname = pg_catalog.current_database())order by 1"
        );
        assert_eq!(
            psql_list_default_access_privileges_catalog_query(),
            "select pg_catalog.pg_get_userbyid(d.defaclrole) as \"owner\", n.nspname as \"schema\", case d.defaclobjtype when 'r' then 'table' when 's' then 'sequence' when 'f' then 'function' when 't' then 'type' when 'n' then 'schema' end as \"type\", pg_catalog.array_to_string(d.defaclacl, e'\\n') as \"access privileges\" from pg_catalog.pg_default_acl d left join pg_catalog.pg_namespace n on n.oid = d.defaclnamespace order by 1, 2, 3"
        );
        assert_eq!(
            catalog_psql_describe_schema_rows(),
            vec![vec![
                Some("public".to_string()),
                Some("postgres".to_string())
            ]]
        );
        assert_eq!(
            catalog_psql_describe_schema_verbose_rows(),
            vec![vec![
                Some("public".to_string()),
                Some("postgres".to_string()),
                None,
                None
            ]]
        );
        assert_eq!(
            psql_describe_type_catalog_query_type(
                "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and (t.typname operator(pg_catalog.~) '^(int4)$' collate pg_catalog.default or pg_catalog.format_type(t.oid, null) operator(pg_catalog.~) '^(int4)$' collate pg_catalog.default) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
            ),
            Some("int4".to_string())
        );
        assert_eq!(
            psql_describe_type_catalog_query_type(
                "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and (t.typname operator(pg_catalog.~) '^(text)$' collate pg_catalog.default or pg_catalog.format_type(t.oid, null) operator(pg_catalog.~) '^(text)$' collate pg_catalog.default) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
            ),
            Some("text".to_string())
        );
        assert_eq!(
            catalog_psql_describe_type_rows("int4"),
            vec![vec![
                Some("pg_catalog".to_string()),
                Some("integer".to_string()),
                None
            ]]
        );
        assert_eq!(
            catalog_psql_describe_type_rows("text"),
            vec![vec![
                Some("pg_catalog".to_string()),
                Some("text".to_string()),
                None
            ]]
        );
        assert_eq!(
            psql_describe_pg_catalog_types_query(),
            "select n.nspname as \"schema\", pg_catalog.format_type(t.oid, null) as \"name\", pg_catalog.obj_description(t.oid, 'pg_type') as \"description\" from pg_catalog.pg_type t left join pg_catalog.pg_namespace n on n.oid = t.typnamespace where (t.typrelid = 0 or (select c.relkind = 'c' from pg_catalog.pg_class c where c.oid = t.typrelid)) and not exists(select 1 from pg_catalog.pg_type el where el.oid = t.typelem and el.typarray = t.oid) and n.nspname operator(pg_catalog.~) '^(pg_catalog)$' collate pg_catalog.default order by 1, 2"
        );
        assert_eq!(
            catalog_psql_describe_type_rows_for_supported_types(),
            vec![
                vec![
                    Some("pg_catalog".to_string()),
                    Some("integer".to_string()),
                    None
                ],
                vec![
                    Some("pg_catalog".to_string()),
                    Some("text".to_string()),
                    None
                ],
            ]
        );
        assert_eq!(
            catalog_psql_describe_type_verbose_rows_for_supported_types(),
            vec![
                vec![
                    Some("pg_catalog".to_string()),
                    Some("integer".to_string()),
                    Some("int4".to_string()),
                    Some("4".to_string()),
                    None,
                    Some("postgres".to_string()),
                    None,
                    None,
                ],
                vec![
                    Some("pg_catalog".to_string()),
                    Some("text".to_string()),
                    Some("text".to_string()),
                    Some("var".to_string()),
                    None,
                    Some("postgres".to_string()),
                    None,
                    None,
                ],
            ]
        );
        assert_eq!(
            catalog_describe_relation_lookup_query_table(
                "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(people)$' collate pg_catalog.default and pg_catalog.pg_table_is_visible(c.oid) order by 2, 3"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            catalog_describe_relation_lookup_query_table(
                "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(people)$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 2, 3"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            catalog_describe_relation_lookup_query_table(
                "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(peo.*)$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 2, 3"
            ),
            Some("peo.*".to_string())
        );
        assert_eq!(
            catalog_describe_relation_lookup_query_table(
                "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where c.relname operator(pg_catalog.~) '^(people)$' collate pg_catalog.default and n.nspname operator(pg_catalog.~) '^(private)$' collate pg_catalog.default order by 2, 3"
            ),
            None
        );
        assert!(catalog_describe_relation_lookup_query_public_namespace(
            "select c.oid, n.nspname, c.relname from pg_catalog.pg_class c left join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname operator(pg_catalog.~) '^(public)$' collate pg_catalog.default order by 2, 3"
        ));
        assert_eq!(
            catalog_describe_relation_lookup_rows(&session, "people"),
            vec![vec![
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
            ]]
        );
        assert_eq!(
            catalog_describe_relation_lookup_rows(&session, "peo.*"),
            vec![vec![
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
            ]]
        );
        assert!(catalog_describe_relation_lookup_rows(&session, "missing").is_empty());
        assert_eq!(
            catalog_describe_relation_lookup_rows_for_public_namespace(&session),
            vec![
                vec![
                    Some(FIRST_USER_RELATION_OID.to_string()),
                    Some("public".to_string()),
                    Some("people".to_string()),
                ],
                vec![
                    Some((FIRST_USER_RELATION_OID + 1).to_string()),
                    Some("public".to_string()),
                    Some("teams".to_string()),
                ],
            ]
        );
        assert_eq!(
            catalog_describe_relation_flags_query_oid(
                "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, '', c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '16384'"
            ),
            Some(FIRST_USER_RELATION_OID)
        );
        assert_eq!(
            catalog_describe_relation_flags_query_oid(
                "select c.relchecks, c.relkind, c.relhasindex, c.relhasrules, c.relhastriggers, c.relrowsecurity, c.relforcerowsecurity, false as relhasoids, c.relispartition, pg_catalog.array_to_string(c.reloptions || array(select 'toast.' || x from pg_catalog.unnest(tc.reloptions) x), ', ') , c.reltablespace, case when c.reloftype = 0 then '' else c.reloftype::pg_catalog.regtype::pg_catalog.text end, c.relpersistence, c.relreplident, am.amname from pg_catalog.pg_class c left join pg_catalog.pg_class tc on (c.reltoastrelid = tc.oid) left join pg_catalog.pg_am am on (c.relam = am.oid) where c.oid = '16384'"
            ),
            Some(FIRST_USER_RELATION_OID)
        );
        assert_eq!(
            catalog_describe_relation_flags_rows(&session, FIRST_USER_RELATION_OID),
            vec![vec![
                Some("0".to_string()),
                Some("r".to_string()),
                Some("f".to_string()),
                Some("f".to_string()),
                Some("f".to_string()),
                Some("f".to_string()),
                Some("f".to_string()),
                Some("f".to_string()),
                Some("f".to_string()),
                Some(String::new()),
                Some("0".to_string()),
                Some(String::new()),
                Some("p".to_string()),
                Some("d".to_string()),
                Some("heap".to_string()),
            ]]
        );
        assert_eq!(
            catalog_describe_attribute_query_oid(
                "select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), (select pg_catalog.pg_get_expr(d.adbin, d.adrelid, true) from pg_catalog.pg_attrdef d where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef), a.attnotnull, (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type t where c.oid = a.attcollation and t.oid = a.atttypid and a.attcollation <> t.typcollation) as attcollation, a.attidentity, a.attgenerated from pg_catalog.pg_attribute a where a.attrelid = '16384' and a.attnum > 0 and not a.attisdropped order by a.attnum"
            ),
            Some(FIRST_USER_RELATION_OID)
        );
        assert_eq!(
            catalog_describe_attribute_rows(&session, FIRST_USER_RELATION_OID),
            vec![
                vec![
                    Some("id".to_string()),
                    Some("integer".to_string()),
                    None,
                    Some("f".to_string()),
                    None,
                    Some(String::new()),
                    Some(String::new()),
                ],
                vec![
                    Some("name".to_string()),
                    Some("text".to_string()),
                    None,
                    Some("f".to_string()),
                    None,
                    Some(String::new()),
                    Some(String::new()),
                ],
            ]
        );
        assert_eq!(
            catalog_describe_verbose_attribute_query_oid(
                "select a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), (select pg_catalog.pg_get_expr(d.adbin, d.adrelid, true) from pg_catalog.pg_attrdef d where d.adrelid = a.attrelid and d.adnum = a.attnum and a.atthasdef), a.attnotnull, (select c.collname from pg_catalog.pg_collation c, pg_catalog.pg_type t where c.oid = a.attcollation and t.oid = a.atttypid and a.attcollation <> t.typcollation) as attcollation, a.attidentity, a.attgenerated, a.attstorage, a.attcompression as attcompression, case when a.attstattarget=-1 then null else a.attstattarget end as attstattarget, pg_catalog.col_description(a.attrelid, a.attnum) from pg_catalog.pg_attribute a where a.attrelid = '16384' and a.attnum > 0 and not a.attisdropped order by a.attnum"
            ),
            Some(FIRST_USER_RELATION_OID)
        );
        assert_eq!(
            catalog_describe_verbose_attribute_rows(&session, FIRST_USER_RELATION_OID),
            vec![
                vec![
                    Some("id".to_string()),
                    Some("integer".to_string()),
                    None,
                    Some("f".to_string()),
                    None,
                    Some(String::new()),
                    Some(String::new()),
                    Some("p".to_string()),
                    Some(String::new()),
                    None,
                    None,
                ],
                vec![
                    Some("name".to_string()),
                    Some("text".to_string()),
                    None,
                    Some("f".to_string()),
                    None,
                    Some(String::new()),
                    Some(String::new()),
                    Some("x".to_string()),
                    Some(String::new()),
                    None,
                    None,
                ],
            ]
        );
        assert_eq!(
            catalog_describe_policy_query_oid(
                "select pol.polname, pol.polpermissive, case when pol.polroles = '{0}' then null else pg_catalog.array_to_string(array(select rolname from pg_catalog.pg_roles where oid = any (pol.polroles) order by 1),',') end, pg_catalog.pg_get_expr(pol.polqual, pol.polrelid), pg_catalog.pg_get_expr(pol.polwithcheck, pol.polrelid), case pol.polcmd when 'r' then 'select' when 'a' then 'insert' when 'w' then 'update' when 'd' then 'delete' end as cmd from pg_catalog.pg_policy pol where pol.polrelid = '16384' order by 1"
            ),
            Some(FIRST_USER_RELATION_OID)
        );
        assert_eq!(
            catalog_describe_statistic_ext_query_oid(
                "select oid, stxrelid::pg_catalog.regclass, stxnamespace::pg_catalog.regnamespace::pg_catalog.text as nsp, stxname, pg_catalog.pg_get_statisticsobjdef_columns(oid) as columns, 'd' = any(stxkind) as ndist_enabled, 'f' = any(stxkind) as deps_enabled, 'm' = any(stxkind) as mcv_enabled, stxstattarget from pg_catalog.pg_statistic_ext where stxrelid = '16384' order by nsp, stxname"
            ),
            Some(FIRST_USER_RELATION_OID)
        );
        assert_eq!(
            catalog_describe_publication_query_oid(
                "select pubname , null , null from pg_catalog.pg_publication p join pg_catalog.pg_publication_namespace pn on p.oid = pn.pnpubid join pg_catalog.pg_class pc on pc.relnamespace = pn.pnnspid where pc.oid ='16384' and pg_catalog.pg_relation_is_publishable('16384') union select pubname , pg_get_expr(pr.prqual, c.oid) , (case when pr.prattrs is not null then (select string_agg(attname, ', ') from pg_catalog.generate_series(0, pg_catalog.array_upper(pr.prattrs::pg_catalog.int2[], 1)) s, pg_catalog.pg_attribute where attrelid = pr.prrelid and attnum = prattrs[s]) else null end) from pg_catalog.pg_publication p join pg_catalog.pg_publication_rel pr on p.oid = pr.prpubid join pg_catalog.pg_class c on c.oid = pr.prrelid where pr.prrelid = '16384' union select pubname , null , null from pg_catalog.pg_publication p where p.puballtables and pg_catalog.pg_relation_is_publishable('16384') order by 1"
            ),
            Some(FIRST_USER_RELATION_OID)
        );
        assert_eq!(
            catalog_describe_inherits_parent_query_oid(
                "select c.oid::pg_catalog.regclass from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhparent and i.inhrelid = '16384' and c.relkind != 'p' and c.relkind != 'i' order by inhseqno"
            ),
            Some(FIRST_USER_RELATION_OID)
        );
        assert_eq!(
            catalog_describe_inherits_child_query_oid(
                "select c.oid::pg_catalog.regclass, c.relkind, inhdetachpending, pg_catalog.pg_get_expr(c.relpartbound, c.oid) from pg_catalog.pg_class c, pg_catalog.pg_inherits i where c.oid = i.inhrelid and i.inhparent = '16384' order by pg_catalog.pg_get_expr(c.relpartbound, c.oid) = 'default', c.oid::pg_catalog.regclass::pg_catalog.text"
            ),
            Some(FIRST_USER_RELATION_OID)
        );
        assert_eq!(
            pg_catalog_tables_query(),
            "select schemaname, tablename, tableowner from pg_catalog.pg_tables where schemaname = 'public' order by tablename"
        );
        assert_eq!(
            pg_catalog_table_rows(&session),
            vec![
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("postgres".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("postgres".to_string()),
                ],
            ]
        );
        assert_eq!(
            pg_catalog_indexes_query(),
            "select schemaname, tablename, indexname, indexdef from pg_catalog.pg_indexes where schemaname = 'public' order by tablename, indexname"
        );
        assert!(pg_catalog_index_rows(&session).is_empty());
        assert_eq!(
            pg_catalog_class_plain_tables_query(),
            "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relkind = 'r' order by c.relname"
        );
        assert_eq!(
            pg_catalog_class_plain_table_rows(&session),
            vec![
                vec![
                    Some(FIRST_USER_RELATION_OID.to_string()),
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("r".to_string()),
                    Some("p".to_string()),
                ],
                vec![
                    Some((FIRST_USER_RELATION_OID + 1).to_string()),
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("r".to_string()),
                    Some("p".to_string()),
                ],
            ]
        );
        assert_eq!(
            pg_catalog_class_plain_tables_in_query_tables(
                "select c.oid, n.nspname, c.relname, c.relkind, c.relpersistence from pg_catalog.pg_class c join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relname in ('teams', 'missing', 'people') and c.relkind = 'r' order by c.relname"
            ),
            Some(vec![
                "teams".to_string(),
                "missing".to_string(),
                "people".to_string(),
            ])
        );
        assert_eq!(
            pg_catalog_class_plain_table_rows_for_tables(
                &session,
                &[
                    "teams".to_string(),
                    "missing".to_string(),
                    "people".to_string(),
                    "teams".to_string(),
                ],
            ),
            vec![
                vec![
                    Some(FIRST_USER_RELATION_OID.to_string()),
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("r".to_string()),
                    Some("p".to_string()),
                ],
                vec![
                    Some((FIRST_USER_RELATION_OID + 1).to_string()),
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("r".to_string()),
                    Some("p".to_string()),
                ],
            ]
        );
        assert_eq!(
            pg_catalog_descriptions_query(),
            "select n.nspname, c.relname, a.attname, d.description from pg_catalog.pg_description d join pg_catalog.pg_class c on c.oid = d.objoid join pg_catalog.pg_namespace n on n.oid = c.relnamespace left join pg_catalog.pg_attribute a on a.attrelid = c.oid and a.attnum = d.objsubid where n.nspname = 'public' and c.relkind = 'r' order by c.relname, d.objsubid"
        );
        assert!(pg_catalog_description_rows(&session).is_empty());
        assert!(psql_list_object_descriptions_query(&canonical_sql(
            "SELECT DISTINCT tt.nspname AS \"Schema\", tt.name AS \"Name\", tt.object AS \"Object\", d.description AS \"Description\"
             FROM (
               SELECT pgc.oid as oid, pgc.tableoid AS tableoid,
               n.nspname as nspname,
               CAST(pgc.conname AS pg_catalog.text) as name, CAST('table constraint' AS pg_catalog.text) as object
               FROM pg_catalog.pg_constraint pgc
               JOIN pg_catalog.pg_class c ON c.oid = pgc.conrelid
               LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
               WHERE n.nspname <> 'pg_catalog' AND n.nspname <> 'information_schema'
                 AND pg_catalog.pg_table_is_visible(c.oid)
             UNION ALL
               SELECT pgc.oid as oid, pgc.tableoid AS tableoid,
               n.nspname as nspname,
               CAST(pgc.conname AS pg_catalog.text) as name, CAST('domain constraint' AS pg_catalog.text) as object
               FROM pg_catalog.pg_constraint pgc
               JOIN pg_catalog.pg_type t ON t.oid = pgc.contypid
               LEFT JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
               WHERE n.nspname <> 'pg_catalog' AND n.nspname <> 'information_schema'
                 AND pg_catalog.pg_type_is_visible(t.oid)
             UNION ALL
               SELECT o.oid as oid, o.tableoid as tableoid,
               n.nspname as nspname,
               CAST(o.opcname AS pg_catalog.text) as name,
               CAST('operator class' AS pg_catalog.text) as object
               FROM pg_catalog.pg_opclass o
               JOIN pg_catalog.pg_am am ON o.opcmethod = am.oid
               JOIN pg_catalog.pg_namespace n ON n.oid = o.opcnamespace
                 AND n.nspname <> 'pg_catalog'
                 AND n.nspname <> 'information_schema'
                 AND pg_catalog.pg_opclass_is_visible(o.oid)
             UNION ALL
               SELECT opf.oid as oid, opf.tableoid as tableoid,
               n.nspname as nspname,
               CAST(opf.opfname AS pg_catalog.text) AS name,
               CAST('operator family' AS pg_catalog.text) as object
               FROM pg_catalog.pg_opfamily opf
               JOIN pg_catalog.pg_am am ON opf.opfmethod = am.oid
               JOIN pg_catalog.pg_namespace n ON opf.opfnamespace = n.oid
                 AND n.nspname <> 'pg_catalog'
                 AND n.nspname <> 'information_schema'
                 AND pg_catalog.pg_opfamily_is_visible(opf.oid)
             UNION ALL
               SELECT r.oid as oid, r.tableoid as tableoid,
               n.nspname as nspname,
               CAST(r.rulename AS pg_catalog.text) as name, CAST('rule' AS pg_catalog.text) as object
               FROM pg_catalog.pg_rewrite r
               JOIN pg_catalog.pg_class c ON c.oid = r.ev_class
               LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
               WHERE r.rulename != '_RETURN'
                 AND n.nspname <> 'pg_catalog'
                 AND n.nspname <> 'information_schema'
                 AND pg_catalog.pg_table_is_visible(c.oid)
             UNION ALL
               SELECT t.oid as oid, t.tableoid as tableoid,
               n.nspname as nspname,
               CAST(t.tgname AS pg_catalog.text) as name, CAST('trigger' AS pg_catalog.text) as object
               FROM pg_catalog.pg_trigger t
               JOIN pg_catalog.pg_class c ON c.oid = t.tgrelid
               LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
               WHERE n.nspname <> 'pg_catalog'
                 AND n.nspname <> 'information_schema'
                 AND pg_catalog.pg_table_is_visible(c.oid)
             ) AS tt
             JOIN pg_catalog.pg_description d ON (tt.oid = d.objoid AND tt.tableoid = d.classoid AND d.objsubid = 0)
             ORDER BY 1, 2, 3;"
        )));
        assert_eq!(
            information_schema_table_rows(&session),
            vec![
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("BASE TABLE".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("BASE TABLE".to_string()),
                ],
            ]
        );
        assert_eq!(
            information_schema_tables_in_query_tables(
                "select table_schema, table_name, table_type from information_schema.tables where table_schema = 'public' and table_name in ('people', 'missing', 'teams') order by table_name"
            ),
            Some(vec![
                "people".to_string(),
                "missing".to_string(),
                "teams".to_string()
            ])
        );
        assert_eq!(
            information_schema_table_rows_for_tables(
                &session,
                &[
                    "people".to_string(),
                    "missing".to_string(),
                    "people".to_string(),
                    "teams".to_string()
                ]
            ),
            vec![
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("BASE TABLE".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("BASE TABLE".to_string()),
                ],
            ]
        );
        assert_eq!(
            information_schema_base_table_discovery_query(),
            "select table_schema, table_name from information_schema.tables where table_type = 'base table' and table_schema not in ('pg_catalog', 'information_schema') order by table_schema, table_name"
        );
        assert_eq!(
            information_schema_base_table_discovery_rows(&session),
            vec![
                vec![Some("public".to_string()), Some("people".to_string())],
                vec![Some("public".to_string()), Some("teams".to_string())],
            ]
        );
        assert_eq!(
            information_schema_rich_tables_query(),
            "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_schema = 'public' order by table_name"
        );
        assert_eq!(
            information_schema_rich_table_rows(&session),
            vec![
                vec![
                    Some("postgres".to_string()),
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("BASE TABLE".to_string()),
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some("YES".to_string()),
                    Some("NO".to_string()),
                    None,
                ],
                vec![
                    Some("postgres".to_string()),
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("BASE TABLE".to_string()),
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some("YES".to_string()),
                    Some("NO".to_string()),
                    None,
                ],
            ]
        );
        assert_eq!(
            information_schema_rich_tables_query_table(
                "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_schema = 'public' and table_name = 'people' order by table_name"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            information_schema_rich_table_rows_for_table(&session, "people"),
            vec![vec![
                Some("postgres".to_string()),
                Some("public".to_string()),
                Some("people".to_string()),
                Some("BASE TABLE".to_string()),
                None,
                None,
                None,
                None,
                None,
                Some("YES".to_string()),
                Some("NO".to_string()),
                None,
            ]]
        );
        assert!(information_schema_rich_table_rows_for_table(&session, "missing").is_empty());
        assert_eq!(
            information_schema_columns_query_table(
                "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' and table_name = 'people' order by ordinal_position"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            information_schema_column_rows(&session, "people"),
            vec![
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    Some("integer".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("name".to_string()),
                    Some("2".to_string()),
                    Some("text".to_string()),
                ],
            ]
        );
        assert_eq!(
            information_schema_all_columns_query(),
            "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
        );
        assert_eq!(
            information_schema_all_column_rows(&session),
            vec![
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    Some("integer".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("name".to_string()),
                    Some("2".to_string()),
                    Some("text".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    Some("integer".to_string()),
                ],
            ]
        );
        assert_eq!(
            information_schema_columns_in_query_tables(
                "select table_schema, table_name, column_name, ordinal_position, data_type from information_schema.columns where table_schema = 'public' and table_name in ('people', 'missing', 'teams') order by table_name, ordinal_position"
            ),
            Some(vec![
                "people".to_string(),
                "missing".to_string(),
                "teams".to_string()
            ])
        );
        assert_eq!(
            information_schema_column_rows_for_tables(
                &session,
                &[
                    "people".to_string(),
                    "missing".to_string(),
                    "people".to_string(),
                    "teams".to_string()
                ]
            ),
            vec![
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    Some("integer".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("name".to_string()),
                    Some("2".to_string()),
                    Some("text".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    Some("integer".to_string()),
                ],
            ]
        );
        assert_eq!(
            information_schema_rich_columns_query(),
            "select table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, udt_schema, udt_name from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
        );
        assert_eq!(
            information_schema_column_details_query_table(
                "select column_name, data_type, is_nullable, column_default from information_schema.columns where table_schema = 'public' and table_name = 'people' order by ordinal_position"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            information_schema_column_detail_rows(&session, "people"),
            vec![
                vec![
                    Some("id".to_string()),
                    Some("integer".to_string()),
                    Some("YES".to_string()),
                    None,
                ],
                vec![
                    Some("name".to_string()),
                    Some("text".to_string()),
                    Some("YES".to_string()),
                    None,
                ],
            ]
        );
        assert_eq!(
            information_schema_rich_column_rows(&session),
            vec![
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("integer".to_string()),
                    Some("pg_catalog".to_string()),
                    Some("int4".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("name".to_string()),
                    Some("2".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("text".to_string()),
                    Some("pg_catalog".to_string()),
                    Some("text".to_string()),
                ],
                vec![
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("integer".to_string()),
                    Some("pg_catalog".to_string()),
                    Some("int4".to_string()),
                ],
            ]
        );
        assert_eq!(
            information_schema_extended_columns_query(),
            "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' order by table_name, ordinal_position"
        );
        assert_eq!(
            information_schema_extended_columns_query_table(
                "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name = 'people' order by ordinal_position"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            information_schema_extended_columns_catalog_query_table(
                "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_catalog = current_database() and table_schema = 'public' and table_name = 'people' order by ordinal_position"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            information_schema_extended_columns_catalog_query_table(
                "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_catalog = 'postgres' and table_schema = 'public' and table_name = 'people' order by ordinal_position"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            information_schema_extended_columns_in_query_tables(
                "select table_catalog, table_schema, table_name, column_name, ordinal_position, column_default, is_nullable, data_type, character_maximum_length, numeric_precision, numeric_precision_radix, numeric_scale, udt_schema, udt_name from information_schema.columns where table_schema = 'public' and table_name in ('teams', 'missing', 'people') order by table_name, ordinal_position"
            ),
            Some(vec![
                "teams".to_string(),
                "missing".to_string(),
                "people".to_string(),
            ])
        );
        assert_eq!(
            information_schema_numeric_metadata(SqlType::Int4),
            (Some(32), Some(2), Some(0))
        );
        assert_eq!(
            information_schema_numeric_metadata(SqlType::Text),
            (None, None, None)
        );
        assert_eq!(
            information_schema_extended_column_rows(&session),
            vec![
                vec![
                    Some("postgres".to_string()),
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("integer".to_string()),
                    None,
                    Some("32".to_string()),
                    Some("2".to_string()),
                    Some("0".to_string()),
                    Some("pg_catalog".to_string()),
                    Some("int4".to_string()),
                ],
                vec![
                    Some("postgres".to_string()),
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("name".to_string()),
                    Some("2".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("text".to_string()),
                    None,
                    None,
                    None,
                    None,
                    Some("pg_catalog".to_string()),
                    Some("text".to_string()),
                ],
                vec![
                    Some("postgres".to_string()),
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("integer".to_string()),
                    None,
                    Some("32".to_string()),
                    Some("2".to_string()),
                    Some("0".to_string()),
                    Some("pg_catalog".to_string()),
                    Some("int4".to_string()),
                ],
            ]
        );
        assert_eq!(
            information_schema_extended_column_rows_for_table(&session, "people"),
            vec![
                vec![
                    Some("postgres".to_string()),
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("integer".to_string()),
                    None,
                    Some("32".to_string()),
                    Some("2".to_string()),
                    Some("0".to_string()),
                    Some("pg_catalog".to_string()),
                    Some("int4".to_string()),
                ],
                vec![
                    Some("postgres".to_string()),
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("name".to_string()),
                    Some("2".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("text".to_string()),
                    None,
                    None,
                    None,
                    None,
                    Some("pg_catalog".to_string()),
                    Some("text".to_string()),
                ],
            ]
        );
        assert!(information_schema_extended_column_rows_for_table(&session, "missing").is_empty());
        assert_eq!(
            information_schema_extended_column_rows_for_tables(
                &session,
                &[
                    "teams".to_string(),
                    "missing".to_string(),
                    "people".to_string(),
                    "teams".to_string(),
                ],
            ),
            vec![
                vec![
                    Some("postgres".to_string()),
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("integer".to_string()),
                    None,
                    Some("32".to_string()),
                    Some("2".to_string()),
                    Some("0".to_string()),
                    Some("pg_catalog".to_string()),
                    Some("int4".to_string()),
                ],
                vec![
                    Some("postgres".to_string()),
                    Some("public".to_string()),
                    Some("people".to_string()),
                    Some("name".to_string()),
                    Some("2".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("text".to_string()),
                    None,
                    None,
                    None,
                    None,
                    Some("pg_catalog".to_string()),
                    Some("text".to_string()),
                ],
                vec![
                    Some("postgres".to_string()),
                    Some("public".to_string()),
                    Some("teams".to_string()),
                    Some("id".to_string()),
                    Some("1".to_string()),
                    None,
                    Some("YES".to_string()),
                    Some("integer".to_string()),
                    None,
                    Some("32".to_string()),
                    Some("2".to_string()),
                    Some("0".to_string()),
                    Some("pg_catalog".to_string()),
                    Some("int4".to_string()),
                ],
            ]
        );
        assert_eq!(
            information_schema_schemata_query(),
            "select schema_name, schema_owner from information_schema.schemata where schema_name = 'public' order by schema_name"
        );
        assert_eq!(
            information_schema_schemata_rows(),
            vec![vec![
                Some("public".to_string()),
                Some("postgres".to_string())
            ]]
        );
        assert_eq!(
            pg_catalog_namespace_query(),
            "select oid, nspname from pg_catalog.pg_namespace where nspname = 'public' order by oid"
        );
        assert_eq!(
            pg_catalog_namespace_rows(),
            vec![vec![
                Some(PUBLIC_NAMESPACE_OID.to_string()),
                Some("public".to_string())
            ]]
        );
        assert_eq!(
            information_schema_table_constraints_query(),
            "select table_schema, table_name, constraint_name, constraint_type from information_schema.table_constraints where table_schema = 'public' order by table_name, constraint_name"
        );
        assert!(information_schema_table_constraint_rows(&session).is_empty());
        assert_eq!(
            information_schema_key_column_usage_query(),
            "select table_schema, table_name, column_name, constraint_name, ordinal_position from information_schema.key_column_usage where table_schema = 'public' order by table_name, ordinal_position"
        );
        assert!(information_schema_key_column_usage_rows(&session).is_empty());
        assert_eq!(
            information_schema_views_query(),
            "select table_catalog, table_schema, table_name, view_definition, check_option, is_updatable, is_insertable_into, is_trigger_updatable, is_trigger_deletable, is_trigger_insertable from information_schema.views where table_schema = 'public' order by table_name"
        );
        assert!(information_schema_view_rows(&session).is_empty());
        assert_eq!(
            pg_catalog_views_query(),
            "select schemaname, viewname, viewowner, definition from pg_catalog.pg_views where schemaname = 'public' order by viewname"
        );
        assert!(pg_catalog_view_rows(&session).is_empty());
        assert_eq!(
            pg_catalog_constraints_query(),
            "select n.nspname, c.relname, con.conname, con.contype from pg_catalog.pg_constraint con join pg_catalog.pg_class c on c.oid = con.conrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' order by c.relname, con.conname"
        );
        assert!(pg_catalog_constraint_rows(&session).is_empty());
        assert_eq!(
            pg_catalog_attrdefs_query(),
            "select n.nspname, c.relname, a.attname, pg_catalog.pg_get_expr(d.adbin, d.adrelid) as default_expr from pg_catalog.pg_attrdef d join pg_catalog.pg_class c on c.oid = d.adrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace join pg_catalog.pg_attribute a on a.attrelid = d.adrelid and a.attnum = d.adnum where n.nspname = 'public' order by c.relname, a.attnum"
        );
        assert!(pg_catalog_attrdef_rows(&session).is_empty());
        assert_eq!(
            catalog_type_rows_by_oid(),
            vec![
                vec![
                    Some("23".to_string()),
                    Some("int4".to_string()),
                    Some("4".to_string()),
                ],
                vec![
                    Some("25".to_string()),
                    Some("text".to_string()),
                    Some("-1".to_string()),
                ],
            ]
        );
        assert_eq!(
            catalog_type_rows_by_name(),
            vec![
                vec![
                    Some("int4".to_string()),
                    Some("23".to_string()),
                    Some("4".to_string()),
                ],
                vec![
                    Some("text".to_string()),
                    Some("25".to_string()),
                    Some("-1".to_string()),
                ],
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
        assert_eq!(
            catalog_attribute_detail_query_table(
                "select attnum, attname, atttypid, attlen from pg_catalog.pg_attribute where attrelid = 'people'::regclass and attnum > 0 order by attnum"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            catalog_attribute_detail_rows(&session, "people").unwrap(),
            vec![
                vec![
                    Some("1".to_string()),
                    Some("id".to_string()),
                    Some("23".to_string()),
                    Some("4".to_string()),
                ],
                vec![
                    Some("2".to_string()),
                    Some("name".to_string()),
                    Some("25".to_string()),
                    Some("-1".to_string()),
                ],
            ]
        );
        assert_eq!(
            pg_catalog_class_attribute_type_query_table(
                "select a.attnum, a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod) as data_type, a.attnotnull from pg_catalog.pg_attribute a join pg_catalog.pg_class c on c.oid = a.attrelid join pg_catalog.pg_namespace n on n.oid = c.relnamespace where n.nspname = 'public' and c.relname = 'people' and a.attnum > 0 and not a.attisdropped order by a.attnum"
            ),
            Some("people".to_string())
        );
        assert_eq!(
            pg_catalog_class_attribute_type_rows(&session, "people").unwrap(),
            vec![
                vec![
                    Some("1".to_string()),
                    Some("id".to_string()),
                    Some("integer".to_string()),
                    Some("f".to_string()),
                ],
                vec![
                    Some("2".to_string()),
                    Some("name".to_string()),
                    Some("text".to_string()),
                    Some("f".to_string()),
                ],
            ]
        );
        assert!(catalog_attribute_rows(&session, "missing").is_none());
    }

    #[test]
    fn catalog_introspection_helpers_expose_relation_oids_and_attribute_details() {
        let mut session = Session::default();
        session.tables.insert(
            "people".to_string(),
            Table {
                oid: FIRST_USER_RELATION_OID,
                name: "people".to_string(),
                columns: vec![CatalogColumn {
                    attnum: 1,
                    def: gpu_db_protocol::ColumnDef {
                        name: "id".to_string(),
                        ty: gpu_db_protocol::SqlType::Int4,
                    },
                }],
                rows: Vec::new(),
            },
        );

        assert_eq!(
            catalog_table_oid_rows(&session),
            vec![vec![
                Some(FIRST_USER_RELATION_OID.to_string()),
                Some("people".to_string()),
            ]]
        );
        assert_eq!(
            catalog_attribute_detail_rows(&session, "people").unwrap(),
            vec![vec![
                Some("1".to_string()),
                Some("id".to_string()),
                Some("23".to_string()),
                Some("4".to_string()),
            ]]
        );
    }

    #[test]
    fn row_filtering_honors_disjunctive_select_groups() {
        let table = Table {
            oid: FIRST_USER_RELATION_OID,
            name: "people".to_string(),
            columns: vec![
                CatalogColumn {
                    attnum: 1,
                    def: gpu_db_protocol::ColumnDef {
                        name: "id".to_string(),
                        ty: gpu_db_protocol::SqlType::Int4,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: gpu_db_protocol::SqlType::Text,
                    },
                },
            ],
            rows: Vec::new(),
        };
        let Command::Select(select) =
            parse_command("SELECT id, name FROM people WHERE (id = 1) OR (name = 'Grace')")
                .unwrap()
        else {
            panic!("expected SELECT plan");
        };

        assert!(row_matches_select_filters(
            &table,
            &[SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            &select,
        )
        .unwrap());
        assert!(row_matches_select_filters(
            &table,
            &[SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            &select,
        )
        .unwrap());
        assert!(!row_matches_select_filters(
            &table,
            &[SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            &select,
        )
        .unwrap());
    }

    #[test]
    fn describe_query_columns_handles_parameterized_select_shapes() {
        let mut session = Session::default();
        session.tables.insert(
            "people".to_string(),
            Table {
                oid: FIRST_USER_RELATION_OID,
                name: "people".to_string(),
                columns: vec![
                    CatalogColumn {
                        attnum: 1,
                        def: gpu_db_protocol::ColumnDef {
                            name: "id".to_string(),
                            ty: gpu_db_protocol::SqlType::Int4,
                        },
                    },
                    CatalogColumn {
                        attnum: 2,
                        def: gpu_db_protocol::ColumnDef {
                            name: "name".to_string(),
                            ty: gpu_db_protocol::SqlType::Text,
                        },
                    },
                ],
                rows: Vec::new(),
            },
        );

        assert_eq!(
            describe_query_columns(
                &session,
                "SELECT name, id FROM people WHERE id = $1 ORDER BY name DESC LIMIT 1",
            ),
            Some(vec![text_column("name"), int4_column("id")])
        );
    }

    #[test]
    fn psql_gdesc_type_rows_formats_supported_row_description_types() {
        assert_eq!(
            psql_describe_query_type_rows(
                "select name as \"column\", pg_catalog.format_type(tp, tpm) as \"type\" from (values ('name', '25'::pg_catalog.oid, -1),('id', '23'::pg_catalog.oid, -1)) s(name, tp, tpm)"
            ),
            Some(vec![
                vec![Some("name".to_string()), Some("text".to_string())],
                vec![Some("id".to_string()), Some("integer".to_string())],
            ])
        );
        assert_eq!(
            psql_describe_query_type_rows(
                "select name as \"column\", pg_catalog.format_type(tp, tpm) as \"type\" from (values ('unsupported', '999999'::pg_catalog.oid, -1)) s(name, tp, tpm)"
            ),
            None
        );
    }

    #[test]
    fn extended_parameter_binding_substitutes_text_and_int_literals() {
        let query = PreparedQuery {
            query: "SELECT id, name FROM people WHERE id = $1 ORDER BY name LIMIT $2".to_string(),
            parameter_type_oids: vec![23, 23],
        };

        assert_eq!(
            bind_query_parameters(&query, &[Some("2".to_string()), Some("1".to_string())]),
            Some("SELECT id, name FROM people WHERE id = 2 ORDER BY name LIMIT 1".to_string())
        );

        let text_query = PreparedQuery {
            query: "SELECT id FROM people WHERE name = $1".to_string(),
            parameter_type_oids: vec![25],
        };
        assert_eq!(
            bind_query_parameters(&text_query, &[Some("O'Brien".to_string())]),
            Some("SELECT id FROM people WHERE name = 'O''Brien'".to_string())
        );
    }

    #[test]
    fn extended_error_path_binding_rejects_parameter_count_mismatch() {
        let inferred_query = PreparedQuery {
            query: "SELECT id FROM people WHERE id = $1".to_string(),
            parameter_type_oids: Vec::new(),
        };

        assert_eq!(expected_parameter_count(&inferred_query), 1);
        assert_eq!(
            bind_query_parameters(&inferred_query, &[Some("2".to_string())]),
            Some("SELECT id FROM people WHERE id = 2".to_string())
        );
        assert_eq!(bind_query_parameters(&inferred_query, &[]), None);
        assert_eq!(
            bind_query_parameters(
                &inferred_query,
                &[Some("2".to_string()), Some("extra".to_string())]
            ),
            None
        );

        let typed_query = PreparedQuery {
            query: "SELECT id FROM people".to_string(),
            parameter_type_oids: vec![23],
        };
        assert_eq!(expected_parameter_count(&typed_query), 1);
        assert_eq!(bind_query_parameters(&typed_query, &[]), None);
    }

    #[test]
    fn extended_prepared_portal_lifecycle_closes_session_local_state() {
        let mut session = Session::default();
        let query = PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![23],
        };
        session.prepared.insert(
            "lookup".to_string(),
            PreparedStatement::Extended(query.clone()),
        );
        session.portals.insert(
            "lookup_portal".to_string(),
            Portal {
                statement_name: "lookup".to_string(),
                query: query.clone(),
                parameters: vec![Some("1".to_string())],
                described: false,
            },
        );
        session.portals.insert(
            "other_portal".to_string(),
            Portal {
                statement_name: "other".to_string(),
                query: query.clone(),
                parameters: vec![Some("2".to_string())],
                described: false,
            },
        );

        session.close_extended_target(DescribeTarget::Portal, "other_portal");
        assert!(session.prepared.contains_key("lookup"));
        assert!(session.portals.contains_key("lookup_portal"));
        assert!(!session.portals.contains_key("other_portal"));

        session.close_extended_target(DescribeTarget::Statement, "lookup");
        assert!(!session.prepared.contains_key("lookup"));
        assert!(!session.portals.contains_key("lookup_portal"));
    }

    #[test]
    fn extended_describe_uses_catalog_columns_for_bound_selects() {
        let mut session = Session::default();
        session.tables.insert(
            "people".to_string(),
            Table {
                oid: FIRST_USER_RELATION_OID,
                name: "people".to_string(),
                columns: vec![
                    CatalogColumn {
                        attnum: 1,
                        def: gpu_db_protocol::ColumnDef {
                            name: "id".to_string(),
                            ty: SqlType::Int4,
                        },
                    },
                    CatalogColumn {
                        attnum: 2,
                        def: gpu_db_protocol::ColumnDef {
                            name: "name".to_string(),
                            ty: SqlType::Text,
                        },
                    },
                ],
                rows: Vec::new(),
            },
        );

        assert_eq!(
            describe_query_columns(
                &session,
                "SELECT name, id FROM people WHERE id = 2 ORDER BY name LIMIT 1"
            )
            .unwrap(),
            vec![text_column("name"), int4_column("id")]
        );
    }

    #[test]
    fn extended_cursor_fetch_count_helpers_use_supported_select_results() {
        assert_eq!(
            parse_declare_cursor(
                "DECLARE _psql_cursor NO SCROLL CURSOR FOR\nSELECT id, name FROM people ORDER BY id"
            ),
            Some((
                "_psql_cursor".to_string(),
                "select id, name from people order by id".to_string()
            ))
        );
        assert_eq!(
            parse_fetch_forward("FETCH FORWARD 2 FROM _psql_cursor"),
            Some(("_psql_cursor".to_string(), 2))
        );
        assert_eq!(
            parse_close_cursor("CLOSE _psql_cursor"),
            Some("_psql_cursor".to_string())
        );

        let mut session = Session::default();
        session.tables.insert(
            "people".to_string(),
            Table {
                oid: FIRST_USER_RELATION_OID,
                name: "people".to_string(),
                columns: vec![
                    CatalogColumn {
                        attnum: 1,
                        def: gpu_db_protocol::ColumnDef {
                            name: "id".to_string(),
                            ty: SqlType::Int4,
                        },
                    },
                    CatalogColumn {
                        attnum: 2,
                        def: gpu_db_protocol::ColumnDef {
                            name: "name".to_string(),
                            ty: SqlType::Text,
                        },
                    },
                ],
                rows: vec![
                    vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
                    vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                    vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
                ],
            },
        );
        let Command::Select(select) =
            parse_command("select id, name from people order by id").unwrap()
        else {
            panic!("expected supported SELECT");
        };
        let result = execute_select_result(&session, &select).unwrap();
        assert_eq!(result.columns, vec![int4_column("id"), text_column("name")]);
        assert_eq!(
            result.rows,
            vec![
                vec![Some("1".to_string()), Some("Ada".to_string())],
                vec![Some("2".to_string()), Some("Linus".to_string())],
                vec![Some("3".to_string()), Some("Grace".to_string())],
            ]
        );
    }

    #[test]
    fn catalog_helpers_parse_catalog_qualified_information_schema_table_filters() {
        let current_database_query = "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_catalog = current_database() and table_schema = 'public' and table_name = 'people' order by table_name";
        let literal_catalog_query = "select table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action from information_schema.tables where table_catalog = 'postgres' and table_schema = 'public' and table_name = 'people' order by table_name";

        assert_eq!(
            information_schema_rich_tables_catalog_query_table(current_database_query),
            Some("people".to_string())
        );
        assert_eq!(
            information_schema_rich_tables_catalog_query_table(literal_catalog_query),
            Some("people".to_string())
        );
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
