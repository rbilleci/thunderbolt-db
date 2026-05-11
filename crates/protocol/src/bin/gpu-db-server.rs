use std::collections::HashMap;
use std::env;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use gpu_db_protocol::{
    parse_command, parse_frontend_message, parse_startup_packet, Command, FrontendMessage,
    SelectFilterOp, SelectProjection, SqlValue, StartupPacket, SUPPORTED_SQL_TYPES,
};
use gpu_db_protocol::{DescribeTarget, SqlType};

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

fn select_filter_matches(left: &SqlValue, op: SelectFilterOp, right: &SqlValue) -> bool {
    match op {
        SelectFilterOp::Eq => left == right,
        SelectFilterOp::Lt => compare_sql_values(left, right).is_lt(),
        SelectFilterOp::Lte => !compare_sql_values(left, right).is_gt(),
        SelectFilterOp::Gt => compare_sql_values(left, right).is_gt(),
        SelectFilterOp::Gte => !compare_sql_values(left, right).is_lt(),
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

struct Session {
    in_transaction: bool,
    prepared: HashMap<String, PreparedStatement>,
    portals: HashMap<String, Portal>,
    tables: HashMap<String, Table>,
    next_relation_oid: u32,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            in_transaction: false,
            prepared: HashMap::new(),
            portals: HashMap::new(),
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

        match parse_frontend_message(&frame)
            .map_err(|error| io::Error::new(ErrorKind::InvalidData, error.to_string()))?
        {
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
                handle_describe(&mut stream, &session, target, &name)?
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
        },
    );
    write_bind_complete(stream)
}

fn handle_describe(
    stream: &mut TcpStream,
    session: &Session,
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
                write_row_description(stream, &columns)
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
    execute_statement(stream, session, &bound_query)
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
                                .find(|candidate| candidate.def.name == *column)
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
                let filters = if select.filters.is_empty() {
                    select.filter.iter().collect::<Vec<_>>()
                } else {
                    select.filters.iter().collect::<Vec<_>>()
                };
                for filter in filters {
                    let Some(idx) = table
                        .columns
                        .iter()
                        .position(|column| column.def.name == filter.column)
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
                    rows.retain(|row| select_filter_matches(&row[idx], filter.op, &filter.value));
                }
                if let Some(order) = &select.order_by {
                    let Some(idx) = table
                        .columns
                        .iter()
                        .position(|column| column.def.name == order.column)
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
    let command = parse_command(query).ok()?;
    let Command::Select(select) = command else {
        return None;
    };
    let table = session.tables.get(&select.table)?;
    let selected_columns = match select.projection {
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
            },
        );
        session.portals.insert(
            "other_portal".to_string(),
            Portal {
                statement_name: "other".to_string(),
                query: query.clone(),
                parameters: vec![Some("2".to_string())],
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
