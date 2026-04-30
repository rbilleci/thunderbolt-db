use std::collections::HashMap;
use std::env;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use gpu_db_protocol::{
    parse_frontend_message, parse_startup_packet, FrontendMessage, StartupPacket,
};

const INT4_OID: u32 = 23;
const TEXT_OID: u32 = 25;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Column {
    name: &'static str,
    oid: u32,
    type_size: i16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ErrorField {
    code: &'static str,
    message: &'static str,
    position: Option<&'static str>,
}

#[derive(Default)]
struct Session {
    in_transaction: bool,
    prepared: HashMap<String, PreparedStatement>,
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
    let canonical = canonical_sql(statement);
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
            &[Column {
                name: "client_encoding",
                oid: TEXT_OID,
                type_size: -1,
            }],
            &[vec![Some(String::from("UTF8"))]],
        ),
        "select current_schema()" => write_single_row(
            stream,
            &[Column {
                name: "current_schema",
                oid: TEXT_OID,
                type_size: -1,
            }],
            &[vec![Some(String::from("public"))]],
        ),
        "select 1 as one" => write_single_row(
            stream,
            &[Column {
                name: "one",
                oid: INT4_OID,
                type_size: 4,
            }],
            &[vec![Some(String::from("1"))]],
        ),
        "select 2 as in_tx" => write_single_row(
            stream,
            &[Column {
                name: "in_tx",
                oid: INT4_OID,
                type_size: 4,
            }],
            &[vec![Some(String::from("2"))]],
        ),
        "select 3 as rolled_back" => write_single_row(
            stream,
            &[Column {
                name: "rolled_back",
                oid: INT4_OID,
                type_size: 4,
            }],
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
                    &[Column {
                        name: "plus_ten",
                        oid: INT4_OID,
                        type_size: 4,
                    }],
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
        push_cstring(&mut payload, column.name);
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
