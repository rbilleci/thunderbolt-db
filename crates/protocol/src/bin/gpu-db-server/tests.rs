use super::*;
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

fn tcp_pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let client = TcpStream::connect(addr).unwrap();
    let (server, _) = listener.accept().unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(1)))
        .unwrap();
    (server, client)
}

fn read_backend_messages(stream: &mut dyn ReadWrite, count: usize) -> Vec<(u8, Vec<u8>)> {
    let mut messages = Vec::with_capacity(count);
    for _ in 0..count {
        let mut tag = [0_u8; 1];
        stream.read_exact(&mut tag).unwrap();
        let mut len = [0_u8; 4];
        stream.read_exact(&mut len).unwrap();
        let payload_len = u32::from_be_bytes(len) as usize - 4;
        let mut payload = vec![0_u8; payload_len];
        stream.read_exact(&mut payload).unwrap();
        messages.push((tag[0], payload));
    }
    messages
}

fn read_backend_tags(stream: &mut dyn ReadWrite, count: usize) -> Vec<u8> {
    let messages = read_backend_messages(stream, count);
    let mut tags = Vec::with_capacity(messages.len());
    for (tag, _) in messages {
        tags.push(tag);
    }
    tags
}

#[test]
fn asyncpg_default_pool_session_reset_query_is_session_control_noop() {
    let mut session = Session::default();
    let (mut writer, mut reader) = tcp_pair();

    run_simple_query(
        &mut writer,
        &mut session,
        "SELECT pg_advisory_unlock_all(); CLOSE ALL; UNLISTEN *; RESET ALL;",
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 7);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C', b'C', b'C', b'C', b'Z']
    );
    assert_eq!(messages[2].1, b"SELECT 1\0".to_vec());

    run_simple_query(&mut writer, &mut session, "SELECT 1 AS one;").unwrap();
    let messages = read_backend_messages(&mut reader, 4);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C', b'Z']
    );
    assert_eq!(messages[2].1, b"SELECT 1\0".to_vec());
}

fn error_field_value(payload: &[u8], field_tag: u8) -> Option<String> {
    let mut idx = 0;
    while idx < payload.len() {
        let tag = payload[idx];
        idx += 1;
        if tag == 0 {
            break;
        }
        let end = payload[idx..]
            .iter()
            .position(|byte| *byte == 0)
            .map(|offset| idx + offset)?;
        let value = std::str::from_utf8(&payload[idx..end]).ok()?;
        if tag == field_tag {
            return Some(value.to_string());
        }
        idx = end + 1;
    }
    None
}

fn test_table(name: &str, rows: Vec<Vec<SqlValue>>) -> Table {
    Table {
        oid: FIRST_USER_RELATION_OID,
        name: name.to_string(),
        columns: vec![CatalogColumn {
            attnum: 1,
            def: gpu_db_protocol::ColumnDef {
                name: "id".to_string(),
                ty: SqlType::Int4,
                domain: None,
                default: None,
            },
        }],
        rows,
        check_constraints: Vec::new(),
        foreign_keys: Vec::new(),
    }
}

#[path = "tests/catalog_introspection.rs"]
mod catalog_introspection;
#[path = "tests/catalog_metadata.rs"]
mod catalog_metadata;
#[path = "tests/copy_dml.rs"]
mod copy_dml;
#[path = "tests/cursor_prepare.rs"]
mod cursor_prepare;
#[path = "tests/extended_bind.rs"]
mod extended_bind;
#[path = "tests/extended_lifecycle.rs"]
mod extended_lifecycle;
#[path = "tests/shared_catalog.rs"]
mod shared_catalog;
