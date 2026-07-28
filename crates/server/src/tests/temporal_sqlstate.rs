//! Simple-query temporal parse SQLSTATE coverage; extended Bind coverage lives beside its codec.

use super::*;

fn binary_bind_payload(portal: &str, statement: &str, value: &[u8]) -> Vec<u8> {
    let mut payload = Vec::new();
    push_cstring(&mut payload, portal);
    push_cstring(&mut payload, statement);
    payload.extend_from_slice(&1_i16.to_be_bytes());
    payload.extend_from_slice(&1_i16.to_be_bytes());
    payload.extend_from_slice(&1_i16.to_be_bytes());
    payload.extend_from_slice(&i32::try_from(value.len()).unwrap().to_be_bytes());
    payload.extend_from_slice(value);
    payload.extend_from_slice(&0_i16.to_be_bytes());
    payload
}

fn assert_single_text_data_row(messages: &[(u8, Vec<u8>)], expected: &str) {
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C', b'Z']
    );
    let mut row = Vec::new();
    row.extend_from_slice(&1_i16.to_be_bytes());
    row.extend_from_slice(&i32::try_from(expected.len()).unwrap().to_be_bytes());
    row.extend_from_slice(expected.as_bytes());
    assert_eq!(messages[1].1, row, "default result format is text");
}

#[test]
fn simple_query_default_assignment_mismatches_return_42804() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    for sql in [
        "CREATE TABLE wire_default_bool (value BOOL DEFAULT 1)",
        "CREATE TABLE wire_default_null (value INT DEFAULT NULL::text)",
    ] {
        client
            .write_all(&tagged(b'Q', &query_payload(sql)))
            .unwrap();
        assert_error_sqlstate(&read_messages(&mut client, 2), b"C42804\0");
    }

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE wire_default_alter (id INT, value BOOL)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    for sql in [
        "ALTER TABLE wire_default_alter ADD COLUMN added BOOL DEFAULT 1",
        "ALTER TABLE wire_default_alter ADD COLUMN added_null INT DEFAULT NULL::text",
        "ALTER TABLE wire_default_alter ALTER COLUMN value SET DEFAULT 1",
        "ALTER TABLE wire_default_alter ALTER COLUMN id SET DEFAULT NULL::text",
    ] {
        client
            .write_all(&tagged(b'Q', &query_payload(sql)))
            .unwrap();
        assert_error_sqlstate(&read_messages(&mut client, 2), b"C42804\0");
    }

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
fn simple_query_sequence_default_target_errors_preserve_42p01_before_42804() {
    const MISSING: &str = "wire_missing_default_sequence";
    const EXISTING: &str = "wire_existing_default_sequence";
    const NON_SEQUENCE: &str = "wire_non_sequence_default_target";

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    for sql in [
        format!("CREATE SEQUENCE {EXISTING}"),
        format!("CREATE TABLE {NON_SEQUENCE} (id INT)"),
        "CREATE TABLE wire_sequence_add (id INT)".to_string(),
        "CREATE TABLE wire_sequence_alter (missing_text TEXT, existing_bool BOOLEAN, missing_int INT, non_sequence_bool BOOLEAN)"
            .to_string(),
    ] {
        client
            .write_all(&tagged(b'Q', &query_payload(&sql)))
            .unwrap();
        assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z'], "{sql}");
    }

    for (sql, state) in [
        (
            format!(
                "CREATE TABLE wire_sequence_create_missing_text (value TEXT DEFAULT nextval('{MISSING}'::regclass))"
            ),
            b"C42P01\0".as_slice(),
        ),
        (
            format!(
                "ALTER TABLE wire_sequence_add ADD COLUMN missing_text TEXT DEFAULT nextval('{MISSING}'::regclass)"
            ),
            b"C42P01\0".as_slice(),
        ),
        (
            format!(
                "ALTER TABLE wire_sequence_alter ALTER COLUMN missing_text SET DEFAULT nextval('{MISSING}'::regclass)"
            ),
            b"C42P01\0".as_slice(),
        ),
        (
            format!(
                "CREATE TABLE wire_sequence_create_missing_int (value INT DEFAULT nextval('{MISSING}'::regclass))"
            ),
            b"C42P01\0".as_slice(),
        ),
        (
            format!(
                "ALTER TABLE wire_sequence_add ADD COLUMN missing_int INT DEFAULT nextval('{MISSING}'::regclass)"
            ),
            b"C42P01\0".as_slice(),
        ),
        (
            format!(
                "ALTER TABLE wire_sequence_alter ALTER COLUMN missing_int SET DEFAULT nextval('{MISSING}'::regclass)"
            ),
            b"C42P01\0".as_slice(),
        ),
        (
            format!(
                "CREATE TABLE wire_sequence_create_existing_bool (value BOOLEAN DEFAULT nextval('{EXISTING}'::regclass))"
            ),
            b"C42804\0".as_slice(),
        ),
        (
            format!(
                "ALTER TABLE wire_sequence_add ADD COLUMN existing_bool BOOLEAN DEFAULT nextval('{EXISTING}'::regclass)"
            ),
            b"C42804\0".as_slice(),
        ),
        (
            format!(
                "ALTER TABLE wire_sequence_alter ALTER COLUMN existing_bool SET DEFAULT nextval('{EXISTING}'::regclass)"
            ),
            b"C42804\0".as_slice(),
        ),
        (
            format!(
                "CREATE TABLE wire_sequence_create_non_sequence_bool (value BOOLEAN DEFAULT nextval('{NON_SEQUENCE}'::regclass))"
            ),
            b"C42804\0".as_slice(),
        ),
        (
            format!(
                "ALTER TABLE wire_sequence_add ADD COLUMN non_sequence_bool BOOLEAN DEFAULT nextval('{NON_SEQUENCE}'::regclass)"
            ),
            b"C42804\0".as_slice(),
        ),
        (
            format!(
                "ALTER TABLE wire_sequence_alter ALTER COLUMN non_sequence_bool SET DEFAULT nextval('{NON_SEQUENCE}'::regclass)"
            ),
            b"C42804\0".as_slice(),
        ),
        (
            format!(
                "CREATE TABLE wire_sequence_order_mismatch_first (a BOOLEAN DEFAULT 1, b TEXT DEFAULT nextval('{MISSING}'::regclass))"
            ),
            b"C42804\0".as_slice(),
        ),
        (
            format!(
                "CREATE TABLE wire_sequence_order_missing_first (a TEXT DEFAULT nextval('{MISSING}'::regclass), b BOOLEAN DEFAULT 1)"
            ),
            b"C42P01\0".as_slice(),
        ),
    ] {
        client
            .write_all(&tagged(b'Q', &query_payload(&sql)))
            .unwrap();
        assert_error_sqlstate(&read_messages(&mut client, 2), state);
    }

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
fn simple_query_add_column_target_errors_precede_default_sqlstates() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE wire_add_default_target (id INT)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    for (sql, state) in [
        (
            "ALTER TABLE wire_missing_add_default_target ADD COLUMN value BOOLEAN DEFAULT 1",
            b"C42P01\0".as_slice(),
        ),
        (
            "ALTER TABLE wire_add_default_target ADD COLUMN id BOOLEAN DEFAULT 1",
            b"C42701\0".as_slice(),
        ),
        (
            "ALTER TABLE wire_add_default_target ADD COLUMN id INT DEFAULT nextval('wire_missing_duplicate_default_sequence'::regclass)",
            b"C42701\0".as_slice(),
        ),
    ] {
        client
            .write_all(&tagged(b'Q', &query_payload(sql)))
            .unwrap();
        assert_error_sqlstate(&read_messages(&mut client, 2), state);
    }

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
fn extended_ddl_default_assignment_mismatch_returns_42804_and_sync_recovers() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE wire_extended_non_sequence_default_target (id INT)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    let mut request = Vec::new();
    request.extend(tagged(
        b'P',
        &parse_payload(
            "extended_default_mismatch",
            "CREATE TABLE extended_default_mismatch (a BOOL DEFAULT nextval('wire_extended_non_sequence_default_target'::regclass), b TEXT DEFAULT nextval('wire_extended_later_missing_default_sequence'::regclass))",
            &[],
        ),
    ));
    request.extend(tagged(
        b'B',
        &bind_payload(
            "extended_default_mismatch_portal",
            "extended_default_mismatch",
        ),
    ));
    request.extend(tagged(
        b'E',
        &execute_payload("extended_default_mismatch_portal", 0),
    ));
    request.extend(tagged(b'S', &[]));
    client.write_all(&request).unwrap();
    let messages = read_messages(&mut client, 4);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'1', b'2', b'E', b'Z']
    );
    assert_error_sqlstate(&messages[2..], b"C42804\0");
    assert_eq!(messages[3].1, vec![b'I'], "Sync restores idle state");

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE extended_default_recovered (id INT)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
fn extended_ddl_missing_sequence_default_returns_42p01_and_sync_recovers() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    let mut request = Vec::new();
    request.extend(tagged(
        b'P',
        &parse_payload(
            "extended_missing_default_sequence",
            "CREATE TABLE extended_missing_default_sequence (value TEXT DEFAULT nextval('wire_extended_missing_default_sequence'::regclass), later BOOLEAN DEFAULT 1)",
            &[],
        ),
    ));
    request.extend(tagged(
        b'B',
        &bind_payload(
            "extended_missing_default_sequence_portal",
            "extended_missing_default_sequence",
        ),
    ));
    request.extend(tagged(
        b'E',
        &execute_payload("extended_missing_default_sequence_portal", 0),
    ));
    request.extend(tagged(b'S', &[]));
    client.write_all(&request).unwrap();
    let messages = read_messages(&mut client, 4);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'1', b'2', b'E', b'Z']
    );
    assert_error_sqlstate(&messages[2..], b"C42P01\0");
    assert_eq!(messages[3].1, vec![b'I'], "Sync restores idle state");

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE extended_missing_default_recovered (id INT)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
fn simple_query_temporal_check_preserves_format_and_field_overflow_sqlstates() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);

    for (sql, state) in [
        (
            "CREATE TABLE simple_bad_datetime (d date, CHECK (d > 'not-a-date'))",
            b"C22007\0".as_slice(),
        ),
        (
            "CREATE TABLE simple_datetime_field (t timestamp, CHECK (t > '2024-01-01 25:00:00'))",
            b"C22008\0".as_slice(),
        ),
        (
            "CREATE TABLE simple_datetime_carrier (t timestamp, \
             CHECK (t > '294276-12-31 23:59:60'::timestamp))",
            b"C22008\0".as_slice(),
        ),
        (
            "CREATE TABLE simple_trailing_timestamp (t timestamp, CHECK (t > '2024-01-01T'))",
            b"C22007\0".as_slice(),
        ),
        (
            "CREATE TABLE simple_time_field_precedence \
             (t timestamp, CHECK (t > '2000-01-01 25:00:00.abc'))",
            b"C22008\0".as_slice(),
        ),
        (
            "CREATE TABLE simple_later_token_precedence \
             (t timestamp, CHECK (t > '2000-02-30 00:00:00.abc'))",
            b"C22007\0".as_slice(),
        ),
        (
            "CREATE TABLE simple_deferred_date_validation \
             (t timestamp, CHECK (t > '2000-02-30 00:00:00'))",
            b"C22008\0".as_slice(),
        ),
        (
            "CREATE TABLE simple_adjacent_t_numeric_precedence \
             (t timestamp, CHECK (t > '2000-01-999999999999999999999T'))",
            b"C22008\0".as_slice(),
        ),
        (
            "CREATE TABLE simple_adjacent_t_calendar_precedence \
             (t timestamp, CHECK (t > '2000-02-30T'))",
            b"C22007\0".as_slice(),
        ),
    ] {
        client
            .write_all(&tagged(b'Q', &query_payload(sql)))
            .unwrap();
        assert_error_sqlstate(&read_messages(&mut client, 2), state);
    }

    let workspace_timestamp = format!("2000-01-01 00:00:00.5{}", "0".repeat(131));
    client
        .write_all(&tagged(
            b'Q',
            &query_payload(&format!(
                "CREATE TABLE simple_workspace_boundary \
                 (t timestamp, CHECK (t >= '{workspace_timestamp}'::timestamp))"
            )),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    let workspace_timestamp_over = format!("2000-01-01 00:00:00.5{}", "0".repeat(132));
    client
        .write_all(&tagged(
            b'Q',
            &query_payload(&format!(
                "CREATE TABLE simple_workspace_over \
                 (t timestamp, CHECK (t >= '{workspace_timestamp_over}'::timestamp))"
            )),
        ))
        .unwrap();
    assert_error_sqlstate(&read_messages(&mut client, 2), b"C22007\0");

    for (table, literal) in [
        ("simple_midnight_24", "2024-01-01 24:00:00.0000000"),
        ("simple_midnight_leap", "2024-01-01 23:59:60.0000000"),
        ("simple_leap_fraction", "2024-01-01 10:00:60.0000005000001"),
        ("simple_leap_binary_midpoint", "2024-01-01 10:00:60.0001255"),
        ("simple_bare_fraction", "2024-01-01 00:00:00."),
        ("simple_two_field_fraction", "2024-01-01 10:20.5"),
        ("simple_two_field_fraction_edge", "2024-01-01 23:59.5"),
    ] {
        client
            .write_all(&tagged(
                b'Q',
                &query_payload(&format!(
                    "CREATE TABLE {table} (t timestamp, CHECK (t >= '{literal}'::timestamp))"
                )),
            ))
            .unwrap();
        assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    }

    for sql in [
        "CREATE TABLE simple_bc_date (d date, CHECK (d >= '4714-11-24 BC'::date))",
        "CREATE TABLE simple_bc_timestamp \
         (t timestamp, CHECK (t >= '4714-11-24 00:00:00 BC'::timestamp))",
    ] {
        client
            .write_all(&tagged(b'Q', &query_payload(sql)))
            .unwrap();
        assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    }

    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}

#[test]
fn raw_extended_binary_temporal_boundaries_return_22008_and_sync_recovers() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let engine = SharedEngine::new();
        handle_connection(&mut stream, &engine).unwrap();
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(&startup_frame()).unwrap();
    let _ = read_messages(&mut client, 9);
    for sql in [
        "CREATE TABLE binary_date_wire (d date)",
        "CREATE TABLE binary_timestamp_wire (t timestamp)",
    ] {
        client
            .write_all(&tagged(b'Q', &query_payload(sql)))
            .unwrap();
        assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    }

    for (statement, sql, oid, value) in [
        (
            "binary_date_boundary",
            "INSERT INTO binary_date_wire (d) VALUES ($1)",
            1082_u32,
            i32::MAX.to_be_bytes().to_vec(),
        ),
        (
            "binary_timestamp_boundary",
            "INSERT INTO binary_timestamp_wire (t) VALUES ($1)",
            1114_u32,
            9_223_371_331_200_000_000_i64.to_be_bytes().to_vec(),
        ),
    ] {
        client
            .write_all(&tagged(b'P', &parse_payload(statement, sql, &[oid])))
            .unwrap();
        assert_eq!(read_tags(&mut client, 1), vec![b'1']);
        let mut bind_and_sync = tagged(b'B', &binary_bind_payload(statement, statement, &value));
        bind_and_sync.extend(tagged(b'S', &[]));
        client.write_all(&bind_and_sync).unwrap();
        let messages = read_messages(&mut client, 2);
        assert_error_sqlstate(&messages, b"C22008\0");
        assert_eq!(messages[1].1, vec![b'I'], "Sync restores idle state");
    }

    for (statement, sql, oid, value) in [
        (
            "binary_date_lower",
            "INSERT INTO binary_date_wire (d) VALUES ($1)",
            1082_u32,
            (-2_451_545_i32).to_be_bytes().to_vec(),
        ),
        (
            "binary_timestamp_lower",
            "INSERT INTO binary_timestamp_wire (t) VALUES ($1)",
            1114_u32,
            (-211_813_488_000_000_000_i64).to_be_bytes().to_vec(),
        ),
    ] {
        client
            .write_all(&tagged(b'P', &parse_payload(statement, sql, &[oid])))
            .unwrap();
        assert_eq!(read_tags(&mut client, 1), vec![b'1']);
        let mut execute_and_sync = tagged(b'B', &binary_bind_payload(statement, statement, &value));
        execute_and_sync.extend(tagged(b'E', &execute_payload(statement, 0)));
        execute_and_sync.extend(tagged(b'S', &[]));
        client.write_all(&execute_and_sync).unwrap();
        assert_eq!(read_tags(&mut client, 3), vec![b'2', b'C', b'Z']);
    }

    for (sql, expected) in [
        ("SELECT d FROM binary_date_wire", "4714-11-24 BC"),
        (
            "SELECT t FROM binary_timestamp_wire",
            "4714-11-24 00:00:00 BC",
        ),
    ] {
        client
            .write_all(&tagged(b'Q', &query_payload(sql)))
            .unwrap();
        assert_single_text_data_row(&read_messages(&mut client, 4), expected);
    }

    client
        .write_all(&tagged(
            b'Q',
            &query_payload("CREATE TABLE binary_temporal_recovered (id int4)"),
        ))
        .unwrap();
    assert_eq!(read_tags(&mut client, 2), vec![b'C', b'Z']);
    client.write_all(&tagged(b'X', &[])).unwrap();
    drop(client);
    server.join().unwrap();
}
