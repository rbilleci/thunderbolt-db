use super::*;

#[test]
fn extended_bind_rejects_named_duplicate_portals_and_replaces_unnamed_portal() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query.clone());
    session.replace_extended_portal(
        "named_portal".to_string(),
        Portal {
            statement_name: "lookup".to_string(),
            query: query.clone(),
            parameters: vec![Some("1".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "named_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"2".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert_eq!(
        session
            .portals
            .get("named_portal")
            .and_then(|portal| portal.parameters.first())
            .cloned(),
        Some(Some("1".to_string()))
    );

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        String::new(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"2".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert_eq!(
        session
            .portals
            .get("")
            .and_then(|portal| portal.parameters.first())
            .cloned(),
        Some(Some("2".to_string()))
    );
}

#[test]
fn extended_bind_reports_duplicate_portal_before_payload_errors() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query.clone());
    session.replace_extended_portal(
        "lookup_portal".to_string(),
        Portal {
            statement_name: "lookup".to_string(),
            query,
            parameters: vec![Some("1".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "lookup_portal".to_string(),
        "lookup".to_string(),
        vec![1],
        vec![None],
        vec![1]
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("42P03".to_string())
    );
    assert_eq!(
        session
            .portals
            .get("lookup_portal")
            .and_then(|portal| portal.parameters.first())
            .cloned(),
        Some(Some("1".to_string()))
    );

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "fresh_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"2".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("fresh_portal"));
}

#[test]
fn extended_bind_rejects_invalid_utf8_text_parameters_without_installing_portal() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE name = $1".to_string(),
        parameter_type_oids: vec![25],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "lookup_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(vec![0xff])],
        Vec::new()
    )
    .unwrap());

    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("lookup_portal"));
}

#[test]
fn extended_bind_rejects_null_parameters_without_installing_portal() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE name = $1".to_string(),
        parameter_type_oids: vec![25],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "lookup_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![None],
        Vec::new()
    )
    .unwrap());

    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("lookup_portal"));
}

#[test]
fn extended_bind_reports_missing_statement_before_format_errors() {
    let mut session = Session::default();
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "missing_portal".to_string(),
        "missing_stmt".to_string(),
        vec![1],
        vec![Some(b"1".to_vec())],
        vec![1]
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("26000".to_string())
    );
    assert!(!session.portals.contains_key("missing_portal"));

    session.replace_extended_statement(
        "lookup".to_string(),
        PreparedQuery {
            query: "SELECT id FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![23],
        },
    );
    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "lookup_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("lookup_portal"));
}

#[test]
fn extended_bind_accepts_binary_int4_parameters_and_recovers() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_statement(
        "lookup".to_string(),
        PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![23],
        },
    );
    let (mut writer, mut reader) = tcp_pair();
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "binary_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: vec![1],
            parameters: vec![Some(1_i32.to_be_bytes().to_vec())],
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!extended_error_pending);
    assert_eq!(
        session
            .portals
            .get("binary_portal")
            .and_then(|portal| portal.parameters.first())
            .cloned(),
        Some(Some("1".to_string()))
    );

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "binary_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "text_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("text_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "text_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_bind_accepts_binary_int4_and_text_result_formats() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_statement(
        "lookup".to_string(),
        PreparedQuery {
            query: "SELECT id, name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![23],
        },
    );
    let (mut writer, mut reader) = tcp_pair();
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "binary_result_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: vec![1, 1],
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!extended_error_pending);
    assert!(session.portals.contains_key("binary_result_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "binary_result_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(
        messages[0].1,
        [
            2_i16.to_be_bytes().as_slice(),
            4_i32.to_be_bytes().as_slice(),
            1_i32.to_be_bytes().as_slice(),
            3_i32.to_be_bytes().as_slice(),
            b"Ada".as_slice(),
        ]
        .concat()
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "text_result_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: vec![0],
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("text_result_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "text_result_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_bind_format_code_count_errors_skip_until_sync_and_recover() {
    for (parameter_format_codes, result_format_codes, expected_message, suffix) in [
        (
            vec![0, 0, 0],
            Vec::new(),
            "bind message has wrong number of parameter format codes",
            "parameter_format_count",
        ),
        (
            Vec::new(),
            vec![0, 0, 0],
            "bind message has wrong number of result format codes",
            "result_format_count",
        ),
    ] {
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
                            domain: None,
                            default: None,
                        },
                    },
                    CatalogColumn {
                        attnum: 2,
                        def: gpu_db_protocol::ColumnDef {
                            name: "name".to_string(),
                            ty: SqlType::Text,
                            domain: None,
                            default: None,
                        },
                    },
                ],
                rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
                check_constraints: Vec::new(),
                foreign_keys: Vec::new(),
            },
        );
        session.replace_extended_statement(
            "lookup".to_string(),
            PreparedQuery {
                query: "SELECT id, name FROM people WHERE id = $1".to_string(),
                parameter_type_oids: vec![23],
            },
        );
        let (mut writer, mut reader) = tcp_pair();
        let mut extended_error_pending = false;

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Bind {
                portal_name: format!("bad_{suffix}_portal"),
                statement_name: "lookup".to_string(),
                parameter_format_codes,
                parameters: vec![Some(b"1".to_vec())],
                result_format_codes,
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 1);
        assert_eq!(messages[0].0, b'E');
        assert_eq!(
            error_field_value(&messages[0].1, b'C'),
            Some("08P01".to_string())
        );
        assert_eq!(
            error_field_value(&messages[0].1, b'M'),
            Some(expected_message.to_string())
        );
        assert!(extended_error_pending);
        assert!(!session
            .portals
            .contains_key(&format!("bad_{suffix}_portal")));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::SimpleQuery(format!("CREATE TABLE skipped_{suffix} (id INT)")),
        )
        .unwrap();
        assert!(!session.tables.contains_key(&format!("skipped_{suffix}")));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Sync,
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
        assert!(!extended_error_pending);

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Bind {
                portal_name: format!("recovered_{suffix}_portal"),
                statement_name: "lookup".to_string(),
                parameter_format_codes: Vec::new(),
                parameters: vec![Some(b"1".to_vec())],
                result_format_codes: vec![0, 0],
            },
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
        assert!(session
            .portals
            .contains_key(&format!("recovered_{suffix}_portal")));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Execute {
                portal_name: format!("recovered_{suffix}_portal"),
                max_rows: 0,
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 2);
        assert_eq!(
            messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            vec![b'D', b'C']
        );
        assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
    }
}

#[test]
fn extended_bind_validates_result_format_code_count_for_select_columns() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let query = PreparedQuery {
        query: "SELECT id, name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_format_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        vec![0, 0, 0]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("bad_format_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "per_column_format_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        vec![0, 0]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("per_column_format_portal"));
}

#[test]
fn extended_bind_validates_result_format_count_for_sql_execute_columns() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup_sql".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT id, name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    session.replace_extended_statement(
        "lookup_exec".to_string(),
        PreparedQuery {
            query: "EXECUTE lookup_sql($1)".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_sql_execute_format_portal".to_string(),
        "lookup_exec".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        vec![0, 0, 0]
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("08P01".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("bind message has wrong number of result format codes".to_string())
    );
    assert!(!session
        .portals
        .contains_key("bad_sql_execute_format_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "good_sql_execute_format_portal".to_string(),
        "lookup_exec".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        vec![0, 0]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session
        .portals
        .contains_key("good_sql_execute_format_portal"));
}

#[test]
fn extended_bind_validates_parameter_format_code_count() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1 AND name = $2".to_string(),
        parameter_type_oids: vec![23, 25],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_parameter_format_portal".to_string(),
        "lookup".to_string(),
        vec![0, 0, 0],
        vec![Some(b"1".to_vec()), Some(b"alice".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("bad_parameter_format_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "per_parameter_format_portal".to_string(),
        "lookup".to_string(),
        vec![0, 0],
        vec![Some(b"1".to_vec()), Some(b"alice".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("per_parameter_format_portal"));
}

#[test]
fn extended_bind_rejects_parameter_count_mismatch_without_installing_portal() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1 AND name = $2".to_string(),
        parameter_type_oids: vec![23, 25],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_count_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        Vec::new()
    )
    .unwrap());

    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("bad_count_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "good_count_portal".to_string(),
        "lookup".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec()), Some(b"Ada".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("good_count_portal"));
}

#[test]
fn extended_portal_execute_max_rows_suspends_and_resumes_select_portal() {
    let mut session = Session::default();
    session.portals.insert(
        "people_portal".to_string(),
        Portal {
            statement_name: "people_stmt".to_string(),
            query: PreparedQuery {
                query: "SELECT id FROM people ORDER BY id".to_string(),
                parameter_type_oids: Vec::new(),
            },
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
            described: false,
            result: Some(SelectResult {
                columns: vec![int4_column("id")],
                rows: vec![
                    vec![Some("1".to_string())],
                    vec![Some("2".to_string())],
                    vec![Some("3".to_string())],
                ],
            }),
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_portal_batch(&mut writer, &mut session, "people_portal", 2).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'D', b'D', b's']);
    let portal = session.portals.get("people_portal").unwrap();
    assert!(!portal.described);
    assert_eq!(portal.position, 2);

    execute_portal_batch(&mut writer, &mut session, "people_portal", 0).unwrap();
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 3\0".to_vec());
    let portal = session.portals.get("people_portal").unwrap();
    assert_eq!(portal.position, 3);
    assert!(portal.completed);

    execute_portal_batch(&mut writer, &mut session, "people_portal", 1).unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"SELECT 0\0".to_vec());
}

#[test]
fn extended_describe_portal_after_max_rows_suspend_preserves_position() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
                vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_portal(
        "people_portal".to_string(),
        Portal {
            statement_name: "people_stmt".to_string(),
            query: PreparedQuery {
                query: "SELECT id, name FROM people ORDER BY id".to_string(),
                parameter_type_oids: Vec::new(),
            },
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_execute(&mut writer, &mut session, "people_portal", 2).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'D', b'D', b's']);
    let portal = session.portals.get("people_portal").unwrap();
    assert!(!portal.described);
    assert_eq!(portal.position, 2);

    assert!(!handle_describe(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "people_portal"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'T']);
    let portal = session.portals.get("people_portal").unwrap();
    assert!(portal.described);
    assert_eq!(portal.position, 2);

    assert!(!handle_execute(&mut writer, &mut session, "people_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 3\0".to_vec());
    let portal = session.portals.get("people_portal").unwrap();
    assert_eq!(portal.position, 3);
    assert!(portal.completed);
}

#[test]
fn extended_describe_nodata_portal_marks_portal_described() {
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
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
            }],
            rows: vec![vec![SqlValue::Int4(1)]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_portal(
        "cursor_declare_portal".to_string(),
        Portal {
            statement_name: "cursor_declare_stmt".to_string(),
            query: PreparedQuery {
                query: "DECLARE described_cursor CURSOR FOR SELECT id FROM people".to_string(),
                parameter_type_oids: Vec::new(),
            },
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_describe(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "cursor_declare_portal"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'n']);
    let portal = session.portals.get("cursor_declare_portal").unwrap();
    assert!(portal.described);
    assert!(portal.result.is_none());
    assert_eq!(portal.position, 0);

    assert!(!handle_execute(&mut writer, &mut session, "cursor_declare_portal", 0).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(session.cursors.contains_key("described_cursor"));
}

#[test]
fn extended_execute_zero_max_rows_exhausts_portal_state() {
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
                    domain: None,
                    default: None,
                },
            }],
            rows: vec![
                vec![SqlValue::Int4(1)],
                vec![SqlValue::Int4(2)],
                vec![SqlValue::Int4(3)],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.portals.insert(
        "people_portal".to_string(),
        Portal {
            statement_name: "people_stmt".to_string(),
            query: PreparedQuery {
                query: "SELECT id FROM people ORDER BY id".to_string(),
                parameter_type_oids: Vec::new(),
            },
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_execute(&mut writer, &mut session, "people_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 4);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'D', b'D', b'C']
    );
    assert_eq!(messages[3].1, b"SELECT 3\0".to_vec());
    let portal = session.portals.get("people_portal").unwrap();
    assert!(!portal.described);
    assert!(portal.result.is_some());
    assert_eq!(portal.position, 3);

    assert!(!handle_execute(&mut writer, &mut session, "people_portal", 1).unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"SELECT 0\0".to_vec());
    assert_eq!(session.portals.get("people_portal").unwrap().position, 3);
}

#[test]
fn extended_execute_leaves_row_description_to_explicit_describe() {
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
                    ty: SqlType::Int4,
                    domain: None,
                    default: None,
                },
            }],
            rows: vec![vec![SqlValue::Int4(1)]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.replace_extended_portal(
        "people_portal".to_string(),
        Portal {
            statement_name: "people_stmt".to_string(),
            query: PreparedQuery {
                query: "SELECT id FROM people".to_string(),
                parameter_type_oids: Vec::new(),
            },
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_describe(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "people_portal"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'T']);
    assert!(session.portals.get("people_portal").unwrap().described);

    assert!(!handle_execute(&mut writer, &mut session, "people_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
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
fn extended_describe_sql_execute_uses_prepared_select_columns() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "EXECUTE lookup(1)".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);

    assert!(!handle_describe(&mut writer, &mut session, DescribeTarget::Statement, "").unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b't', b'T']);

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "exec_portal".to_string(),
        String::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_execute(&mut writer, &mut session, "exec_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_sql_execute_bind_infers_sql_prepared_parameter_types() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: vec![vec![SqlValue::Int4(2), SqlValue::Text("Ada".to_string())]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1 AND name = $2".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid(), SqlType::Text.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "EXECUTE lookup($1, $2)".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    let query = match session.prepared.get("") {
        Some(PreparedStatement::Extended(query)) => query,
        _ => panic!("expected unnamed extended prepared statement"),
    };
    assert_eq!(
        query.parameter_type_oids,
        vec![SqlType::Int4.postgres_oid(), SqlType::Text.postgres_oid()]
    );

    assert!(!handle_describe(&mut writer, &mut session, DescribeTarget::Statement, "").unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b't', b'T']);

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "exec_portal".to_string(),
        String::new(),
        Vec::new(),
        vec![Some(b"2".to_vec()), Some(b"Ada".to_vec())],
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_execute(&mut writer, &mut session, "exec_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_sql_execute_bind_revalidates_sql_prepared_inferred_types() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT id, name FROM people WHERE name = $1 LIMIT $2".to_string(),
            parameter_type_oids: vec![SqlType::Text.postgres_oid(), SqlType::Int4.postgres_oid()],
        }),
    );
    session.replace_extended_statement(
        "lookup_exec".to_string(),
        PreparedQuery {
            query: "EXECUTE lookup($1, $2)".to_string(),
            parameter_type_oids: vec![0, 0],
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_sql_execute_bind_portal".to_string(),
        "lookup_exec".to_string(),
        Vec::new(),
        vec![Some(b"Ada".to_vec()), Some(b"not-an-int".to_vec())],
        Vec::new(),
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("22P02".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("invalid input syntax for parameter type oid 23: \"not-an-int\"".to_string())
    );
    assert!(!session.portals.contains_key("bad_sql_execute_bind_portal"));
}

#[test]
fn extended_sql_execute_describe_reports_literal_parameter_errors() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "EXECUTE lookup('not-an-int')".to_string(),
        Vec::new(),
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert!(String::from_utf8_lossy(&messages[0].1)
        .contains("invalid input syntax for parameter type oid 23: \"not-an-int\""));
    assert!(!session.prepared.contains_key(""));
}

#[test]
fn extended_cursor_sql_execute_describe_reports_literal_parameter_errors() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "DECLARE lookup_cursor CURSOR FOR EXECUTE lookup('not-an-int')".to_string(),
        Vec::new(),
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert!(String::from_utf8_lossy(&messages[0].1)
        .contains("invalid input syntax for parameter type oid 23: \"not-an-int\""));
    assert!(!session.prepared.contains_key(""));
    assert!(!session.cursors.contains_key("lookup_cursor"));
}

#[test]
fn extended_sql_execute_rejects_conflicting_reused_placeholder_types() {
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
                        domain: None,
                        default: None,
                    },
                },
                CatalogColumn {
                    attnum: 2,
                    def: gpu_db_protocol::ColumnDef {
                        name: "name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1 AND name = $2".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid(), SqlType::Text.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "EXECUTE lookup($1, $1)".to_string(),
        Vec::new(),
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert!(String::from_utf8_lossy(&messages[0].1)
        .contains("inconsistent parameter types for SQL EXECUTE placeholder"));
    assert!(!session.prepared.contains_key(""));
}

#[test]
fn extended_sql_execute_sparse_outer_placeholders_skip_unused_binds() {
    let query = PreparedQuery {
        query: "EXECUTE lookup($2, 'Ada')".to_string(),
        parameter_type_oids: vec![0, SqlType::Int4.postgres_oid()],
    };

    let bound = bind_query_parameters(
        &query,
        &[
            Some("not-an-int-and-not-used".to_string()),
            Some("2".to_string()),
        ],
    )
    .unwrap();
    assert_eq!(bound, "EXECUTE lookup(2, 'Ada')");

    assert_eq!(
        bind_query_parameters(
            &query,
            &[
                Some("still-not-used".to_string()),
                Some("not-an-int".to_string()),
            ],
        ),
        Err(BindParameterError::InvalidTextRepresentation {
            oid: SqlType::Int4.postgres_oid(),
            value: "not-an-int".to_string(),
        })
    );
}
