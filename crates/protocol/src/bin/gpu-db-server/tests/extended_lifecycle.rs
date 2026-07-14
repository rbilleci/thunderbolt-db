use super::*;

#[test]
fn extended_parameter_binding_substitutes_text_and_int_literals() {
    let query = PreparedQuery {
        query: "SELECT id, name FROM people WHERE id = $1 ORDER BY name LIMIT $2".to_string(),
        parameter_type_oids: vec![23, 23],
    };

    assert_eq!(
        bind_query_parameters(&query, &[Some("2".to_string()), Some("1".to_string())]),
        Ok("SELECT id, name FROM people WHERE id = 2 ORDER BY name LIMIT 1".to_string())
    );

    let text_query = PreparedQuery {
        query: "SELECT id FROM people WHERE name = $1".to_string(),
        parameter_type_oids: vec![25],
    };
    assert_eq!(
        bind_query_parameters(&text_query, &[Some("O'Brien".to_string())]),
        Ok("SELECT id FROM people WHERE name = 'O''Brien'".to_string())
    );
    assert_eq!(
        bind_query_parameters(&text_query, &[Some("Ada $1".to_string())]),
        Ok("SELECT id FROM people WHERE name = 'Ada $1'".to_string())
    );
}

#[test]
fn extended_parameter_binding_replaces_exact_placeholder_tokens() {
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1 OR id = $10 ORDER BY id LIMIT $11".to_string(),
        parameter_type_oids: vec![23; 11],
    };
    let parameters = (1..=11)
        .map(|idx| Some(idx.to_string()))
        .collect::<Vec<_>>();

    assert_eq!(
        replace_unquoted_placeholder(&query.query, "$1", "7"),
        "SELECT id FROM people WHERE id = 7 OR id = $10 ORDER BY id LIMIT $11"
    );
    assert_eq!(
        bind_query_parameters(&query, &parameters),
        Ok("SELECT id FROM people WHERE id = 1 OR id = 10 ORDER BY id LIMIT 11".to_string())
    );
}

#[test]
fn extended_parameter_binding_matches_zero_padded_nonzero_placeholders() {
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $01 OR id = $002 ORDER BY id LIMIT $0003"
            .to_string(),
        parameter_type_oids: vec![23, 23, 23],
    };

    assert_eq!(max_placeholder_index(&query.query), 3);
    assert!(!contains_zero_placeholder(&query.query));
    assert_eq!(
        replace_unquoted_placeholder(&query.query, "$1", "7"),
        "SELECT id FROM people WHERE id = 7 OR id = $002 ORDER BY id LIMIT $0003"
    );
    assert_eq!(
        bind_query_parameters(
            &query,
            &[
                Some("1".to_string()),
                Some("2".to_string()),
                Some("2".to_string()),
            ],
        ),
        Ok("SELECT id FROM people WHERE id = 1 OR id = 2 ORDER BY id LIMIT 2".to_string())
    );
}

#[test]
fn extended_parameter_binding_ignores_quoted_placeholder_literals() {
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE name = '$1' OR name = 'O''$2' OR id = $1".to_string(),
        parameter_type_oids: vec![23],
    };

    assert_eq!(max_placeholder_index(&query.query), 1);
    assert_eq!(expected_parameter_count(&query), 1);
    assert_eq!(
        bind_query_parameters(&query, &[Some("2".to_string())]),
        Ok("SELECT id FROM people WHERE name = '$1' OR name = 'O''$2' OR id = 2".to_string())
    );
    assert_eq!(
        replace_parameter_placeholders_with_dummy_literals(
            "SELECT id FROM people WHERE name = '$1' OR id = $1"
        ),
        "SELECT id FROM people WHERE name = '$1' OR id = 1"
    );
}

#[test]
fn extended_parameter_binding_ignores_commented_placeholder_literals() {
    let query = strip_sql_comments(
        "SELECT id FROM people WHERE id = $1 -- ignored $2\n\
         ORDER BY id /* ignored $3 */ LIMIT $2",
    );
    let prepared = PreparedQuery {
        query,
        parameter_type_oids: vec![23, 23],
    };

    assert_eq!(max_placeholder_index(&prepared.query), 2);
    assert_eq!(expected_parameter_count(&prepared), 2);
    assert_eq!(
        bind_query_parameters(&prepared, &[Some("2".to_string()), Some("1".to_string())]),
        Ok("SELECT id FROM people WHERE id = 2 \nORDER BY id   LIMIT 1".to_string())
    );
    assert_eq!(
        strip_sql_comments("SELECT '$1 -- still text', id FROM people WHERE id = $1"),
        "SELECT '$1 -- still text', id FROM people WHERE id = $1"
    );
    assert_eq!(
        strip_sql_comments(
            "SELECT $$-- still text$$, $tag$/* still text */$tag$, id FROM people WHERE id = $1"
        ),
        "SELECT $$-- still text$$, $tag$/* still text */$tag$, id FROM people WHERE id = $1"
    );
    assert_eq!(
        strip_sql_comments(r#"SELECT "-- still identifier", id FROM people WHERE id = $1"#),
        r#"SELECT "-- still identifier", id FROM people WHERE id = $1"#
    );
    assert_eq!(
        strip_sql_comments(
            "SELECT id FROM people WHERE id = $1 /* outer $2 /* inner $3 */ done $4 */ LIMIT $2"
        ),
        "SELECT id FROM people WHERE id = $1   LIMIT $2"
    );
}

#[test]
fn extended_parse_rejects_zero_placeholder_without_installing_statement() {
    let mut session = Session::default();
    assert!(contains_zero_placeholder(
        "SELECT name FROM people WHERE id = $0"
    ));
    assert!(contains_zero_placeholder(
        "SELECT name FROM people WHERE id = $00"
    ));
    assert!(!contains_zero_placeholder(
        "SELECT '$0' AS literal, name FROM people WHERE id = $1"
    ));
    assert!(!contains_zero_placeholder(
        "SELECT name FROM people WHERE id = $01"
    ));
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
    let (mut writer, mut reader) = tcp_pair();
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Parse {
            statement_name: "bad_zero".to_string(),
            query: "SELECT name FROM people WHERE id = $0".to_string(),
            parameter_type_oids: Vec::new(),
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("42P02".to_string())
    );
    assert!(!session.prepared.contains_key("bad_zero"));
    assert!(extended_error_pending);

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "skipped_zero".to_string(),
            statement_name: "bad_zero".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert!(!session.portals.contains_key("skipped_zero"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "good".to_string(),
        "SELECT name FROM people WHERE id = $1".to_string(),
        vec![23]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(session.prepared.contains_key("good"));
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
        Ok("SELECT id FROM people WHERE id = 2".to_string())
    );
    assert_eq!(
        bind_query_parameters(&inferred_query, &[]),
        Err(BindParameterError::CountMismatch)
    );
    assert_eq!(
        bind_query_parameters(
            &inferred_query,
            &[Some("2".to_string()), Some("extra".to_string())]
        ),
        Err(BindParameterError::CountMismatch)
    );

    let typed_query = PreparedQuery {
        query: "SELECT id FROM people".to_string(),
        parameter_type_oids: vec![23],
    };
    assert_eq!(expected_parameter_count(&typed_query), 1);
    assert_eq!(
        bind_query_parameters(&typed_query, &[]),
        Err(BindParameterError::CountMismatch)
    );
}

#[test]
fn extended_parse_infers_supported_parameter_types_from_select_shape() {
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
        resolve_prepared_parameter_type_oids(
            &session,
            "SELECT id FROM people WHERE id > $1 AND name = $2 LIMIT $3",
            Vec::new(),
        ),
        vec![23, 25, 23]
    );
    assert_eq!(
        resolve_prepared_parameter_type_oids(
            &session,
            "SELECT id FROM people WHERE $1 <= id AND $2 = name LIMIT $3",
            Vec::new(),
        ),
        vec![23, 25, 23]
    );
    assert_eq!(
        resolve_prepared_parameter_type_oids(
            &session,
            "SELECT id FROM people WHERE id = $1",
            vec![25],
        ),
        vec![25]
    );
}

#[test]
fn extended_bind_rejects_invalid_values_for_inferred_int4_parameters() {
    let query = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };

    assert_eq!(
        bind_query_parameters(&query, &[Some("not-an-int".to_string())]),
        Err(BindParameterError::InvalidTextRepresentation {
            oid: 23,
            value: "not-an-int".to_string(),
        })
    );
}

#[test]
fn extended_bind_rejects_invalid_int4_values_without_installing_portal() {
    let mut session = Session::default();
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query);
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_int4_portal".to_string(),
        "lookup".to_string(),
        vec![0],
        vec![Some(b"not-an-int".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.portals.contains_key("bad_int4_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "good_int4_portal".to_string(),
        "lookup".to_string(),
        vec![0],
        vec![Some(b"1".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(session.portals.contains_key("good_int4_portal"));
}

#[test]
fn extended_execute_supports_parameterized_cursor_declarations() {
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
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "".to_string(),
        "DECLARE _psql_cursor NO SCROLL CURSOR FOR SELECT id, name FROM people WHERE id > $1 ORDER BY id".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "".to_string(),
        "".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_execute(&mut writer, &mut session, "", 0).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    let cursor = session.cursors.get("_psql_cursor").unwrap();
    assert_eq!(cursor.rows.len(), 2);
    assert_eq!(
        cursor.rows[0],
        vec![Some("2".to_string()), Some("Linus".to_string())]
    );
}

#[test]
fn extended_cursor_declare_portal_describe_and_close_keep_session_cursor() {
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
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "cursor_stmt".to_string(),
        "DECLARE raw_cursor CURSOR FOR SELECT id, name FROM people WHERE id > $1 ORDER BY id"
            .to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "cursor_portal".to_string(),
        "cursor_stmt".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec())],
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_describe(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "cursor_portal",
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'n']);

    assert!(!handle_execute(&mut writer, &mut session, "cursor_portal", 1).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(session.cursors.contains_key("raw_cursor"));

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "cursor_portal",
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(!session.portals.contains_key("cursor_portal"));
    assert!(session.cursors.contains_key("raw_cursor"));

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Statement,
        "cursor_stmt",
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(!session.prepared.contains_key("cursor_stmt"));
    assert!(session.cursors.contains_key("raw_cursor"));

    execute_statement(
        &mut writer,
        &mut session,
        "FETCH FORWARD 1 FROM raw_cursor",
        true,
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 3);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'T', b'D', b'C']
    );
    assert_eq!(messages[2].1, b"FETCH 1\0".to_vec());
    assert_eq!(session.cursors.get("raw_cursor").unwrap().position, 1);
}

#[test]
fn extended_cursor_declaration_supports_sql_prepared_execute_binds() {
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
    session.prepared.insert(
        "lookup".to_string(),
        PreparedStatement::Sql(PreparedQuery {
            query: "SELECT id, name FROM people WHERE id >= $1 ORDER BY id".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "cursor_exec_stmt".to_string(),
        "DECLARE raw_exec_cursor CURSOR FOR EXECUTE lookup($1)".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    let query = match session.prepared.get("cursor_exec_stmt") {
        Some(PreparedStatement::Extended(query)) => query,
        _ => panic!("expected extended cursor declaration statement"),
    };
    assert_eq!(
        query.parameter_type_oids,
        vec![SqlType::Int4.postgres_oid()]
    );

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "cursor_exec_portal".to_string(),
        "cursor_exec_stmt".to_string(),
        Vec::new(),
        vec![Some(b"2".to_vec())],
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_describe(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "cursor_exec_portal",
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'n']);

    assert!(!handle_execute(&mut writer, &mut session, "cursor_exec_portal", 0).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    let cursor = session.cursors.get("raw_exec_cursor").unwrap();
    assert_eq!(cursor.rows.len(), 2);
    assert_eq!(
        cursor.rows[0],
        vec![Some("2".to_string()), Some("Linus".to_string())]
    );
}

#[test]
fn extended_cursor_sql_execute_parse_rejects_conflicting_explicit_oid() {
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
            query: "SELECT id, name FROM people WHERE id >= $1 ORDER BY id".to_string(),
            parameter_type_oids: vec![SqlType::Int4.postgres_oid()],
        }),
    );
    let (mut writer, mut reader) = tcp_pair();
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Parse {
            statement_name: "cursor_exec_stmt".to_string(),
            query: "DECLARE raw_exec_cursor CURSOR FOR EXECUTE lookup($1)".to_string(),
            parameter_type_oids: vec![SqlType::Text.postgres_oid()],
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("42P08".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("inconsistent parameter types for SQL EXECUTE placeholder".to_string())
    );
    assert!(extended_error_pending);
    assert!(!session.prepared.contains_key("cursor_exec_stmt"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "skipped_cursor_portal".to_string(),
            statement_name: "cursor_exec_stmt".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert!(!session.portals.contains_key("skipped_cursor_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "cursor_exec_stmt".to_string(),
        "DECLARE raw_exec_cursor CURSOR FOR EXECUTE lookup($1)".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    let query = match session.prepared.get("cursor_exec_stmt") {
        Some(PreparedStatement::Extended(query)) => query,
        _ => panic!("expected recovered extended cursor declaration statement"),
    };
    assert_eq!(
        query.parameter_type_oids,
        vec![SqlType::Int4.postgres_oid()]
    );
}

#[test]
fn extended_cursor_declaration_rejects_malformed_result_format_arity() {
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
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "cursor_stmt".to_string(),
        "DECLARE raw_cursor CURSOR FOR SELECT id, name FROM people ORDER BY id".to_string(),
        Vec::new(),
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);

    assert!(handle_bind(
        &mut writer,
        &mut session,
        "bad_cursor_portal".to_string(),
        "cursor_stmt".to_string(),
        Vec::new(),
        Vec::new(),
        vec![0, 0],
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("08P01".to_string())
    );
    assert!(!session.portals.contains_key("bad_cursor_portal"));

    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "good_cursor_portal".to_string(),
        "cursor_stmt".to_string(),
        Vec::new(),
        Vec::new(),
        vec![0],
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    assert!(!handle_execute(&mut writer, &mut session, "good_cursor_portal", 0).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(session.cursors.contains_key("raw_cursor"));
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
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );
    session.portals.insert(
        "other_portal".to_string(),
        Portal {
            statement_name: "other".to_string(),
            query: query.clone(),
            parameters: vec![Some("2".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
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
fn extended_close_accepts_missing_statement_and_portal_names() {
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

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Statement,
        "missing_statement"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(session.prepared.contains_key("lookup"));
    assert!(session.portals.contains_key("lookup_portal"));

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "missing_portal"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(session.prepared.contains_key("lookup"));
    assert!(session.portals.contains_key("lookup_portal"));

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Portal,
        "lookup_portal"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(session.prepared.contains_key("lookup"));
    assert!(!session.portals.contains_key("lookup_portal"));
}

#[test]
fn extended_close_statement_cascades_to_portals_before_execute_recovery() {
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
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
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

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Statement,
        "lookup"
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(!session.prepared.contains_key("lookup"));
    assert!(!session.portals.contains_key("lookup_portal"));

    assert!(handle_execute(&mut writer, &mut session, "lookup_portal", 0).unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "lookup_again".to_string(),
        "SELECT name FROM people WHERE id = $1".to_string(),
        vec![23]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "lookup_again_portal".to_string(),
        "lookup_again".to_string(),
        Vec::new(),
        vec![Some(b"2".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!handle_execute(&mut writer, &mut session, "lookup_again_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 2);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'D', b'C']
    );
    assert_eq!(messages[1].1, b"SELECT 1\0".to_vec());
}

#[test]
fn extended_close_portal_removes_only_portal_before_execute_recovery() {
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
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let query = PreparedQuery {
        query: "SELECT name FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement("lookup".to_string(), query.clone());
    session.replace_extended_portal(
        "closed_portal".to_string(),
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
    let mut extended_error_pending = false;

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Close {
            target: DescribeTarget::Portal,
            name: "closed_portal".to_string(),
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert!(!extended_error_pending);
    assert!(session.prepared.contains_key("lookup"));
    assert!(!session.portals.contains_key("closed_portal"));

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "closed_portal".to_string(),
            max_rows: 0,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("34000".to_string())
    );
    assert!(extended_error_pending);

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_closed_portal (id INT)".to_string()),
    )
    .unwrap();
    assert!(!session.tables.contains_key("skipped_closed_portal"));

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
            portal_name: "rebound_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"2".to_vec())],
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "rebound_portal".to_string(),
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
fn extended_describe_missing_targets_skip_until_sync_and_recover() {
    for (target, missing_name, expected_code, suffix) in [
        (
            DescribeTarget::Statement,
            "missing_statement",
            "26000",
            "statement",
        ),
        (DescribeTarget::Portal, "missing_portal", "34000", "portal"),
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
            FrontendMessage::Describe {
                target,
                name: missing_name.to_string(),
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 1);
        assert_eq!(messages[0].0, b'E');
        assert_eq!(
            error_field_value(&messages[0].1, b'C'),
            Some(expected_code.to_string())
        );
        assert!(extended_error_pending);

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::SimpleQuery(format!(
                "CREATE TABLE skipped_describe_{suffix} (id INT)"
            )),
        )
        .unwrap();
        assert!(!session
            .tables
            .contains_key(&format!("skipped_describe_{suffix}")));

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
                result_format_codes: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
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
fn extended_close_missing_targets_do_not_enter_error_recovery() {
    for (target, missing_name, suffix) in [
        (DescribeTarget::Statement, "missing_statement", "statement"),
        (DescribeTarget::Portal, "missing_portal", "portal"),
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
            FrontendMessage::Close {
                target,
                name: missing_name.to_string(),
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 1);
        assert_eq!(messages[0].0, b'3');
        assert!(!extended_error_pending);
        assert!(session.prepared.contains_key("lookup"));

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::SimpleQuery(format!("CREATE TABLE skipped_close_{suffix} (id INT)")),
        )
        .unwrap();
        assert!(session
            .tables
            .contains_key(&format!("skipped_close_{suffix}")));
        assert_eq!(
            read_backend_tags(&mut reader, 2),
            vec![b'C', b'Z'],
            "simple query after missing Close should not be skipped"
        );

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Bind {
                portal_name: format!("recovered_{suffix}_portal"),
                statement_name: "lookup".to_string(),
                parameter_format_codes: Vec::new(),
                parameters: vec![Some(b"1".to_vec())],
                result_format_codes: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
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
fn extended_execute_missing_portal_skips_until_sync_and_recovers() {
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
        FrontendMessage::Execute {
            portal_name: "missing_portal".to_string(),
            max_rows: 1,
        },
    )
    .unwrap();
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("34000".to_string())
    );
    assert!(extended_error_pending);

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_execute (id INT)".to_string()),
    )
    .unwrap();
    assert!(!session.tables.contains_key("skipped_execute"));

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
            portal_name: "recovered_execute_portal".to_string(),
            statement_name: "lookup".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);

    handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "recovered_execute_portal".to_string(),
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
fn extended_close_does_not_remove_sql_prepared_statements() {
    let mut session = Session::default();
    session
        .prepared
        .insert("golden_stmt".to_string(), PreparedStatement::AddTen);
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_close(
        &mut writer,
        &mut session,
        DescribeTarget::Statement,
        "golden_stmt"
    )
    .unwrap());

    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'3']);
    assert_eq!(
        session.prepared.get("golden_stmt"),
        Some(&PreparedStatement::AddTen)
    );
}

#[test]
fn extended_parse_rejects_named_duplicates_and_replaces_unnamed_state() {
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
    let first = PreparedQuery {
        query: "SELECT id FROM people".to_string(),
        parameter_type_oids: Vec::new(),
    };
    session.replace_extended_statement("named".to_string(), first.clone());
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        "named".to_string(),
        "SELECT name FROM people".to_string(),
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert_eq!(
        session.prepared.get("named"),
        Some(&PreparedStatement::Extended(first))
    );

    let unnamed = PreparedQuery {
        query: "SELECT id FROM people WHERE id = $1".to_string(),
        parameter_type_oids: vec![23],
    };
    session.replace_extended_statement(String::new(), unnamed);
    session.replace_extended_portal(
        String::new(),
        Portal {
            statement_name: String::new(),
            query: PreparedQuery {
                query: "SELECT id FROM people WHERE id = $1".to_string(),
                parameter_type_oids: vec![23],
            },
            parameters: vec![Some("1".to_string())],
            result_format_codes: Vec::new(),
            described: false,
            result: None,
            position: 0,
            completed: false,
        },
    );

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        String::new(),
        "SELECT name FROM people WHERE name = $1".to_string(),
        vec![25]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(!session.portals.contains_key(""));
    assert_eq!(
        session.prepared.get(""),
        Some(&PreparedStatement::Extended(PreparedQuery {
            query: "SELECT name FROM people WHERE name = $1".to_string(),
            parameter_type_oids: vec![25],
        }))
    );
}

#[test]
fn extended_parse_accepts_bounded_dml_statements() {
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
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "insert_people".to_string(),
        "INSERT INTO people (id, name) VALUES ($1, $2)".to_string(),
        Vec::new()
    )
    .unwrap());

    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert_eq!(
        session.prepared.get("insert_people"),
        Some(&PreparedStatement::Extended(PreparedQuery {
            query: "INSERT INTO people (id, name) VALUES ($1, $2)".to_string(),
            parameter_type_oids: vec![23, 25],
        }))
    );
}

#[test]
fn extended_dml_insert_update_delete_execute_and_recover() {
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
    let (mut writer, mut reader) = tcp_pair();

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "insert_people".to_string(),
        "INSERT INTO people (id, name) VALUES ($1, $2)".to_string(),
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "insert_portal".to_string(),
        "insert_people".to_string(),
        Vec::new(),
        vec![Some(b"1".to_vec()), Some(b"Ada".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!handle_execute(&mut writer, &mut session, "insert_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"INSERT 0 1\0");

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "update_people".to_string(),
        "UPDATE people SET name = $1 WHERE id = $2".to_string(),
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "update_portal".to_string(),
        "update_people".to_string(),
        Vec::new(),
        vec![Some(b"Grace".to_vec()), Some(b"1".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!handle_execute(&mut writer, &mut session, "update_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"UPDATE 1\0");

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "delete_people".to_string(),
        "DELETE FROM people WHERE name = $1".to_string(),
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert!(!handle_bind(
        &mut writer,
        &mut session,
        "delete_portal".to_string(),
        "delete_people".to_string(),
        Vec::new(),
        vec![Some(b"Grace".to_vec())],
        Vec::new()
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
    assert!(!handle_execute(&mut writer, &mut session, "delete_portal", 0).unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"DELETE 1\0");
    assert!(session.tables["people"].rows.is_empty());
}

#[test]
fn extended_parse_rejects_unsupported_copy_explicitly_without_installing_statement() {
    let mut session = Session::default();
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        "copy_people".to_string(),
        "/* comment */ COPY (SELECT * FROM people) TO STDOUT".to_string(),
        Vec::new()
    )
    .unwrap());

    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("0A000".to_string())
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some("COPY is not supported by the compatibility endpoint".to_string())
    );
    assert!(!session.prepared.contains_key("copy_people"));
}

#[test]
fn extended_parse_rejects_unsupported_cursor_options_without_installing_statement() {
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
            rows: Vec::new(),
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        "bad_cursor_options".to_string(),
        "DECLARE bad_cursor BINARY CURSOR FOR SELECT id FROM people".to_string(),
        Vec::new()
    )
    .unwrap());

    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'E');
    assert_eq!(
        error_field_value(&messages[0].1, b'M'),
        Some(
            "cursor declaration options are not supported by the compatibility endpoint"
                .to_string()
        )
    );
    assert!(!session.prepared.contains_key("bad_cursor_options"));
}

#[test]
fn extended_parse_rejects_extra_parameter_type_oids_without_installing_statement() {
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
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_parse(
        &mut writer,
        &mut session,
        "lookup".to_string(),
        "SELECT name FROM people WHERE id = $1".to_string(),
        vec![23, 25]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'E']);
    assert!(!session.prepared.contains_key("lookup"));

    assert!(!handle_parse(
        &mut writer,
        &mut session,
        "lookup".to_string(),
        "SELECT name FROM people WHERE id = $1".to_string(),
        vec![23]
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
    assert_eq!(
        session.prepared.get("lookup"),
        Some(&PreparedStatement::Extended(PreparedQuery {
            query: "SELECT name FROM people WHERE id = $1".to_string(),
            parameter_type_oids: vec![23],
        }))
    );
}

#[test]
fn extended_parse_errors_skip_until_sync_and_recover() {
    for (statement_name, query, parameter_type_oids, expected_code, expected_message, suffix) in [
        (
            "bad_type",
            "SELECT name FROM people WHERE id = $1",
            vec![16],
            "0A000",
            "only text and int4 extended-query parameters are supported",
            "bad_type",
        ),
        (
            "too_many_oids",
            "SELECT name FROM people WHERE id = $1",
            vec![23, 25],
            "08P01",
            "parse message has too many parameter type oids",
            "too_many_oids",
        ),
        (
            "unsupported_update_without_where",
            "UPDATE people SET name = $1",
            vec![25],
            "0A000",
            "extended query protocol only supports relational SELECT",
            "unsupported_update_without_where",
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
        let (mut writer, mut reader) = tcp_pair();
        let mut extended_error_pending = false;

        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Parse {
                statement_name: statement_name.to_string(),
                query: query.to_string(),
                parameter_type_oids,
            },
        )
        .unwrap();
        let messages = read_backend_messages(&mut reader, 1);
        assert_eq!(messages[0].0, b'E');
        assert_eq!(
            error_field_value(&messages[0].1, b'C'),
            Some(expected_code.to_string())
        );
        assert_eq!(
            error_field_value(&messages[0].1, b'M'),
            Some(expected_message.to_string())
        );
        assert!(extended_error_pending);
        assert!(!session.prepared.contains_key(statement_name));

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
            FrontendMessage::Parse {
                statement_name: format!("recovered_{suffix}"),
                query: "SELECT name FROM people WHERE id = $1".to_string(),
                parameter_type_oids: vec![23],
            },
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'1']);
        handle_frontend_message(
            &mut writer,
            &mut session,
            &mut extended_error_pending,
            FrontendMessage::Bind {
                portal_name: format!("recovered_{suffix}_portal"),
                statement_name: format!("recovered_{suffix}"),
                parameter_format_codes: Vec::new(),
                parameters: vec![Some(b"1".to_vec())],
                result_format_codes: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(read_backend_tags(&mut reader, 1), vec![b'2']);
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
