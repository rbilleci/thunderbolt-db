use super::*;

#[test]
fn canonical_sql_collapses_case_whitespace_and_semicolons() {
    assert_eq!(
        canonical_sql("  SELECT   1   AS One ; ; "),
        "select 1 as one"
    );
}

#[test]
fn recognizes_pg_dumpall_tablespace_metadata_query() {
    let canonical = canonical_sql(
        "SELECT oid, spcname, pg_catalog.pg_get_userbyid(spcowner) AS spcowner, pg_catalog.pg_tablespace_location(oid), spcacl, acldefault('t', spcowner) AS acldefault, array_to_string(spcoptions, ', '),pg_catalog.shobj_description(oid, 'pg_tablespace') FROM pg_catalog.pg_tablespace WHERE spcname !~ '^pg_' ORDER BY 1",
    );
    assert!(is_pg_dumpall_tablespace_metadata_query(&canonical));
}

#[test]
fn copy_statement_detection_skips_leading_comments() {
    assert!(is_copy_statement(
        "/* copy boundary */ -- line comment\nCOPY copy_people TO STDOUT;"
    ));
    assert!(is_copy_statement(
        "/* outer /* nested */ done */ COPY copy_people FROM STDIN;"
    ));
    assert!(!is_copy_statement(
        "/* copy-looking comment */ SELECT 'COPY people TO STDOUT'"
    ));
    assert!(!is_copy_statement(
        "/* unterminated COPY copy_people TO STDOUT"
    ));
}

#[test]
fn copy_to_stdout_table_detection_is_narrow() {
    assert_eq!(
        parse_copy_to_stdout_table("COPY public.people TO STDOUT;"),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions::TEXT,
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table("/* comment */ COPY people TO STDOUT"),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions::TEXT,
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table("COPY people TO STDOUT WITH CSV"),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions::CSV,
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table("COPY people TO STDOUT WITH CSV HEADER"),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions::CSV_HEADER,
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table(
            "COPY people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER '|')"
        ),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions {
                format: CopyFormat::Csv,
                header: true,
                delimiter: '|',
                quote: '"',
                escape: '"',
            }
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table(
            "COPY people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER '|', QUOTE '''', ESCAPE '\\')"
        ),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions {
                format: CopyFormat::Csv,
                header: true,
                delimiter: '|',
                quote: '\'',
                escape: '\\',
            }
        })
    );
    assert_eq!(
        parse_copy_to_stdout_table("COPY people TO STDOUT WITH (FORMAT csv, QUOTE '|')"),
        Some(gpu_db_protocol::CopyToStdout {
            table: "people".to_string(),
            options: CopyOptions {
                format: CopyFormat::Csv,
                header: false,
                delimiter: ',',
                quote: '|',
                escape: '|',
            }
        })
    );
    assert_eq!(parse_copy_to_stdout_table("COPY people FROM STDIN"), None);
    assert_eq!(
        parse_copy_to_stdout_table("COPY (SELECT * FROM people) TO STDOUT"),
        None
    );
    assert_eq!(
        parse_copy_to_stdout_table("COPY people TO STDOUT WITH (FORMAT csv, NULL '')"),
        None
    );
}

#[test]
fn copy_from_stdin_table_detection_is_narrow() {
    assert_eq!(
        parse_copy_from_stdin("COPY public.people FROM STDIN;"),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: None,
            options: CopyOptions::TEXT,
        })
    );
    assert_eq!(
        parse_copy_from_stdin("/* comment */ COPY people (id, name) FROM STDIN"),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            options: CopyOptions::TEXT,
        })
    );
    assert_eq!(
        parse_copy_from_stdin("COPY people FROM STDIN WITH CSV"),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: None,
            options: CopyOptions::CSV,
        })
    );
    assert_eq!(
        parse_copy_from_stdin("COPY public.people (id, name) FROM STDIN WITH CSV"),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            options: CopyOptions::CSV,
        })
    );
    assert_eq!(
        parse_copy_from_stdin("COPY public.people (id, name) FROM STDIN WITH CSV HEADER"),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            options: CopyOptions::CSV_HEADER,
        })
    );
    assert_eq!(
        parse_copy_from_stdin(
            "COPY public.people (id, name) FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER '|')"
        ),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            options: CopyOptions {
                format: CopyFormat::Csv,
                header: true,
                delimiter: '|',
                quote: '"',
                escape: '"',
            }
        })
    );
    assert_eq!(
        parse_copy_from_stdin(
            "COPY public.people (id, name) FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER '|', QUOTE '''', ESCAPE '\\')"
        ),
        Some(gpu_db_protocol::CopyFromStdin {
            table: "people".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            options: CopyOptions {
                format: CopyFormat::Csv,
                header: true,
                delimiter: '|',
                quote: '\'',
                escape: '\\',
            }
        })
    );
    assert_eq!(parse_copy_from_stdin("COPY people TO STDOUT"), None);
    assert_eq!(
        parse_copy_from_stdin("COPY (SELECT * FROM people) FROM STDIN"),
        None
    );
    assert_eq!(
        parse_copy_from_stdin("COPY people FROM STDIN WITH (FORMAT csv, DELIMITER '|', QUOTE '|')"),
        None
    );
}

#[test]
fn truncate_table_detection_is_narrow() {
    assert_eq!(
        parse_truncate_table("TRUNCATE TABLE ONLY public.people;"),
        Some(ParsedTruncateTable {
            table: "people".to_string(),
            restart_identity: false,
        })
    );
    assert_eq!(
        parse_truncate_table("/* restore */ TRUNCATE TABLE people;"),
        Some(ParsedTruncateTable {
            table: "people".to_string(),
            restart_identity: false,
        })
    );
    assert_eq!(
        parse_truncate_table("TRUNCATE people"),
        Some(ParsedTruncateTable {
            table: "people".to_string(),
            restart_identity: false,
        })
    );
    assert_eq!(
        parse_truncate_table("TRUNCATE TABLE public.people RESTART IDENTITY"),
        Some(ParsedTruncateTable {
            table: "people".to_string(),
            restart_identity: true,
        })
    );
    assert_eq!(
        parse_truncate_table("TRUNCATE TABLE public.people CONTINUE IDENTITY"),
        None
    );
    assert_eq!(parse_truncate_table("TRUNCATE TABLE people CASCADE"), None);
    assert_eq!(parse_truncate_table("TRUNCATE TABLE people, teams"), None);
    assert_eq!(parse_truncate_table("TRUNCATE TABLE private.people"), None);
    assert_eq!(
        parse_truncate_table("TRUNCATE TABLE public.people CASCADE"),
        None
    );
    assert_eq!(parse_truncate_table("TRUNCATE TABLE \"people\""), None);
}

#[test]
fn drop_table_detection_is_narrow() {
    assert_eq!(
        parse_drop_table("DROP TABLE IF EXISTS public.people;"),
        Some(DropTable {
            tables: vec!["people".to_string()],
            if_exists: true,
        })
    );
    assert_eq!(
        parse_drop_table("/* restore */ DROP TABLE people;"),
        Some(DropTable {
            tables: vec!["people".to_string()],
            if_exists: false,
        })
    );
    assert_eq!(
        parse_drop_table("DROP TABLE public.people, teams"),
        Some(DropTable {
            tables: vec!["people".to_string(), "teams".to_string()],
            if_exists: false,
        })
    );
    assert_eq!(parse_drop_table("DROP TABLE public.people CASCADE"), None);
    assert_eq!(parse_drop_table("DROP SCHEMA IF EXISTS public"), None);
    assert_eq!(parse_drop_table("DROP TABLE \"people\""), None);
    assert_eq!(parse_drop_table("DROP TABLE private.people, teams"), None);
    assert_eq!(
        parse_alter_table_drop_constraint(
            "ALTER TABLE IF EXISTS ONLY public.accounts DROP CONSTRAINT IF EXISTS accounts_pkey;"
        ),
        Some(DropConstraint {
            table: "accounts".to_string(),
            constraint: "accounts_pkey".to_string(),
            table_if_exists: true,
            if_exists: true,
        })
    );
    assert_eq!(
        parse_alter_table_drop_constraint(
            "ALTER TABLE ONLY accounts DROP CONSTRAINT accounts_pkey;"
        ),
        Some(DropConstraint {
            table: "accounts".to_string(),
            constraint: "accounts_pkey".to_string(),
            table_if_exists: false,
            if_exists: false,
        })
    );
    assert_eq!(
        parse_alter_table_drop_constraint("ALTER TABLE accounts DROP COLUMN id"),
        None
    );
}

#[test]
fn simple_copy_to_stdout_emits_copyout_data_done_and_recovers() {
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
                vec![SqlValue::Int4(2), SqlValue::Text("Tab\tName".to_string())],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(&mut writer, &mut session, "COPY people TO STDOUT", true).unwrap();

    let messages = read_backend_messages(&mut reader, 5);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'c', b'C']
    );
    assert_eq!(messages[0].1, vec![0, 0, 2, 0, 0, 0, 0]);
    assert_eq!(messages[1].1, b"1\tAda\n");
    assert_eq!(messages[2].1, b"2\tTab\\tName\n");
    assert_eq!(messages[4].1, b"COPY 2\0");

    execute_statement(
        &mut writer,
        &mut session,
        "SELECT name FROM people WHERE id = 1",
        true,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'T', b'D', b'C']);
}

#[test]
fn simple_copy_to_stdout_with_csv_quotes_fields_and_recovers() {
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
                vec![
                    SqlValue::Int4(2),
                    SqlValue::Text("Grace, \"Amazing\"".to_string()),
                ],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "COPY people TO STDOUT WITH CSV",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 5);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'c', b'C']
    );
    assert_eq!(messages[1].1, b"1,Ada\n");
    assert_eq!(messages[2].1, b"2,\"Grace, \"\"Amazing\"\"\"\n");
    assert_eq!(messages[4].1, b"COPY 2\0");

    execute_statement(
        &mut writer,
        &mut session,
        "SELECT name FROM people WHERE id = 2",
        true,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'T', b'D', b'C']);
}

#[test]
fn simple_copy_to_stdout_with_csv_header_emits_column_names_and_recovers() {
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
                        name: "full_name".to_string(),
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

    execute_statement(
        &mut writer,
        &mut session,
        "COPY people TO STDOUT WITH CSV HEADER",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 5);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'c', b'C']
    );
    assert_eq!(messages[1].1, b"id,full_name\n");
    assert_eq!(messages[2].1, b"1,Ada\n");
    assert_eq!(messages[4].1, b"COPY 1\0");

    execute_statement(
        &mut writer,
        &mut session,
        "SELECT full_name FROM people WHERE id = 1",
        true,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'T', b'D', b'C']);
}

#[test]
fn simple_copy_to_stdout_with_parenthesized_csv_delimiter_recovers() {
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
                        name: "full_name".to_string(),
                        ty: SqlType::Text,
                        domain: None,
                        default: None,
                    },
                },
            ],
            rows: vec![
                vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
                vec![
                    SqlValue::Int4(2),
                    SqlValue::Text("Grace|Hopper".to_string()),
                ],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "COPY people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER '|')",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 6);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'd', b'c', b'C']
    );
    assert_eq!(messages[1].1, b"id|full_name\n");
    assert_eq!(messages[2].1, b"1|Ada\n");
    assert_eq!(messages[3].1, b"2|\"Grace|Hopper\"\n");
    assert_eq!(messages[5].1, b"COPY 2\0");

    execute_statement(
        &mut writer,
        &mut session,
        "SELECT full_name FROM people WHERE id = 2",
        true,
    )
    .unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'T', b'D', b'C']);
}

#[test]
fn simple_copy_from_stdin_accepts_data_done_and_recovers() {
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
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("COPY people FROM STDIN".to_string())
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'G');
    assert_eq!(messages[0].1, vec![0, 0, 2, 0, 0, 0, 0]);
    assert!(session.copy_in.is_some());

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"1\tAda\n2\tGrace\\tHopper\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Grace\tHopper".to_string())
            ],
        ]
    );

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn simple_copy_from_stdin_with_parenthesized_csv_delimiter_recovers() {
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
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery(
            "COPY people FROM STDIN WITH (FORMAT csv, HEADER, DELIMITER '|')".to_string()
        )
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'G');

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"id|name\n1|Ada\n2|\"Grace|Hopper\"\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Grace|Hopper".to_string())
            ],
        ]
    );

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn simple_copy_from_stdin_with_csv_accepts_quoted_data_and_recovers() {
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
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("COPY people FROM STDIN WITH CSV".to_string())
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'G');
    assert_eq!(messages[0].1, vec![0, 0, 2, 0, 0, 0, 0]);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"1,Ada\n2,\"Grace, \"\"Hopper\"\"\"\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Grace, \"Hopper\"".to_string())
            ],
        ]
    );

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn simple_copy_from_stdin_with_csv_header_skips_header_and_recovers() {
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
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("COPY people FROM STDIN WITH CSV HEADER".to_string())
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'G');

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"id,name\n1,Ada\n2,\"Grace, \"\"Hopper\"\"\"\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Grace, \"Hopper\"".to_string())
            ],
        ]
    );

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn extended_copy_to_stdout_emits_copyout_data_done_and_recovers() {
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
                vec![
                    SqlValue::Int4(2),
                    SqlValue::Text("Grace|Hopper".to_string()),
                ],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Parse {
            statement_name: "copy_out".to_string(),
            query: "COPY people TO STDOUT WITH (FORMAT csv, HEADER, DELIMITER '|')".to_string(),
            parameter_type_oids: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "copy_out_portal".to_string(),
            statement_name: "copy_out".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Describe {
            target: DescribeTarget::Portal,
            name: "copy_out_portal".to_string(),
        }
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'1', b'2', b'n']);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "copy_out_portal".to_string(),
            max_rows: 0,
        }
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 6);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'H', b'd', b'd', b'd', b'c', b'C']
    );
    assert_eq!(messages[1].1, b"id|name\n");
    assert_eq!(messages[2].1, b"1|Ada\n");
    assert_eq!(messages[3].1, b"2|\"Grace|Hopper\"\n");
    assert_eq!(messages[5].1, b"COPY 2\0");

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn extended_copy_from_stdin_accepts_data_and_copyfail_does_not_mutate() {
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
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Parse {
            statement_name: "copy_in".to_string(),
            query: "COPY people (id, name) FROM STDIN WITH (FORMAT csv, HEADER, DELIMITER '|')"
                .to_string(),
            parameter_type_oids: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "copy_in_portal".to_string(),
            statement_name: "copy_in".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "copy_in_portal".to_string(),
            max_rows: 0,
        }
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 3);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'1', b'2', b'G']
    );
    assert!(session.copy_in.is_some());

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"id|name\n1|Ada\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyFail("client aborted copy".to_string())
    )
    .unwrap());
    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(
        messages.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
        vec![b'E']
    );
    assert_eq!(
        error_field_value(&messages[0].1, b'C'),
        Some("57014".to_string())
    );
    assert!(session.tables["people"].rows.is_empty());
    assert!(extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Parse {
            statement_name: "copy_in_retry".to_string(),
            query: "COPY people (id, name) FROM STDIN WITH (FORMAT csv, HEADER, DELIMITER '|')"
                .to_string(),
            parameter_type_oids: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Bind {
            portal_name: "copy_in_retry_portal".to_string(),
            statement_name: "copy_in_retry".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        }
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Execute {
            portal_name: "copy_in_retry_portal".to_string(),
            max_rows: 0,
        }
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'1', b'2', b'G']);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"id|name\n1|Ada\n2|\"Grace|Hopper\"\n".to_vec())
    )
    .unwrap());
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'C']);
    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Grace|Hopper".to_string())
            ],
        ]
    );

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("SELECT name FROM people WHERE id = 2".to_string())
    )
    .unwrap());
    assert_eq!(
        read_backend_tags(&mut reader, 4),
        vec![b'T', b'D', b'C', b'Z']
    );
}

#[test]
fn truncate_table_clears_rows_and_recovers() {
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
                    default: Some(ColumnDefault::SequenceNextVal {
                        sequence: "people_id_seq".to_string(),
                        create_if_missing: false,
                    }),
                },
            }],
            rows: vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(2)]],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    session.sequences.insert(
        "people_id_seq".to_string(),
        Sequence {
            oid: FIRST_USER_RELATION_OID + 1,
            name: "people_id_seq".to_string(),
            last_value: 2,
            is_called: true,
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "TRUNCATE TABLE ONLY public.people RESTART IDENTITY",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"TRUNCATE TABLE\0");
    assert!(session.tables["people"].rows.is_empty());
    assert_eq!(session.sequences["people_id_seq"].last_value, 1);
    assert!(!session.sequences["people_id_seq"].is_called);

    execute_statement(&mut writer, &mut session, "SELECT id FROM people", true).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'T', b'C']);
}

#[test]
fn simple_relational_delete_removes_matching_rows_and_recovers() {
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
                vec![
                    SqlValue::Int4(4),
                    SqlValue::Text("Ada Lovelace".to_string()),
                ],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "DELETE FROM people WHERE id = 2 OR name LIKE 'Ada%'",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"DELETE 3\0");
    assert_eq!(
        session.tables["people"].rows,
        vec![vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())]]
    );

    execute_statement(&mut writer, &mut session, "SELECT id FROM people", true).unwrap();
    assert_eq!(read_backend_tags(&mut reader, 3), vec![b'T', b'D', b'C']);
}

#[test]
fn simple_relational_update_changes_matching_rows_and_recovers() {
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
                vec![
                    SqlValue::Int4(4),
                    SqlValue::Text("Ada Lovelace".to_string()),
                ],
            ],
            check_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        },
    );
    let (mut writer, mut reader) = tcp_pair();

    execute_statement(
        &mut writer,
        &mut session,
        "UPDATE people SET name = 'Updated' WHERE id = 2 OR name LIKE 'Ada%'",
        true,
    )
    .unwrap();

    let messages = read_backend_messages(&mut reader, 1);
    assert_eq!(messages[0].0, b'C');
    assert_eq!(messages[0].1, b"UPDATE 3\0");
    assert_eq!(
        session.tables["people"].rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Updated".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("Updated".to_string())],
            vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            vec![SqlValue::Int4(4), SqlValue::Text("Updated".to_string())],
        ]
    );

    execute_statement(&mut writer, &mut session, "SELECT id FROM people", true).unwrap();
    assert_eq!(
        read_backend_tags(&mut reader, 6),
        vec![b'T', b'D', b'D', b'D', b'D', b'C']
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
fn split_simple_query_preserves_semicolons_inside_text_literals() {
    assert_eq!(
        split_simple_query(
            r#"INSERT INTO commands VALUES ('SELECT 1;'); PREPARE "lookup;name"(int4) AS SELECT 'Ada'';Lovelace';"#
        ),
        vec![
            "INSERT INTO commands VALUES ('SELECT 1;')",
            r#"PREPARE "lookup;name"(int4) AS SELECT 'Ada'';Lovelace'"#
        ]
    );
}

#[test]
fn split_simple_query_preserves_semicolons_inside_sql_comments() {
    assert_eq!(
        split_simple_query(
            "/* comment ; /* nested ; */ done */ PREPARE lookup(int4) AS SELECT id FROM people WHERE id = $1; \
             -- comment ; before execute\n\
             EXECUTE lookup(1);"
        ),
        vec![
            "PREPARE lookup(int4) AS SELECT id FROM people WHERE id = $1",
            "EXECUTE lookup(1)"
        ]
    );
}

#[test]
fn frontend_function_call_unsupported_error_is_explicit() {
    assert_eq!(
        unsupported_frontend_message(&FrontendMessage::FunctionCall {
            function_oid: 42,
            argument_format_codes: vec![],
            arguments: vec![],
            result_format_code: 0,
        }),
        "FunctionCall is not supported by the compatibility endpoint"
    );
    assert_eq!(
        unsupported_frontend_message(&FrontendMessage::CopyData(Vec::new())),
        "frontend COPY data flow is not supported by the compatibility endpoint"
    );
    assert_eq!(
        unsupported_frontend_message(&FrontendMessage::CopyDone),
        "frontend COPY data flow is not supported by the compatibility endpoint"
    );
    assert_eq!(
        unsupported_frontend_message(&FrontendMessage::CopyFail(
            "client aborted copy".to_string()
        )),
        "frontend COPY data flow is not supported by the compatibility endpoint"
    );
}

#[test]
fn frontend_function_call_error_skips_until_sync_and_recovers() {
    let mut session = Session::default();
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::FunctionCall {
            function_oid: 42,
            argument_format_codes: vec![],
            arguments: vec![],
            result_format_code: 0,
        }
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
        Some("FunctionCall is not supported by the compatibility endpoint".to_string())
    );
    assert!(extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_function_call (id INT)".to_string())
    )
    .unwrap());
    assert!(!session.tables.contains_key("skipped_function_call"));

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE recovered_function_call (id INT)".to_string())
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert!(session.tables.contains_key("recovered_function_call"));
}

#[test]
fn frontend_copy_data_error_skips_until_sync_and_recovers() {
    let mut session = Session::default();
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyData(b"1\tAda\n".to_vec())
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
        Some("frontend COPY data flow is not supported by the compatibility endpoint".to_string())
    );
    assert!(extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_copy_frame (id INT)".to_string())
    )
    .unwrap());
    assert!(!session.tables.contains_key("skipped_copy_frame"));

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE recovered_copy_frame (id INT)".to_string())
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert!(session.tables.contains_key("recovered_copy_frame"));
}

#[test]
fn frontend_copy_done_error_skips_until_sync_and_recovers() {
    let mut session = Session::default();
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyDone
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
        Some("frontend COPY data flow is not supported by the compatibility endpoint".to_string())
    );
    assert!(extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_copy_done (id INT)".to_string())
    )
    .unwrap());
    assert!(!session.tables.contains_key("skipped_copy_done"));

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE recovered_copy_done (id INT)".to_string())
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert!(session.tables.contains_key("recovered_copy_done"));
}

#[test]
fn frontend_copy_fail_error_skips_until_sync_and_recovers() {
    let mut session = Session::default();
    let mut extended_error_pending = false;
    let (mut writer, mut reader) = tcp_pair();

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::CopyFail("client aborted copy".to_string())
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
        Some("frontend COPY data flow is not supported by the compatibility endpoint".to_string())
    );
    assert!(extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE skipped_copy_fail (id INT)".to_string())
    )
    .unwrap());
    assert!(!session.tables.contains_key("skipped_copy_fail"));

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::Sync
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 1), vec![b'Z']);
    assert!(!extended_error_pending);

    assert!(handle_frontend_message(
        &mut writer,
        &mut session,
        &mut extended_error_pending,
        FrontendMessage::SimpleQuery("CREATE TABLE recovered_copy_fail (id INT)".to_string())
    )
    .unwrap());
    assert_eq!(read_backend_tags(&mut reader, 2), vec![b'C', b'Z']);
    assert!(session.tables.contains_key("recovered_copy_fail"));
}
