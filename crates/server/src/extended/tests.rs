use super::*;
use gpu_db_facade::{
    BoundPreparedStatement, DbError, DbValue, ErrorCategory, PreparedStatement, QueryOutcome,
    SharedEngine, SharedSession, SubmissionRequest,
};

fn submit_text(
    engine: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    engine
        .submit(session, SubmissionRequest::Text(sql))
        .into_immediate()
}

fn submit_prepared(
    engine: &SharedEngine,
    session: &mut SharedSession,
    bound: &BoundPreparedStatement,
) -> Result<QueryOutcome, DbError> {
    engine
        .submit(session, SubmissionRequest::Prepared(bound))
        .into_immediate()
}

#[test]
fn session_cursor_and_parse_shape_errors_keep_the_frozen_psql_diagnostics() {
    let mut extended = ExtendedSession::default();
    for error in [
        extended.fetch_sql_cursor("missing", Some(1)).unwrap_err(),
        extended
            .close_sql_cursor(crate::sql_cursor::SqlCursorCloseTarget::Named(
                "missing".to_string(),
            ))
            .unwrap_err(),
    ] {
        assert_eq!(error.code, "34000");
        assert_eq!(error.message, "cursor does not exist");
    }

    let error = extended_prepared_parse_error(DbError {
        category: ErrorCategory::Syntax,
        message: "invalid SQL parameter reference".to_string(),
    });
    assert_eq!(error.code, "42P02");
    assert_eq!(error.message, "there is no parameter $0");
}

#[test]
fn materialized_cursor_lifecycle_distinguishes_idle_and_transaction_ownership() {
    let outcome = || QueryOutcome::Rows {
        columns: vec![ColumnMeta {
            name: "value".to_string(),
            logical_type: LogicalType::Int4,
            numeric_typmod: None,
        }],
        rows: vec![
            vec![DbValue::Int4(1)],
            vec![DbValue::Int4(2)],
            vec![DbValue::Int4(3)],
        ],
    };
    let mut extended = ExtendedSession::default();
    extended
        .install_sql_cursor("idle_cursor".to_string(), outcome(), false)
        .unwrap();
    extended
        .install_sql_cursor("transaction_cursor".to_string(), outcome(), true)
        .unwrap();

    let QueryOutcome::Command { tag, .. } =
        extended.move_sql_cursor("idle_cursor", Some(2)).unwrap()
    else {
        panic!("MOVE must return a command outcome");
    };
    assert_eq!(tag, CommandTag::Other("MOVE 2".to_string()));
    assert_eq!(
        extended.fetch_sql_cursor("idle_cursor", None).unwrap(),
        QueryOutcome::Returning {
            tag: CommandTag::Other("FETCH".to_string()),
            columns: vec![ColumnMeta {
                name: "value".to_string(),
                logical_type: LogicalType::Int4,
                numeric_typmod: None,
            }],
            rows: vec![vec![DbValue::Int4(3)]],
            rows_affected: 1,
        }
    );

    extended.finish_transaction_boundary(false);
    assert!(extended.fetch_sql_cursor("idle_cursor", Some(0)).is_ok());
    let error = extended
        .fetch_sql_cursor("transaction_cursor", Some(0))
        .unwrap_err();
    assert_eq!(error.code, "34000");
    assert_eq!(error.message, "cursor does not exist");

    let duplicate = extended
        .ensure_sql_cursor_name_available("idle_cursor")
        .unwrap_err();
    assert_eq!(duplicate.code, "42P03");
    assert_eq!(duplicate.message, "cursor already exists");
}

#[test]
fn extended_cursor_wrapper_describes_no_data_and_survives_its_implicit_cycle() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    let mut extended = ExtendedSession::default();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE cursor_wrapper_accounts (id int4)",
    )
    .unwrap();
    let target = ExtendedSession::analyze_sql_prepare(
        &engine,
        &session,
        "SELECT id FROM cursor_wrapper_accounts WHERE id = $1",
        &[Some(LogicalType::Int4)],
    )
    .unwrap();
    extended
        .install_sql_prepared("cursor_target".to_string(), target)
        .unwrap();

    extended.complete_transaction_action(TransactionAction::BeginImplicit, true);
    let request = extended
        .prepare_request(
            String::new(),
            "DECLARE implicit_cursor CURSOR FOR EXECUTE cursor_target($1)".to_string(),
            &[23],
            SessionTransactionStatus::InTransaction,
        )
        .unwrap();
    assert!(
        !request
            .cursor_declaration
            .as_ref()
            .unwrap()
            .transaction_bound
    );
    let analysis = ExtendedSession::analyze_prepare(&engine, &mut session, &request);
    extended.complete_parse(request, analysis).unwrap();
    assert_eq!(
        message_tags(
            &extended
                .describe(
                    DescribeTarget::Statement,
                    "",
                    SessionTransactionStatus::InTransaction,
                )
                .unwrap()
        ),
        vec![b't', b'n']
    );
    extended
        .bind(String::new(), "", &[], &[Some(b"2".to_vec())], &[])
        .unwrap();
    assert!(matches!(
        extended.execution_request("").unwrap(),
        Some(ExecutionRequest::Query(_))
    ));
    extended
        .set_execution_outcome(
            "",
            Ok(QueryOutcome::Rows {
                columns: vec![ColumnMeta {
                    name: "id".to_string(),
                    logical_type: LogicalType::Int4,
                    numeric_typmod: None,
                }],
                rows: vec![vec![DbValue::Int4(2)]],
            }),
        )
        .unwrap();
    assert_eq!(
        message_tags(&extended.encode_execute("", 0).unwrap()),
        vec![b'C']
    );

    extended.complete_transaction_action(TransactionAction::CommitImplicit, true);
    extended.finish_transaction_boundary(false);
    assert!(matches!(
        extended
            .fetch_sql_cursor("implicit_cursor", Some(1))
            .unwrap(),
        QueryOutcome::Returning { rows, .. } if rows == vec![vec![DbValue::Int4(2)]]
    ));

    let explicit = extended
        .prepare_request(
            "explicit".to_string(),
            "DECLARE explicit_cursor CURSOR FOR EXECUTE cursor_target($1)".to_string(),
            &[23],
            SessionTransactionStatus::InTransaction,
        )
        .unwrap();
    assert!(
        explicit
            .cursor_declaration
            .as_ref()
            .unwrap()
            .transaction_bound
    );
}

#[test]
fn extended_parameterized_select_cursor_keeps_the_frozen_fetch_count_boundary() {
    let mut extended = ExtendedSession::default();
    let error = extended
        .prepare_request(
            String::new(),
            "DECLARE _psql_cursor CURSOR FOR SELECT id FROM accounts WHERE id = $1".to_string(),
            &[23],
            SessionTransactionStatus::Idle,
        )
        .unwrap_err();
    assert_eq!(error.code, "0A000");
    assert_eq!(
        error.message,
        "parameterized cursor declarations are not supported"
    );
}

#[test]
fn bind_arity_error_keeps_the_frozen_psql_diagnostic() {
    let statement = Statement {
        prepared: PreparedStatement::parse("SELECT 1").unwrap(),
        sql_execute_bind_arguments: None,
        deferred_execution_error: None,
        cursor_declaration: None,
        copy: None,
        copy_target: None,
        parameter_oids: vec![23, 25],
        columns: Vec::new(),
    };
    let error = validate_bind_shape(&statement, &[], &[Some(b"1".to_vec())]).unwrap_err();
    assert_eq!(error.code, "08P01");
    assert_eq!(error.message, "bind message has wrong number of parameters");
}

#[test]
fn extended_sql_execute_arity_keeps_the_frozen_prepared_diagnostic() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE arity_accounts (id int4)",
    )
    .unwrap();
    let target = ExtendedSession::analyze_sql_prepare(
        &engine,
        &session,
        "SELECT id FROM arity_accounts WHERE id = $1",
        &[Some(LogicalType::Int4)],
    )
    .unwrap();
    let error = sql_execute_bind_arguments(&target, &[], &[]).unwrap_err();
    assert_eq!(error.code, "08P01");
    assert_eq!(
        error.message,
        "bound parameter count does not match prepared statement"
    );
}

#[test]
fn extended_relational_parse_errors_keep_the_bounded_protocol_diagnostic() {
    for parsed in [
        PreparedStatement::parse(
            "SELECT id FROM accounts JOIN teams ON id = account_id WHERE id = $1",
        )
        .unwrap_err(),
        DbError {
            category: ErrorCategory::Syntax,
            message: "query shape is not supported by the compatibility stub".to_string(),
        },
    ] {
        let error = extended_prepared_parse_error(parsed);
        assert_eq!(error.code, "0A000");
        assert_eq!(
            error.message,
            "extended query protocol only supports relational SELECT"
        );
    }
}

#[test]
fn post_startup_password_and_sasl_messages_are_protocol_errors() {
    for message in [
        FrontendMessage::PasswordMessage("secret".to_string()),
        FrontendMessage::SaslInitialResponse {
            mechanism: "SCRAM-SHA-256".to_string(),
            initial_response: Some(b"n,,n=,r=late".to_vec()),
        },
        FrontendMessage::SaslResponse(b"c=biws,r=late,p=proof".to_vec()),
    ] {
        let Dispatch::Response(Err(error)) =
            ExtendedSession::default().dispatch(message, SessionTransactionStatus::Idle)
        else {
            panic!("post-startup authentication message was not rejected");
        };
        assert_eq!(error.code, "08P01");
    }
}

#[test]
fn parse_bind_and_describe_are_effect_free_and_catalog_typed() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE accounts (id int4 PRIMARY KEY, balance int8)",
    )
    .unwrap();
    let mut extended = ExtendedSession::default();
    assert_eq!(
        extended
            .parse(
                "update_balance".to_string(),
                "UPDATE accounts SET balance = balance + $2 WHERE id = $1 RETURNING balance",
                &[],
                |sql, hints| engine.prepare_statement(&session, sql, hints),
            )
            .unwrap()[0],
        b'1'
    );
    assert_eq!(
        extended
            .bind(
                "portal".to_string(),
                "update_balance",
                &[],
                &[Some(b"7".to_vec()), Some(b"-2".to_vec())],
                &[],
            )
            .unwrap()[0],
        b'2'
    );
    let statement_description = extended
        .describe(
            DescribeTarget::Statement,
            "update_balance",
            SessionTransactionStatus::Idle,
        )
        .unwrap();
    assert_eq!(statement_description[0], b't');
    assert!(statement_description.contains(&b'T'));
    assert!(extended.execution_request("portal").unwrap().is_some());
}

#[test]
fn extended_sql_execute_binds_outer_parameters_to_the_session_prepared_target() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE execute_bind_accounts (id int4 PRIMARY KEY, name text)",
    )
    .unwrap();
    submit_text(
        &engine,
        &mut session,
        "INSERT INTO execute_bind_accounts VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();
    let target = engine
        .prepare_statement(
            &session,
            "SELECT id FROM execute_bind_accounts WHERE id = $1 AND id = $2 AND name = $3 ORDER BY id",
            &[
                Some(LogicalType::Int4),
                Some(LogicalType::Int4),
                Some(LogicalType::Text),
            ],
        )
        .unwrap();
    let mut extended = ExtendedSession::default();
    extended
        .install_sql_prepared("lookup".to_string(), target)
        .unwrap();

    let request = extended
        .prepare_request(
            "outer_lookup".to_string(),
            "EXECUTE lookup($1, $1, $2)".to_string(),
            &[],
            SessionTransactionStatus::Idle,
        )
        .unwrap();
    let analysis = ExtendedSession::analyze_prepare(&engine, &mut session, &request);
    extended.complete_parse(request, analysis).unwrap();
    assert_eq!(
        extended.statements["outer_lookup"].parameter_oids,
        vec![23, 25]
    );
    extended
        .describe_revalidated(
            &engine,
            &session,
            DescribeTarget::Statement,
            "outer_lookup",
            SessionTransactionStatus::Idle,
        )
        .unwrap();
    extended
        .bind(
            "lookup_portal".to_string(),
            "outer_lookup",
            &[],
            &[Some(b"1".to_vec()), Some(b"Ada".to_vec())],
            &[],
        )
        .unwrap();
    let request = extended
        .execution_request("lookup_portal")
        .unwrap()
        .unwrap();
    assert!(matches!(
        submit_prepared(&engine, &mut session, request.query_bound()).unwrap(),
        QueryOutcome::Rows { rows, .. } if rows == vec![vec![DbValue::Int4(1)]]
    ));
}

#[test]
fn extended_sql_execute_describes_then_rejects_a_deferred_negative_limit() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE deferred_limit_accounts (id int4)",
    )
    .unwrap();
    let plan = ExtendedSession::analyze_sql_prepare_plan(
        &engine,
        &session,
        "SELECT id FROM deferred_limit_accounts ORDER BY id LIMIT -1",
        &[],
    )
    .unwrap();
    assert_eq!(
        plan.deferred_execution_error.as_ref().unwrap().message,
        "LIMIT must not be negative"
    );
    let mut extended = ExtendedSession::default();
    extended
        .install_sql_prepared_plan("negative_limit".to_string(), plan)
        .unwrap();
    let request = extended
        .prepare_request(
            "negative_execute".to_string(),
            "EXECUTE negative_limit".to_string(),
            &[],
            SessionTransactionStatus::Idle,
        )
        .unwrap();
    let analysis = ExtendedSession::analyze_prepare(&engine, &mut session, &request);
    extended.complete_parse(request, analysis).unwrap();
    assert_eq!(
        message_tags(
            &extended
                .describe(
                    DescribeTarget::Statement,
                    "negative_execute",
                    SessionTransactionStatus::Idle,
                )
                .unwrap()
        ),
        vec![b't', b'T']
    );
    extended
        .bind(
            "negative_portal".to_string(),
            "negative_execute",
            &[],
            &[],
            &[],
        )
        .unwrap();
    let error = extended.execution_request("negative_portal").unwrap_err();
    assert_eq!(error.message, "LIMIT must not be negative");
}

#[test]
fn extended_sql_execute_binds_typed_literals_before_outer_parameters() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE execute_literal_accounts (id int4 PRIMARY KEY, name text)",
    )
    .unwrap();
    submit_text(
        &engine,
        &mut session,
        "INSERT INTO execute_literal_accounts VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    let target = engine
        .prepare_statement(
            &session,
            "SELECT id FROM execute_literal_accounts WHERE id = $1 AND name = $2",
            &[Some(LogicalType::Int4), Some(LogicalType::Text)],
        )
        .unwrap();
    let mut extended = ExtendedSession::default();
    extended
        .install_sql_prepared("literal_lookup".to_string(), target)
        .unwrap();

    let request = extended
        .prepare_request(
            "outer_literal_lookup".to_string(),
            "EXECUTE literal_lookup(($1)::int4, 'Ada')".to_string(),
            &[],
            SessionTransactionStatus::Idle,
        )
        .unwrap();
    let analysis = ExtendedSession::analyze_prepare(&engine, &mut session, &request);
    extended.complete_parse(request, analysis).unwrap();
    assert_eq!(
        extended.statements["outer_literal_lookup"].parameter_oids,
        vec![23]
    );
    extended
        .bind(
            "literal_lookup_portal".to_string(),
            "outer_literal_lookup",
            &[],
            &[Some(b"1".to_vec())],
            &[],
        )
        .unwrap();
    let request = extended
        .execution_request("literal_lookup_portal")
        .unwrap()
        .unwrap();
    assert!(matches!(
        submit_prepared(&engine, &mut session, request.query_bound()).unwrap(),
        QueryOutcome::Rows { rows, .. } if rows == vec![vec![DbValue::Int4(1)]]
    ));
    let error = extended
        .prepare_request(
            "invalid_literal_lookup".to_string(),
            "EXECUTE literal_lookup('not-an-int', 'Ada')".to_string(),
            &[],
            SessionTransactionStatus::Idle,
        )
        .unwrap_err();
    assert_eq!(error.code, "22P02");
    assert_eq!(
        error.message,
        "invalid input syntax for parameter type oid 23: \"not-an-int\""
    );
    let error = extended
        .prepare_request(
            "conflicting_literal_lookup".to_string(),
            "EXECUTE literal_lookup($1, $1)".to_string(),
            &[],
            SessionTransactionStatus::Idle,
        )
        .unwrap_err();
    assert_eq!(error.code, "42P08");
    assert_eq!(
        error.message,
        "inconsistent parameter types for SQL EXECUTE placeholder"
    );
}

#[test]
fn describe_emits_canonical_oids_sizes_and_numeric_typmod_for_all_logical_types() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE numeric_metadata (\
             i2 INT2, i4 INT4, i8 INT8, amount NUMERIC(12,4), active BOOL, note TEXT,\
             day DATE, created_at TIMESTAMP, ident UUID\
         )",
    )
    .unwrap();
    let mut extended = ExtendedSession::default();
    extended
        .parse(
            "numeric_lookup".to_string(),
            "SELECT i2, i4, i8, amount, active, note, day, created_at, ident FROM numeric_metadata",
            &[],
            |sql, hints| engine.prepare_statement(&session, sql, hints),
        )
        .unwrap();
    assert_eq!(
        extended.statements["numeric_lookup"].columns,
        vec![
            ColumnMeta {
                name: "i2".to_string(),
                logical_type: LogicalType::Int2,
                numeric_typmod: None
            },
            ColumnMeta {
                name: "i4".to_string(),
                logical_type: LogicalType::Int4,
                numeric_typmod: None
            },
            ColumnMeta {
                name: "i8".to_string(),
                logical_type: LogicalType::Int8,
                numeric_typmod: None
            },
            ColumnMeta {
                name: "amount".to_string(),
                logical_type: LogicalType::Numeric,
                numeric_typmod: Some((12, 4))
            },
            ColumnMeta {
                name: "active".to_string(),
                logical_type: LogicalType::Bool,
                numeric_typmod: None
            },
            ColumnMeta {
                name: "note".to_string(),
                logical_type: LogicalType::Text,
                numeric_typmod: None
            },
            ColumnMeta {
                name: "day".to_string(),
                logical_type: LogicalType::Date,
                numeric_typmod: None
            },
            ColumnMeta {
                name: "created_at".to_string(),
                logical_type: LogicalType::Timestamp,
                numeric_typmod: None
            },
            ColumnMeta {
                name: "ident".to_string(),
                logical_type: LogicalType::Uuid,
                numeric_typmod: None
            },
        ],
        "Parse retains the facade's full typed result description before wire encoding"
    );
    let description = extended
        .describe(
            DescribeTarget::Statement,
            "numeric_lookup",
            SessionTransactionStatus::Idle,
        )
        .unwrap();
    let tags = message_tags(&description);
    let row_description = message_payloads(&description)
        .into_iter()
        .zip(tags)
        .find_map(|(payload, tag)| (tag == b'T').then_some(payload))
        .expect("Describe statement must emit a RowDescription");
    let field_count = i16::from_be_bytes(row_description[..2].try_into().unwrap());
    assert_eq!(field_count, 9);
    let mut offset = 2;
    let mut actual = Vec::new();
    for _ in 0..field_count {
        let name_end = row_description[offset..]
            .iter()
            .position(|byte| *byte == 0)
            .map(|length| offset + length)
            .expect("column-name terminator");
        let name = std::str::from_utf8(&row_description[offset..name_end])
            .unwrap()
            .to_string();
        offset = name_end + 1;
        let table_oid = u32::from_be_bytes(row_description[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let attribute_number =
            i16::from_be_bytes(row_description[offset..offset + 2].try_into().unwrap());
        offset += 2;
        let oid = u32::from_be_bytes(row_description[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let type_size = i16::from_be_bytes(row_description[offset..offset + 2].try_into().unwrap());
        offset += 2;
        let typmod = i32::from_be_bytes(row_description[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let format = i16::from_be_bytes(row_description[offset..offset + 2].try_into().unwrap());
        offset += 2;
        assert_eq!((table_oid, attribute_number, format), (0, 0, 0));
        actual.push((name, oid, type_size, typmod));
    }
    assert_eq!(offset, row_description.len());
    assert_eq!(
        actual,
        vec![
            ("i2".to_string(), 21, 2, -1),
            ("i4".to_string(), 23, 4, -1),
            ("i8".to_string(), 20, 8, -1),
            ("amount".to_string(), 1700, -1, 786_440),
            ("active".to_string(), 16, 1, -1),
            ("note".to_string(), 25, -1, -1),
            ("day".to_string(), 1082, 4, -1),
            ("created_at".to_string(), 1114, 8, -1),
            ("ident".to_string(), 2950, 16, -1),
        ],
        "Describe must emit canonical PostgreSQL OIDs/type sizes, including numeric typmod"
    );
}

#[test]
fn parse_and_describe_follow_only_the_session_private_catalog() {
    let engine = SharedEngine::new();
    let mut creator = engine.open_session();
    let observer = engine.open_session();
    submit_text(&engine, &mut creator, "BEGIN").unwrap();
    submit_text(
        &engine,
        &mut creator,
        "CREATE TABLE private_extended_predecessor (id int4)",
    )
    .unwrap();
    submit_text(
        &engine,
        &mut creator,
        "CREATE TABLE private_extended (id int4 PRIMARY KEY, value text)",
    )
    .unwrap();

    let mut extended = ExtendedSession::default();
    extended
        .parse(
            "private_insert".to_string(),
            "INSERT INTO private_extended VALUES ($1, $2) RETURNING id, value",
            &[],
            |sql, hints| engine.prepare_statement(&creator, sql, hints),
        )
        .unwrap();
    assert_eq!(
        extended.statements["private_insert"].parameter_oids,
        vec![23, 25]
    );
    assert_eq!(
        extended.statements["private_insert"]
            .columns
            .iter()
            .map(|column| (column.name.as_str(), column.logical_type))
            .collect::<Vec<_>>(),
        vec![("id", LogicalType::Int4), ("value", LogicalType::Text)]
    );
    assert_eq!(
        message_tags(
            &extended
                .describe_revalidated(
                    &engine,
                    &creator,
                    DescribeTarget::Statement,
                    "private_insert",
                    creator.transaction_status(),
                )
                .unwrap()
        ),
        vec![b't', b'T']
    );

    let mut observer_extended = ExtendedSession::default();
    assert_eq!(
        observer_extended
            .parse(
                "observer".to_string(),
                "SELECT value FROM private_extended WHERE id = $1",
                &[],
                |sql, hints| engine.prepare_statement(&observer, sql, hints),
            )
            .unwrap_err()
            .code,
        "42P01"
    );

    submit_text(&engine, &mut creator, "ROLLBACK").unwrap();
    assert_eq!(
        extended
            .describe_revalidated(
                &engine,
                &creator,
                DescribeTarget::Statement,
                "private_insert",
                creator.transaction_status(),
            )
            .unwrap_err()
            .code,
        "42P01"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn private_parse_bind_execute_reads_its_write_and_commits_once() {
    let engine = gpu_db_engine::Engine::new_local();
    engine.set_shard_residency_enabled(true);
    let engine = SharedEngine::from_engine(engine);
    let mut creator = engine.open_session();
    let mut observer = engine.open_session();
    submit_text(&engine, &mut creator, "BEGIN").unwrap();
    submit_text(
        &engine,
        &mut creator,
        "CREATE TABLE private_pbe (id int4 PRIMARY KEY, value int8)",
    )
    .unwrap();

    let mut extended = ExtendedSession::default();
    extended
        .parse(
            "insert_private".to_string(),
            "INSERT INTO private_pbe VALUES ($1, $2) RETURNING value",
            &[],
            |sql, hints| engine.prepare_statement(&creator, sql, hints),
        )
        .unwrap();
    assert_eq!(
        message_tags(
            &extended
                .describe_revalidated(
                    &engine,
                    &creator,
                    DescribeTarget::Statement,
                    "insert_private",
                    creator.transaction_status(),
                )
                .unwrap()
        ),
        vec![b't', b'T']
    );
    extended
        .bind(
            "insert_portal".to_string(),
            "insert_private",
            &[],
            &[Some(b"1".to_vec()), Some(b"99".to_vec())],
            &[],
        )
        .unwrap();
    let insert = extended
        .execution_request("insert_portal")
        .unwrap()
        .unwrap();
    assert!(matches!(
        submit_prepared(&engine, &mut creator, insert.query_bound()).unwrap(),
        QueryOutcome::Returning { rows, rows_affected: 1, .. }
            if rows == vec![vec![DbValue::Int8(99)]]
    ));

    let mut observer_extended = ExtendedSession::default();
    assert_eq!(
        observer_extended
            .parse(
                "observer_before_commit".to_string(),
                "SELECT value FROM private_pbe WHERE id = $1",
                &[],
                |sql, hints| engine.prepare_statement(&observer, sql, hints),
            )
            .unwrap_err()
            .code,
        "42P01"
    );

    extended
        .parse(
            "select_private".to_string(),
            "SELECT value FROM private_pbe WHERE id = $1",
            &[],
            |sql, hints| engine.prepare_statement(&creator, sql, hints),
        )
        .unwrap();
    extended
        .bind(
            "select_portal".to_string(),
            "select_private",
            &[],
            &[Some(b"1".to_vec())],
            &[],
        )
        .unwrap();
    let select = extended
        .execution_request("select_portal")
        .unwrap()
        .unwrap();
    assert!(matches!(
        submit_prepared(&engine, &mut creator, select.query_bound()).unwrap(),
        QueryOutcome::Rows { rows, .. }
            if rows == vec![vec![DbValue::Int8(99)]]
    ));

    submit_text(&engine, &mut creator, "COMMIT").unwrap();
    observer_extended
        .parse(
            "observer_after_commit".to_string(),
            "SELECT value FROM private_pbe WHERE id = $1",
            &[],
            |sql, hints| engine.prepare_statement(&observer, sql, hints),
        )
        .unwrap();
    observer_extended
        .bind(
            "observer_portal".to_string(),
            "observer_after_commit",
            &[],
            &[Some(b"1".to_vec())],
            &[],
        )
        .unwrap();
    let observer_select = observer_extended
        .execution_request("observer_portal")
        .unwrap()
        .unwrap();
    assert!(matches!(
        submit_prepared(&engine, &mut observer, observer_select.query_bound()).unwrap(),
        QueryOutcome::Rows { rows, .. }
            if rows == vec![vec![DbValue::Int8(99)]]
    ));
}

#[test]
fn named_duplicates_fail_and_unnamed_entries_replace() {
    let engine = SharedEngine::new();
    let session = engine.open_session();
    let mut extended = ExtendedSession::default();
    for _ in 0..2 {
        extended
            .parse("".to_string(), "BEGIN", &[], |sql, hints| {
                engine.prepare_statement(&session, sql, hints)
            })
            .unwrap();
    }
    extended
        .parse("named".to_string(), "BEGIN", &[], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    assert_eq!(
        extended
            .parse("named".to_string(), "BEGIN", &[], |sql, hints| {
                engine.prepare_statement(&session, sql, hints)
            })
            .unwrap_err()
            .code,
        "42P05"
    );
    assert!(matches!(
        extended.dispatch(
            FrontendMessage::Parse {
                statement_name: "named".to_string(),
                query: "not valid sql".to_string(),
                parameter_type_oids: Vec::new(),
            },
            SessionTransactionStatus::Idle,
        ),
        Dispatch::Response(Err(error)) if error.code != "42P05"
    ));

    extended
        .bind("named_portal".to_string(), "named", &[], &[], &[])
        .unwrap();
    assert_eq!(
        extended
            .bind("named_portal".to_string(), "named", &[], &[], &[])
            .unwrap_err()
            .code,
        "42P03"
    );
    assert_eq!(
        extended
            .bind("named_portal".to_string(), "missing", &[], &[], &[])
            .unwrap_err()
            .code,
        "26000"
    );
    assert_eq!(
        extended
            .bind("named_portal".to_string(), "named", &[], &[], &[0, 0],)
            .unwrap_err()
            .code,
        "42P03"
    );
    for _ in 0..2 {
        extended.bind("".to_string(), "", &[], &[], &[]).unwrap();
    }
    assert_eq!(
        message_tags(
            &extended
                .bind("no_data_formats".to_string(), "named", &[], &[], &[0, 1],)
                .unwrap()
        ),
        vec![b'2']
    );
}

#[test]
fn portals_follow_transaction_and_statement_lifetimes() {
    let engine = SharedEngine::new();
    let session = engine.open_session();
    let mut extended = ExtendedSession::default();
    extended
        .parse("statement".to_string(), "BEGIN", &[], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    extended
        .bind("portal".to_string(), "statement", &[], &[], &[])
        .unwrap();

    extended.sync(true);
    assert!(extended.execution_request("portal").is_ok());
    extended.sync(false);
    assert_eq!(
        extended.execution_request("portal").unwrap_err().code,
        "34000"
    );

    extended
        .bind("portal".to_string(), "statement", &[], &[], &[])
        .unwrap();
    assert_eq!(
        message_tags(
            &extended
                .close(DescribeTarget::Statement, "statement")
                .unwrap()
        ),
        vec![b'3']
    );
    assert!(extended.execution_request("portal").is_err());
    assert_eq!(
        message_tags(
            &extended
                .close(DescribeTarget::Statement, "missing")
                .unwrap()
        ),
        vec![b'3']
    );
    assert_eq!(
        message_tags(&extended.close(DescribeTarget::Portal, "missing").unwrap()),
        vec![b'3']
    );
}

#[test]
fn unnamed_replacement_starts_only_after_statement_and_shape_validation() {
    let engine = SharedEngine::new();
    let session = engine.open_session();
    let mut extended = ExtendedSession::default();
    extended
        .parse("".to_string(), "BEGIN", &[], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    extended.bind("".to_string(), "", &[], &[], &[]).unwrap();
    extended.bind("old".to_string(), "", &[], &[], &[]).unwrap();
    assert!(extended
        .parse("".to_string(), "not valid sql", &[], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .is_err());
    assert!(extended
        .describe(
            DescribeTarget::Statement,
            "",
            SessionTransactionStatus::Idle,
        )
        .is_err());
    assert!(extended.execution_request("").is_ok());
    assert!(extended.execution_request("old").is_ok());

    extended
        .parse("named".to_string(), "BEGIN", &[], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    extended
        .bind("".to_string(), "named", &[], &[], &[])
        .unwrap();
    assert!(extended
        .bind("".to_string(), "missing", &[], &[], &[])
        .is_err());
    assert!(extended.execution_request("").is_ok());

    // Once lookup, shape validation, and the failed-transaction gate pass, CreatePortal owns
    // the unnamed replacement. A later codec error leaves the previous unnamed portal gone.
    extended
        .parse("typed".to_string(), "BEGIN", &[23], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    assert_eq!(
        extended
            .bind("".to_string(), "typed", &[2], &[Some(b"7".to_vec())], &[],)
            .unwrap_err()
            .code,
        "22023"
    );
    assert!(extended.execution_request("").is_err());
}

#[test]
fn empty_extended_statement_reaches_empty_query_response() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    let mut extended = ExtendedSession::default();
    extended
        .parse("empty".to_string(), " ; ", &[], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    extended
        .bind("empty_portal".to_string(), "empty", &[], &[], &[])
        .unwrap();
    let request = extended.execution_request("empty_portal").unwrap().unwrap();
    let outcome = submit_prepared(&engine, &mut session, request.query_bound());
    extended
        .set_execution_outcome("empty_portal", outcome)
        .unwrap();
    assert_eq!(
        message_tags(&extended.encode_execute("empty_portal", 0).unwrap()),
        vec![b'I']
    );
    assert_eq!(
        message_tags(&extended.encode_execute("empty_portal", 0).unwrap()),
        vec![b'I']
    );
}

#[test]
fn completed_non_row_portal_rejects_a_second_execute() {
    let engine = SharedEngine::new();
    let session = engine.open_session();
    let mut extended = ExtendedSession::default();
    extended
        .parse("begin".to_string(), "BEGIN", &[], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    extended
        .bind("portal".to_string(), "begin", &[], &[], &[])
        .unwrap();
    extended
        .set_execution_outcome(
            "portal",
            Ok(QueryOutcome::Command {
                tag: CommandTag::Begin,
                rows_affected: None,
            }),
        )
        .unwrap();
    assert_eq!(
        message_tags(&extended.encode_execute("portal", 0).unwrap()),
        vec![b'C']
    );
    assert_eq!(
        extended.encode_execute("portal", 0).unwrap_err().code,
        "55000"
    );
}

#[test]
fn failed_transaction_preserves_postgresql_message_precedence() {
    let engine = SharedEngine::new();
    let mut engine_session = engine.open_session();
    submit_text(
        &engine,
        &mut engine_session,
        "CREATE TABLE precedence_rows (id int4)",
    )
    .unwrap();
    let mut extended = ExtendedSession::default();
    extended
        .parse("normal".to_string(), "BEGIN", &[], |sql, hints| {
            engine.prepare_statement(&engine_session, sql, hints)
        })
        .unwrap();
    extended
        .parse("exit".to_string(), "ROLLBACK", &[], |sql, hints| {
            engine.prepare_statement(&engine_session, sql, hints)
        })
        .unwrap();
    extended
        .parse("exit_typed".to_string(), "ROLLBACK", &[23], |sql, hints| {
            engine.prepare_statement(&engine_session, sql, hints)
        })
        .unwrap();
    extended
        .parse("empty".to_string(), " ; ", &[], |sql, hints| {
            engine.prepare_statement(&engine_session, sql, hints)
        })
        .unwrap();
    extended
        .parse(
            "rows".to_string(),
            "SELECT id FROM precedence_rows",
            &[],
            |sql, hints| engine.prepare_statement(&engine_session, sql, hints),
        )
        .unwrap();
    extended
        .bind("cached".to_string(), "normal", &[], &[], &[])
        .unwrap();
    extended
        .bind("rows_portal".to_string(), "rows", &[], &[], &[])
        .unwrap();
    extended
        .bind("empty_portal".to_string(), "empty", &[], &[], &[])
        .unwrap();

    let response = extended.dispatch(
        FrontendMessage::Parse {
            statement_name: "malformed".to_string(),
            query: "not valid sql".to_string(),
            parameter_type_oids: Vec::new(),
        },
        SessionTransactionStatus::FailedTransaction,
    );
    assert!(matches!(response, Dispatch::Response(Err(error)) if error.code != "25P02"));

    let Dispatch::Prepare(empty_request) = extended.dispatch(
        FrontendMessage::Parse {
            statement_name: "empty_while_failed".to_string(),
            query: " ; ".to_string(),
            parameter_type_oids: Vec::new(),
        },
        SessionTransactionStatus::FailedTransaction,
    ) else {
        panic!("an empty Parse must bypass the failed-transaction gate");
    };
    let empty_prepared =
        ExtendedSession::analyze_prepare(&engine, &mut engine_session, &empty_request);
    assert!(extended
        .complete_parse(*empty_request, empty_prepared)
        .is_ok());

    let response = extended.dispatch(
        FrontendMessage::Parse {
            statement_name: "blocked".to_string(),
            query: "BEGIN".to_string(),
            parameter_type_oids: Vec::new(),
        },
        SessionTransactionStatus::FailedTransaction,
    );
    assert!(matches!(response, Dispatch::Response(Err(error)) if error.code == "25P02"));

    let response = extended.dispatch(
        FrontendMessage::Bind {
            portal_name: "blocked".to_string(),
            statement_name: "missing".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        },
        SessionTransactionStatus::FailedTransaction,
    );
    assert!(matches!(response, Dispatch::Response(Err(error)) if error.code == "26000"));

    let response = extended.dispatch(
        FrontendMessage::Bind {
            portal_name: "blocked".to_string(),
            statement_name: "normal".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: Vec::new(),
        },
        SessionTransactionStatus::FailedTransaction,
    );
    assert!(matches!(response, Dispatch::Response(Err(error)) if error.code == "08P01"));

    let response = extended.dispatch(
        FrontendMessage::Bind {
            portal_name: "blocked".to_string(),
            statement_name: "normal".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        },
        SessionTransactionStatus::FailedTransaction,
    );
    assert!(matches!(response, Dispatch::Response(Err(error)) if error.code == "25P02"));

    let response = extended.dispatch(
        FrontendMessage::Bind {
            portal_name: "blocked_result_shape".to_string(),
            statement_name: "normal".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: vec![0, 0],
        },
        SessionTransactionStatus::FailedTransaction,
    );
    assert!(matches!(response, Dispatch::Response(Err(error)) if error.code == "25P02"));

    let response = extended.dispatch(
        FrontendMessage::Execute {
            portal_name: "missing".to_string(),
            max_rows: 0,
        },
        SessionTransactionStatus::FailedTransaction,
    );
    assert!(matches!(response, Dispatch::Response(Err(error)) if error.code == "34000"));

    let response = extended.dispatch(
        FrontendMessage::Execute {
            portal_name: "cached".to_string(),
            max_rows: 0,
        },
        SessionTransactionStatus::FailedTransaction,
    );
    assert!(matches!(response, Dispatch::Response(Err(error)) if error.code == "25P02"));

    assert!(matches!(
        extended.dispatch(
            FrontendMessage::Execute {
                portal_name: "empty_portal".to_string(),
                max_rows: 0,
            },
            SessionTransactionStatus::FailedTransaction,
        ),
        Dispatch::Execute { .. }
    ));

    assert_eq!(
        extended
            .describe_revalidated(
                &engine,
                &engine_session,
                DescribeTarget::Statement,
                "missing",
                SessionTransactionStatus::FailedTransaction,
            )
            .unwrap_err()
            .code,
        "26000"
    );
    assert!(extended
        .describe_revalidated(
            &engine,
            &engine_session,
            DescribeTarget::Statement,
            "normal",
            SessionTransactionStatus::FailedTransaction,
        )
        .is_ok());
    assert_eq!(
        extended
            .describe_revalidated(
                &engine,
                &engine_session,
                DescribeTarget::Statement,
                "rows",
                SessionTransactionStatus::FailedTransaction,
            )
            .unwrap_err()
            .code,
        "25P02"
    );
    assert_eq!(
        extended
            .describe_revalidated(
                &engine,
                &engine_session,
                DescribeTarget::Portal,
                "missing",
                SessionTransactionStatus::FailedTransaction,
            )
            .unwrap_err()
            .code,
        "34000"
    );
    assert!(extended
        .describe_revalidated(
            &engine,
            &engine_session,
            DescribeTarget::Portal,
            "cached",
            SessionTransactionStatus::FailedTransaction,
        )
        .is_ok());
    assert_eq!(
        extended
            .describe_revalidated(
                &engine,
                &engine_session,
                DescribeTarget::Portal,
                "rows_portal",
                SessionTransactionStatus::FailedTransaction,
            )
            .unwrap_err()
            .code,
        "25P02"
    );

    let response = extended.dispatch(
        FrontendMessage::Bind {
            portal_name: "exit".to_string(),
            statement_name: "exit".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: Vec::new(),
            result_format_codes: Vec::new(),
        },
        SessionTransactionStatus::FailedTransaction,
    );
    assert!(matches!(response, Dispatch::Bind(_)));

    let response = extended.dispatch(
        FrontendMessage::Bind {
            portal_name: "exit_typed".to_string(),
            statement_name: "exit_typed".to_string(),
            parameter_format_codes: Vec::new(),
            parameters: vec![Some(b"1".to_vec())],
            result_format_codes: Vec::new(),
        },
        SessionTransactionStatus::FailedTransaction,
    );
    assert!(matches!(response, Dispatch::Response(Err(error)) if error.code == "25P02"));
}

#[test]
fn lifecycle_actions_own_skip_and_simple_query_boundaries() {
    let mut extended = ExtendedSession::default();
    assert_eq!(
        extended.before_dispatch_transaction_action(
            &FrontendMessage::Parse {
                statement_name: String::new(),
                query: "BEGIN".to_string(),
                parameter_type_oids: Vec::new(),
            },
            SessionTransactionStatus::Idle,
        ),
        TransactionAction::BeginImplicit
    );
    extended.complete_transaction_action(TransactionAction::BeginImplicit, true);
    assert_eq!(
        TransactionAction::CommitImplicit.failure_cleanup(),
        TransactionAction::RollbackImplicit
    );
    assert_eq!(
        extended.simple_query_completion_action(&Ok(QueryOutcome::Command {
            tag: CommandTag::Rollback,
            rows_affected: None,
        })),
        TransactionAction::None
    );

    extended.fail();
    assert_eq!(
        extended.skipping_frame_action(b'B'),
        SkippingFrameAction::Discard
    );
    assert_eq!(
        extended.skipping_frame_action(b'S'),
        SkippingFrameAction::Sync
    );
}

#[test]
fn parse_declares_typed_unused_trailing_parameters() {
    let engine = SharedEngine::new();
    let session = engine.open_session();
    let mut extended = ExtendedSession::default();
    extended
        .parse("lookup".to_string(), "BEGIN", &[23], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    assert_eq!(extended.statements["lookup"].parameter_oids, vec![23]);
    assert_eq!(
        message_tags(
            &extended
                .bind(
                    "portal".to_string(),
                    "lookup",
                    &[],
                    &[Some(b"7".to_vec())],
                    &[],
                )
                .unwrap()
        ),
        vec![b'2']
    );
    assert_eq!(
        extended
            .bind("missing_value".to_string(), "lookup", &[], &[], &[])
            .unwrap_err()
            .code,
        "08P01"
    );

    extended
        .parse("empty_typed".to_string(), " ; ", &[23], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    assert_eq!(extended.statements["empty_typed"].parameter_oids, vec![23]);
    assert!(extended
        .bind(
            "empty_typed_portal".to_string(),
            "empty_typed",
            &[],
            &[Some(b"8".to_vec())],
            &[],
        )
        .is_ok());
}

#[test]
fn execute_chunks_a_cached_result_and_never_requests_reexecution() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE accounts (id int4 PRIMARY KEY)",
    )
    .unwrap();
    let mut extended = ExtendedSession::default();
    extended
        .parse(
            "lookup".to_string(),
            "SELECT id FROM accounts",
            &[],
            |sql, hints| engine.prepare_statement(&session, sql, hints),
        )
        .unwrap();
    extended
        .bind("portal".to_string(), "lookup", &[], &[], &[])
        .unwrap();
    extended
        .set_execution_outcome(
            "portal",
            Ok(QueryOutcome::Rows {
                columns: vec![ColumnMeta {
                    name: "id".to_string(),
                    logical_type: LogicalType::Int4,
                    numeric_typmod: None,
                }],
                rows: vec![
                    vec![DbValue::Int4(1)],
                    vec![DbValue::Int4(2)],
                    vec![DbValue::Int4(3)],
                ],
            }),
        )
        .unwrap();
    assert!(extended.execution_request("portal").unwrap().is_none());

    let first = extended.encode_execute("portal", 2).unwrap();
    assert_eq!(message_tags(&first), vec![b'D', b'D', b's']);
    let second = extended.encode_execute("portal", 2).unwrap();
    assert_eq!(message_tags(&second), vec![b'D', b'C']);
    let repeated = extended.encode_execute("portal", 0).unwrap();
    assert_eq!(message_tags(&repeated), vec![b'C']);
    assert_eq!(message_payloads(&repeated), vec![b"SELECT 0\0".to_vec()]);

    extended
        .bind("exact".to_string(), "lookup", &[], &[], &[])
        .unwrap();
    extended
        .set_execution_outcome(
            "exact",
            Ok(QueryOutcome::Rows {
                columns: vec![ColumnMeta {
                    name: "id".to_string(),
                    logical_type: LogicalType::Int4,
                    numeric_typmod: None,
                }],
                rows: vec![vec![DbValue::Int4(1)], vec![DbValue::Int4(2)]],
            }),
        )
        .unwrap();
    assert_eq!(
        message_tags(&extended.encode_execute("exact", 2).unwrap()),
        vec![b'D', b'D', b's']
    );
    let exact_eof = extended.encode_execute("exact", 2).unwrap();
    assert_eq!(message_tags(&exact_eof), vec![b'C']);
    assert_eq!(message_payloads(&exact_eof), vec![b"SELECT 0\0".to_vec()]);
}

#[test]
fn bind_accepts_binary_bool_results_after_supported_parameters() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE notes (id int4 PRIMARY KEY, active bool)",
    )
    .unwrap();
    let mut extended = ExtendedSession::default();
    extended
        .parse(
            "insert_note".to_string(),
            "INSERT INTO notes VALUES ($1, true) RETURNING active",
            &[],
            |sql, hints| engine.prepare_statement(&session, sql, hints),
        )
        .unwrap();
    assert_eq!(
        message_tags(
            &extended
                .bind(
                    "portal".to_string(),
                    "insert_note",
                    &[],
                    &[Some(b"1".to_vec())],
                    &[1],
                )
                .unwrap(),
        ),
        vec![b'2']
    );
    assert!(extended.execution_request("portal").is_ok());
}

#[test]
fn bind_codec_errors_use_postgresql_semantic_sqlstates() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(&engine, &mut session, "CREATE TABLE codec_rows (id int4)").unwrap();
    let mut extended = ExtendedSession::default();
    for (statement, oid) in [
        ("int2_arg", 21),
        ("int4_arg", 23),
        ("int8_arg", 20),
        ("numeric_arg", 1700),
        ("bool_arg", 16),
        ("text_arg", 25),
        ("date_arg", 1082),
        ("timestamp_arg", 1114),
        ("uuid_arg", 2950),
    ] {
        extended
            .parse(statement.to_string(), "BEGIN", &[oid], |sql, hints| {
                engine.prepare_statement(&session, sql, hints)
            })
            .unwrap();
    }
    extended
        .parse(
            "rows".to_string(),
            "SELECT id FROM codec_rows",
            &[],
            |sql, hints| engine.prepare_statement(&session, sql, hints),
        )
        .unwrap();

    // Each facade logical type is exercised at the server codec boundary. Fixed-width values
    // use a deliberately short binary payload; variable-width NUMERIC/TEXT use malformed wire
    // shapes (short header / invalid UTF-8) because arbitrary payload widths are otherwise valid.
    for (portal, statement, formats, values, expected) in vec![
        (
            "unsupported",
            "int4_arg",
            vec![2],
            vec![Some(b"7".to_vec())],
            "22023",
        ),
        (
            "bad_int2_text",
            "int2_arg",
            vec![0],
            vec![Some(b"int2?".to_vec())],
            "22P02",
        ),
        (
            "bad_int4_text",
            "int4_arg",
            vec![0],
            vec![Some(b"int4?".to_vec())],
            "22P02",
        ),
        (
            "bad_int8_text",
            "int8_arg",
            vec![0],
            vec![Some(b"int8?".to_vec())],
            "22P02",
        ),
        (
            "bad_numeric_text",
            "numeric_arg",
            vec![0],
            vec![Some(b"numeric?".to_vec())],
            "22P02",
        ),
        (
            "bad_numeric_nondecimal_separator",
            "numeric_arg",
            vec![0],
            vec![Some(b"0x__2".to_vec())],
            "22P02",
        ),
        (
            "bad_bool_text",
            "bool_arg",
            vec![0],
            vec![Some(b"maybe".to_vec())],
            "22P02",
        ),
        (
            "bad_text_utf8",
            "text_arg",
            vec![0],
            vec![Some(vec![0xff])],
            "22P02",
        ),
        (
            "bad_date_text",
            "date_arg",
            vec![0],
            vec![Some(b"date?".to_vec())],
            "22P02",
        ),
        (
            "bad_timestamp_text",
            "timestamp_arg",
            vec![0],
            vec![Some(b"timestamp?".to_vec())],
            "22P02",
        ),
        (
            "bad_uuid_text",
            "uuid_arg",
            vec![0],
            vec![Some(b"uuid?".to_vec())],
            "22P02",
        ),
        (
            "bad_int2_binary",
            "int2_arg",
            vec![1],
            vec![Some(vec![0])],
            "22P03",
        ),
        (
            "bad_int4_binary",
            "int4_arg",
            vec![1],
            vec![Some(vec![0; 3])],
            "22P03",
        ),
        (
            "bad_int8_binary",
            "int8_arg",
            vec![1],
            vec![Some(vec![0; 7])],
            "22P03",
        ),
        (
            "bad_numeric_binary",
            "numeric_arg",
            vec![1],
            vec![Some(vec![0; 7])],
            "22P03",
        ),
        (
            "bad_numeric_dscale_reserved",
            "numeric_arg",
            vec![1],
            vec![Some(vec![0, 0, 0, 0, 0, 0, 0x40, 0])],
            "22P03",
        ),
        (
            "numeric_dscale_out_of_range",
            "numeric_arg",
            vec![1],
            vec![Some(vec![0, 0, 0, 0, 0, 0, 1, 0])],
            "22003",
        ),
        (
            "bad_bool_binary",
            "bool_arg",
            vec![1],
            vec![Some(vec![])],
            "22P03",
        ),
        (
            "bad_text_binary",
            "text_arg",
            vec![1],
            vec![Some(vec![0xff])],
            "22P03",
        ),
        (
            "bad_date_binary",
            "date_arg",
            vec![1],
            vec![Some(vec![0; 3])],
            "22P03",
        ),
        (
            "bad_timestamp_binary",
            "timestamp_arg",
            vec![1],
            vec![Some(vec![0; 7])],
            "22P03",
        ),
        (
            "bad_uuid_binary",
            "uuid_arg",
            vec![1],
            vec![Some(vec![0; 15])],
            "22P03",
        ),
        (
            "bad_text_binary_utf8",
            "text_arg",
            vec![1],
            vec![Some(vec![0xff])],
            "22P03",
        ),
        (
            "bad_text_nul",
            "text_arg",
            vec![0],
            vec![Some(b"nul\0byte".to_vec())],
            "22P02",
        ),
        (
            "bad_text_binary_nul",
            "text_arg",
            vec![1],
            vec![Some(b"nul\0byte".to_vec())],
            "22P03",
        ),
    ] {
        assert_eq!(
            extended
                .bind(portal.to_string(), statement, &formats, &values, &[])
                .unwrap_err()
                .code,
            expected
        );
    }
    let error = extended
        .bind(
            "bad_int4_message".to_string(),
            "int4_arg",
            &[0],
            &[Some(b"not-an-int".to_vec())],
            &[],
        )
        .unwrap_err();
    assert_eq!(error.code, "22P02");
    assert_eq!(
        error.message,
        "invalid input syntax for parameter type oid 23: \"not-an-int\""
    );
    for (portal, value) in [
        ("numeric_hex", b"0x2a".as_slice()),
        ("numeric_decimal_underscores", b"1_500.25_00".as_slice()),
        ("numeric_exponent_underscores", b"1e1_0".as_slice()),
    ] {
        assert!(
            extended
                .bind(
                    portal.to_string(),
                    "numeric_arg",
                    &[0],
                    &[Some(value.to_vec())],
                    &[],
                )
                .is_ok(),
            "server bind accepts PG16 finite NUMERIC text {value:?}"
        );
    }
    for (portal, statement, value) in [
        ("int2_range", "int2_arg", b"0x8_000".as_slice()),
        ("int4_range", "int4_arg", b"0o2_000_000_0000".as_slice()),
        (
            "int8_range",
            "int8_arg",
            b"0x8000_0000_0000_0000".as_slice(),
        ),
    ] {
        assert_eq!(
            extended
                .bind(
                    portal.to_string(),
                    statement,
                    &[0],
                    &[Some(value.to_vec())],
                    &[]
                )
                .unwrap_err()
                .code,
            "22003",
            "well-formed {statement} text outside its signed range must not become 22P02"
        );
    }
    assert_eq!(
        extended
            .bind("bad_result_count".to_string(), "rows", &[], &[], &[0, 0])
            .unwrap_err()
            .code,
        "08P01"
    );
    assert_eq!(
        extended
            .bind(
                "bad_binary_before_result_count".to_string(),
                "int4_arg",
                &[1],
                &[Some(vec![0, 1, 2])],
                &[0, 0],
            )
            .unwrap_err()
            .code,
        "22P03"
    );
    assert_eq!(
        extended
            .bind("bad_result".to_string(), "rows", &[], &[], &[2])
            .unwrap_err()
            .code,
        "22023"
    );
}

#[test]
fn ddl_changed_returning_type_fails_before_the_prepared_write() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE prepared_shape (x int4)",
    )
    .unwrap();
    let mut extended = ExtendedSession::default();
    extended
        .parse(
            "stale_insert".to_string(),
            "INSERT INTO prepared_shape VALUES ('hello') RETURNING x",
            &[],
            |sql, hints| engine.prepare_statement(&session, sql, hints),
        )
        .unwrap();
    submit_text(&engine, &mut session, "DROP TABLE prepared_shape").unwrap();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE prepared_shape (x text)",
    )
    .unwrap();

    let statement_error = extended
        .describe_revalidated(
            &engine,
            &session,
            DescribeTarget::Statement,
            "stale_insert",
            SessionTransactionStatus::Idle,
        )
        .unwrap_err();
    assert_eq!(statement_error.code, "0A000");

    // Bind still owns the Parse-time int4 metadata. Execute must revalidate that proof and
    // fail before mutation admission; binary result encoding can never become a post-commit
    // failure.
    extended
        .bind("stale_portal".to_string(), "stale_insert", &[], &[], &[1])
        .unwrap();
    let portal_error = extended
        .describe_revalidated(
            &engine,
            &session,
            DescribeTarget::Portal,
            "stale_portal",
            SessionTransactionStatus::Idle,
        )
        .unwrap_err();
    assert_eq!(portal_error.code, "0A000");
    let request = extended.execution_request("stale_portal").unwrap().unwrap();
    let error = submit_prepared(&engine, &mut session, request.query_bound()).unwrap_err();
    assert_eq!(error.category, gpu_db_facade::ErrorCategory::Unsupported);

    let QueryOutcome::Rows { rows, .. } =
        submit_text(&engine, &mut session, "SELECT x FROM prepared_shape").unwrap()
    else {
        panic!("SELECT must return rows");
    };
    assert!(
        rows.is_empty(),
        "the rejected prepared write must be effect-free"
    );
}

#[test]
fn ddl_changed_numeric_typmod_rejects_stale_statement_and_portal_before_execution() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE prepared_numeric_shape (amount NUMERIC(12,2))",
    )
    .unwrap();
    let mut extended = ExtendedSession::default();
    extended
        .parse(
            "stale_numeric_insert".to_string(),
            "INSERT INTO prepared_numeric_shape VALUES ($1) RETURNING amount",
            &[],
            |sql, hints| engine.prepare_statement(&session, sql, hints),
        )
        .unwrap();
    extended
        .bind(
            "stale_numeric_portal".to_string(),
            "stale_numeric_insert",
            &[],
            &[Some(b"1.00".to_vec())],
            &[1],
        )
        .unwrap();

    submit_text(&engine, &mut session, "DROP TABLE prepared_numeric_shape").unwrap();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE prepared_numeric_shape (amount NUMERIC(12,3))",
    )
    .unwrap();

    for (target, name) in [
        (DescribeTarget::Statement, "stale_numeric_insert"),
        (DescribeTarget::Portal, "stale_numeric_portal"),
    ] {
        let error = extended
            .describe_revalidated(
                &engine,
                &session,
                target,
                name,
                SessionTransactionStatus::Idle,
            )
            .expect_err("a changed NUMERIC typmod must invalidate cached metadata");
        assert_eq!(error.code, "0A000");
    }

    // The portal's immutable bound description is rechecked before engine admission, so a
    // typmod-only schema change cannot mutate the replacement relation under stale metadata.
    let request = extended
        .execution_request("stale_numeric_portal")
        .unwrap()
        .unwrap();
    let error = submit_prepared(&engine, &mut session, request.query_bound()).unwrap_err();
    assert_eq!(error.category, gpu_db_facade::ErrorCategory::Unsupported);
    let QueryOutcome::Rows { rows, .. } = submit_text(
        &engine,
        &mut session,
        "SELECT amount FROM prepared_numeric_shape",
    )
    .unwrap() else {
        panic!("SELECT must return rows");
    };
    assert!(
        rows.is_empty(),
        "stale portal must fail before write admission"
    );
}

#[test]
fn ddl_changed_inferred_parameter_type_rejects_describe_before_cached_metadata() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE prepared_parameter_shape (x int4)",
    )
    .unwrap();
    let mut extended = ExtendedSession::default();
    extended
        .parse(
            "stale_parameter".to_string(),
            "INSERT INTO prepared_parameter_shape VALUES ($1)",
            &[],
            |sql, hints| engine.prepare_statement(&session, sql, hints),
        )
        .unwrap();

    submit_text(&engine, &mut session, "DROP TABLE prepared_parameter_shape").unwrap();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE prepared_parameter_shape (x text)",
    )
    .unwrap();

    // The prepared statement has no result columns, so this specifically proves stale
    // ParameterDescription is rejected. Returning Err means no ParameterDescription or
    // NoData bytes were constructed for the caller to emit.
    let error = extended
        .describe_revalidated(
            &engine,
            &session,
            DescribeTarget::Statement,
            "stale_parameter",
            SessionTransactionStatus::Idle,
        )
        .unwrap_err();
    assert_eq!(error.code, "42804");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_w1_executes_once_through_engine_session_and_returns_typed_rows() {
    let engine = gpu_db_engine::Engine::new_local();
    engine.set_auto_admit_on_commit(true);
    let engine = SharedEngine::from_engine(engine);
    let mut session = engine.open_session();
    submit_text(
        &engine,
        &mut session,
        "CREATE TABLE accounts (id int4 PRIMARY KEY, balance int8)",
    )
    .unwrap();
    submit_text(
        &engine,
        &mut session,
        "INSERT INTO accounts VALUES (7, 100)",
    )
    .unwrap();

    let mut extended = ExtendedSession::default();
    extended
        .parse(
            "w1".to_string(),
            "UPDATE accounts SET balance = balance + $2::int8 \
             WHERE id = $1::int4 RETURNING balance",
            &[],
            |sql, hints| engine.prepare_statement(&session, sql, hints),
        )
        .unwrap();
    extended
        .bind(
            "w1_portal".to_string(),
            "w1",
            &[1],
            &[
                Some(7_i32.to_be_bytes().to_vec()),
                Some((-9_i64).to_be_bytes().to_vec()),
            ],
            &[1],
        )
        .unwrap();
    let request = extended.execution_request("w1_portal").unwrap().unwrap();
    let outcome = submit_prepared(&engine, &mut session, request.query_bound());
    assert!(matches!(
        &outcome,
        Ok(QueryOutcome::Returning { rows, rows_affected: 1, .. })
            if rows == &vec![vec![DbValue::Int8(91)]]
    ));
    extended
        .set_execution_outcome("w1_portal", outcome)
        .unwrap();
    assert!(extended.execution_request("w1_portal").unwrap().is_none());
    assert_eq!(
        message_tags(&extended.encode_execute("w1_portal", 0).unwrap()),
        vec![b'D', b'C']
    );
}

fn message_tags(bytes: &[u8]) -> Vec<u8> {
    let mut tags = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        tags.push(bytes[offset]);
        let len = u32::from_be_bytes(bytes[offset + 1..offset + 5].try_into().unwrap()) as usize;
        offset += len + 1;
    }
    tags
}

fn message_payloads(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut payloads = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let len = u32::from_be_bytes(bytes[offset + 1..offset + 5].try_into().unwrap()) as usize;
        payloads.push(bytes[offset + 5..offset + len + 1].to_vec());
        offset += len + 1;
    }
    payloads
}
