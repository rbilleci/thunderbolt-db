use super::*;
use gpu_db_facade::{
    BoundPreparedStatement, DbError, DbValue, QueryOutcome, SharedEngine, SharedSession,
    SubmissionRequest,
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
fn bind_rejects_unsupported_binary_results_after_supported_parameters() {
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
        extended
            .bind(
                "portal".to_string(),
                "insert_note",
                &[],
                &[Some(b"1".to_vec())],
                &[1],
            )
            .unwrap_err()
            .code,
        "0A000"
    );
    assert!(extended.execution_request("portal").is_err());
}

#[test]
fn bind_codec_errors_use_postgresql_semantic_sqlstates() {
    let engine = SharedEngine::new();
    let mut session = engine.open_session();
    submit_text(&engine, &mut session, "CREATE TABLE codec_rows (id int4)").unwrap();
    let mut extended = ExtendedSession::default();
    extended
        .parse("int_arg".to_string(), "BEGIN", &[23], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    extended
        .parse("uuid_arg".to_string(), "BEGIN", &[2950], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    extended
        .parse("text_arg".to_string(), "BEGIN", &[25], |sql, hints| {
            engine.prepare_statement(&session, sql, hints)
        })
        .unwrap();
    extended
        .parse(
            "rows".to_string(),
            "SELECT id FROM codec_rows",
            &[],
            |sql, hints| engine.prepare_statement(&session, sql, hints),
        )
        .unwrap();

    for (portal, statement, formats, values, expected) in [
        (
            "unsupported",
            "int_arg",
            vec![2],
            vec![Some(b"7".to_vec())],
            "22023",
        ),
        (
            "bad_text",
            "int_arg",
            vec![0],
            vec![Some(b"not-an-int".to_vec())],
            "22P02",
        ),
        (
            "bad_int_binary",
            "int_arg",
            vec![1],
            vec![Some(vec![0, 1, 2])],
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
            "bad_text_utf8",
            "text_arg",
            vec![0],
            vec![Some(vec![0xff])],
            "22P02",
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
                "int_arg",
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
