use super::*;

fn submit_text(
    shared: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    shared
        .submit(session, SubmissionRequest::Text(sql))
        .into_immediate()
}

#[test]
fn canonical_session_submission_owns_transaction_control() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_text(&shared, &mut session, "BEGIN").unwrap();
    assert!(session.in_transaction());
    submit_text(&shared, &mut session, "ROLLBACK").unwrap();
    assert!(!session.in_transaction());
}

#[test]
fn compatibility_reads_stay_outside_mutation_admission() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_text(&shared, &mut session, "SET answer=forty-two").unwrap();
    submit_text(&shared, &mut session, "CREATE SEQUENCE seq").unwrap();
    let (visible_before, wal_before) = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };

    assert!(matches!(
        submit_text(&shared, &mut session, "GET answer").unwrap(),
        QueryOutcome::Command { .. }
    ));
    assert!(matches!(
        submit_text(&shared, &mut session, "SELECT pg_advisory_unlock_all()").unwrap(),
        QueryOutcome::Rows { rows, .. } if rows == vec![vec![DbValue::Null]]
    ));
    assert_eq!(
        submit_text(&shared, &mut session, "SELECT currval('seq')")
            .unwrap_err()
            .category,
        ErrorCategory::InvalidRequest,
        "currval is a real WAL-neutral session read and is undefined before nextval"
    );
    assert_eq!(
        submit_text(&shared, &mut session, "SELECT bounded_fn()")
            .unwrap_err()
            .category,
        ErrorCategory::Engine,
        "arbitrary function calls must use catalog lookup rather than the cleanup escape hatch"
    );

    let engine = shared.read_engine().unwrap();
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
}

#[test]
fn unresolved_insert_returning_fails_before_sequence_or_wal_claim() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    let (visible_before, wal_before) = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };
    let error = submit_text(
        &shared,
        &mut session,
        "INSERT INTO missing VALUES (1) RETURNING id",
    )
    .unwrap_err();
    assert_eq!(error.category, ErrorCategory::UndefinedRelation);
    assert_eq!(error.message, "relation does not exist");
    let engine = shared.read_engine().unwrap();
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
}

#[test]
fn typed_copy_uses_canonical_admission_and_explicit_transaction_publication() {
    let shared = SharedEngine::new();
    let mut writer = shared.open_session();
    let mut observer = shared.open_session();
    submit_text(
        &shared,
        &mut writer,
        "CREATE TABLE copy_people (id INT, name TEXT)",
    )
    .unwrap();
    let copy = CopyFromStdin {
        table: "copy_people".to_string(),
        columns: Some(vec!["id".to_string(), "name".to_string()]),
        options: gpu_db_sql::CopyOptions::CSV,
    };

    let described = shared
        .submit(&mut writer, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .unwrap();
    let QueryOutcome::CopyIn { target } = described else {
        panic!("COPY start must return typed input columns")
    };
    assert_eq!(
        target
            .columns()
            .iter()
            .map(|column| (&*column.name, column.logical_type))
            .collect::<Vec<_>>(),
        vec![("id", LogicalType::Int4), ("name", LogicalType::Text)]
    );

    let (visible_before, wal_before) = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };
    assert_eq!(
        shared
            .submit(
                &mut writer,
                SubmissionRequest::CopyFrom {
                    target: &target,
                    rows: vec![vec![DbValue::Int4(1), DbValue::Text("Ada".to_string())]],
                },
            )
            .into_immediate()
            .unwrap(),
        QueryOutcome::Command {
            tag: CommandTag::Copy,
            rows_affected: Some(1),
        }
    );
    let engine = shared.read_engine().unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert_eq!(engine.visible_up_to(), visible_before + 1);

    submit_text(&shared, &mut writer, "BEGIN").unwrap();
    let wal_before_private = shared.read_engine().unwrap().durable_wal_records().len();
    shared
        .submit(
            &mut writer,
            SubmissionRequest::CopyFrom {
                target: &target,
                rows: vec![vec![DbValue::Int4(2), DbValue::Null]],
            },
        )
        .into_immediate()
        .unwrap();
    let QueryOutcome::Rows { rows, .. } = submit_text(
        &shared,
        &mut writer,
        "SELECT name FROM copy_people WHERE id = 2",
    )
    .unwrap() else {
        panic!("writer COPY row must be readable in its private GPU generation")
    };
    assert_eq!(rows, vec![vec![DbValue::Null]]);
    let QueryOutcome::Rows { rows, .. } = submit_text(
        &shared,
        &mut observer,
        "SELECT name FROM copy_people WHERE id = 2",
    )
    .unwrap() else {
        panic!("observer SELECT must return rows outcome")
    };
    assert!(rows.is_empty(), "uncommitted COPY must not publish");
    assert_eq!(
        shared.read_engine().unwrap().durable_wal_records().len(),
        wal_before_private,
        "staged COPY claims no WAL"
    );
    submit_text(&shared, &mut writer, "ROLLBACK").unwrap();
    assert_eq!(
        shared.read_engine().unwrap().durable_wal_records().len(),
        wal_before_private
    );

    submit_text(&shared, &mut writer, "BEGIN").unwrap();
    shared
        .submit(
            &mut writer,
            SubmissionRequest::CopyFrom {
                target: &target,
                rows: vec![vec![DbValue::Int4(3), DbValue::Text(String::new())]],
            },
        )
        .into_immediate()
        .unwrap();
    submit_text(&shared, &mut writer, "COMMIT").unwrap();
    assert_eq!(
        shared.read_engine().unwrap().durable_wal_records().len(),
        wal_before_private + 1,
        "COMMIT owns the sole WAL claim for staged COPY"
    );
    let QueryOutcome::Rows { rows, .. } = submit_text(
        &shared,
        &mut observer,
        "SELECT name FROM copy_people WHERE id = 3",
    )
    .unwrap() else {
        panic!("observer SELECT must return rows outcome")
    };
    assert_eq!(rows, vec![vec![DbValue::Text(String::new())]]);
}

/// Retain protocol COPY target proofs across a concurrent pair while the table is device resident.
/// This is the facade-level counterpart to the pgwire lifetime hazard: every acknowledged COPY
/// must remain visible after the pair completes, including a NULL in an unreferenced column.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn concurrent_gpu_copy_targets_retain_every_acknowledged_row() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(64);
    engine.set_shard_index_probe_enabled(true);
    engine.set_shard_batched_point_read_enabled(true);
    engine.set_auto_admit_on_commit(true);
    let shared = Arc::new(SharedEngine::from_engine(engine));
    let mut setup = shared.open_session();
    submit_text(
        &shared,
        &mut setup,
        "CREATE TABLE facade_copy_hazard (id INT PRIMARY KEY, note INT)",
    )
    .unwrap();
    let seed = (0..128)
        .map(|id| format!("({id}, {id})"))
        .collect::<Vec<_>>()
        .join(",");
    submit_text(
        &shared,
        &mut setup,
        &format!("INSERT INTO facade_copy_hazard (id, note) VALUES {seed}"),
    )
    .unwrap();

    let copy = CopyFromStdin {
        table: "facade_copy_hazard".to_string(),
        columns: Some(vec!["id".to_string(), "note".to_string()]),
        options: gpu_db_sql::CopyOptions::CSV,
    };
    for (id, note) in [
        (1_001, DbValue::Null),
        (1_002, DbValue::Int4(0)),
        (1_003, DbValue::Int4(7)),
    ] {
        let QueryOutcome::CopyIn { target } = shared
            .submit(&mut setup, SubmissionRequest::CopyFromStart(&copy))
            .into_immediate()
            .unwrap()
        else {
            panic!("COPY start must return a target proof")
        };
        assert!(matches!(
            shared
                .submit(
                    &mut setup,
                    SubmissionRequest::CopyFrom {
                        target: &target,
                        rows: vec![vec![DbValue::Int4(id), note]],
                    },
                )
                .into_immediate()
                .unwrap(),
            QueryOutcome::Command {
                tag: CommandTag::Copy,
                rows_affected: Some(1),
            }
        ));
    }

    let mut left_session = shared.open_session();
    let QueryOutcome::CopyIn {
        target: left_target,
    } = shared
        .submit(&mut left_session, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .unwrap()
    else {
        panic!("left COPY start must return a target proof")
    };
    let mut right_session = shared.open_session();
    let QueryOutcome::CopyIn {
        target: right_target,
    } = shared
        .submit(&mut right_session, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .unwrap()
    else {
        panic!("right COPY start must return a target proof")
    };
    let barrier = Arc::new(std::sync::Barrier::new(2));
    std::thread::scope(|scope| {
        let left_shared = Arc::clone(&shared);
        let left_barrier = Arc::clone(&barrier);
        scope.spawn(move || {
            left_barrier.wait();
            assert!(matches!(
                left_shared
                    .submit(
                        &mut left_session,
                        SubmissionRequest::CopyFrom {
                            target: &left_target,
                            rows: vec![vec![DbValue::Int4(1_004), DbValue::Null]],
                        },
                    )
                    .into_immediate()
                    .unwrap(),
                QueryOutcome::Command {
                    tag: CommandTag::Copy,
                    rows_affected: Some(1),
                }
            ));
        });
        let right_shared = Arc::clone(&shared);
        let right_barrier = Arc::clone(&barrier);
        scope.spawn(move || {
            right_barrier.wait();
            assert!(matches!(
                right_shared
                    .submit(
                        &mut right_session,
                        SubmissionRequest::CopyFrom {
                            target: &right_target,
                            rows: vec![vec![DbValue::Int4(1_005), DbValue::Int4(0)]],
                        },
                    )
                    .into_immediate()
                    .unwrap(),
                QueryOutcome::Command {
                    tag: CommandTag::Copy,
                    rows_affected: Some(1),
                }
            ));
        });
    });

    let QueryOutcome::Rows { rows, .. } = submit_text(
        &shared,
        &mut setup,
        "SELECT id, note FROM facade_copy_hazard WHERE id >= 1001 ORDER BY id",
    )
    .unwrap() else {
        panic!("COPY rows must remain queryable")
    };
    assert_eq!(
        rows,
        vec![
            vec![DbValue::Int4(1_001), DbValue::Null],
            vec![DbValue::Int4(1_002), DbValue::Int4(0)],
            vec![DbValue::Int4(1_003), DbValue::Int4(7)],
            vec![DbValue::Int4(1_004), DbValue::Null],
            vec![DbValue::Int4(1_005), DbValue::Int4(0)],
        ]
    );
}

#[test]
fn copy_description_errors_are_pre_effect_and_fail_an_explicit_transaction() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_text(
        &shared,
        &mut session,
        "CREATE TABLE valid_copy_target (id INT PRIMARY KEY)",
    )
    .unwrap();
    let valid = CopyFromStdin {
        table: "valid_copy_target".to_string(),
        columns: None,
        options: gpu_db_sql::CopyOptions::TEXT,
    };
    let QueryOutcome::CopyIn {
        target: valid_target,
    } = shared
        .submit(&mut session, SubmissionRequest::CopyFromStart(&valid))
        .into_immediate()
        .unwrap()
    else {
        panic!("valid COPY start must return a target")
    };
    let missing = CopyFromStdin {
        table: "missing_copy_target".to_string(),
        columns: None,
        options: gpu_db_sql::CopyOptions::TEXT,
    };
    let before = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };
    let error = shared
        .submit(&mut session, SubmissionRequest::CopyFromStart(&missing))
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::UndefinedRelation);
    let engine = shared.read_engine().unwrap();
    assert_eq!(
        (engine.visible_up_to(), engine.durable_wal_records().len()),
        before
    );

    submit_text(&shared, &mut session, "BEGIN").unwrap();
    let error = shared
        .submit(&mut session, SubmissionRequest::CopyFromStart(&missing))
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::UndefinedRelation);
    assert_eq!(
        session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );
    let error = shared
        .revalidate_copy_target(&session, &valid_target)
        .unwrap_err();
    assert_eq!(
        error.category,
        ErrorCategory::InFailedTransaction,
        "failed-transaction precedence must outrank target/catalog revalidation"
    );
    submit_text(&shared, &mut session, "ROLLBACK").unwrap();
    let engine = shared.read_engine().unwrap();
    assert_eq!(
        (engine.visible_up_to(), engine.durable_wal_records().len()),
        before
    );
}

#[test]
fn zero_row_copy_target_blocks_typed_reset_until_protocol_drop() {
    let shared = SharedEngine::new();
    let mut copy_session = shared.open_session();
    let mut reset_session = shared.open_session();
    submit_text(
        &shared,
        &mut reset_session,
        "CREATE TABLE copy_reset_guard (id INT PRIMARY KEY)",
    )
    .unwrap();
    let copy = CopyFromStdin {
        table: "copy_reset_guard".to_string(),
        columns: None,
        options: gpu_db_sql::CopyOptions::TEXT,
    };
    let QueryOutcome::CopyIn { target } = shared
        .submit(&mut copy_session, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .unwrap()
    else {
        panic!("COPY start must retain a protocol target")
    };
    let before = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };

    let error = submit_text(&shared, &mut reset_session, "TRUNCATE copy_reset_guard").unwrap_err();
    assert_eq!(error.category, ErrorCategory::Serialization, "{error:?}");
    let engine = shared.read_engine().unwrap();
    assert_eq!(
        (engine.visible_up_to(), engine.durable_wal_records().len()),
        before,
        "even a zero-row COPY target owns the relation until protocol cancellation/drop"
    );
    drop(target);
}

#[test]
fn copy_target_proof_rejects_drop_recreate_before_any_copy_effect() {
    let shared = SharedEngine::new();
    let mut copy_session = shared.open_session();
    let mut ddl_session = shared.open_session();
    submit_text(
        &shared,
        &mut ddl_session,
        "CREATE TABLE copy_generation (id INT PRIMARY KEY, name TEXT)",
    )
    .unwrap();
    let copy = CopyFromStdin {
        table: "copy_generation".to_string(),
        columns: Some(vec!["id".to_string(), "name".to_string()]),
        options: gpu_db_sql::CopyOptions::CSV,
    };
    let QueryOutcome::CopyIn { target } = shared
        .submit(&mut copy_session, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .unwrap()
    else {
        panic!("COPY start must retain its target proof")
    };

    submit_text(&shared, &mut ddl_session, "DROP TABLE copy_generation").unwrap();
    submit_text(
        &shared,
        &mut ddl_session,
        "CREATE TABLE copy_generation (id INT PRIMARY KEY, name TEXT)",
    )
    .unwrap();
    let before_copy = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };
    let error = shared
        .submit(
            &mut copy_session,
            SubmissionRequest::CopyFrom {
                target: &target,
                rows: Vec::new(),
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::Serialization);
    {
        let engine = shared.read_engine().unwrap();
        assert_eq!(
            (engine.visible_up_to(), engine.durable_wal_records().len()),
            before_copy,
            "stale COPY 0 must fail before sequence, WAL, or publication"
        );
    }
    let error = shared
        .submit(
            &mut copy_session,
            SubmissionRequest::CopyFrom {
                target: &target,
                rows: vec![vec![DbValue::Int4(9), DbValue::Text("stale".to_string())]],
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::Serialization);
    let engine = shared.read_engine().unwrap();
    assert_eq!(
        (engine.visible_up_to(), engine.durable_wal_records().len()),
        before_copy,
        "a stale COPY target must fail before sequence, WAL, or publication"
    );
    let QueryOutcome::Rows { rows, .. } =
        submit_text(&shared, &mut copy_session, "SELECT id FROM copy_generation").unwrap()
    else {
        panic!("replacement table SELECT must return rows")
    };
    assert!(rows.is_empty());
}

#[test]
fn explicit_transaction_copy_revalidates_its_target_before_staging_empty_rows() {
    let shared = SharedEngine::new();
    let mut copy_session = shared.open_session();
    let mut ddl_session = shared.open_session();
    submit_text(
        &shared,
        &mut ddl_session,
        "CREATE TABLE copy_txn_stale (id INT PRIMARY KEY, name TEXT)",
    )
    .unwrap();
    submit_text(&shared, &mut copy_session, "BEGIN").unwrap();
    let copy = CopyFromStdin {
        table: "copy_txn_stale".to_string(),
        columns: Some(vec!["id".to_string(), "name".to_string()]),
        options: gpu_db_sql::CopyOptions::CSV,
    };
    let QueryOutcome::CopyIn { target } = shared
        .submit(&mut copy_session, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .unwrap()
    else {
        panic!("transaction COPY start must return a target")
    };
    assert_eq!(
        target.transaction_identity, None,
        "a published target remains portable but must be revalidated against the transaction"
    );
    for ddl in [
        "DROP TABLE copy_txn_stale",
        "CREATE TABLE copy_txn_stale (id INT PRIMARY KEY, name TEXT)",
    ] {
        submit_text(&shared, &mut ddl_session, ddl).unwrap();
    }
    let before_copy = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };
    let error = shared
        .submit(
            &mut copy_session,
            SubmissionRequest::CopyFrom {
                target: &target,
                rows: Vec::new(),
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::Serialization);
    assert_eq!(
        copy_session.transaction_status(),
        SessionTransactionStatus::FailedTransaction
    );
    let engine = shared.read_engine().unwrap();
    assert_eq!(
        (engine.visible_up_to(), engine.durable_wal_records().len()),
        before_copy,
        "stale explicit COPY 0 must not stage a delta, claim WAL, or publish"
    );
    submit_text(&shared, &mut copy_session, "ROLLBACK").unwrap();
}

#[test]
fn prepared_copy_to_rejects_a_mutating_bound_owner_before_any_effect() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_text(
        &shared,
        &mut session,
        "CREATE TABLE copy_to_owner_guard (id INT PRIMARY KEY)",
    )
    .unwrap();
    let mutating = shared
        .prepare_statement(
            &session,
            "INSERT INTO copy_to_owner_guard (id) VALUES (7) RETURNING id",
            &[],
        )
        .unwrap()
        .bind_values(&[])
        .unwrap();
    let copy = CopyToStdout {
        table: "copy_to_owner_guard".to_string(),
        columns: Some(vec!["id".to_string()]),
        options: gpu_db_sql::CopyOptions::CSV,
    };
    let before = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };
    let error = shared
        .submit(
            &mut session,
            SubmissionRequest::PreparedCopyTo {
                copy: &copy,
                bound: &mutating,
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::InvalidRequest);
    let engine = shared.read_engine().unwrap();
    assert_eq!(
        (engine.visible_up_to(), engine.durable_wal_records().len()),
        before,
        "mismatched prepared COPY TO owner must fail before mutation admission"
    );
    let QueryOutcome::Rows { rows, .. } =
        submit_text(&shared, &mut session, "SELECT id FROM copy_to_owner_guard").unwrap()
    else {
        panic!("guard table SELECT must return rows")
    };
    assert!(rows.is_empty());
}

#[test]
fn copy_target_is_bound_to_the_shared_engine_that_described_it() {
    let source = SharedEngine::new();
    let destination = SharedEngine::new();
    let mut source_session = source.open_session();
    let mut destination_session = destination.open_session();
    for (shared, session) in [
        (&source, &mut source_session),
        (&destination, &mut destination_session),
    ] {
        submit_text(
            shared,
            session,
            "CREATE TABLE copy_engine_owner (id INT PRIMARY KEY, name TEXT)",
        )
        .unwrap();
    }
    let copy = CopyFromStdin {
        table: "copy_engine_owner".to_string(),
        columns: Some(vec!["id".to_string(), "name".to_string()]),
        options: gpu_db_sql::CopyOptions::CSV,
    };
    let QueryOutcome::CopyIn {
        target: source_target,
    } = source
        .submit(&mut source_session, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .unwrap()
    else {
        panic!("source COPY start must return a target")
    };
    let QueryOutcome::CopyIn {
        target: destination_target,
    } = destination
        .submit(
            &mut destination_session,
            SubmissionRequest::CopyFromStart(&copy),
        )
        .into_immediate()
        .unwrap()
    else {
        panic!("destination COPY start must return a target")
    };
    assert_eq!(
        source_target.proof, destination_target.proof,
        "the regression requires independently built engines with value-identical relation proofs"
    );
    assert_ne!(source_target, destination_target);

    let before = {
        let engine = destination.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };
    let error = destination
        .revalidate_copy_target(&destination_session, &source_target)
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::InvalidRequest);
    let error = destination
        .submit(
            &mut destination_session,
            SubmissionRequest::CopyFrom {
                target: &source_target,
                rows: vec![vec![
                    DbValue::Int4(9),
                    DbValue::Text("wrong engine".to_string()),
                ]],
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::InvalidRequest);
    let engine = destination.read_engine().unwrap();
    assert_eq!(
        (engine.visible_up_to(), engine.durable_wal_records().len()),
        before,
        "cross-engine COPY must fail before transaction identity, WAL, or publication"
    );
    let QueryOutcome::Rows { rows, .. } = submit_text(
        &destination,
        &mut destination_session,
        "SELECT id FROM copy_engine_owner",
    )
    .unwrap() else {
        panic!("destination table SELECT must return rows")
    };
    assert!(rows.is_empty());
}

#[test]
fn copy_target_is_bound_to_its_transaction_catalog_context() {
    let shared = SharedEngine::new();
    let mut session = shared.open_session();
    submit_text(&shared, &mut session, "BEGIN").unwrap();
    submit_text(
        &shared,
        &mut session,
        "CREATE TABLE copy_txn_owner (id INT PRIMARY KEY, name TEXT)",
    )
    .unwrap();
    let copy = CopyFromStdin {
        table: "copy_txn_owner".to_string(),
        columns: Some(vec!["id".to_string(), "name".to_string()]),
        options: gpu_db_sql::CopyOptions::CSV,
    };
    let QueryOutcome::CopyIn { target: stale } = shared
        .submit(&mut session, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .unwrap()
    else {
        panic!("transaction-private COPY start must return a target")
    };
    let private_txn_id = stale
        .transaction_identity
        .expect("target must retain its private transaction identity");
    submit_text(&shared, &mut session, "ROLLBACK").unwrap();
    submit_text(
        &shared,
        &mut session,
        "CREATE TABLE copy_txn_owner (id INT PRIMARY KEY, name TEXT)",
    )
    .unwrap();
    let QueryOutcome::CopyIn { target: current } = shared
        .submit(&mut session, SubmissionRequest::CopyFromStart(&copy))
        .into_immediate()
        .unwrap()
    else {
        panic!("published COPY start must return a target")
    };
    assert_eq!(
        stale.proof, current.proof,
        "the regression requires rollback/recreate to reuse a value-identical relation proof"
    );
    assert_eq!(current.transaction_identity, None);
    assert_eq!(stale.transaction_identity, Some(private_txn_id));

    let before = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };
    let error = shared.revalidate_copy_target(&session, &stale).unwrap_err();
    assert_eq!(error.category, ErrorCategory::Serialization);
    let error = shared
        .submit(
            &mut session,
            SubmissionRequest::CopyFrom {
                target: &stale,
                rows: vec![vec![
                    DbValue::Int4(9),
                    DbValue::Text("stale transaction".to_string()),
                ]],
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::Serialization);
    let engine = shared.read_engine().unwrap();
    assert_eq!(
        (engine.visible_up_to(), engine.durable_wal_records().len()),
        before,
        "a transaction-bound target must fail before WAL or publication after rollback"
    );
}

#[test]
fn private_and_published_value_identical_targets_keep_overlay_provenance() {
    let shared = SharedEngine::new();
    let mut private_session = shared.open_session();
    let mut published_session = shared.open_session();
    submit_text(&shared, &mut private_session, "BEGIN").unwrap();
    submit_text(
        &shared,
        &mut private_session,
        "CREATE TABLE copy_overlay_twin (id INT PRIMARY KEY, name TEXT)",
    )
    .unwrap();
    let copy = CopyFromStdin {
        table: "copy_overlay_twin".to_string(),
        columns: Some(vec!["id".to_string(), "name".to_string()]),
        options: gpu_db_sql::CopyOptions::CSV,
    };
    let QueryOutcome::CopyIn {
        target: private_target,
    } = submit_copy_from_start_with_hook(&shared, &mut private_session, &copy, || {
        submit_text(
            &shared,
            &mut published_session,
            "CREATE TABLE copy_overlay_twin (id INT PRIMARY KEY, name TEXT)",
        )
        .unwrap();
    })
    .unwrap()
    else {
        panic!("private COPY start must return a target")
    };
    let QueryOutcome::CopyIn {
        target: published_target,
    } = shared
        .submit(
            &mut published_session,
            SubmissionRequest::CopyFromStart(&copy),
        )
        .into_immediate()
        .unwrap()
    else {
        panic!("published COPY start must return a target")
    };
    assert_eq!(
        private_target.proof, published_target.proof,
        "the adversarial private/global twin must have value-identical metadata"
    );
    assert!(
        private_target.transaction_identity.is_some(),
        "overlay provenance, not proof equality, must bind the private target"
    );
    assert_eq!(published_target.transaction_identity, None);

    submit_text(&shared, &mut private_session, "ROLLBACK").unwrap();
    let before = {
        let engine = shared.read_engine().unwrap();
        (engine.visible_up_to(), engine.durable_wal_records().len())
    };
    let error = shared
        .submit(
            &mut published_session,
            SubmissionRequest::CopyFrom {
                target: &private_target,
                rows: vec![vec![
                    DbValue::Int4(9),
                    DbValue::Text("must not cross overlay".to_string()),
                ]],
            },
        )
        .into_immediate()
        .unwrap_err();
    assert_eq!(error.category, ErrorCategory::Serialization);
    let engine = shared.read_engine().unwrap();
    assert_eq!(
        (engine.visible_up_to(), engine.durable_wal_records().len()),
        before,
        "rolled-back overlay target must not mutate its published value-identical twin"
    );
}
