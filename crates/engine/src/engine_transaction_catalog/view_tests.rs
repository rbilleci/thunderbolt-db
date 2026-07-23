use super::*;
use std::sync::mpsc;
use std::time::Duration;

fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
    gpu_db_sql::ParsedCommand::parse(sql).unwrap()
}

fn operation_payload(record: &gpu_db_wal::WalRecord) -> Arc<[u8]> {
    let envelope = gpu_db_wal::decode_canonical_record_payload(&record.payload)
        .unwrap()
        .unwrap();
    let operation = envelope
        .fragments
        .iter()
        .find(|fragment| fragment.kind != gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus)
        .unwrap();
    Engine::decode_engine_operation(&operation.body).unwrap()
}

fn select(sql: &str) -> Select {
    let Command::Select(select) = parse_command(sql).unwrap() else {
        panic!("expected SELECT");
    };
    select
}

#[test]
fn transactional_view_is_private_describable_durable_and_recoverable() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            2_000,
            parsed("CREATE TABLE view_source (id int4, note text)"),
        )
        .unwrap();
    engine.submit_transaction(2_001, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            2_001,
            parsed("CREATE VIEW private_view AS SELECT id, note FROM view_source WHERE id >= 10"),
        )
        .unwrap();

    assert!(!engine
        .catalog_snapshot()
        .relational_views
        .contains_key("private_view"));
    let prepared = gpu_db_sql::PreparedCommand::parse("SELECT * FROM private_view").unwrap();
    let description = engine
        .describe_prepared_command_in_transaction(2_001, &prepared, &[])
        .unwrap();
    assert_eq!(
        description
            .result_columns
            .iter()
            .map(|column| (column.name.as_str(), column.ty))
            .collect::<Vec<_>>(),
        vec![("id", SqlType::Int4), ("note", SqlType::Text)]
    );
    assert!(matches!(
        engine.describe_prepared_command(&prepared, &[]),
        Err(ExecuteError::UndefinedRelation(name)) if name == "private_view"
    ));
    assert!(engine
        .execute_relational_select_in_transaction(2_001, &select("SELECT * FROM private_view"))
        .unwrap()
        .rows
        .is_empty());
    assert_eq!(engine.durable_wal_records().len(), 1);

    engine.submit_transaction(2_001, parsed("COMMIT")).unwrap();
    let records = engine.durable_wal_records();
    assert_eq!(records.len(), 2);
    let envelope = gpu_db_wal::decode_canonical_record_payload(&records[1].payload)
        .unwrap()
        .unwrap();
    assert_eq!(envelope.header.operation_count, 2);
    assert_eq!(envelope.header.table_block_count, 0);
    assert_eq!(
        envelope.fragments[0].kind,
        gpu_db_wal::CanonicalFragmentKind::CatalogMutation
    );
    let payload = operation_payload(&records[1]);
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("transactional view must use one typed transaction record");
    };
    assert_eq!(record.catalog_commands.len(), 1);
    assert!(record.created_table_identities.is_empty());
    assert_eq!(record.view_operations.len(), 1);
    let identity = &record.view_operations[0];
    assert_eq!((identity.command_index, identity.ordinal), (0, 0));
    assert!(identity.target_before.is_none());
    assert_eq!(
        identity
            .dependencies
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["view_source"]
    );
    assert_eq!(
        identity.dependencies["view_source"].kind,
        BinaryCatalogRelationKind::Table
    );
    assert_eq!(identity.target_after.kind, BinaryCatalogRelationKind::View);
    assert_eq!(
        record.operation_order,
        vec![BinaryTransactionOperationIdentity::Catalog { command_index: 0 }]
    );

    let request_digest = gpu_db_wal::canonical_request_digest(&payload);
    let (_, affected_rows) = engine
        .commit_state()
        .resolve_transaction_retry_digest_outcome(2_001, request_digest)
        .unwrap()
        .unwrap();
    assert_eq!(affected_rows, 0);
    assert!(engine
        .commit_state()
        .resolve_transaction_retry_digest_outcome(2_001, [0x6d; 32])
        .unwrap_err()
        .to_string()
        .contains("different request"));

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.catalog_snapshot().relational_views["private_view"],
        engine.catalog_snapshot().relational_views["private_view"]
    );
    assert_eq!(
        recovered
            .describe_prepared_command(&prepared, &[])
            .unwrap()
            .result_columns,
        description.result_columns
    );
    assert!(recovered
        .execute_relational_select(&select("SELECT * FROM private_view"))
        .unwrap()
        .rows
        .is_empty());
}

#[test]
fn private_layered_views_replace_in_order_and_rollback_without_effect() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            2_010,
            parsed("CREATE TABLE layered_source (a int4, b text)"),
        )
        .unwrap();
    engine.submit_transaction(2_011, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            2_011,
            parsed("CREATE VIEW layered_one AS SELECT a, b FROM layered_source"),
        )
        .unwrap();
    engine
        .submit_transaction(
            2_011,
            parsed("CREATE OR REPLACE VIEW layered_one AS SELECT a FROM layered_source"),
        )
        .unwrap();
    engine
        .submit_transaction(
            2_011,
            parsed("CREATE VIEW layered_two AS SELECT * FROM layered_one"),
        )
        .unwrap();

    let prepared = gpu_db_sql::PreparedCommand::parse("SELECT * FROM layered_two").unwrap();
    let private = engine
        .describe_prepared_command_in_transaction(2_011, &prepared, &[])
        .unwrap();
    assert_eq!(
        private
            .result_columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        vec!["a"]
    );
    let snapshot = engine.transaction_snapshot_handle(2_011).unwrap();
    let catalog = snapshot.transaction_catalog();
    let first_oid = catalog.relational_views["layered_one"].oid;
    let second_oid = catalog.relational_views["layered_two"].oid;
    assert_ne!(first_oid, second_oid);

    engine
        .submit_transaction(2_011, parsed("ROLLBACK"))
        .unwrap();
    assert!(engine.catalog_snapshot().relational_views.is_empty());
    assert_eq!(engine.durable_wal_records().len(), 1);

    engine.submit_transaction(2_012, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            2_012,
            parsed("CREATE VIEW layered_one AS SELECT a, b FROM layered_source"),
        )
        .unwrap();
    engine
        .submit_transaction(
            2_012,
            parsed("CREATE OR REPLACE VIEW layered_one AS SELECT a FROM layered_source"),
        )
        .unwrap();
    engine.submit_transaction(2_012, parsed("COMMIT")).unwrap();

    let records = engine.durable_wal_records();
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("view replacements must use a typed transaction record");
    };
    assert_eq!(record.view_operations.len(), 2);
    assert!(record.view_operations[0].target_before.is_none());
    assert_eq!(
        record.view_operations[1].target_before,
        Some(record.view_operations[0].target_after.clone())
    );
    assert_eq!(
        record.view_operations[0].target_after.oid,
        record.view_operations[1].target_after.oid
    );
    assert_ne!(
        record.view_operations[0].target_after.digest,
        record.view_operations[1].target_after.digest
    );
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.catalog_snapshot().relational_views["layered_one"],
        engine.catalog_snapshot().relational_views["layered_one"]
    );
}

#[test]
fn view_dependency_or_target_aba_serializes_before_wal() {
    for begin in ["BEGIN", "BEGIN ISOLATION LEVEL REPEATABLE READ"] {
        for replace_target in [false, true] {
            let engine = Engine::new_local();
            engine
                .submit_transaction(2_020, parsed("CREATE TABLE aba_source (id int4)"))
                .unwrap();
            if replace_target {
                engine
                    .submit_transaction(
                        2_021,
                        parsed("CREATE VIEW aba_view AS SELECT id FROM aba_source"),
                    )
                    .unwrap();
            }
            engine.submit_transaction(2_022, parsed(begin)).unwrap();
            let ddl = if replace_target {
                "CREATE OR REPLACE VIEW aba_view AS SELECT id FROM aba_source"
            } else {
                "CREATE VIEW aba_view AS SELECT id FROM aba_source"
            };
            engine.submit_transaction(2_022, parsed(ddl)).unwrap();

            if replace_target {
                engine
                    .submit_transaction(2_023, parsed("DROP VIEW aba_view"))
                    .unwrap();
                engine
                    .submit_transaction(
                        2_024,
                        parsed("CREATE VIEW aba_view AS SELECT id FROM aba_source"),
                    )
                    .unwrap();
            } else {
                engine
                    .submit_transaction(2_023, parsed("DROP TABLE aba_source"))
                    .unwrap();
                engine
                    .submit_transaction(2_024, parsed("CREATE TABLE aba_source (id int4)"))
                    .unwrap();
            }
            let wal_before = engine.durable_wal_records().len();
            let error = engine
                .submit_transaction(2_022, parsed("COMMIT"))
                .unwrap_err();
            assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
            assert_eq!(engine.durable_wal_records().len(), wal_before);
            engine
                .submit_transaction(2_022, parsed("ROLLBACK"))
                .unwrap();
        }
    }
}

#[test]
fn read_committed_rebuilds_private_view_chain_after_same_catalog_publication() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(2_025, parsed("CREATE TABLE rc_view_source (id int4)"))
        .unwrap();
    engine.submit_transaction(2_026, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            2_026,
            parsed("CREATE VIEW rc_view_one AS SELECT id FROM rc_view_source"),
        )
        .unwrap();
    let first = engine
        .transaction_snapshot_handle(2_026)
        .unwrap()
        .transaction_catalog()
        .relational_views["rc_view_one"]
        .clone();

    engine
        .submit_transaction(2_027, parsed("SET rc_view_rebase = yes"))
        .unwrap();
    engine
        .submit_transaction(
            2_026,
            parsed("CREATE VIEW rc_view_two AS SELECT * FROM rc_view_one"),
        )
        .unwrap();
    let private = engine
        .transaction_snapshot_handle(2_026)
        .unwrap()
        .transaction_catalog();
    assert_eq!(private.relational_views["rc_view_one"], first);
    assert!(private.relational_views.contains_key("rc_view_two"));
    engine.submit_transaction(2_026, parsed("COMMIT")).unwrap();

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_eq!(
        recovered.catalog_snapshot().relational_views["rc_view_one"],
        first
    );
    assert_eq!(
        recovered
            .describe_prepared_command(
                &gpu_db_sql::PreparedCommand::parse("SELECT * FROM rc_view_two").unwrap(),
                &[],
            )
            .unwrap()
            .result_columns[0]
            .name,
        "id"
    );
}

#[test]
fn catalog_latch_makes_view_dependency_capture_atomic_with_concurrent_drop() {
    let engine = Arc::new(Engine::new_local());
    engine
        .submit_transaction(2_028, parsed("CREATE TABLE latched_view_source (id int4)"))
        .unwrap();
    let source_oid = engine.catalog_snapshot().relational_catalog["latched_view_source"].oid;
    engine
        .submit_transaction(2_029, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();

    let (release_tx, release_rx) = mpsc::channel();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let racer_engine = Arc::clone(&engine);
    let racer = std::thread::spawn(move || {
        release_rx.recv().unwrap();
        started_tx.send(()).unwrap();
        done_tx
            .send(racer_engine.submit_transaction(2_030, parsed("DROP TABLE latched_view_source")))
            .unwrap();
    });

    let statement = parsed("CREATE VIEW latched_view AS SELECT id FROM latched_view_source");
    let (command, source) = statement.into_parts();
    let hook_engine = Arc::clone(&engine);
    engine
        .execute_catalog_in_transaction_instrumented(2_029, command, source, None, || {
            assert!(matches!(
                hook_engine.catalog_latch.try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ));
            release_tx.send(()).unwrap();
            started_rx.recv().unwrap();
            assert!(matches!(
                done_rx.recv_timeout(Duration::from_millis(100)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
        })
        .unwrap();
    done_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    racer.join().unwrap();

    let snapshot = engine.transaction_snapshot_handle(2_029).unwrap();
    let delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let TransactionOperation::Catalog(staged) = &delta.operations[0] else {
        panic!("expected a catalog operation");
    };
    assert_eq!(
        staged.view_identity.as_ref().unwrap().targets[0].dependencies["latched_view_source"].oid,
        source_oid
    );
    drop(delta);
    let wal_before = engine.durable_wal_records().len();
    let error = engine
        .submit_transaction(2_029, parsed("COMMIT"))
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine
        .submit_transaction(2_029, parsed("ROLLBACK"))
        .unwrap();
}

#[test]
fn view_replay_rejects_every_typed_identity_tamper_before_catalog_effect() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(2_030, parsed("CREATE TABLE proof_source (id int4)"))
        .unwrap();
    engine.submit_transaction(2_031, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            2_031,
            parsed("CREATE VIEW proof_view_one AS SELECT id FROM proof_source"),
        )
        .unwrap();
    engine
        .submit_transaction(
            2_031,
            parsed("CREATE VIEW proof_view AS SELECT * FROM proof_view_one"),
        )
        .unwrap();
    engine.submit_transaction(2_031, parsed("COMMIT")).unwrap();
    let payload = operation_payload(engine.durable_wal_records().last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("view commit must decode as a transaction");
    };

    let apply_target = Engine::new_local();
    apply_target
        .submit_transaction(2_032, parsed("CREATE TABLE proof_source (id int4)"))
        .unwrap();
    let replay_entry = LogEntry {
        term: 1,
        index: 2,
        payload: Arc::from(&b""[..]),
    };
    let candidates = [
        {
            let mut tampered = record.clone();
            tampered.view_operations[0]
                .dependencies
                .get_mut("proof_source")
                .unwrap()
                .oid += 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.view_operations[1]
                .dependencies
                .get_mut("proof_view_one")
                .unwrap()
                .digest[0] ^= 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.view_operations[1].target_after.digest[0] ^= 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.view_operations[1].ordinal += 1;
            tampered
        },
    ];
    for tampered in candidates {
        let before = apply_target.catalog_snapshot();
        let mut catalog = apply_target.ddl_catalog().clone();
        let error = apply_target
            .apply_binary_transaction_record(&replay_entry, &mut catalog, tampered)
            .unwrap_err();
        assert!(
            error.to_string().contains("CREATE VIEW")
                || error.to_string().contains("stored-view")
                || error.to_string().contains("statement position"),
            "{error}"
        );
        let after = Engine::catalog_snapshot_from_working(&catalog, before.commit_seq);
        assert!(after.same_contents(&before));
        assert!(!catalog.relational_views.contains_key("proof_view_one"));
        assert!(!catalog.relational_views.contains_key("proof_view"));
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn ordered_catalog_view_gpu_permutations_reset_and_recovery_are_nonvacuous() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            2_040,
            parsed("CREATE TABLE gpu_view_source (id int4 PRIMARY KEY, value int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(2_041, parsed("INSERT INTO gpu_view_source VALUES (1, 10)"))
        .unwrap();

    engine.submit_transaction(2_042, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            2_042,
            parsed(
                "CREATE VIEW gpu_view_early AS \
                 SELECT id, value FROM gpu_view_source WHERE id >= 1 ORDER BY id",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(2_042, parsed("INSERT INTO gpu_view_source VALUES (2, 20)"))
        .unwrap();
    engine
        .submit_transaction(
            2_042,
            parsed("CREATE TABLE gpu_view_private (id int4 PRIMARY KEY, value int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(2_042, parsed("INSERT INTO gpu_view_private VALUES (7, 70)"))
        .unwrap();
    engine
        .submit_transaction(
            2_042,
            parsed(
                "CREATE VIEW gpu_view_private_one AS \
                 SELECT id, value FROM gpu_view_private",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            2_042,
            parsed(
                "CREATE OR REPLACE VIEW gpu_view_private_one AS \
                 SELECT value FROM gpu_view_private",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            2_042,
            parsed("CREATE VIEW gpu_view_layered AS SELECT * FROM gpu_view_private_one"),
        )
        .unwrap();
    engine
        .submit_transaction(2_042, parsed("TRUNCATE gpu_view_source"))
        .unwrap();
    engine
        .submit_transaction(2_042, parsed("INSERT INTO gpu_view_source VALUES (3, 30)"))
        .unwrap();

    let description = engine
        .describe_prepared_command_in_transaction(
            2_042,
            &gpu_db_sql::PreparedCommand::parse("SELECT * FROM gpu_view_layered").unwrap(),
            &[],
        )
        .unwrap();
    assert_eq!(
        description
            .result_columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        vec!["value"]
    );
    let private_early = engine
        .execute_relational_select_in_transaction(2_042, &select("SELECT * FROM gpu_view_early"))
        .unwrap();
    assert!(matches!(
        private_early.executed_target,
        DeviceTarget::Gpu(_)
    ));
    assert_eq!(
        private_early.rows.into_boxed(),
        vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]
    );
    let private_layered = engine
        .execute_relational_select_in_transaction(2_042, &select("SELECT * FROM gpu_view_layered"))
        .unwrap();
    assert!(matches!(
        private_layered.executed_target,
        DeviceTarget::Gpu(_)
    ));
    assert_eq!(
        private_layered.rows.into_boxed(),
        vec![vec![SqlValue::Int4(70)]]
    );
    assert_eq!(
        engine
            .execute_resident_expr_select_sql_in_transaction(
                2_042,
                "SELECT relname FROM pg_catalog.pg_class \
                 WHERE relname = 'gpu_view_layered'",
            )
            .unwrap()
            .rows,
        vec![vec![SqlValue::Text("gpu_view_layered".to_string())]]
    );
    assert!(engine
        .execute_resident_expr_select_sql(
            "SELECT relname FROM pg_catalog.pg_class WHERE relname = 'gpu_view_layered'",
        )
        .unwrap()
        .rows
        .is_empty());

    engine.submit_transaction(2_042, parsed("COMMIT")).unwrap();
    let records = engine.durable_wal_records();
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("mixed view transaction must use one typed transaction record");
    };
    assert_eq!(record.view_operations.len(), 4);
    assert_eq!(
        record
            .view_operations
            .iter()
            .map(|identity| identity.ordinal)
            .collect::<Vec<_>>(),
        vec![0, 4, 5, 6]
    );
    assert_eq!(
        record.view_operations[2].target_before,
        Some(record.view_operations[1].target_after.clone())
    );
    assert_eq!(
        record.view_operations[3]
            .dependencies
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["gpu_view_private", "gpu_view_private_one"]
    );

    for (sql, expected) in [
        (
            "SELECT * FROM gpu_view_early",
            vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]],
        ),
        (
            "SELECT * FROM gpu_view_layered",
            vec![vec![SqlValue::Int4(70)]],
        ),
    ] {
        let result = engine.execute_relational_select(&select(sql)).unwrap();
        assert!(matches!(result.executed_target, DeviceTarget::Gpu(_)));
        assert_eq!(result.rows.into_boxed(), expected);
    }

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    recovered.set_shard_residency_enabled(true);
    recovered.set_auto_admit_on_commit(true);
    let recovered_early = recovered
        .execute_relational_select(&select("SELECT * FROM gpu_view_early"))
        .unwrap();
    assert!(matches!(
        recovered_early.executed_target,
        DeviceTarget::Gpu(_)
    ));
    assert_eq!(
        recovered_early.rows.into_boxed(),
        vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]
    );
    let recovered_layered = recovered
        .execute_relational_select(&select("SELECT * FROM gpu_view_layered"))
        .unwrap();
    assert!(matches!(
        recovered_layered.executed_target,
        DeviceTarget::Gpu(_)
    ));
    assert_eq!(
        recovered_layered.rows.into_boxed(),
        vec![vec![SqlValue::Int4(70)]]
    );
}
