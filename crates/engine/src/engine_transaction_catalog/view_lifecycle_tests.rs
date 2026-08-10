use super::*;
use std::sync::mpsc;
use std::time::Duration;

fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
    gpu_db_sql::ParsedCommand::parse(sql).unwrap()
}

fn select(sql: &str) -> Select {
    let Command::Select(select) = parse_command(sql).unwrap() else {
        panic!("expected SELECT");
    };
    select
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

fn codec5_catalog_record(
    durable: &gpu_db_wal::WalRecord,
) -> (BinaryTransactionRecord, gpu_db_wal::CanonicalDigest) {
    let envelope = gpu_db_wal::decode_canonical_record_payload(&durable.payload)
        .unwrap()
        .unwrap();
    let record = crate::typed_insert_aggregate::decode_catalog_composition_for_test(
        &envelope.header,
        &envelope.fragments,
    )
    .unwrap()
    .expect("mixed view/INSERT codec-5 record must carry S3");
    (record, envelope.header.request_digest)
}

#[test]
fn transactional_view_rename_drop_recreate_is_private_ordered_and_recoverable() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            3_000,
            parsed("CREATE TABLE lifecycle_source (id int4, note text)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            3_001,
            parsed("CREATE VIEW lifecycle_view AS SELECT id, note FROM lifecycle_source"),
        )
        .unwrap();
    engine
        .submit_transaction(
            3_006,
            parsed("COMMENT ON VIEW lifecycle_view IS 'lifecycle metadata'"),
        )
        .unwrap();
    engine
        .submit_transaction(
            3_007,
            parsed("GRANT SELECT ON VIEW lifecycle_view TO PUBLIC"),
        )
        .unwrap();
    let original = engine.catalog_snapshot().relational_views["lifecycle_view"].clone();
    assert!(original.acl["public"].contains(&TablePrivilege::Select));
    let wal_before = engine.durable_wal_records().len();

    engine.submit_transaction(3_002, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            3_002,
            parsed("ALTER VIEW lifecycle_view RENAME TO lifecycle_renamed"),
        )
        .unwrap();
    let published = engine.catalog_snapshot();
    assert!(published.relational_views.contains_key("lifecycle_view"));
    assert!(!published.relational_views.contains_key("lifecycle_renamed"));
    let private = engine
        .transaction_snapshot_handle(3_002)
        .unwrap()
        .transaction_catalog();
    assert!(!private.relational_views.contains_key("lifecycle_view"));
    assert_eq!(
        private.relational_views["lifecycle_renamed"].oid,
        original.oid
    );
    assert_eq!(
        private.relational_views["lifecycle_renamed"].acl,
        original.acl
    );
    assert_eq!(
        private
            .relational_comments
            .get(&RelationalCommentTarget::View {
                view: "lifecycle_renamed".to_string(),
            })
            .map(String::as_str),
        Some("lifecycle metadata")
    );
    assert!(!private
        .relational_comments
        .contains_key(&RelationalCommentTarget::View {
            view: "lifecycle_view".to_string(),
        }));
    let prepared = gpu_db_sql::PreparedCommand::parse("SELECT * FROM lifecycle_renamed").unwrap();
    assert_eq!(
        engine
            .describe_prepared_command_in_transaction(3_002, &prepared, &[])
            .unwrap()
            .result_columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        vec!["id", "note"]
    );
    assert!(matches!(
        engine.describe_prepared_command_in_transaction(
            3_002,
            &gpu_db_sql::PreparedCommand::parse("SELECT * FROM lifecycle_view").unwrap(),
            &[],
        ),
        Err(ExecuteError::UndefinedRelation(name)) if name == "lifecycle_view"
    ));
    engine
        .submit_transaction(
            3_002,
            parsed("INSERT INTO lifecycle_source VALUES (1, 'kept')"),
        )
        .unwrap();
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(
                3_002,
                &select("SELECT * FROM lifecycle_renamed"),
            )
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("kept".to_string()),]]
    );

    engine
        .submit_transaction(
            3_002,
            parsed("DROP VIEW IF EXISTS missing_lifecycle_view, lifecycle_renamed"),
        )
        .unwrap();
    assert!(engine
        .transaction_snapshot_handle(3_002)
        .unwrap()
        .transaction_catalog()
        .relational_views
        .is_empty());
    engine
        .submit_transaction(
            3_002,
            parsed(
                "CREATE VIEW lifecycle_renamed AS \
                 SELECT note FROM lifecycle_source",
            ),
        )
        .unwrap();
    let replacement = engine
        .transaction_snapshot_handle(3_002)
        .unwrap()
        .transaction_catalog()
        .relational_views["lifecycle_renamed"]
        .clone();
    assert_ne!(replacement.oid, original.oid);
    assert!(!replacement.acl.contains_key("public"));
    assert!(!engine
        .transaction_snapshot_handle(3_002)
        .unwrap()
        .transaction_catalog()
        .relational_comments
        .contains_key(&RelationalCommentTarget::View {
            view: "lifecycle_renamed".to_string(),
        }));
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    engine.submit_transaction(3_002, parsed("COMMIT")).unwrap();
    let records = engine.durable_wal_records();
    assert_eq!(records.len(), wal_before + 1);
    let (record, request_digest) = codec5_catalog_record(records.last().unwrap());
    assert!(record.view_operations.is_empty());
    assert_eq!(record.view_lifecycle_operations.len(), 3);
    assert_eq!(
        record
            .view_lifecycle_operations
            .iter()
            .map(|identity| (identity.command_index, identity.ordinal))
            .collect::<Vec<_>>(),
        vec![(0, 0), (1, 2), (2, 3)]
    );
    assert_eq!(
        record.operation_order,
        vec![
            BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
            BinaryTransactionOperationIdentity::Insert {
                table: "lifecycle_source".to_string(),
            },
            BinaryTransactionOperationIdentity::Catalog { command_index: 1 },
            BinaryTransactionOperationIdentity::Catalog { command_index: 2 },
        ]
    );
    assert!(record.table_identities.contains_key("lifecycle_source"));
    assert!(record.mutations.is_empty());
    let rename = &record.view_lifecycle_operations[0].targets[0];
    assert_eq!(rename.target_before.as_ref().unwrap().oid, original.oid);
    assert_eq!(rename.target_after.as_ref().unwrap().oid, original.oid);
    assert_eq!(rename.after_name.as_deref(), Some("lifecycle_renamed"));
    let drop = &record.view_lifecycle_operations[1].targets;
    assert_eq!(drop.len(), 2);
    assert!(drop[0].target_before.is_none());
    assert!(drop[0].dependencies.is_empty());
    assert_eq!(drop[1].target_before.as_ref().unwrap().oid, original.oid);
    assert!(drop[1].target_after.is_none());
    let create = &record.view_lifecycle_operations[2].targets[0];
    assert!(create.target_before.is_none());
    assert_eq!(create.target_after.as_ref().unwrap().oid, replacement.oid);
    let (_, affected_rows) = engine
        .commit_state()
        .resolve_transaction_retry_digest_outcome(3_002, request_digest)
        .unwrap()
        .unwrap();
    assert_eq!(affected_rows, 1);
    assert!(engine
        .commit_state()
        .resolve_transaction_retry_digest_outcome(3_002, [0x3d; 32])
        .unwrap_err()
        .to_string()
        .contains("different request"));

    let committed = engine.catalog_snapshot();
    assert!(!committed.relational_views.contains_key("lifecycle_view"));
    assert_eq!(committed.relational_views["lifecycle_renamed"], replacement);
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(committed.as_ref()));
    assert_eq!(
        recovered
            .describe_prepared_command(&prepared, &[])
            .unwrap()
            .result_columns[0]
            .name,
        "note"
    );
    assert_eq!(recovered.relational_view_comment("lifecycle_renamed"), None);
    assert!(!recovered
        .relational_relation_acl("lifecycle_renamed")
        .unwrap()
        .contains_key("public"));
    assert_eq!(
        recovered
            .execute_relational_select(&select("SELECT * FROM lifecycle_renamed"))
            .unwrap()
            .rows,
        vec![vec![SqlValue::Text("kept".to_string())]]
    );
}

#[test]
fn multi_view_drop_respects_dependencies_and_rollback_including_absent_targets() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(3_010, parsed("CREATE TABLE drop_source (id int4)"))
        .unwrap();
    engine
        .submit_transaction(
            3_011,
            parsed("CREATE VIEW drop_base AS SELECT id FROM drop_source"),
        )
        .unwrap();
    engine
        .submit_transaction(
            3_012,
            parsed("CREATE VIEW drop_layer AS SELECT * FROM drop_base"),
        )
        .unwrap();
    let published = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();

    engine.submit_transaction(3_013, parsed("BEGIN")).unwrap();
    let dependency_error = engine
        .submit_transaction(3_013, parsed("DROP VIEW drop_base"))
        .unwrap_err();
    assert!(
        dependency_error.to_string().contains("depends"),
        "{dependency_error}"
    );
    assert!(engine
        .transaction_snapshot_handle(3_013)
        .unwrap()
        .transaction_catalog()
        .same_contents(published.as_ref()));
    engine
        .submit_transaction(
            3_013,
            parsed("DROP VIEW IF EXISTS absent_drop_view, drop_layer, drop_base"),
        )
        .unwrap();
    assert!(engine
        .transaction_snapshot_handle(3_013)
        .unwrap()
        .transaction_catalog()
        .relational_views
        .is_empty());
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine
        .submit_transaction(3_013, parsed("ROLLBACK"))
        .unwrap();
    assert!(engine.catalog_snapshot().same_contents(published.as_ref()));
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    engine.submit_transaction(3_014, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(3_014, parsed("DROP VIEW IF EXISTS absent_drop_view"))
        .unwrap();
    engine.submit_transaction(3_014, parsed("COMMIT")).unwrap();
    assert!(engine.catalog_snapshot().same_contents(published.as_ref()));
    let payload = operation_payload(engine.durable_wal_records().last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("absent DROP VIEW must retain typed request identity");
    };
    let target = &record.view_lifecycle_operations[0].targets[0];
    assert!(target.target_before.is_none());
    assert!(target.dependencies.is_empty());
    assert!(target.target_after.is_none());
}

#[test]
fn view_lifecycle_rebases_only_over_byte_identical_catalog_and_rejects_aba_before_wal() {
    for begin in ["BEGIN", "BEGIN ISOLATION LEVEL REPEATABLE READ"] {
        let engine = Engine::new_local();
        engine
            .submit_transaction(3_020, parsed("CREATE TABLE aba_lifecycle_source (id int4)"))
            .unwrap();
        engine
            .submit_transaction(
                3_021,
                parsed(
                    "CREATE VIEW aba_lifecycle_view AS \
                     SELECT id FROM aba_lifecycle_source",
                ),
            )
            .unwrap();
        engine.submit_transaction(3_022, parsed(begin)).unwrap();
        engine
            .submit_transaction(
                3_022,
                parsed(
                    "ALTER VIEW aba_lifecycle_view \
                     RENAME TO aba_lifecycle_private",
                ),
            )
            .unwrap();

        engine
            .submit_transaction(3_023, parsed("SET lifecycle_rebase = unchanged"))
            .unwrap();
        engine
            .submit_transaction(
                3_022,
                parsed("ALTER VIEW aba_lifecycle_private RENAME TO aba_lifecycle_view"),
            )
            .unwrap();

        engine
            .submit_transaction(3_024, parsed("DROP VIEW aba_lifecycle_view"))
            .unwrap();
        engine
            .submit_transaction(
                3_025,
                parsed(
                    "CREATE VIEW aba_lifecycle_view AS \
                     SELECT id FROM aba_lifecycle_source",
                ),
            )
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        let error = engine
            .submit_transaction(3_022, parsed("COMMIT"))
            .unwrap_err();
        assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        engine
            .submit_transaction(3_022, parsed("ROLLBACK"))
            .unwrap();
    }
}

#[test]
fn catalog_latch_makes_view_lifecycle_capture_atomic_with_concurrent_drop() {
    let engine = Arc::new(Engine::new_local());
    engine
        .submit_transaction(
            3_026,
            parsed("CREATE TABLE latched_lifecycle_source (id int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            3_027,
            parsed(
                "CREATE VIEW latched_lifecycle_view AS \
                 SELECT id FROM latched_lifecycle_source",
            ),
        )
        .unwrap();
    let view_oid = engine.catalog_snapshot().relational_views["latched_lifecycle_view"].oid;
    engine
        .submit_transaction(3_028, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();

    let (release_tx, release_rx) = mpsc::channel();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let racer_engine = Arc::clone(&engine);
    let racer = std::thread::spawn(move || {
        release_rx.recv().unwrap();
        started_tx.send(()).unwrap();
        done_tx
            .send(
                racer_engine.submit_transaction(3_029, parsed("DROP VIEW latched_lifecycle_view")),
            )
            .unwrap();
    });

    let statement = parsed(
        "ALTER VIEW latched_lifecycle_view \
         RENAME TO latched_lifecycle_private",
    );
    let (command, source) = statement.into_parts();
    let hook_engine = Arc::clone(&engine);
    engine
        .execute_catalog_in_transaction_instrumented(3_028, command, source, None, || {
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

    let snapshot = engine.transaction_snapshot_handle(3_028).unwrap();
    let delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let TransactionOperation::Catalog(staged) = &delta.operations[0] else {
        panic!("expected a catalog operation");
    };
    assert_eq!(
        staged.view_identity.as_ref().unwrap().targets[0]
            .target_before
            .as_ref()
            .unwrap()
            .oid,
        view_oid
    );
    drop(delta);
    let wal_before = engine.durable_wal_records().len();
    let error = engine
        .submit_transaction(3_028, parsed("COMMIT"))
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine
        .submit_transaction(3_028, parsed("ROLLBACK"))
        .unwrap();
}

#[test]
fn view_lifecycle_replay_tamper_is_clone_first_and_effect_free() {
    let source = Engine::new_local();
    source
        .submit_transaction(
            3_030,
            parsed("CREATE TABLE proof_lifecycle_source (id int4)"),
        )
        .unwrap();
    source
        .submit_transaction(
            3_031,
            parsed(
                "CREATE VIEW proof_lifecycle_view AS \
                 SELECT id FROM proof_lifecycle_source",
            ),
        )
        .unwrap();
    let prefix = source.durable_wal_records();
    source.submit_transaction(3_032, parsed("BEGIN")).unwrap();
    source
        .submit_transaction(
            3_032,
            parsed(
                "ALTER VIEW proof_lifecycle_view \
                 RENAME TO proof_lifecycle_renamed",
            ),
        )
        .unwrap();
    source
        .submit_transaction(3_032, parsed("DROP VIEW proof_lifecycle_renamed"))
        .unwrap();
    source.submit_transaction(3_032, parsed("COMMIT")).unwrap();
    let payload = operation_payload(source.durable_wal_records().last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("view lifecycle commit must decode");
    };

    let target = Engine::recover_from_durable_wal(&prefix).unwrap();
    let before = target.catalog_snapshot();
    let entry = LogEntry {
        term: 1,
        index: before.commit_seq + 1,
        payload: Arc::from(&b""[..]),
    };
    let candidates = [
        {
            let mut tampered = record.clone();
            tampered.view_lifecycle_operations[0].targets[0]
                .target_before
                .as_mut()
                .unwrap()
                .oid += 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.view_lifecycle_operations[0].targets[0]
                .dependencies
                .get_mut("proof_lifecycle_source")
                .unwrap()
                .digest[0] ^= 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.view_lifecycle_operations[0].targets[0].after_name =
                Some("smuggled_view".to_string());
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.view_lifecycle_operations[1].ordinal = 0;
            tampered
        },
    ];
    for tampered in candidates {
        let mut catalog = target.ddl_catalog().clone();
        assert!(target
            .apply_binary_transaction_record(&entry, &mut catalog, tampered)
            .is_err());
        let after = Engine::catalog_snapshot_from_working(&catalog, before.commit_seq);
        assert!(after.same_contents(before.as_ref()));
        assert!(catalog
            .relational_views
            .contains_key("proof_lifecycle_view"));
        assert!(!catalog
            .relational_views
            .contains_key("proof_lifecycle_renamed"));
    }
}

#[test]
fn post_durable_view_lifecycle_failure_is_recovery_owned() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(3_040, parsed("CREATE TABLE durable_view_source (id int4)"))
        .unwrap();
    engine
        .submit_transaction(
            3_041,
            parsed("CREATE VIEW durable_view AS SELECT id FROM durable_view_source"),
        )
        .unwrap();
    engine
        .submit_transaction(
            3_043,
            parsed("COMMENT ON VIEW durable_view IS 'durable metadata'"),
        )
        .unwrap();
    engine
        .submit_transaction(3_044, parsed("GRANT SELECT ON VIEW durable_view TO PUBLIC"))
        .unwrap();
    engine.submit_transaction(3_042, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            3_042,
            parsed("ALTER VIEW durable_view RENAME TO durable_view_renamed"),
        )
        .unwrap();
    engine.fail_next_transaction_post_durable_apply();
    let error = engine
        .submit_transaction(3_042, parsed("COMMIT"))
        .unwrap_err();
    assert!(error.is_indeterminate(), "{error}");

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    let catalog = recovered.catalog_snapshot();
    assert!(!catalog.relational_views.contains_key("durable_view"));
    assert!(catalog
        .relational_views
        .contains_key("durable_view_renamed"));
    assert_eq!(
        recovered
            .relational_view_comment("durable_view_renamed")
            .as_deref(),
        Some("durable metadata")
    );
    assert!(recovered
        .relational_relation_acl("durable_view_renamed")
        .unwrap()["public"]
        .contains(&TablePrivilege::Select));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn ordered_catalog_view_lifecycle_gpu_null_catalog_and_recovery_are_nonvacuous() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            3_050,
            parsed("CREATE TABLE gpu_lifecycle_source (id int4 PRIMARY KEY, note text)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            3_051,
            parsed("INSERT INTO gpu_lifecycle_source VALUES (1, NULL), (2, 'two')"),
        )
        .unwrap();
    engine
        .submit_transaction(
            3_052,
            parsed(
                "CREATE VIEW gpu_lifecycle_view AS \
                 SELECT id, note FROM gpu_lifecycle_source ORDER BY id",
            ),
        )
        .unwrap();

    engine.submit_transaction(3_053, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            3_053,
            parsed(
                "ALTER VIEW gpu_lifecycle_view \
                 RENAME TO gpu_lifecycle_renamed",
            ),
        )
        .unwrap();
    let renamed = engine
        .execute_relational_select_in_transaction(
            3_053,
            &select("SELECT * FROM gpu_lifecycle_renamed"),
        )
        .unwrap();
    assert!(matches!(renamed.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(
        renamed.rows.into_boxed(),
        vec![
            vec![SqlValue::Int4(1), SqlValue::Null],
            vec![SqlValue::Int4(2), SqlValue::Text("two".to_string()),],
        ]
    );
    assert_eq!(
        engine
            .execute_resident_expr_select_sql_in_transaction(
                3_053,
                "SELECT relname FROM pg_catalog.pg_class \
                 WHERE relname = 'gpu_lifecycle_renamed'",
            )
            .unwrap()
            .rows,
        vec![vec![SqlValue::Text("gpu_lifecycle_renamed".to_string(),)]]
    );
    assert!(!engine
        .transaction_snapshot_handle(3_053)
        .unwrap()
        .transaction_catalog()
        .relational_views
        .contains_key("gpu_lifecycle_view"));
    let old_binding_error = engine
        .execute_relational_select_in_transaction(
            3_053,
            &select("SELECT * FROM gpu_lifecycle_view"),
        )
        .unwrap_err();
    assert!(
        old_binding_error.to_string().contains("gpu_lifecycle_view"),
        "{old_binding_error}"
    );
    assert!(engine
        .execute_resident_expr_select_sql_in_transaction(
            3_053,
            "SELECT relname FROM pg_catalog.pg_class \
             WHERE relname = 'gpu_lifecycle_view'",
        )
        .unwrap()
        .rows
        .is_empty());

    engine
        .submit_transaction(3_053, parsed("DROP VIEW gpu_lifecycle_renamed"))
        .unwrap();
    engine
        .submit_transaction(
            3_053,
            parsed(
                "CREATE VIEW gpu_lifecycle_final AS \
                 SELECT id, note FROM gpu_lifecycle_source ORDER BY id",
            ),
        )
        .unwrap();
    engine.submit_transaction(3_053, parsed("COMMIT")).unwrap();

    for candidate in [
        Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap(),
        engine,
    ] {
        candidate.set_shard_residency_enabled(true);
        candidate.set_auto_admit_on_commit(true);
        let result = candidate
            .execute_relational_select(&select("SELECT * FROM gpu_lifecycle_final"))
            .unwrap();
        assert!(matches!(result.executed_target, DeviceTarget::Gpu(_)));
        assert_eq!(
            result.rows.into_boxed(),
            vec![
                vec![SqlValue::Int4(1), SqlValue::Null],
                vec![SqlValue::Int4(2), SqlValue::Text("two".to_string()),],
            ]
        );
        assert!(candidate
            .execute_resident_expr_select_sql(
                "SELECT relname FROM pg_catalog.pg_class \
                 WHERE relname = 'gpu_lifecycle_renamed'",
            )
            .unwrap()
            .rows
            .is_empty());
    }
}
