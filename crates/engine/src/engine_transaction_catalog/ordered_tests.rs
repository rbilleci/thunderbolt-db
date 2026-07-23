use super::*;
use crate::engine_transaction_reset::table_schema_digest;

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

#[test]
fn ordered_catalog_multiple_create_binds_typed_identity_retry_and_replay() {
    let engine = Engine::new_local();
    engine.submit_transaction(1_000, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            1_000,
            parsed("CREATE TABLE ordered_host_a (id serial PRIMARY KEY, value int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            1_000,
            parsed("CREATE TABLE ordered_host_b (id serial PRIMARY KEY, value text)"),
        )
        .unwrap();

    let private = engine.transaction_snapshot_handle(1_000).unwrap();
    let private_catalog = private.transaction_catalog();
    let first = private_catalog.relational_catalog["ordered_host_a"].clone();
    let second = private_catalog.relational_catalog["ordered_host_b"].clone();
    assert!(first.oid < second.oid);
    let sequence = private_catalog.relational_sequences["ordered_host_a_id_seq"].clone();
    let second_sequence = private_catalog.relational_sequences["ordered_host_b_id_seq"].clone();
    assert!(!engine
        .catalog_snapshot()
        .relational_catalog
        .contains_key("ordered_host_a"));
    let prepared = gpu_db_sql::PreparedCommand::parse(
        "INSERT INTO ordered_host_b VALUES ($1, $2) RETURNING id, value",
    )
    .unwrap();
    let description = engine
        .describe_prepared_command_in_transaction(1_000, &prepared, &[])
        .unwrap();
    assert_eq!(
        description.parameter_types,
        vec![SqlType::Int4, SqlType::Text]
    );
    assert!(engine.describe_prepared_command(&prepared, &[]).is_err());

    engine.submit_transaction(1_000, parsed("COMMIT")).unwrap();
    let records = engine.durable_wal_records();
    assert_eq!(records.len(), 1);
    let payload = operation_payload(&records[0]);
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("ordered catalog commit must use one resolved transaction record");
    };
    assert_eq!(
        record
            .catalog_commands
            .iter()
            .map(|operation| operation.ordinal)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        record
            .created_table_identities
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["ordered_host_a", "ordered_host_b"]
    );
    assert_eq!(
        record.operation_order,
        vec![
            BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
            BinaryTransactionOperationIdentity::Catalog { command_index: 1 },
        ]
    );
    assert_eq!(
        record.created_table_identities["ordered_host_a"].table_oid,
        first.oid
    );
    assert_eq!(
        record.created_table_identities["ordered_host_b"].schema_digest,
        table_schema_digest(&second).unwrap()
    );
    let catalog_output = record.catalog_output.as_ref().unwrap();
    assert_eq!(
        catalog_output.relational_next_oid,
        private_catalog.relational_next_oid
    );
    assert_eq!(
        catalog_output.relational_next_column_id,
        private_catalog.relational_next_column_id
    );
    assert_eq!(
        catalog_output.created_sequence_oids,
        BTreeMap::from([
            ("ordered_host_a_id_seq".to_string(), sequence.oid),
            ("ordered_host_b_id_seq".to_string(), second_sequence.oid,),
        ])
    );
    assert_eq!(record.statement_digests.len(), record.operation_order.len());
    assert_eq!(
        record.sequence_input_oids,
        BTreeMap::from([
            ((0, "ordered_host_a_id_seq".to_string()), sequence.oid),
            (
                (1, "ordered_host_b_id_seq".to_string()),
                second_sequence.oid,
            ),
        ])
    );

    let replay_entry = LogEntry {
        term: 1,
        index: 1,
        payload: Arc::from(&b""[..]),
    };
    let apply_target = Engine::new_local();
    let mut wrong_sequence_oid = record.clone();
    *wrong_sequence_oid
        .sequence_input_oids
        .get_mut(&(0, "ordered_host_a_id_seq".to_string()))
        .unwrap() += 1;
    let mut catalog = apply_target.ddl_catalog().clone();
    let error = apply_target
        .apply_binary_transaction_record(&replay_entry, &mut catalog, wrong_sequence_oid)
        .unwrap_err();
    assert!(error.to_string().contains("sequence input"), "{error}");

    let mut unrelated_sequence_advance = record.clone();
    unrelated_sequence_advance
        .operation_order
        .push(BinaryTransactionOperationIdentity::Insert {
            table: "ordered_host_a".to_string(),
        });
    unrelated_sequence_advance.statement_digests.push(
        transaction_statement_digest(
            &parse_command("INSERT INTO ordered_host_a (value) VALUES (7)").unwrap(),
        )
        .unwrap(),
    );
    unrelated_sequence_advance.sequence_input_oids.insert(
        (2, "ordered_host_b_id_seq".to_string()),
        second_sequence.oid,
    );
    unrelated_sequence_advance
        .sequence_advances
        .insert("ordered_host_b_id_seq".to_string(), (1, true));
    let mut catalog = apply_target.ddl_catalog().clone();
    let error = apply_target
        .apply_binary_transaction_record(&replay_entry, &mut catalog, unrelated_sequence_advance)
        .unwrap_err();
    assert!(
        error.to_string().contains("exact default dependency"),
        "{error}"
    );

    let mut impossible_mutation_family = record.clone();
    impossible_mutation_family
        .operation_order
        .push(BinaryTransactionOperationIdentity::Delete {
            table: "ordered_host_a".to_string(),
        });
    impossible_mutation_family.statement_digests.push(
        transaction_statement_digest(
            &parse_command("DELETE FROM ordered_host_a WHERE id = 9").unwrap(),
        )
        .unwrap(),
    );
    impossible_mutation_family
        .mutations
        .push(BinaryTransactionMutation::Insert {
            table: "ordered_host_a".to_string(),
            row_id: 9,
            row_encoded: encode_relational_row(&[SqlValue::Int4(9), SqlValue::Int4(7)]),
        });
    let mut catalog = apply_target.ddl_catalog().clone();
    let error = apply_target
        .apply_binary_transaction_record(&replay_entry, &mut catalog, impossible_mutation_family)
        .unwrap_err();
    assert!(error.to_string().contains("no row operation"), "{error}");

    let request_digest = gpu_db_wal::canonical_request_digest(&payload);
    let (_, affected_rows) = engine
        .commit_state()
        .resolve_transaction_retry_digest_outcome(1_000, request_digest)
        .unwrap()
        .unwrap();
    assert_eq!(affected_rows, 0);
    assert!(engine
        .commit_state()
        .resolve_transaction_retry_digest_outcome(1_000, [0xA5; 32])
        .unwrap_err()
        .to_string()
        .contains("different request"));

    let envelope = gpu_db_wal::decode_canonical_record_payload(&records[0].payload)
        .unwrap()
        .unwrap();
    assert_eq!(envelope.header.catalog_after_epoch, 1);
    assert_eq!(envelope.header.table_block_count, 2);
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.catalog_snapshot().relational_catalog["ordered_host_a"],
        first
    );
    assert_eq!(
        recovered.catalog_snapshot().relational_catalog["ordered_host_b"],
        second
    );
    assert_eq!(
        recovered.catalog_snapshot().relational_sequences["ordered_host_a_id_seq"],
        sequence
    );
    assert_eq!(
        recovered.catalog_snapshot().relational_sequences["ordered_host_b_id_seq"],
        second_sequence
    );
    assert_eq!(
        recovered
            .commit_state()
            .resolve_transaction_retry_digest_outcome(1_000, request_digest)
            .unwrap()
            .unwrap()
            .1,
        0
    );
}

#[test]
fn ordered_catalog_read_committed_rebases_all_private_operations() {
    let engine = Engine::new_local();
    engine.submit_transaction(1_010, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            1_010,
            parsed("CREATE TABLE ordered_rc_a (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    let first_before = engine
        .transaction_snapshot_handle(1_010)
        .unwrap()
        .transaction_catalog()
        .relational_catalog["ordered_rc_a"]
        .clone();

    engine
        .submit_transaction(1_011, parsed("SET ordered_rebase = yes"))
        .unwrap();
    engine
        .submit_transaction(
            1_010,
            parsed("CREATE TABLE ordered_rc_b (id int4, note text)"),
        )
        .unwrap();
    let snapshot = engine.transaction_snapshot_handle(1_010).unwrap();
    assert_eq!(snapshot.boundary, engine.visible_up_to());
    let rebased = snapshot.transaction_catalog();
    assert_eq!(rebased.relational_catalog["ordered_rc_a"], first_before);
    assert!(rebased.relational_catalog.contains_key("ordered_rc_b"));
    assert_eq!(
        engine
            .relational_copy_columns_in_transaction(1_010, "ordered_rc_b")
            .unwrap()
            .len(),
        2
    );

    engine.submit_transaction(1_010, parsed("COMMIT")).unwrap();
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_eq!(
        recovered.catalog_snapshot().relational_catalog["ordered_rc_a"],
        first_before
    );
    assert!(recovered
        .catalog_snapshot()
        .relational_catalog
        .contains_key("ordered_rc_b"));
}

#[test]
fn ordered_catalog_existing_sequence_default_binds_stable_input_oid() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            1_005,
            parsed("CREATE SEQUENCE ordered_existing_default_seq"),
        )
        .unwrap();
    let sequence =
        engine.catalog_snapshot().relational_sequences["ordered_existing_default_seq"].clone();
    engine.submit_transaction(1_006, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            1_006,
            parsed(
                "CREATE TABLE ordered_existing_default (\
                 id int4 DEFAULT nextval('ordered_existing_default_seq'::regclass), value int4)",
            ),
        )
        .unwrap();
    engine.submit_transaction(1_006, parsed("COMMIT")).unwrap();

    let records = engine.durable_wal_records();
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("ordered existing-sequence catalog commit must use a transaction record");
    };
    assert_eq!(
        record.sequence_input_oids,
        BTreeMap::from([(
            (0, "ordered_existing_default_seq".to_string(),),
            sequence.oid
        )])
    );
    assert!(record.sequence_advances.is_empty());

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.catalog_snapshot().relational_sequences["ordered_existing_default_seq"],
        sequence
    );
    assert!(recovered
        .catalog_snapshot()
        .relational_catalog
        .contains_key("ordered_existing_default"));
}

#[test]
fn ordered_catalog_repeatable_read_allocator_aba_rejects_before_wal() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(1_020, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    for table in ["ordered_rr_a", "ordered_rr_b"] {
        engine
            .submit_transaction(1_020, parsed(&format!("CREATE TABLE {table} (id int4)")))
            .unwrap();
    }
    engine
        .submit_transaction(1_021, parsed("CREATE TABLE ordered_aba (id int4)"))
        .unwrap();
    engine
        .submit_transaction(1_022, parsed("DROP TABLE ordered_aba"))
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    let error = engine
        .submit_transaction(1_020, parsed("COMMIT"))
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(!engine
        .catalog_snapshot()
        .relational_catalog
        .contains_key("ordered_rr_a"));
    engine
        .submit_transaction(1_020, parsed("ROLLBACK"))
        .unwrap();
}

#[test]
fn ordered_catalog_unsupported_family_is_rejected_without_new_effect() {
    let engine = Engine::new_local();
    engine.submit_transaction(1_030, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(1_030, parsed("CREATE TABLE ordered_supported (id int4)"))
        .unwrap();
    let generation_before = engine
        .transaction_snapshot_handle(1_030)
        .unwrap()
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .generation;
    let error = engine
        .submit_transaction(
            1_030,
            parsed(
                "CREATE MATERIALIZED VIEW ordered_unsupported AS SELECT id FROM ordered_supported WITH NO DATA",
            ),
        )
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Unsupported(_)));
    let snapshot = engine.transaction_snapshot_handle(1_030).unwrap();
    let delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(delta.generation, generation_before);
    assert_eq!(delta.operations.len(), 1);
    drop(delta);
    assert!(engine.durable_wal_records().is_empty());
    engine
        .submit_transaction(1_030, parsed("ROLLBACK"))
        .unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn ordered_catalog_gpu_mixed_permutations_reset_and_recovery_are_nonvacuous() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            1_040,
            parsed("CREATE TABLE ordered_existing (id int4 PRIMARY KEY, value int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(1_041, parsed("INSERT INTO ordered_existing VALUES (1, 10)"))
        .unwrap();

    engine.submit_transaction(1_042, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(1_042, parsed("INSERT INTO ordered_existing VALUES (2, 20)"))
        .unwrap();
    engine
        .submit_transaction(
            1_042,
            parsed("CREATE TABLE ordered_gpu_a (id int4 PRIMARY KEY, value int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(1_042, parsed("INSERT INTO ordered_gpu_a VALUES (1, 11)"))
        .unwrap();
    engine
        .submit_transaction(
            1_042,
            parsed("CREATE TABLE ordered_gpu_b (id int4 PRIMARY KEY, value text)"),
        )
        .unwrap();
    let prepared_reset = gpu_db_sql::PreparedCommand::parse("TRUNCATE ordered_gpu_a").unwrap();
    let prepared_reset_description = engine
        .describe_prepared_command_in_transaction(1_042, &prepared_reset, &[])
        .unwrap();
    engine
        .submit_transaction(
            1_042,
            MutationRequest::new(prepared_reset.bind(&[]).unwrap())
                .with_expected_catalog_version(prepared_reset_description.catalog_version),
        )
        .unwrap();
    engine
        .submit_transaction(1_043, parsed("SET ordered_catalog_rebase = yes"))
        .unwrap();
    engine
        .submit_transaction(1_042, parsed("INSERT INTO ordered_gpu_a VALUES (2, 22)"))
        .unwrap();
    engine
        .submit_transaction(
            1_042,
            parsed("INSERT INTO ordered_gpu_b VALUES (3, 'three')"),
        )
        .unwrap();
    engine
        .submit_transaction(1_042, parsed("TRUNCATE ordered_existing"))
        .unwrap();
    engine
        .submit_transaction(
            1_042,
            parsed("CREATE TABLE ordered_gpu_c (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(1_042, parsed("INSERT INTO ordered_gpu_c VALUES (4)"))
        .unwrap();
    engine
        .submit_transaction(1_042, parsed("INSERT INTO ordered_existing VALUES (5, 50)"))
        .unwrap();

    let private_catalog_rows = engine
        .execute_resident_expr_select_sql_in_transaction(
            1_042,
            "SELECT c.relname FROM pg_catalog.pg_class c \
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
             WHERE c.relname = 'ordered_gpu_b'",
        )
        .unwrap()
        .rows;
    assert_eq!(
        private_catalog_rows,
        vec![vec![SqlValue::Text("ordered_gpu_b".to_string())]]
    );
    assert!(engine
        .execute_resident_expr_select_sql(
            "SELECT relname FROM pg_catalog.pg_class WHERE relname = 'ordered_gpu_b'"
        )
        .unwrap()
        .rows
        .is_empty());
    assert_eq!(
        engine
            .execute_resident_expr_select_sql_in_transaction(
                1_042,
                "SELECT id, value FROM ordered_gpu_a ORDER BY id",
            )
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(2), SqlValue::Int4(22)]]
    );

    engine.submit_transaction(1_042, parsed("COMMIT")).unwrap();
    let records = engine.durable_wal_records();
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("mixed catalog program must use one transaction record");
    };
    assert_eq!(
        record
            .catalog_commands
            .iter()
            .map(|operation| operation.ordinal)
            .collect::<Vec<_>>(),
        vec![1, 3, 8]
    );
    assert_eq!(
        record
            .table_resets
            .iter()
            .map(|reset| reset.ordinal)
            .collect::<Vec<_>>(),
        vec![4, 7]
    );
    assert_eq!(record.mutations.len(), 4);
    assert_eq!(
        record.operation_order,
        vec![
            BinaryTransactionOperationIdentity::Insert {
                table: "ordered_existing".to_string(),
            },
            BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
            BinaryTransactionOperationIdentity::Insert {
                table: "ordered_gpu_a".to_string(),
            },
            BinaryTransactionOperationIdentity::Catalog { command_index: 1 },
            BinaryTransactionOperationIdentity::TableReset {
                table: "ordered_gpu_a".to_string(),
            },
            BinaryTransactionOperationIdentity::Insert {
                table: "ordered_gpu_a".to_string(),
            },
            BinaryTransactionOperationIdentity::Insert {
                table: "ordered_gpu_b".to_string(),
            },
            BinaryTransactionOperationIdentity::TableReset {
                table: "ordered_existing".to_string(),
            },
            BinaryTransactionOperationIdentity::Catalog { command_index: 2 },
            BinaryTransactionOperationIdentity::Insert {
                table: "ordered_gpu_c".to_string(),
            },
            BinaryTransactionOperationIdentity::Insert {
                table: "ordered_existing".to_string(),
            },
        ]
    );
    assert_eq!(
        record
            .table_identities
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["ordered_existing"]
    );

    for (sql, expected) in [
        (
            "SELECT id, value FROM ordered_gpu_a ORDER BY id",
            vec![vec![SqlValue::Int4(2), SqlValue::Int4(22)]],
        ),
        (
            "SELECT id FROM ordered_gpu_c ORDER BY id",
            vec![vec![SqlValue::Int4(4)]],
        ),
        (
            "SELECT id, value FROM ordered_existing ORDER BY id",
            vec![vec![SqlValue::Int4(5), SqlValue::Int4(50)]],
        ),
    ] {
        assert_eq!(
            engine.execute_resident_expr_select_sql(sql).unwrap().rows,
            expected
        );
    }

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    recovered.set_shard_residency_enabled(true);
    recovered.set_auto_admit_on_commit(true);
    assert_eq!(
        recovered
            .execute_resident_expr_select_sql(
                "SELECT relname FROM pg_catalog.pg_class WHERE relname = 'ordered_gpu_b'",
            )
            .unwrap()
            .rows,
        vec![vec![SqlValue::Text("ordered_gpu_b".to_string())]]
    );
    assert_eq!(
        recovered
            .execute_resident_expr_select_sql("SELECT id, value FROM ordered_gpu_a ORDER BY id",)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(2), SqlValue::Int4(22)]]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn ordered_catalog_post_durable_failure_is_recovery_owned() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.submit_transaction(1_050, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            1_050,
            parsed("CREATE TABLE ordered_durable_a (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            1_050,
            parsed("CREATE TABLE ordered_durable_b (id int4 PRIMARY KEY, value int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            1_050,
            parsed(
                "CREATE VIEW ordered_durable_view AS \
                 SELECT id, value FROM ordered_durable_b",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(1_050, parsed("INSERT INTO ordered_durable_b VALUES (1, 9)"))
        .unwrap();
    engine.fail_next_transaction_post_durable_apply();
    let error = engine
        .submit_transaction(1_050, parsed("COMMIT"))
        .unwrap_err();
    assert!(error.is_indeterminate());
    assert!(engine.is_commit_path_poisoned());
    let records = engine.durable_wal_records();
    assert_eq!(records.len(), 1);

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .relational_catalog
        .contains_key("ordered_durable_a"));
    let Command::Select(view_select) = parse_command("SELECT * FROM ordered_durable_view").unwrap()
    else {
        panic!("expected SELECT");
    };
    assert_eq!(
        recovered
            .execute_relational_select(&view_select)
            .unwrap()
            .rows
            .into_boxed(),
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(9)]]
    );
    let payload = operation_payload(&records[0]);
    let request_digest = gpu_db_wal::canonical_request_digest(&payload);
    assert_eq!(
        recovered
            .commit_state()
            .resolve_transaction_retry_digest_outcome(1_050, request_digest)
            .unwrap()
            .unwrap()
            .1,
        1
    );
}
