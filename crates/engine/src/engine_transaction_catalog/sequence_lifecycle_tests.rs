use super::*;

fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
    gpu_db_sql::ParsedCommand::parse(sql).unwrap()
}

#[test]
fn private_serial_sequence_rename_binds_generated_oid_and_recovers() {
    let engine = Engine::new_local();
    engine.submit_transaction(5_060, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            5_060,
            parsed("CREATE TABLE renamed_private_serial (id SERIAL, note TEXT)"),
        )
        .unwrap();
    let private_before = engine
        .transaction_snapshot_handle(5_060)
        .unwrap()
        .transaction_catalog();
    let generated_oid = private_before.relational_sequences["renamed_private_serial_id_seq"].oid;

    engine
        .submit_transaction(
            5_060,
            parsed(
                "ALTER SEQUENCE renamed_private_serial_id_seq \
                 RENAME TO renamed_private_serial_sequence",
            ),
        )
        .unwrap();
    let private_after = engine
        .transaction_snapshot_handle(5_060)
        .unwrap()
        .transaction_catalog();
    assert_eq!(
        private_after.relational_sequences["renamed_private_serial_sequence"].oid,
        generated_oid
    );
    assert_eq!(
        table_default_sequence(&private_after, "renamed_private_serial"),
        "renamed_private_serial_sequence"
    );

    engine
        .submit_transaction(5_060, parsed("COMMIT"))
        .expect("the generated sequence must bind by stable OID after its private rename");
    let committed = engine.catalog_snapshot();
    assert_eq!(
        committed.relational_sequences["renamed_private_serial_sequence"].oid,
        generated_oid
    );
    assert_eq!(
        table_default_sequence(&committed, "renamed_private_serial"),
        "renamed_private_serial_sequence"
    );

    let records = engine.durable_wal_records();
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("renamed generated sequence must use typed transaction WAL");
    };
    assert_eq!(
        record
            .catalog_output
            .as_ref()
            .unwrap()
            .created_sequence_oids,
        BTreeMap::from([("renamed_private_serial_id_seq".to_string(), generated_oid,)])
    );
    assert_eq!(
        record.sequence_lifecycle_operations.last().unwrap().targets[0]
            .target_after
            .as_ref()
            .unwrap()
            .oid,
        generated_oid
    );

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(committed.as_ref()));
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

fn table_default_sequence(catalog: &CatalogSnapshot, table: &str) -> String {
    catalog.relational_catalog[table]
        .columns
        .iter()
        .find_map(|column| match &column.default {
            Some(ColumnDefault::SequenceNextVal { sequence, .. }) => Some(sequence.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("table {table:?} has no sequence default"))
}

#[test]
fn transactional_create_sequence_is_private_rollback_complete_and_recoverable() {
    let engine = Engine::new_local();
    let allocator_before = engine.catalog_snapshot().relational_next_oid;
    let wal_before = engine.durable_wal_records().len();

    engine.submit_transaction(5_000, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(5_000, parsed("CREATE SEQUENCE private_sequence"))
        .unwrap();
    assert!(engine
        .relational_catalog_sequence("private_sequence")
        .is_none());
    let private = engine
        .transaction_snapshot_handle(5_000)
        .unwrap()
        .transaction_catalog();
    assert_eq!(
        private.relational_sequences["private_sequence"].oid,
        allocator_before
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    engine
        .submit_transaction(5_000, parsed("ROLLBACK"))
        .unwrap();
    assert!(engine
        .relational_catalog_sequence("private_sequence")
        .is_none());
    assert_eq!(
        engine.catalog_snapshot().relational_next_oid,
        allocator_before
    );

    engine.submit_transaction(5_001, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(5_001, parsed("CREATE SEQUENCE private_sequence"))
        .unwrap();
    engine.submit_transaction(5_001, parsed("COMMIT")).unwrap();

    let sequence = engine
        .relational_catalog_sequence("private_sequence")
        .unwrap();
    assert_eq!(sequence.oid, allocator_before);
    assert_eq!((sequence.last_value, sequence.is_called), (1, false));
    let records = engine.durable_wal_records();
    assert_eq!(records.len(), wal_before + 1);
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("sequence lifecycle must use one typed transaction record");
    };
    assert_eq!(record.sequence_lifecycle_operations.len(), 1);
    let target = &record.sequence_lifecycle_operations[0].targets[0];
    assert!(target.target_before.is_none());
    assert_eq!(
        target.target_after.as_ref().map(|identity| identity.oid),
        Some(allocator_before)
    );
    assert!(record.sequence_advances.is_empty());
    assert!(record.sequence_advances_by_oid.is_empty());

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

#[test]
fn restart_and_rename_preserve_oid_defaults_comments_rollback_and_recovery() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(5_010, parsed("CREATE SEQUENCE lifecycle_sequence"))
        .unwrap();
    engine
        .submit_transaction(
            5_011,
            parsed("COMMENT ON SEQUENCE lifecycle_sequence IS 'stable metadata'"),
        )
        .unwrap();
    engine
        .submit_transaction(
            5_012,
            parsed(
                "CREATE TABLE lifecycle_owner \
                 (id int4 DEFAULT nextval('lifecycle_sequence'::regclass), note text)",
            ),
        )
        .unwrap();
    let original = engine
        .relational_catalog_sequence("lifecycle_sequence")
        .unwrap();
    let oid = original.oid;
    let wal_before = engine.durable_wal_records().len();

    engine.submit_transaction(5_013, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            5_013,
            parsed("ALTER SEQUENCE lifecycle_sequence RESTART WITH 41"),
        )
        .unwrap();
    engine
        .submit_transaction(
            5_013,
            parsed("ALTER SEQUENCE lifecycle_sequence RENAME TO lifecycle_sequence_private"),
        )
        .unwrap();
    let private = engine
        .transaction_snapshot_handle(5_013)
        .unwrap()
        .transaction_catalog();
    let renamed = &private.relational_sequences["lifecycle_sequence_private"];
    assert_eq!(renamed.oid, oid);
    assert_eq!((renamed.last_value, renamed.is_called), (41, false));
    assert_eq!(
        table_default_sequence(&private, "lifecycle_owner"),
        "lifecycle_sequence_private"
    );
    assert_eq!(
        private
            .relational_comments
            .get(&RelationalCommentTarget::Sequence {
                sequence: "lifecycle_sequence_private".to_string(),
            })
            .map(String::as_str),
        Some("stable metadata")
    );
    assert!(engine
        .relational_catalog_sequence("lifecycle_sequence_private")
        .is_none());
    assert_eq!(
        table_default_sequence(&engine.catalog_snapshot(), "lifecycle_owner"),
        "lifecycle_sequence"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    engine
        .submit_transaction(5_013, parsed("ROLLBACK"))
        .unwrap();
    let rolled_back = engine
        .relational_catalog_sequence("lifecycle_sequence")
        .unwrap();
    assert_eq!(rolled_back.oid, oid);
    assert_eq!((rolled_back.last_value, rolled_back.is_called), (1, false));

    engine.submit_transaction(5_014, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            5_014,
            parsed("ALTER SEQUENCE lifecycle_sequence RESTART 41"),
        )
        .unwrap();
    engine
        .submit_transaction(
            5_014,
            parsed("ALTER SEQUENCE lifecycle_sequence RENAME TO lifecycle_sequence_final"),
        )
        .unwrap();
    engine.submit_transaction(5_014, parsed("COMMIT")).unwrap();

    let committed = engine
        .relational_catalog_sequence("lifecycle_sequence_final")
        .unwrap();
    assert_eq!(committed.oid, oid);
    assert_eq!((committed.last_value, committed.is_called), (41, false));
    assert_eq!(
        table_default_sequence(&engine.catalog_snapshot(), "lifecycle_owner"),
        "lifecycle_sequence_final"
    );
    assert_eq!(
        engine
            .relational_sequence_comment("lifecycle_sequence_final")
            .as_deref(),
        Some("stable metadata")
    );

    let records = engine.durable_wal_records();
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("sequence lifecycle commit must decode");
    };
    assert_eq!(record.sequence_lifecycle_operations.len(), 2);
    assert_eq!(
        record
            .sequence_lifecycle_operations
            .iter()
            .map(|identity| (identity.command_index, identity.ordinal))
            .collect::<Vec<_>>(),
        vec![(0, 0), (1, 1)]
    );
    assert!(record
        .sequence_lifecycle_operations
        .iter()
        .flat_map(|operation| &operation.targets)
        .all(|target| {
            target
                .target_before
                .as_ref()
                .zip(target.target_after.as_ref())
                .is_some_and(|(before, after)| before.oid == oid && after.oid == oid)
        }));

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

#[test]
fn multi_drop_sequence_is_atomic_dependency_checked_and_rollback_complete() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(5_020, parsed("CREATE SEQUENCE drop_sequence_a"))
        .unwrap();
    engine
        .submit_transaction(5_021, parsed("CREATE SEQUENCE drop_sequence_b"))
        .unwrap();
    engine
        .submit_transaction(
            5_022,
            parsed(
                "CREATE TABLE drop_sequence_owner \
                 (id int4 DEFAULT nextval('drop_sequence_a'::regclass))",
            ),
        )
        .unwrap();
    let b_oid = engine
        .relational_catalog_sequence("drop_sequence_b")
        .unwrap()
        .oid;
    let wal_before = engine.durable_wal_records().len();

    engine.submit_transaction(5_023, parsed("BEGIN")).unwrap();
    let dependency = engine
        .submit_transaction(
            5_023,
            parsed("DROP SEQUENCE drop_sequence_b, drop_sequence_a"),
        )
        .unwrap_err();
    assert!(
        dependency
            .to_string()
            .contains("cannot drop sequence \"drop_sequence_a\"")
            && dependency.to_string().contains("depends on it"),
        "{dependency}"
    );
    let private = engine
        .transaction_snapshot_handle(5_023)
        .unwrap()
        .transaction_catalog();
    assert!(private.relational_sequences.contains_key("drop_sequence_a"));
    assert!(private.relational_sequences.contains_key("drop_sequence_b"));
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    engine
        .submit_transaction(
            5_023,
            parsed("DROP SEQUENCE IF EXISTS missing_drop_sequence, drop_sequence_b"),
        )
        .unwrap();
    let private = engine
        .transaction_snapshot_handle(5_023)
        .unwrap()
        .transaction_catalog();
    assert!(!private.relational_sequences.contains_key("drop_sequence_b"));
    assert!(engine
        .relational_catalog_sequence("drop_sequence_b")
        .is_some());
    engine
        .submit_transaction(5_023, parsed("ROLLBACK"))
        .unwrap();
    assert_eq!(
        engine
            .relational_catalog_sequence("drop_sequence_b")
            .unwrap()
            .oid,
        b_oid
    );

    engine.submit_transaction(5_024, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            5_024,
            parsed("DROP SEQUENCE IF EXISTS missing_drop_sequence, drop_sequence_b"),
        )
        .unwrap();
    engine.submit_transaction(5_024, parsed("COMMIT")).unwrap();
    assert!(engine
        .relational_catalog_sequence("drop_sequence_b")
        .is_none());
    assert!(engine
        .relational_catalog_sequence("drop_sequence_a")
        .is_some());
    let payload = operation_payload(engine.durable_wal_records().last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("sequence drop commit must decode");
    };
    let targets = &record.sequence_lifecycle_operations[0].targets;
    assert_eq!(targets.len(), 2);
    assert!(targets[0].target_before.is_none());
    assert_eq!(
        targets[1].target_before.as_ref().map(|target| target.oid),
        Some(b_oid)
    );

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

#[test]
fn drop_recreate_sequence_uses_a_new_oid_and_cannot_inherit_metadata() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(5_030, parsed("CREATE SEQUENCE aba_sequence"))
        .unwrap();
    engine
        .submit_transaction(
            5_031,
            parsed("COMMENT ON SEQUENCE aba_sequence IS 'old object'"),
        )
        .unwrap();
    let original_oid = engine
        .relational_catalog_sequence("aba_sequence")
        .unwrap()
        .oid;

    engine.submit_transaction(5_032, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(5_032, parsed("DROP SEQUENCE aba_sequence"))
        .unwrap();
    engine
        .submit_transaction(5_032, parsed("CREATE SEQUENCE aba_sequence"))
        .unwrap();
    let private = engine
        .transaction_snapshot_handle(5_032)
        .unwrap()
        .transaction_catalog();
    let replacement_oid = private.relational_sequences["aba_sequence"].oid;
    assert_ne!(replacement_oid, original_oid);
    assert!(!private
        .relational_comments
        .contains_key(&RelationalCommentTarget::Sequence {
            sequence: "aba_sequence".to_string(),
        }));
    engine.submit_transaction(5_032, parsed("COMMIT")).unwrap();

    let replacement = engine.relational_catalog_sequence("aba_sequence").unwrap();
    assert_eq!(replacement.oid, replacement_oid);
    assert_ne!(replacement.oid, original_oid);
    assert_eq!(engine.relational_sequence_comment("aba_sequence"), None);
    let payload = operation_payload(engine.durable_wal_records().last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("sequence ABA commit must decode");
    };
    assert_eq!(
        record.sequence_lifecycle_operations[0].targets[0]
            .target_before
            .as_ref()
            .map(|target| target.oid),
        Some(original_oid)
    );
    assert_eq!(
        record.sequence_lifecycle_operations[1].targets[0]
            .target_after
            .as_ref()
            .map(|target| target.oid),
        Some(replacement_oid)
    );

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_sequence("aba_sequence")
            .unwrap()
            .oid,
        replacement_oid
    );
    assert_eq!(recovered.relational_sequence_comment("aba_sequence"), None);
}

#[test]
fn sequence_lifecycle_guards_value_users_and_release_on_rollback() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(5_040, parsed("CREATE SEQUENCE guarded_sequence"))
        .unwrap();
    let oid = engine
        .relational_catalog_sequence("guarded_sequence")
        .unwrap()
        .oid;

    let reader = engine.table_access.lease();
    reader.acquire_shared([oid]).unwrap();
    let restart = engine
        .submit_transaction(
            5_041,
            parsed("ALTER SEQUENCE guarded_sequence RESTART WITH 9"),
        )
        .unwrap_err();
    assert!(
        matches!(restart, ExecuteError::Serialization(_)),
        "{restart}"
    );
    drop(reader);

    engine.submit_transaction(5_042, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            5_042,
            parsed("ALTER SEQUENCE guarded_sequence RESTART WITH 9"),
        )
        .unwrap();
    let nextval = engine
        .submit_transaction(
            5_043,
            parsed("SELECT nextval('guarded_sequence'::regclass)"),
        )
        .unwrap_err();
    assert!(
        matches!(nextval, ExecuteError::Serialization(_)),
        "{nextval}"
    );
    engine
        .submit_transaction(5_042, parsed("ROLLBACK"))
        .unwrap();

    engine
        .submit_transaction(
            5_044,
            parsed("SELECT nextval('guarded_sequence'::regclass)"),
        )
        .unwrap();
    let sequence = engine
        .relational_catalog_sequence("guarded_sequence")
        .unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (1, true));
}

#[test]
fn ordinary_value_transitions_refuse_private_create_and_restart_before_wal() {
    let engine = Engine::new_local_test_engine();

    engine.submit_transaction(5_045, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(5_045, parsed("CREATE SEQUENCE private_created_sequence"))
        .unwrap();
    let private_create_wal = engine.durable_wal_records().len();
    for sql in [
        "SELECT nextval('private_created_sequence'::regclass)",
        "SELECT setval('private_created_sequence'::regclass, 12, true)",
    ] {
        let error = engine.submit_transaction(5_045, parsed(sql)).unwrap_err();
        assert!(matches!(&error, ExecuteError::Unsupported(_)), "{error}");
        assert!(error.to_string().contains("transaction-private"), "{error}");
        assert_eq!(engine.durable_wal_records().len(), private_create_wal);
    }
    engine
        .submit_transaction(5_045, parsed("ROLLBACK"))
        .unwrap();
    assert!(engine
        .relational_catalog_sequence("private_created_sequence")
        .is_none());

    engine
        .submit_transaction(5_046, parsed("CREATE SEQUENCE private_restarted_sequence"))
        .unwrap();
    let published_before = engine
        .relational_catalog_sequence("private_restarted_sequence")
        .unwrap();
    let private_restart_wal = engine.durable_wal_records().len();
    engine.submit_transaction(5_047, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            5_047,
            parsed("ALTER SEQUENCE private_restarted_sequence RESTART WITH 20"),
        )
        .unwrap();
    for sql in [
        "SELECT nextval('private_restarted_sequence'::regclass)",
        "SELECT setval('private_restarted_sequence'::regclass, 21, false)",
    ] {
        let error = engine.submit_transaction(5_047, parsed(sql)).unwrap_err();
        assert!(matches!(&error, ExecuteError::Unsupported(_)), "{error}");
        assert!(error.to_string().contains("transaction-private"), "{error}");
        assert_eq!(engine.durable_wal_records().len(), private_restart_wal);
    }
    let published_after = engine
        .relational_catalog_sequence("private_restarted_sequence")
        .unwrap();
    assert_eq!(
        (
            published_after.last_value,
            published_after.is_called,
            published_after.oid,
        ),
        (
            published_before.last_value,
            published_before.is_called,
            published_before.oid,
        )
    );
    engine
        .submit_transaction(5_047, parsed("ROLLBACK"))
        .unwrap();
}

#[test]
fn sequence_lifecycle_rebases_without_losing_private_identity() {
    for (offset, begin) in [
        (0_u64, "BEGIN"),
        (10_u64, "BEGIN ISOLATION LEVEL REPEATABLE READ"),
    ] {
        let engine = Engine::new_local();
        engine
            .submit_transaction(5_060 + offset, parsed("CREATE SEQUENCE sequence_rebase"))
            .unwrap();
        let oid = engine
            .relational_catalog_sequence("sequence_rebase")
            .unwrap()
            .oid;
        engine
            .submit_transaction(5_061 + offset, parsed(begin))
            .unwrap();
        engine
            .submit_transaction(
                5_061 + offset,
                parsed("ALTER SEQUENCE sequence_rebase RESTART WITH 23"),
            )
            .unwrap();
        engine
            .submit_transaction(
                5_062 + offset,
                parsed("SET sequence_rebase_unrelated = yes"),
            )
            .unwrap();
        engine
            .submit_transaction(
                5_061 + offset,
                parsed("ALTER SEQUENCE sequence_rebase RENAME TO sequence_rebase_final"),
            )
            .unwrap();
        engine
            .submit_transaction(5_061 + offset, parsed("COMMIT"))
            .unwrap();
        let sequence = engine
            .relational_catalog_sequence("sequence_rebase_final")
            .unwrap();
        assert_eq!(sequence.oid, oid);
        assert_eq!((sequence.last_value, sequence.is_called), (23, false));
    }
}

#[test]
fn concurrent_private_create_sequence_race_serializes_before_second_wal() {
    let engine = Engine::new_local();
    engine.submit_transaction(5_080, parsed("BEGIN")).unwrap();
    engine.submit_transaction(5_081, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(5_080, parsed("CREATE SEQUENCE raced_sequence"))
        .unwrap();
    engine
        .submit_transaction(5_081, parsed("CREATE SEQUENCE raced_sequence"))
        .unwrap();
    engine.submit_transaction(5_080, parsed("COMMIT")).unwrap();
    let wal_after_winner = engine.durable_wal_records().len();
    let loser = engine
        .submit_transaction(5_081, parsed("COMMIT"))
        .unwrap_err();
    assert!(matches!(loser, ExecuteError::Serialization(_)), "{loser}");
    assert_eq!(engine.durable_wal_records().len(), wal_after_winner);
    engine
        .submit_transaction(5_081, parsed("ROLLBACK"))
        .unwrap();
    assert!(engine
        .relational_catalog_sequence("raced_sequence")
        .is_some());
}

#[test]
fn post_durable_sequence_lifecycle_failure_is_recovery_owned() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(5_090, parsed("CREATE SEQUENCE durable_sequence"))
        .unwrap();
    let oid = engine
        .relational_catalog_sequence("durable_sequence")
        .unwrap()
        .oid;
    let wal_before = engine.durable_wal_records().len();
    engine.submit_transaction(5_091, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            5_091,
            parsed("ALTER SEQUENCE durable_sequence RESTART WITH 77"),
        )
        .unwrap();
    engine.fail_next_transaction_post_durable_apply();
    let failed = engine
        .submit_transaction(5_091, parsed("COMMIT"))
        .unwrap_err();
    assert!(matches!(failed, ExecuteError::Indeterminate(_)), "{failed}");
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    let later = engine
        .submit_transaction(5_092, parsed("CREATE SEQUENCE later_sequence"))
        .unwrap_err();
    assert!(
        later.to_string().contains("restart recovery required"),
        "{later}"
    );

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    let sequence = recovered
        .relational_catalog_sequence("durable_sequence")
        .unwrap();
    assert_eq!(sequence.oid, oid);
    assert_eq!((sequence.last_value, sequence.is_called), (77, false));
}

#[test]
fn truncate_restart_identity_is_private_ordered_and_recoverable() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            5_045,
            "CREATE TABLE restart_identity_owner (id SERIAL PRIMARY KEY, note TEXT)",
        )
        .unwrap();
    engine
        .execute_text(
            5_046,
            "INSERT INTO restart_identity_owner (note) VALUES ('old-a'), ('old-b')",
        )
        .unwrap();
    let sequence_name = "restart_identity_owner_id_seq";
    let sequence_oid = engine
        .relational_catalog_sequence(sequence_name)
        .unwrap()
        .oid;
    let select =
        match parse_command("SELECT id, note FROM restart_identity_owner ORDER BY id").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };

    engine.execute_text(5_047, "BEGIN").unwrap();
    let wal_before = engine.durable_wal_records().len();
    engine
        .execute_text(5_047, "TRUNCATE restart_identity_owner RESTART IDENTITY")
        .unwrap();
    let private = engine
        .transaction_snapshot_handle(5_047)
        .unwrap()
        .transaction_catalog();
    let private_sequence = &private.relational_sequences[sequence_name];
    assert_eq!(private_sequence.oid, sequence_oid);
    assert_eq!(
        (private_sequence.last_value, private_sequence.is_called),
        (1, false)
    );
    let outside = engine.execute_relational_select(&select).unwrap_err();
    assert!(
        matches!(outside, ExecuteError::Serialization(_)),
        "the retained reset guard must exclude an outside reader: {outside}"
    );
    let published_sequence = engine.relational_catalog_sequence(sequence_name).unwrap();
    assert_eq!(
        (published_sequence.last_value, published_sequence.is_called),
        (2, true),
        "the private reset must not publish its value state"
    );
    assert!(engine
        .execute_relational_select_in_transaction(5_047, &select)
        .unwrap()
        .rows
        .is_empty());
    engine
        .execute_text(
            5_047,
            "INSERT INTO restart_identity_owner (note) VALUES ('private')",
        )
        .unwrap();
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(5_047, &select)
            .unwrap()
            .rows,
        vec![vec![
            SqlValue::Int4(1),
            SqlValue::Text("private".to_string())
        ]]
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine.execute_text(5_047, "ROLLBACK").unwrap();
    assert_eq!(
        engine
            .execute_relational_select(&select)
            .unwrap()
            .rows
            .len(),
        2
    );
    let sequence = engine.relational_catalog_sequence(sequence_name).unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (2, true));

    engine.execute_text(5_048, "BEGIN").unwrap();
    engine
        .execute_text(5_048, "TRUNCATE restart_identity_owner RESTART IDENTITY")
        .unwrap();
    engine
        .execute_text(
            5_048,
            "INSERT INTO restart_identity_owner (note) VALUES ('committed')",
        )
        .unwrap();
    engine.execute_text(5_048, "COMMIT").unwrap();
    assert_eq!(
        engine.execute_relational_select(&select).unwrap().rows,
        vec![vec![
            SqlValue::Int4(1),
            SqlValue::Text("committed".to_string())
        ]]
    );
    let sequence = engine.relational_catalog_sequence(sequence_name).unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (1, true));

    let records = engine.durable_wal_records();
    assert_eq!(records.len(), wal_before + 1);
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("TRUNCATE RESTART transaction must decode");
    };
    assert_eq!(record.sequence_reset_operations.len(), 1);
    assert_eq!(
        record.sequence_advances_by_oid.get(&sequence_oid),
        Some(&(1, true))
    );
    assert!(record.sequence_advances.is_empty());

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        engine.execute_relational_select(&select).unwrap().rows
    );
    let recovered_sequence = recovered
        .relational_catalog_sequence(sequence_name)
        .unwrap();
    assert_eq!(
        (recovered_sequence.last_value, recovered_sequence.is_called),
        (1, true)
    );
}

#[test]
fn truncate_restart_before_owned_sequence_rename_commits_and_recovers_in_order() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            5_053,
            "CREATE TABLE reset_then_rename_owner (id SERIAL PRIMARY KEY, note TEXT)",
        )
        .unwrap();
    engine
        .execute_text(
            5_054,
            "INSERT INTO reset_then_rename_owner (note) VALUES ('old-a'), ('old-b')",
        )
        .unwrap();
    let original_sequence = "reset_then_rename_owner_id_seq";
    let renamed_sequence = "reset_then_rename_sequence";
    let sequence_oid = engine
        .relational_catalog_sequence(original_sequence)
        .unwrap()
        .oid;

    engine.submit_transaction(5_055, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            5_055,
            parsed("TRUNCATE reset_then_rename_owner RESTART IDENTITY"),
        )
        .unwrap();
    engine
        .submit_transaction(
            5_055,
            parsed(
                "ALTER SEQUENCE reset_then_rename_owner_id_seq \
                 RENAME TO reset_then_rename_sequence",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(5_056, parsed("SET reset_then_rename_rebase = yes"))
        .unwrap();
    engine
        .submit_transaction(
            5_055,
            parsed("ALTER SEQUENCE reset_then_rename_sequence RESTART WITH 1"),
        )
        .expect("READ COMMITTED rebase must retain the rebound reset output");
    engine
        .submit_transaction(5_055, parsed("COMMIT"))
        .expect("a later sequence rename must not invalidate an earlier ordered reset proof");

    let committed = engine.catalog_snapshot();
    assert_eq!(
        table_default_sequence(&committed, "reset_then_rename_owner"),
        renamed_sequence
    );
    let sequence = &committed.relational_sequences[renamed_sequence];
    assert_eq!(sequence.oid, sequence_oid);
    assert_eq!((sequence.last_value, sequence.is_called), (1, false));
    assert!(!committed
        .relational_sequences
        .contains_key(original_sequence));
    let select = match parse_command("SELECT id, note FROM reset_then_rename_owner").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert!(engine
        .execute_relational_select(&select)
        .unwrap()
        .rows
        .is_empty());

    let records = engine.durable_wal_records();
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("reset-then-rename transaction must decode");
    };
    assert!(matches!(
        record.operation_order.as_slice(),
        [
            BinaryTransactionOperationIdentity::TableReset { .. },
            BinaryTransactionOperationIdentity::Catalog { .. },
            BinaryTransactionOperationIdentity::Catalog { .. }
        ]
    ));
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(committed.as_ref()));
    assert!(recovered
        .execute_relational_select(&select)
        .unwrap()
        .rows
        .is_empty());
}

#[test]
fn shadowed_restart_identity_remains_an_ordered_sequence_barrier() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            5_049,
            "CREATE TABLE restart_barrier_owner (id SERIAL PRIMARY KEY, note TEXT)",
        )
        .unwrap();
    engine
        .execute_text(
            5_050,
            "INSERT INTO restart_barrier_owner (note) VALUES ('old-a'), ('old-b')",
        )
        .unwrap();
    let sequence_name = "restart_barrier_owner_id_seq";
    let sequence_oid = engine
        .relational_catalog_sequence(sequence_name)
        .unwrap()
        .oid;
    let wal_before = engine.durable_wal_records().len();

    engine.execute_text(5_051, "BEGIN").unwrap();
    engine
        .execute_text(5_051, "TRUNCATE restart_barrier_owner RESTART IDENTITY")
        .unwrap();
    engine
        .execute_text(
            5_051,
            "INSERT INTO restart_barrier_owner (note) VALUES ('shadowed')",
        )
        .unwrap();
    engine
        .execute_text(5_051, "TRUNCATE restart_barrier_owner CONTINUE IDENTITY")
        .unwrap();
    engine
        .execute_text(
            5_051,
            "INSERT INTO restart_barrier_owner (note) VALUES ('survivor')",
        )
        .unwrap();
    engine.execute_text(5_051, "COMMIT").unwrap();

    let select =
        match parse_command("SELECT id, note FROM restart_barrier_owner ORDER BY id").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
    assert_eq!(
        engine.execute_relational_select(&select).unwrap().rows,
        vec![vec![
            SqlValue::Int4(2),
            SqlValue::Text("survivor".to_string())
        ]]
    );
    let sequence = engine.relational_catalog_sequence(sequence_name).unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (2, true));

    let records = engine.durable_wal_records();
    assert_eq!(records.len(), wal_before + 1);
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("shadowed reset transaction must decode");
    };
    assert_eq!(record.operation_order.len(), 4);
    assert_eq!(record.table_resets.len(), 1);
    assert_eq!(record.table_resets[0].ordinal, 2);
    assert_eq!(record.sequence_reset_operations.len(), 1);
    assert_eq!(record.sequence_reset_operations[0].ordinal, 0);
    assert_eq!(
        record.sequence_advances_by_oid.get(&sequence_oid),
        Some(&(2, true))
    );

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        engine.execute_relational_select(&select).unwrap().rows
    );
    let recovered_sequence = recovered
        .relational_catalog_sequence(sequence_name)
        .unwrap();
    assert_eq!(
        (recovered_sequence.last_value, recovered_sequence.is_called),
        (2, true)
    );
}

#[test]
fn dml_on_both_sides_of_rename_and_restart_binds_one_stable_sequence_oid() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(5_052, "CREATE SEQUENCE ordered_value_sequence")
        .unwrap();
    engine
        .execute_text(
            5_053,
            "CREATE TABLE ordered_value_owner \
             (id INT DEFAULT nextval('ordered_value_sequence'::regclass), note TEXT)",
        )
        .unwrap();
    engine
        .execute_text(
            5_054,
            "INSERT INTO ordered_value_owner (id, note) VALUES (99, 'seed')",
        )
        .unwrap();
    let sequence_oid = engine
        .relational_catalog_sequence("ordered_value_sequence")
        .unwrap()
        .oid;
    let wal_before = engine.durable_wal_records().len();

    engine.execute_text(5_055, "BEGIN").unwrap();
    engine
        .execute_text(
            5_055,
            "INSERT INTO ordered_value_owner (note) VALUES ('before-rename')",
        )
        .unwrap();
    engine
        .submit_transaction(
            5_055,
            parsed(
                "ALTER SEQUENCE ordered_value_sequence RENAME TO ordered_value_sequence_private",
            ),
        )
        .unwrap();
    engine
        .execute_text(
            5_055,
            "INSERT INTO ordered_value_owner (note) VALUES ('after-rename')",
        )
        .unwrap();
    engine
        .submit_transaction(
            5_055,
            parsed("ALTER SEQUENCE ordered_value_sequence_private RESTART WITH 40"),
        )
        .unwrap();
    engine
        .execute_text(
            5_055,
            "INSERT INTO ordered_value_owner (note) VALUES ('after-restart')",
        )
        .unwrap();
    engine.execute_text(5_055, "COMMIT").unwrap();

    let select =
        match parse_command("SELECT id, note FROM ordered_value_owner ORDER BY id").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
    assert_eq!(
        engine.execute_relational_select(&select).unwrap().rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("before-rename".to_string())
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("after-rename".to_string())
            ],
            vec![
                SqlValue::Int4(40),
                SqlValue::Text("after-restart".to_string())
            ],
            vec![SqlValue::Int4(99), SqlValue::Text("seed".to_string())],
        ]
    );
    let sequence = engine
        .relational_catalog_sequence("ordered_value_sequence_private")
        .unwrap();
    assert_eq!(sequence.oid, sequence_oid);
    assert_eq!((sequence.last_value, sequence.is_called), (40, true));
    assert_eq!(
        table_default_sequence(&engine.catalog_snapshot(), "ordered_value_owner"),
        "ordered_value_sequence_private"
    );

    let records = engine.durable_wal_records();
    assert_eq!(
        records.len(),
        wal_before + 3,
        "the two pre-RESTART defaults publish independently before the user envelope"
    );
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("ordered sequence/DML transaction must decode");
    };
    assert_eq!(
        record.sequence_advances_by_oid,
        BTreeMap::from([(sequence_oid, (40, true))])
    );
    assert!(record.sequence_advances.is_empty());
    assert_eq!(
        record.sequence_input_oids.len(),
        1,
        "only the post-RESTART private default belongs to sequence_advances_by_oid"
    );
    assert!(record
        .sequence_input_oids
        .values()
        .all(|oid| *oid == sequence_oid));
    assert_eq!(record.sequence_value_references.len(), 2);
    assert!(record
        .sequence_value_references
        .iter()
        .all(|reference| reference.sequence_oid == sequence_oid));
    assert_eq!(record.sequence_lifecycle_operations.len(), 2);

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        engine.execute_relational_select(&select).unwrap().rows
    );
    let recovered_sequence = recovered
        .relational_catalog_sequence("ordered_value_sequence_private")
        .unwrap();
    assert_eq!(recovered_sequence.oid, sequence_oid);
    assert_eq!(
        (recovered_sequence.last_value, recovered_sequence.is_called),
        (40, true)
    );
}

#[test]
fn restart_after_final_default_advance_is_a_durable_stable_oid_barrier() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(5_060, "CREATE SEQUENCE restart_value_barrier")
        .unwrap();
    engine
        .execute_text(
            5_061,
            "CREATE TABLE restart_value_owner \
             (id INT DEFAULT nextval('restart_value_barrier'::regclass), note TEXT)",
        )
        .unwrap();
    let sequence_oid = engine
        .relational_catalog_sequence("restart_value_barrier")
        .unwrap()
        .oid;

    engine.execute_text(5_062, "BEGIN").unwrap();
    engine
        .execute_text(
            5_062,
            "INSERT INTO restart_value_owner (note) VALUES ('before-restart')",
        )
        .unwrap();
    engine
        .submit_transaction(
            5_062,
            parsed("ALTER SEQUENCE restart_value_barrier RESTART WITH 23"),
        )
        .unwrap();
    engine.execute_text(5_062, "COMMIT").unwrap();

    let sequence = engine
        .relational_catalog_sequence("restart_value_barrier")
        .unwrap();
    assert_eq!((sequence.last_value, sequence.is_called), (23, false));
    let records = engine.durable_wal_records();
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("restart barrier transaction must decode");
    };
    assert!(record.sequence_advances_by_oid.is_empty());
    assert_eq!(record.sequence_value_references.len(), 1);
    assert_eq!(
        record.sequence_value_references[0].sequence_oid,
        sequence_oid
    );

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let recovered_sequence = recovered
        .relational_catalog_sequence("restart_value_barrier")
        .unwrap();
    assert_eq!(
        (recovered_sequence.last_value, recovered_sequence.is_called),
        (23, false)
    );
}

#[test]
fn private_sequence_table_and_rows_publish_as_one_ordered_identity_envelope() {
    let engine = Engine::new_local_test_engine();
    let allocator_before = engine.catalog_snapshot().relational_next_oid;
    let wal_before = engine.durable_wal_records().len();

    engine.submit_transaction(5_056, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(5_056, parsed("CREATE SEQUENCE private_order_sequence"))
        .unwrap();
    engine
        .submit_transaction(
            5_056,
            parsed(
                "CREATE TABLE private_order_owner \
                 (id INT DEFAULT nextval('private_order_sequence'::regclass), note TEXT)",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            5_056,
            parsed("INSERT INTO private_order_owner (note) VALUES ('before-rename')"),
        )
        .unwrap();
    engine
        .submit_transaction(
            5_056,
            parsed("ALTER SEQUENCE private_order_sequence RENAME TO private_order_sequence_final"),
        )
        .unwrap();
    engine
        .submit_transaction(
            5_056,
            parsed("INSERT INTO private_order_owner (note) VALUES ('after-rename')"),
        )
        .unwrap();

    assert!(engine
        .relational_catalog_sequence("private_order_sequence")
        .is_none());
    assert!(engine
        .relational_catalog_table("private_order_owner")
        .is_none());
    let private = engine
        .transaction_snapshot_handle(5_056)
        .unwrap()
        .transaction_catalog();
    let sequence_oid = private.relational_sequences["private_order_sequence_final"].oid;
    let table_oid = private.relational_catalog["private_order_owner"].oid;
    assert_eq!(sequence_oid, allocator_before);
    assert_eq!(table_oid, allocator_before + 1);
    assert_eq!(
        table_default_sequence(&private, "private_order_owner"),
        "private_order_sequence_final"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine.submit_transaction(5_056, parsed("COMMIT")).unwrap();

    let select =
        match parse_command("SELECT id, note FROM private_order_owner ORDER BY id").unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        };
    assert_eq!(
        engine.execute_relational_select(&select).unwrap().rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("before-rename".to_string())
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("after-rename".to_string())
            ],
        ]
    );
    let sequence = engine
        .relational_catalog_sequence("private_order_sequence_final")
        .unwrap();
    assert_eq!(sequence.oid, sequence_oid);
    assert_eq!((sequence.last_value, sequence.is_called), (2, true));

    let records = engine.durable_wal_records();
    assert_eq!(records.len(), wal_before + 1);
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("private sequence/table transaction must decode");
    };
    assert_eq!(record.catalog_commands.len(), 3);
    assert_eq!(record.sequence_lifecycle_operations.len(), 2);
    assert_eq!(
        record.created_table_identities["private_order_owner"].table_oid,
        table_oid
    );
    assert_eq!(
        record.sequence_advances_by_oid,
        BTreeMap::from([(sequence_oid, (2, true))])
    );
    assert!(record
        .sequence_input_oids
        .values()
        .all(|oid| *oid == sequence_oid));

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        engine.execute_relational_select(&select).unwrap().rows
    );
    assert_eq!(
        recovered
            .relational_catalog_sequence("private_order_sequence_final")
            .unwrap()
            .oid,
        sequence_oid
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transactional_sequence_lifecycle_gpu_null_differential() {
    fn fixture() -> Engine {
        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine.set_shard_size_target(64);
        engine
            .submit_transaction(5_100, parsed("CREATE SEQUENCE gpu_sequence"))
            .unwrap();
        engine
            .submit_transaction(
                5_101,
                parsed(
                    "CREATE TABLE gpu_sequence_owner \
                     (id INT DEFAULT nextval('gpu_sequence'::regclass), marker INT, note TEXT)",
                ),
            )
            .unwrap();
        engine
            .submit_transaction(
                5_102,
                parsed("INSERT INTO gpu_sequence_owner VALUES (99, NULL, 'seed')"),
            )
            .unwrap();
        engine
    }

    fn lifecycle(engine: &Engine, txn_id: u64, explicit: bool) {
        let ids = if explicit {
            [txn_id; 5]
        } else {
            // Each omitted default publishes a separate system envelope from the same canonical
            // allocator, so direct test-driver user identities leave room for those claims.
            [txn_id, txn_id + 10, txn_id + 20, txn_id + 30, txn_id + 40]
        };
        if explicit {
            engine.submit_transaction(txn_id, parsed("BEGIN")).unwrap();
        }
        engine
            .submit_transaction(
                ids[0],
                parsed(
                    "INSERT INTO gpu_sequence_owner (marker, note) \
                     VALUES (NULL, 'before-rename')",
                ),
            )
            .unwrap();
        engine
            .submit_transaction(
                ids[1],
                parsed("ALTER SEQUENCE gpu_sequence RENAME TO gpu_sequence_final"),
            )
            .unwrap();
        engine
            .submit_transaction(
                ids[2],
                parsed(
                    "INSERT INTO gpu_sequence_owner (marker, note) \
                     VALUES (7, NULL)",
                ),
            )
            .unwrap();
        engine
            .submit_transaction(
                ids[3],
                parsed("ALTER SEQUENCE gpu_sequence_final RESTART WITH 40"),
            )
            .unwrap();
        engine
            .submit_transaction(
                ids[4],
                parsed(
                    "INSERT INTO gpu_sequence_owner (marker, note) \
                     VALUES (NULL, NULL)",
                ),
            )
            .unwrap();
        if explicit {
            let private = engine
                .execute_relational_select_in_transaction(
                    txn_id,
                    &match parse_command(
                        "SELECT id, marker, note FROM gpu_sequence_owner ORDER BY id",
                    )
                    .unwrap()
                    {
                        Command::Select(select) => select,
                        other => panic!("expected SELECT, got {other:?}"),
                    },
                )
                .unwrap();
            assert!(matches!(private.executed_target, DeviceTarget::Gpu(_)));
            assert_eq!(private.fallback_reason, None);
            engine.submit_transaction(txn_id, parsed("COMMIT")).unwrap();
        }
    }

    let candidate = fixture();
    lifecycle(&candidate, 5_103, true);
    let reference = fixture();
    lifecycle(&reference, 5_104, false);
    let select = match parse_command("SELECT id, marker, note FROM gpu_sequence_owner ORDER BY id")
        .unwrap()
    {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    let candidate_rows = candidate.execute_relational_select(&select).unwrap();
    let reference_rows = reference.execute_relational_select(&select).unwrap();
    assert!(matches!(
        candidate_rows.executed_target,
        DeviceTarget::Gpu(_)
    ));
    assert!(matches!(
        reference_rows.executed_target,
        DeviceTarget::Gpu(_)
    ));
    assert_eq!(candidate_rows.fallback_reason, None);
    assert_eq!(reference_rows.fallback_reason, None);
    assert_eq!(candidate_rows.rows, reference_rows.rows);
    assert_eq!(
        candidate_rows.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Null,
                SqlValue::Text("before-rename".to_string()),
            ],
            vec![SqlValue::Int4(2), SqlValue::Int4(7), SqlValue::Null],
            vec![SqlValue::Int4(40), SqlValue::Null, SqlValue::Null],
            vec![
                SqlValue::Int4(99),
                SqlValue::Null,
                SqlValue::Text("seed".to_string()),
            ],
        ]
    );
    let candidate_sequence = candidate
        .relational_catalog_sequence("gpu_sequence_final")
        .unwrap();
    let reference_sequence = reference
        .relational_catalog_sequence("gpu_sequence_final")
        .unwrap();
    assert_eq!(candidate_sequence.oid, reference_sequence.oid);
    assert_eq!(
        (candidate_sequence.last_value, candidate_sequence.is_called),
        (reference_sequence.last_value, reference_sequence.is_called)
    );
    assert_eq!(
        (candidate_sequence.last_value, candidate_sequence.is_called),
        (40, true)
    );

    let recovered = Engine::recover_from_durable_wal(&candidate.durable_wal_records()).unwrap();
    let recovered_rows = recovered.execute_relational_select(&select).unwrap();
    assert!(matches!(
        recovered_rows.executed_target,
        DeviceTarget::Gpu(_)
    ));
    assert_eq!(recovered_rows.fallback_reason, None);
    assert_eq!(recovered_rows.rows, candidate_rows.rows);
}

#[test]
fn sequence_lifecycle_replay_tamper_is_clone_first_and_effect_free() {
    let source = Engine::new_local();
    source
        .submit_transaction(5_050, parsed("CREATE SEQUENCE proof_sequence"))
        .unwrap();
    source
        .submit_transaction(
            5_051,
            parsed(
                "CREATE TABLE proof_sequence_owner \
                 (id int4 DEFAULT nextval('proof_sequence'::regclass))",
            ),
        )
        .unwrap();
    let prefix = source.durable_wal_records();
    source.submit_transaction(5_052, parsed("BEGIN")).unwrap();
    source
        .submit_transaction(
            5_052,
            parsed("ALTER SEQUENCE proof_sequence RESTART WITH 17"),
        )
        .unwrap();
    source
        .submit_transaction(
            5_052,
            parsed("ALTER SEQUENCE proof_sequence RENAME TO proof_sequence_final"),
        )
        .unwrap();
    source.submit_transaction(5_052, parsed("COMMIT")).unwrap();
    let payload = operation_payload(source.durable_wal_records().last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("sequence lifecycle commit must decode");
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
            tampered.sequence_lifecycle_operations[0].targets[0]
                .target_before
                .as_mut()
                .unwrap()
                .oid += 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.sequence_lifecycle_operations[0].targets[0]
                .target_after
                .as_mut()
                .unwrap()
                .digest[0] ^= 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.sequence_lifecycle_operations[1].targets[0]
                .dependencies_after
                .values_mut()
                .next()
                .unwrap()
                .column_id += 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.sequence_lifecycle_operations[1].ordinal = 0;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered
                .catalog_output
                .as_mut()
                .unwrap()
                .relational_next_oid = u32::MAX;
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
        assert!(catalog.relational_sequences.contains_key("proof_sequence"));
        assert!(!catalog
            .relational_sequences
            .contains_key("proof_sequence_final"));
        assert_eq!(
            table_default_sequence(&after, "proof_sequence_owner"),
            "proof_sequence"
        );
    }
}
