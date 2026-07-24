use super::*;

fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
    gpu_db_sql::ParsedCommand::parse(sql).unwrap()
}

fn operation_payload(record: &WalRecord) -> Arc<[u8]> {
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

fn encoded_operation(codec: u8, body: &[u8]) -> Arc<[u8]> {
    let mut operation = Vec::with_capacity(ENGINE_OPERATION_MAGIC.len() + 12 + body.len());
    operation.extend_from_slice(ENGINE_OPERATION_MAGIC);
    operation.push(codec);
    operation.extend_from_slice(&[0; 3]);
    operation.extend_from_slice(&(body.len() as u64).to_le_bytes());
    operation.extend_from_slice(body);
    Arc::from(operation)
}

fn seed_legacy_index_cursor(engine: &Engine, next_oid: u32) {
    let mut catalog = engine
        .catalog_latch
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    catalog.index_oid_epoch_current = false;
    catalog.legacy_recovery_floor_prepared = true;
    catalog.legacy_recovery_next_index_oid = next_oid;
    catalog.legacy_recovery_index_oids_assigned = false;
    engine.publish_catalog_snapshot(&catalog, engine.committed_seq(), 0);
}

#[test]
fn literal_legacy_sql_and_typed_prefixes_cross_the_index_identity_boundary_once() {
    const LEGACY_TABLE: &[u8] = br#"{"CreateTable":{"table":"mixed_index_codec","columns":[{"name":"id","ty":"Int4","domain":null,"default":null},{"name":"code","ty":"Int4","domain":null,"default":null}],"primary_key":null,"unique_constraints":[],"check_constraints":[]}}"#;
    const CURRENT_INDEX: &[u8] = br#"{"CreateIndex":{"name":"mixed_index_codec_code","table":"mixed_index_codec","column":"code","columns":["code"],"unique":false}}"#;

    let operation = |codec: u8, body: &[u8]| {
        let mut operation = Vec::with_capacity(ENGINE_OPERATION_MAGIC.len() + 12 + body.len());
        operation.extend_from_slice(ENGINE_OPERATION_MAGIC);
        operation.push(codec);
        operation.extend_from_slice(&[0; 3]);
        operation.extend_from_slice(&(body.len() as u64).to_le_bytes());
        operation.extend_from_slice(body);
        operation
    };

    let codec_1 = operation(
        ENGINE_OPERATION_CODEC_LEGACY_SQL,
        b"CREATE TABLE codec_one_index (id int4 PRIMARY KEY)",
    );
    let codec_1_replay = Engine::decode_engine_operation(&codec_1).unwrap();
    assert!(!Engine::engine_command_uses_current_index_semantics(
        &codec_1_replay
    ));
    assert!(matches!(
        Engine::decode_engine_command(&codec_1_replay).unwrap(),
        Some(Command::CreateTable(_))
    ));

    let legacy_replay = Engine::decode_engine_operation(&operation(
        ENGINE_OPERATION_CODEC_TYPED_COMMAND_V1,
        LEGACY_TABLE,
    ))
    .unwrap();
    let current_replay = Engine::decode_engine_operation(&operation(
        ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
        CURRENT_INDEX,
    ))
    .unwrap();
    assert!(!Engine::engine_command_uses_current_index_semantics(
        &legacy_replay
    ));
    assert!(Engine::engine_command_uses_current_index_semantics(
        &current_replay
    ));
    let recovered = Engine::recover_from_durable_wal(&[
        WalRecord {
            txn_id: 6_880,
            payload: legacy_replay,
        },
        WalRecord {
            txn_id: 6_881,
            payload: current_replay,
        },
    ])
    .unwrap();
    let table = &recovered.catalog_snapshot().relational_catalog["mixed_index_codec"];
    assert_eq!(table.indexes.len(), 1);
    assert_eq!(table.indexes[0].name, "mixed_index_codec_code");
    assert_eq!(table.indexes[0].table, table.name);
}

#[test]
fn codec_four_drop_if_exists_rejects_a_wrong_kind_without_partial_catalog_apply() {
    const CURRENT_DROP: &[u8] = br#"{"DropIndex":{"names":["codec_four_drop_idx","codec_four_drop_wrong_kind"],"if_exists":true}}"#;
    let prefix = [
        WalRecord {
            txn_id: 6_882,
            payload: Arc::from(&b"CREATE TABLE codec_four_drop_owner (id int4, code int4)"[..]),
        },
        WalRecord {
            txn_id: 6_883,
            payload: Arc::from(
                &b"CREATE INDEX codec_four_drop_idx ON codec_four_drop_owner (code)"[..],
            ),
        },
        WalRecord {
            txn_id: 6_884,
            payload: Arc::from(&b"CREATE TABLE codec_four_drop_wrong_kind (id int4)"[..]),
        },
    ];
    let current = WalRecord {
        txn_id: 6_885,
        payload: Engine::decode_engine_operation(&encoded_operation(
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2,
            CURRENT_DROP,
        ))
        .unwrap(),
    };
    let complete_source = prefix
        .iter()
        .cloned()
        .chain(std::iter::once(current.clone()))
        .collect::<Vec<_>>();

    let recovered = Engine::new_local();
    recovered
        .prepare_legacy_index_oid_recovery(&complete_source)
        .unwrap();
    recovered.begin_recovery_replay();
    recovered.replay_durable_records(&prefix).unwrap();
    let before = recovered.catalog_snapshot();
    let error = recovered
        .replay_durable_records(std::slice::from_ref(&current))
        .expect_err("codec 4 must not reinterpret an existing table as an absent index");
    assert!(
        error
            .to_string()
            .contains("relation \"codec_four_drop_wrong_kind\" is not an index"),
        "{error}"
    );
    assert!(recovered.catalog_snapshot().same_contents(before.as_ref()));
    assert!(
        recovered.catalog_snapshot().relational_catalog["codec_four_drop_owner"]
            .indexes
            .iter()
            .any(|index| index.name == "codec_four_drop_idx")
    );
}

#[test]
fn legacy_index_at_int4_max_allows_current_rename_drop_and_repeat_recovery() {
    let legacy = [
        WalRecord {
            txn_id: 6_886,
            payload: Arc::from(&b"CREATE TABLE legacy_max_index_owner (id int4, code int4)"[..]),
        },
        WalRecord {
            txn_id: 6_887,
            payload: Arc::from(
                &b"CREATE INDEX legacy_max_index_name ON legacy_max_index_owner (code)"[..],
            ),
        },
    ];
    let source = Engine::new_local();
    seed_legacy_index_cursor(&source, MAX_CATALOG_OID);
    source.begin_recovery_replay();
    source.replay_durable_records(&legacy).unwrap();
    source.finish_recovery_replay().unwrap();
    let migrated =
        &source.catalog_snapshot().relational_catalog["legacy_max_index_owner"].indexes[0];
    assert_eq!(migrated.oid, MAX_CATALOG_OID);

    source
        .execute_text(
            6_888,
            "ALTER INDEX legacy_max_index_name RENAME TO legacy_max_index_renamed",
        )
        .expect("allocation-neutral current DDL must accept the exhausted sentinel");
    let renamed = source.catalog_snapshot();
    assert_eq!(
        renamed.relational_catalog["legacy_max_index_owner"].indexes[0].oid,
        MAX_CATALOG_OID
    );
    assert_eq!(renamed.relational_next_oid, MAX_CATALOG_OID + 1);
    source
        .execute_text(6_889, "DROP INDEX legacy_max_index_renamed")
        .expect("allocation-neutral DROP must retain the exhausted sentinel");
    let committed = source.catalog_snapshot();
    assert!(committed.relational_catalog["legacy_max_index_owner"]
        .indexes
        .is_empty());
    assert_eq!(committed.relational_next_oid, MAX_CATALOG_OID + 1);

    let records = source.durable_wal_records();
    let repeated = Engine::new_local();
    seed_legacy_index_cursor(&repeated, MAX_CATALOG_OID);
    repeated
        .prepare_legacy_index_oid_recovery(&records)
        .unwrap();
    repeated.begin_recovery_replay();
    repeated.replay_durable_records(&records).unwrap();
    repeated.finish_recovery_replay().unwrap();
    assert!(repeated
        .catalog_snapshot()
        .same_contents(committed.as_ref()));
}

#[test]
fn complete_legacy_prefix_places_early_index_oids_above_later_relations() {
    const EARLY: &[u8] = b"CREATE TABLE legacy_oid_early (id int4 PRIMARY KEY, code int4 UNIQUE)";
    let late_columns = (0..4_050)
        .map(|column| format!("s{column} serial"))
        .collect::<Vec<_>>()
        .join(",");
    let late = format!("CREATE TABLE legacy_oid_late ({late_columns})");
    let records = vec![
        WalRecord {
            txn_id: 6_890,
            payload: Arc::from(EARLY),
        },
        WalRecord {
            txn_id: 6_891,
            payload: Arc::from(late.into_bytes()),
        },
    ];

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let before_current = recovered.catalog_snapshot();
    let early = &before_current.relational_catalog["legacy_oid_early"];
    assert_eq!(early.indexes.len(), 2);
    let max_late_relation_oid = before_current
        .relational_sequences
        .values()
        .map(|sequence| sequence.oid)
        .chain(
            before_current
                .relational_catalog
                .values()
                .map(|table| table.oid),
        )
        .max()
        .unwrap();
    assert!(max_late_relation_oid >= FIRST_LEGACY_RECOVERY_INDEX_OID);
    assert!(early
        .indexes
        .iter()
        .all(|index| index.oid > max_late_relation_oid));

    recovered
        .submit_transaction(
            6_892,
            gpu_db_sql::ParsedCommand::parse(
                "CREATE INDEX legacy_oid_post_boundary ON legacy_oid_early (code)",
            )
            .unwrap(),
        )
        .unwrap();
    let after_current = recovered.catalog_snapshot();
    let post = after_current.relational_catalog["legacy_oid_early"]
        .indexes
        .iter()
        .find(|index| index.name == "legacy_oid_post_boundary")
        .unwrap();
    assert!(post.oid > early.indexes.iter().map(|index| index.oid).max().unwrap());
    assert_eq!(after_current.relational_next_oid, post.oid + 1);
    let restarted = Engine::recover_from_durable_wal(&recovered.durable_wal_records()).unwrap();
    assert!(restarted
        .catalog_snapshot()
        .same_contents(after_current.as_ref()));
}

#[test]
fn legacy_add_primary_and_unique_constraints_prepare_one_complete_index_prefix() {
    let records = [
        (
            6_892,
            "CREATE TABLE legacy_add_prefix (id serial, payload int4)",
        ),
        (6_893, "CREATE TABLE legacy_add_owner (id int4, code int4)"),
        (
            6_894,
            "ALTER TABLE ONLY legacy_add_owner \
             ADD CONSTRAINT legacy_add_owner_pkey PRIMARY KEY (id)",
        ),
        (
            6_895,
            "ALTER TABLE ONLY legacy_add_owner \
             ADD CONSTRAINT legacy_add_owner_code_key UNIQUE (code)",
        ),
    ]
    .into_iter()
    .map(|(txn_id, sql)| WalRecord {
        txn_id,
        payload: Arc::from(sql.as_bytes()),
    })
    .collect::<Vec<_>>();

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let catalog = recovered.catalog_snapshot();
    let owner = &catalog.relational_catalog["legacy_add_owner"];
    assert_eq!(
        owner
            .indexes
            .iter()
            .map(|index| (
                index.name.as_str(),
                index.primary_key,
                index.unique_constraint
            ))
            .collect::<Vec<_>>(),
        vec![
            ("legacy_add_owner_pkey", true, false),
            ("legacy_add_owner_code_key", false, true),
        ]
    );
    let relation_high_water = catalog
        .relational_catalog
        .values()
        .map(|table| table.oid)
        .chain(
            catalog
                .relational_sequences
                .values()
                .map(|sequence| sequence.oid),
        )
        .max()
        .unwrap();
    assert!(owner
        .indexes
        .iter()
        .all(|index| index.oid > relation_high_water));

    let chunked = Engine::new_local();
    chunked.prepare_legacy_index_oid_recovery(&records).unwrap();
    chunked.begin_recovery_replay();
    for record in &records {
        chunked
            .replay_durable_records(std::slice::from_ref(record))
            .unwrap();
    }
    chunked.finish_recovery_replay().unwrap();
    assert!(chunked.catalog_snapshot().same_contents(catalog.as_ref()));
}

#[test]
fn legacy_constraint_backed_index_rename_and_drop_keep_historical_replay_policy() {
    let records = [
        (
            6_895,
            "CREATE TABLE legacy_constraint_indexes \
             (id int4 PRIMARY KEY, code int4 UNIQUE)",
        ),
        (
            6_896,
            "COMMENT ON INDEX legacy_constraint_indexes_pkey IS 'legacy primary index'",
        ),
        (
            6_897,
            "COMMENT ON CONSTRAINT legacy_constraint_indexes_pkey \
             ON legacy_constraint_indexes IS 'legacy primary constraint'",
        ),
        (
            6_898,
            "COMMENT ON INDEX legacy_constraint_indexes_code_key IS 'legacy unique index'",
        ),
        (
            6_899,
            "COMMENT ON CONSTRAINT legacy_constraint_indexes_code_key \
             ON legacy_constraint_indexes IS 'legacy unique constraint'",
        ),
        (
            6_900,
            "ALTER INDEX legacy_constraint_indexes_pkey \
             RENAME TO legacy_constraint_indexes_pkey_old",
        ),
        (
            6_901,
            "DROP INDEX legacy_constraint_indexes_code_key, \
             legacy_constraint_indexes_code_key",
        ),
    ]
    .into_iter()
    .map(|(txn_id, sql)| WalRecord {
        txn_id,
        payload: Arc::from(sql.as_bytes()),
    })
    .collect::<Vec<_>>();
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let table = &recovered.catalog_snapshot().relational_catalog["legacy_constraint_indexes"];
    assert_eq!(table.indexes.len(), 1);
    assert_eq!(table.indexes[0].name, "legacy_constraint_indexes_pkey_old");
    assert!(table.indexes[0].primary_key);
    assert_eq!(
        recovered
            .relational_index_comment("legacy_constraint_indexes_pkey_old")
            .as_deref(),
        Some("legacy primary index")
    );
    assert_eq!(
        recovered
            .relational_constraint_comment(
                "legacy_constraint_indexes",
                "legacy_constraint_indexes_pkey_old"
            )
            .as_deref(),
        Some("legacy primary constraint")
    );
    assert_eq!(
        recovered.relational_index_comment("legacy_constraint_indexes_code_key"),
        None
    );
    assert_eq!(
        recovered.relational_constraint_comment(
            "legacy_constraint_indexes",
            "legacy_constraint_indexes_code_key"
        ),
        None
    );
}

#[test]
fn current_index_epoch_keeps_later_index_neutral_transaction_opcodes_byte_stable() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            6_900,
            parsed("CREATE TABLE index_epoch_source (id int4, code int4)"),
        )
        .unwrap();

    engine.submit_transaction(6_901, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            6_901,
            parsed("CREATE INDEX index_epoch_first ON index_epoch_source (code)"),
        )
        .unwrap();
    engine.submit_transaction(6_901, parsed("COMMIT")).unwrap();

    engine.submit_transaction(6_902, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            6_902,
            parsed("CREATE VIEW index_epoch_view AS SELECT id FROM index_epoch_source"),
        )
        .unwrap();
    engine.submit_transaction(6_902, parsed("COMMIT")).unwrap();

    engine.submit_transaction(6_903, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            6_903,
            parsed("CREATE TABLE index_epoch_plain (id int4, code int4)"),
        )
        .unwrap();
    engine.submit_transaction(6_903, parsed("COMMIT")).unwrap();

    engine.submit_transaction(6_904, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            6_904,
            parsed("CREATE INDEX index_epoch_second ON index_epoch_plain (code)"),
        )
        .unwrap();
    engine.submit_transaction(6_904, parsed("COMMIT")).unwrap();

    let records = engine.durable_wal_records();
    assert_eq!(&operation_payload(&records[1])[..3], &[255, 1, 16]);
    assert_eq!(
        &operation_payload(&records[2])[..3],
        &[255, 1, 12],
        "pure CREATE VIEW retains the accepted opcode after the index boundary"
    );
    assert_eq!(
        &operation_payload(&records[3])[..3],
        &[255, 1, 10],
        "index-neutral CREATE TABLE retains the accepted opcode after the index boundary"
    );
    assert_eq!(&operation_payload(&records[4])[..3], &[255, 1, 16]);

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));

    let mut forbidden = records;
    forbidden.push(WalRecord {
        txn_id: 6_905,
        payload: Arc::from(&b"CREATE TABLE forbidden_late_legacy (id int4 PRIMARY KEY)"[..]),
    });
    let error = match Engine::recover_from_durable_wal(&forbidden) {
        Ok(_) => panic!("an index-sensitive legacy record crossed the current boundary"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("index identity boundary"),
        "{error}"
    );
}

#[test]
fn first_index_neutral_catalog_change_after_legacy_indexes_durably_migrates_once() {
    let legacy = WalRecord {
        txn_id: 6_910,
        payload: Arc::from(
            &b"CREATE TABLE legacy_view_source (id int4 PRIMARY KEY, code int4)"[..],
        ),
    };
    let engine = Engine::recover_from_durable_wal(&[legacy]).unwrap();
    let legacy_index_oid =
        engine.catalog_snapshot().relational_catalog["legacy_view_source"].indexes[0].oid;

    engine.submit_transaction(6_911, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            6_911,
            parsed("CREATE VIEW first_migration_view AS SELECT code FROM legacy_view_source"),
        )
        .unwrap();
    engine.submit_transaction(6_911, parsed("COMMIT")).unwrap();

    let records = engine.durable_wal_records();
    let payload = operation_payload(records.last().unwrap());
    assert_eq!(
        &payload[..3],
        &[255, 1, 16],
        "the first post-legacy catalog transaction must durably mark the allocator migration"
    );
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("migration marker must remain one ordered transaction record");
    };
    assert_eq!(
        record.catalog_epoch,
        BinaryTransactionCatalogEpoch::IndexIdentityV1
    );
    assert_eq!(record.view_lifecycle_operations.len(), 1);
    assert!(record.index_lifecycle_operations.is_empty());

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let catalog = recovered.catalog_snapshot();
    assert_eq!(
        catalog.relational_catalog["legacy_view_source"].indexes[0].oid,
        legacy_index_oid
    );
    assert!(catalog
        .relational_views
        .contains_key("first_migration_view"));
    assert!(catalog.same_contents(engine.catalog_snapshot().as_ref()));
}

#[test]
fn empty_legacy_index_prefix_still_transitions_a_neutral_catalog_transaction() {
    let legacy = [
        WalRecord {
            txn_id: 6_912,
            payload: Arc::from(&b"CREATE TABLE empty_index_prefix_source (id int4)"[..]),
        },
        WalRecord {
            txn_id: 6_913,
            payload: Arc::from(&b"DROP INDEX IF EXISTS empty_index_prefix_absent"[..]),
        },
    ];
    let engine = Engine::recover_from_durable_wal(&legacy).unwrap();
    let legacy_catalog = engine.catalog_snapshot();
    assert!(!legacy_catalog.index_oid_epoch_current);
    assert!(!legacy_catalog.legacy_recovery_index_oids_assigned);

    engine.submit_transaction(6_914, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            6_914,
            parsed("CREATE TABLE empty_index_prefix_current (id int4)"),
        )
        .unwrap();
    engine.submit_transaction(6_914, parsed("COMMIT")).unwrap();

    let records = engine.durable_wal_records();
    let payload = operation_payload(records.last().unwrap());
    assert_eq!(
        &payload[..3],
        &[255, 1, 16],
        "the first admitted catalog command must finalize even an empty legacy index prefix"
    );
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("the allocator transition must remain one ordered transaction record");
    };
    assert_eq!(
        record.catalog_epoch,
        BinaryTransactionCatalogEpoch::IndexIdentityV1
    );
    assert!(record.index_lifecycle_operations.is_empty());

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .relational_catalog
        .contains_key("empty_index_prefix_current"));
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

#[test]
fn transition_selector_tamper_rejects_before_wal_without_catalog_effect() {
    let legacy = WalRecord {
        txn_id: 6_915,
        payload: Arc::from(
            &b"CREATE TABLE transition_tamper_source \
               (id int4 PRIMARY KEY, code int4)"[..],
        ),
    };
    let engine = Engine::recover_from_durable_wal(&[legacy]).unwrap();
    let published_before = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();
    engine.submit_transaction(6_916, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            6_916,
            parsed(
                "CREATE VIEW transition_tamper_view AS \
                 SELECT code FROM transition_tamper_source",
            ),
        )
        .unwrap();
    let snapshot = engine.transaction_snapshot_handle(6_916).unwrap();
    {
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let TransactionOperation::Catalog(staged) = &mut delta.operations[0] else {
            panic!("expected catalog operation");
        };
        assert!(staged.index_epoch_transition);
        Arc::make_mut(staged).index_epoch_transition = false;
    }

    let error = engine
        .submit_transaction(6_916, parsed("COMMIT"))
        .expect_err("a forged opcode selector must reject before claiming WAL");
    assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
    assert!(error.to_string().contains("catalog epoch"), "{error}");
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(engine
        .catalog_snapshot()
        .same_contents(published_before.as_ref()));
    assert!(!engine
        .catalog_snapshot()
        .relational_views
        .contains_key("transition_tamper_view"));
    engine
        .submit_transaction(6_916, parsed("ROLLBACK"))
        .unwrap();
}

#[test]
fn legacy_sequence_rename_and_drop_keep_historical_default_bindings() {
    let records = [
        (6_930, "CREATE SEQUENCE legacy_rename_sequence"),
        (
            6_931,
            "CREATE TABLE legacy_rename_owner \
             (id int4 DEFAULT nextval('legacy_rename_sequence'::regclass))",
        ),
        (
            6_932,
            "ALTER SEQUENCE legacy_rename_sequence \
             RENAME TO legacy_renamed_sequence",
        ),
        (6_933, "CREATE SEQUENCE legacy_drop_sequence"),
        (
            6_934,
            "CREATE TABLE legacy_drop_owner \
             (id int4 DEFAULT nextval('legacy_drop_sequence'::regclass))",
        ),
        (6_935, "DROP SEQUENCE legacy_drop_sequence"),
    ]
    .into_iter()
    .map(|(txn_id, sql)| WalRecord {
        txn_id,
        payload: Arc::from(sql.as_bytes()),
    })
    .collect::<Vec<_>>();

    let recovered = Engine::recover_from_durable_wal(&records)
        .expect("acknowledged generic-SQL WAL must retain its historical sequence semantics");
    let catalog = recovered.catalog_snapshot();
    assert!(!catalog
        .relational_sequences
        .contains_key("legacy_rename_sequence"));
    assert!(catalog
        .relational_sequences
        .contains_key("legacy_renamed_sequence"));
    assert!(!catalog
        .relational_sequences
        .contains_key("legacy_drop_sequence"));
    for (table, expected_sequence) in [
        ("legacy_rename_owner", "legacy_rename_sequence"),
        ("legacy_drop_owner", "legacy_drop_sequence"),
    ] {
        let default = catalog.relational_catalog[table]
            .columns
            .iter()
            .find_map(|column| match &column.default {
                Some(ColumnDefault::SequenceNextVal { sequence, .. }) => Some(sequence.as_str()),
                _ => None,
            });
        assert_eq!(
            default,
            Some(expected_sequence),
            "legacy replay must not retroactively rewrite acknowledged default text"
        );
    }
}

#[test]
fn split_checkpoint_replay_inherits_the_current_index_identity_epoch() {
    let legacy = WalRecord {
        txn_id: 6_920,
        payload: Arc::from(
            &b"CREATE TABLE split_epoch_source (id int4 PRIMARY KEY, code int4)"[..],
        ),
    };
    let source = Engine::recover_from_durable_wal(&[legacy]).unwrap();

    source.submit_transaction(6_921, parsed("BEGIN")).unwrap();
    source
        .submit_transaction(
            6_921,
            parsed("CREATE VIEW split_epoch_boundary AS SELECT code FROM split_epoch_source"),
        )
        .unwrap();
    source.submit_transaction(6_921, parsed("COMMIT")).unwrap();

    source.submit_transaction(6_922, parsed("BEGIN")).unwrap();
    source
        .submit_transaction(
            6_922,
            parsed("CREATE VIEW split_epoch_suffix AS SELECT id FROM split_epoch_source"),
        )
        .unwrap();
    source.submit_transaction(6_922, parsed("COMMIT")).unwrap();

    let records = source.durable_wal_records();
    assert_eq!(records.len(), 3);
    assert_eq!(&operation_payload(&records[1])[..3], &[255, 1, 16]);
    assert_eq!(
        &operation_payload(&records[2])[..3],
        &[255, 1, 12],
        "the post-boundary view keeps the established byte-stable opcode"
    );

    // Match checkpoint/lane reopen: bind the complete identity source, replay the checkpoint
    // through its migration marker, then admit the neutral suffix in a separate scanner call.
    let recovered = Engine::new_local();
    recovered
        .prepare_legacy_index_oid_recovery(&records)
        .unwrap();
    recovered.begin_recovery_replay();
    recovered
        .replay_durable_records(&records[..2])
        .expect("the checkpoint prefix must assign legacy identities and cross the boundary");
    recovered
        .replay_durable_records(&records[2..])
        .expect("the suffix scanner must inherit the crossed index-identity epoch");
    recovered.finish_recovery_replay().unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(source.catalog_snapshot().as_ref()));
}
