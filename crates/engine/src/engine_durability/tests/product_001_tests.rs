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

struct DurableFixture {
    directory: std::path::PathBuf,
    wal: std::path::PathBuf,
}

impl DurableFixture {
    fn new(label: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "gpu-db-product-001-durability-{label}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("create durable fixture directory");
        Self {
            wal: directory.join("canonical.wal"),
            directory,
        }
    }
}

impl Drop for DurableFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn durable_ids(engine: &Engine, table: &str) -> Vec<i32> {
    let Command::Select(select) =
        parse_command(&format!("SELECT id FROM {table} ORDER BY id")).expect("parse id query")
    else {
        panic!("expected SELECT")
    };
    engine
        .execute_relational_select(&select)
        .expect("execute id query")
        .rows
        .iter()
        .map(|row| match row[0] {
            SqlValue::Int4(id) => id,
            ref other => panic!("expected int4 id, got {other:?}"),
        })
        .collect()
}

#[test]
fn product_001_pre_fsync_failure_has_no_visibility_or_recovery_effect_and_retry_is_clean() {
    let fixture = DurableFixture::new("pre-fsync");
    let mut engine = Engine::with_durable_wal_segment(&fixture.wal);
    engine
        .execute_text(71_000, "CREATE TABLE product_fsync (id INT PRIMARY KEY)")
        .expect("durable fixture DDL");
    let durable_before = engine.durable_wal_records();
    let visible_before = engine.visible_up_to();
    let flushed_before = engine.wal_flushed_count();

    engine.simulate_next_wal_flush_failure();
    let failure = engine.execute_text(71_001, "INSERT INTO product_fsync VALUES (1)");
    assert!(
        matches!(
            failure,
            Err(ExecuteError::Engine(EngineError::Durability(_)))
        ),
        "a pre-fsync failure must not acknowledge a mutation: {failure:?}"
    );
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(engine.wal_flushed_count(), flushed_before);
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert!(durable_ids(&engine, "product_fsync").is_empty());

    // Restart before retrying: recovery must see exactly the prefix acknowledged before the
    // injected fsync failure, never a transient live row or a buffered failed record.
    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal).expect("recover prefix");
    assert_eq!(recovered.durable_wal_records(), durable_before);
    assert!(durable_ids(&recovered, "product_fsync").is_empty());

    recovered
        .execute_text(71_001, "INSERT INTO product_fsync VALUES (1)")
        .expect("the same unacknowledged transaction identity retries cleanly");
    drop(recovered);
    let reopened =
        Engine::open_durable_wal_segment_auto(&fixture.wal).expect("recover acknowledged retry");
    assert_eq!(durable_ids(&reopened, "product_fsync"), vec![1]);
}

#[test]
fn product_001_post_durable_transaction_apply_failure_wedges_live_and_recovers_once() {
    let fixture = DurableFixture::new("post-durable");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    engine
        .execute_text(
            72_000,
            "CREATE TABLE product_indeterminate (id INT PRIMARY KEY, value INT)",
        )
        .expect("durable fixture DDL");
    engine
        .submit_transaction(72_001, parsed("BEGIN"))
        .expect("begin explicit transaction");
    engine
        .submit_transaction(
            72_001,
            parsed("INSERT INTO product_indeterminate VALUES (1, 10)"),
        )
        .expect("stage explicit write");
    engine.fail_next_transaction_post_durable_apply();
    let failure = engine
        .submit_transaction(72_001, parsed("COMMIT"))
        .expect_err("post-durable apply fault must be indeterminate");
    assert!(
        failure.is_indeterminate(),
        "expected indeterminate failure: {failure}"
    );
    assert!(engine.is_commit_path_poisoned());
    assert!(
        engine
            .execute_text(72_002, "INSERT INTO product_indeterminate VALUES (2, 20)")
            .is_err(),
        "a post-durable live engine must fail closed until restart recovery"
    );

    // The durable record owns the outcome: replay installs the explicit commit exactly once,
    // clears the live wedge, and the recovered canonical WAL remains appendable.
    drop(engine);
    let recovered =
        Engine::open_durable_wal_segment_auto(&fixture.wal).expect("recover indeterminate commit");
    assert_eq!(durable_ids(&recovered, "product_indeterminate"), vec![1]);
    let recovered_records = recovered.durable_wal_records();
    let committed = recovered_records
        .iter()
        .find(|record| record.txn_id == 72_001)
        .expect("post-durable explicit transaction remains in canonical WAL");
    let request_digest = gpu_db_wal::canonical_request_digest(&operation_payload(committed));
    assert_eq!(
        recovered
            .commit_state()
            .resolve_transaction_retry_digest_outcome(72_001, request_digest)
            .expect("recovery indexes the durable transaction retry outcome")
            .expect("matching retry digest resolves")
            .1,
        1,
        "the recovered terminal outcome reports one applied user row"
    );
    let records_before_second_reopen = recovered_records.len();
    drop(recovered);
    let recovered_again =
        Engine::open_durable_wal_segment_auto(&fixture.wal).expect("repeat recovery is exact");
    assert_eq!(
        recovered_again.durable_wal_records().len(),
        records_before_second_reopen,
        "a second reopen must not append or duplicate the recovered explicit transaction"
    );
    assert_eq!(
        durable_ids(&recovered_again, "product_indeterminate"),
        vec![1]
    );
    recovered_again
        .execute_text(72_002, "INSERT INTO product_indeterminate VALUES (2, 20)")
        .expect("recovered engine continues appending");
    drop(recovered_again);
    let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("second recovery after continued append");
    assert_eq!(durable_ids(&reopened, "product_indeterminate"), vec![1, 2]);
    assert_eq!(
        reopened.durable_wal_records().len(),
        records_before_second_reopen + 1,
        "only the acknowledged post-recovery append extends the canonical WAL"
    );
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

#[test]
fn ordinary_sequence_transition_retry_and_recovery_never_double_consume() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(7_000, parsed("CREATE SEQUENCE retry_value"))
        .unwrap();
    let wal_before = engine.durable_wal_records().len();

    let first = engine
        .submit_transaction(7_001, parsed("SELECT nextval('retry_value'::regclass)"))
        .unwrap();
    let TransactionAdmissionResult::SequenceValue(first) = first else {
        panic!("nextval must return its durable value");
    };
    assert_eq!(first.value, 1);
    assert_eq!(first.transition_txn_id, 7_001);
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);

    let retry = engine
        .submit_transaction(7_001, parsed("SELECT nextval('retry_value'::regclass)"))
        .unwrap();
    let TransactionAdmissionResult::SequenceValue(retry) = retry else {
        panic!("same-id retry must return its memoized value");
    };
    assert_eq!(retry, first);
    assert_eq!(
        engine.durable_wal_records().len(),
        wal_before + 1,
        "same-id retry must not append or consume"
    );
    let transition_record = engine.durable_wal_records()[wal_before].clone();
    engine
        .submit_transaction(
            7_002,
            parsed("ALTER SEQUENCE retry_value RENAME TO retry_value_renamed"),
        )
        .unwrap();
    let retry_after_rename = engine
        .submit_transaction(7_001, parsed("SELECT nextval('retry_value'::regclass)"))
        .unwrap();
    let TransactionAdmissionResult::SequenceValue(retry_after_rename) = retry_after_rename else {
        panic!("retry must resolve before re-binding a renamed source name");
    };
    assert_eq!(retry_after_rename.value, 1);

    let records = engine.durable_wal_records();
    let payload = operation_payload(&transition_record);
    let BinaryWalRecord::SequenceValueTransition(transition) =
        decode_binary_record(&payload).unwrap()
    else {
        panic!("ordinary nextval must use the typed transition opcode");
    };
    assert_eq!(transition.transition_txn_id, 7_001);
    assert_eq!(
        (transition.prior_last_value, transition.prior_is_called),
        (1, false)
    );
    assert_eq!(
        (transition.new_last_value, transition.new_is_called),
        (1, true)
    );

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let recovered_wal_before = recovered.durable_wal_records().len();
    let recovered_retry = recovered
        .submit_transaction(7_001, parsed("SELECT nextval('retry_value'::regclass)"))
        .unwrap();
    let TransactionAdmissionResult::SequenceValue(recovered_retry) = recovered_retry else {
        panic!("recovered same-id retry must return its memoized value");
    };
    assert_eq!(recovered_retry.value, 1);
    assert_eq!(recovered.durable_wal_records().len(), recovered_wal_before);
    let next = recovered
        .submit_transaction(
            7_010,
            parsed("SELECT nextval('retry_value_renamed'::regclass)"),
        )
        .unwrap();
    let TransactionAdmissionResult::SequenceValue(next) = next else {
        panic!("new nextval must return a value");
    };
    assert_eq!(next.value, 2);
}

#[test]
fn ordinary_default_transition_precedes_and_is_referenced_by_user_wal() {
    let engine = Engine::new_local_test_engine();
    engine
        .submit_transaction(7_100, parsed("CREATE SEQUENCE default_wal_value"))
        .unwrap();
    engine
        .submit_transaction(
            7_101,
            parsed(
                "CREATE TABLE default_wal_owner \
                 (id INT DEFAULT nextval('default_wal_value'::regclass), note INT)",
            ),
        )
        .unwrap();
    let sequence_oid = engine
        .relational_catalog_sequence("default_wal_value")
        .unwrap()
        .oid;
    let table = engine
        .relational_catalog_table("default_wal_owner")
        .unwrap();
    let id_column = table
        .columns
        .iter()
        .find(|column| column.name == "id")
        .unwrap();

    let wal_before_rollback = engine.durable_wal_records().len();
    engine.submit_transaction(7_102, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            7_102,
            parsed("INSERT INTO default_wal_owner (note) VALUES (NULL)"),
        )
        .unwrap();
    engine
        .submit_transaction(7_102, parsed("ROLLBACK"))
        .unwrap();
    assert_eq!(
        engine.durable_wal_records().len(),
        wal_before_rollback + 1,
        "rollback keeps only the independently durable default transition"
    );
    assert_eq!(
        {
            let sequence = engine
                .relational_catalog_sequence("default_wal_value")
                .unwrap();
            (sequence.last_value, sequence.is_called)
        },
        (1, true)
    );

    let wal_before_commit = engine.durable_wal_records().len();
    engine
        .submit_transaction(
            7_110,
            parsed("INSERT INTO default_wal_owner (note) VALUES (9)"),
        )
        .unwrap();
    let records = engine.durable_wal_records();
    assert_eq!(records.len(), wal_before_commit + 2);
    let transition_payload = operation_payload(&records[records.len() - 2]);
    let BinaryWalRecord::SequenceValueTransition(transition) =
        decode_binary_record(&transition_payload).unwrap()
    else {
        panic!("default must first publish one ordinary transition");
    };
    assert_eq!(transition.parent_txn_id, 7_110);
    assert_eq!(transition.returned_value, 2);
    assert!(matches!(
        transition.operation,
        BinarySequenceValueOperation::Default
    ));

    let user_payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(user) = decode_binary_record(&user_payload).unwrap() else {
        panic!("default INSERT must publish one referenced user transaction");
    };
    let [reference] = user.sequence_value_references.as_slice() else {
        panic!("user WAL must carry exactly one materialized default reference");
    };
    assert_eq!(reference.transition_txn_id, transition.transition_txn_id);
    assert_eq!(reference.parent_txn_id, 7_110);
    assert_eq!(reference.sequence_oid, sequence_oid);
    assert_eq!(reference.returned_value, 2);
    assert_eq!(reference.table_oid, table.oid);
    assert_eq!(reference.column_id, id_column.id);
    assert_eq!(reference.staging_row_ordinal, 0);
    let BinaryTransactionMutation::Insert { row_id, .. } = &user.mutations[0] else {
        panic!("materialized default must bind one INSERT entity");
    };
    assert_eq!(reference.row_id, *row_id);
    assert!(!reference.final_value_overwritten);
    assert!(reference.default_expression);

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let recovered_sequence = recovered
        .relational_catalog_sequence("default_wal_value")
        .unwrap();
    assert_eq!(
        (recovered_sequence.last_value, recovered_sequence.is_called),
        (2, true)
    );
    let select = match parse_command("SELECT id, note FROM default_wal_owner").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        vec![vec![SqlValue::Int4(2), SqlValue::Int4(9)]]
    );
    let recovered_wal_before_retry = recovered.durable_wal_records().len();
    let retry = recovered
        .submit_transaction(
            7_110,
            parsed("INSERT INTO default_wal_owner (note) VALUES (9)"),
        )
        .unwrap();
    let TransactionAdmissionResult::Dml(retry) = retry else {
        panic!("recovered default INSERT retry must resolve its terminal result");
    };
    assert_eq!(retry.rows_affected, 1);
    assert_eq!(
        recovered.durable_wal_records().len(),
        recovered_wal_before_retry,
        "recovered retry must append neither a transition nor a user envelope"
    );
    let mismatch = recovered
        .submit_transaction(
            7_110,
            parsed("INSERT INTO default_wal_owner (note) VALUES (10)"),
        )
        .unwrap_err();
    assert!(
        mismatch.to_string().contains("different request"),
        "{mismatch}"
    );
    assert_eq!(
        recovered
            .relational_catalog_sequence("default_wal_value")
            .unwrap()
            .last_value,
        2
    );
}

#[test]
fn failed_default_retry_reuses_the_durable_expression_identity() {
    let engine = Engine::new_local_test_engine();
    engine
        .submit_transaction(7_200, parsed("CREATE SEQUENCE failed_default_value"))
        .unwrap();
    engine
        .submit_transaction(
            7_201,
            parsed(
                "CREATE TABLE failed_default_owner \
                 (id INT DEFAULT nextval('failed_default_value'::regclass), \
                  required INT, marker INT DEFAULT 0)",
            ),
        )
        .unwrap();
    let statement = "INSERT INTO failed_default_owner (marker) VALUES (9)";
    let wal_before = engine.durable_wal_records().len();
    let first = engine
        .submit_transaction(7_202, parsed(statement))
        .unwrap_err();
    assert!(
        first.to_string().contains("provide every column"),
        "{first}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert_eq!(
        engine
            .relational_catalog_sequence("failed_default_value")
            .unwrap()
            .last_value,
        1
    );

    let retry = engine
        .submit_transaction(7_202, parsed(statement))
        .unwrap_err();
    assert!(
        retry.to_string().contains("provide every column"),
        "{retry}"
    );
    assert_eq!(
        engine.durable_wal_records().len(),
        wal_before + 1,
        "same request must reuse the durable expression outcome"
    );
    let mismatch = engine
        .submit_transaction(
            7_202,
            parsed("INSERT INTO failed_default_owner (marker) VALUES (10)"),
        )
        .unwrap_err();
    assert!(
        mismatch.to_string().contains("different request"),
        "{mismatch}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    let cross_family = engine
        .submit_transaction(7_202, parsed("CREATE TABLE wrong_parent_claim (id INT)"))
        .unwrap_err();
    assert!(
        cross_family.to_string().contains("different request"),
        "{cross_family}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert!(!engine
        .catalog_snapshot()
        .relational_catalog
        .contains_key("wrong_parent_claim"));
    for legacy_error in [
        engine
            .execute_text(7_202, "CREATE TABLE legacy_wrong_parent_claim (id INT)")
            .unwrap_err(),
        engine
            .execute_dml_concurrent(
                7_202,
                "UPDATE failed_default_owner SET marker = 1 WHERE marker = 9",
            )
            .unwrap_err(),
        engine
            .execute_dml_concurrent_instrumented(
                7_202,
                "DELETE FROM failed_default_owner WHERE marker = 9",
                || panic!("cross-family identity rejection must precede DML preparation"),
            )
            .unwrap_err(),
    ] {
        assert!(
            legacy_error.to_string().contains("different request"),
            "{legacy_error}"
        );
    }
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert!(!engine
        .catalog_snapshot()
        .relational_catalog
        .contains_key("legacy_wrong_parent_claim"));

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    let recovered_wal_before = recovered.durable_wal_records().len();
    let retry = recovered
        .submit_transaction(7_202, parsed(statement))
        .unwrap_err();
    assert!(
        retry.to_string().contains("provide every column"),
        "{retry}"
    );
    assert_eq!(
        recovered.durable_wal_records().len(),
        recovered_wal_before,
        "recovery must retain the expression-identity outcome without a user envelope"
    );
    assert_eq!(
        recovered
            .relational_catalog_sequence("failed_default_value")
            .unwrap()
            .last_value,
        1
    );
    let recovered_cross_family = recovered
        .submit_transaction(
            7_202,
            parsed("CREATE TABLE recovered_wrong_parent_claim (id INT)"),
        )
        .unwrap_err();
    assert!(
        recovered_cross_family
            .to_string()
            .contains("different request"),
        "{recovered_cross_family}"
    );
    let recovered_legacy = recovered
        .execute_text(
            7_202,
            "CREATE TABLE recovered_legacy_wrong_parent_claim (id INT)",
        )
        .unwrap_err();
    assert!(
        recovered_legacy.to_string().contains("different request"),
        "{recovered_legacy}"
    );
}

#[test]
fn sequence_transition_and_reference_tamper_are_clone_first_and_effect_free() {
    let source = Engine::new_local_test_engine();
    source
        .submit_transaction(7_300, parsed("CREATE SEQUENCE tamper_value"))
        .unwrap();
    source
        .submit_transaction(
            7_301,
            parsed(
                "CREATE TABLE tamper_owner \
                 (id INT DEFAULT nextval('tamper_value'::regclass), note INT)",
            ),
        )
        .unwrap();
    source
        .submit_transaction(7_302, parsed("INSERT INTO tamper_owner (note) VALUES (8)"))
        .unwrap();
    let records = source.durable_wal_records();
    let transition_position = records.len() - 2;
    let BinaryWalRecord::SequenceValueTransition(transition) =
        decode_binary_record(&operation_payload(&records[transition_position])).unwrap()
    else {
        panic!("expected typed sequence transition");
    };
    let user_payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(user) = decode_binary_record(&user_payload).unwrap() else {
        panic!("expected referenced user transaction");
    };
    let wrapped_base_len =
        usize::try_from(u64::from_le_bytes(user_payload[3..11].try_into().unwrap())).unwrap();
    let reference_count_at = 11 + wrapped_base_len;
    let mut huge_reference_count = user_payload.to_vec();
    huge_reference_count[reference_count_at..reference_count_at + 4]
        .copy_from_slice(&u32::MAX.to_le_bytes());
    let recovery_error = match Engine::recover_from_durable_wal(&[WalRecord {
        txn_id: records.last().unwrap().txn_id,
        payload: Arc::from(huge_reference_count),
    }]) {
        Ok(_) => panic!("recovery must reject an oversized sequence-reference count"),
        Err(error) => error,
    };
    assert!(
        recovery_error
            .to_string()
            .contains("reference count does not match remaining bytes"),
        "{recovery_error}"
    );

    let transition_target =
        Engine::recover_from_durable_wal(&records[..transition_position]).unwrap();
    let before = transition_target.catalog_snapshot();
    let entry = LogEntry {
        term: 1,
        index: before.commit_seq + 1,
        payload: Arc::from(&b""[..]),
    };
    let mut wrong_prior = transition.clone();
    wrong_prior.prior_last_value = 11;
    wrong_prior.new_last_value = 11;
    wrong_prior.returned_value = 11;
    assert!(valid_sequence_value_transition(&wrong_prior));
    let mut catalog = transition_target.ddl_catalog().clone();
    let error = transition_target
        .apply_sequence_value_transition_record(&entry, &mut catalog, wrong_prior)
        .unwrap_err();
    assert!(error.to_string().contains("prior state"), "{error}");
    assert!(
        Engine::catalog_snapshot_from_working(&catalog, before.commit_seq)
            .same_contents(before.as_ref())
    );
    assert!(transition_target
        .sequence_value_outcomes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_empty());

    let user_target = Engine::recover_from_durable_wal(&records[..records.len() - 1]).unwrap();
    let before = user_target.catalog_snapshot();
    let select = match parse_command("SELECT id, note FROM tamper_owner").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert!(user_target
        .execute_relational_select(&select)
        .unwrap()
        .rows
        .is_empty());
    let entry = LogEntry {
        term: 1,
        index: before.commit_seq + 1,
        payload: Arc::from(&b""[..]),
    };
    for tampered in [
        {
            let mut tampered = user.clone();
            tampered.sequence_value_references[0].returned_value += 1;
            tampered
        },
        {
            let mut tampered = user.clone();
            tampered.sequence_value_references[0].table_oid += 1;
            tampered
        },
        {
            let mut tampered = user.clone();
            tampered.sequence_value_references[0].transition_txn_id += 1;
            tampered
        },
        {
            let mut tampered = user.clone();
            tampered.sequence_value_references[0].row_id += 1;
            tampered
        },
        {
            let mut tampered = user.clone();
            let BinaryTransactionMutation::Insert { row_encoded, .. } = &mut tampered.mutations[0]
            else {
                panic!("expected final INSERT mutation");
            };
            *row_encoded = encode_relational_row(&[SqlValue::Int4(99), SqlValue::Int4(8)]);
            tampered
        },
        {
            let mut tampered = user.clone();
            tampered.sequence_value_references[0].final_value_overwritten = true;
            tampered
        },
    ] {
        let mut catalog = user_target.ddl_catalog().clone();
        assert!(user_target
            .apply_binary_transaction_record(&entry, &mut catalog, tampered)
            .is_err());
        assert!(
            Engine::catalog_snapshot_from_working(&catalog, before.commit_seq)
                .same_contents(before.as_ref())
        );
        assert!(user_target
            .execute_relational_select(&select)
            .unwrap()
            .rows
            .is_empty());
    }
}

#[test]
fn sequence_default_row_binding_preserves_later_update_and_delete() {
    let engine = Engine::new_local_test_engine();
    engine
        .submit_transaction(7_400, parsed("CREATE SEQUENCE rebound_default"))
        .unwrap();
    engine
        .submit_transaction(
            7_410,
            parsed(
                "CREATE TABLE rebound_owner \
                 (id INT DEFAULT nextval('rebound_default'::regclass), note INT)",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(7_415, parsed("CREATE TABLE rebound_noise (id INT)"))
        .unwrap();

    engine.submit_transaction(7_420, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(7_420, parsed("INSERT INTO rebound_owner (note) VALUES (1)"))
        .unwrap();
    engine
        .submit_transaction(7_430, parsed("INSERT INTO rebound_noise VALUES (9)"))
        .unwrap();
    engine
        .submit_transaction(
            7_420,
            parsed("UPDATE rebound_owner SET id = 77 WHERE note = 1"),
        )
        .unwrap();
    engine.submit_transaction(7_420, parsed("COMMIT")).unwrap();
    let records = engine.durable_wal_records();
    let BinaryWalRecord::Transaction(updated) =
        decode_binary_record(&operation_payload(records.last().unwrap())).unwrap()
    else {
        panic!("expected referenced update transaction");
    };
    let [updated_reference] = updated.sequence_value_references.as_slice() else {
        panic!("updated INSERT must retain one default reference");
    };
    assert!(updated_reference.final_value_overwritten);
    let BinaryTransactionMutation::Insert {
        row_id,
        row_encoded,
        ..
    } = &updated.mutations[0]
    else {
        panic!("insert-then-update must coalesce to INSERT");
    };
    assert_eq!(updated_reference.row_id, *row_id);
    assert_eq!(
        decode_relational_row(
            row_encoded,
            &engine
                .catalog_snapshot()
                .relational_catalog
                .get("rebound_owner")
                .unwrap()
                .columns,
        )
        .unwrap(),
        vec![SqlValue::Int4(77), SqlValue::Int4(1)]
    );

    engine.submit_transaction(7_440, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(7_440, parsed("INSERT INTO rebound_owner (note) VALUES (2)"))
        .unwrap();
    engine
        .submit_transaction(7_440, parsed("DELETE FROM rebound_owner WHERE note = 2"))
        .unwrap();
    engine.submit_transaction(7_440, parsed("COMMIT")).unwrap();
    let records = engine.durable_wal_records();
    let BinaryWalRecord::Transaction(deleted) =
        decode_binary_record(&operation_payload(records.last().unwrap())).unwrap()
    else {
        panic!("expected referenced delete transaction");
    };
    let [deleted_reference] = deleted.sequence_value_references.as_slice() else {
        panic!("deleted INSERT must retain one default reference");
    };
    assert!(deleted_reference.final_value_overwritten);
    assert!(deleted.mutations.is_empty());

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let select = match parse_command("SELECT id, note FROM rebound_owner").unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    };
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        vec![vec![SqlValue::Int4(77), SqlValue::Int4(1)]]
    );
    assert_eq!(
        recovered
            .relational_catalog_sequence("rebound_default")
            .unwrap()
            .last_value,
        2
    );
}

#[test]
fn compatibility_assigned_ids_advance_the_shared_transition_allocator() {
    let engine = Engine::new_local_test_engine();
    engine.execute_text(90, "BEGIN").unwrap();
    assert_eq!(
        engine.allocate_transaction_id().unwrap(),
        91,
        "a compatibility-assigned user id must be reserved before an engine-owned transition"
    );
    engine.execute_text(90, "ROLLBACK").unwrap();
}

#[test]
fn sequence_default_route_catalog_pin_rejects_drift_before_transition() {
    let engine = Engine::new_local_test_engine();
    engine
        .submit_transaction(1, parsed("CREATE SEQUENCE route_default"))
        .unwrap();
    engine
        .submit_transaction(
            2,
            parsed(
                "CREATE TABLE route_owner \
                 (id INT DEFAULT nextval('route_default'::regclass), note INT)",
            ),
        )
        .unwrap();
    let insert = parsed("INSERT INTO route_owner (note) VALUES (9)");
    let command = insert.command().clone();
    let (specialized, route_catalog_version) = engine.insert_sequence_default_route(&command);
    assert!(specialized);
    let route_catalog_version = route_catalog_version.unwrap();

    engine
        .submit_transaction(
            3,
            parsed("ALTER TABLE route_owner ALTER COLUMN id DROP DEFAULT"),
        )
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    let sequence_before = engine.relational_catalog_sequence("route_default").unwrap();
    let error = engine
        .execute_sequence_default_autocommit(
            10,
            command,
            Some(
                crate::engine_mutation_admission::CatalogVersionExpectation::SequenceRoute(
                    route_catalog_version,
                ),
            ),
        )
        .unwrap_err();
    assert!(matches!(&error, ExecuteError::Serialization(_)), "{error}");
    assert!(error.to_string().contains("dependencies"), "{error}");
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    let sequence_after = engine.relational_catalog_sequence("route_default").unwrap();
    assert_eq!(
        (sequence_after.last_value, sequence_after.is_called),
        (sequence_before.last_value, sequence_before.is_called)
    );
    assert!(engine.transaction_snapshot_handle(10).is_none());
}

#[test]
fn negative_sequence_default_route_rejects_add_default_before_transition() {
    let engine = Engine::new_local_test_engine();
    engine
        .submit_transaction(1, parsed("CREATE SEQUENCE added_route_default"))
        .unwrap();
    engine
        .submit_transaction(
            2,
            parsed("CREATE TABLE added_route_owner (id INT, note INT)"),
        )
        .unwrap();
    let insert_source = "INSERT INTO added_route_owner (note) VALUES (9)";
    let insert = parsed(insert_source);
    let command = insert.command().clone();
    let (specialized, route_catalog_version) = engine.insert_sequence_default_route(&command);
    assert!(
        !specialized,
        "the admitted generation has no sequence default"
    );
    let route_catalog_version = route_catalog_version.unwrap();

    engine
        .submit_transaction(
            3,
            parsed(
                "ALTER TABLE added_route_owner ALTER COLUMN id \
                 SET DEFAULT nextval('added_route_default'::regclass)",
            ),
        )
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    let sequence_before = engine
        .relational_catalog_sequence("added_route_default")
        .unwrap();
    let error = engine
        .execute_parsed_dml_concurrent_with_catalog(
            10,
            command,
            insert_source,
            Some(
                crate::engine_mutation_admission::CatalogVersionExpectation::SequenceRoute(
                    route_catalog_version,
                ),
            ),
        )
        .unwrap_err();
    assert!(matches!(&error, ExecuteError::Serialization(_)), "{error}");
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    let sequence_after = engine
        .relational_catalog_sequence("added_route_default")
        .unwrap();
    assert_eq!(
        (sequence_after.last_value, sequence_after.is_called),
        (sequence_before.last_value, sequence_before.is_called)
    );
    assert!(engine.transaction_snapshot_handle(10).is_none());
}
