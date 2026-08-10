use super::*;

#[path = "product_001_tests/product_001_copy_tests.rs"]
mod product_001_copy_tests;
#[path = "product_001_tests/product_001_mixed_dml_tests.rs"]
mod product_001_mixed_dml_tests;

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

fn operation_payload_if_not_codec5(record: &WalRecord) -> Option<Arc<[u8]>> {
    let envelope = gpu_db_wal::decode_canonical_record_payload(&record.payload)
        .unwrap()
        .unwrap();
    (!Engine::canonical_envelope_is_codec5(&envelope)).then(|| operation_payload(record))
}

fn codec5_replay_metadata(
    record: &WalRecord,
) -> crate::typed_insert_aggregate::SemanticsV2ReplayMetadata {
    let envelope = gpu_db_wal::decode_canonical_record_payload(&record.payload)
        .expect("decode codec-5 canonical envelope")
        .expect("codec-5 record must use a canonical envelope");
    assert!(Engine::canonical_envelope_is_codec5(&envelope));
    let fragments = envelope
        .fragments
        .iter()
        .map(|fragment| gpu_db_wal::CanonicalFragmentRef {
            kind: fragment.kind,
            body: &fragment.body,
        })
        .collect::<Vec<_>>();
    crate::typed_insert_aggregate::decode_closed_semantics_v2_replay(
        &envelope.header,
        &envelope.outcome,
        &fragments,
    )
    .expect("codec-5 INSERT must pass strict closure")
    .expect("live INSERT must select semantics-v2")
    .metadata()
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
    let retry_record = reopened
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == 71_001)
        .expect("acknowledged retry remains in canonical WAL");
    let retry_envelope = gpu_db_wal::decode_canonical_record_payload(&retry_record.payload)
        .expect("decode retry record")
        .expect("retry uses a canonical envelope");
    assert!(
        Engine::canonical_envelope_is_codec5(&retry_envelope),
        "the clean retry must not enter a displaced INSERT authority"
    );
}

#[test]
fn product_001_codec5_pre_durable_crash_prefixes_reopen_empty_and_retry_once() {
    use crate::engine_transaction_delta::TransactionTerminalPreDurableFault;

    let boundaries = [
        (
            "before-replication-proposal",
            TransactionTerminalPreDurableFault::ReplicationProposal,
        ),
        (
            "before-canonical-wal-append",
            TransactionTerminalPreDurableFault::CanonicalWalAppend,
        ),
        (
            "before-wal-flush",
            TransactionTerminalPreDurableFault::WalFlush,
        ),
    ];

    for (ordinal, (label, boundary)) in boundaries.into_iter().enumerate() {
        let fixture = DurableFixture::new(label);
        let engine = Engine::with_durable_wal_segment(&fixture.wal);
        let create_txn = 71_100 + u64::try_from(ordinal).unwrap() * 10;
        let insert_txn = create_txn + 1;
        engine
            .execute_text(
                create_txn,
                "CREATE TABLE product_codec5_prefix (id INT PRIMARY KEY, value BIGINT)",
            )
            .expect("durable crash-prefix fixture DDL");
        let durable_before = engine.durable_wal_records();
        let visible_before = engine.visible_up_to();
        let flushed_before = engine.wal_flushed_count();

        engine.fail_next_transaction_terminal_pre_durable_at(boundary);
        let failure = engine.execute_text(
            insert_txn,
            "INSERT INTO product_codec5_prefix VALUES (1, 9000000001)",
        );
        assert!(
            matches!(
                failure,
                Err(ExecuteError::Engine(EngineError::Durability(_)))
            ),
            "{label} must fail before acknowledgement: {failure:?}"
        );
        assert_eq!(engine.visible_up_to(), visible_before, "{label}");
        assert_eq!(engine.wal_flushed_count(), flushed_before, "{label}");
        assert_eq!(engine.durable_wal_records(), durable_before, "{label}");
        assert!(durable_ids(&engine, "product_codec5_prefix").is_empty());

        // Dropping here is the crash. Recovery must consume only the acknowledged prefix; in
        // particular the tentative codec-5 append at WalFlush must not become replayable.
        drop(engine);
        let recovered =
            Engine::open_durable_wal_segment_auto(&fixture.wal).expect("recover crash prefix");
        assert_eq!(recovered.durable_wal_records(), durable_before, "{label}");
        assert!(durable_ids(&recovered, "product_codec5_prefix").is_empty());

        recovered
            .execute_text(
                insert_txn,
                "INSERT INTO product_codec5_prefix VALUES (1, 9000000001)",
            )
            .expect("the unacknowledged codec-5 request must retry through the same authority");
        drop(recovered);
        let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("recover the acknowledged retry");
        assert_eq!(durable_ids(&reopened, "product_codec5_prefix"), vec![1]);
        let retry_record = reopened
            .durable_wal_records()
            .into_iter()
            .find(|record| record.txn_id == insert_txn)
            .expect("the acknowledged retry remains in canonical WAL");
        let retry_envelope = gpu_db_wal::decode_canonical_record_payload(&retry_record.payload)
            .expect("decode crash-prefix retry record")
            .expect("crash-prefix retry uses a canonical envelope");
        assert!(
            Engine::canonical_envelope_is_codec5(&retry_envelope),
            "{label} retry must not enter the displaced resolved INSERT authority"
        );
    }
}

#[test]
fn product_001_private_create_index_crash_prefixes_retry_once() {
    use crate::engine_transaction_delta::TransactionTerminalPreDurableFault;

    let boundaries = [
        (
            "before-replication-proposal",
            TransactionTerminalPreDurableFault::ReplicationProposal,
        ),
        (
            "before-canonical-wal-append",
            TransactionTerminalPreDurableFault::CanonicalWalAppend,
        ),
        (
            "before-wal-flush",
            TransactionTerminalPreDurableFault::WalFlush,
        ),
    ];
    let stage_private_index = |engine: &Engine, txn_id| {
        for sql in [
            "BEGIN",
            "CREATE TABLE product_private_index_prefix (id INT, value INT)",
            "CREATE UNIQUE INDEX product_private_index_prefix_id \
             ON product_private_index_prefix (id)",
            "INSERT INTO product_private_index_prefix VALUES (1, 10), (2, 20)",
        ] {
            engine
                .submit_transaction(txn_id, parsed(sql))
                .expect("stage transaction-private CREATE INDEX and typed INSERT");
        }
    };

    for (ordinal, (label, boundary)) in boundaries.into_iter().enumerate() {
        let fixture = DurableFixture::new(&format!("private-create-index-{label}"));
        let engine = Engine::with_durable_wal_segment(&fixture.wal);
        let txn_id = 71_300 + u64::try_from(ordinal).unwrap();
        let durable_before = engine.durable_wal_records();
        let visible_before = engine.visible_up_to();
        stage_private_index(&engine, txn_id);
        engine.fail_next_transaction_terminal_pre_durable_at(boundary);
        let failure = engine.submit_transaction(txn_id, parsed("COMMIT"));
        assert!(
            matches!(
                failure,
                Err(ExecuteError::Engine(EngineError::Durability(_)))
            ),
            "{label} must fail before acknowledgement: {failure:?}"
        );
        assert_eq!(engine.durable_wal_records(), durable_before, "{label}");
        assert_eq!(engine.visible_up_to(), visible_before, "{label}");
        assert!(
            engine
                .relational_catalog_table("product_private_index_prefix")
                .is_none(),
            "{label} must not publish the private table or index"
        );

        drop(engine);
        let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("recover the private CREATE INDEX crash prefix");
        assert_eq!(recovered.durable_wal_records(), durable_before, "{label}");
        assert!(
            recovered
                .relational_catalog_table("product_private_index_prefix")
                .is_none(),
            "{label} fresh reopen must not materialize the private table or index"
        );

        stage_private_index(&recovered, txn_id);
        recovered
            .submit_transaction(txn_id, parsed("COMMIT"))
            .expect("the unacknowledged private CREATE INDEX transaction retries through codec-5");
        assert_eq!(
            durable_ids(&recovered, "product_private_index_prefix"),
            vec![1, 2],
            "{label} retry publishes both private indexed rows exactly once"
        );
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            recovered.relational_named_index_covered_rows("product_private_index_prefix"),
            Some(2),
            "{label} retry publishes the first private named-index generation exactly once"
        );
        let record = recovered
            .durable_wal_records()
            .into_iter()
            .find(|record| record.txn_id == txn_id)
            .expect("the acknowledged private-index retry remains in canonical WAL");
        assert!(codec5_replay_metadata(&record).initial_table_absent);

        drop(recovered);
        let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("fresh reopen must recover the acknowledged private-index retry");
        assert_eq!(
            durable_ids(&reopened, "product_private_index_prefix"),
            vec![1, 2],
            "{label} fresh replay owns the private indexed rowset"
        );
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            reopened.relational_named_index_covered_rows("product_private_index_prefix"),
            Some(2),
            "{label} fresh replay owns the first private named-index generation"
        );
    }
}

#[test]
fn product_001_private_create_index_post_durable_recovery_owns_once() {
    let fixture = DurableFixture::new("private-create-index-post-durable");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_350;
    let records_before = engine.durable_wal_records().len();
    for sql in [
        "BEGIN",
        "CREATE TABLE product_private_index_indeterminate (id INT, value INT)",
        "CREATE UNIQUE INDEX product_private_index_indeterminate_id \
         ON product_private_index_indeterminate (id)",
        "INSERT INTO product_private_index_indeterminate VALUES (1, 10), (2, 20)",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage post-durable private CREATE INDEX transaction");
    }
    engine.fail_next_transaction_post_durable_apply();
    let failure = engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect_err("post-durable private-index apply fault must be indeterminate");
    assert!(
        failure.is_indeterminate(),
        "expected indeterminate failure: {failure}"
    );
    assert!(engine.is_commit_path_poisoned());
    assert_eq!(engine.durable_wal_records().len(), records_before + 1);

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("recover the indeterminate private CREATE INDEX transaction");
    assert_eq!(recovered.durable_wal_records().len(), records_before + 1);
    assert_eq!(
        durable_ids(&recovered, "product_private_index_indeterminate"),
        vec![1, 2]
    );
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        recovered.relational_named_index_covered_rows("product_private_index_indeterminate"),
        Some(2),
        "post-durable recovery must publish the first private named-index generation"
    );
    let record = recovered
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == txn_id)
        .expect("the durable private-index transaction remains in canonical WAL");
    assert!(codec5_replay_metadata(&record).initial_table_absent);

    let records_before_second_reopen = recovered.durable_wal_records().len();
    drop(recovered);
    let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("repeat private-index recovery is exact");
    assert_eq!(
        reopened.durable_wal_records().len(),
        records_before_second_reopen,
        "repeat recovery must not append or duplicate the private CREATE INDEX transaction"
    );
    assert_eq!(
        durable_ids(&reopened, "product_private_index_indeterminate"),
        vec![1, 2]
    );
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        reopened.relational_named_index_covered_rows("product_private_index_indeterminate"),
        Some(2),
        "repeat recovery must retain the first private named-index generation"
    );
}

#[test]
fn product_001_existing_table_s3_index_replacement_crash_prefixes_retry_once() {
    use crate::engine_transaction_delta::TransactionTerminalPreDurableFault;

    let boundaries = [
        (
            "before-replication-proposal",
            TransactionTerminalPreDurableFault::ReplicationProposal,
        ),
        (
            "before-canonical-wal-append",
            TransactionTerminalPreDurableFault::CanonicalWalAppend,
        ),
        (
            "before-wal-flush",
            TransactionTerminalPreDurableFault::WalFlush,
        ),
    ];
    let stage_transition = |engine: &Engine, txn_id| {
        for sql in [
            "BEGIN",
            "DROP INDEX product_existing_s3_index_prefix_original_code_region",
            "CREATE UNIQUE INDEX product_existing_s3_index_prefix_replacement_code_region \
             ON product_existing_s3_index_prefix (code, region)",
            "INSERT INTO product_existing_s3_index_prefix VALUES \
             (3, 30, 'west'), (4, 40, 'south'), (5, NULL, 'south')",
        ] {
            engine
                .submit_transaction(txn_id, parsed(sql))
                .expect("stage populated-table S3 CREATE INDEX transition");
        }
    };

    for (ordinal, (label, boundary)) in boundaries.into_iter().enumerate() {
        let fixture = DurableFixture::new(&format!("existing-s3-index-{label}"));
        let engine = Engine::with_durable_wal_segment(&fixture.wal);
        engine
            .execute_text(
                71_380 + u64::try_from(ordinal).unwrap() * 10,
                "CREATE TABLE product_existing_s3_index_prefix \
                 (id INT PRIMARY KEY, code INT, region TEXT)",
            )
            .expect("create populated-table S3 transition fixture");
        engine
            .execute_text(
                71_381 + u64::try_from(ordinal).unwrap() * 10,
                "INSERT INTO product_existing_s3_index_prefix VALUES \
                 (1, NULL, 'north'), (2, 20, 'east')",
            )
            .expect("seed the published prefix through codec-5");
        engine
            .execute_text(
                71_382 + u64::try_from(ordinal).unwrap() * 10,
                "CREATE UNIQUE INDEX product_existing_s3_index_prefix_original_code_region \
                 ON product_existing_s3_index_prefix (code, region)",
            )
            .expect("create the published predecessor index through codec-5");
        let durable_before = engine.durable_wal_records();
        let visible_before = engine.visible_up_to();
        let next_oid_before = engine.catalog_snapshot().relational_next_oid;
        let txn_id = 71_383 + u64::try_from(ordinal).unwrap();
        stage_transition(&engine, txn_id);
        engine.fail_next_transaction_terminal_pre_durable_at(boundary);
        let failure = engine.submit_transaction(txn_id, parsed("COMMIT"));
        assert!(
            matches!(
                failure,
                Err(ExecuteError::Engine(EngineError::Durability(_)))
            ),
            "{label} must fail before acknowledgement: {failure:?}"
        );
        assert_eq!(engine.durable_wal_records(), durable_before, "{label}");
        assert_eq!(engine.visible_up_to(), visible_before, "{label}");
        assert_eq!(
            engine.catalog_snapshot().relational_next_oid,
            next_oid_before,
            "{label} must not publish the private S3 index identity"
        );
        assert!(
            engine
                .relational_catalog_table("product_existing_s3_index_prefix")
                .unwrap()
                .indexes
                .iter()
                .any(|index| index.name == "product_existing_s3_index_prefix_original_code_region"),
            "{label} must retain the published predecessor index"
        );
        assert!(
            engine
                .relational_catalog_table("product_existing_s3_index_prefix")
                .unwrap()
                .indexes
                .iter()
                .all(|index| {
                    index.name != "product_existing_s3_index_prefix_replacement_code_region"
                }),
            "{label} must not publish the private replacement index"
        );

        drop(engine);
        let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("recover the unacknowledged populated-table S3 transition");
        assert_eq!(recovered.durable_wal_records(), durable_before, "{label}");
        assert_eq!(
            durable_ids(&recovered, "product_existing_s3_index_prefix"),
            vec![1, 2],
            "{label} recovery must retain only the published prefix"
        );
        stage_transition(&recovered, txn_id);
        recovered
            .submit_transaction(txn_id, parsed("COMMIT"))
            .expect("same transaction ID retries the populated-table S3 transition once");
        assert_eq!(
            durable_ids(&recovered, "product_existing_s3_index_prefix"),
            vec![1, 2, 3, 4, 5],
            "{label} retry publishes every prefix and private row exactly once"
        );
        assert!(
            recovered
                .relational_catalog_table("product_existing_s3_index_prefix")
                .unwrap()
                .indexes
                .iter()
                .all(|index| index.name != "product_existing_s3_index_prefix_original_code_region"),
            "{label} retry retires the original S3 index"
        );
        assert!(
            recovered
                .relational_catalog_table("product_existing_s3_index_prefix")
                .unwrap()
                .indexes
                .iter()
                .any(|index| {
                    index.name == "product_existing_s3_index_prefix_replacement_code_region"
                }),
            "{label} retry publishes the S3 replacement index"
        );
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            recovered.relational_named_index_covered_rows("product_existing_s3_index_prefix"),
            Some(10),
            "{label} retry publishes complete S3-created named-index coverage"
        );
        let record = recovered
            .durable_wal_records()
            .into_iter()
            .find(|record| record.txn_id == txn_id)
            .expect("the acknowledged retry remains in canonical WAL");
        assert!(
            !codec5_replay_metadata(&record).initial_table_absent,
            "{label} retry must preserve the published-table predecessor"
        );

        drop(recovered);
        let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("fresh reopen must recover the acknowledged S3 transition");
        assert_eq!(
            durable_ids(&reopened, "product_existing_s3_index_prefix"),
            vec![1, 2, 3, 4, 5],
            "{label} fresh replay owns the complete populated-table rowset"
        );
        let reopened_table = reopened
            .relational_catalog_table("product_existing_s3_index_prefix")
            .expect("fresh replay retains the transitioned table catalog");
        assert!(
            reopened_table
                .indexes
                .iter()
                .all(|index| index.name != "product_existing_s3_index_prefix_original_code_region"),
            "{label} fresh replay must not resurrect the retired index"
        );
        assert!(
            reopened_table.indexes.iter().any(|index| {
                index.name == "product_existing_s3_index_prefix_replacement_code_region"
            }),
            "{label} fresh replay must retain the replacement index"
        );
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            reopened.relational_named_index_covered_rows("product_existing_s3_index_prefix"),
            Some(10),
            "{label} fresh replay owns complete S3-created named-index coverage"
        );
    }
}

#[test]
fn product_001_existing_table_s3_index_replacement_post_durable_recovery_owns_once() {
    let fixture = DurableFixture::new("existing-s3-index-post-durable");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    engine
        .execute_text(
            71_420,
            "CREATE TABLE product_existing_s3_index_indeterminate \
             (id INT PRIMARY KEY, code INT, region TEXT)",
        )
        .expect("create populated-table S3 indeterminate fixture");
    engine
        .execute_text(
            71_421,
            "INSERT INTO product_existing_s3_index_indeterminate VALUES \
             (1, NULL, 'north'), (2, 20, 'east')",
        )
        .expect("seed the published S3 indeterminate prefix");
    engine
        .execute_text(
            71_422,
            "CREATE UNIQUE INDEX product_existing_s3_index_indeterminate_original_code_region \
             ON product_existing_s3_index_indeterminate (code, region)",
        )
        .expect("create the published S3 indeterminate predecessor index");
    let records_before = engine.durable_wal_records().len();
    let txn_id = 71_423;
    for sql in [
        "BEGIN",
        "DROP INDEX product_existing_s3_index_indeterminate_original_code_region",
        "CREATE UNIQUE INDEX product_existing_s3_index_indeterminate_replacement_code_region \
         ON product_existing_s3_index_indeterminate (code, region)",
        "INSERT INTO product_existing_s3_index_indeterminate VALUES \
         (3, 30, 'west'), (4, 40, 'south'), (5, NULL, 'south')",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage populated-table S3 post-durable transition");
    }
    engine.fail_next_transaction_post_durable_apply();
    let failure = engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect_err("post-durable populated-table S3 fault must be indeterminate");
    assert!(
        failure.is_indeterminate(),
        "expected indeterminate failure: {failure}"
    );
    assert!(engine.is_commit_path_poisoned());
    assert_eq!(engine.durable_wal_records().len(), records_before + 1);

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("recover the indeterminate populated-table S3 transition");
    assert_eq!(recovered.durable_wal_records().len(), records_before + 1);
    assert_eq!(
        durable_ids(&recovered, "product_existing_s3_index_indeterminate"),
        vec![1, 2, 3, 4, 5]
    );
    assert!(recovered
        .relational_catalog_table("product_existing_s3_index_indeterminate")
        .unwrap()
        .indexes
        .iter()
        .all(|index| {
            index.name != "product_existing_s3_index_indeterminate_original_code_region"
        }));
    assert!(recovered
        .relational_catalog_table("product_existing_s3_index_indeterminate")
        .unwrap()
        .indexes
        .iter()
        .any(|index| {
            index.name == "product_existing_s3_index_indeterminate_replacement_code_region"
        }));
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        recovered.relational_named_index_covered_rows("product_existing_s3_index_indeterminate"),
        Some(10),
        "post-durable recovery must publish complete S3-created named-index coverage"
    );

    drop(recovered);
    let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("repeat recovery of the populated-table S3 transition is exact");
    assert_eq!(
        durable_ids(&reopened, "product_existing_s3_index_indeterminate"),
        vec![1, 2, 3, 4, 5]
    );
    let reopened_table = reopened
        .relational_catalog_table("product_existing_s3_index_indeterminate")
        .expect("repeat recovery retains the transitioned table catalog");
    assert!(
        reopened_table.indexes.iter().all(|index| {
            index.name != "product_existing_s3_index_indeterminate_original_code_region"
        }),
        "repeat recovery must not resurrect the retired index"
    );
    assert!(
        reopened_table.indexes.iter().any(|index| {
            index.name == "product_existing_s3_index_indeterminate_replacement_code_region"
        }),
        "repeat recovery must retain the replacement index"
    );
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        reopened.relational_named_index_covered_rows("product_existing_s3_index_indeterminate"),
        Some(10),
        "repeat recovery retains complete S3-created named-index coverage"
    );
}

#[test]
fn product_001_existing_table_s3_index_unique_failure_rolls_back_without_publication() {
    let fixture = DurableFixture::new("existing-s3-index-unique-rollback");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    engine
        .execute_text(
            71_430,
            "CREATE TABLE product_existing_s3_index_unique_failure \
             (id INT PRIMARY KEY, code INT, region TEXT)",
        )
        .expect("create populated-table S3 unique-failure fixture");
    engine
        .execute_text(
            71_431,
            "INSERT INTO product_existing_s3_index_unique_failure VALUES (1, 10, 'north')",
        )
        .expect("seed the published unique-failure prefix");
    let durable_before = engine.durable_wal_records();
    let visible_before = engine.visible_up_to();
    let next_oid_before = engine.catalog_snapshot().relational_next_oid;
    let txn_id = 71_432;
    for sql in [
        "BEGIN",
        "INSERT INTO product_existing_s3_index_unique_failure VALUES (2, 20, 'west')",
        "CREATE UNIQUE INDEX product_existing_s3_index_unique_failure_code_region \
         ON product_existing_s3_index_unique_failure (code, region)",
        "INSERT INTO product_existing_s3_index_unique_failure VALUES (3, 30, 'south')",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage populated-table S3 unique-failure transition");
    }
    let duplicate = engine
        .submit_transaction(
            txn_id,
            parsed("INSERT INTO product_existing_s3_index_unique_failure VALUES (4, 30, 'south')"),
        )
        .expect_err("private S3-created unique index must reject the duplicate before WAL");
    assert!(
        duplicate.to_string().contains("duplicate key"),
        "S3-created index must retain its PostgreSQL diagnostic: {duplicate}"
    );
    engine
        .submit_transaction(txn_id, parsed("ROLLBACK"))
        .expect("rollback after populated-table S3 unique rejection");
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(
        engine.catalog_snapshot().relational_next_oid,
        next_oid_before
    );
    assert_eq!(
        durable_ids(&engine, "product_existing_s3_index_unique_failure"),
        vec![1]
    );
    assert!(engine
        .relational_catalog_table("product_existing_s3_index_unique_failure")
        .unwrap()
        .indexes
        .iter()
        .all(|index| index.name != "product_existing_s3_index_unique_failure_code_region"));

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("fresh reopen must retain only the published unique-failure prefix");
    assert_eq!(recovered.durable_wal_records(), durable_before);
    assert_eq!(
        durable_ids(&recovered, "product_existing_s3_index_unique_failure"),
        vec![1]
    );
    assert!(recovered
        .relational_catalog_table("product_existing_s3_index_unique_failure")
        .unwrap()
        .indexes
        .iter()
        .all(|index| index.name != "product_existing_s3_index_unique_failure_code_region"));
}

#[test]
fn product_001_private_create_index_unique_failure_rolls_back_without_publication() {
    let fixture = DurableFixture::new("private-create-index-unique-rollback");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_360;
    let durable_before = engine.durable_wal_records();
    let visible_before = engine.visible_up_to();
    let next_oid_before = engine.catalog_snapshot().relational_next_oid;

    for sql in [
        "BEGIN",
        "CREATE TABLE product_private_index_unique_failure (id INT, value INT)",
        "CREATE UNIQUE INDEX product_private_index_unique_failure_id \
         ON product_private_index_unique_failure (id)",
        "INSERT INTO product_private_index_unique_failure VALUES (1, 10)",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private CREATE INDEX unique-failure transaction");
    }
    let duplicate = engine
        .submit_transaction(
            txn_id,
            parsed("INSERT INTO product_private_index_unique_failure VALUES (1, 20)"),
        )
        .expect_err(
            "private named unique index must reject the duplicate before codec-5 admission",
        );
    assert!(
        duplicate.to_string().contains("duplicate key"),
        "private named unique index must retain its PostgreSQL diagnostic: {duplicate}"
    );
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);

    engine
        .submit_transaction(txn_id, parsed("ROLLBACK"))
        .expect("rollback after private named-index duplicate rejection");
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(
        engine.catalog_snapshot().relational_next_oid,
        next_oid_before,
        "private named-index rollback must not leak catalog allocation"
    );
    assert!(
        engine
            .relational_catalog_table("product_private_index_unique_failure")
            .is_none(),
        "rollback must not publish the private table or named index"
    );

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("private named-index rollback leaves an empty durable prefix");
    assert_eq!(recovered.durable_wal_records(), durable_before);
    assert_eq!(recovered.visible_up_to(), visible_before);
    assert_eq!(
        recovered.catalog_snapshot().relational_next_oid,
        next_oid_before
    );
    assert!(
        recovered
            .relational_catalog_table("product_private_index_unique_failure")
            .is_none(),
        "fresh reopen must not materialize the rolled-back private table or named index"
    );
}

#[test]
fn product_001_codec5_private_create_sequence_table_rename_crash_prefixes_retry_once() {
    use crate::engine_transaction_delta::TransactionTerminalPreDurableFault;

    let boundaries = [
        (
            "before-replication-proposal",
            TransactionTerminalPreDurableFault::ReplicationProposal,
        ),
        (
            "before-canonical-wal-append",
            TransactionTerminalPreDurableFault::CanonicalWalAppend,
        ),
        (
            "before-wal-flush",
            TransactionTerminalPreDurableFault::WalFlush,
        ),
    ];

    let stage_private_create = |engine: &Engine, txn_id| {
        for sql in [
            "BEGIN",
            "CREATE SEQUENCE product_private_prefix_sequence",
            "CREATE TABLE product_private_prefix_owner \
             (id INT DEFAULT nextval('product_private_prefix_sequence'::regclass), \
              row_key INT PRIMARY KEY, note TEXT)",
            "INSERT INTO product_private_prefix_owner (row_key, note) VALUES (1, 'before-rename')",
            "ALTER SEQUENCE product_private_prefix_sequence \
             RENAME TO product_private_prefix_sequence_final",
            "INSERT INTO product_private_prefix_owner (row_key, note) VALUES (2, 'after-rename')",
            "ALTER SEQUENCE product_private_prefix_sequence_final RESTART WITH 40",
            "INSERT INTO product_private_prefix_owner (row_key, note) VALUES (3, 'after-restart')",
        ] {
            engine
                .submit_transaction(txn_id, parsed(sql))
                .expect("stage private sequence/table/rename transaction");
        }
    };

    for (ordinal, (label, boundary)) in boundaries.into_iter().enumerate() {
        let fixture = DurableFixture::new(&format!("private-create-{label}"));
        let engine = Engine::with_durable_wal_segment(&fixture.wal);
        let txn_id = 71_200 + u64::try_from(ordinal).unwrap();
        let durable_before = engine.durable_wal_records();
        let visible_before = engine.visible_up_to();
        let flushed_before = engine.wal_flushed_count();

        stage_private_create(&engine, txn_id);
        engine.fail_next_transaction_terminal_pre_durable_at(boundary);
        let failure = engine.submit_transaction(txn_id, parsed("COMMIT"));
        assert!(
            matches!(
                failure,
                Err(ExecuteError::Engine(EngineError::Durability(_)))
            ),
            "{label} must fail before acknowledgement: {failure:?}"
        );
        assert_eq!(engine.visible_up_to(), visible_before, "{label}");
        assert_eq!(engine.wal_flushed_count(), flushed_before, "{label}");
        assert_eq!(engine.durable_wal_records(), durable_before, "{label}");
        assert!(
            engine
                .relational_catalog_table("product_private_prefix_owner")
                .is_none(),
            "{label} must not publish the private table"
        );
        assert!(
            engine
                .relational_catalog_sequence("product_private_prefix_sequence_final")
                .is_none(),
            "{label} must not publish the renamed private sequence"
        );

        // Dropping here models a process crash. The prefix may contain neither the first table
        // generation nor the final stable sequence name, because both belong to this one codec-5
        // terminal record.
        drop(engine);
        let recovered =
            Engine::open_durable_wal_segment_auto(&fixture.wal).expect("recover crash prefix");
        assert_eq!(recovered.durable_wal_records(), durable_before, "{label}");
        assert!(
            recovered
                .relational_catalog_table("product_private_prefix_owner")
                .is_none(),
            "{label} fresh reopen must not materialize the private table"
        );
        assert!(
            recovered
                .relational_catalog_sequence("product_private_prefix_sequence_final")
                .is_none(),
            "{label} fresh reopen must not materialize the renamed private sequence"
        );

        stage_private_create(&recovered, txn_id);
        recovered
            .submit_transaction(txn_id, parsed("COMMIT"))
            .expect("the unacknowledged private transaction retries through codec-5");
        assert_eq!(
            durable_ids(&recovered, "product_private_prefix_owner"),
            vec![1, 2, 40],
            "{label} retry publishes every pre-rename, post-rename, and post-restart row exactly once"
        );
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            recovered.relational_named_index_covered_rows("product_private_prefix_owner"),
            Some(3),
            "{label} retry must publish the transaction-created primary index with every row"
        );
        let sequence = recovered
            .relational_catalog_sequence("product_private_prefix_sequence_final")
            .expect("retry publishes final sequence binding");
        assert_eq!(
            (sequence.last_value, sequence.is_called),
            (40, true),
            "{label}"
        );
        let retry_record = recovered
            .durable_wal_records()
            .into_iter()
            .find(|record| record.txn_id == txn_id)
            .expect("the acknowledged retry remains in canonical WAL");
        let metadata = codec5_replay_metadata(&retry_record);
        assert!(
            metadata.initial_table_absent,
            "{label} retry must retain the S7-authenticated first-table predecessor"
        );

        drop(recovered);
        let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("recover acknowledged private retry");
        assert_eq!(
            durable_ids(&reopened, "product_private_prefix_owner"),
            vec![1, 2, 40],
            "{label} replay owns the same first-table generation"
        );
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            reopened.relational_named_index_covered_rows("product_private_prefix_owner"),
            Some(3),
            "{label} fresh replay must republish the transaction-created primary index"
        );
        let sequence = reopened
            .relational_catalog_sequence("product_private_prefix_sequence_final")
            .expect("replay restores final sequence binding");
        assert_eq!(
            (sequence.last_value, sequence.is_called),
            (40, true),
            "{label}"
        );
    }
}

#[cfg(feature = "test-support")]
#[test]
fn product_001_codec5_private_create_replay_parks_persistent_719_then_uses_fresh_context() {
    let Ok(runtime) = gpu_db_execution::CudaDriverRuntime::probe() else {
        return;
    };
    if !runtime.snapshot().driver_available || runtime.snapshot().device_count == 0 {
        return;
    }

    let fixture = DurableFixture::new("private-create-persistent-719");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_245;
    for sql in [
        "BEGIN",
        "CREATE SEQUENCE product_private_719_sequence",
        "CREATE TABLE product_private_719_owner \
         (id INT DEFAULT nextval('product_private_719_sequence'::regclass), \
          row_key INT PRIMARY KEY, note TEXT)",
        "INSERT INTO product_private_719_owner (row_key, note) VALUES (1, 'before-rename')",
        "ALTER SEQUENCE product_private_719_sequence \
         RENAME TO product_private_719_sequence_final",
        "INSERT INTO product_private_719_owner (row_key, note) VALUES (2, 'after-rename')",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage one codec-5 private CREATE/INSERT authority");
    }
    engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect("durably commit the private CREATE/INSERT authority");
    let durable = engine.durable_wal_records();
    drop(engine);

    // The first completion failure returns UnknownQuiescence; dropping that owner consumes the
    // second injected failure during its bounded drain and quarantines the old-context leases.
    // Recovery must then replay the immutable codec-5 record exactly once on a new context.
    gpu_db_execution::fail_owned_stream_syncs_with_cuda_code_for_test(2, 719);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("persistent typed-generation 719 must use one fresh-context replay retry");
    assert_eq!(Engine::recovery_attempt_count(), 2);
    assert!(recovered.uses_dedicated_recovery_cuda_contexts_for_test());
    assert_eq!(recovered.durable_wal_records(), durable);
    assert_eq!(
        durable_ids(&recovered, "product_private_719_owner"),
        vec![1, 2]
    );
    let sequence = recovered
        .relational_catalog_sequence("product_private_719_sequence_final")
        .expect("fresh replay retains the renamed private sequence");
    assert_eq!((sequence.last_value, sequence.is_called), (2, true));
}

#[test]
fn product_001_codec5_private_serial_table_rename_crash_prefixes_retry_once() {
    use crate::engine_transaction_delta::TransactionTerminalPreDurableFault;

    let boundaries = [
        (
            "before-replication-proposal",
            TransactionTerminalPreDurableFault::ReplicationProposal,
        ),
        (
            "before-canonical-wal-append",
            TransactionTerminalPreDurableFault::CanonicalWalAppend,
        ),
        (
            "before-wal-flush",
            TransactionTerminalPreDurableFault::WalFlush,
        ),
    ];
    let stage_private_serial = |engine: &Engine, txn_id| {
        for sql in [
            "BEGIN",
            "CREATE TABLE product_private_serial_prefix_owner \
             (id serial PRIMARY KEY, note text)",
            "ALTER SEQUENCE product_private_serial_prefix_owner_id_seq \
             RENAME TO product_private_serial_prefix_owner_id_seq_final",
            "INSERT INTO product_private_serial_prefix_owner (note) \
             VALUES ('first'), ('second')",
        ] {
            engine
                .submit_transaction(txn_id, parsed(sql))
                .expect("stage private serial/rename/typed-INSERT transaction");
        }
    };

    for (ordinal, (label, boundary)) in boundaries.into_iter().enumerate() {
        let fixture = DurableFixture::new(&format!("private-serial-{label}"));
        let engine = Engine::with_durable_wal_segment(&fixture.wal);
        let txn_id = 71_203 + u64::try_from(ordinal).unwrap();
        let durable_before = engine.durable_wal_records();
        let visible_before = engine.visible_up_to();
        let next_oid_before = engine.catalog_snapshot().relational_next_oid;

        stage_private_serial(&engine, txn_id);
        engine.fail_next_transaction_terminal_pre_durable_at(boundary);
        let failure = engine.submit_transaction(txn_id, parsed("COMMIT"));
        assert!(
            matches!(
                failure,
                Err(ExecuteError::Engine(EngineError::Durability(_)))
            ),
            "{label} must fail before acknowledgement: {failure:?}"
        );
        assert_eq!(engine.durable_wal_records(), durable_before, "{label}");
        assert_eq!(engine.visible_up_to(), visible_before, "{label}");
        assert_eq!(
            engine.catalog_snapshot().relational_next_oid,
            next_oid_before,
            "{label} must not publish the generated table/sequence OIDs"
        );
        assert!(
            engine
                .relational_catalog_table("product_private_serial_prefix_owner")
                .is_none(),
            "{label} must not publish the private serial table"
        );
        assert!(
            engine
                .relational_catalog_sequence("product_private_serial_prefix_owner_id_seq_final")
                .is_none(),
            "{label} must not publish the renamed generated sequence"
        );

        drop(engine);
        let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("recover the unacknowledged private serial prefix");
        assert_eq!(recovered.durable_wal_records(), durable_before, "{label}");
        assert_eq!(
            recovered.catalog_snapshot().relational_next_oid,
            next_oid_before,
            "{label} fresh reopen must not materialize generated catalog identity"
        );
        assert!(
            recovered
                .relational_catalog_table("product_private_serial_prefix_owner")
                .is_none(),
            "{label} fresh reopen must not materialize the private serial table"
        );

        stage_private_serial(&recovered, txn_id);
        recovered
            .submit_transaction(txn_id, parsed("COMMIT"))
            .expect("the unacknowledged private serial transaction retries through codec-5");
        assert_eq!(
            durable_ids(&recovered, "product_private_serial_prefix_owner"),
            vec![1, 2],
            "{label} retry publishes every generated serial row exactly once"
        );
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            recovered.relational_named_index_covered_rows("product_private_serial_prefix_owner"),
            Some(2),
            "{label} retry publishes the generated primary index exactly once"
        );
        let sequence = recovered
            .relational_catalog_sequence("product_private_serial_prefix_owner_id_seq_final")
            .expect("retry publishes the renamed generated sequence");
        assert_eq!(
            (sequence.last_value, sequence.is_called),
            (2, true),
            "{label}"
        );
        assert!(
            recovered
                .relational_catalog_sequence("product_private_serial_prefix_owner_id_seq")
                .is_none(),
            "{label} retry must retain only the final generated sequence name"
        );
        let retry_record = recovered
            .durable_wal_records()
            .into_iter()
            .find(|record| record.txn_id == txn_id)
            .expect("the acknowledged private serial retry remains in canonical WAL");
        assert!(codec5_replay_metadata(&retry_record).initial_table_absent);

        drop(recovered);
        let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("fresh reopen recovers the acknowledged private serial retry");
        assert_eq!(
            durable_ids(&reopened, "product_private_serial_prefix_owner"),
            vec![1, 2],
            "{label} fresh replay owns the private serial rowset"
        );
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            reopened.relational_named_index_covered_rows("product_private_serial_prefix_owner"),
            Some(2),
            "{label} fresh replay owns the generated primary index"
        );
        let sequence = reopened
            .relational_catalog_sequence("product_private_serial_prefix_owner_id_seq_final")
            .expect("fresh replay restores the final generated sequence name");
        assert_eq!(
            (sequence.last_value, sequence.is_called),
            (2, true),
            "{label}"
        );
    }
}

#[test]
fn product_001_private_serial_table_rename_post_durable_recovery_owns_once() {
    let fixture = DurableFixture::new("private-serial-post-durable");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_206;
    let records_before = engine.durable_wal_records().len();
    for sql in [
        "BEGIN",
        "CREATE TABLE product_private_serial_indeterminate_owner \
         (id serial PRIMARY KEY, note text)",
        "ALTER SEQUENCE product_private_serial_indeterminate_owner_id_seq \
         RENAME TO product_private_serial_indeterminate_owner_id_seq_final",
        "INSERT INTO product_private_serial_indeterminate_owner (note) \
         VALUES ('first'), ('second')",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private serial/rename/typed-INSERT post-durable transaction");
    }
    engine.fail_next_transaction_post_durable_apply();
    let failure = engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect_err("post-durable private serial apply fault must be indeterminate");
    assert!(
        failure.is_indeterminate(),
        "expected indeterminate private serial failure: {failure}"
    );
    assert!(engine.is_commit_path_poisoned());
    assert_eq!(engine.durable_wal_records().len(), records_before + 1);

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("recover the durable private serial transaction");
    assert_eq!(recovered.durable_wal_records().len(), records_before + 1);
    assert_eq!(
        durable_ids(&recovered, "product_private_serial_indeterminate_owner"),
        vec![1, 2]
    );
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        recovered.relational_named_index_covered_rows("product_private_serial_indeterminate_owner"),
        Some(2),
        "post-durable recovery must publish the generated primary index"
    );
    let sequence = recovered
        .relational_catalog_sequence("product_private_serial_indeterminate_owner_id_seq_final")
        .expect("recovery publishes the renamed generated sequence");
    assert_eq!((sequence.last_value, sequence.is_called), (2, true));
    assert!(
        recovered
            .relational_catalog_sequence("product_private_serial_indeterminate_owner_id_seq")
            .is_none(),
        "recovery must not restore the pre-rename generated sequence name"
    );
    let record = recovered
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == txn_id)
        .expect("durable private serial transaction remains present");
    assert!(codec5_replay_metadata(&record).initial_table_absent);

    let records_before_repeat_reopen = recovered.durable_wal_records().len();
    drop(recovered);
    let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("repeat private serial recovery is exact");
    assert_eq!(
        reopened.durable_wal_records().len(),
        records_before_repeat_reopen,
        "repeat recovery must not append or duplicate the private serial transaction"
    );
    assert_eq!(
        durable_ids(&reopened, "product_private_serial_indeterminate_owner"),
        vec![1, 2]
    );
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        reopened.relational_named_index_covered_rows("product_private_serial_indeterminate_owner"),
        Some(2),
        "repeat recovery must retain the generated primary index"
    );
    let sequence = reopened
        .relational_catalog_sequence("product_private_serial_indeterminate_owner_id_seq_final")
        .expect("repeat recovery retains the renamed generated sequence");
    assert_eq!((sequence.last_value, sequence.is_called), (2, true));
}

#[test]
fn product_001_private_serial_table_rename_duplicate_rolls_back_without_publication() {
    let fixture = DurableFixture::new("private-serial-duplicate-rollback");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_207;
    let durable_before = engine.durable_wal_records();
    let visible_before = engine.visible_up_to();
    let next_oid_before = engine.catalog_snapshot().relational_next_oid;
    for sql in [
        "BEGIN",
        "CREATE TABLE product_private_serial_duplicate_owner \
         (id serial PRIMARY KEY, note text)",
        "ALTER SEQUENCE product_private_serial_duplicate_owner_id_seq \
         RENAME TO product_private_serial_duplicate_owner_id_seq_final",
        "INSERT INTO product_private_serial_duplicate_owner (note) VALUES ('first')",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private serial/rename/valid typed INSERT before duplicate");
    }
    let duplicate = engine
        .submit_transaction(
            txn_id,
            parsed("INSERT INTO product_private_serial_duplicate_owner (id, note) VALUES (1, 'duplicate')"),
        )
        .expect_err("private serial primary key must reject the duplicate before codec-5 admission");
    assert!(
        duplicate
            .to_string()
            .to_ascii_lowercase()
            .contains("duplicate key"),
        "private serial duplicate must retain its PostgreSQL diagnostic: {duplicate}"
    );
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);

    engine
        .submit_transaction(txn_id, parsed("ROLLBACK"))
        .expect("rollback after private serial duplicate rejection");
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(
        engine.catalog_snapshot().relational_next_oid,
        next_oid_before,
        "private serial rollback must not leak generated table or sequence OIDs"
    );
    assert!(engine
        .relational_catalog_table("product_private_serial_duplicate_owner")
        .is_none());
    assert!(engine
        .relational_catalog_sequence("product_private_serial_duplicate_owner_id_seq")
        .is_none());
    assert!(engine
        .relational_catalog_sequence("product_private_serial_duplicate_owner_id_seq_final")
        .is_none());

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("private serial rollback leaves no durable prefix");
    assert_eq!(recovered.durable_wal_records(), durable_before);
    assert_eq!(recovered.visible_up_to(), visible_before);
    assert_eq!(
        recovered.catalog_snapshot().relational_next_oid,
        next_oid_before,
        "fresh reopen must not consume private serial catalog identity"
    );
    assert!(recovered
        .relational_catalog_table("product_private_serial_duplicate_owner")
        .is_none());
    assert!(recovered
        .relational_catalog_sequence("product_private_serial_duplicate_owner_id_seq_final")
        .is_none());
}

#[test]
fn product_001_private_create_sequence_table_rename_rollback_publishes_nothing() {
    let fixture = DurableFixture::new("private-create-rollback");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_300;
    let durable_before = engine.durable_wal_records();
    let visible_before = engine.visible_up_to();
    let next_oid_before = engine.catalog_snapshot().relational_next_oid;

    for sql in [
        "BEGIN",
        "CREATE SEQUENCE product_private_rollback_sequence",
        "CREATE TABLE product_private_rollback_owner \
         (id INT DEFAULT nextval('product_private_rollback_sequence'::regclass), \
          row_key INT PRIMARY KEY, note TEXT)",
        "INSERT INTO product_private_rollback_owner (row_key, note) VALUES (1, 'before-rename')",
        "ALTER SEQUENCE product_private_rollback_sequence \
         RENAME TO product_private_rollback_sequence_final",
        "INSERT INTO product_private_rollback_owner (row_key, note) VALUES (2, 'after-rename')",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private rollback transaction");
    }
    engine
        .submit_transaction(txn_id, parsed("ROLLBACK"))
        .expect("rollback private first-table transaction");
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(
        engine.catalog_snapshot().relational_next_oid,
        next_oid_before
    );
    assert!(engine
        .relational_catalog_table("product_private_rollback_owner")
        .is_none());
    assert!(engine
        .relational_catalog_sequence("product_private_rollback_sequence")
        .is_none());
    assert!(engine
        .relational_catalog_sequence("product_private_rollback_sequence_final")
        .is_none());

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("rollback leaves an empty durable prefix");
    assert_eq!(recovered.durable_wal_records(), durable_before);
    assert_eq!(
        recovered.catalog_snapshot().relational_next_oid,
        next_oid_before
    );
    assert!(recovered
        .relational_catalog_table("product_private_rollback_owner")
        .is_none());
    assert!(recovered
        .relational_catalog_sequence("product_private_rollback_sequence_final")
        .is_none());
}

#[test]
fn product_001_private_create_sequence_table_rename_check_failure_rolls_back_without_publication() {
    let fixture = DurableFixture::new("private-create-check-rollback");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    engine
        .execute_text(
            71_348,
            "CREATE TABLE product_private_check_parent (id INT PRIMARY KEY)",
        )
        .expect("create the published CHECK/FK parent");
    engine
        .execute_text(
            71_349,
            "INSERT INTO product_private_check_parent VALUES (7)",
        )
        .expect("seed the published CHECK/FK parent");
    engine
        .execute_text(71_350, "CREATE DOMAIN product_private_check_tally AS INT")
        .expect("create the published domain used by the private CHECK table");
    let txn_id = 71_351;
    let durable_before = engine.durable_wal_records();
    let visible_before = engine.visible_up_to();
    let next_oid_before = engine.catalog_snapshot().relational_next_oid;

    for sql in [
        "BEGIN",
        "CREATE SEQUENCE product_private_check_sequence",
        "CREATE TABLE product_private_check_owner \
         (id INT DEFAULT nextval('product_private_check_sequence'::regclass), parent_id INT, \
          row_key INT PRIMARY KEY, tally product_private_check_tally, \
          CONSTRAINT product_private_check_positive CHECK (tally > 0))",
        "ALTER TABLE ONLY product_private_check_owner \
         ADD CONSTRAINT product_private_check_owner_parent \
         FOREIGN KEY (parent_id) REFERENCES product_private_check_parent(id)",
        "INSERT INTO product_private_check_owner (row_key, parent_id, tally) VALUES (1, 7, 1)",
        "ALTER SEQUENCE product_private_check_sequence \
         RENAME TO product_private_check_sequence_final",
        "ALTER SEQUENCE product_private_check_sequence_final RESTART WITH 40",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private check transaction");
    }
    let check = engine
        .submit_transaction(
            txn_id,
            parsed(
                "INSERT INTO product_private_check_owner (row_key, parent_id, tally) VALUES (2, 7, 0)",
            ),
        )
        .expect_err("private first-table CHECK must reject before codec-5 admission");
    assert!(
        check.to_string().contains("check constraint"),
        "private CHECK must retain its PostgreSQL diagnostic: {check}"
    );
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);

    engine
        .submit_transaction(txn_id, parsed("ROLLBACK"))
        .expect("rollback after private CHECK rejection");
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(
        engine.catalog_snapshot().relational_next_oid,
        next_oid_before
    );
    assert!(engine
        .relational_catalog_table("product_private_check_owner")
        .is_none());
    assert!(engine
        .relational_catalog_sequence("product_private_check_sequence")
        .is_none());
    assert!(engine
        .relational_catalog_sequence("product_private_check_sequence_final")
        .is_none());

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("private CHECK rollback leaves an empty durable prefix");
    assert_eq!(recovered.durable_wal_records(), durable_before);
    assert_eq!(recovered.visible_up_to(), visible_before);
    assert_eq!(
        recovered.catalog_snapshot().relational_next_oid,
        next_oid_before
    );
    assert!(recovered
        .relational_catalog_table("product_private_check_owner")
        .is_none());
    assert!(recovered
        .relational_catalog_sequence("product_private_check_sequence_final")
        .is_none());
}

#[test]
fn product_001_private_create_sequence_table_foreign_key_rename_reopens_through_codec5() {
    let fixture = DurableFixture::new("private-create-foreign-key");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    engine
        .execute_text(71_359, "CREATE DOMAIN product_private_fk_tally AS INT")
        .expect("create the domain used by the private FK table");

    let txn_id = 71_362;
    for sql in [
        "BEGIN",
        "CREATE SEQUENCE product_private_fk_sequence",
        "CREATE TABLE product_private_fk_parent (id INT PRIMARY KEY)",
        "CREATE TABLE product_private_fk_owner \
         (id INT DEFAULT nextval('product_private_fk_sequence'::regclass), parent_id INT, \
          row_key INT PRIMARY KEY, tally product_private_fk_tally, \
          CONSTRAINT product_private_fk_tally_positive CHECK (tally > 0))",
        "ALTER TABLE ONLY product_private_fk_owner ADD CONSTRAINT product_private_fk_owner_parent \
         FOREIGN KEY (parent_id) REFERENCES product_private_fk_parent(id)",
        "INSERT INTO product_private_fk_parent VALUES (7)",
        "INSERT INTO product_private_fk_owner (row_key, parent_id, tally) VALUES (1, 7, 11)",
        "ALTER SEQUENCE product_private_fk_sequence \
         RENAME TO product_private_fk_sequence_final",
        "INSERT INTO product_private_fk_owner (row_key, parent_id, tally) VALUES (2, 7, 22)",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private sequence/table/FK/rename transaction");
    }
    engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect("private FK transaction must retain codec-5 S3 composition");
    assert_eq!(durable_ids(&engine, "product_private_fk_owner"), vec![1, 2]);
    assert_eq!(durable_ids(&engine, "product_private_fk_parent"), vec![7]);
    let sequence = engine
        .relational_catalog_sequence("product_private_fk_sequence_final")
        .expect("private FK transaction publishes the renamed sequence");
    assert_eq!((sequence.last_value, sequence.is_called), (2, true));
    let record = engine
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == txn_id)
        .expect("private FK transaction emits one canonical record");
    let metadata = codec5_replay_metadata(&record);
    assert!(metadata.initial_table_absent);

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("reopen private FK transaction from codec-5 WAL");
    assert_eq!(
        durable_ids(&recovered, "product_private_fk_owner"),
        vec![1, 2]
    );
    assert_eq!(
        durable_ids(&recovered, "product_private_fk_parent"),
        vec![7]
    );
    let sequence = recovered
        .relational_catalog_sequence("product_private_fk_sequence_final")
        .expect("reopen retains renamed private sequence");
    assert_eq!((sequence.last_value, sequence.is_called), (2, true));
}

#[test]
fn product_001_private_create_domain_table_insert_reopens_through_codec5() {
    let fixture = DurableFixture::new("private-create-domain");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_364;
    for sql in [
        "BEGIN",
        "CREATE DOMAIN product_private_inline_amount AS INT",
        "CREATE TABLE product_private_inline_domain_owner \
         (id INT PRIMARY KEY, amount product_private_inline_amount, \
          CONSTRAINT product_private_inline_amount_positive CHECK (amount > 0))",
        "INSERT INTO product_private_inline_domain_owner VALUES (1, 11), (2, 22)",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private domain/table/INSERT transaction");
    }
    engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect("private domain/table/INSERT transaction must retain codec-5 S3 composition");
    assert_eq!(
        durable_ids(&engine, "product_private_inline_domain_owner"),
        vec![1, 2]
    );
    assert!(engine
        .relational_catalog_domain("product_private_inline_amount")
        .is_some());
    assert_eq!(
        engine.relational_named_index_covered_rows("product_private_inline_domain_owner"),
        Some(2)
    );
    let record = engine
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == txn_id)
        .expect("private domain transaction emits one canonical record");
    let metadata = codec5_replay_metadata(&record);
    assert!(metadata.initial_table_absent);

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("reopen private domain transaction from codec-5 WAL");
    assert_eq!(
        durable_ids(&recovered, "product_private_inline_domain_owner"),
        vec![1, 2]
    );
    assert!(recovered
        .relational_catalog_domain("product_private_inline_amount")
        .is_some());
    assert_eq!(
        recovered.relational_named_index_covered_rows("product_private_inline_domain_owner"),
        Some(2)
    );
}

#[test]
fn product_001_private_create_domain_table_check_failure_rolls_back_without_publication() {
    let fixture = DurableFixture::new("private-create-domain-check-rollback");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_365;
    let durable_before = engine.durable_wal_records();
    let visible_before = engine.visible_up_to();
    let next_oid_before = engine.catalog_snapshot().relational_next_oid;
    for sql in [
        "BEGIN",
        "CREATE DOMAIN product_private_inline_failure_amount AS INT",
        "CREATE TABLE product_private_inline_failure_owner \
         (id INT PRIMARY KEY, amount product_private_inline_failure_amount, \
          CONSTRAINT product_private_inline_failure_amount_positive CHECK (amount > 0))",
        "INSERT INTO product_private_inline_failure_owner VALUES (1, 11)",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private domain/table/valid INSERT before CHECK failure");
    }
    let check = engine
        .submit_transaction(
            txn_id,
            parsed("INSERT INTO product_private_inline_failure_owner VALUES (2, -1)"),
        )
        .expect_err("private domain CHECK must reject the bad typed row at statement admission");
    assert!(
        check.to_string().to_ascii_lowercase().contains("check"),
        "private domain CHECK must retain its PostgreSQL diagnostic: {check}"
    );
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);

    engine
        .submit_transaction(txn_id, parsed("ROLLBACK"))
        .expect("rollback after private domain CHECK rejection");
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(
        engine.catalog_snapshot().relational_next_oid,
        next_oid_before,
        "failed private domain composition must not consume a catalog OID"
    );
    assert!(engine
        .relational_catalog_domain("product_private_inline_failure_amount")
        .is_none());
    assert!(engine
        .relational_catalog_table("product_private_inline_failure_owner")
        .is_none());

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("fresh reopen must retain no failed private-domain transaction prefix");
    assert_eq!(recovered.durable_wal_records(), durable_before);
    assert_eq!(recovered.visible_up_to(), visible_before);
    assert_eq!(
        recovered.catalog_snapshot().relational_next_oid,
        next_oid_before
    );
    assert!(recovered
        .relational_catalog_domain("product_private_inline_failure_amount")
        .is_none());
    assert!(recovered
        .relational_catalog_table("product_private_inline_failure_owner")
        .is_none());
}

#[test]
fn product_001_private_create_domain_table_crash_prefixes_retry_once() {
    use crate::engine_transaction_delta::TransactionTerminalPreDurableFault;

    let boundaries = [
        (
            "before-replication-proposal",
            TransactionTerminalPreDurableFault::ReplicationProposal,
        ),
        (
            "before-canonical-wal-append",
            TransactionTerminalPreDurableFault::CanonicalWalAppend,
        ),
        (
            "before-wal-flush",
            TransactionTerminalPreDurableFault::WalFlush,
        ),
    ];
    let stage_private_domain = |engine: &Engine, txn_id| {
        for sql in [
            "BEGIN",
            "CREATE DOMAIN product_private_inline_prefix_amount AS INT",
            "CREATE TABLE product_private_inline_prefix_owner \
             (id INT PRIMARY KEY, amount product_private_inline_prefix_amount, \
              CONSTRAINT product_private_inline_prefix_amount_positive CHECK (amount > 0))",
            "INSERT INTO product_private_inline_prefix_owner VALUES (1, 11), (2, 22)",
        ] {
            engine
                .submit_transaction(txn_id, parsed(sql))
                .expect("stage private domain/table/INSERT crash-prefix transaction");
        }
    };

    for (ordinal, (label, boundary)) in boundaries.into_iter().enumerate() {
        let fixture = DurableFixture::new(&format!("private-create-domain-{label}"));
        let engine = Engine::with_durable_wal_segment(&fixture.wal);
        let txn_id = 71_366 + u64::try_from(ordinal).unwrap();
        let durable_before = engine.durable_wal_records();
        let visible_before = engine.visible_up_to();
        let next_oid_before = engine.catalog_snapshot().relational_next_oid;
        stage_private_domain(&engine, txn_id);
        engine.fail_next_transaction_terminal_pre_durable_at(boundary);
        let failure = engine.submit_transaction(txn_id, parsed("COMMIT"));
        assert!(
            matches!(
                failure,
                Err(ExecuteError::Engine(EngineError::Durability(_)))
            ),
            "{label} must fail before acknowledgement: {failure:?}"
        );
        assert_eq!(engine.durable_wal_records(), durable_before, "{label}");
        assert_eq!(engine.visible_up_to(), visible_before, "{label}");
        assert_eq!(
            engine.catalog_snapshot().relational_next_oid,
            next_oid_before,
            "{label} must not publish a private-domain catalog allocation"
        );
        assert!(engine
            .relational_catalog_domain("product_private_inline_prefix_amount")
            .is_none());
        assert!(engine
            .relational_catalog_table("product_private_inline_prefix_owner")
            .is_none());

        drop(engine);
        let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("fresh reopen must retain no pre-durable private-domain prefix");
        assert_eq!(recovered.durable_wal_records(), durable_before, "{label}");
        assert_eq!(recovered.visible_up_to(), visible_before, "{label}");
        assert_eq!(
            recovered.catalog_snapshot().relational_next_oid,
            next_oid_before,
            "{label} fresh reopen must not consume a private-domain catalog allocation"
        );
        assert!(recovered
            .relational_catalog_domain("product_private_inline_prefix_amount")
            .is_none());
        assert!(recovered
            .relational_catalog_table("product_private_inline_prefix_owner")
            .is_none());

        stage_private_domain(&recovered, txn_id);
        recovered
            .submit_transaction(txn_id, parsed("COMMIT"))
            .expect("the unacknowledged private-domain transaction retries through codec-5");
        assert_eq!(
            durable_ids(&recovered, "product_private_inline_prefix_owner"),
            vec![1, 2],
            "{label} retry publishes the private-domain rowset exactly once"
        );
        assert!(recovered
            .relational_catalog_domain("product_private_inline_prefix_amount")
            .is_some());
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            recovered.relational_named_index_covered_rows("product_private_inline_prefix_owner"),
            Some(2),
            "{label} retry publishes the private-domain primary index exactly once"
        );
        let record = recovered
            .durable_wal_records()
            .into_iter()
            .find(|record| record.txn_id == txn_id)
            .expect("the acknowledged private-domain retry remains in canonical WAL");
        assert!(codec5_replay_metadata(&record).initial_table_absent);

        drop(recovered);
        let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("fresh reopen must recover the acknowledged private-domain retry");
        assert_eq!(
            durable_ids(&reopened, "product_private_inline_prefix_owner"),
            vec![1, 2],
            "{label} fresh replay owns the retried private-domain rowset"
        );
        assert!(reopened
            .relational_catalog_domain("product_private_inline_prefix_amount")
            .is_some());
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            reopened.relational_named_index_covered_rows("product_private_inline_prefix_owner"),
            Some(2),
            "{label} fresh replay restores the private-domain primary index"
        );
    }
}

#[test]
fn product_001_private_create_domain_table_post_durable_recovery_owns_once() {
    let fixture = DurableFixture::new("private-create-domain-post-durable");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 71_371;
    let records_before = engine.durable_wal_records().len();
    for sql in [
        "BEGIN",
        "CREATE DOMAIN product_private_inline_indeterminate_amount AS INT",
        "CREATE TABLE product_private_inline_indeterminate_owner \
         (id INT PRIMARY KEY, amount product_private_inline_indeterminate_amount, \
          CONSTRAINT product_private_inline_indeterminate_amount_positive CHECK (amount > 0))",
        "INSERT INTO product_private_inline_indeterminate_owner VALUES (1, 11), (2, 22)",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private domain/table/INSERT post-durable transaction");
    }
    engine.fail_next_transaction_post_durable_apply();
    let failure = engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect_err("post-durable private-domain apply fault must be indeterminate");
    assert!(
        failure.is_indeterminate(),
        "expected indeterminate private-domain failure: {failure}"
    );
    assert!(engine.is_commit_path_poisoned());
    assert_eq!(engine.durable_wal_records().len(), records_before + 1);

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("recovery owns the durable private-domain transaction");
    assert_eq!(recovered.durable_wal_records().len(), records_before + 1);
    assert_eq!(
        durable_ids(&recovered, "product_private_inline_indeterminate_owner"),
        vec![1, 2]
    );
    assert!(recovered
        .relational_catalog_domain("product_private_inline_indeterminate_amount")
        .is_some());
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        recovered.relational_named_index_covered_rows("product_private_inline_indeterminate_owner"),
        Some(2),
        "recovery publishes the transaction-created private-domain primary index"
    );
    let record = recovered
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == txn_id)
        .expect("durable private-domain transaction remains present");
    assert!(codec5_replay_metadata(&record).initial_table_absent);

    let records_before_second_reopen = recovered.durable_wal_records().len();
    drop(recovered);
    let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("repeat recovery of the private-domain transaction is exact");
    assert_eq!(
        reopened.durable_wal_records().len(),
        records_before_second_reopen,
        "repeat recovery must not append or duplicate the private-domain transaction"
    );
    assert_eq!(
        durable_ids(&reopened, "product_private_inline_indeterminate_owner"),
        vec![1, 2]
    );
    assert!(reopened
        .relational_catalog_domain("product_private_inline_indeterminate_amount")
        .is_some());
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        reopened.relational_named_index_covered_rows("product_private_inline_indeterminate_owner"),
        Some(2),
        "repeat recovery retains the private-domain primary index"
    );
}

#[test]
fn product_001_private_create_sequence_table_foreign_key_failure_rolls_back_without_publication() {
    let fixture = DurableFixture::new("private-create-foreign-key-rollback");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    engine
        .execute_text(
            71_369,
            "CREATE DOMAIN product_private_fk_failure_tally AS INT",
        )
        .expect("create the domain used by the private FK failure table");

    let txn_id = 71_372;
    let durable_before = engine.durable_wal_records();
    let visible_before = engine.visible_up_to();
    let next_oid_before = engine.catalog_snapshot().relational_next_oid;
    for sql in [
        "BEGIN",
        "CREATE SEQUENCE product_private_fk_failure_sequence",
        "CREATE TABLE product_private_fk_failure_parent (id INT PRIMARY KEY)",
        "CREATE TABLE product_private_fk_failure_owner \
         (id INT DEFAULT nextval('product_private_fk_failure_sequence'::regclass), \
          row_key INT PRIMARY KEY, parent_id INT, \
          tally product_private_fk_failure_tally, \
          CONSTRAINT product_private_fk_failure_tally_positive CHECK (tally > 0))",
        "ALTER TABLE ONLY product_private_fk_failure_owner ADD CONSTRAINT product_private_fk_failure_owner_parent \
         FOREIGN KEY (parent_id) REFERENCES product_private_fk_failure_parent(id)",
        "INSERT INTO product_private_fk_failure_parent VALUES (7)",
        "INSERT INTO product_private_fk_failure_owner (row_key, parent_id, tally) VALUES (1, 7, 1)",
        "ALTER SEQUENCE product_private_fk_failure_sequence \
         RENAME TO product_private_fk_failure_sequence_final",
        "ALTER SEQUENCE product_private_fk_failure_sequence_final RESTART WITH 40",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private sequence/table/FK/rename transaction");
    }
    let foreign_key = engine
        .submit_transaction(
            txn_id,
            parsed(
                "INSERT INTO product_private_fk_failure_owner (row_key, parent_id, tally) VALUES (2, 99, 2)",
            ),
        )
        .expect_err("private first-table foreign key must reject at the INSERT statement");
    assert!(
        foreign_key
            .to_string()
            .to_ascii_lowercase()
            .contains("foreign key"),
        "private FK must retain its PostgreSQL diagnostic: {foreign_key}"
    );
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);

    engine
        .submit_transaction(txn_id, parsed("ROLLBACK"))
        .expect("rollback after private FK rejection");
    assert_eq!(engine.durable_wal_records(), durable_before);
    assert_eq!(engine.visible_up_to(), visible_before);
    assert_eq!(
        engine.catalog_snapshot().relational_next_oid,
        next_oid_before
    );
    assert!(engine
        .relational_catalog_table("product_private_fk_failure_owner")
        .is_none());
    assert!(engine
        .relational_catalog_table("product_private_fk_failure_parent")
        .is_none());
    assert!(engine
        .relational_catalog_sequence("product_private_fk_failure_sequence")
        .is_none());
    assert!(engine
        .relational_catalog_sequence("product_private_fk_failure_sequence_final")
        .is_none());

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("private FK rollback leaves only the published parent prefix");
    assert_eq!(recovered.durable_wal_records(), durable_before);
    assert_eq!(recovered.visible_up_to(), visible_before);
    assert_eq!(
        recovered.catalog_snapshot().relational_next_oid,
        next_oid_before
    );
    assert!(recovered
        .relational_catalog_table("product_private_fk_failure_owner")
        .is_none());
    assert!(recovered
        .relational_catalog_table("product_private_fk_failure_parent")
        .is_none());
    assert!(recovered
        .relational_catalog_sequence("product_private_fk_failure_sequence_final")
        .is_none());
}

#[test]
fn product_001_private_create_sequence_table_foreign_key_crash_prefixes_retry_once() {
    use crate::engine_transaction_delta::TransactionTerminalPreDurableFault;

    let boundaries = [
        (
            "before-replication-proposal",
            TransactionTerminalPreDurableFault::ReplicationProposal,
        ),
        (
            "before-canonical-wal-append",
            TransactionTerminalPreDurableFault::CanonicalWalAppend,
        ),
        (
            "before-wal-flush",
            TransactionTerminalPreDurableFault::WalFlush,
        ),
    ];

    let stage_private_fk = |engine: &Engine, txn_id| {
        for sql in [
            "BEGIN",
            "CREATE SEQUENCE product_private_fk_prefix_sequence",
            "CREATE TABLE product_private_fk_prefix_parent (id INT PRIMARY KEY)",
            "CREATE TABLE product_private_fk_prefix_owner \
             (id INT DEFAULT nextval('product_private_fk_prefix_sequence'::regclass), \
              row_key INT PRIMARY KEY, parent_id INT, \
              tally product_private_fk_prefix_tally, \
              CONSTRAINT product_private_fk_prefix_tally_positive CHECK (tally > 0))",
            "ALTER TABLE ONLY product_private_fk_prefix_owner ADD CONSTRAINT product_private_fk_prefix_owner_parent \
             FOREIGN KEY (parent_id) REFERENCES product_private_fk_prefix_parent(id)",
            "INSERT INTO product_private_fk_prefix_parent VALUES (7)",
            "INSERT INTO product_private_fk_prefix_owner (row_key, parent_id, tally) VALUES (1, 7, 11)",
            "ALTER SEQUENCE product_private_fk_prefix_sequence \
             RENAME TO product_private_fk_prefix_sequence_final",
            "INSERT INTO product_private_fk_prefix_owner (row_key, parent_id, tally) VALUES (2, 7, 22)",
            "ALTER SEQUENCE product_private_fk_prefix_sequence_final RESTART WITH 40",
            "INSERT INTO product_private_fk_prefix_owner (row_key, parent_id, tally) VALUES (3, 7, 33)",
        ] {
            engine
                .submit_transaction(txn_id, parsed(sql))
                .expect("stage private sequence/table/FK/rename transaction");
        }
    };

    for (ordinal, (label, boundary)) in boundaries.into_iter().enumerate() {
        let fixture = DurableFixture::new(&format!("private-create-foreign-key-{label}"));
        let engine = Engine::with_durable_wal_segment(&fixture.wal);
        engine
            .execute_text(
                71_390 + u64::try_from(ordinal).unwrap() * 10,
                "CREATE DOMAIN product_private_fk_prefix_tally AS INT",
            )
            .expect("create the domain used by the private retry table");

        let txn_id = 71_500 + u64::try_from(ordinal).unwrap();
        let durable_before = engine.durable_wal_records();
        let visible_before = engine.visible_up_to();
        stage_private_fk(&engine, txn_id);
        engine.fail_next_transaction_terminal_pre_durable_at(boundary);
        let failure = engine.submit_transaction(txn_id, parsed("COMMIT"));
        assert!(
            matches!(
                failure,
                Err(ExecuteError::Engine(EngineError::Durability(_)))
            ),
            "{label} must fail before acknowledgement: {failure:?}"
        );
        assert_eq!(engine.durable_wal_records(), durable_before, "{label}");
        assert_eq!(engine.visible_up_to(), visible_before, "{label}");
        assert!(
            engine
                .relational_catalog_table("product_private_fk_prefix_owner")
                .is_none(),
            "{label} must not publish the private FK table"
        );
        assert!(
            engine
                .relational_catalog_table("product_private_fk_prefix_parent")
                .is_none(),
            "{label} must not publish the private FK parent"
        );
        assert!(
            engine
                .relational_catalog_sequence("product_private_fk_prefix_sequence_final")
                .is_none(),
            "{label} must not publish the renamed private FK sequence"
        );

        drop(engine);
        let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("recover private FK crash prefix");
        assert_eq!(recovered.durable_wal_records(), durable_before, "{label}");
        assert!(
            recovered
                .relational_catalog_table("product_private_fk_prefix_owner")
                .is_none(),
            "{label} fresh reopen must not materialize the private FK table"
        );
        assert!(
            recovered
                .relational_catalog_table("product_private_fk_prefix_parent")
                .is_none(),
            "{label} fresh reopen must not materialize the private FK parent"
        );

        stage_private_fk(&recovered, txn_id);
        recovered
            .submit_transaction(txn_id, parsed("COMMIT"))
            .expect("the unacknowledged private FK transaction retries through codec-5");
        assert_eq!(
            durable_ids(&recovered, "product_private_fk_prefix_owner"),
            vec![1, 2, 40],
            "{label} retry publishes every pre-rename, post-rename, and post-restart row exactly once"
        );
        assert_eq!(
            durable_ids(&recovered, "product_private_fk_prefix_parent"),
            vec![7],
            "{label} retry publishes the private FK parent exactly once"
        );
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            recovered.relational_named_index_covered_rows("product_private_fk_prefix_owner"),
            Some(3),
            "{label} retry must publish the transaction-created FK owner's primary index with every row"
        );
        let sequence = recovered
            .relational_catalog_sequence("product_private_fk_prefix_sequence_final")
            .expect("retry publishes final private FK sequence binding");
        assert_eq!(
            (sequence.last_value, sequence.is_called),
            (40, true),
            "{label}"
        );
        let record = recovered
            .durable_wal_records()
            .into_iter()
            .find(|record| record.txn_id == txn_id)
            .expect("the acknowledged retry remains in canonical WAL");
        assert!(codec5_replay_metadata(&record).initial_table_absent);

        drop(recovered);
        let reopened = Engine::open_durable_wal_segment_auto(&fixture.wal)
            .expect("fresh reopen must recover the acknowledged private FK retry");
        assert_eq!(
            durable_ids(&reopened, "product_private_fk_prefix_owner"),
            vec![1, 2, 40],
            "{label} fresh replay owns the retried private FK rowset"
        );
        assert_eq!(
            durable_ids(&reopened, "product_private_fk_prefix_parent"),
            vec![7],
            "{label} fresh replay owns the retried private FK parent"
        );
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            reopened.relational_named_index_covered_rows("product_private_fk_prefix_owner"),
            Some(3),
            "{label} fresh replay must republish the retried private FK primary index"
        );
        let sequence = reopened
            .relational_catalog_sequence("product_private_fk_prefix_sequence_final")
            .expect("fresh replay restores the final private FK sequence binding");
        assert_eq!(
            (sequence.last_value, sequence.is_called),
            (40, true),
            "{label}"
        );
    }
}

#[test]
fn product_001_private_create_sequence_table_foreign_key_post_durable_recovery_owns_once() {
    let fixture = DurableFixture::new("private-create-foreign-key-post-durable");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    engine
        .execute_text(
            71_379,
            "CREATE DOMAIN product_private_fk_indeterminate_tally AS INT",
        )
        .expect("create the domain used by the private recovery table");

    let txn_id = 71_382;
    let records_before = engine.durable_wal_records().len();
    for sql in [
        "BEGIN",
        "CREATE SEQUENCE product_private_fk_indeterminate_sequence",
        "CREATE TABLE product_private_fk_indeterminate_parent (id INT PRIMARY KEY)",
        "CREATE TABLE product_private_fk_indeterminate_owner \
         (id INT DEFAULT nextval('product_private_fk_indeterminate_sequence'::regclass), \
          row_key INT PRIMARY KEY, parent_id INT, \
          tally product_private_fk_indeterminate_tally, \
          CONSTRAINT product_private_fk_indeterminate_tally_positive CHECK (tally > 0))",
        "ALTER TABLE ONLY product_private_fk_indeterminate_owner ADD CONSTRAINT product_private_fk_indeterminate_owner_parent \
         FOREIGN KEY (parent_id) REFERENCES product_private_fk_indeterminate_parent(id)",
        "INSERT INTO product_private_fk_indeterminate_parent VALUES (7)",
        "INSERT INTO product_private_fk_indeterminate_owner (row_key, parent_id, tally) VALUES (1, 7, 11)",
        "ALTER SEQUENCE product_private_fk_indeterminate_sequence \
         RENAME TO product_private_fk_indeterminate_sequence_final",
        "INSERT INTO product_private_fk_indeterminate_owner (row_key, parent_id, tally) VALUES (2, 7, 22)",
        "ALTER SEQUENCE product_private_fk_indeterminate_sequence_final RESTART WITH 40",
        "INSERT INTO product_private_fk_indeterminate_owner (row_key, parent_id, tally) VALUES (3, 7, 33)",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private sequence/table/FK/rename transaction");
    }
    engine.fail_next_transaction_post_durable_apply();
    let failure = engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect_err("post-durable private FK apply fault must be indeterminate");
    assert!(
        failure.is_indeterminate(),
        "expected indeterminate failure: {failure}"
    );
    assert!(engine.is_commit_path_poisoned());
    assert_eq!(engine.durable_wal_records().len(), records_before + 1);

    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("recover indeterminate private FK transaction");
    assert_eq!(recovered.durable_wal_records().len(), records_before + 1);
    assert_eq!(
        durable_ids(&recovered, "product_private_fk_indeterminate_owner"),
        vec![1, 2, 40]
    );
    assert_eq!(
        durable_ids(&recovered, "product_private_fk_indeterminate_parent"),
        vec![7]
    );
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        recovered.relational_named_index_covered_rows("product_private_fk_indeterminate_owner"),
        Some(3),
        "post-durable recovery must publish the transaction-created FK owner's primary index"
    );
    let sequence = recovered
        .relational_catalog_sequence("product_private_fk_indeterminate_sequence_final")
        .expect("recovery publishes the final FK sequence binding");
    assert_eq!((sequence.last_value, sequence.is_called), (40, true));
    let record = recovered
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == txn_id)
        .expect("durable private FK transaction remains present");
    assert!(codec5_replay_metadata(&record).initial_table_absent);

    let records_before_second_reopen = recovered.durable_wal_records().len();
    drop(recovered);
    let reopened =
        Engine::open_durable_wal_segment_auto(&fixture.wal).expect("repeat FK recovery is exact");
    assert_eq!(
        reopened.durable_wal_records().len(),
        records_before_second_reopen,
        "repeat recovery must not append or duplicate the private FK transaction"
    );
    assert_eq!(
        durable_ids(&reopened, "product_private_fk_indeterminate_owner"),
        vec![1, 2, 40]
    );
    assert_eq!(
        durable_ids(&reopened, "product_private_fk_indeterminate_parent"),
        vec![7]
    );
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        reopened.relational_named_index_covered_rows("product_private_fk_indeterminate_owner"),
        Some(3),
        "repeat recovery must retain the transaction-created FK owner's primary index"
    );
    let sequence = reopened
        .relational_catalog_sequence("product_private_fk_indeterminate_sequence_final")
        .expect("repeat recovery retains the final FK sequence binding");
    assert_eq!((sequence.last_value, sequence.is_called), (40, true));
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
    let envelope = gpu_db_wal::decode_canonical_record_payload(&committed.payload)
        .expect("decode durable canonical record")
        .expect("post-durable typed INSERT uses a canonical envelope");
    assert!(
        Engine::canonical_envelope_is_codec5(&envelope),
        "the recovery-owned transaction must retain the one codec-5 INSERT authority"
    );
    let fragments = envelope
        .fragments
        .iter()
        .map(|fragment| gpu_db_wal::CanonicalFragmentRef {
            kind: fragment.kind,
            body: &fragment.body,
        })
        .collect::<Vec<_>>();
    let replay = crate::typed_insert_aggregate::decode_closed_semantics_v2_replay(
        &envelope.header,
        &envelope.outcome,
        &fragments,
    )
    .expect("strictly close durable codec-5 transaction")
    .expect("durable INSERT selects semantics-v2 replay");
    let request_digest = replay.metadata().request_digest;
    assert_eq!(request_digest, envelope.header.request_digest);
    assert_eq!(replay.metadata().affected_rows, 1);
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
fn product_001_private_create_sequence_table_rename_post_durable_recovery_owns_once() {
    let fixture = DurableFixture::new("private-create-post-durable");
    let engine = Engine::with_durable_wal_segment(&fixture.wal);
    let txn_id = 72_100;
    let records_before = engine.durable_wal_records().len();

    for sql in [
        "BEGIN",
        "CREATE SEQUENCE product_private_indeterminate_sequence",
        "CREATE TABLE product_private_indeterminate_owner \
         (id INT DEFAULT nextval('product_private_indeterminate_sequence'::regclass), \
          row_key INT PRIMARY KEY, note TEXT)",
        "INSERT INTO product_private_indeterminate_owner (row_key, note) VALUES (1, 'before-rename')",
        "ALTER SEQUENCE product_private_indeterminate_sequence \
         RENAME TO product_private_indeterminate_sequence_final",
        "INSERT INTO product_private_indeterminate_owner (row_key, note) VALUES (2, 'after-rename')",
        "ALTER SEQUENCE product_private_indeterminate_sequence_final RESTART WITH 40",
        "INSERT INTO product_private_indeterminate_owner (row_key, note) VALUES (3, 'after-restart')",
    ] {
        engine
            .submit_transaction(txn_id, parsed(sql))
            .expect("stage private first-table transaction");
    }
    engine.fail_next_transaction_post_durable_apply();
    let failure = engine
        .submit_transaction(txn_id, parsed("COMMIT"))
        .expect_err("post-durable first-table apply fault must be indeterminate");
    assert!(
        failure.is_indeterminate(),
        "expected indeterminate failure: {failure}"
    );
    assert!(engine.is_commit_path_poisoned());
    assert_eq!(engine.durable_wal_records().len(), records_before + 1);
    assert!(
        engine
            .execute_text(
                72_101,
                "CREATE TABLE product_private_indeterminate_decoy (id INT)"
            )
            .is_err(),
        "the indeterminate engine must fail closed before a second catalog authority can publish"
    );

    // The acknowledged codec-5 record owns both S3 catalog composition and the first GPU table
    // generation. Restart recovery is therefore the sole completion path.
    drop(engine);
    let recovered = Engine::open_durable_wal_segment_auto(&fixture.wal)
        .expect("recover indeterminate private first-table transaction");
    assert_eq!(recovered.durable_wal_records().len(), records_before + 1);
    assert_eq!(
        durable_ids(&recovered, "product_private_indeterminate_owner"),
        vec![1, 2, 40]
    );
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        recovered.relational_named_index_covered_rows("product_private_indeterminate_owner"),
        Some(3),
        "post-durable recovery must publish the transaction-created primary index"
    );
    let sequence = recovered
        .relational_catalog_sequence("product_private_indeterminate_sequence_final")
        .expect("recovery publishes the final stable sequence binding");
    assert_eq!((sequence.last_value, sequence.is_called), (40, true));
    let record = recovered
        .durable_wal_records()
        .into_iter()
        .find(|record| record.txn_id == txn_id)
        .expect("durable first-table transaction remains present");
    let metadata = codec5_replay_metadata(&record);
    assert!(metadata.initial_table_absent);
    let request_digest = metadata.request_digest;
    assert_eq!(
        recovered
            .commit_state()
            .resolve_transaction_retry_digest_outcome(txn_id, request_digest)
            .expect("recovery indexes the exact transaction retry outcome")
            .expect("matching retry digest resolves")
            .1,
        3,
        "the recovered terminal outcome reports every user row"
    );

    let records_before_second_reopen = recovered.durable_wal_records().len();
    drop(recovered);
    let reopened =
        Engine::open_durable_wal_segment_auto(&fixture.wal).expect("repeat recovery is exact");
    assert_eq!(
        reopened.durable_wal_records().len(),
        records_before_second_reopen,
        "repeat recovery must not append or duplicate the private transaction"
    );
    assert_eq!(
        durable_ids(&reopened, "product_private_indeterminate_owner"),
        vec![1, 2, 40]
    );
    #[cfg(feature = "probe-timing")]
    assert_eq!(
        reopened.relational_named_index_covered_rows("product_private_indeterminate_owner"),
        Some(3),
        "repeat recovery must retain the transaction-created primary index"
    );
    let sequence = reopened
        .relational_catalog_sequence("product_private_indeterminate_sequence_final")
        .expect("repeat recovery retains the final sequence binding");
    assert_eq!((sequence.last_value, sequence.is_called), (40, true));
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
    let table = engine
        .relational_catalog_table("default_wal_owner")
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

    let user = codec5_replay_metadata(records.last().unwrap());
    assert_eq!(user.stable_transaction_id, 7_110);
    assert_eq!(user.display_oid, table.oid);
    assert_eq!(user.statement_count, 1);
    assert_eq!(user.affected_rows, 1);
    assert_eq!(user.row_allocator_high_water, user.row_allocator_before + 1);

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
fn explicit_sequence_default_cells_use_catalog_order_and_only_requested_cells_transition() {
    let engine = Engine::new_local_test_engine();
    engine
        .submit_transaction(
            7_150,
            parsed("CREATE SEQUENCE explicit_default_cells_value"),
        )
        .unwrap();
    engine
        .submit_transaction(
            7_151,
            parsed(
                "CREATE TABLE explicit_default_cells \
                 (a INT DEFAULT nextval('explicit_default_cells_value'::regclass), \
                  b INT DEFAULT nextval('explicit_default_cells_value'::regclass), \
                  marker INT UNIQUE)",
            ),
        )
        .unwrap();

    // PostgreSQL evaluates volatile defaults row-major and then in catalog-column order, rather
    // than INSERT source-column order. The source says (b, a), but a receives 1 and b receives 2.
    let reversed_source = "INSERT INTO explicit_default_cells (b, a, marker) \
                           VALUES (DEFAULT, DEFAULT, 1)";
    let command = parsed(reversed_source);
    assert!(engine.insert_sequence_default_route(command.command()).0);
    let wal_before = engine.durable_wal_records().len();
    let TransactionAdmissionResult::Dml(result) =
        engine.submit_transaction(7_200, command).unwrap()
    else {
        panic!("explicit sequence defaults must use the DML route");
    };
    assert_eq!(result.rows_affected, 1);
    let records = engine.durable_wal_records();
    let transitions = records[wal_before..]
        .iter()
        .filter_map(|record| {
            let payload = operation_payload_if_not_codec5(record)?;
            let BinaryWalRecord::SequenceValueTransition(transition) =
                decode_binary_record(&payload).unwrap()
            else {
                return None;
            };
            Some((
                transition.statement_ordinal,
                transition.expression_ordinal,
                transition.returned_value,
            ))
        })
        .collect::<Vec<_>>();
    assert_eq!(transitions, vec![(0, 0, 1), (0, 1, 2)]);
    assert_eq!(
        codec5_replay_metadata(records.last().unwrap()).affected_rows,
        1
    );

    // Both sequence-default columns are present in this statement, but each row asks for just
    // one. Reserve the skipped potential positions so the durable expression identities remain
    // row-major/catalog-order coordinates.
    let wal_before = engine.durable_wal_records().len();
    let TransactionAdmissionResult::Dml(result) = engine
        .submit_transaction(
            7_300,
            parsed(
                "INSERT INTO explicit_default_cells (b, a, marker) \
                 VALUES (DEFAULT, 40, 2), (50, DEFAULT, 3)",
            ),
        )
        .unwrap()
    else {
        panic!("mixed explicit defaults must use the DML route");
    };
    assert_eq!(result.rows_affected, 2);
    let records = engine.durable_wal_records();
    let transitions = records[wal_before..]
        .iter()
        .filter_map(|record| {
            let payload = operation_payload_if_not_codec5(record)?;
            let BinaryWalRecord::SequenceValueTransition(transition) =
                decode_binary_record(&payload).unwrap()
            else {
                return None;
            };
            Some((transition.expression_ordinal, transition.returned_value))
        })
        .collect::<Vec<_>>();
    assert_eq!(transitions, vec![(1, 3), (2, 4)]);
    assert_eq!(
        codec5_replay_metadata(records.last().unwrap()).affected_rows,
        2
    );

    // The implicit target list is catalog ordered too. Explicit NULL and scalar literals are
    // supplied values, so neither can take the sequence-default transition route.
    let implicit_default =
        parsed("INSERT INTO explicit_default_cells VALUES (DEFAULT, DEFAULT, 4)");
    assert!(
        engine
            .insert_sequence_default_route(implicit_default.command())
            .0
    );
    let TransactionAdmissionResult::Dml(result) =
        engine.submit_transaction(7_400, implicit_default).unwrap()
    else {
        panic!("implicit catalog-order defaults must use the DML route");
    };
    assert_eq!(result.rows_affected, 1);
    let literal = parsed("INSERT INTO explicit_default_cells (a, b, marker) VALUES (99, 100, 7)");
    assert!(!engine.insert_sequence_default_route(literal.command()).0);
    #[cfg(feature = "probe-timing")]
    let probe_before = engine.insert_probe_snapshot();
    let TransactionAdmissionResult::Dml(literal_result) =
        engine.submit_transaction(7_500, literal).unwrap()
    else {
        panic!("supplied sequence-table values must use the typed INSERT terminal");
    };
    assert_eq!(literal_result.rows_affected, 1);
    let null = parsed("INSERT INTO explicit_default_cells (a, b, marker) VALUES (NULL, 100, 8)");
    assert!(!engine.insert_sequence_default_route(null.command()).0);
    let TransactionAdmissionResult::Dml(null_result) =
        engine.submit_transaction(7_600, null).unwrap()
    else {
        panic!("supplied NULL sequence-table values must use the typed INSERT terminal");
    };
    assert_eq!(null_result.rows_affected, 1);
    #[cfg(feature = "probe-timing")]
    {
        let probe = engine.insert_probe_snapshot().delta_since(probe_before);
        assert_eq!(probe.successful_insert_statements, 2);
    }
    assert_eq!(
        engine
            .relational_catalog_sequence("explicit_default_cells_value")
            .unwrap()
            .last_value,
        6,
        "explicit NULL and literal values must not consume the sequence"
    );

    // An omitted published default remains a per-row request, not a single statement request.
    let omitted = parsed("INSERT INTO explicit_default_cells (marker) VALUES (5), (6)");
    assert!(engine.insert_sequence_default_route(omitted.command()).0);
    let TransactionAdmissionResult::Dml(result) =
        engine.submit_transaction(7_700, omitted).unwrap()
    else {
        panic!("omitted published defaults must use the DML route");
    };
    assert_eq!(result.rows_affected, 2);

    // Bind keeps DEFAULT as a semantic request while preserving the parameter's bound provenance.
    engine
        .submit_transaction(
            7_800,
            parsed(
                "CREATE TABLE prepared_explicit_default_cell \
                 (id INT DEFAULT nextval('explicit_default_cells_value'::regclass), \
                  marker INT UNIQUE)",
            ),
        )
        .unwrap();
    let bound = gpu_db_sql::PreparedCommand::parse(
        "INSERT INTO prepared_explicit_default_cell (id, marker) VALUES (DEFAULT, $1)",
    )
    .unwrap()
    .bind(&[SqlValue::Int4(77)])
    .unwrap();
    let Command::Insert(insert) = bound.command() else {
        panic!("bound statement must retain INSERT semantics");
    };
    assert!(matches!(
        insert.rows[0][0],
        InsertCell::Default {
            provenance: gpu_db_sql::InsertDefaultProvenance::SqlKeyword
        }
    ));
    assert!(matches!(
        insert.rows[0][1],
        InsertCell::Value {
            value: SqlValue::Int4(77),
            provenance: gpu_db_sql::InsertValueProvenance::BoundParameter { index: 1 }
        }
    ));
    assert!(engine.insert_sequence_default_route(bound.command()).0);
    let TransactionAdmissionResult::Dml(result) = engine.submit_transaction(7_900, bound).unwrap()
    else {
        panic!("prepared explicit DEFAULT must use the DML route");
    };
    assert_eq!(result.rows_affected, 1);

    // DEFAULT remains an ordinary literal default when the published column is not nextval.
    engine
        .submit_transaction(
            8_000,
            parsed(
                "CREATE TABLE literal_explicit_default_cell \
                 (id INT DEFAULT 19, marker INT UNIQUE)",
            ),
        )
        .unwrap();
    let literal_default =
        parsed("INSERT INTO literal_explicit_default_cell (id, marker) VALUES (DEFAULT, 1)");
    assert!(
        !engine
            .insert_sequence_default_route(literal_default.command())
            .0
    );
    assert!(matches!(
        engine.submit_transaction(8_100, literal_default).unwrap(),
        TransactionAdmissionResult::Command | TransactionAdmissionResult::Dml(_)
    ));
    assert_eq!(
        durable_ids(&engine, "literal_explicit_default_cell"),
        vec![19]
    );

    let Command::Select(select) =
        parse_command("SELECT a, b, marker FROM explicit_default_cells ORDER BY marker").unwrap()
    else {
        panic!("expected SELECT");
    };
    assert_eq!(
        engine.execute_relational_select(&select).unwrap().rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(2), SqlValue::Int4(1)],
            vec![SqlValue::Int4(40), SqlValue::Int4(3), SqlValue::Int4(2)],
            vec![SqlValue::Int4(4), SqlValue::Int4(50), SqlValue::Int4(3)],
            vec![SqlValue::Int4(5), SqlValue::Int4(6), SqlValue::Int4(4)],
            vec![SqlValue::Int4(7), SqlValue::Int4(8), SqlValue::Int4(5)],
            vec![SqlValue::Int4(9), SqlValue::Int4(10), SqlValue::Int4(6)],
            vec![SqlValue::Int4(99), SqlValue::Int4(100), SqlValue::Int4(7)],
            vec![SqlValue::Null, SqlValue::Int4(100), SqlValue::Int4(8)],
        ]
    );
}

#[test]
fn explicit_sequence_default_gap_retry_recovery_preserves_nontransactional_transition() {
    let engine = Engine::new_local_test_engine();
    engine
        .submit_transaction(7_170, parsed("CREATE SEQUENCE explicit_default_gap_value"))
        .unwrap();
    engine
        .submit_transaction(
            7_171,
            parsed(
                "CREATE TABLE explicit_default_gap \
                 (id INT DEFAULT nextval('explicit_default_gap_value'::regclass), \
                  marker INT UNIQUE)",
            ),
        )
        .unwrap();

    let first = parsed("INSERT INTO explicit_default_gap (id, marker) VALUES (DEFAULT, 1)");
    assert!(engine.insert_sequence_default_route(first.command()).0);
    let TransactionAdmissionResult::Dml(result) = engine.submit_transaction(7_200, first).unwrap()
    else {
        panic!("explicit DEFAULT must enter the sequence DML route");
    };
    assert_eq!(result.rows_affected, 1);

    let failing = "INSERT INTO explicit_default_gap (id, marker) VALUES (DEFAULT, 1)";
    let wal_before_failure = engine.durable_wal_records().len();
    let failure = engine
        .submit_transaction(7_300, parsed(failing))
        .unwrap_err();
    assert_eq!(
        engine
            .relational_catalog_sequence("explicit_default_gap_value")
            .unwrap()
            .last_value,
        2,
        "a failed explicit DEFAULT statement keeps its independently durable gap: {failure}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before_failure + 1);
    assert!(engine.submit_transaction(7_300, parsed(failing)).is_err());
    assert_eq!(
        engine.durable_wal_records().len(),
        wal_before_failure + 1,
        "the exact retry reuses its durable explicit-default transition"
    );

    let TransactionAdmissionResult::Dml(result) = engine
        .submit_transaction(
            7_400,
            parsed("INSERT INTO explicit_default_gap (id, marker) VALUES (DEFAULT, 3)"),
        )
        .unwrap()
    else {
        panic!("explicit DEFAULT success must return DML metadata");
    };
    assert_eq!(result.rows_affected, 1);
    assert_eq!(durable_ids(&engine, "explicit_default_gap"), vec![1, 3]);

    engine.submit_transaction(7_500, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            7_500,
            parsed("INSERT INTO explicit_default_gap (id, marker) VALUES (DEFAULT, 4)"),
        )
        .unwrap();
    engine
        .submit_transaction(7_500, parsed("ROLLBACK"))
        .unwrap();
    assert_eq!(durable_ids(&engine, "explicit_default_gap"), vec![1, 3]);
    assert_eq!(
        engine
            .relational_catalog_sequence("explicit_default_gap_value")
            .unwrap()
            .last_value,
        4,
        "explicit DEFAULT remains nontransactional across user rollback"
    );

    let records = engine.durable_wal_records();
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(durable_ids(&recovered, "explicit_default_gap"), vec![1, 3]);
    assert_eq!(
        recovered
            .relational_catalog_sequence("explicit_default_gap_value")
            .unwrap()
            .last_value,
        4
    );
    let recovered_wal_before_retry = recovered.durable_wal_records().len();
    let TransactionAdmissionResult::Dml(retry) = recovered
        .submit_transaction(
            7_400,
            parsed("INSERT INTO explicit_default_gap (id, marker) VALUES (DEFAULT, 3)"),
        )
        .unwrap()
    else {
        panic!("recovered explicit DEFAULT retry must resolve terminal DML metadata");
    };
    assert_eq!(retry.rows_affected, 1);
    assert_eq!(
        recovered.durable_wal_records().len(),
        recovered_wal_before_retry,
        "recovery retry must append no duplicate transition or user envelope"
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
                  required INT PRIMARY KEY, marker INT DEFAULT 0)",
            ),
        )
        .unwrap();
    let statement = "INSERT INTO failed_default_owner (marker) VALUES (9)";
    let wal_before = engine.durable_wal_records().len();
    let first = engine
        .submit_transaction(7_202, parsed(statement))
        .unwrap_err();
    assert!(first.to_string().contains("not-null constraint"), "{first}");
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
    assert!(retry.to_string().contains("not-null constraint"), "{retry}");
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
    assert!(retry.to_string().contains("not-null constraint"), "{retry}");
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
fn sequence_transition_tamper_is_clone_first_and_codec5_reference_replay_is_effect_free() {
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
    let user = codec5_replay_metadata(records.last().unwrap());
    assert_eq!(user.stable_transaction_id, 7_302);
    assert_eq!(user.affected_rows, 1);

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
    // The codec-5 S5 reference is validated by strict closure and then again against the durable
    // sequence-outcome index during recovery.  Replaying the exact acknowledged record through
    // that production route must install the row once; the rehashed S5 field-level hostile cases
    // live in the codec-5 Q1 sabotage matrix rather than the historical binary decoder tests.
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered.execute_relational_select(&select).unwrap().rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(8)]]
    );
    assert!(user_target
        .catalog_snapshot()
        .same_contents(before.as_ref()));
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
    let updated = gpu_db_wal::decode_canonical_record_payload(&records.last().unwrap().payload)
        .expect("decode insert-then-update canonical record")
        .expect("insert-then-update uses a canonical envelope");
    assert!(
        Engine::canonical_envelope_is_codec5(&updated),
        "insert-then-update must retain codec-5 as its sole INSERT authority"
    );
    let fragments = updated
        .fragments
        .iter()
        .map(|fragment| gpu_db_wal::CanonicalFragmentRef {
            kind: fragment.kind,
            body: &fragment.body,
        })
        .collect::<Vec<_>>();
    let replay = crate::typed_insert_aggregate::decode_closed_semantics_v2_replay(
        &updated.header,
        &updated.outcome,
        &fragments,
    )
    .expect("strictly close insert-then-update codec-5 record")
    .expect("insert-then-update selects semantics-v2 replay");
    assert_eq!(replay.metadata().affected_rows, 1);

    engine.submit_transaction(7_440, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(7_440, parsed("INSERT INTO rebound_owner (note) VALUES (2)"))
        .unwrap();
    engine
        .submit_transaction(7_440, parsed("DELETE FROM rebound_owner WHERE note = 2"))
        .unwrap();
    engine.submit_transaction(7_440, parsed("COMMIT")).unwrap();
    let records = engine.durable_wal_records();
    let deleted = gpu_db_wal::decode_canonical_record_payload(&records.last().unwrap().payload)
        .expect("decode insert-then-delete canonical record")
        .expect("insert-then-delete uses a canonical envelope");
    assert!(
        Engine::canonical_envelope_is_codec5(&deleted),
        "insert-then-delete must retain codec-5 as its sole INSERT authority"
    );
    let fragments = deleted
        .fragments
        .iter()
        .map(|fragment| gpu_db_wal::CanonicalFragmentRef {
            kind: fragment.kind,
            body: &fragment.body,
        })
        .collect::<Vec<_>>();
    let replay = crate::typed_insert_aggregate::decode_closed_semantics_v2_replay(
        &deleted.header,
        &deleted.outcome,
        &fragments,
    )
    .expect("strictly close insert-then-delete codec-5 record")
    .expect("insert-then-delete selects semantics-v2 replay");
    assert_eq!(replay.metadata().affected_rows, 1);

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
