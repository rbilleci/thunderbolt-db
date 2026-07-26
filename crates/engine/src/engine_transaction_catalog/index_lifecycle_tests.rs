use super::*;
use std::sync::mpsc;
use std::time::Duration;

mod cold_host_staging_tests;
mod constraint_identity_tests;
mod gpu_accounting_tests;
mod index_null_validation_tests;
mod namespace_tests;

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

fn index_named<'a>(table: &'a RelationalTable, name: &str) -> &'a RelationalIndex {
    table
        .indexes
        .iter()
        .find(|index| index.name == name)
        .unwrap_or_else(|| panic!("missing index {name:?}"))
}

fn select(sql: &str) -> Select {
    match parse_command(sql).unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    }
}

fn seed_shared_pg_class_oid_for_boundary_test(engine: &Engine, next_oid: u32) {
    let mut catalog = engine
        .catalog_latch
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    catalog.relational_next_oid = next_oid;
    engine.publish_catalog_snapshot(&catalog, engine.committed_seq(), 0);
}

#[test]
fn transactional_index_lifecycle_is_private_ordered_stable_and_recoverable() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            4_000,
            parsed("CREATE TABLE index_lifecycle (id int4, code int4, region int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_001,
            parsed("CREATE INDEX index_lifecycle_code ON index_lifecycle (code)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_002,
            parsed("COMMENT ON INDEX index_lifecycle_code IS 'stable metadata'"),
        )
        .unwrap();
    let original_table = engine.relational_catalog_table("index_lifecycle").unwrap();
    let original = index_named(&original_table, "index_lifecycle_code").clone();
    let allocator_before = engine.catalog_snapshot().relational_next_oid;
    let wal_before = engine.durable_wal_records().len();

    engine.submit_transaction(4_003, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_003,
            parsed(
                "ALTER INDEX index_lifecycle_code \
                 RENAME TO index_lifecycle_code_private",
            ),
        )
        .unwrap();
    assert!(
        index_named(
            &engine.relational_catalog_table("index_lifecycle").unwrap(),
            "index_lifecycle_code"
        )
        .oid == original.oid
    );
    assert!(engine
        .relational_catalog_table("index_lifecycle")
        .unwrap()
        .indexes
        .iter()
        .all(|index| index.name != "index_lifecycle_code_private"));

    let private = engine
        .transaction_snapshot_handle(4_003)
        .unwrap()
        .transaction_catalog();
    let renamed = index_named(
        &private.relational_catalog["index_lifecycle"],
        "index_lifecycle_code_private",
    );
    assert_eq!(renamed.oid, original.oid);
    assert_eq!(
        private
            .relational_comments
            .get(&RelationalCommentTarget::Index {
                index: "index_lifecycle_code_private".to_string(),
            })
            .map(String::as_str),
        Some("stable metadata")
    );

    engine
        .submit_transaction(
            4_003,
            parsed(
                "DROP INDEX IF EXISTS missing_index_lifecycle, \
                 index_lifecycle_code_private",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_003,
            parsed(
                "CREATE UNIQUE INDEX index_lifecycle_code_private \
                 ON index_lifecycle (code, region)",
            ),
        )
        .unwrap();
    let final_private = engine
        .transaction_snapshot_handle(4_003)
        .unwrap()
        .transaction_catalog();
    let replacement = index_named(
        &final_private.relational_catalog["index_lifecycle"],
        "index_lifecycle_code_private",
    )
    .clone();
    assert_ne!(replacement.oid, original.oid);
    assert_eq!(replacement.oid, allocator_before);
    assert!(replacement.unique);
    assert_eq!(replacement.key_columns, ["code", "region"]);
    assert!(!final_private
        .relational_comments
        .contains_key(&RelationalCommentTarget::Index {
            index: "index_lifecycle_code_private".to_string(),
        }));
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    engine.submit_transaction(4_003, parsed("COMMIT")).unwrap();
    let records = engine.durable_wal_records();
    assert_eq!(records.len(), wal_before + 1);
    let payload = operation_payload(records.last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("index lifecycle must use one typed transaction record");
    };
    assert_eq!(record.index_lifecycle_operations.len(), 3);
    assert_eq!(
        record
            .index_lifecycle_operations
            .iter()
            .map(|identity| (identity.command_index, identity.ordinal))
            .collect::<Vec<_>>(),
        vec![(0, 0), (1, 1), (2, 2)]
    );
    assert_eq!(
        record.operation_order,
        vec![
            BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
            BinaryTransactionOperationIdentity::Catalog { command_index: 1 },
            BinaryTransactionOperationIdentity::Catalog { command_index: 2 },
        ]
    );
    let rename = &record.index_lifecycle_operations[0].targets[0];
    assert_eq!(rename.owner_name.as_deref(), Some("index_lifecycle"));
    assert_eq!(rename.index_before.as_ref().unwrap().oid, original.oid);
    assert_eq!(rename.index_after.as_ref().unwrap().oid, original.oid);
    assert_eq!(
        rename.after_name.as_deref(),
        Some("index_lifecycle_code_private")
    );
    let drop = &record.index_lifecycle_operations[1].targets;
    assert_eq!(drop.len(), 2);
    assert!(drop[0].owner_name.is_none());
    assert!(drop[0].table_before.is_none());
    assert!(drop[0].index_before.is_none());
    assert_eq!(drop[1].index_before.as_ref().unwrap().oid, original.oid);
    assert!(drop[1].table_after.is_some());
    assert!(drop[1].index_after.is_none());
    let create = &record.index_lifecycle_operations[2].targets[0];
    assert!(create.index_before.is_none());
    assert_eq!(
        create.table_before.as_ref().unwrap().oid,
        original_table.oid
    );
    assert_eq!(create.index_after.as_ref().unwrap().oid, replacement.oid);
    assert_eq!(
        record.catalog_output.as_ref().unwrap().relational_next_oid,
        allocator_before + 1
    );

    let request_digest = gpu_db_wal::canonical_request_digest(&payload);
    assert_eq!(
        engine
            .commit_state()
            .resolve_transaction_retry_digest_outcome(4_003, request_digest)
            .unwrap()
            .unwrap()
            .1,
        0
    );
    assert!(engine
        .commit_state()
        .resolve_transaction_retry_digest_outcome(4_003, [0x49; 32])
        .unwrap_err()
        .to_string()
        .contains("different request"));

    let committed = engine.catalog_snapshot();
    assert_eq!(
        index_named(
            &committed.relational_catalog["index_lifecycle"],
            "index_lifecycle_code_private"
        ),
        &replacement
    );
    assert_eq!(committed.relational_next_oid, allocator_before + 1);
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(committed.as_ref()));
    assert_eq!(
        recovered.relational_index_comment("index_lifecycle_code_private"),
        None
    );
}

#[test]
fn multi_drop_index_is_atomic_dependency_checked_and_rollback_complete() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            4_010,
            parsed(
                "CREATE TABLE index_drop (id int4 PRIMARY KEY, a int4, b int4, \
                 UNIQUE (a))",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(4_011, parsed("CREATE INDEX index_drop_b ON index_drop (b)"))
        .unwrap();
    engine
        .submit_transaction(
            4_012,
            parsed("CREATE INDEX index_drop_ab ON index_drop (a, b)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_015,
            parsed("CREATE TABLE index_drop_peer (id int4, code int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_016,
            parsed("CREATE INDEX index_drop_peer_code ON index_drop_peer (code)"),
        )
        .unwrap();
    let published = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();
    engine.submit_transaction(4_013, parsed("BEGIN")).unwrap();

    for sql in [
        "DROP INDEX index_drop_b, index_drop_b",
        "DROP INDEX index_drop_b, missing_index_drop",
        "DROP INDEX IF EXISTS index_drop",
    ] {
        assert!(
            engine.submit_transaction(4_013, parsed(sql)).is_err(),
            "{sql}"
        );
        assert!(engine
            .transaction_snapshot_handle(4_013)
            .unwrap()
            .transaction_catalog()
            .same_contents(published.as_ref()));
    }
    engine
        .submit_transaction(
            4_013,
            parsed(
                "DROP INDEX IF EXISTS missing_index_drop, index_drop_b, \
                 index_drop_ab, index_drop_peer_code",
            ),
        )
        .unwrap();
    let private = engine
        .transaction_snapshot_handle(4_013)
        .unwrap()
        .transaction_catalog();
    assert_eq!(
        private.relational_catalog["index_drop"]
            .indexes
            .iter()
            .map(|index| index.name.as_str())
            .collect::<Vec<_>>(),
        vec!["index_drop_pkey", "index_drop_a_key"]
    );
    assert!(private.relational_catalog["index_drop_peer"]
        .indexes
        .is_empty());
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine
        .submit_transaction(4_013, parsed("ROLLBACK"))
        .unwrap();
    assert!(engine.catalog_snapshot().same_contents(published.as_ref()));
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    engine.submit_transaction(4_014, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_014,
            parsed("DROP INDEX IF EXISTS definitely_absent_index"),
        )
        .unwrap();
    engine.submit_transaction(4_014, parsed("COMMIT")).unwrap();
    assert!(engine.catalog_snapshot().same_contents(published.as_ref()));
    let payload = operation_payload(engine.durable_wal_records().last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("absent DROP INDEX must retain typed request identity");
    };
    let target = &record.index_lifecycle_operations[0].targets[0];
    assert!(target.owner_name.is_none());
    assert!(target.table_before.is_none());
    assert!(target.index_before.is_none());
    assert!(target.table_after.is_none());
    assert!(target.index_after.is_none());
}

#[test]
fn index_lifecycle_excludes_same_table_writers_and_releases_on_rollback() {
    let engine = Arc::new(Engine::new_local());
    engine
        .submit_transaction(
            4_020,
            parsed("CREATE TABLE index_guard (id int4, code int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_021,
            parsed("CREATE INDEX index_guard_code ON index_guard (code)"),
        )
        .unwrap();
    engine.submit_transaction(4_022, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_022,
            parsed("ALTER INDEX index_guard_code RENAME TO index_guard_private"),
        )
        .unwrap();
    let wal_before = engine.durable_wal_records().len();

    let writer = engine
        .submit_transaction(4_023, parsed("INSERT INTO index_guard VALUES (1, 10)"))
        .unwrap_err();
    assert!(matches!(writer, ExecuteError::Serialization(_)), "{writer}");
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    let ddl = engine
        .submit_transaction(4_024, parsed("DROP INDEX index_guard_code"))
        .unwrap_err();
    assert!(matches!(ddl, ExecuteError::Serialization(_)), "{ddl}");
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    engine
        .submit_transaction(4_022, parsed("ROLLBACK"))
        .unwrap();
    engine
        .submit_transaction(4_025, parsed("DROP INDEX index_guard_code"))
        .unwrap();
    assert!(engine
        .relational_catalog_table("index_guard")
        .unwrap()
        .indexes
        .is_empty());
}

#[test]
fn serializable_index_commit_rejects_before_replication_and_retry_remains_clean() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            4_018,
            parsed("CREATE TABLE serializable_index_owner (id int4, code int4)"),
        )
        .unwrap();
    let catalog_before = engine.catalog_snapshot();
    let allocator_before = catalog_before.relational_next_oid;
    let wal_before = engine.durable_wal_records().len();

    engine.submit_transaction(4_019, parsed("BEGIN")).unwrap();
    // Admission normally rejects SERIALIZABLE before BEGIN. Inject the characteristic directly
    // into an otherwise valid empty snapshot to adversarially exercise the commit-path invariant:
    // even a corrupt/internal caller must be rejected before it can propose a replication entry.
    let current = engine.transaction_snapshot_handle(4_019).unwrap();
    let replacement = engine.capture_transaction_snapshot(
        current.boundary,
        TransactionCharacteristics {
            isolation: TransactionIsolation::Serializable,
            ..current.characteristics
        },
    );
    engine
        .active_snapshots
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .replace_transaction_snapshot(4_019, &current, replacement, false)
        .unwrap();
    engine
        .submit_transaction(
            4_019,
            parsed(
                "CREATE INDEX serializable_index_code \
                 ON serializable_index_owner (code)",
            ),
        )
        .unwrap();
    assert_eq!(
        index_named(
            &engine
                .transaction_snapshot_handle(4_019)
                .unwrap()
                .transaction_catalog()
                .relational_catalog["serializable_index_owner"],
            "serializable_index_code",
        )
        .oid,
        allocator_before
    );
    let marks_before = engine.replication_watermarks();
    let error = engine
        .submit_transaction(4_019, parsed("COMMIT"))
        .expect_err("SERIALIZABLE must reject before replication proposal");
    assert!(matches!(error, ExecuteError::Unsupported(_)), "{error}");
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.replication_watermarks(), marks_before);
    assert_eq!(engine.committed_seq(), catalog_before.commit_seq);
    assert!(engine
        .catalog_snapshot()
        .same_contents(catalog_before.as_ref()));
    assert_eq!(
        engine.catalog_snapshot().relational_next_oid,
        allocator_before
    );
    assert!(engine
        .commit_state()
        .resolve_transaction_retry_digest_outcome(4_019, [0x71; 32])
        .unwrap()
        .is_none());

    engine
        .submit_transaction(4_019, parsed("ROLLBACK"))
        .unwrap();
    engine.submit_transaction(4_020, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_020,
            parsed(
                "CREATE INDEX serializable_index_code \
                 ON serializable_index_owner (code)",
            ),
        )
        .unwrap();
    engine.submit_transaction(4_020, parsed("COMMIT")).unwrap();
    let marks_after = engine.replication_watermarks();
    assert_eq!(marks_after.commit_index, marks_before.commit_index + 1);
    assert_eq!(marks_after.applied_index, marks_before.applied_index + 1);
    assert_eq!(marks_after.visible_index, marks_before.visible_index + 1);
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert_eq!(
        index_named(
            &engine
                .relational_catalog_table("serializable_index_owner")
                .unwrap(),
            "serializable_index_code",
        )
        .oid,
        allocator_before
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

#[test]
fn index_oid_allocation_rolls_back_and_catalog_allocator_aba_is_excluded() {
    let rollback = Engine::new_local();
    rollback
        .submit_transaction(
            4_026,
            parsed("CREATE TABLE index_oid_rollback (id int4, code int4, region int4)"),
        )
        .unwrap();
    let before = rollback.catalog_snapshot().relational_next_oid;
    rollback.submit_transaction(4_027, parsed("BEGIN")).unwrap();
    rollback
        .submit_transaction(
            4_027,
            parsed("CREATE INDEX rolled_index_oid ON index_oid_rollback (code)"),
        )
        .unwrap();
    assert_eq!(
        index_named(
            &rollback
                .transaction_snapshot_handle(4_027)
                .unwrap()
                .transaction_catalog()
                .relational_catalog["index_oid_rollback"],
            "rolled_index_oid"
        )
        .oid,
        before
    );
    rollback
        .submit_transaction(4_027, parsed("ROLLBACK"))
        .unwrap();
    assert_eq!(rollback.catalog_snapshot().relational_next_oid, before);
    rollback
        .submit_transaction(
            4_028,
            parsed("CREATE INDEX committed_index_oid ON index_oid_rollback (region)"),
        )
        .unwrap();
    assert_eq!(
        index_named(
            &rollback
                .relational_catalog_table("index_oid_rollback")
                .unwrap(),
            "committed_index_oid"
        )
        .oid,
        before
    );

    for begin in ["BEGIN", "BEGIN ISOLATION LEVEL REPEATABLE READ"] {
        let engine = Engine::new_local();
        engine
            .submit_transaction(
                4_066,
                parsed("CREATE TABLE index_rebase (id int4, code int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(
                4_067,
                parsed("CREATE INDEX index_rebase_code ON index_rebase (code)"),
            )
            .unwrap();
        let oid = index_named(
            &engine.relational_catalog_table("index_rebase").unwrap(),
            "index_rebase_code",
        )
        .oid;
        engine.submit_transaction(4_068, parsed(begin)).unwrap();
        engine
            .submit_transaction(
                4_068,
                parsed("ALTER INDEX index_rebase_code RENAME TO index_rebase_private"),
            )
            .unwrap();
        engine
            .submit_transaction(4_069, parsed("SET index_rebase_unrelated = yes"))
            .unwrap();
        engine
            .submit_transaction(
                4_068,
                parsed("ALTER INDEX index_rebase_private RENAME TO index_rebase_final"),
            )
            .unwrap();
        engine.submit_transaction(4_068, parsed("COMMIT")).unwrap();
        assert_eq!(
            index_named(
                &engine.relational_catalog_table("index_rebase").unwrap(),
                "index_rebase_final"
            )
            .oid,
            oid
        );
    }

    for begin in ["BEGIN", "BEGIN ISOLATION LEVEL REPEATABLE READ"] {
        let engine = Engine::new_local();
        engine
            .submit_transaction(
                4_060,
                parsed("CREATE TABLE index_aba_target (id int4, code int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(
                4_061,
                parsed("CREATE TABLE index_aba_peer (id int4, code int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(
                4_062,
                parsed("CREATE INDEX index_aba_code ON index_aba_target (code)"),
            )
            .unwrap();
        engine.submit_transaction(4_063, parsed(begin)).unwrap();
        engine
            .submit_transaction(
                4_063,
                parsed("ALTER INDEX index_aba_code RENAME TO index_aba_private"),
            )
            .unwrap();

        let wal_before_aba = engine.durable_wal_records().len();
        let create_aba = engine.submit_transaction(
            4_064,
            parsed("CREATE TABLE index_aba_ephemeral (id int4 PRIMARY KEY)"),
        );
        if let Err(error) = create_aba {
            assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
            assert_eq!(engine.durable_wal_records().len(), wal_before_aba);
            engine
                .submit_transaction(4_063, parsed("ROLLBACK"))
                .unwrap();
            continue;
        }
        engine
            .submit_transaction(4_065, parsed("DROP TABLE index_aba_ephemeral"))
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        let error = engine
            .submit_transaction(4_063, parsed("COMMIT"))
            .unwrap_err();
        assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert!(engine
            .relational_catalog_table("index_aba_target")
            .unwrap()
            .indexes
            .iter()
            .any(|index| index.name == "index_aba_code"));
        engine
            .submit_transaction(4_063, parsed("ROLLBACK"))
            .unwrap();
    }
}

#[test]
fn table_and_index_share_pg_class_oid_boundary_through_rollback_retry_and_replay() {
    let source = Engine::new_local();
    seed_shared_pg_class_oid_for_boundary_test(&source, 19_999);
    source
        .submit_transaction(
            4_026_100,
            parsed("CREATE TABLE oid_boundary_owner (id int4, code int4)"),
        )
        .unwrap();
    assert_eq!(
        source
            .relational_catalog_table("oid_boundary_owner")
            .unwrap()
            .oid,
        19_999
    );
    assert_eq!(source.catalog_snapshot().relational_next_oid, 20_000);

    source
        .submit_transaction(4_026_101, parsed("BEGIN"))
        .unwrap();
    source
        .submit_transaction(
            4_026_101,
            parsed("CREATE INDEX oid_boundary_index ON oid_boundary_owner (code)"),
        )
        .unwrap();
    assert_eq!(
        index_named(
            &source
                .transaction_snapshot_handle(4_026_101)
                .unwrap()
                .transaction_catalog()
                .relational_catalog["oid_boundary_owner"],
            "oid_boundary_index",
        )
        .oid,
        20_000
    );
    source
        .submit_transaction(4_026_101, parsed("ROLLBACK"))
        .unwrap();
    assert_eq!(source.catalog_snapshot().relational_next_oid, 20_000);

    source
        .submit_transaction(4_026_102, parsed("BEGIN"))
        .unwrap();
    source
        .submit_transaction(
            4_026_102,
            parsed("CREATE INDEX oid_boundary_index ON oid_boundary_owner (code)"),
        )
        .unwrap();
    source
        .submit_transaction(
            4_026_102,
            parsed("CREATE TABLE oid_boundary_successor (id int4)"),
        )
        .unwrap();
    source
        .submit_transaction(4_026_102, parsed("COMMIT"))
        .unwrap();

    let committed = source.catalog_snapshot();
    let index_oid = index_named(
        &committed.relational_catalog["oid_boundary_owner"],
        "oid_boundary_index",
    )
    .oid;
    let successor_oid = committed.relational_catalog["oid_boundary_successor"].oid;
    assert_eq!((index_oid, successor_oid), (20_000, 20_001));
    assert_eq!(committed.relational_next_oid, 20_002);
    let pg_class_oids = committed
        .relational_catalog
        .values()
        .flat_map(|table| {
            std::iter::once(table.oid).chain(table.indexes.iter().map(|index| index.oid))
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        pg_class_oids.len(),
        committed.relational_catalog.len()
            + committed
                .relational_catalog
                .values()
                .map(|table| table.indexes.len())
                .sum::<usize>(),
        "every relation and index must own one distinct pg_class OID"
    );

    let recovered = Engine::new_local();
    seed_shared_pg_class_oid_for_boundary_test(&recovered, 19_999);
    recovered.begin_recovery_replay();
    recovered
        .replay_durable_records(&source.durable_wal_records())
        .unwrap();
    recovered.finish_recovery_replay().unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(committed.as_ref()));
}

#[test]
fn exhausted_pg_class_highwater_accepts_index_neutral_recovery_and_rejects_reuse() {
    let source = Engine::new_local();
    seed_shared_pg_class_oid_for_boundary_test(&source, MAX_CATALOG_OID);
    source
        .submit_transaction(4_026_103, parsed("CREATE TABLE max_oid_owner (id int4)"))
        .unwrap();
    source
        .submit_transaction(4_026_104, parsed("SET max_oid_neutral = yes"))
        .unwrap();
    let committed = source.catalog_snapshot();
    assert_eq!(
        committed.relational_catalog["max_oid_owner"].oid,
        MAX_CATALOG_OID
    );
    assert_eq!(committed.relational_next_oid, MAX_CATALOG_OID + 1);
    let wal_before = source.durable_wal_records().len();
    let error = source
        .submit_transaction(4_026_105, parsed("CREATE TABLE max_oid_reuse (id int4)"))
        .expect_err("the exhausted high-water must not wrap or allocate out of domain");
    assert!(
        error.to_string().contains("OID allocation exhausted"),
        "{error}"
    );
    assert_eq!(source.durable_wal_records().len(), wal_before);
    assert!(source.catalog_snapshot().same_contents(committed.as_ref()));
    assert!(!source
        .catalog_snapshot()
        .relational_catalog
        .contains_key("max_oid_reuse"));

    let records = source.durable_wal_records();
    let recovered = Engine::new_local();
    seed_shared_pg_class_oid_for_boundary_test(&recovered, MAX_CATALOG_OID);
    recovered.begin_recovery_replay();
    recovered.replay_durable_records(&records).unwrap();
    recovered.finish_recovery_replay().unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(committed.as_ref()));
}

#[test]
fn legacy_index_oid_domain_edge_is_atomic_when_two_identities_are_synthesized() {
    let engine = Engine::new_local();
    let mut one = engine.ddl_catalog().clone();
    one.index_oid_epoch_current = false;
    one.legacy_recovery_floor_prepared = true;
    one.legacy_recovery_next_index_oid = MAX_CATALOG_OID;
    assert_eq!(
        one.migrated_legacy_index_oid(&BTreeSet::new()).unwrap(),
        MAX_CATALOG_OID
    );
    let error = one
        .migrated_legacy_index_oid(&BTreeSet::new())
        .expect_err("the second legacy index identity exceeds the Int4 catalog domain");
    assert!(
        error.to_string().contains("GPU catalog Int4 domain"),
        "{error}"
    );

    let mut catalog = engine.ddl_catalog().clone();
    catalog.index_oid_epoch_current = false;
    catalog.legacy_recovery_floor_prepared = true;
    catalog.legacy_recovery_next_index_oid = MAX_CATALOG_OID;
    let next_oid_before = catalog.relational_next_oid;
    let Command::CreateTable(create) = parse_command(
        "CREATE TABLE legacy_max_oid_atomic \
         (id int4 PRIMARY KEY, code int4 UNIQUE)",
    )
    .unwrap() else {
        panic!("expected CREATE TABLE");
    };
    let error = engine
        .apply_create_table_legacy_replay(&mut catalog, create, 77, 0)
        .expect_err("a two-index legacy record must reject atomically at the domain edge");
    assert!(
        error.to_string().contains("GPU catalog Int4 domain"),
        "{error}"
    );
    assert_eq!(catalog.relational_next_oid, next_oid_before);
    assert_eq!(catalog.legacy_recovery_next_index_oid, MAX_CATALOG_OID);
    assert!(!catalog
        .relational_catalog
        .contains_key("legacy_max_oid_atomic"));
}

#[test]
fn repeated_nullable_index_key_is_rejected_before_identity_or_wal_and_recovers_absent() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            4_026_110,
            parsed("CREATE TABLE repeated_index_key (id int4, nullable_key int4)"),
        )
        .unwrap();
    let before = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();
    engine
        .submit_transaction(4_026_111, parsed("BEGIN"))
        .unwrap();
    let error = engine
        .submit_transaction(
            4_026_111,
            parsed(
                "CREATE UNIQUE INDEX repeated_nullable_index \
                 ON repeated_index_key (nullable_key, nullable_key)",
            ),
        )
        .expect_err("a repeated nullable key column must reject before device publication");
    assert!(
        error
            .to_string()
            .contains("column \"nullable_key\" appears twice in index definition"),
        "{error}"
    );
    let private = engine
        .transaction_snapshot_handle(4_026_111)
        .unwrap()
        .transaction_catalog();
    assert!(private.relational_catalog["repeated_index_key"]
        .indexes
        .is_empty());
    assert_eq!(private.relational_next_oid, before.relational_next_oid);
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine
        .submit_transaction(4_026_111, parsed("ROLLBACK"))
        .unwrap();
    assert!(engine.catalog_snapshot().same_contents(before.as_ref()));

    let implicit = engine
        .submit_transaction(
            4_026_112,
            parsed("CREATE TABLE repeated_implicit (a int4, UNIQUE (a, a))"),
        )
        .expect_err("implicit constraint indexes share the duplicate-column guard");
    assert!(
        implicit
            .to_string()
            .contains("column \"a\" appears twice in index definition"),
        "{implicit}"
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

#[test]
fn transactional_create_table_binds_shared_pg_class_oid_highwater_and_recovers() {
    let source = Engine::new_local();
    let allocator_before = source.catalog_snapshot().relational_next_oid;
    let prefix = source.durable_wal_records();
    source.submit_transaction(4_029, parsed("BEGIN")).unwrap();
    source
        .submit_transaction(
            4_029,
            parsed(
                "CREATE TABLE implicit_index_identity \
                 (id int4 PRIMARY KEY, code int4 UNIQUE)",
            ),
        )
        .unwrap();
    source
        .submit_transaction(
            4_029,
            parsed(
                "CREATE INDEX implicit_index_ephemeral \
                 ON implicit_index_identity (code)",
            ),
        )
        .unwrap();
    source
        .submit_transaction(
            4_029,
            parsed(
                "ALTER INDEX implicit_index_ephemeral \
                 RENAME TO implicit_index_ephemeral_renamed",
            ),
        )
        .unwrap();
    source
        .submit_transaction(
            4_029,
            parsed(
                "DROP INDEX IF EXISTS implicit_index_absent, \
                 implicit_index_ephemeral_renamed",
            ),
        )
        .unwrap();
    source.submit_transaction(4_029, parsed("COMMIT")).unwrap();

    let committed = source.catalog_snapshot();
    let table = &committed.relational_catalog["implicit_index_identity"];
    assert_eq!(
        table
            .indexes
            .iter()
            .map(|index| index.oid)
            .collect::<Vec<_>>(),
        vec![allocator_before + 1, allocator_before + 2]
    );
    assert_eq!(committed.relational_next_oid, allocator_before + 4);

    let payload = operation_payload(source.durable_wal_records().last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("transactional CREATE TABLE must use one typed record");
    };
    assert_eq!(record.index_lifecycle_operations.len(), 3);
    assert_eq!(
        record
            .index_lifecycle_operations
            .iter()
            .map(|operation| (
                operation.command_index,
                operation.ordinal,
                operation.targets.len()
            ))
            .collect::<Vec<_>>(),
        vec![(1, 1, 1), (2, 2, 1), (3, 3, 2)]
    );
    assert!(matches!(
        record.index_lifecycle_operations[2].targets[0],
        BinaryTransactionIndexLifecycleTargetIdentity {
            index_before: None,
            index_after: None,
            ..
        }
    ));
    assert_eq!(
        record.operation_order,
        vec![
            BinaryTransactionOperationIdentity::Catalog { command_index: 0 },
            BinaryTransactionOperationIdentity::Catalog { command_index: 1 },
            BinaryTransactionOperationIdentity::Catalog { command_index: 2 },
            BinaryTransactionOperationIdentity::Catalog { command_index: 3 },
        ]
    );
    assert_eq!(
        record.catalog_output.as_ref().unwrap().relational_next_oid,
        allocator_before + 4
    );

    let recovered = Engine::recover_from_durable_wal(&source.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(committed.as_ref()));

    let target = Engine::recover_from_durable_wal(&prefix).unwrap();
    let before = target.catalog_snapshot();
    let entry = LogEntry {
        term: 1,
        index: before.commit_seq + 1,
        payload: Arc::from(&b""[..]),
    };
    let assert_rejected_without_effect = |tampered: BinaryTransactionRecord| {
        let mut catalog = target.ddl_catalog().clone();
        assert!(target
            .apply_binary_transaction_record(&entry, &mut catalog, tampered)
            .is_err());
        assert!(
            Engine::catalog_snapshot_from_working(&catalog, before.commit_seq)
                .same_contents(before.as_ref())
        );
    };

    let mut high_water = record.clone();
    high_water
        .catalog_output
        .as_mut()
        .unwrap()
        .relational_next_oid = u32::MAX;
    assert_rejected_without_effect(high_water);

    let mut swapped_implicit_oids = record.clone();
    swapped_implicit_oids
        .created_table_index_identities
        .get_mut("implicit_index_identity")
        .unwrap()
        .swap(0, 1);
    assert_rejected_without_effect(swapped_implicit_oids);

    let mut reordered = record.clone();
    reordered.operation_order.swap(1, 2);
    assert_rejected_without_effect(reordered);

    let mut wrong_statement_digest = record;
    wrong_statement_digest.statement_digests[1][0] ^= 0x5a;
    assert_rejected_without_effect(wrong_statement_digest);
}

#[test]
fn catalog_latch_captures_one_index_identity_or_serializes_the_racer() {
    let engine = Arc::new(Engine::new_local());
    engine
        .submit_transaction(
            4_030,
            parsed("CREATE TABLE index_latch (id int4, code int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_031,
            parsed("CREATE INDEX index_latch_code ON index_latch (code)"),
        )
        .unwrap();
    let original = index_named(
        &engine.relational_catalog_table("index_latch").unwrap(),
        "index_latch_code",
    )
    .clone();
    engine
        .submit_transaction(4_032, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();

    let (release_tx, release_rx) = mpsc::channel();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let racer_engine = Arc::clone(&engine);
    let racer = std::thread::spawn(move || {
        release_rx.recv().unwrap();
        started_tx.send(()).unwrap();
        done_tx
            .send(racer_engine.submit_transaction(4_033, parsed("DROP INDEX index_latch_code")))
            .unwrap();
    });

    let statement = parsed(
        "ALTER INDEX index_latch_code \
         RENAME TO index_latch_private",
    );
    let (command, source) = statement.into_parts();
    let hook_engine = Arc::clone(&engine);
    engine
        .execute_catalog_in_transaction_instrumented(4_032, command, source, None, || {
            assert!(matches!(
                hook_engine.catalog_latch.try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ));
            release_tx.send(()).unwrap();
            started_rx.recv().unwrap();
        })
        .unwrap();
    let raced = done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(matches!(raced, Err(ExecuteError::Serialization(_))));
    racer.join().unwrap();

    let snapshot = engine.transaction_snapshot_handle(4_032).unwrap();
    let delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let TransactionOperation::Catalog(staged) = &delta.operations[0] else {
        panic!("expected index catalog operation");
    };
    let target = &staged.index_identity.as_ref().unwrap().targets[0];
    assert_eq!(target.index_before.as_ref().unwrap().oid, original.oid);
    assert_eq!(target.index_after.as_ref().unwrap().oid, original.oid);
    drop(delta);
    engine
        .submit_transaction(4_032, parsed("ROLLBACK"))
        .unwrap();
}

#[test]
fn index_lifecycle_replay_tamper_is_clone_first_and_effect_free() {
    let source = Engine::new_local();
    source
        .submit_transaction(
            4_040,
            parsed("CREATE TABLE index_proof (id int4, code int4, region int4)"),
        )
        .unwrap();
    source
        .submit_transaction(
            4_041,
            parsed("CREATE INDEX index_proof_code ON index_proof (code)"),
        )
        .unwrap();
    let prefix = source.durable_wal_records();
    source.submit_transaction(4_042, parsed("BEGIN")).unwrap();
    source
        .submit_transaction(
            4_042,
            parsed("ALTER INDEX index_proof_code RENAME TO index_proof_renamed"),
        )
        .unwrap();
    source
        .submit_transaction(4_042, parsed("DROP INDEX index_proof_renamed"))
        .unwrap();
    source
        .submit_transaction(
            4_042,
            parsed("CREATE INDEX index_proof_final ON index_proof (code, region)"),
        )
        .unwrap();
    source.submit_transaction(4_042, parsed("COMMIT")).unwrap();
    let payload = operation_payload(source.durable_wal_records().last().unwrap());
    let BinaryWalRecord::Transaction(record) = decode_binary_record(&payload).unwrap() else {
        panic!("index lifecycle commit must decode");
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
            tampered.index_lifecycle_operations[0].targets[0]
                .table_before
                .as_mut()
                .unwrap()
                .oid += 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.index_lifecycle_operations[0].targets[0]
                .index_before
                .as_mut()
                .unwrap()
                .digest[0] ^= 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.index_lifecycle_operations[0].targets[0].owner_name =
                Some("smuggled_owner".to_string());
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.index_lifecycle_operations[1].targets[0]
                .table_after
                .as_mut()
                .unwrap()
                .digest[0] ^= 1;
            tampered
        },
        {
            let mut tampered = record.clone();
            tampered.index_lifecycle_operations[1].ordinal = 0;
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
        assert!(
            index_named(
                &catalog.relational_catalog["index_proof"],
                "index_proof_code"
            )
            .oid > 0
        );
        assert!(catalog.relational_catalog["index_proof"]
            .indexes
            .iter()
            .all(|index| index.name != "index_proof_final"));
    }
}

#[test]
fn post_durable_index_lifecycle_failure_is_recovery_owned() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            4_050,
            parsed("CREATE TABLE durable_index (id int4, code int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_051,
            parsed("CREATE INDEX durable_index_code ON durable_index (code)"),
        )
        .unwrap();
    let oid = index_named(
        &engine.relational_catalog_table("durable_index").unwrap(),
        "durable_index_code",
    )
    .oid;
    engine.submit_transaction(4_052, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_052,
            parsed("ALTER INDEX durable_index_code RENAME TO durable_index_renamed"),
        )
        .unwrap();
    engine.fail_next_transaction_post_durable_apply();
    let error = engine
        .submit_transaction(4_052, parsed("COMMIT"))
        .unwrap_err();
    assert!(error.is_indeterminate(), "{error}");

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    let table = recovered.relational_catalog_table("durable_index").unwrap();
    assert!(table
        .indexes
        .iter()
        .all(|index| index.name != "durable_index_code"));
    assert_eq!(index_named(&table, "durable_index_renamed").oid, oid);
}

/// PRODUCT-001 device proof: CREATE UNIQUE validates the transaction's unified private relation,
/// NULL-bearing compound keys are omitted, later private DML sees the staged index, rename preserves
/// its stable allocation identity, multi-target DROP retires every old identity, and recovery
/// reconstructs only the final catalog/index generation.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transactional_index_lifecycle_gpu_private_dml_null_residency_and_recovery() {
    let dropped = Engine::new_local();
    dropped.set_shard_residency_enabled(true);
    dropped.set_auto_admit_on_commit(true);
    dropped
        .submit_transaction(
            4_090,
            parsed("CREATE TABLE gpu_index_final_drop (id int4, code int4)"),
        )
        .unwrap();
    dropped
        .submit_transaction(
            4_091,
            parsed("INSERT INTO gpu_index_final_drop VALUES (1, 7)"),
        )
        .unwrap();
    dropped.submit_transaction(4_092, parsed("BEGIN")).unwrap();
    dropped
        .submit_transaction(
            4_092,
            parsed(
                "CREATE UNIQUE INDEX gpu_index_final_drop_code \
                 ON gpu_index_final_drop (code)",
            ),
        )
        .unwrap();
    dropped
        .submit_transaction(4_092, parsed("DROP INDEX gpu_index_final_drop_code"))
        .unwrap();
    dropped
        .submit_transaction(
            4_092,
            parsed("INSERT INTO gpu_index_final_drop VALUES (2, 7)"),
        )
        .expect("private DML after the ordered DROP must not retain the removed constraint");
    dropped.submit_transaction(4_092, parsed("COMMIT")).unwrap();
    assert!(dropped
        .relational_catalog_table("gpu_index_final_drop")
        .unwrap()
        .indexes
        .is_empty());
    assert!(!dropped
        .read_state
        .residency
        .named_index_coverage_complete
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .contains_key("gpu_index_final_drop"));
    assert!(dropped
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .keys()
        .all(|(table, _, _)| table != "gpu_index_final_drop"));
    let dropped_recovered =
        Engine::recover_from_durable_wal(&dropped.durable_wal_records()).unwrap();
    dropped_recovered
        .submit_transaction(
            4_093,
            parsed("INSERT INTO gpu_index_final_drop VALUES (3, 7)"),
        )
        .expect("recovery must retain the final absent-index constraint state");
    let dropped_rows = dropped_recovered
        .execute_relational_select(&select(
            "SELECT id, code FROM gpu_index_final_drop ORDER BY id",
        ))
        .unwrap();
    assert!(matches!(dropped_rows.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(dropped_rows.fallback_reason, None);
    assert_eq!(dropped_rows.rows.len(), 3);

    let rejected = Engine::new_local();
    rejected.set_shard_residency_enabled(true);
    rejected.set_auto_admit_on_commit(true);
    rejected
        .submit_transaction(
            4_100,
            parsed(
                "CREATE TABLE gpu_index_reject \
                 (id int4 PRIMARY KEY, code int4, region text)",
            ),
        )
        .unwrap();
    rejected
        .submit_transaction(
            4_101,
            parsed("INSERT INTO gpu_index_reject VALUES (1, 7, 'west')"),
        )
        .unwrap();
    rejected.submit_transaction(4_102, parsed("BEGIN")).unwrap();
    rejected
        .submit_transaction(
            4_102,
            parsed("INSERT INTO gpu_index_reject VALUES (2, 7, 'west')"),
        )
        .unwrap();
    let reject_wal = rejected.durable_wal_records().len();
    let error = rejected
        .submit_transaction(
            4_102,
            parsed(
                "CREATE UNIQUE INDEX gpu_index_reject_code_region \
                 ON gpu_index_reject (code, region)",
            ),
        )
        .expect_err("private duplicate must reject CREATE UNIQUE before WAL");
    assert!(
        matches!(error, ExecuteError::Engine(EngineError::UniqueViolation(_))),
        "{error}"
    );
    assert_eq!(rejected.durable_wal_records().len(), reject_wal);
    assert!(rejected
        .transaction_snapshot_handle(4_102)
        .unwrap()
        .transaction_catalog()
        .relational_catalog["gpu_index_reject"]
        .indexes
        .iter()
        .all(|index| index.name != "gpu_index_reject_code_region"));
    rejected
        .submit_transaction(4_102, parsed("ROLLBACK"))
        .unwrap();

    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.set_shard_size_target(64);
    engine
        .submit_transaction(
            4_110,
            parsed(
                "CREATE TABLE gpu_index_lifecycle \
                 (id int4 PRIMARY KEY, code int4, region text)",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_111,
            parsed(
                "INSERT INTO gpu_index_lifecycle VALUES \
                 (1, NULL, 'north'), (2, NULL, 'north'), (3, 7, 'west')",
            ),
        )
        .unwrap();

    engine.submit_transaction(4_112, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_112,
            parsed("INSERT INTO gpu_index_lifecycle VALUES (4, NULL, 'west')"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_112,
            parsed(
                "CREATE UNIQUE INDEX gpu_index_code_region \
                 ON gpu_index_lifecycle (code, region)",
            ),
        )
        .expect("multiple NULL compound keys are SQL-distinct");
    let private_catalog = engine
        .execute_resident_expr_select_sql_in_transaction(
            4_112,
            "SELECT relname FROM pg_catalog.pg_class \
             WHERE relname = 'gpu_index_code_region'",
        )
        .unwrap();
    assert!(matches!(
        private_catalog.executed_target,
        DeviceTarget::Gpu(_)
    ));
    assert_eq!(private_catalog.fallback_reason, None);
    assert_eq!(
        private_catalog.rows,
        vec![vec![SqlValue::Text("gpu_index_code_region".to_string())]]
    );
    assert!(engine
        .execute_resident_expr_select_sql(
            "SELECT relname FROM pg_catalog.pg_class \
             WHERE relname = 'gpu_index_code_region'",
        )
        .unwrap()
        .rows
        .is_empty());

    engine
        .submit_transaction(
            4_112,
            parsed(
                "INSERT INTO gpu_index_lifecycle VALUES \
                 (5, 8, 'south'), (6, NULL, 'south')",
            ),
        )
        .unwrap();
    let duplicate = engine
        .submit_transaction(
            4_112,
            parsed("INSERT INTO gpu_index_lifecycle VALUES (7, 8, 'south')"),
        )
        .expect_err("DML after CREATE must use the private unique index");
    assert!(
        matches!(
            duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{duplicate}"
    );
    let private_rows = engine
        .execute_relational_select_in_transaction(
            4_112,
            &select("SELECT id, code, region FROM gpu_index_lifecycle ORDER BY id"),
        )
        .unwrap();
    assert!(matches!(private_rows.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(private_rows.fallback_reason, None);
    assert_eq!(private_rows.rows.len(), 6);
    engine.submit_transaction(4_112, parsed("COMMIT")).unwrap();

    let created_table = engine
        .relational_catalog_table("gpu_index_lifecycle")
        .unwrap();
    let created_index = index_named(&created_table, "gpu_index_code_region").clone();
    let created_key = crate::engine_residency::index_probe_key_id(
        &created_table,
        &created_index,
        created_table
            .indexes
            .iter()
            .position(|index| index.oid == created_index.oid)
            .unwrap(),
    )
    .unwrap();
    let (created_shard, created_ptr) = {
        let cache = engine
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache
            .iter()
            .find_map(|((table, shard, key), entry)| {
                (table == "gpu_index_lifecycle" && *key == created_key)
                    .then(|| (*shard, entry.device_index.as_ref().unwrap().device_ptr()))
            })
            .unwrap_or_else(|| {
                panic!(
                    "missing created index key {created_key}; cache={:?}",
                    cache.keys().collect::<Vec<_>>()
                )
            })
    };

    engine.submit_transaction(4_113, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_113,
            parsed("ALTER INDEX gpu_index_code_region RENAME TO gpu_index_code_region_renamed"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_113,
            parsed(
                "CREATE INDEX gpu_index_region_code \
                 ON gpu_index_lifecycle (region, code)",
            ),
        )
        .unwrap();
    engine.submit_transaction(4_113, parsed("COMMIT")).unwrap();
    let renamed_table = engine
        .relational_catalog_table("gpu_index_lifecycle")
        .unwrap();
    let renamed = index_named(&renamed_table, "gpu_index_code_region_renamed");
    let auxiliary = index_named(&renamed_table, "gpu_index_region_code").clone();
    assert_eq!(renamed.oid, created_index.oid);
    let renamed_manifest = engine
        .read_state
        .residency
        .named_index_coverage_complete
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get("gpu_index_lifecycle")
        .cloned();
    assert_eq!(
        renamed_manifest,
        Some((renamed_table.oid, renamed_table.indexes.clone())),
        "rename/create publication must install the exact final catalog manifest"
    );
    let auxiliary_key = crate::engine_residency::COMPOUND_KEY_ID_FLAG | auxiliary.oid as usize;
    {
        let cache = engine
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(
            cache[&(
                "gpu_index_lifecycle".to_string(),
                created_shard,
                created_key,
            )]
                .device_index
                .as_ref()
                .unwrap()
                .device_ptr(),
            created_ptr,
            "rename retains the stable OID-keyed device allocation"
        );
        assert!(
            cache.contains_key(&(
                "gpu_index_lifecycle".to_string(),
                created_shard,
                auxiliary_key,
            )),
            "missing auxiliary key {auxiliary_key}; cache={:?}",
            cache.keys().collect::<Vec<_>>()
        );
    }

    engine.submit_transaction(4_114, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_114,
            parsed(
                "DROP INDEX IF EXISTS absent_gpu_index, \
                 gpu_index_code_region_renamed, gpu_index_region_code",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_114,
            parsed(
                "CREATE UNIQUE INDEX gpu_index_code_region \
                 ON gpu_index_lifecycle (code, region)",
            ),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_114,
            parsed("INSERT INTO gpu_index_lifecycle VALUES (8, 9, 'east')"),
        )
        .unwrap();
    engine.submit_transaction(4_114, parsed("COMMIT")).unwrap();

    let final_table = engine
        .relational_catalog_table("gpu_index_lifecycle")
        .unwrap();
    let final_index = index_named(&final_table, "gpu_index_code_region").clone();
    assert_ne!(final_index.oid, created_index.oid);
    assert_ne!(final_index.oid, auxiliary.oid);
    let final_key = crate::engine_residency::COMPOUND_KEY_ID_FLAG | final_index.oid as usize;
    {
        let cache = engine
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for retired in [created_key, auxiliary_key] {
            assert!(!cache.contains_key(&(
                "gpu_index_lifecycle".to_string(),
                created_shard,
                retired,
            )));
        }
        assert!(cache.contains_key(&("gpu_index_lifecycle".to_string(), created_shard, final_key,)));
    }
    assert_eq!(
        engine
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get("gpu_index_lifecycle")
            .cloned(),
        Some((final_table.oid, final_table.indexes.clone()))
    );

    let records = engine.durable_wal_records();
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let recovered_table = recovered
        .relational_catalog_table("gpu_index_lifecycle")
        .unwrap();
    assert_eq!(
        index_named(&recovered_table, "gpu_index_code_region").oid,
        final_index.oid
    );
    assert_eq!(
        recovered
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get("gpu_index_lifecycle")
            .cloned(),
        Some((recovered_table.oid, recovered_table.indexes.clone()))
    );
    let recovered_rows = recovered
        .execute_relational_select(&select(
            "SELECT id, code, region FROM gpu_index_lifecycle ORDER BY id",
        ))
        .unwrap();
    assert!(matches!(
        recovered_rows.executed_target,
        DeviceTarget::Gpu(_)
    ));
    assert_eq!(recovered_rows.fallback_reason, None);
    assert_eq!(recovered_rows.rows.len(), 7);
    recovered
        .submit_transaction(
            4_115,
            parsed("INSERT INTO gpu_index_lifecycle VALUES (9, NULL, 'east')"),
        )
        .expect("recovery preserves NULLS DISTINCT");
    let recovered_duplicate = recovered
        .submit_transaction(
            4_116,
            parsed("INSERT INTO gpu_index_lifecycle VALUES (10, 9, 'east')"),
        )
        .expect_err("recovered unique index must reject a present duplicate");
    assert!(
        matches!(
            recovered_duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{recovered_duplicate}"
    );
}

/// PRODUCT-001 cold-authority proof: CREATE UNIQUE must validate every retained chunk as one
/// device relation. A duplicate split across chunks rejects before WAL; a transaction-private
/// tombstone then removes that conflict, NULL-bearing keys remain SQL-distinct, and DML after the
/// CREATE consumes the staged unique index without deauthorizing or reconstructing host rows.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transactional_unique_index_validates_cold_chunks_and_private_tombstones() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            4_200,
            "CREATE TABLE cold_index_lifecycle \
             (id int4 PRIMARY KEY, code int4, tag text)",
        )
        .unwrap();
    let values = (0..1000)
        .map(|id| match id {
            250 | 750 => format!("({id}, NULL, 'nullable')"),
            900 => " (900, 17, 'tag-0017')".trim().to_string(),
            _ => format!("({id}, {id}, 'tag-{id:04}')"),
        })
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            4_201,
            &format!("INSERT INTO cold_index_lifecycle VALUES {values}"),
        )
        .unwrap();
    engine.set_relational_residency_budget_bytes(0, 8192);
    let count = engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM cold_index_lifecycle"))
        .unwrap();
    assert!(matches!(count.executed_target, DeviceTarget::Gpu(_)));
    engine
        .execute_text(
            4_202,
            "INSERT INTO cold_index_lifecycle VALUES (1000, 1000, 'enter')",
        )
        .unwrap();
    assert!(
        engine
            .table_chunk_authoritative("cold_index_lifecycle")
            .is_some(),
        "the fixture must enter chunk authority"
    );
    let global_cold = engine.read_streaming_cold_chunks();
    assert!(
        global_cold["cold_index_lifecycle"].chunks.len() > 2,
        "the duplicate fixture must span multiple retained chunks"
    );
    drop(global_cold);
    // The 8KiB admission cap above exists only to force chunk authority. Exact proof accounting
    // correctly refuses that lease before it can stage a complete cross-window pair, so give the
    // immutable cold validator a fitting but still bounded working set. The dedicated tiny-plan
    // sabotage below owns the effect-free ResourceExhausted + same-transaction retry contract.
    let proof_budget = 1024 * 1024;
    engine.set_relational_residency_budget_bytes(0, proof_budget);

    engine.submit_transaction(4_203, parsed("BEGIN")).unwrap();
    let wal_before = engine.durable_wal_records().len();
    let duplicate = engine
        .submit_transaction(
            4_203,
            parsed(
                "CREATE UNIQUE INDEX cold_index_code_tag \
                 ON cold_index_lifecycle (code, tag)",
            ),
        )
        .expect_err("a duplicate split across cold chunks must reject");
    assert!(
        matches!(
            duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{duplicate}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(engine
        .transaction_snapshot_handle(4_203)
        .unwrap()
        .transaction_catalog()
        .relational_catalog["cold_index_lifecycle"]
        .indexes
        .iter()
        .all(|index| index.name != "cold_index_code_tag"));

    engine
        .submit_transaction(
            4_203,
            parsed("DELETE FROM cold_index_lifecycle WHERE id = 900"),
        )
        .unwrap();
    let private_after_delete = engine
        .execute_relational_select_in_transaction(
            4_203,
            &select("SELECT id FROM cold_index_lifecycle WHERE id = 900"),
        )
        .unwrap();
    assert!(matches!(
        private_after_delete.executed_target,
        DeviceTarget::Gpu(_)
    ));
    assert!(
        private_after_delete.rows.is_empty(),
        "the transaction-private cold tombstone must mask the deleted duplicate"
    );
    engine
        .submit_transaction(
            4_203,
            parsed(
                "CREATE UNIQUE INDEX cold_index_code_tag \
                 ON cold_index_lifecycle (code, tag)",
            ),
        )
        .expect("the private tombstone and NULLS DISTINCT must make the relation unique");
    let validation_peak = engine
        .read_state
        .residency
        .cold_index_validation_peak_device_bytes
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        validation_peak > 0 && validation_peak <= proof_budget,
        "all sliced payloads, sidecars, unified input, and operator scratch must stay within the \
         configured budget: peak={validation_peak}"
    );
    engine
        .submit_transaction(
            4_203,
            parsed(
                "INSERT INTO cold_index_lifecycle VALUES \
                 (1001, NULL, 'nullable'), (1002, NULL, 'nullable'), \
                 (1003, 1003, 'post-create')",
            ),
        )
        .unwrap();
    let post_create_duplicate = engine
        .submit_transaction(
            4_203,
            parsed("INSERT INTO cold_index_lifecycle VALUES (1004, 17, 'tag-0017')"),
        )
        .expect_err("post-CREATE private DML must consume the staged unique index");
    assert!(
        matches!(
            post_create_duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{post_create_duplicate}"
    );
    // The transaction-private tombstone must not ABA-stamp the global cold authority before the
    // one WAL/publication owner installs the resolved record.
    let transaction = engine.transaction_snapshot_handle(4_203).unwrap();
    let deleted_entity = transaction
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .operations
        .iter()
        .find_map(|operation| {
            let TransactionOperation::Row(operation) = operation else {
                return None;
            };
            let PreparedMutation::Delete { table, .. } = &operation.delta.mutation else {
                return None;
            };
            operation
                .delta
                .write_set
                .rows
                .iter()
                .find(|row| &row.table == table)
                .and_then(|row| {
                    crate::engine_residency::parse_relational_row_id(
                        &row.row_key,
                        &relational_key_prefix(table),
                    )
                })
        })
        .expect("the private DELETE retains its stable entity identity");
    let global_before_commit = engine.read_streaming_cold_chunks();
    let global_entity_matches = global_before_commit["cold_index_lifecycle"]
        .chunks
        .iter()
        .flat_map(|chunk| chunk.entity_ids.iter())
        .filter(|&&entity_id| entity_id == deleted_entity)
        .count();
    let global_entity_bounds = global_before_commit["cold_index_lifecycle"]
        .chunks
        .iter()
        .flat_map(|chunk| chunk.entity_ids.iter().copied())
        .fold(None, |bounds, entity_id| match bounds {
            None => Some((entity_id, entity_id)),
            Some((min, max)) => Some((min.min(entity_id), max.max(entity_id))),
        });
    assert_eq!(
        global_entity_matches, 1,
        "delete entity={deleted_entity}, global bounds={global_entity_bounds:?}"
    );
    drop(global_before_commit);
    let private_table =
        transaction.transaction_catalog().relational_catalog["cold_index_lifecycle"].clone();
    assert!(
        engine
            .class_row_by_entity_identity(&private_table, deleted_entity, engine.committed_seq())
            .is_some(),
        "the global device generation must retain the transaction's delete target until COMMIT"
    );
    drop(transaction);
    engine.submit_transaction(4_203, parsed("COMMIT")).unwrap();
    assert_eq!(engine.chunk_class_deauths(), 0);
    assert!(
        engine
            .table_chunk_authoritative("cold_index_lifecycle")
            .is_some(),
        "transactional index publication must preserve chunk authority"
    );
    let table = engine
        .relational_catalog_table("cold_index_lifecycle")
        .unwrap();
    let index = index_named(&table, "cold_index_code_tag");
    assert!(index.unique);
    assert_eq!(index.key_columns, ["code", "tag"]);
    let published_duplicate = engine
        .execute_text(
            4_206,
            "INSERT INTO cold_index_lifecycle VALUES (1004, 17, 'tag-0017')",
        )
        .expect_err("cold publication must rebuild probes from the final catalog");
    assert!(
        matches!(
            published_duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{published_duplicate}"
    );
    assert_eq!(engine.chunk_class_deauths(), 0);

    let rows = engine
        .execute_relational_select(&select("SELECT id FROM cold_index_lifecycle ORDER BY id"))
        .unwrap();
    assert!(matches!(rows.executed_target, DeviceTarget::Gpu(_)));
    assert_eq!(rows.fallback_reason, None);
    assert_eq!(rows.rows.len(), 1003);
    assert!(rows
        .rows
        .iter()
        .all(|row| row.to_vec() != vec![SqlValue::Int4(900)]));

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    let recovered_table = recovered
        .relational_catalog_table("cold_index_lifecycle")
        .unwrap();
    assert_eq!(index_named(&recovered_table, "cold_index_code_tag"), index);
    recovered
        .submit_transaction(
            4_204,
            parsed("INSERT INTO cold_index_lifecycle VALUES (1004, NULL, 'nullable')"),
        )
        .expect("recovery must retain NULLS DISTINCT");
    let recovered_duplicate = recovered
        .submit_transaction(
            4_205,
            parsed("INSERT INTO cold_index_lifecycle VALUES (1005, 17, 'tag-0017')"),
        )
        .expect_err("recovery must rebuild the unique-index semantics");
    assert!(
        matches!(
            recovered_duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{recovered_duplicate}"
    );
}

/// PRODUCT-001 authority-selection sabotage: an ordinary store-authoritative relation may retain
/// an auxiliary cold cache beside its hot shards. Private DML changes only the transaction shard
/// generation, so CREATE UNIQUE must ignore the stale cache. Losing both retained representations
/// for an existing nonempty relation must fail closed rather than become a zero-row shortcut.
/// Once the hot generation is device-authoritative, losing its shards must also reject an ordinary
/// CREATE INDEX before WAL instead of falling back to the auxiliary cold cache.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn unique_index_prefers_private_hot_shards_over_auxiliary_cold_and_refuses_missing_authority() {
    struct RestoreChunkClassEntry(bool);
    impl Drop for RestoreChunkClassEntry {
        fn drop(&mut self) {
            crate::engine_streaming_exec::CHUNK_CLASS_ENTRY_ENABLED_TEST
                .store(self.0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let previous = crate::engine_streaming_exec::CHUNK_CLASS_ENTRY_ENABLED_TEST
        .swap(false, std::sync::atomic::Ordering::Relaxed);
    let _restore_chunk_class_entry = RestoreChunkClassEntry(previous);

    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            4_210,
            "CREATE TABLE index_authority_owner \
             (id int4 PRIMARY KEY, code int4)",
        )
        .unwrap();
    engine
        .execute_text(
            4_211,
            "INSERT INTO index_authority_owner VALUES (1, 11), (2, 12)",
        )
        .unwrap();
    engine.set_relational_residency_budget_bytes(0, 8192);
    let count = engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM index_authority_owner"))
        .unwrap();
    assert!(matches!(count.executed_target, DeviceTarget::Gpu(_)));
    let auxiliary_cold = engine
        .read_streaming_cold_chunks()
        .get("index_authority_owner")
        .cloned()
        .expect("the streaming read must retain an auxiliary cold cache");
    assert!(
        engine
            .table_chunk_authoritative("index_authority_owner")
            .is_none(),
        "a read cache is not representation authority"
    );

    engine.set_relational_residency_budget_bytes(0, u64::MAX);
    engine.clear_gpu_memory_pressured(0);
    engine.set_shard_residency_enabled(true);
    engine
        .populate_relational_residency_snapshot("index_authority_owner")
        .unwrap();
    assert!(engine
        .read_residency_shards()
        .get("index_authority_owner")
        .is_some_and(|shards| !shards.is_empty()));
    assert!(Arc::ptr_eq(
        engine
            .read_streaming_cold_chunks()
            .get("index_authority_owner")
            .expect("hot admission must not rewrite the auxiliary cache"),
        &auxiliary_cold,
    ));

    engine.submit_transaction(4_212, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_212,
            parsed("INSERT INTO index_authority_owner VALUES (3, 12)"),
        )
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    let duplicate = engine
        .submit_transaction(
            4_212,
            parsed(
                "CREATE UNIQUE INDEX index_authority_owner_code \
                 ON index_authority_owner (code)",
            ),
        )
        .expect_err("private hot duplicate must not be hidden by the stale auxiliary cache");
    assert!(
        matches!(
            duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{duplicate}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine
        .submit_transaction(4_212, parsed("ROLLBACK"))
        .unwrap();

    engine
        .submit_transaction(4_213, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    engine
        .execute_relational_select_in_transaction(
            4_213,
            &select("SELECT id FROM index_authority_owner WHERE id = 1"),
        )
        .unwrap();
    let snapshot = engine.transaction_snapshot_handle(4_213).unwrap();
    {
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut shards = (*delta.resident_shards).clone();
        shards.remove("index_authority_owner");
        delta.resident_shards = Arc::new(shards);
        let mut cold = (*delta.streaming_cold_chunks).clone();
        cold.remove("index_authority_owner");
        delta.streaming_cold_chunks = Arc::new(cold);
    }
    let missing = engine
        .submit_transaction(
            4_213,
            parsed(
                "CREATE UNIQUE INDEX index_authority_missing_code \
                 ON index_authority_owner (code)",
            ),
        )
        .expect_err("missing retained authority must not be interpreted as an empty relation");
    assert!(
        matches!(missing, ExecuteError::Serialization(_)),
        "{missing}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine
        .submit_transaction(4_213, parsed("ROLLBACK"))
        .unwrap();
    drop(snapshot);

    engine.set_table_device_authoritative("index_authority_owner", true);
    let removed = engine
        .read_state
        .residency
        .with_shards_mut_for_table("index_authority_owner", |shards| {
            shards.remove("index_authority_owner")
        });
    assert!(
        removed.is_some(),
        "the sabotage must remove the formerly authoritative hot generation"
    );
    assert!(engine.table_device_authoritative("index_authority_owner"));
    assert!(engine
        .read_residency_shards()
        .get("index_authority_owner")
        .is_none_or(Vec::is_empty));
    assert!(Arc::ptr_eq(
        engine
            .read_streaming_cold_chunks()
            .get("index_authority_owner")
            .expect("the stale auxiliary cold cache must remain present"),
        &auxiliary_cold,
    ));

    let lost_hot_wal_before = engine.durable_wal_records().len();
    engine.submit_transaction(4_214, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_214,
            parsed(
                "CREATE INDEX index_authority_lost_hot_code \
                 ON index_authority_owner (code)",
            ),
        )
        .expect("non-UNIQUE metadata may stage before physical publication preflight");
    let lost_hot = engine
        .submit_transaction(4_214, parsed("COMMIT"))
        .expect_err("stale cold cache must not replace lost device-authoritative shards");
    assert!(
        matches!(lost_hot, ExecuteError::Serialization(_)),
        "{lost_hot}"
    );
    assert_eq!(
        engine.durable_wal_records().len(),
        lost_hot_wal_before,
        "lost hot authority must be rejected before WAL"
    );
    assert!(
        engine
            .relational_catalog_table("index_authority_owner")
            .unwrap()
            .indexes
            .iter()
            .all(|index| index.name != "index_authority_lost_hot_code"),
        "failed publication preflight must not leak catalog metadata"
    );
    engine
        .submit_transaction(4_214, parsed("ROLLBACK"))
        .unwrap();
}

/// PRODUCT-001 retained-generation sabotage: neither a locally valid strict subset nor descriptor
/// cardinality is authority over the transaction's exact shard generation. Omitting one side of a
/// cross-shard duplicate and tearing a descriptor away from its allocation header must both fail
/// before the one-row shortcut, leave no catalog/WAL effect, and permit a same-transaction retry
/// after restoring the exact shard generation and removing the real duplicate through private GPU
/// DML.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn unique_index_rejects_torn_hot_row_count_before_shortcut_and_retries() {
    let mut engine = Engine::new_local_test_engine();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(1);
    engine
        .execute_text(
            4_215,
            "CREATE TABLE torn_hot_index_owner \
             (id int4 PRIMARY KEY, code int4)",
        )
        .unwrap();
    engine
        .execute_text(4_216, "INSERT INTO torn_hot_index_owner VALUES (1, 19)")
        .unwrap();
    engine
        .populate_relational_residency_snapshot("torn_hot_index_owner")
        .unwrap();
    engine
        .execute_text(
            4_217,
            "INSERT INTO torn_hot_index_owner VALUES (2, 20), (3, 19)",
        )
        .unwrap();
    let resident_shards = engine.read_residency_shards();
    assert!(
        resident_shards["torn_hot_index_owner"]
            .iter()
            .filter(|shard| shard.row_count != 0)
            .count()
            >= 2,
        "the duplicate must span distinct resident shards: {:?}",
        resident_shards["torn_hot_index_owner"]
    );

    engine
        .submit_transaction(4_218, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    engine
        .execute_relational_select_in_transaction(
            4_218,
            &select("SELECT id FROM torn_hot_index_owner WHERE id = 1"),
        )
        .unwrap();
    let snapshot = engine.transaction_snapshot_handle(4_218).unwrap();
    let intact_shards = {
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let intact = Arc::clone(&delta.resident_shards);
        let mut omitted = (*intact).clone();
        let shards = omitted
            .get_mut("torn_hot_index_owner")
            .expect("fixture must retain the hot generation");
        let omitted_at = shards
            .iter()
            .position(|shard| shard.row_count != 0)
            .expect("fixture must retain a nonempty shard");
        shards.remove(omitted_at);
        assert!(shards.iter().any(|shard| shard.row_count != 0));
        delta.resident_shards = Arc::new(omitted);
        intact
    };

    let wal_before = engine.durable_wal_records().len();
    let omitted = engine
        .submit_transaction(
            4_218,
            parsed(
                "CREATE UNIQUE INDEX torn_hot_index_code \
                 ON torn_hot_index_owner (code)",
            ),
        )
        .expect_err("a valid strict subset is not the proven transaction generation");
    assert!(
        matches!(omitted, ExecuteError::Serialization(_)),
        "{omitted}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(
        snapshot.transaction_catalog().relational_catalog["torn_hot_index_owner"]
            .indexes
            .iter()
            .all(|index| index.name != "torn_hot_index_code")
    );

    {
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        delta.resident_shards = Arc::clone(&intact_shards);
        let mut torn = (*intact_shards).clone();
        let shard = torn
            .get_mut("torn_hot_index_owner")
            .and_then(|shards| shards.iter_mut().find(|shard| shard.row_count == 1))
            .expect("fixture must retain a one-row device shard");
        shard.row_count = 0;
        delta.resident_shards = Arc::new(torn);
    }
    let torn = engine
        .submit_transaction(
            4_218,
            parsed(
                "CREATE UNIQUE INDEX torn_hot_index_code \
                 ON torn_hot_index_owner (code)",
            ),
        )
        .expect_err("descriptor row count must agree with the device header");
    assert!(matches!(torn, ExecuteError::Serialization(_)), "{torn}");
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(
        snapshot.transaction_catalog().relational_catalog["torn_hot_index_owner"]
            .indexes
            .iter()
            .all(|index| index.name != "torn_hot_index_code")
    );

    snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .resident_shards = intact_shards;
    engine
        .submit_transaction(
            4_218,
            parsed("DELETE FROM torn_hot_index_owner WHERE id = 3"),
        )
        .unwrap();
    engine
        .submit_transaction(
            4_218,
            parsed(
                "CREATE UNIQUE INDEX torn_hot_index_code \
                 ON torn_hot_index_owner (code)",
            ),
        )
        .expect("the restored authority plus private delete must be retryable");
    engine.submit_transaction(4_218, parsed("COMMIT")).unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    assert!(engine
        .relational_catalog_table("torn_hot_index_owner")
        .unwrap()
        .indexes
        .iter()
        .any(|index| index.name == "torn_hot_index_code"));
    drop(snapshot);

    let duplicate = engine
        .submit_transaction(
            4_219,
            parsed("INSERT INTO torn_hot_index_owner VALUES (4, 19)"),
        )
        .expect_err("the retried UNIQUE index must govern later DML");
    assert!(
        matches!(
            duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{duplicate}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cold_unique_budget_failure_is_effect_free_and_retryable_in_the_same_transaction() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            4_220,
            "CREATE TABLE cold_index_retry (id int4 PRIMARY KEY, code int4)",
        )
        .unwrap();
    let values = (0..600)
        .map(|id| format!("({id}, {id})"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            4_221,
            &format!("INSERT INTO cold_index_retry VALUES {values}"),
        )
        .unwrap();
    engine.set_relational_residency_budget_bytes(0, 8192);
    engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM cold_index_retry"))
        .unwrap();
    engine
        .execute_text(4_222, "INSERT INTO cold_index_retry VALUES (600, 600)")
        .unwrap();
    assert!(engine
        .table_chunk_authoritative("cold_index_retry")
        .is_some());

    engine.set_relational_residency_budget_bytes(0, 512);
    engine.submit_transaction(4_223, parsed("BEGIN")).unwrap();
    let wal_before = engine.durable_wal_records().len();
    let catalog_before = engine
        .transaction_snapshot_handle(4_223)
        .unwrap()
        .transaction_catalog();
    let error = engine
        .submit_transaction(
            4_223,
            parsed("CREATE UNIQUE INDEX cold_index_retry_code ON cold_index_retry (code)"),
        )
        .expect_err("one-row device geometry cannot fit the deliberately tiny budget");
    assert!(
        matches!(error, ExecuteError::ResourceExhausted(_)),
        "{error}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    let after_failure = engine
        .transaction_snapshot_handle(4_223)
        .unwrap()
        .transaction_catalog();
    assert!(after_failure.same_contents(catalog_before.as_ref()));
    assert_eq!(
        after_failure.relational_next_oid,
        catalog_before.relational_next_oid
    );

    // The tiny 512-byte refusal owns the failure assertion. Retry with enough headroom for the
    // complete cross-window proof; the 8 KiB value above exists only to establish cold authority.
    engine.set_relational_residency_budget_bytes(0, 1024 * 1024);
    engine
        .submit_transaction(
            4_223,
            parsed("CREATE UNIQUE INDEX cold_index_retry_code ON cold_index_retry (code)"),
        )
        .expect("the same transaction can retry from unchanged typed authority");
    engine.submit_transaction(4_223, parsed("COMMIT")).unwrap();
    assert!(
        index_named(
            &engine.relational_catalog_table("cold_index_retry").unwrap(),
            "cold_index_retry_code"
        )
        .unique
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cold_unique_tiny_tile_plan_is_bounded_effect_free_and_retryable() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            4_224,
            "CREATE TABLE bounded_cold_index (id int4 PRIMARY KEY, code int4)",
        )
        .unwrap();
    let values = (0..64)
        .map(|id| format!("({id}, {id})"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            4_225,
            &format!("INSERT INTO bounded_cold_index VALUES {values}"),
        )
        .unwrap();
    engine.set_relational_residency_budget_bytes(0, 8192);
    engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM bounded_cold_index"))
        .unwrap();
    engine
        .execute_text(4_226, "INSERT INTO bounded_cold_index VALUES (64, 64)")
        .unwrap();
    assert!(engine
        .table_chunk_authoritative("bounded_cold_index")
        .is_some());

    engine
        .read_state
        .residency
        .cold_index_validation_batch_rows_override
        .store(1, std::sync::atomic::Ordering::Relaxed);
    engine
        .read_state
        .residency
        .cold_index_validation_max_batches_override
        .store(2, std::sync::atomic::Ordering::Relaxed);
    engine
        .read_state
        .residency
        .cold_index_validation_peak_planned_windows
        .store(0, std::sync::atomic::Ordering::Relaxed);
    engine
        .read_state
        .residency
        .cold_index_validation_subset_proofs
        .store(0, std::sync::atomic::Ordering::Relaxed);

    engine.submit_transaction(4_227, parsed("BEGIN")).unwrap();
    let snapshot = engine.transaction_snapshot_handle(4_227).unwrap();
    let catalog_before = snapshot.transaction_catalog();
    let wal_before = engine.durable_wal_records().len();
    let bounded = engine
        .submit_transaction(
            4_227,
            parsed(
                "CREATE UNIQUE INDEX bounded_cold_index_code \
                 ON bounded_cold_index (code)",
            ),
        )
        .expect_err("the forced tiny-tile plan must refuse before unbounded framing or work");
    assert!(
        matches!(bounded, ExecuteError::ResourceExhausted(_)),
        "{bounded}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(snapshot
        .transaction_catalog()
        .same_contents(catalog_before.as_ref()));
    assert!(
        engine
            .read_state
            .residency
            .cold_index_validation_peak_planned_windows
            .load(std::sync::atomic::Ordering::Relaxed)
            <= 3
    );
    assert_eq!(
        engine
            .read_state
            .residency
            .cold_index_validation_subset_proofs
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "bounded planning must refuse before launching a quadratic proof prefix"
    );

    engine
        .read_state
        .residency
        .cold_index_validation_batch_rows_override
        .store(0, std::sync::atomic::Ordering::Relaxed);
    engine
        .read_state
        .residency
        .cold_index_validation_max_batches_override
        .store(0, std::sync::atomic::Ordering::Relaxed);
    engine.set_relational_residency_budget_bytes(0, 1024 * 1024);
    engine
        .submit_transaction(
            4_227,
            parsed(
                "CREATE UNIQUE INDEX bounded_cold_index_code \
                 ON bounded_cold_index (code)",
            ),
        )
        .expect("the unchanged transaction authority must retry with a bounded fitting plan");
    engine.submit_transaction(4_227, parsed("COMMIT")).unwrap();
    assert!(
        index_named(
            &engine
                .relational_catalog_table("bounded_cold_index")
                .unwrap(),
            "bounded_cold_index_code"
        )
        .unique
    );
    drop(snapshot);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn cold_unique_rejects_torn_private_chunk_generation_before_proof() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            4_228,
            "CREATE TABLE torn_cold_index (id int4 PRIMARY KEY, code int4)",
        )
        .unwrap();
    let values = (0..1000)
        .map(|id| {
            let code = if id == 0 || id == 999 { 77 } else { id + 1000 };
            format!("({id}, {code})")
        })
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            4_229,
            &format!("INSERT INTO torn_cold_index VALUES {values}"),
        )
        .unwrap();
    engine.set_relational_residency_budget_bytes(0, 8192);
    engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM torn_cold_index"))
        .unwrap();
    engine
        .execute_text(4_230, "INSERT INTO torn_cold_index VALUES (1000, 3000)")
        .unwrap();
    assert!(engine
        .table_chunk_authoritative("torn_cold_index")
        .is_some());
    // The 8 KiB cap exists only to force cold-class entry. Generation sabotage, rather than the
    // separately covered bounded-plan refusal, owns this test; give the complete validation proof
    // enough headroom to reach its cross-chunk duplicate verdict.
    engine.set_relational_residency_budget_bytes(0, 1024 * 1024);

    engine.submit_transaction(4_231, parsed("BEGIN")).unwrap();
    let snapshot = engine.transaction_snapshot_handle(4_231).unwrap();
    let intact_cold = {
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let intact = Arc::clone(&delta.streaming_cold_chunks);
        let entry = intact
            .get("torn_cold_index")
            .expect("fixture must retain private cold authority");
        assert!(
            entry.chunks.len() > 2,
            "the duplicate endpoints must span retained chunks"
        );
        let torn_entry = Engine::clone_cold_without_chunk_for_test(entry, 0);
        let mut torn = (*intact).clone();
        torn.insert("torn_cold_index".to_string(), torn_entry);
        delta.streaming_cold_chunks = Arc::new(torn);
        intact
    };

    let wal_before = engine.durable_wal_records().len();
    let torn = engine
        .submit_transaction(
            4_231,
            parsed(
                "CREATE UNIQUE INDEX torn_cold_index_code \
                 ON torn_cold_index (code)",
            ),
        )
        .expect_err("a self-consistent strict chunk subset is not the proven private generation");
    assert!(matches!(torn, ExecuteError::Serialization(_)), "{torn}");
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(
        snapshot.transaction_catalog().relational_catalog["torn_cold_index"]
            .indexes
            .iter()
            .all(|index| index.name != "torn_cold_index_code")
    );

    snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .streaming_cold_chunks = intact_cold;
    let intact_duplicate = engine
        .submit_transaction(
            4_231,
            parsed(
                "CREATE UNIQUE INDEX torn_cold_index_code \
                 ON torn_cold_index (code)",
            ),
        )
        .expect_err("restored complete authority must rediscover the cross-chunk duplicate");
    assert!(
        matches!(
            intact_duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "{intact_duplicate}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine
        .submit_transaction(4_231, parsed("ROLLBACK"))
        .unwrap();
    drop(snapshot);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn hot_unique_budget_counts_only_owner_inputs_not_unrelated_residency() {
    let mut engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            4_230,
            "CREATE TABLE hot_index_owner (id int4 PRIMARY KEY, code int4)",
        )
        .unwrap();
    let owner_values = (0..128)
        .map(|id| format!("({id}, {id})"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            4_231,
            &format!("INSERT INTO hot_index_owner VALUES {owner_values}"),
        )
        .unwrap();
    engine
        .execute_text(
            4_232,
            "CREATE TABLE hot_index_unrelated \
             (id int4 PRIMARY KEY, a int4, b int4, c int4, d int4)",
        )
        .unwrap();
    let unrelated_values = (0..20_000)
        .map(|id| format!("({id}, {id}, {id}, {id}, {id})"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            4_233,
            &format!("INSERT INTO hot_index_unrelated VALUES {unrelated_values}"),
        )
        .unwrap();

    engine.submit_transaction(4_234, parsed("BEGIN")).unwrap();
    let snapshot = engine.transaction_snapshot_handle(4_234).unwrap();
    let shards = snapshot.transaction_shards();
    let owner_input =
        super::index_identity::pinned_index_validation_input_bytes(&shards["hot_index_owner"], 0)
            .unwrap();
    let unrelated_input = super::index_identity::pinned_index_validation_input_bytes(
        &shards["hot_index_unrelated"],
        0,
    )
    .unwrap();
    let scratch_headroom = 256 * 1024;
    let budget = owner_input + scratch_headroom;
    assert!(
        owner_input + unrelated_input > budget,
        "the unrelated resident operand must exceed the configured headroom"
    );
    drop(snapshot);
    engine.set_relational_residency_budget_bytes(0, budget);
    engine
        .read_state
        .residency
        .cold_index_validation_peak_device_bytes
        .store(0, std::sync::atomic::Ordering::Relaxed);
    engine
        .submit_transaction(
            4_234,
            parsed("CREATE UNIQUE INDEX hot_index_owner_code ON hot_index_owner (code)"),
        )
        .expect("unrelated resident allocations are not operands of this proof");
    let scoped_peak = engine
        .read_state
        .residency
        .cold_index_validation_peak_device_bytes
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        scoped_peak > 0 && scoped_peak <= scratch_headroom,
        "query-local scratch peak {scoped_peak} exceeded headroom {scratch_headroom}"
    );
    assert!(
        engine.relational_resident_bytes_for_gpu(0) > budget,
        "the test must retain unrelated device allocations beyond the query-local budget"
    );
    // Restore a globally coherent admission budget before publication. The assertion above is
    // specifically about query-local validation operands; canonical index publication remains a
    // global residency transaction and must not overcommit the configured device budget.
    engine.set_relational_residency_budget_bytes(0, u64::MAX);
    engine.submit_transaction(4_234, parsed("COMMIT")).unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn unique_validation_does_not_hold_the_global_budget_lock_across_device_work() {
    let engine = Arc::new(Engine::new_local_test_engine());
    engine
        .execute_text(
            4_240,
            "CREATE TABLE index_validation_owner (id int4 PRIMARY KEY, code int4)",
        )
        .unwrap();
    let owner_values = (0..512)
        .map(|id| format!("({id}, {id})"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            4_241,
            &format!("INSERT INTO index_validation_owner VALUES {owner_values}"),
        )
        .unwrap();
    engine
        .execute_text(
            4_242,
            "CREATE TABLE index_validation_allocator (id int4 PRIMARY KEY, value int4)",
        )
        .unwrap();
    engine
        .execute_text(
            4_243,
            "INSERT INTO index_validation_allocator VALUES (1, 10), (2, 20)",
        )
        .unwrap();
    engine
        .read_state
        .residency
        .purge_shard_pk_index_for_table("index_validation_allocator");

    engine.submit_transaction(4_244, parsed("BEGIN")).unwrap();
    let reached = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    engine.set_index_validation_pause_hook(Arc::clone(&reached), Arc::clone(&resume));
    let validator = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            engine.submit_transaction(
                4_244,
                parsed(
                    "CREATE UNIQUE INDEX index_validation_owner_code \
                     ON index_validation_owner (code)",
                ),
            )
        })
    };
    reached.wait();

    let (allocated_tx, allocated_rx) = mpsc::channel();
    let allocator = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            allocated_tx
                .send(engine.publish_relational_resident_indexes("index_validation_allocator"))
                .unwrap();
        })
    };
    let allocation = allocated_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("unrelated named-index allocation blocked behind validation")
        .expect("unrelated named-index allocation succeeds while validation is paused");
    assert!(allocation
        .indexes
        .iter()
        .any(|entry| entry.index == "index_validation_allocator_pkey"));
    resume.wait();
    validator
        .join()
        .expect("validation thread")
        .expect("validation completes after the sabotage pause");
    allocator.join().expect("allocator thread");
    engine
        .submit_transaction(4_244, parsed("ROLLBACK"))
        .unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn count_only_unique_validation_reads_no_fake_value_column_for_narrow_bool_keys() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(4_245, "CREATE TABLE narrow_bool_unique (b bool)")
        .unwrap();
    let mut unique_values = (0..1024).map(|_| "(NULL)".to_string()).collect::<Vec<_>>();
    unique_values.extend(["(TRUE)".to_string(), "(FALSE)".to_string()]);
    engine
        .execute_text(
            4_246,
            &format!(
                "INSERT INTO narrow_bool_unique VALUES {}",
                unique_values.join(",")
            ),
        )
        .unwrap();
    let shards = engine.read_residency_shards();
    let narrow_bytes = shards["narrow_bool_unique"][0]
        .device_memory
        .as_ref()
        .unwrap()
        .metadata()
        .allocated_bytes;
    assert!(
        narrow_bytes < unique_values.len() as u64 * 4,
        "the fixture must be narrower than the removed fake int4 value bound"
    );
    drop(shards);
    engine.submit_transaction(4_247, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            4_247,
            parsed("CREATE UNIQUE INDEX narrow_bool_unique_b ON narrow_bool_unique (b)"),
        )
        .expect("NULLS DISTINCT plus TRUE/FALSE is unique on the count-only GPU path");
    engine.submit_transaction(4_247, parsed("COMMIT")).unwrap();

    engine
        .execute_text(4_248, "CREATE TABLE narrow_bool_duplicate (b bool)")
        .unwrap();
    let mut duplicate_values = (0..1024).map(|_| "(NULL)".to_string()).collect::<Vec<_>>();
    duplicate_values.extend(["(TRUE)".to_string(), "(TRUE)".to_string()]);
    engine
        .execute_text(
            4_249,
            &format!(
                "INSERT INTO narrow_bool_duplicate VALUES {}",
                duplicate_values.join(",")
            ),
        )
        .unwrap();
    engine.submit_transaction(4_250, parsed("BEGIN")).unwrap();
    let duplicate = engine
        .submit_transaction(
            4_250,
            parsed("CREATE UNIQUE INDEX narrow_bool_duplicate_b ON narrow_bool_duplicate (b)"),
        )
        .expect_err("the count-only GPU verdict must find the duplicate TRUE");
    assert!(
        matches!(
            duplicate,
            ExecuteError::Engine(EngineError::UniqueViolation(_))
        ),
        "narrow COUNT-only input returned the wrong error: {duplicate}"
    );
    engine
        .submit_transaction(4_250, parsed("ROLLBACK"))
        .unwrap();
}
