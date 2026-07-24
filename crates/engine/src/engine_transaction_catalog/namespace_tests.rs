use super::*;

#[test]
fn drop_index_if_exists_rejects_wrong_kind_atomically_in_both_live_paths() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            9_100,
            "CREATE TABLE drop_index_kind_owner (id int4, code int4)",
        )
        .unwrap();
    engine
        .execute_text(
            9_101,
            "CREATE INDEX drop_index_kind_target ON drop_index_kind_owner (code)",
        )
        .unwrap();
    engine
        .execute_text(9_102, "CREATE TABLE drop_index_kind_table (id int4)")
        .unwrap();
    engine
        .execute_text(
            9_103,
            "CREATE VIEW drop_index_kind_view AS \
             SELECT id FROM drop_index_kind_table",
        )
        .unwrap();
    engine
        .execute_text(9_104, "CREATE SEQUENCE drop_index_kind_sequence")
        .unwrap();
    let before = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();

    for (ordinal, wrong_kind) in [
        "drop_index_kind_table",
        "drop_index_kind_view",
        "drop_index_kind_sequence",
    ]
    .into_iter()
    .enumerate()
    {
        for (if_ordinal, if_exists) in ["", " IF EXISTS"].into_iter().enumerate() {
            let sql = format!("DROP INDEX{if_exists} drop_index_kind_target, {wrong_kind}");
            let autocommit = engine
                .execute_text(9_110 + ordinal as u64 * 10 + if_ordinal as u64, &sql)
                .expect_err(
                    "IF EXISTS suppresses only an absent name, never a wrong relation kind",
                );
            assert!(
                autocommit
                    .to_string()
                    .contains(&format!("relation \"{wrong_kind}\" is not an index")),
                "{autocommit}"
            );
            assert_eq!(engine.durable_wal_records().len(), wal_before);
            assert!(engine.catalog_snapshot().same_contents(before.as_ref()));

            let txn_id = 9_150 + ordinal as u64 * 10 + if_ordinal as u64;
            engine.submit_transaction(txn_id, parsed("BEGIN")).unwrap();
            let explicit = engine
                .submit_transaction(txn_id, parsed(&sql))
                .expect_err("the transaction-private path must share wrong-kind classification");
            assert!(
                explicit
                    .to_string()
                    .contains(&format!("relation \"{wrong_kind}\" is not an index")),
                "{explicit}"
            );
            let snapshot = engine.transaction_snapshot_handle(txn_id).unwrap();
            assert!(snapshot.transaction_delta_is_empty());
            assert!(snapshot
                .transaction_catalog()
                .same_contents(before.as_ref()));
            assert_eq!(engine.durable_wal_records().len(), wal_before);
            drop(snapshot);
            engine
                .submit_transaction(txn_id, parsed("ROLLBACK"))
                .unwrap();
        }
    }

    engine
        .execute_text(
            9_190,
            "DROP INDEX IF EXISTS absent_drop_index_kind, drop_index_kind_target",
        )
        .unwrap();
    assert!(
        engine.catalog_snapshot().relational_catalog["drop_index_kind_owner"]
            .indexes
            .iter()
            .all(|index| index.name != "drop_index_kind_target")
    );
}

#[test]
fn rename_index_destination_uses_the_complete_schema_relation_namespace() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            9_200,
            "CREATE TABLE rename_index_namespace_owner (id int4, code int4)",
        )
        .unwrap();
    engine
        .execute_text(
            9_201,
            "CREATE INDEX rename_index_namespace_source \
             ON rename_index_namespace_owner (code)",
        )
        .unwrap();
    engine
        .execute_text(9_202, "CREATE TABLE rename_index_namespace_table (id int4)")
        .unwrap();
    engine
        .execute_text(
            9_203,
            "CREATE VIEW rename_index_namespace_view AS \
             SELECT id FROM rename_index_namespace_table",
        )
        .unwrap();
    engine
        .execute_text(
            9_204,
            "CREATE MATERIALIZED VIEW rename_index_namespace_matview AS \
             SELECT id FROM rename_index_namespace_table WITH NO DATA",
        )
        .unwrap();
    engine
        .execute_text(9_205, "CREATE SEQUENCE rename_index_namespace_sequence")
        .unwrap();
    let before = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();

    for (ordinal, destination) in [
        "rename_index_namespace_table",
        "rename_index_namespace_view",
        "rename_index_namespace_matview",
        "rename_index_namespace_sequence",
    ]
    .into_iter()
    .enumerate()
    {
        let txn_id = 9_210 + ordinal as u64;
        engine.submit_transaction(txn_id, parsed("BEGIN")).unwrap();
        let error = engine
            .submit_transaction(
                txn_id,
                parsed(&format!(
                    "ALTER INDEX rename_index_namespace_source RENAME TO {destination}"
                )),
            )
            .expect_err("an index cannot shadow another schema relation family");
        assert!(
            error.to_string().contains(destination)
                && (error.to_string().contains("not absent")
                    || error.to_string().contains("already exists")),
            "{error}"
        );
        let private = engine.transaction_snapshot_handle(txn_id).unwrap();
        assert!(private.transaction_delta_is_empty());
        assert!(private.transaction_catalog().same_contents(before.as_ref()));
        drop(private);
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(
            engine.catalog_snapshot().relational_next_oid,
            before.relational_next_oid
        );
        engine
            .submit_transaction(txn_id, parsed("ROLLBACK"))
            .unwrap();
    }

    assert!(engine.catalog_snapshot().same_contents(before.as_ref()));
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered.catalog_snapshot().same_contents(before.as_ref()));
}

fn assert_failed_transaction_catalog_statement_is_effect_free(
    engine: &Engine,
    txn_id: TxnId,
    sql: &str,
    before: &CatalogSnapshot,
    wal_before: usize,
) {
    engine.submit_transaction(txn_id, parsed("BEGIN")).unwrap();
    let error = engine
        .submit_transaction(txn_id, parsed(sql))
        .expect_err("shared pg_class namespace conflict must reject");
    let private = engine.transaction_snapshot_handle(txn_id).unwrap();
    assert!(
        private.transaction_delta_is_empty(),
        "{sql:?} left a private operation after {error}"
    );
    assert!(private.transaction_catalog().same_contents(before));
    drop(private);
    assert!(engine.catalog_snapshot().same_contents(before));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine
        .submit_transaction(txn_id, parsed("ROLLBACK"))
        .unwrap();
}

#[test]
fn current_table_implicit_index_and_view_names_share_one_pg_class_authority() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            9_300,
            "CREATE TABLE current_namespace_owner (id int4, code int4)",
        )
        .unwrap();
    engine
        .execute_text(
            9_301,
            "CREATE INDEX current_namespace_index \
             ON current_namespace_owner (code)",
        )
        .unwrap();
    engine
        .execute_text(9_302, "CREATE TABLE current_namespace_table (id int4)")
        .unwrap();
    engine
        .execute_text(
            9_303,
            "CREATE VIEW current_namespace_view AS \
             SELECT id FROM current_namespace_owner",
        )
        .unwrap();
    let before = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();
    let autocommit = engine
        .execute_text(
            9_304,
            "ALTER TABLE current_namespace_table RENAME TO current_namespace_index",
        )
        .expect_err("a table rename cannot shadow an index relation");
    assert!(
        autocommit
            .to_string()
            .contains("relation \"current_namespace_index\" already exists"),
        "{autocommit}"
    );
    assert!(engine.catalog_snapshot().same_contents(before.as_ref()));
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    for (ordinal, sql) in [
        "CREATE TABLE current_namespace_index (id int4)",
        "CREATE TABLE implicit_self_shadow \
         (id int4, CONSTRAINT implicit_self_shadow UNIQUE (id))",
        "CREATE TABLE implicit_table_shadow \
         (id int4, CONSTRAINT current_namespace_table UNIQUE (id))",
        "CREATE TABLE implicit_index_shadow \
         (id int4, CONSTRAINT current_namespace_index UNIQUE (id))",
        "CREATE VIEW current_namespace_index AS \
         SELECT id FROM current_namespace_owner",
        "ALTER VIEW current_namespace_view RENAME TO current_namespace_index",
        "ALTER TABLE current_namespace_table RENAME TO current_namespace_index",
        "DROP VIEW IF EXISTS current_namespace_index",
    ]
    .into_iter()
    .enumerate()
    {
        assert_failed_transaction_catalog_statement_is_effect_free(
            &engine,
            9_310 + ordinal as u64,
            sql,
            before.as_ref(),
            wal_before,
        );
    }

    assert_eq!(
        engine.catalog_snapshot().relational_next_oid,
        before.relational_next_oid
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered.catalog_snapshot().same_contents(before.as_ref()));
}

#[test]
fn current_sequence_and_materialized_view_commands_cannot_shadow_an_index() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            9_350,
            "CREATE TABLE reverse_namespace_owner (id int4, code int4)",
        )
        .unwrap();
    engine
        .execute_text(
            9_351,
            "CREATE INDEX reverse_namespace_index \
             ON reverse_namespace_owner (code)",
        )
        .unwrap();
    engine
        .execute_text(9_352, "CREATE SEQUENCE reverse_namespace_sequence")
        .unwrap();
    engine
        .execute_text(
            9_353,
            "CREATE MATERIALIZED VIEW reverse_namespace_matview AS \
             SELECT id FROM reverse_namespace_owner WITH NO DATA",
        )
        .unwrap();
    let before = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();

    for (ordinal, sql) in [
        "CREATE SEQUENCE reverse_namespace_index",
        "ALTER SEQUENCE reverse_namespace_sequence \
         RENAME TO reverse_namespace_index",
        "DROP SEQUENCE IF EXISTS reverse_namespace_index",
        "CREATE MATERIALIZED VIEW reverse_namespace_index AS \
         SELECT id FROM reverse_namespace_owner WITH NO DATA",
        "ALTER MATERIALIZED VIEW reverse_namespace_matview \
         RENAME TO reverse_namespace_index",
        "REFRESH MATERIALIZED VIEW reverse_namespace_index",
        "DROP MATERIALIZED VIEW IF EXISTS reverse_namespace_index",
    ]
    .into_iter()
    .enumerate()
    {
        let error = engine
            .execute_text(9_360 + ordinal as u64, sql)
            .expect_err("a current pg_class family must not select or shadow an index");
        assert!(
            error.to_string().contains("already exists")
                || error.to_string().contains("not a sequence")
                || error.to_string().contains("not a materialized view"),
            "{sql:?}: {error}"
        );
        assert!(engine.catalog_snapshot().same_contents(before.as_ref()));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
    }

    assert_eq!(
        engine
            .catalog_snapshot()
            .pg_class_relation_kind("reverse_namespace_index")
            .unwrap(),
        Some(PgClassRelationKind::Index)
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered.catalog_snapshot().same_contents(before.as_ref()));
}

#[test]
fn reverse_family_oid_exhaustion_rejects_before_wal_and_without_partial_catalog_state() {
    let engine = Engine::new_local();
    engine
        .execute_text(9_380, "CREATE TABLE reverse_oid_owner (id int4, code int4)")
        .unwrap();
    seed_shared_pg_class_oid_for_boundary_test(&engine, MAX_CATALOG_OID);
    engine
        .execute_text(
            9_381,
            "CREATE INDEX reverse_oid_boundary ON reverse_oid_owner (code)",
        )
        .unwrap();
    assert_eq!(
        index_named(
            &engine.catalog_snapshot().relational_catalog["reverse_oid_owner"],
            "reverse_oid_boundary",
        )
        .oid,
        MAX_CATALOG_OID
    );
    let before = engine.catalog_snapshot();
    let wal_before = engine.durable_wal_records().len();

    for (ordinal, sql) in [
        "CREATE SEQUENCE reverse_oid_exhausted_sequence",
        "CREATE MATERIALIZED VIEW reverse_oid_exhausted_matview AS \
         SELECT id FROM reverse_oid_owner WITH NO DATA",
    ]
    .into_iter()
    .enumerate()
    {
        let error = engine
            .execute_text(9_382 + ordinal as u64, sql)
            .expect_err("shared pg_class OID exhaustion must reject before WAL");
        assert!(
            error.to_string().contains("OID allocation exhausted"),
            "{sql:?}: {error}"
        );
        assert!(engine.catalog_snapshot().same_contents(before.as_ref()));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
    }
}

#[test]
fn legacy_sequence_index_shadow_defaults_replay_then_current_epoch_fails_closed() {
    let legacy_records = [
        gpu_db_wal::WalRecord {
            txn_id: 9_700,
            payload: Arc::from(
                &b"CREATE TABLE legacy_sequence_index_owner \
                   (id int4, CONSTRAINT legacy_sequence_index_shadow UNIQUE (id))"[..],
            ),
        },
        gpu_db_wal::WalRecord {
            txn_id: 9_701,
            payload: Arc::from(&b"CREATE SEQUENCE legacy_sequence_index_shadow"[..]),
        },
        gpu_db_wal::WalRecord {
            txn_id: 9_702,
            payload: Arc::from(
                &b"CREATE TABLE legacy_sequence_defaults \
                   (id int4, bucket int4 DEFAULT \
                    nextval('legacy_sequence_index_shadow'::regclass))"[..],
            ),
        },
        gpu_db_wal::WalRecord {
            txn_id: 9_703,
            payload: Arc::from(&b"INSERT INTO legacy_sequence_defaults (id) VALUES (1)"[..]),
        },
        gpu_db_wal::WalRecord {
            txn_id: 9_704,
            payload: Arc::from(
                &b"ALTER TABLE legacy_sequence_defaults ADD COLUMN extra int4 \
                   DEFAULT nextval('legacy_sequence_index_shadow'::regclass)"[..],
            ),
        },
        gpu_db_wal::WalRecord {
            txn_id: 9_705,
            payload: Arc::from(
                &b"ALTER TABLE legacy_sequence_defaults ALTER COLUMN bucket \
                   SET DEFAULT nextval('legacy_sequence_index_shadow'::regclass)"[..],
            ),
        },
    ];
    let engine = Engine::recover_from_durable_wal(&legacy_records).unwrap();
    let legacy = engine.catalog_snapshot();
    assert!(legacy.relational_catalog["legacy_sequence_index_owner"]
        .indexes
        .iter()
        .any(|index| index.name == "legacy_sequence_index_shadow"));
    let sequence = &legacy.relational_sequences["legacy_sequence_index_shadow"];
    assert_eq!(sequence.last_value, 2);
    assert!(sequence.is_called);
    assert!(legacy
        .pg_class_relation_kind("legacy_sequence_index_shadow")
        .is_err());

    engine
        .execute_text(
            9_706,
            "CREATE INDEX legacy_sequence_epoch_boundary \
             ON legacy_sequence_defaults (id)",
        )
        .unwrap();
    let complete = engine.durable_wal_records();
    let expected = engine.catalog_snapshot();
    let recovered = Engine::recover_from_durable_wal(&complete).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(expected.as_ref()));

    let split = complete.len() - 1;
    let chunked = Engine::new_local();
    chunked
        .prepare_legacy_index_oid_recovery(&complete)
        .unwrap();
    chunked.begin_recovery_replay();
    chunked.replay_durable_records(&complete[..split]).unwrap();
    chunked.replay_durable_records(&complete[split..]).unwrap();
    chunked.finish_recovery_replay().unwrap();
    assert!(chunked.catalog_snapshot().same_contents(expected.as_ref()));

    let before_current = recovered.catalog_snapshot();
    let error = recovered
        .execute_text(
            9_707,
            "SELECT nextval('legacy_sequence_index_shadow'::regclass)",
        )
        .expect_err("current sequence resolution must not guess through legacy ambiguity");
    assert!(
        error.to_string().contains("multiple pg_class relations"),
        "{error}"
    );
    assert!(recovered
        .catalog_snapshot()
        .same_contents(before_current.as_ref()));
}

#[test]
fn post_boundary_legacy_neutral_sequence_record_uses_current_namespace_in_split_recovery() {
    let source = Engine::new_local();
    source
        .execute_text(
            9_720,
            "CREATE TABLE neutral_epoch_owner (id int4, code int4)",
        )
        .unwrap();
    source
        .execute_text(
            9_721,
            "CREATE INDEX neutral_epoch_shadow ON neutral_epoch_owner (code)",
        )
        .unwrap();

    let legacy_command = parse_command("CREATE SEQUENCE neutral_epoch_shadow").unwrap();
    let legacy_payload = Engine::encode_legacy_replay_command_for_test(&legacy_command).unwrap();
    assert!(!Engine::engine_command_uses_current_index_semantics(
        &legacy_payload
    ));
    let suffix = {
        let commit = source.commit_state();
        Engine::canonical_wal_record(
            &commit,
            9_722,
            source.committed_seq() + 1,
            0,
            &legacy_payload,
        )
        .unwrap()
    };
    let mut complete = source.durable_wal_records();
    complete.push(suffix);

    let complete_error = match Engine::recover_from_durable_wal(&complete) {
        Ok(_) => panic!("post-boundary neutral legacy record must use the central namespace"),
        Err(error) => error,
    };
    assert!(
        complete_error
            .to_string()
            .contains("relation \"neutral_epoch_shadow\" already exists"),
        "{complete_error}"
    );

    let split = complete.len() - 1;
    let recovered = Engine::new_local();
    recovered
        .prepare_legacy_index_oid_recovery(&complete)
        .unwrap();
    recovered.begin_recovery_replay();
    recovered
        .replay_durable_records(&complete[..split])
        .unwrap();
    let before_suffix = recovered.catalog_snapshot();
    let split_error = recovered
        .replay_durable_records(&complete[split..])
        .expect_err("chunked replay must retain the one-way current epoch");
    assert!(
        split_error
            .to_string()
            .contains("relation \"neutral_epoch_shadow\" already exists"),
        "{split_error}"
    );
    assert!(recovered
        .catalog_snapshot()
        .same_contents(before_suffix.as_ref()));
    assert!(!recovered
        .catalog_snapshot()
        .relational_sequences
        .contains_key("neutral_epoch_shadow"));
}

#[test]
fn current_table_index_sequence_and_index_view_oid_bundles_are_clone_first_and_recoverable() {
    for (next_oid, sql) in [
        (
            MAX_CATALOG_OID,
            "CREATE TABLE oid_atomic_serial (id serial)",
        ),
        (
            MAX_CATALOG_OID - 1,
            "CREATE TABLE oid_atomic_index_sequence \
             (id serial PRIMARY KEY)",
        ),
    ] {
        let engine = Engine::new_local();
        seed_shared_pg_class_oid_for_boundary_test(&engine, next_oid);
        let before = engine.catalog_snapshot();
        let wal_before = engine.durable_wal_records().len();
        assert_failed_transaction_catalog_statement_is_effect_free(
            &engine,
            9_400 + u64::from(next_oid == MAX_CATALOG_OID),
            sql,
            before.as_ref(),
            wal_before,
        );
        assert_eq!(engine.catalog_snapshot().relational_next_oid, next_oid);
    }

    let table_bundle = Engine::new_local();
    seed_shared_pg_class_oid_for_boundary_test(&table_bundle, MAX_CATALOG_OID - 2);
    table_bundle
        .submit_transaction(9_410, parsed("BEGIN"))
        .unwrap();
    table_bundle
        .submit_transaction(
            9_410,
            parsed("CREATE TABLE oid_bundle (id serial PRIMARY KEY)"),
        )
        .unwrap();
    table_bundle
        .submit_transaction(9_410, parsed("COMMIT"))
        .unwrap();
    let table_catalog = table_bundle.catalog_snapshot();
    let table = &table_catalog.relational_catalog["oid_bundle"];
    assert_eq!(table.oid, MAX_CATALOG_OID - 2);
    assert_eq!(table.indexes[0].oid, MAX_CATALOG_OID - 1);
    assert_eq!(
        table_catalog.relational_sequences["oid_bundle_id_seq"].oid,
        MAX_CATALOG_OID
    );
    assert_eq!(table_catalog.relational_next_oid, MAX_CATALOG_OID + 1);
    let table_recovered = Engine::new_local();
    seed_shared_pg_class_oid_for_boundary_test(&table_recovered, MAX_CATALOG_OID - 2);
    table_recovered.begin_recovery_replay();
    table_recovered
        .replay_durable_records(&table_bundle.durable_wal_records())
        .unwrap();
    table_recovered.finish_recovery_replay().unwrap();
    assert!(table_recovered
        .catalog_snapshot()
        .same_contents(table_catalog.as_ref()));

    let mixed = Engine::new_local();
    seed_shared_pg_class_oid_for_boundary_test(&mixed, MAX_CATALOG_OID - 2);
    mixed
        .execute_text(9_420, "CREATE TABLE oid_mixed_owner (id int4, code int4)")
        .unwrap();
    mixed.submit_transaction(9_421, parsed("BEGIN")).unwrap();
    mixed
        .submit_transaction(
            9_421,
            parsed("CREATE INDEX oid_mixed_index ON oid_mixed_owner (code)"),
        )
        .unwrap();
    mixed
        .submit_transaction(
            9_421,
            parsed(
                "CREATE VIEW oid_mixed_view AS \
                 SELECT id FROM oid_mixed_owner",
            ),
        )
        .unwrap();
    mixed.submit_transaction(9_421, parsed("COMMIT")).unwrap();
    let mixed_catalog = mixed.catalog_snapshot();
    assert_eq!(
        index_named(
            &mixed_catalog.relational_catalog["oid_mixed_owner"],
            "oid_mixed_index",
        )
        .oid,
        MAX_CATALOG_OID - 1
    );
    assert_eq!(
        mixed_catalog.relational_views["oid_mixed_view"].oid,
        MAX_CATALOG_OID
    );
    assert_eq!(mixed_catalog.relational_next_oid, MAX_CATALOG_OID + 1);
    let mixed_recovered = Engine::new_local();
    seed_shared_pg_class_oid_for_boundary_test(&mixed_recovered, MAX_CATALOG_OID - 2);
    mixed_recovered.begin_recovery_replay();
    mixed_recovered
        .replay_durable_records(&mixed.durable_wal_records())
        .unwrap();
    mixed_recovered.finish_recovery_replay().unwrap();
    assert!(mixed_recovered
        .catalog_snapshot()
        .same_contents(mixed_catalog.as_ref()));

    let mixed_failure = Engine::new_local();
    mixed_failure
        .execute_text(
            9_430,
            "CREATE TABLE oid_mixed_failure_owner (id int4, code int4)",
        )
        .unwrap();
    seed_shared_pg_class_oid_for_boundary_test(&mixed_failure, MAX_CATALOG_OID);
    let published_before = mixed_failure.catalog_snapshot();
    let wal_before = mixed_failure.durable_wal_records().len();
    mixed_failure
        .submit_transaction(9_431, parsed("BEGIN"))
        .unwrap();
    mixed_failure
        .submit_transaction(
            9_431,
            parsed(
                "CREATE INDEX oid_mixed_failure_index \
                 ON oid_mixed_failure_owner (code)",
            ),
        )
        .unwrap();
    let error = mixed_failure
        .submit_transaction(
            9_431,
            parsed(
                "CREATE VIEW oid_mixed_failure_view AS \
                 SELECT id FROM oid_mixed_failure_owner",
            ),
        )
        .expect_err("the view allocation after MAX must reject privately");
    assert!(
        error.to_string().contains("OID allocation exhausted"),
        "{error}"
    );
    let private = mixed_failure
        .transaction_snapshot_handle(9_431)
        .unwrap()
        .transaction_catalog();
    assert_eq!(private.relational_next_oid, MAX_CATALOG_OID + 1);
    assert!(!private
        .relational_views
        .contains_key("oid_mixed_failure_view"));
    assert!(private.relational_catalog["oid_mixed_failure_owner"]
        .indexes
        .iter()
        .any(|index| index.name == "oid_mixed_failure_index"));
    assert!(mixed_failure
        .catalog_snapshot()
        .same_contents(published_before.as_ref()));
    assert_eq!(mixed_failure.durable_wal_records().len(), wal_before);
    mixed_failure
        .submit_transaction(9_431, parsed("ROLLBACK"))
        .unwrap();
}

#[test]
fn legacy_shadowed_pg_class_bindings_recover_but_current_commands_fail_closed() {
    let legacy_records = [
        gpu_db_wal::WalRecord {
            txn_id: 9_500,
            payload: Arc::from(&b"CREATE TABLE legacy_namespace_source (id int4)"[..]),
        },
        gpu_db_wal::WalRecord {
            txn_id: 9_501,
            payload: Arc::from(
                &b"CREATE TABLE legacy_namespace_owner \
                   (id int4, code int4, \
                    CONSTRAINT legacy_view_shadow UNIQUE (code), \
                    CONSTRAINT legacy_table_shadow UNIQUE (id))"[..],
            ),
        },
        gpu_db_wal::WalRecord {
            txn_id: 9_502,
            payload: Arc::from(&b"CREATE TABLE legacy_table_shadow (id int4)"[..]),
        },
        gpu_db_wal::WalRecord {
            txn_id: 9_503,
            payload: Arc::from(
                &b"CREATE VIEW legacy_view_shadow AS \
                   SELECT id FROM legacy_namespace_source"[..],
            ),
        },
    ];
    let engine = Engine::recover_from_durable_wal(&legacy_records).unwrap();
    let recovered = engine.catalog_snapshot();
    assert!(recovered
        .relational_catalog
        .contains_key("legacy_table_shadow"));
    assert!(recovered
        .relational_views
        .contains_key("legacy_view_shadow"));
    let owner = &recovered.relational_catalog["legacy_namespace_owner"];
    assert!(owner
        .indexes
        .iter()
        .any(|index| index.name == "legacy_table_shadow"));
    assert!(owner
        .indexes
        .iter()
        .any(|index| index.name == "legacy_view_shadow"));
    assert!(recovered
        .pg_class_relation_kind("legacy_table_shadow")
        .is_err());
    assert!(recovered
        .pg_class_relation_kind("legacy_view_shadow")
        .is_err());

    let wal_before = engine.durable_wal_records().len();
    for (ordinal, sql) in [
        "CREATE TABLE legacy_view_shadow (id int4)",
        "CREATE INDEX legacy_ambiguous_owner_index \
         ON legacy_table_shadow (id)",
        "ALTER INDEX legacy_view_shadow RENAME TO legacy_index_renamed",
        "DROP INDEX IF EXISTS legacy_view_shadow",
        "CREATE OR REPLACE VIEW legacy_view_shadow AS \
         SELECT id FROM legacy_namespace_source",
        "ALTER VIEW legacy_view_shadow RENAME TO legacy_view_renamed",
        "DROP VIEW IF EXISTS legacy_view_shadow",
    ]
    .into_iter()
    .enumerate()
    {
        let error = engine
            .execute_text(9_510 + ordinal as u64, sql)
            .expect_err("a current command must not choose one side of an ambiguous binding");
        assert!(
            error.to_string().contains("multiple pg_class relations"),
            "{sql:?}: {error}"
        );
        assert!(engine.catalog_snapshot().same_contents(recovered.as_ref()));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
    }
}

#[test]
fn legacy_opcode_12_preserves_an_acknowledged_view_index_shadow() {
    let prefix = [
        gpu_db_wal::WalRecord {
            txn_id: 9_600,
            payload: Arc::from(&b"CREATE TABLE legacy_opcode_source (id int4)"[..]),
        },
        gpu_db_wal::WalRecord {
            txn_id: 9_601,
            payload: Arc::from(
                &b"CREATE TABLE legacy_opcode_owner \
                   (id int4, CONSTRAINT legacy_opcode_shadow UNIQUE (id))"[..],
            ),
        },
    ];
    let model = Engine::recover_from_durable_wal(&prefix).unwrap();
    let before = model.catalog_snapshot();
    let command =
        parse_command("CREATE VIEW legacy_opcode_shadow AS SELECT id FROM legacy_opcode_source")
            .unwrap();
    let Command::CreateView(create) = command.clone() else {
        panic!("expected CREATE VIEW");
    };
    let mut working = model.ddl_catalog().clone();
    model
        .with_apply_catalog(Some(Arc::clone(&before)), || {
            model.apply_create_view_legacy_replay(&mut working, create)
        })
        .unwrap();
    let after = Engine::catalog_snapshot_from_working(&working, before.commit_seq);
    let source = &before.relational_catalog["legacy_opcode_source"];
    let view = &after.relational_views["legacy_opcode_shadow"];
    let record = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
        allocator_high_water: 0,
        catalog_commands: vec![BinaryTransactionCatalogCommand {
            ordinal: 0,
            command: command.clone(),
        }],
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: Some(BinaryTransactionCatalogOutput {
            relational_next_oid: after.relational_next_oid,
            relational_next_column_id: after.relational_next_column_id,
            created_sequence_oids: BTreeMap::new(),
        }),
        view_operations: vec![BinaryTransactionViewOperationIdentity {
            command_index: 0,
            ordinal: 0,
            target_before: None,
            dependencies: BTreeMap::from([(
                "legacy_opcode_source".to_string(),
                BinaryCatalogRelationIdentity {
                    kind: BinaryCatalogRelationKind::Table,
                    oid: source.oid,
                    digest: crate::engine_transaction_reset::table_schema_digest(source).unwrap(),
                },
            )]),
            target_after: BinaryCatalogRelationIdentity {
                kind: BinaryCatalogRelationKind::View,
                oid: view.oid,
                digest: crate::engine_transaction_catalog::view_identity::view_semantic_digest(
                    view,
                )
                .unwrap(),
            },
        }],
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        operation_order: vec![BinaryTransactionOperationIdentity::Catalog { command_index: 0 }],
        statement_digests: vec![transaction_statement_digest(&command).unwrap()],
        sequence_input_oids: BTreeMap::new(),
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: Vec::new(),
    };
    let payload = try_encode_binary_transaction(&record).unwrap();
    assert_eq!(payload[2], 12, "historical CREATE VIEW must use opcode 12");
    let records = prefix
        .into_iter()
        .chain(std::iter::once(gpu_db_wal::WalRecord {
            txn_id: 9_602,
            payload: Arc::from(payload),
        }))
        .collect::<Vec<_>>();
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    let catalog = recovered.catalog_snapshot();
    assert!(catalog
        .relational_views
        .contains_key("legacy_opcode_shadow"));
    assert!(catalog.relational_catalog["legacy_opcode_owner"]
        .indexes
        .iter()
        .any(|index| index.name == "legacy_opcode_shadow"));
    assert!(catalog
        .pg_class_relation_kind("legacy_opcode_shadow")
        .is_err());
    let before_current = recovered.catalog_snapshot();
    let error = recovered
        .execute_text(9_603, "DROP VIEW IF EXISTS legacy_opcode_shadow")
        .expect_err("a current command must not guess through the historical ambiguity");
    assert!(
        error.to_string().contains("multiple pg_class relations"),
        "{error}"
    );
    assert!(recovered
        .catalog_snapshot()
        .same_contents(before_current.as_ref()));
}

#[test]
fn post_boundary_byte_stable_opcode_12_rejects_a_legacy_ambiguous_dependency() {
    let legacy_prefix = [
        gpu_db_wal::WalRecord {
            txn_id: 9_610,
            payload: Arc::from(
                &b"CREATE TABLE epoch_policy_owner (id int4, code int4, \
                   CONSTRAINT epoch_policy_ambiguous UNIQUE (code))"[..],
            ),
        },
        gpu_db_wal::WalRecord {
            txn_id: 9_611,
            payload: Arc::from(&b"CREATE TABLE epoch_policy_ambiguous (id int4)"[..]),
        },
    ];
    let source = Engine::recover_from_durable_wal(&legacy_prefix).unwrap();
    assert!(source
        .catalog_snapshot()
        .pg_class_relation_kind("epoch_policy_ambiguous")
        .is_err());
    source.submit_transaction(9_612, parsed("BEGIN")).unwrap();
    source
        .submit_transaction(
            9_612,
            parsed(
                "CREATE INDEX epoch_policy_boundary \
                 ON epoch_policy_owner (id)",
            ),
        )
        .unwrap();
    source.submit_transaction(9_612, parsed("COMMIT")).unwrap();
    let boundary_records = source.durable_wal_records();
    assert_eq!(
        &operation_payload(boundary_records.last().unwrap())[..3],
        &[255, 1, 16]
    );

    let before = source.catalog_snapshot();
    let command = parse_command(
        "CREATE VIEW epoch_policy_forbidden AS \
         SELECT id FROM epoch_policy_ambiguous",
    )
    .unwrap();
    let Command::CreateView(create) = command.clone() else {
        panic!("expected CREATE VIEW");
    };
    let mut legacy_postimage = source.ddl_catalog().clone();
    source
        .with_apply_catalog(Some(Arc::clone(&before)), || {
            source.apply_create_view_legacy_replay(&mut legacy_postimage, create)
        })
        .unwrap();
    let after = Engine::catalog_snapshot_from_working(&legacy_postimage, before.commit_seq);
    let table = &before.relational_catalog["epoch_policy_ambiguous"];
    let view = &after.relational_views["epoch_policy_forbidden"];
    let record = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
        allocator_high_water: 0,
        catalog_commands: vec![BinaryTransactionCatalogCommand {
            ordinal: 0,
            command: command.clone(),
        }],
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: Some(BinaryTransactionCatalogOutput {
            relational_next_oid: after.relational_next_oid,
            relational_next_column_id: after.relational_next_column_id,
            created_sequence_oids: BTreeMap::new(),
        }),
        view_operations: vec![BinaryTransactionViewOperationIdentity {
            command_index: 0,
            ordinal: 0,
            target_before: None,
            dependencies: BTreeMap::from([(
                "epoch_policy_ambiguous".to_string(),
                BinaryCatalogRelationIdentity {
                    kind: BinaryCatalogRelationKind::Table,
                    oid: table.oid,
                    digest: crate::engine_transaction_reset::table_schema_digest(table).unwrap(),
                },
            )]),
            target_after: BinaryCatalogRelationIdentity {
                kind: BinaryCatalogRelationKind::View,
                oid: view.oid,
                digest: crate::engine_transaction_catalog::view_identity::view_semantic_digest(
                    view,
                )
                .unwrap(),
            },
        }],
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        operation_order: vec![BinaryTransactionOperationIdentity::Catalog { command_index: 0 }],
        statement_digests: vec![transaction_statement_digest(&command).unwrap()],
        sequence_input_oids: BTreeMap::new(),
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: Vec::new(),
    };
    let payload = try_encode_binary_transaction(&record).unwrap();
    assert_eq!(payload[2], 12, "the neutral suffix remains byte-stable");
    let mut complete = boundary_records;
    let payload = Arc::from(payload);
    let suffix = {
        let commit = source.commit_state();
        Engine::canonical_wal_record(&commit, 9_613, source.committed_seq() + 1, 0, &payload)
            .unwrap()
    };
    complete.push(suffix);

    let recovered = Engine::new_local();
    recovered
        .prepare_legacy_index_oid_recovery(&complete)
        .unwrap();
    recovered.begin_recovery_replay();
    recovered
        .replay_durable_records(&complete[..complete.len() - 1])
        .unwrap();
    let before_suffix = recovered.catalog_snapshot();
    let error = recovered
        .replay_durable_records(&complete[complete.len() - 1..])
        .expect_err("post-boundary opcode 12 must use the current central namespace");
    assert!(
        error.to_string().contains("multiple pg_class relations"),
        "{error}"
    );
    assert!(recovered
        .catalog_snapshot()
        .same_contents(before_suffix.as_ref()));
    assert!(!recovered
        .catalog_snapshot()
        .relational_views
        .contains_key("epoch_policy_forbidden"));
}
