use super::*;

fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
    gpu_db_sql::ParsedCommand::parse(sql).unwrap()
}

fn table_oid(engine: &Engine, table: &str) -> u32 {
    engine
        .relational_catalog_table(table)
        .unwrap_or_else(|| panic!("missing fixture relation {table}"))
        .oid
}

#[test]
fn copy_target_proof_owns_guard_through_every_protocol_clone() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE copy_guard (id INT, value INT)")
        .unwrap();
    let oid = table_oid(&engine, "copy_guard");
    let (_columns, proof) = engine.relational_copy_target("copy_guard").unwrap();
    let protocol_clone = proof.clone();
    let reset = engine.table_access.lease();

    assert!(matches!(
        reset.acquire_exclusive([oid]),
        Err(ExecuteError::Serialization(_))
    ));
    drop(proof);
    assert!(matches!(
        reset.acquire_exclusive([oid]),
        Err(ExecuteError::Serialization(_))
    ));
    drop(protocol_clone);
    reset.acquire_exclusive([oid]).unwrap();
}

#[test]
fn transaction_copy_proof_retains_only_shared_target_after_terminal_and_parent_upgrade() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE copy_tx_target (id INT)")
        .unwrap();
    engine
        .execute_text(2, "CREATE TABLE copy_tx_peer (id INT)")
        .unwrap();
    let target_oid = table_oid(&engine, "copy_tx_target");
    let peer_oid = table_oid(&engine, "copy_tx_peer");
    engine.submit_transaction(3, parsed("BEGIN")).unwrap();
    let (_columns, proof) = engine
        .relational_copy_target_in_transaction(3, "copy_tx_target")
        .unwrap();
    let snapshot = engine.transaction_snapshot_handle(3).unwrap();
    snapshot.table_access.acquire_shared([peer_oid]).unwrap();
    snapshot
        .table_access
        .acquire_exclusive([target_oid])
        .unwrap();
    engine.submit_transaction(3, parsed("ROLLBACK")).unwrap();
    drop(snapshot);

    let later = engine.table_access.lease();
    later.acquire_exclusive([peer_oid]).unwrap();
    later.acquire_shared([target_oid]).unwrap();
    assert!(matches!(
        later.acquire_exclusive([target_oid]),
        Err(ExecuteError::Serialization(_))
    ));
    drop(proof);
    later.acquire_exclusive([target_oid]).unwrap();
}

#[test]
fn every_public_retained_probe_claims_table_access_before_sampling_a_root() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE probe_guard (a INT, b BIGINT, v INT, UNIQUE (a, b))",
        )
        .unwrap();
    let oid = table_oid(&engine, "probe_guard");
    let reset = engine.table_access.lease();
    reset.acquire_exclusive([oid]).unwrap();
    let select = match parse_command("SELECT v FROM probe_guard WHERE a = 1").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };

    assert!(matches!(
        engine.execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
            std::slice::from_ref(&select)
        ),
        Err(ExecuteError::Serialization(_))
    ));
    assert!(matches!(
        engine.bench_sharded_point_lookup_batch("probe_guard", "a", &["v".to_string()], &[1]),
        Err(ExecuteError::Serialization(_))
    ));
    assert!(matches!(
        engine.prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "probe_guard",
            ["a", "b"],
            &["v"]
        ),
        Err(ExecuteError::Serialization(_))
    ));
    assert!(engine
        .relational_retained_device_read_view("probe_guard")
        .is_none());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn retained_device_view_uses_one_generation_atomic_entry_during_replacement() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(
            1,
            "CREATE TABLE retained_generation (id INT PRIMARY KEY, value INT)",
        )
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO retained_generation VALUES (1, 10)")
        .unwrap();
    engine
        .execute_text(
            3,
            "CREATE TABLE retained_generation_other (id INT PRIMARY KEY, value INT)",
        )
        .unwrap();
    engine
        .execute_text(
            4,
            "INSERT INTO retained_generation_other VALUES (1, 10), (2, 20)",
        )
        .unwrap();
    install_test_single_buffer_residency(&mut engine, "retained_generation");
    install_test_single_buffer_residency(&mut engine, "retained_generation_other");
    let first = engine
        .read_state
        .residency
        .snapshots
        .load()
        .get("retained_generation")
        .cloned()
        .unwrap();
    let second = engine
        .read_state
        .residency
        .snapshots
        .load()
        .get("retained_generation_other")
        .cloned()
        .unwrap();
    assert_ne!(first.descriptor.row_count, second.descriptor.row_count);
    let first_memory = first.device_memory.as_ref().unwrap();
    let second_memory = second.device_memory.as_ref().unwrap();
    assert_ne!(first_memory.device_ptr(), second_memory.device_ptr());

    // Reproduce the publisher's legal transient order: the legacy lifecycle side map already
    // carries generation B while the immutable entry map still publishes generation A. A read
    // facade that performs two independent loads returns A metadata over B bytes.
    engine.read_state.residency.with_snapshots_mut(|snapshots| {
        snapshots.insert("retained_generation".to_string(), first.clone())
    });
    engine
        .read_state
        .residency
        .device_memory
        .insert("retained_generation".to_string(), Arc::clone(second_memory));
    let view = engine
        .relational_retained_device_read_view("retained_generation")
        .expect("generation A remains a valid retained entry");
    assert_eq!(view.snapshot_handle().row_count, first.descriptor.row_count);
    assert_eq!(view.device_ptr(), first_memory.device_ptr());
    assert_ne!(view.device_ptr(), second_memory.device_ptr());
}

#[test]
fn typed_insert_compatibility_surface_uses_the_immediate_terminal_and_releases_its_guard() {
    let mut engine = Engine::with_batching(8, Duration::from_secs(60));
    engine
        .execute_text(1, "CREATE TABLE queued_guard (id INT, value INT)")
        .unwrap();
    let oid = table_oid(&engine, "queued_guard");
    engine
        .enqueue_set_text(2, "INSERT INTO queued_guard VALUES (1, 10)", Instant::now())
        .unwrap();
    assert_eq!(
        engine.batcher().len(),
        0,
        "INSERT compatibility ingress must not retain a second queued write authority"
    );

    let reset = engine.table_access.lease();
    reset.acquire_exclusive([oid]).unwrap();
}

#[test]
fn table_affecting_ddl_claimants_conflict_before_effect() {
    let mut engine = Engine::with_batching(8, Duration::from_secs(60));
    engine
        .execute_text(1, "CREATE TABLE ddl_guard (id INT)")
        .unwrap();
    let oid = table_oid(&engine, "ddl_guard");
    let durable_before = engine.durable_wal_records().len();
    let reset = engine.table_access.lease();
    reset.acquire_exclusive([oid]).unwrap();

    for error in [
        engine
            .execute_text(2, "ALTER TABLE ddl_guard ADD COLUMN value INT")
            .unwrap_err(),
        engine
            .execute_text(3, "CREATE TABLE ddl_guard_peer (id INT)")
            .unwrap_err(),
        engine
            .enqueue_set_text(4, "DROP TABLE ddl_guard", Instant::now())
            .unwrap_err(),
    ] {
        assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
    }
    let raw_error = engine
        .commit_mutation(
            5,
            Arc::from(&b"ALTER TABLE ddl_guard ADD COLUMN other INT"[..]),
        )
        .unwrap_err();
    assert!(
        raw_error
            .to_string()
            .contains("incompatible retained access guard"),
        "{raw_error}"
    );
    assert_eq!(engine.batcher().len(), 0);
    assert_eq!(engine.durable_wal_records().len(), durable_before);

    drop(reset);
    let table = engine.relational_catalog_table("ddl_guard").unwrap();
    assert_eq!(table.columns.len(), 1);
    assert!(engine.relational_catalog_table("ddl_guard_peer").is_none());
}

#[test]
fn exact_index_and_constraint_claimants_ignore_unrelated_table_reset() {
    let engine = Engine::with_batching(8, Duration::from_secs(60));
    engine
        .execute_text(1, "CREATE TABLE ddl_exact_target (id INT)")
        .unwrap();
    engine
        .execute_text(2, "CREATE TABLE ddl_exact_reset_peer (id INT)")
        .unwrap();
    let target_oid = table_oid(&engine, "ddl_exact_target");
    let peer_oid = table_oid(&engine, "ddl_exact_reset_peer");
    let reset = engine.table_access.lease();
    reset.acquire_exclusive([peer_oid]).unwrap();

    engine
        .execute_text(
            3,
            "CREATE INDEX ddl_exact_target_idx ON ddl_exact_target (id)",
        )
        .unwrap();
    engine
        .execute_text(
            4,
            "ALTER TABLE ddl_exact_target ADD CONSTRAINT ddl_exact_target_pkey PRIMARY KEY (id)",
        )
        .unwrap();

    let target_reset = engine.table_access.lease();
    target_reset.acquire_exclusive([target_oid]).unwrap();
    let error = engine
        .execute_text(
            5,
            "CREATE INDEX ddl_exact_target_idx_2 ON ddl_exact_target (id)",
        )
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
}

#[test]
fn terminal_serialized_and_ddl_retries_resolve_before_fresh_table_access() {
    let engine = Engine::new_local_test_engine();
    engine
        .submit_transaction(
            20,
            parsed("CREATE TABLE retry_guard (id SERIAL PRIMARY KEY, value INT)"),
        )
        .unwrap();
    engine
        .submit_transaction(21, parsed("INSERT INTO retry_guard (value) VALUES (10)"))
        .unwrap();
    // The omitted SERIAL default is a separately durable sequence transition and claims the
    // allocator's next canonical identity (22) before the enclosing INSERT publishes.
    engine
        .submit_transaction(23, parsed("CREATE TABLE retry_guard_peer (id INT)"))
        .unwrap();
    let wal_after_commits = engine.durable_wal_records().len();
    let reset = engine.table_access.lease();
    reset
        .acquire_exclusive([table_oid(&engine, "retry_guard")])
        .unwrap();

    engine
        .submit_transaction(21, parsed("INSERT INTO retry_guard (value) VALUES (10)"))
        .expect("a sequence-default INSERT retry has no new table access");
    engine
        .submit_transaction(23, parsed("CREATE TABLE retry_guard_peer (id INT)"))
        .expect("a terminal DDL retry resolves before its conservative catalog guard");
    let mismatch = engine
        .submit_transaction(21, parsed("INSERT INTO retry_guard (value) VALUES (11)"))
        .unwrap_err();
    assert!(
        mismatch.to_string().contains("different request"),
        "mismatched identity must win over the unrelated reset conflict: {mismatch}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_after_commits);
}

#[test]
fn canonical_dml_and_enqueue_retries_resolve_before_reset_guards() {
    let mut engine = Engine::with_batching(8, Duration::from_secs(60));
    engine
        .execute_text(30, "CREATE TABLE ingress_retry (id INT, value INT)")
        .unwrap();
    let direct = "INSERT INTO ingress_retry VALUES (1, 10)";
    engine.execute_dml_concurrent(31, direct).unwrap();
    let queued = "INSERT INTO ingress_retry VALUES (2, 20)";
    engine.enqueue_set_text(32, queued, Instant::now()).unwrap();
    engine.flush_admin().unwrap();
    let committed = engine.committed_seq();
    let durable = engine.durable_wal_records().len();

    let reset = engine.table_access.lease();
    reset
        .acquire_exclusive([table_oid(&engine, "ingress_retry")])
        .unwrap();
    engine
        .execute_dml_concurrent(31, direct)
        .expect("concurrent DML exact retry must not claim fresh shared access");
    engine
        .enqueue_set_text(32, queued, Instant::now())
        .expect("queued exact retry must not claim fresh shared access");
    assert_eq!(engine.batcher().len(), 0);
    assert_eq!(engine.committed_seq(), committed);
    assert_eq!(engine.durable_wal_records().len(), durable);
}

#[test]
fn raced_table_access_failure_rechecks_new_terminal_identity() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(40, "CREATE TABLE retry_race (id INT, value INT)")
        .unwrap();
    let text = "INSERT INTO retry_race VALUES (1, 10)";
    let digest = gpu_db_wal::canonical_request_digest(text.as_bytes());

    let outcome = engine
        .sabotage_retry_access_race(41, digest, || {
            engine.execute_dml_concurrent(41, text).unwrap();
            Err(ExecuteError::Serialization(
                "deterministic post-lookup guard race".to_string(),
            ))
        })
        .unwrap();
    assert!(matches!(
        outcome,
        crate::engine_transaction_reset::StableRetryOr::Terminal(1)
    ));
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id FROM retry_race")
            .unwrap()
            .rows
            .len(),
        1
    );
}

#[test]
fn raw_binary_inserts_are_rejected_before_dependency_guards() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE binary_a (id INT, value INT)")
        .unwrap();
    engine
        .execute_text(2, "CREATE TABLE binary_b (id INT, value INT)")
        .unwrap();
    let oid_b = table_oid(&engine, "binary_b");
    let reset = engine.table_access.lease();
    reset.acquire_exclusive([oid_b]).unwrap();

    let record = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::IndexIdentityV1,
        allocator_high_water: 3,
        catalog_commands: Vec::new(),
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: None,
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: Vec::new(),
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: vec![
            BinaryTransactionMutation::Insert {
                table: "binary_a".to_string(),
                row_id: 1,
                row_encoded: encode_relational_row(&[SqlValue::Int4(1), SqlValue::Int4(10)]),
            },
            BinaryTransactionMutation::Insert {
                table: "binary_b".to_string(),
                row_id: 2,
                row_encoded: encode_relational_row(&[SqlValue::Int4(2), SqlValue::Int4(20)]),
            },
        ],
    };
    let payload: Arc<[u8]> = Arc::from(try_encode_binary_transaction(&record).unwrap());
    let durable_before = engine.durable_wal_records().len();
    let error = engine.commit_mutation(3, payload).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("must enter typed transaction admission"),
        "{error}"
    );
    assert_eq!(engine.durable_wal_records().len(), durable_before);
}

#[test]
fn raw_binary_table_reset_is_rejected_before_wal_claim() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE binary_reset (id INT)")
        .unwrap();
    let catalog = engine.read_state.latest_catalog();
    let table = catalog.relational_catalog["binary_reset"].clone();
    let identities = crate::engine_transaction_reset::table_access_dependency_identities(
        &catalog.relational_catalog,
        &table,
    )
    .unwrap();
    let record = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::IndexIdentityV1,
        allocator_high_water: 1,
        catalog_commands: Vec::new(),
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: None,
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: Vec::new(),
        table_resets: vec![BinaryTransactionTableReset {
            ordinal: 0,
            table: table.name.clone(),
            table_oid: table.oid,
            schema_digest: [0; 32],
            source_commit_seq: 0,
            before_digest: [0; 32],
            expected_rows: 0,
            after_empty_digest: [0; 32],
            dependency_identities: identities,
        }],
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::new(),
        mutations: Vec::new(),
    };
    let payload: Arc<[u8]> = Arc::from(try_encode_binary_transaction(&record).unwrap());
    let durable_before = engine.durable_wal_records().len();
    let error = engine.commit_mutation(2, Arc::clone(&payload)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("require typed transaction admission"),
        "{error}"
    );
    let batch = engine
        .commit_mutation_batch(&[(3, Arc::clone(&payload))])
        .unwrap_err();
    assert!(!batch.requeue);
    assert!(
        batch
            .error
            .to_string()
            .contains("require typed transaction admission"),
        "{}",
        batch.error
    );
    assert_eq!(engine.durable_wal_records().len(), durable_before);
}

#[test]
fn identity_bound_row_apply_rejects_oid_and_schema_aba_before_state_changes() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE identity_apply (id INT)")
        .unwrap();
    let table = engine.read_state.latest_catalog().relational_catalog["identity_apply"].clone();
    let schema_digest = crate::engine_transaction_reset::table_schema_digest(&table).unwrap();
    let base = BinaryTransactionRecord {
        catalog_epoch: BinaryTransactionCatalogEpoch::IndexIdentityV1,
        allocator_high_water: 2,
        catalog_commands: Vec::new(),
        created_table_identities: BTreeMap::new(),
        created_table_index_identities: BTreeMap::new(),
        catalog_output: None,
        view_operations: Vec::new(),
        view_lifecycle_operations: Vec::new(),
        index_lifecycle_operations: Vec::new(),
        sequence_lifecycle_operations: Vec::new(),
        sequence_reset_operations: Vec::new(),
        sequence_advances_by_oid: BTreeMap::new(),
        operation_order: Vec::new(),
        statement_digests: Vec::new(),
        sequence_input_oids: BTreeMap::new(),
        sequence_value_references: Vec::new(),
        table_resets: Vec::new(),
        sequence_advances: BTreeMap::new(),
        table_identities: BTreeMap::from([(
            table.name.clone(),
            BinaryTransactionTableIdentity {
                table_oid: table.oid,
                schema_digest,
            },
        )]),
        mutations: vec![BinaryTransactionMutation::Insert {
            table: table.name.clone(),
            row_id: 1,
            row_encoded: encode_relational_row(&[SqlValue::Int4(7)]),
        }],
    };
    let entry = LogEntry {
        term: 1,
        index: engine.committed_seq() + 1,
        payload: Arc::from(&b""[..]),
    };

    for identity in [
        BinaryTransactionTableIdentity {
            table_oid: table.oid + 1,
            schema_digest,
        },
        BinaryTransactionTableIdentity {
            table_oid: table.oid,
            schema_digest: [0xA5; 32],
        },
    ] {
        let mut record = base.clone();
        record.table_identities.insert(table.name.clone(), identity);
        let mut catalog = engine
            .catalog_latch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let error = engine
            .apply_binary_transaction_record(&entry, &mut catalog, record)
            .unwrap_err();
        assert!(error.to_string().contains("changed from OID"), "{error}");
    }
    assert!(engine
        .execute_relational_select_text("SELECT id FROM identity_apply")
        .unwrap()
        .rows
        .is_empty());
}
