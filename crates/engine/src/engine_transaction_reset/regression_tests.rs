use super::*;

fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
    gpu_db_sql::ParsedCommand::parse(sql).unwrap()
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn typed_transaction_resets_preserve_order_identity_consumption_and_recovery() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            100,
            parsed("CREATE TABLE reset_order (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(101, parsed("INSERT INTO reset_order VALUES (1)"))
        .unwrap();

    engine.submit_transaction(102, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(102, parsed("INSERT INTO reset_order VALUES (2)"))
        .unwrap();
    engine
        .submit_transaction(102, parsed("TRUNCATE TABLE reset_order"))
        .unwrap();
    let private_empty = match parse_command("SELECT id FROM reset_order").unwrap() {
        Command::Select(select) => engine
            .execute_relational_select_in_transaction(102, &select)
            .unwrap(),
        _ => unreachable!(),
    };
    assert!(private_empty.rows.is_empty());
    engine.submit_transaction(102, parsed("COMMIT")).unwrap();
    assert!(engine
        .execute_relational_select_text("SELECT id FROM reset_order")
        .unwrap()
        .rows
        .is_empty());
    let reset_table = engine.catalog_snapshot().relational_catalog["reset_order"].clone();
    assert_eq!(
        engine.zero_row_resident_generation_boundary(&reset_table),
        Some(engine.committed_seq()),
        "a reset-only commit must publish a real typed zero-row GPU generation"
    );
    assert!(engine.table_device_authoritative("reset_order"));
    let records = engine.durable_wal_records();
    let reset_only = match decode_binary_record(&records.last().unwrap().payload).unwrap() {
        BinaryWalRecord::Transaction(record) => record,
        _ => panic!("expected typed reset transaction"),
    };
    assert_eq!(reset_only.table_resets.len(), 1);
    assert!(reset_only.mutations.is_empty());
    assert!(
        reset_only.allocator_high_water > 2,
        "the insert shadowed by TRUNCATE still consumes its claimed identity"
    );
    let envelope = gpu_db_wal::decode_canonical_record_payload(&records.last().unwrap().payload)
        .unwrap()
        .unwrap();
    assert_eq!(
        envelope.fragments[0].kind,
        gpu_db_wal::CanonicalFragmentKind::TableReset
    );
    assert_eq!(envelope.header.table_block_count, 1);

    // The typed zero-row generation is ordinary device authority, not a terminal special
    // case: a following autocommit write may reuse a key retired by the reset and must rebuild
    // the mandatory named-index coverage on the fresh lineage.
    engine
        .submit_transaction(150, parsed("INSERT INTO reset_order VALUES (1)"))
        .unwrap();
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id FROM reset_order")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1)]]
    );

    engine.submit_transaction(103, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(103, parsed("TRUNCATE reset_order"))
        .unwrap();
    engine
        .submit_transaction(103, parsed("INSERT INTO reset_order VALUES (3)"))
        .unwrap();
    let private_after_insert = match parse_command("SELECT id FROM reset_order").unwrap() {
        Command::Select(select) => engine
            .execute_relational_select_in_transaction(103, &select)
            .unwrap(),
        _ => unreachable!(),
    };
    assert_eq!(private_after_insert.rows, vec![vec![SqlValue::Int4(3)]]);
    engine.submit_transaction(103, parsed("COMMIT")).unwrap();
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id FROM reset_order")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(3)]]
    );

    engine.submit_transaction(104, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(104, parsed("TRUNCATE reset_order"))
        .unwrap();
    engine
        .submit_transaction(104, parsed("INSERT INTO reset_order VALUES (4)"))
        .unwrap();
    engine
        .submit_transaction(104, parsed("TRUNCATE reset_order"))
        .unwrap();
    engine
        .submit_transaction(104, parsed("INSERT INTO reset_order VALUES (5)"))
        .unwrap();
    engine.submit_transaction(104, parsed("COMMIT")).unwrap();
    let repeated = match decode_binary_record(&engine.durable_wal_records().last().unwrap().payload)
        .unwrap()
    {
        BinaryWalRecord::Transaction(record) => record,
        _ => panic!("expected repeated typed reset transaction"),
    };
    assert_eq!(repeated.table_resets.len(), 1);
    assert_eq!(repeated.table_resets[0].ordinal, 2);
    assert_eq!(repeated.mutations.len(), 1);
    assert!(matches!(
        &repeated.mutations[0],
        BinaryTransactionMutation::Insert { row_encoded, .. }
            if row_encoded == &encode_relational_row(&[SqlValue::Int4(5)])
    ));
    let expected = vec![vec![SqlValue::Int4(5)]];
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id FROM reset_order")
            .unwrap()
            .rows,
        expected
    );

    let durable = engine.durable_wal_records();
    let recovered = Engine::recover_from_durable_wal(&durable).unwrap();
    assert_eq!(
        recovered
            .execute_relational_select_text("SELECT id FROM reset_order")
            .unwrap()
            .rows,
        expected
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn copy_and_explicit_transaction_roots_feed_reset_recovery() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            180,
            parsed("CREATE TABLE copy_reset_root (id int4 PRIMARY KEY, value int4)"),
        )
        .unwrap();
    let copy = gpu_db_sql::parse_copy_from_stdin(
        "COPY copy_reset_root (id, value) FROM STDIN WITH (FORMAT csv)",
    )
    .unwrap();
    let copy_rows = vec![
        vec![SqlValue::Int4(1), SqlValue::Int4(10)],
        vec![SqlValue::Int4(2), SqlValue::Int4(20)],
    ];
    assert_eq!(
        engine
            .execute_relational_copy_rows(181, &copy, copy_rows.clone())
            .unwrap(),
        2
    );
    let copy_root = engine.committed_seq();
    assert_eq!(
        engine.test_table_root_index("copy_reset_root"),
        copy_root,
        "typed COPY must advance the canonical table root"
    );
    engine
        .submit_transaction(182, parsed("TRUNCATE copy_reset_root"))
        .unwrap();

    engine
        .submit_transaction(
            183,
            parsed("CREATE TABLE txn_reset_root (id int4 PRIMARY KEY, value int4)"),
        )
        .unwrap();
    engine.submit_transaction(184, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(184, parsed("INSERT INTO txn_reset_root VALUES (7, 70)"))
        .unwrap();
    engine.submit_transaction(184, parsed("COMMIT")).unwrap();
    let transaction_root = engine.committed_seq();
    assert_eq!(
        engine.test_table_root_index("txn_reset_root"),
        transaction_root,
        "an explicit transaction mutation must advance the canonical table root"
    );
    engine
        .submit_transaction(185, parsed("TRUNCATE txn_reset_root"))
        .unwrap();

    let durable = engine.durable_wal_records();
    let recovered = Engine::recover_from_durable_wal(&durable).unwrap();
    for table in ["copy_reset_root", "txn_reset_root"] {
        assert!(
            recovered
                .execute_relational_select_text(&format!("SELECT id FROM {table}"))
                .unwrap()
                .rows
                .is_empty(),
            "{table} must recover at its reset root"
        );
    }
    let recovered_wal = recovered.durable_wal_records().len();
    assert_eq!(
        recovered
            .execute_relational_copy_rows(181, &copy, copy_rows)
            .unwrap(),
        2,
        "recovery retains COPY's exact terminal affected-row outcome"
    );
    assert_eq!(recovered.durable_wal_records().len(), recovered_wal);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn exact_reset_source_digest_rejects_same_cardinality_mutation_and_replay_mismatch() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            900,
            parsed("CREATE TABLE reset_digest_guard (id INT PRIMARY KEY, note TEXT)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            901,
            parsed("INSERT INTO reset_digest_guard VALUES (1, NULL), (2, 'two')"),
        )
        .unwrap();
    engine.submit_transaction(902, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(902, parsed("TRUNCATE reset_digest_guard"))
        .unwrap();
    let durable_before = engine.durable_wal_records().len();
    let table = engine.catalog_snapshot().relational_catalog["reset_digest_guard"].clone();
    let staged = engine.transaction_snapshot_handle(902).unwrap();
    let reset = staged
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .operations
        .iter()
        .find_map(|operation| match operation {
            TransactionOperation::TableReset(reset) => Some(reset.as_ref().clone()),
            TransactionOperation::Catalog(_)
            | TransactionOperation::Row(_)
            | TransactionOperation::TypedInsert(_) => None,
        })
        .unwrap();
    assert_eq!(
        reset.before_digest,
        engine
            .table_reset_device_root_proof(&table, reset.source_commit_seq, engine.committed_seq(),)
            .unwrap()
            .1
    );
    let shards = engine.read_state.residency.shards.load_full();
    let shard = shards["reset_digest_guard"]
        .iter()
        .find(|shard| shard.row_count != 0)
        .expect("global source shard must retain pre-reset rows")
        .clone();
    let descriptor = engine.resident_snapshot_for_shard(&shard, &table);
    let source_before = engine
        .table_reset_digest_source(
            &table,
            DeviceTableDigestSource {
                snapshot: &descriptor,
                memory: shard.device_memory.as_ref().unwrap(),
                row_count: shard.row_count as u64,
                row_ids: shard
                    .row_id_region
                    .as_ref()
                    .map(|region| (region.as_ref(), 0)),
                boundary: engine.committed_seq() as i64,
                deleted_by: shard
                    .deleted_by_region
                    .as_ref()
                    .map(|region| (region.as_ref(), 0)),
                created_by: shard
                    .created_by_region
                    .as_ref()
                    .map(|region| (region.as_ref(), 0)),
            },
        )
        .unwrap();
    let id_offset = resident_device_int4_column_offset(&descriptor, &table, 0).unwrap();
    shard
        .device_memory
        .as_ref()
        .unwrap()
        .append_owned_chunks([CudaOwnedDeviceMemoryChunk {
            byte_offset: id_offset,
            bytes: 99_i32.to_le_bytes().to_vec(),
        }])
        .unwrap();
    assert_eq!(
        shard
            .device_memory
            .as_ref()
            .unwrap()
            .read_resident_i32_column(id_offset, 1)
            .unwrap(),
        vec![99]
    );
    let source_after = engine
        .table_reset_digest_source(
            &table,
            DeviceTableDigestSource {
                snapshot: &descriptor,
                memory: shard.device_memory.as_ref().unwrap(),
                row_count: shard.row_count as u64,
                row_ids: shard
                    .row_id_region
                    .as_ref()
                    .map(|region| (region.as_ref(), 0)),
                boundary: engine.committed_seq() as i64,
                deleted_by: shard
                    .deleted_by_region
                    .as_ref()
                    .map(|region| (region.as_ref(), 0)),
                created_by: shard
                    .created_by_region
                    .as_ref()
                    .map(|region| (region.as_ref(), 0)),
            },
        )
        .unwrap();
    assert_ne!(source_before, source_after);
    assert_ne!(
        reset.before_digest,
        engine
            .table_reset_device_root_proof(&table, reset.source_commit_seq, engine.committed_seq(),)
            .unwrap()
            .1
    );
    let error = engine
        .submit_transaction(902, parsed("COMMIT"))
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
    assert_eq!(engine.durable_wal_records().len(), durable_before);
    engine.submit_transaction(902, parsed("ROLLBACK")).unwrap();

    let clean = Engine::new_local();
    clean.set_shard_residency_enabled(true);
    clean.set_auto_admit_on_commit(true);
    clean
        .submit_transaction(
            910,
            parsed("CREATE TABLE reset_digest_replay (id INT PRIMARY KEY, note TEXT)"),
        )
        .unwrap();
    clean
        .submit_transaction(
            911,
            parsed("INSERT INTO reset_digest_replay VALUES (1, NULL), (2, 'two')"),
        )
        .unwrap();
    clean
        .submit_transaction(912, parsed("TRUNCATE reset_digest_replay"))
        .unwrap();
    let durable = clean.durable_wal_records();
    let reset_entry = durable.last().unwrap().clone();
    let prefix = Engine::recover_from_durable_wal(&durable[..durable.len() - 1]).unwrap();
    let mut record = match decode_binary_record(&reset_entry.payload).unwrap() {
        BinaryWalRecord::Transaction(record) => record,
        _ => panic!("expected typed reset transaction"),
    };
    record.table_resets[0].before_digest[0] ^= 0x80;
    let replay_entry = LogEntry {
        term: 1,
        index: prefix.committed_seq() + 1,
        payload: Arc::clone(&reset_entry.payload),
    };
    let mut catalog = prefix
        .catalog_latch
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let error = prefix
        .apply_binary_transaction_record(&replay_entry, &mut catalog, record)
        .unwrap_err();
    assert!(
        error.to_string().contains("before-root proof mismatch"),
        "{error}"
    );
    assert_eq!(
        prefix
            .execute_relational_select_text("SELECT id FROM reset_digest_replay ORDER BY id")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(2)]]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn multiple_transactional_resets_publish_one_atomic_typed_record() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            105,
            parsed("CREATE TABLE reset_many_a (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            106,
            parsed("CREATE TABLE reset_many_b (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(107, parsed("INSERT INTO reset_many_a VALUES (1)"))
        .unwrap();
    engine
        .submit_transaction(108, parsed("INSERT INTO reset_many_b VALUES (1)"))
        .unwrap();

    engine.submit_transaction(109, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(109, parsed("TRUNCATE reset_many_b"))
        .unwrap();
    engine
        .submit_transaction(109, parsed("INSERT INTO reset_many_a VALUES (2)"))
        .unwrap();
    engine
        .submit_transaction(109, parsed("TRUNCATE reset_many_a"))
        .unwrap();
    engine.submit_transaction(109, parsed("COMMIT")).unwrap();

    let records = engine.durable_wal_records();
    let record = match decode_binary_record(&records.last().unwrap().payload).unwrap() {
        BinaryWalRecord::Transaction(record) => record,
        _ => panic!("expected typed reset transaction"),
    };
    assert_eq!(
        record
            .table_resets
            .iter()
            .map(|reset| (reset.table.as_str(), reset.ordinal))
            .collect::<Vec<_>>(),
        vec![("reset_many_b", 0), ("reset_many_a", 2)]
    );
    assert!(record.mutations.is_empty());
    let envelope = gpu_db_wal::decode_canonical_record_payload(&records.last().unwrap().payload)
        .unwrap()
        .unwrap();
    assert_eq!(envelope.header.table_block_count, 2);
    for table_name in ["reset_many_a", "reset_many_b"] {
        assert!(engine
            .execute_relational_select_text(&format!("SELECT id FROM {table_name}"))
            .unwrap()
            .rows
            .is_empty());
        let table = engine.catalog_snapshot().relational_catalog[table_name].clone();
        assert_eq!(
            engine.zero_row_resident_generation_boundary(&table),
            Some(engine.committed_seq())
        );
    }

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    for table_name in ["reset_many_a", "reset_many_b"] {
        assert!(recovered
            .execute_relational_select_text(&format!("SELECT id FROM {table_name}"))
            .unwrap()
            .rows
            .is_empty());
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn repeatable_read_first_access_after_reset_binds_the_typed_empty_root() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            110,
            parsed("CREATE TABLE reset_fence_target (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            111,
            parsed("CREATE TABLE reset_fence_other (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(112, parsed("INSERT INTO reset_fence_target VALUES (1)"))
        .unwrap();
    engine
        .submit_transaction(113, parsed("INSERT INTO reset_fence_other VALUES (7)"))
        .unwrap();

    engine
        .submit_transaction(114, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    let unrelated = match parse_command("SELECT id FROM reset_fence_other").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(114, &unrelated)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(7)]]
    );
    let retained_boundary = engine.transaction_snapshot_handle(114).unwrap().boundary;

    engine.submit_transaction(115, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(115, parsed("TRUNCATE reset_fence_target"))
        .unwrap();
    engine.submit_transaction(115, parsed("COMMIT")).unwrap();
    assert!(engine.committed_seq() > retained_boundary);

    let target = match parse_command("SELECT id FROM reset_fence_target").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    let old_transaction = engine
        .execute_relational_select_in_transaction(114, &target)
        .unwrap();
    assert!(old_transaction.rows.is_empty());
    assert!(engine
        .transaction_snapshot_handle(114)
        .unwrap()
        .table_is_rewrite_fenced("reset_fence_target"));
    engine.submit_transaction(114, parsed("ROLLBACK")).unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn repeatable_read_fenced_root_preserves_later_private_insert_and_commit() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            116,
            parsed("CREATE TABLE fenced_private_target (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            117,
            parsed("CREATE TABLE fenced_private_other (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(118, parsed("INSERT INTO fenced_private_target VALUES (1)"))
        .unwrap();
    engine
        .submit_transaction(119, parsed("INSERT INTO fenced_private_other VALUES (9)"))
        .unwrap();

    engine
        .submit_transaction(120, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    let other = match parse_command("SELECT id FROM fenced_private_other").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(120, &other)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(9)]]
    );

    engine
        .submit_transaction(121, parsed("TRUNCATE fenced_private_target"))
        .unwrap();
    engine
        .submit_transaction(120, parsed("INSERT INTO fenced_private_target VALUES (2)"))
        .unwrap();
    let target = match parse_command("SELECT id FROM fenced_private_target").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(120, &target)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(2)]],
        "a historical rewrite fence substitutes the base once; it must not mask the private overlay"
    );
    let duplicate = engine
        .submit_transaction(120, parsed("INSERT INTO fenced_private_target VALUES (2)"))
        .unwrap_err();
    assert!(
        duplicate.to_string().contains("duplicate key"),
        "private uniqueness must see the post-fence insert: {duplicate}"
    );
    engine.submit_transaction(120, parsed("COMMIT")).unwrap();
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id FROM fenced_private_target")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(2)]]
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .execute_relational_select_text("SELECT id FROM fenced_private_target")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(2)]]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn post_fence_writer_still_conflicts_with_repeatable_read_private_insert() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            130,
            parsed("CREATE TABLE fenced_conflict_target (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            131,
            parsed("CREATE TABLE fenced_conflict_other (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(132, parsed("INSERT INTO fenced_conflict_target VALUES (1)"))
        .unwrap();
    engine
        .submit_transaction(133, parsed("INSERT INTO fenced_conflict_other VALUES (9)"))
        .unwrap();
    engine
        .submit_transaction(134, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    let other = match parse_command("SELECT id FROM fenced_conflict_other").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    engine
        .execute_relational_select_in_transaction(134, &other)
        .unwrap();
    engine
        .submit_transaction(135, parsed("TRUNCATE fenced_conflict_target"))
        .unwrap();
    engine
        .submit_transaction(134, parsed("INSERT INTO fenced_conflict_target VALUES (2)"))
        .unwrap();
    engine
        .submit_transaction(136, parsed("INSERT INTO fenced_conflict_target VALUES (2)"))
        .unwrap();

    let error = engine
        .submit_transaction(134, parsed("COMMIT"))
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
    engine.submit_transaction(134, parsed("ROLLBACK")).unwrap();
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id FROM fenced_conflict_target")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(2)]]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn every_live_autocommit_truncate_surface_emits_typed_reset_wal() {
    fn last_reset(engine: &Engine, expected_rows: u64) -> BinaryTransactionTableReset {
        let durable = engine.durable_wal_records();
        let record = match decode_binary_record(&durable.last().unwrap().payload).unwrap() {
            BinaryWalRecord::Transaction(record) => record,
            _ => panic!("live TRUNCATE must emit a typed transaction record"),
        };
        assert_eq!(record.table_resets.len(), 1);
        assert!(record.mutations.is_empty());
        let reset = record.table_resets.into_iter().next().unwrap();
        assert_eq!(reset.table, "autocommit_reset");
        assert_eq!(reset.expected_rows, expected_rows);
        let envelope =
            gpu_db_wal::decode_canonical_record_payload(&durable.last().unwrap().payload)
                .unwrap()
                .unwrap();
        assert_eq!(
            envelope.fragments[0].kind,
            gpu_db_wal::CanonicalFragmentKind::TableReset
        );
        reset
    }

    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            200,
            parsed("CREATE TABLE autocommit_reset (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            201,
            parsed("INSERT INTO autocommit_reset VALUES (1), (2), (3)"),
        )
        .unwrap();
    engine
        .submit_transaction(202, parsed("DELETE FROM autocommit_reset WHERE id = 2"))
        .unwrap();
    let before_submit = engine.durable_wal_records().len();
    let table = engine.catalog_snapshot().relational_catalog["autocommit_reset"].clone();
    let source_commit_seq = engine.test_table_root_index("autocommit_reset");
    let expected_before = engine
        .table_reset_device_root_proof(&table, source_commit_seq, engine.committed_seq())
        .unwrap()
        .1;
    engine
        .submit_transaction(203, parsed("TRUNCATE autocommit_reset"))
        .unwrap();
    assert!(engine.transaction_snapshot_handle(203).is_none());
    assert_eq!(engine.durable_wal_records().len(), before_submit + 1);
    let first = last_reset(&engine, 2);
    assert_ne!(first.source_commit_seq, 0);
    assert_eq!(first.before_digest, expected_before);

    engine
        .submit_transaction(204, parsed("INSERT INTO autocommit_reset VALUES (4)"))
        .unwrap();
    let before_raw = engine.durable_wal_records().len();
    engine
        .execute_text(205, "TRUNCATE TABLE ONLY autocommit_reset")
        .unwrap();
    assert_eq!(engine.durable_wal_records().len(), before_raw + 1);
    last_reset(&engine, 1);

    engine
        .submit_transaction(206, parsed("INSERT INTO autocommit_reset VALUES (5)"))
        .unwrap();
    let before_enqueue = engine.durable_wal_records().len();
    engine
        .enqueue_set_text(207, "TRUNCATE autocommit_reset", Instant::now())
        .unwrap();
    assert_eq!(
        engine.batcher().len(),
        0,
        "TRUNCATE never enters SQL batching"
    );
    assert_eq!(engine.durable_wal_records().len(), before_enqueue + 1);
    last_reset(&engine, 1);
    assert!(engine
        .execute_relational_select_text("SELECT id FROM autocommit_reset")
        .unwrap()
        .rows
        .is_empty());

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .execute_relational_select_text("SELECT id FROM autocommit_reset")
        .unwrap()
        .rows
        .is_empty());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn retained_repeatable_read_access_serializes_a_competing_reset_pre_effect() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            120,
            parsed("CREATE TABLE guarded_reset_target (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(121, parsed("INSERT INTO guarded_reset_target VALUES (1)"))
        .unwrap();

    engine
        .submit_transaction(122, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    let target = match parse_command("SELECT id FROM guarded_reset_target").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(122, &target)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1)]]
    );

    engine.submit_transaction(123, parsed("BEGIN")).unwrap();
    let wal_before = engine.durable_wal_records().len();
    let error = engine
        .submit_transaction(123, parsed("TRUNCATE guarded_reset_target"))
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(engine
        .transaction_snapshot_handle(123)
        .unwrap()
        .transaction_delta_is_empty());
    engine.submit_transaction(123, parsed("ROLLBACK")).unwrap();
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(122, &target)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1)]]
    );
    engine.submit_transaction(122, parsed("ROLLBACK")).unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn reset_dependency_guard_conflicts_with_a_retained_foreign_key_parent_access() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            124,
            parsed("CREATE TABLE guarded_reset_parent (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            125,
            parsed("CREATE TABLE guarded_reset_child (id int4 PRIMARY KEY, parent_id int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            126,
            parsed("ALTER TABLE ONLY guarded_reset_child ADD CONSTRAINT guarded_reset_child_parent_fk FOREIGN KEY (parent_id) REFERENCES guarded_reset_parent(id)"),
        )
        .unwrap();
    engine
        .submit_transaction(127, parsed("INSERT INTO guarded_reset_parent VALUES (1)"))
        .unwrap();
    engine
        .submit_transaction(128, parsed("INSERT INTO guarded_reset_child VALUES (1, 1)"))
        .unwrap();

    engine
        .submit_transaction(129, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    let parent = match parse_command("SELECT id FROM guarded_reset_parent").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(129, &parent)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1)]]
    );

    engine.submit_transaction(130, parsed("BEGIN")).unwrap();
    let wal_before = engine.durable_wal_records().len();
    let error = engine
        .submit_transaction(130, parsed("TRUNCATE guarded_reset_child"))
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Serialization(_)));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(engine
        .transaction_snapshot_handle(130)
        .unwrap()
        .transaction_delta_is_empty());
    engine.submit_transaction(130, parsed("ROLLBACK")).unwrap();
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id FROM guarded_reset_child")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1)]]
    );
    engine.submit_transaction(129, parsed("ROLLBACK")).unwrap();
}

#[test]
fn transactional_reset_rejects_an_inbound_foreign_key_pre_effect() {
    let engine = Engine::new_local();
    engine
        .submit_transaction(
            130,
            parsed("CREATE TABLE reset_parent (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            131,
            parsed("CREATE TABLE reset_child (id int4 PRIMARY KEY, parent_id int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            132,
            parsed("ALTER TABLE ONLY reset_child ADD CONSTRAINT reset_child_parent_fk FOREIGN KEY (parent_id) REFERENCES reset_parent(id)"),
        )
        .unwrap();
    engine.submit_transaction(133, parsed("BEGIN")).unwrap();
    let wal_before = engine.durable_wal_records().len();

    let error = engine
        .submit_transaction(133, parsed("TRUNCATE reset_parent"))
        .unwrap_err();
    assert!(matches!(error, ExecuteError::Unsupported(_)));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(engine
        .transaction_snapshot_handle(133)
        .unwrap()
        .transaction_delta_is_empty());
    engine.submit_transaction(133, parsed("ROLLBACK")).unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transactional_reset_post_durable_failure_is_recovery_owned() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            140,
            parsed("CREATE TABLE durable_reset_target (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(141, parsed("INSERT INTO durable_reset_target VALUES (1)"))
        .unwrap();
    engine.submit_transaction(142, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(142, parsed("TRUNCATE durable_reset_target"))
        .unwrap();
    let durable_before = engine.durable_wal_records().len();
    engine.fail_next_transaction_post_durable_apply();

    let error = engine
        .submit_transaction(142, parsed("COMMIT"))
        .unwrap_err();
    assert!(error.is_indeterminate());
    assert!(engine.is_commit_path_poisoned());
    assert!(engine.transaction_snapshot_handle(142).is_some());
    let records = engine.durable_wal_records();
    assert_eq!(records.len(), durable_before + 1);
    let record = match decode_binary_record(&records.last().unwrap().payload).unwrap() {
        BinaryWalRecord::Transaction(record) => record,
        _ => panic!("expected typed reset transaction"),
    };
    assert_eq!(record.table_resets.len(), 1);
    assert!(record.mutations.is_empty());

    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert!(recovered
        .execute_relational_select_text("SELECT id FROM durable_reset_target")
        .unwrap()
        .rows
        .is_empty());
}

#[test]
fn old_transaction_rejects_drop_recreate_oid_aba_before_staging_rows() {
    for (suffix, replacement) in [
        ("same", "CREATE TABLE aba_same (id int4 PRIMARY KEY)"),
        (
            "different",
            "CREATE TABLE aba_different (id int4 PRIMARY KEY, payload int8)",
        ),
    ] {
        let engine = Engine::new_local_test_engine();
        let table = format!("aba_{suffix}");
        engine
            .submit_transaction(
                300,
                parsed(&format!("CREATE TABLE {table} (id int4 PRIMARY KEY)")),
            )
            .unwrap();
        engine
            .submit_transaction(304, parsed("CREATE TABLE aba_anchor (id int4 PRIMARY KEY)"))
            .unwrap();
        engine
            .submit_transaction(305, parsed("INSERT INTO aba_anchor VALUES (7)"))
            .unwrap();
        let original_oid = engine.catalog_snapshot().relational_catalog[&table].oid;
        engine
            .submit_transaction(301, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
            .unwrap();
        let anchor = match parse_command("SELECT id FROM aba_anchor").unwrap() {
            Command::Select(select) => select,
            _ => unreachable!(),
        };
        engine
            .execute_relational_select_in_transaction(301, &anchor)
            .unwrap();
        engine
            .submit_transaction(302, parsed(&format!("DROP TABLE {table}")))
            .unwrap();
        engine.submit_transaction(303, parsed(replacement)).unwrap();
        assert_ne!(
            engine.catalog_snapshot().relational_catalog[&table].oid,
            original_oid
        );
        let wal_before = engine.durable_wal_records().len();

        let error = engine
            .submit_transaction(301, parsed(&format!("INSERT INTO {table} (id) VALUES (1)")))
            .unwrap_err();

        assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert!(engine
            .transaction_snapshot_handle(301)
            .unwrap()
            .transaction_delta_is_empty());
        engine.submit_transaction(301, parsed("ROLLBACK")).unwrap();
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn empty_plain_table_reset_publishes_a_real_zero_row_device_root_and_recovers() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            310,
            parsed("CREATE TABLE empty_reset (id int4 PRIMARY KEY)"),
        )
        .unwrap();

    engine
        .submit_transaction(311, parsed("TRUNCATE empty_reset"))
        .unwrap();

    let table = engine.catalog_snapshot().relational_catalog["empty_reset"].clone();
    assert_eq!(
        engine.zero_row_resident_generation_boundary(&table),
        Some(2)
    );
    assert!(engine.table_device_authoritative("empty_reset"));
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    let recovered_table = recovered.catalog_snapshot().relational_catalog["empty_reset"].clone();
    assert!(recovered
        .zero_row_resident_generation_boundary(&recovered_table)
        .is_some());
    assert!(recovered
        .execute_relational_select_text("SELECT id FROM empty_reset")
        .unwrap()
        .rows
        .is_empty());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn autocommit_reset_clean_rejection_releases_the_id_for_an_exact_retry() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            320,
            parsed("CREATE TABLE reset_retry_clean (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(321, parsed("INSERT INTO reset_retry_clean VALUES (1)"))
        .unwrap();
    let oid = engine.catalog_snapshot().relational_catalog["reset_retry_clean"].oid;
    let reader = engine.table_access.lease();
    reader.acquire_shared([oid]).unwrap();
    let wal_before = engine.durable_wal_records().len();

    let error = engine
        .submit_transaction(322, parsed("TRUNCATE reset_retry_clean"))
        .unwrap_err();

    assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
    assert!(engine.transaction_snapshot_handle(322).is_none());
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    drop(reader);
    engine
        .submit_transaction(322, parsed("TRUNCATE reset_retry_clean"))
        .unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn autocommit_reset_retry_is_exact_idempotent_mismatch_safe_and_recoverable() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    for table in ["reset_retry_exact", "reset_retry_other"] {
        engine
            .submit_transaction(
                330 + u64::from(table.ends_with("other")),
                parsed(&format!("CREATE TABLE {table} (id int4 PRIMARY KEY)")),
            )
            .unwrap();
        engine
            .submit_transaction(
                332 + u64::from(table.ends_with("other")),
                parsed(&format!("INSERT INTO {table} VALUES (1)")),
            )
            .unwrap();
    }
    engine
        .submit_transaction(340, parsed("TRUNCATE reset_retry_exact"))
        .unwrap();
    let wal_after = engine.durable_wal_records().len();
    engine
        .submit_transaction(340, parsed("TRUNCATE reset_retry_exact"))
        .unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_after);
    let mismatch = engine
        .submit_transaction(340, parsed("TRUNCATE reset_retry_other"))
        .unwrap_err();
    assert!(
        mismatch.to_string().contains("different request"),
        "{mismatch}"
    );
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id FROM reset_retry_other")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1)]]
    );

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    let recovered_wal = recovered.durable_wal_records().len();
    recovered
        .submit_transaction(340, parsed("TRUNCATE reset_retry_exact"))
        .unwrap();
    assert_eq!(recovered.durable_wal_records().len(), recovered_wal);
    let recovered_mismatch = recovered
        .submit_transaction(340, parsed("TRUNCATE reset_retry_other"))
        .unwrap_err();
    assert!(
        recovered_mismatch.to_string().contains("different request"),
        "{recovered_mismatch}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn autocommit_reset_wal_flush_failure_is_cleanly_retryable_once() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            350,
            parsed("CREATE TABLE reset_retry_flush (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(351, parsed("INSERT INTO reset_retry_flush VALUES (1)"))
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    engine.simulate_next_wal_flush_failure();

    let error = engine
        .submit_transaction(352, parsed("TRUNCATE reset_retry_flush"))
        .unwrap_err();

    assert!(matches!(
        error,
        ExecuteError::Engine(EngineError::Durability(_))
    ));
    assert!(engine.transaction_snapshot_handle(352).is_none());
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    engine
        .submit_transaction(352, parsed("TRUNCATE reset_retry_flush"))
        .unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
    engine
        .submit_transaction(352, parsed("TRUNCATE reset_retry_flush"))
        .unwrap();
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn reset_visibility_proof_counts_nullable_rows_and_mvcc_lanes_not_null_payloads() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            360,
            parsed("CREATE TABLE reset_nullable (id int4 PRIMARY KEY, payload int4)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            361,
            parsed("INSERT INTO reset_nullable VALUES (1, NULL), (2, 20), (3, NULL)"),
        )
        .unwrap();
    engine
        .submit_transaction(362, parsed("DELETE FROM reset_nullable WHERE id = 2"))
        .unwrap();
    engine
        .submit_transaction(363, parsed("TRUNCATE reset_nullable"))
        .unwrap();

    let record = match decode_binary_record(&engine.durable_wal_records().last().unwrap().payload)
        .unwrap()
    {
        BinaryWalRecord::Transaction(record) => record,
        _ => panic!("expected typed reset WAL"),
    };
    assert_eq!(record.table_resets[0].expected_rows, 2);
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .execute_relational_select_text("SELECT id FROM reset_nullable")
        .unwrap()
        .rows
        .is_empty());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn grouped_drop_recreate_root_matches_record_replay_before_reset() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    let create = [(
        1,
        Arc::from(b"CREATE TABLE grouped_reset (id int4 PRIMARY KEY)".as_slice()),
    )];
    if let Err(failure) = engine.commit_mutation_batch(&create) {
        panic!("grouped reset create failed: {}", failure.error);
    }
    engine
        .execute_text(2, "INSERT INTO grouped_reset VALUES (1)")
        .expect("typed INSERT before grouped reset");
    let lifecycle = [
        (3, Arc::from(b"DROP TABLE grouped_reset".as_slice())),
        (
            4,
            Arc::from(b"CREATE TABLE grouped_reset (name text)".as_slice()),
        ),
    ];
    if let Err(failure) = engine.commit_mutation_batch(&lifecycle) {
        panic!("grouped reset lifecycle failed: {}", failure.error);
    }
    assert_eq!(
        engine.test_table_root_index("grouped_reset"),
        0,
        "the final OID must start at its own empty root"
    );
    engine
        .populate_relational_residency_snapshot("grouped_reset")
        .unwrap();

    engine
        .submit_transaction(500, parsed("TRUNCATE grouped_reset"))
        .unwrap();
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .execute_relational_select_text("SELECT name FROM grouped_reset")
        .unwrap()
        .rows
        .is_empty());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn repeatable_read_late_access_keeps_its_catalog_root_across_metadata_churn() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(600, parsed("CREATE TABLE rr_catalog_anchor (id int4)"))
        .unwrap();
    engine
        .submit_transaction(601, parsed("CREATE TABLE rr_catalog_target (id int4)"))
        .unwrap();
    engine
        .submit_transaction(602, parsed("INSERT INTO rr_catalog_anchor VALUES (1)"))
        .unwrap();
    engine
        .submit_transaction(603, parsed("INSERT INTO rr_catalog_target VALUES (7)"))
        .unwrap();
    engine
        .submit_transaction(604, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    let anchor = match parse_command("SELECT id FROM rr_catalog_anchor").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    engine
        .execute_relational_select_in_transaction(604, &anchor)
        .unwrap();

    engine
        .submit_transaction(
            605,
            parsed("ALTER TABLE rr_catalog_target RENAME TO rr_catalog_retired"),
        )
        .unwrap();
    engine
        .submit_transaction(606, parsed("CREATE TABLE rr_catalog_target (id int4)"))
        .unwrap();
    engine
        .submit_transaction(607, parsed("INSERT INTO rr_catalog_target VALUES (99)"))
        .unwrap();

    let target = match parse_command("SELECT id FROM rr_catalog_target").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(604, &target)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(7)]],
        "the held catalog binding must retain the old OID/root, not see its name's replacement"
    );
    engine.submit_transaction(604, parsed("ROLLBACK")).unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn read_committed_rebase_clears_prior_statement_rewrite_fence_membership() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            620,
            parsed("CREATE TABLE rc_fence_target (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(621, parsed("CREATE TABLE rc_fence_anchor (id int4)"))
        .unwrap();
    engine
        .submit_transaction(622, parsed("INSERT INTO rc_fence_target VALUES (1)"))
        .unwrap();
    engine
        .submit_transaction(623, parsed("INSERT INTO rc_fence_anchor VALUES (7)"))
        .unwrap();
    engine
        .submit_transaction(624, parsed("BEGIN ISOLATION LEVEL READ COMMITTED"))
        .unwrap();
    let anchor = match parse_command("SELECT id FROM rc_fence_anchor").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    engine
        .execute_relational_select_in_transaction(624, &anchor)
        .unwrap();
    let captured_before_reset = engine.transaction_snapshot_handle(624).unwrap();

    engine
        .submit_transaction(625, parsed("TRUNCATE rc_fence_target"))
        .unwrap();
    engine
        .acquire_transaction_table_access(&captured_before_reset, ["rc_fence_target".to_string()])
        .unwrap();
    assert!(captured_before_reset.table_is_rewrite_fenced("rc_fence_target"));
    engine
        .submit_transaction(626, parsed("INSERT INTO rc_fence_target VALUES (2)"))
        .unwrap();

    let target = match parse_command("SELECT id FROM rc_fence_target").unwrap() {
        Command::Select(select) => select,
        _ => unreachable!(),
    };
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(624, &target)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(2)]],
        "the next READ COMMITTED boundary contains the reset and must expose later rows"
    );
    assert!(!engine
        .transaction_snapshot_handle(624)
        .unwrap()
        .table_is_rewrite_fenced("rc_fence_target"));
    engine.submit_transaction(624, parsed("ROLLBACK")).unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn reset_drop_recreate_retires_root_and_fence_after_snapshot_epoch() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .submit_transaction(
            380,
            parsed("CREATE TABLE reset_churn (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    engine
        .submit_transaction(381, parsed("INSERT INTO reset_churn VALUES (1)"))
        .unwrap();
    let old_oid = engine.relational_catalog_table("reset_churn").unwrap().oid;
    engine
        .submit_transaction(382, parsed("BEGIN ISOLATION LEVEL REPEATABLE READ"))
        .unwrap();
    engine
        .submit_transaction(383, parsed("TRUNCATE reset_churn"))
        .unwrap();
    let fence = *engine
        .read_state
        .table_rewrite_fences
        .load()
        .get(&old_oid)
        .unwrap();
    assert!(fence > engine.transaction_snapshot_handle(382).unwrap().boundary);

    engine
        .submit_transaction(384, parsed("DROP TABLE reset_churn"))
        .unwrap();
    assert_eq!(engine.test_table_root_index("reset_churn"), 0);
    assert_eq!(
        engine.read_state.table_rewrite_fences.load().get(&old_oid),
        Some(&fence),
        "the old snapshot epoch still requires the dropped OID's fence"
    );

    engine.submit_transaction(382, parsed("ROLLBACK")).unwrap();
    engine
        .submit_transaction(385, parsed("CREATE TABLE reset_churn_peer (id int4)"))
        .unwrap();
    assert!(!engine
        .read_state
        .table_rewrite_fences
        .load()
        .contains_key(&old_oid));
    engine
        .submit_transaction(
            386,
            parsed("CREATE TABLE reset_churn (id int4 PRIMARY KEY)"),
        )
        .unwrap();
    assert_eq!(
        engine.test_table_root_index("reset_churn"),
        0,
        "a recreated name must start from its new OID's empty root"
    );

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .execute_relational_select_text("SELECT id FROM reset_churn")
        .unwrap()
        .rows
        .is_empty());
}
