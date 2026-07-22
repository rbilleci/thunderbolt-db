use super::*;

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn chunk_authoritative_reset_fence_supports_old_rr_and_rc_private_writes_and_recovery() {
    let mut engine = Engine::new_local_test_engine();
    let mut seq = 0_u64;
    if !gpu_available(&mut engine, &mut seq) {
        return;
    }
    seq += 1;
    engine
        .execute_text(
            seq,
            "CREATE TABLE chunk_reset_rr (id INT PRIMARY KEY, v INT)",
        )
        .unwrap();
    seq += 1;
    engine
        .execute_text(
            seq,
            "CREATE TABLE chunk_reset_rc (id INT PRIMARY KEY, v INT)",
        )
        .unwrap();
    seq += 1;
    engine
        .execute_text(seq, "CREATE TABLE chunk_reset_anchor (id INT PRIMARY KEY)")
        .unwrap();
    let rows = (0..900)
        .map(|id| format!("({id}, {id})"))
        .collect::<Vec<_>>()
        .join(",");
    for table in ["chunk_reset_rr", "chunk_reset_rc"] {
        seq += 1;
        engine
            .execute_text(seq, &format!("INSERT INTO {table} VALUES {rows}"))
            .unwrap();
    }
    seq += 1;
    engine
        .execute_text(seq, "INSERT INTO chunk_reset_anchor VALUES (1)")
        .unwrap();
    engine.set_relational_residency_budget_bytes(0, 8192);
    for table in ["chunk_reset_rr", "chunk_reset_rc"] {
        engine
            .execute_relational_select(&select(&format!("SELECT COUNT(*) FROM {table}")))
            .unwrap();
        seq += 1;
        engine
            .execute_text(seq, &format!("INSERT INTO {table} VALUES (100000, 100000)"))
            .unwrap();
        assert!(engine.table_chunk_authoritative(table).is_some());
    }
    engine.clear_relational_residency_budget_bytes(0);

    seq += 1;
    let rr = seq;
    engine
        .execute_text(rr, "BEGIN ISOLATION LEVEL REPEATABLE READ")
        .unwrap();
    engine
        .execute_relational_select_in_transaction(rr, &select("SELECT id FROM chunk_reset_anchor"))
        .unwrap();
    seq += 1;
    engine.execute_text(seq, "TRUNCATE chunk_reset_rr").unwrap();
    assert!(engine
        .execute_relational_select_in_transaction(rr, &select("SELECT id FROM chunk_reset_rr"))
        .unwrap()
        .rows
        .is_empty());
    engine
        .execute_text(rr, "INSERT INTO chunk_reset_rr VALUES (200001, 1)")
        .unwrap();
    engine
        .execute_text(rr, "INSERT INTO chunk_reset_rr VALUES (200002, 2)")
        .unwrap();
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(
                rr,
                &select("SELECT id FROM chunk_reset_rr ORDER BY id"),
            )
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(200001)], vec![SqlValue::Int4(200002)]]
    );
    engine.execute_text(rr, "COMMIT").unwrap();

    seq += 1;
    let rc = seq;
    engine
        .execute_text(rc, "BEGIN ISOLATION LEVEL READ COMMITTED")
        .unwrap();
    seq += 1;
    engine.execute_text(seq, "TRUNCATE chunk_reset_rc").unwrap();
    assert!(engine
        .execute_relational_select_in_transaction(rc, &select("SELECT id FROM chunk_reset_rc"))
        .unwrap()
        .rows
        .is_empty());
    engine
        .execute_text(rc, "INSERT INTO chunk_reset_rc VALUES (300001, 1)")
        .unwrap();
    engine
        .execute_text(rc, "INSERT INTO chunk_reset_rc VALUES (300002, 2)")
        .unwrap();
    assert_eq!(
        engine
            .execute_relational_select_in_transaction(
                rc,
                &select("SELECT id FROM chunk_reset_rc ORDER BY id"),
            )
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(300001)], vec![SqlValue::Int4(300002)]]
    );
    engine.execute_text(rc, "COMMIT").unwrap();

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    for (table, first) in [("chunk_reset_rr", 200001), ("chunk_reset_rc", 300001)] {
        assert_eq!(
            recovered
                .execute_relational_select(&select(&format!("SELECT id FROM {table} ORDER BY id")))
                .unwrap()
                .rows,
            vec![vec![SqlValue::Int4(first)], vec![SqlValue::Int4(first + 1)]]
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn reset_rebase_and_repeated_chunk_resets_keep_typed_roots_and_recover() {
    fn assert_private_empty_root(engine: &Engine, txn_id: u64, table: &str) {
        let snapshot = engine
            .transaction_snapshot_handle(txn_id)
            .expect("active transaction snapshot");
        let shards = snapshot.transaction_shards();
        let root = shards.get(table).expect("private table root");
        assert_eq!(root.len(), 1, "typed empty root is one allocated shard");
        assert_eq!(root[0].row_count, 0);
        assert!(root[0].device_memory.is_some());
    }

    let mut engine = Engine::new_local_test_engine();
    let mut seq = 0_u64;
    if !gpu_available(&mut engine, &mut seq) {
        return;
    }
    for table in [
        "reset_rebase_hot",
        "reset_rebase_chunk",
        "reset_rebase_peer",
    ] {
        seq += 1;
        engine
            .execute_text(
                seq,
                &format!("CREATE TABLE {table} (id INT PRIMARY KEY, v INT)"),
            )
            .unwrap();
    }
    let rows = (0..900)
        .map(|id| format!("({id}, {id})"))
        .collect::<Vec<_>>()
        .join(",");
    for table in ["reset_rebase_hot", "reset_rebase_chunk"] {
        seq += 1;
        engine
            .execute_text(seq, &format!("INSERT INTO {table} VALUES {rows}"))
            .unwrap();
    }
    engine.set_relational_residency_budget_bytes(0, 8192);
    engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM reset_rebase_chunk"))
        .unwrap();
    seq += 1;
    engine
        .execute_text(
            seq,
            "INSERT INTO reset_rebase_chunk VALUES (100000, 100000)",
        )
        .unwrap();
    assert!(engine
        .table_chunk_authoritative("reset_rebase_chunk")
        .is_some());
    engine.clear_relational_residency_budget_bytes(0);

    seq += 1;
    let hot_txn = seq;
    engine
        .execute_text(hot_txn, "BEGIN ISOLATION LEVEL READ COMMITTED")
        .unwrap();
    engine
        .execute_text(hot_txn, "TRUNCATE reset_rebase_hot")
        .unwrap();
    seq += 1;
    engine
        .execute_text(seq, "INSERT INTO reset_rebase_peer VALUES (1, 1)")
        .unwrap();
    assert!(engine
        .execute_relational_select_in_transaction(
            hot_txn,
            &select("SELECT id FROM reset_rebase_hot"),
        )
        .unwrap()
        .rows
        .is_empty());
    assert_private_empty_root(&engine, hot_txn, "reset_rebase_hot");
    engine
        .execute_text(hot_txn, "INSERT INTO reset_rebase_hot VALUES (200001, 1)")
        .unwrap();
    engine.execute_text(hot_txn, "COMMIT").unwrap();

    seq += 1;
    let chunk_txn = seq;
    engine
        .execute_text(chunk_txn, "BEGIN ISOLATION LEVEL READ COMMITTED")
        .unwrap();
    engine
        .execute_text(chunk_txn, "TRUNCATE reset_rebase_chunk")
        .unwrap();
    seq += 1;
    engine
        .execute_text(seq, "INSERT INTO reset_rebase_peer VALUES (2, 2)")
        .unwrap();
    assert!(engine
        .execute_relational_select_in_transaction(
            chunk_txn,
            &select("SELECT id FROM reset_rebase_chunk"),
        )
        .unwrap()
        .rows
        .is_empty());
    assert_private_empty_root(&engine, chunk_txn, "reset_rebase_chunk");
    engine
        .execute_text(chunk_txn, "TRUNCATE reset_rebase_chunk")
        .unwrap();
    engine
        .execute_text(
            chunk_txn,
            "INSERT INTO reset_rebase_chunk VALUES (300001, 1)",
        )
        .unwrap();
    engine
        .execute_text(chunk_txn, "TRUNCATE reset_rebase_chunk")
        .unwrap();
    engine
        .execute_text(
            chunk_txn,
            "INSERT INTO reset_rebase_chunk VALUES (300002, 2)",
        )
        .unwrap();
    engine.execute_text(chunk_txn, "COMMIT").unwrap();

    let record = match decode_binary_record(&engine.durable_wal_records().last().unwrap().payload)
        .unwrap()
    {
        BinaryWalRecord::Transaction(record) => record,
        _ => panic!("expected typed reset transaction"),
    };
    assert_eq!(record.table_resets.len(), 1);
    assert_eq!(record.table_resets[0].ordinal, 3);
    assert_eq!(record.mutations.len(), 1);

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    for (table, expected) in [("reset_rebase_hot", 200001), ("reset_rebase_chunk", 300002)] {
        assert_eq!(
            recovered
                .execute_relational_select(&select(&format!("SELECT id FROM {table}")))
                .unwrap()
                .rows,
            vec![vec![SqlValue::Int4(expected)]]
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn stale_non_authoritative_cold_cache_cannot_override_a_current_hot_reset_root() {
    let _forced_over_cap = ChunkKeyIndexCapOverride::tiny();
    let mut engine = Engine::new_local_test_engine();
    let mut seq = 0_u64;
    if !gpu_available(&mut engine, &mut seq) {
        return;
    }
    let rows = |base: i32| {
        (0..800)
            .map(|i| format!("({}, {i})", base + i))
            .collect::<Vec<_>>()
            .join(",")
    };
    seq += 1;
    engine
        .execute_text(
            seq,
            "CREATE TABLE reset_cap_owner (id INT PRIMARY KEY, v INT)",
        )
        .unwrap();
    seq += 1;
    engine
        .execute_text(
            seq,
            "CREATE TABLE reset_stale_cold (id INT PRIMARY KEY, v INT)",
        )
        .unwrap();
    seq += 1;
    engine
        .execute_text(
            seq,
            &format!("INSERT INTO reset_cap_owner VALUES {}", rows(0)),
        )
        .unwrap();
    seq += 1;
    engine
        .execute_text(
            seq,
            &format!("INSERT INTO reset_stale_cold VALUES {}", rows(10000)),
        )
        .unwrap();
    engine.set_relational_residency_budget_bytes(0, 4096);
    engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM reset_cap_owner"))
        .unwrap();
    seq += 1;
    engine
        .execute_text(seq, "INSERT INTO reset_cap_owner VALUES (100000, 1)")
        .unwrap();
    let first_bytes = engine.chunk_key_bloom_bytes();
    assert!(first_bytes > 0);
    crate::engine_streaming_exec::CHUNK_KEY_BLOOM_CAP_BYTES_TEST.store(
        first_bytes + first_bytes / 2,
        std::sync::atomic::Ordering::Relaxed,
    );
    engine
        .execute_relational_select(&select("SELECT COUNT(*) FROM reset_stale_cold"))
        .unwrap();
    assert!(engine
        .table_chunk_authoritative("reset_stale_cold")
        .is_none());
    assert!(engine
        .read_streaming_cold_chunks()
        .contains_key("reset_stale_cold"));

    engine.clear_relational_residency_budget_bytes(0);
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .populate_relational_residency_snapshot("reset_stale_cold")
        .unwrap();
    // Warmup installed a real current shard root. This test seam marks that already-backed root
    // authoritative so the following DML maintains it in place while the declined cold cache is
    // deliberately retained as the stale competitor under test.
    engine.set_table_device_authoritative("reset_stale_cold", true);
    assert!(engine.table_device_authoritative("reset_stale_cold"));
    assert!(engine
        .read_streaming_cold_chunks()
        .contains_key("reset_stale_cold"));
    seq += 1;
    engine
        .execute_text(seq, "INSERT INTO reset_stale_cold VALUES (200000, 9)")
        .unwrap();
    seq += 1;
    engine
        .execute_text(seq, "TRUNCATE reset_stale_cold")
        .unwrap();
    let reset = match decode_binary_record(&engine.durable_wal_records().last().unwrap().payload)
        .unwrap()
    {
        BinaryWalRecord::Transaction(record) => record.table_resets.into_iter().next().unwrap(),
        _ => panic!("expected typed table reset WAL"),
    };
    assert_eq!(reset.expected_rows, 801);
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .execute_relational_select(&select("SELECT id FROM reset_stale_cold"))
        .unwrap()
        .rows
        .is_empty());
}
