/// `capacity > row_count` pads each i32 section to `capacity` (real values then zero headroom);
/// the header still records `row_count`; section offsets derive from `capacity`.
#[test]
fn capacity_padding_reserves_headroom_with_capacity_offsets() {
    let (names, types) = int4_cols();
    let rows = int4_rows(3);
    let capacity = 8;
    let payload = build_relational_device_payload_with_capacity(&names, &types, &rows, capacity)
        .unwrap()
        .0;
    // Layout: 8-byte header, int4 col0 (capacity*4), int4 col1 (capacity*4).
    assert_eq!(payload.len(), 8 + 2 * capacity * 4);
    let header = u64::from_le_bytes(payload[0..8].try_into().unwrap());
    assert_eq!(header, 3, "header records the live row count, not capacity");
    let at = |sec: usize, row: usize| -> i32 {
        let off = 8 + sec * capacity * 4 + row * 4;
        i32::from_le_bytes(payload[off..off + 4].try_into().unwrap())
    };
    // col0 (id): 0,1,2 then zero headroom.
    assert_eq!((at(0, 0), at(0, 2), at(0, 3), at(0, 7)), (0, 2, 0, 0));
    // col1 (balance): 0,10,20 then zero headroom.
    assert_eq!((at(1, 0), at(1, 2), at(1, 3)), (0, 20, 0));
}

/// An OPEN (capacity > row_count) payload rejects variable-length text; `capacity == row_count`
/// (dense) still accepts it. And `capacity < row_count` is always rejected.
#[test]
fn open_payload_rejects_text_and_undersize() {
    let names = vec!["id".to_string(), "name".to_string()];
    let types = vec![SqlType::Int4, SqlType::Text];
    let rows = vec![vec![SqlValue::Int4(1), SqlValue::Text("a".to_string())]];
    assert!(build_relational_device_payload_with_capacity(&names, &types, &rows, 1).is_ok());
    assert!(build_relational_device_payload_with_capacity(&names, &types, &rows, 4).is_err());
    let (names, types) = int4_cols();
    let rows = int4_rows(5);
    assert!(build_relational_device_payload_with_capacity(&names, &types, &rows, 3).is_err());
}

#[test]
fn named_index_budget_estimate_charges_distinct_device_keys_once() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE estimate_idx (id INT PRIMARY KEY, tenant_id INT, status INT)",
        )
        .unwrap();
    engine
        .execute_text(2, "CREATE INDEX estimate_idx_id_copy ON estimate_idx (id)")
        .unwrap();
    engine
        .execute_text(
            3,
            "CREATE INDEX estimate_idx_status ON estimate_idx (tenant_id, status)",
        )
        .unwrap();
    let table = engine.relational_catalog_table("estimate_idx").unwrap();
    let row_count = 3usize;
    let capacity = 8usize;
    let table_size = 32u64;
    let per_distinct_key = gpu_db_execution::resident_index_allocated_bytes(
        (table_size - 1) as u32,
        capacity as u64,
    )
    .unwrap();
    assert_eq!(
        estimated_named_index_bytes_for_shard(&table, row_count, capacity),
        Some(per_distinct_key * 2),
        "the PK and duplicate id index share one raw key allocation; the compound index owns one fingerprint allocation"
    );
}

/// S-F/R-1: the per-shard device hash index obeys the same hard cap as base payloads.
/// At a cap equal to the admitted shard+identity bytes, the optional index declines and the
/// sharded point path still returns the correct row through its GPU scan fallback.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_device_index_declines_at_residency_budget() {
    let mut e = Engine::new_local();
    e.set_auto_admit_on_commit(true);
    e.set_shard_residency_enabled(true);
    e.set_shard_index_probe_enabled(true);
    e.execute_text(1, "CREATE TABLE capped_shard_index (id INT, balance INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO capped_shard_index VALUES (10, 100), (20, 200), (30, 300)",
    )
    .unwrap();
    let admitted = e
        .populate_relational_residency_snapshot("capped_shard_index")
        .unwrap();
    if admitted.device_memory_proof.is_none() {
        return;
    }
    let budget = e.relational_resident_bytes_for_gpu(0);
    e.set_relational_residency_budget_bytes(0, budget);
    let table = e.relational_catalog_table("capped_shard_index").unwrap();
    let id = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let balance = crate::rel_exec_helpers::relational_column_index(&table, "balance").unwrap();
    assert!(
        e.gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            id,
            &[id, balance],
            &[20],
        )
        .expect("batched GPU route completed")
        .is_none(),
        "the capped optional index must decline"
    );
    let result = e
        .execute_relational_select_text(
            "SELECT id, balance FROM capped_shard_index WHERE id = 20",
        )
        .expect("the SQL route falls back to the device scan");
    assert_eq!(
        result.rows.row(0),
        &[SqlValue::Int4(20), SqlValue::Int4(200)]
    );
    assert!(
        e.read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .all(|entry| entry.device_index.is_none()),
        "no over-budget shard index may remain retained"
    );
    assert!(e.relational_resident_bytes_for_gpu(0) <= budget);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn active_transaction_pins_remain_globally_accounted_and_non_evictable() {
    let mut e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(10, "CREATE TABLE pinned_budget_a (id INT, value INT)")
        .unwrap();
    e.execute_text(11, "INSERT INTO pinned_budget_a VALUES (1, 10)")
        .unwrap();
    e.execute_text(12, "CREATE TABLE pinned_budget_b (id INT, value INT)")
        .unwrap();
    e.execute_text(13, "INSERT INTO pinned_budget_b VALUES (2, 20)")
        .unwrap();
    let current = e.relational_resident_bytes_for_gpu(0);
    let replacement = e.relational_resident_table_bytes_for_gpu("pinned_budget_b", 0);
    assert!(current > replacement && replacement > 0);

    e.execute_text(14, "BEGIN").unwrap();
    e.set_relational_residency_budget_bytes(0, current);
    let result = e.admit_relational_residency_snapshot(
        "pinned_budget_b",
        0,
        replacement.saturating_add(1),
    );
    assert!(result.is_err(), "a pinned base generation must not be evicted");
    let shards = e.read_residency_shards();
    assert!(shards.contains_key("pinned_budget_a"));
    assert!(shards.contains_key("pinned_budget_b"));
    assert_eq!(e.relational_resident_bytes_for_gpu(0), current);
    e.execute_text(14, "ROLLBACK").unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn same_table_replacement_cannot_hide_snapshot_retained_payload_from_budget() {
    let mut e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.execute_text(
        15,
        "CREATE TABLE pinned_same_table (id INT, value INT)",
    )
    .unwrap();
    e.execute_text(16, "INSERT INTO pinned_same_table VALUES (1, 10)")
        .unwrap();
    repair_test_relational_host_copy(&e, "pinned_same_table");
    let first = e
        .populate_relational_residency_snapshot("pinned_same_table")
        .unwrap();
    if first.device_memory_proof.is_none() {
        return;
    }
    let first_ptr = e
        .read_residency_shards()
        .get("pinned_same_table")
        .and_then(|shards| shards.first())
        .and_then(|shard| shard.device_memory.as_ref())
        .map(|memory| memory.device_ptr())
        .unwrap();
    let exact_current = e.relational_resident_bytes_for_gpu(0);
    assert!(exact_current > 0);

    e.execute_text(17, "BEGIN").unwrap();
    e.set_relational_residency_budget_bytes(0, exact_current);
    let replacement = e.populate_relational_residency_snapshot("pinned_same_table");
    assert!(
        replacement.is_err(),
        "a same-size replacement must count the transaction-pinned old allocation instead of excluding the target table wholesale"
    );
    let after_ptr = e
        .read_residency_shards()
        .get("pinned_same_table")
        .and_then(|shards| shards.first())
        .and_then(|shard| shard.device_memory.as_ref())
        .map(|memory| memory.device_ptr())
        .unwrap();
    assert_eq!(after_ptr, first_ptr, "failed admission cannot publish a replacement");
    assert_eq!(e.relational_resident_bytes_for_gpu(0), exact_current);
    e.execute_text(17, "ROLLBACK").unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_retained_index_stays_charged_after_cache_purge() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_index_probe_enabled(true);
    e.execute_text(
        20,
        "CREATE TABLE pinned_index_budget (id int4 PRIMARY KEY, value int4)",
    )
    .unwrap();
    e.execute_text(
        21,
        "INSERT INTO pinned_index_budget VALUES (1, 10), (2, 20), (3, 30)",
    )
    .unwrap();
    let table = e.relational_catalog_table("pinned_index_budget").unwrap();
    let id = crate::rel_exec_helpers::relational_column_index(&table, "id").unwrap();
    let value = crate::rel_exec_helpers::relational_column_index(&table, "value").unwrap();
    let _ = e
        .gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            id,
            &[id, value],
            &[2],
        )
        .unwrap();
    // Isolate the retained CudaResidentDeviceMemory charge from route-plan descriptor charges.
    e.read_state
        .residency
        .sharded_point_routes
        .store(Arc::new(BTreeMap::new()));
    e.read_state
        .residency
        .compound_point_routes
        .store(Arc::new(BTreeMap::new()));
    let index_allocations = e
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .values()
        .filter_map(|entry| entry.device_index.as_ref())
        .map(|memory| (memory.device_ptr(), memory.metadata().allocated_bytes))
        .collect::<BTreeMap<_, _>>();
    let index_bytes = index_allocations.values().sum::<u64>();
    assert!(index_bytes > 0, "the fixture must own a device index");
    let before = e.relational_resident_bytes_for_gpu(0);

    e.execute_text(22, "BEGIN").unwrap();
    e.read_state
        .residency
        .purge_shard_pk_index_for_table("pinned_index_budget");
    assert!(e
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_empty());
    assert_eq!(
        e.relational_resident_bytes_for_gpu(0),
        before,
        "the active snapshot's retired index allocation must remain charged"
    );
    e.execute_text(22, "ROLLBACK").unwrap();
    assert!(
        e.relational_resident_bytes_for_gpu(0) <= before.saturating_sub(index_bytes),
        "at least the purged index allocation must leave with the final transaction generation"
    );
    assert!(e
        .transaction_retained_gpu_allocations
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_empty());
}
