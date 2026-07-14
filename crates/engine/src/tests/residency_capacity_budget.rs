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

/// S-F/R-1: the per-shard device hash index obeys the same hard cap as base payloads.
/// At a cap equal to the admitted shard+identity bytes, the optional index declines and the
/// sharded point path still returns the correct row through its GPU-scan/host-routing fallback.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sharded_device_index_declines_at_residency_budget() {
    let mut e = Engine::new_local_cpu_oracle();
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
    let result = e
        .gather_sharded_int4_point_lookups_batched(
            e.committed_seq(),
            &table,
            id,
            &[id, balance],
            &[20],
        )
        .expect("the capped index declines to the correct sharded fallback");
    assert_eq!(result.values, vec![20, 200]);
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
