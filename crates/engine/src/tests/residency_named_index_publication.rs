pub(super) fn resident_named_index_cache_entry(
    engine: &Engine,
    table: &str,
    shard_id: u32,
    key_id: usize,
) -> (Arc<CudaResidentDeviceMemory>, u32, u32, usize) {
    let cache = engine
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = cache
        .get(&(table.to_string(), shard_id, key_id))
        .unwrap_or_else(|| {
            panic!(
                "named index cache entry ({table}, {shard_id}, {key_id}); present={:?}",
                cache.keys().collect::<Vec<_>>()
            )
        });
    (
        entry
            .device_index
            .clone()
            .expect("named index must not be a declined cache entry"),
        entry.table_mask,
        entry.hash_shift,
        entry.row_count,
    )
}

pub(super) fn resident_named_index_physical_hit_count(
    index: &Arc<CudaResidentDeviceMemory>,
    table_mask: u32,
    hash_shift: u32,
    row_count: usize,
    needle: i32,
) -> u32 {
    index
        .submit_multi_shard_i32_write_locate(
            &[WriteLocateShard {
                index: Arc::clone(index),
                table_mask,
                hash_shift,
                row_count: row_count as u32,
            }],
            &[needle],
            u32::try_from(row_count.max(8)).expect("test row count fits u32"),
        )
        .expect("probe named resident index")
        .count[0]
}

/// PRODUCT-002: publication covers the primary and non-unique compound secondary index, accepts
/// repeated secondary keys, and incrementally extends the exact device allocation through INSERT
/// and UPDATE physical-version appends. The NULL-bearing column in the second table is deliberately
/// unreferenced: it must not turn exact NOT-NULL named indexes into a whole-table decline.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_named_indexes_publish_and_extend_without_rebuild() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.set_shard_size_target(64);
    engine
        .execute_text(
            1,
            "CREATE TABLE named_idx (id INT, tenant_id INT, status SMALLINT, PRIMARY KEY (id))",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE INDEX named_idx_by_status ON named_idx (tenant_id, status)",
        )
        .unwrap();
    engine
        .execute_text(3, "INSERT INTO named_idx VALUES (1, 7, 2), (2, 7, 2)")
        .unwrap();

    let report = engine
        .publish_relational_resident_indexes("named_idx")
        .expect("publish every named index");
    assert_eq!(report.table, "named_idx");
    assert_eq!(report.indexed_rows, 2);
    assert_eq!(report.shard_count, 1);
    assert!(report.allocated_bytes > 0);
    assert_eq!(
        report
            .indexes
            .iter()
            .map(|entry| (
                entry.index.as_str(),
                entry.key_columns.as_slice(),
                entry.unique,
                entry.indexed_rows,
            ))
            .collect::<Vec<_>>(),
        vec![
            ("named_idx_pkey", ["id".to_string()].as_slice(), true, 2),
            (
                "named_idx_by_status",
                ["tenant_id".to_string(), "status".to_string()].as_slice(),
                false,
                2,
            ),
        ]
    );

    let table = engine.relational_catalog_table("named_idx").unwrap();
    let secondary_ordinal = table
        .indexes
        .iter()
        .position(|index| index.name == "named_idx_by_status")
        .unwrap();
    let secondary_key_id = crate::engine_residency::index_probe_key_id(
        &table,
        &table.indexes[secondary_ordinal],
        secondary_ordinal,
    )
    .unwrap();
    let shard_id = engine.read_residency_shards()["named_idx"]
        .iter()
        .find(|shard| shard.row_count != 0)
        .expect("non-empty named-index shard")
        .shard_id;
    let (index_before, table_mask, hash_shift, rows_before) =
        resident_named_index_cache_entry(&engine, "named_idx", shard_id, secondary_key_id);
    assert_eq!(rows_before, 2);
    let old_key = crate::engine_residency::compound_key_fingerprint(&[7, 2]);
    assert_eq!(
        resident_named_index_physical_hit_count(
            &index_before,
            table_mask,
            hash_shift,
            rows_before,
            old_key,
        ),
        2,
        "duplicate non-unique compound keys occupy distinct candidate slots"
    );

    let rebuilds_before = engine
        .read_state
        .residency
        .lane_diag_rebuilds
        .load(std::sync::atomic::Ordering::Relaxed);
    engine
        .execute_text(4, "INSERT INTO named_idx VALUES (3, 7, 2)")
        .unwrap();
    let (index_after_insert, table_mask, hash_shift, rows_after_insert) =
        resident_named_index_cache_entry(&engine, "named_idx", shard_id, secondary_key_id);
    assert!(Arc::ptr_eq(&index_before, &index_after_insert));
    assert_eq!(rows_after_insert, 3);
    assert_eq!(
        resident_named_index_physical_hit_count(
            &index_after_insert,
            table_mask,
            hash_shift,
            rows_after_insert,
            old_key,
        ),
        3
    );

    engine
        .execute_text(5, "UPDATE named_idx SET status = 3 WHERE id = 1")
        .unwrap();
    let (index_after_update, table_mask, hash_shift, rows_after_update) =
        resident_named_index_cache_entry(&engine, "named_idx", shard_id, secondary_key_id);
    assert!(Arc::ptr_eq(&index_before, &index_after_update));
    assert_eq!(rows_after_update, 4);
    let moved_key = crate::engine_residency::compound_key_fingerprint(&[7, 3]);
    assert_eq!(
        resident_named_index_physical_hit_count(
            &index_after_update,
            table_mask,
            hash_shift,
            rows_after_update,
            moved_key,
        ),
        1,
        "secondary-key movement must publish the new physical version"
    );
    assert_eq!(
        engine
            .read_state
            .residency
            .lane_diag_rebuilds
            .load(std::sync::atomic::Ordering::Relaxed),
        rebuilds_before,
        "post-publication mutations must extend O(k), not rebuild O(rows)"
    );

    engine
        .execute_text(
            6,
            "INSERT INTO named_idx VALUES (4, 8, 1), (5, 8, 2), (6, 8, 3), (7, 9, 1), (8, 9, 2), (9, 9, 3), (10, 10, 1), (11, 10, 2), (12, 10, 3)",
        )
        .unwrap();
    let (index_after_threshold, _, _, rows_after_threshold) =
        resident_named_index_cache_entry(&engine, "named_idx", shard_id, secondary_key_id);
    assert!(Arc::ptr_eq(&index_before, &index_after_threshold));
    assert_eq!(rows_after_threshold, 13);
    assert_eq!(
        engine
            .read_state
            .residency
            .lane_diag_rebuilds
            .load(std::sync::atomic::Ordering::Relaxed),
        rebuilds_before,
        "capacity-sized publication must survive well past the former four-row load threshold"
    );

    let republished = engine
        .publish_relational_resident_indexes("named_idx")
        .expect("current named indexes remain fully published");
    assert_eq!(republished.indexed_rows, 13);
    assert_eq!(
        republished
            .indexes
            .iter()
            .map(|entry| entry.indexed_rows)
            .collect::<Vec<_>>(),
        vec![13, 13]
    );
    assert_eq!(
        engine
            .read_state
            .residency
            .lane_diag_rebuilds
            .load(std::sync::atomic::Ordering::Relaxed),
        rebuilds_before,
        "publication proof over a maintained generation is cache-only"
    );

    engine
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&("named_idx".to_string(), shard_id, secondary_key_id));
    assert!(
        engine
            .execute_text(7, "INSERT INTO named_idx VALUES (13, 11, 1)")
            .is_err(),
        "a missing mandatory secondary must fail loudly; durable apply cannot downgrade to a partial index set"
    );

    let null_engine = Engine::new_local();
    null_engine.set_shard_residency_enabled(true);
    null_engine.set_auto_admit_on_commit(true);
    null_engine
        .execute_text(
            1,
            "CREATE TABLE named_idx_null (id INT, tenant_id INT, status SMALLINT, optional_code INT, PRIMARY KEY (id))",
        )
        .unwrap();
    null_engine
        .execute_text(
            2,
            "CREATE INDEX named_idx_null_by_status ON named_idx_null (tenant_id, status)",
        )
        .unwrap();
    null_engine
        .execute_text(
            3,
            "INSERT INTO named_idx_null VALUES (1, 8, 4, NULL), (2, 8, 4, 9)",
        )
        .unwrap();
    let null_report = null_engine
        .publish_relational_resident_indexes("named_idx_null")
        .expect("unreferenced NULL bitmap does not decline named indexes");
    assert_eq!(null_report.indexes.len(), 2);
    assert_eq!(null_report.indexed_rows, 2);
    let null_rebuilds_before = null_engine
        .read_state
        .residency
        .lane_diag_rebuilds
        .load(std::sync::atomic::Ordering::Relaxed);
    let null_visits_before = null_engine
        .read_state
        .residency
        .named_index_publication_shard_visits
        .load(std::sync::atomic::Ordering::Relaxed);
    null_engine
        .execute_text(4, "INSERT INTO named_idx_null VALUES (3, 8, 4, NULL)")
        .unwrap();
    assert_eq!(
        null_engine
            .read_state
            .residency
            .named_index_publication_shard_visits
            .load(std::sync::atomic::Ordering::Relaxed)
            - null_visits_before,
        2,
        "rollover publication visits one new shard for each of two named indexes"
    );
    let rollover_report = null_engine
        .publish_relational_resident_indexes("named_idx_null")
        .expect("NULL-bearing rollover publishes every named index before visibility");
    assert_eq!(rollover_report.indexed_rows, 3);
    assert_eq!(rollover_report.shard_count, 2);
    assert!(
        rollover_report
            .indexes
            .iter()
            .all(|entry| entry.indexed_rows == 3 && entry.shard_count == 2)
    );
    assert_eq!(
        null_engine
            .read_state
            .residency
            .lane_diag_rebuilds
            .load(std::sync::atomic::Ordering::Relaxed),
        null_rebuilds_before + 2,
        "rollover builds exactly the two named indexes for the new k-row shard"
    );
}

/// A capacity-horizon directory must remain one device allocation through every physical row that
/// can be appended to its open shard. This combines distinct inserts, duplicate secondary postings,
/// and an MVCC version append so the no-premature-purge guarantee covers the complete mutation mix.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_named_index_survives_full_physical_append_horizon_without_rebuild() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.set_shard_size_target(64);
    engine
        .execute_text(
            1,
            "CREATE TABLE capacity_idx (id INT, tenant_id INT, status SMALLINT, PRIMARY KEY (id))",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE INDEX capacity_idx_by_status ON capacity_idx (tenant_id, status)",
        )
        .unwrap();
    engine
        .execute_text(3, "INSERT INTO capacity_idx VALUES (1, 7, 2), (2, 7, 2)")
        .unwrap();
    engine
        .publish_relational_resident_indexes("capacity_idx")
        .unwrap();

    let table = engine.relational_catalog_table("capacity_idx").unwrap();
    let secondary_ordinal = table
        .indexes
        .iter()
        .position(|index| index.name == "capacity_idx_by_status")
        .unwrap();
    let secondary_key_id = crate::engine_residency::index_probe_key_id(
        &table,
        &table.indexes[secondary_ordinal],
        secondary_ordinal,
    )
    .unwrap();
    let shards = engine.read_residency_shards();
    let shard = shards["capacity_idx"]
        .iter()
        .find(|shard| shard.row_count != 0)
        .expect("one non-empty capacity fixture shard");
    assert_eq!(
        shard.capacity, 64,
        "the fixture keeps one open 64-row shard"
    );
    let shard_id = shard.shard_id;
    let (index_before, table_mask, hash_shift, rows_before) =
        resident_named_index_cache_entry(&engine, "capacity_idx", shard_id, secondary_key_id);
    let duplicate_key = crate::engine_residency::compound_key_fingerprint(&[7, 2]);
    assert_eq!(
        resident_named_index_physical_hit_count(
            &index_before,
            table_mask,
            hash_shift,
            rows_before,
            duplicate_key,
        ),
        2,
        "duplicate secondary keys must retain separate postings before the MVCC append"
    );
    let rebuilds_before = engine
        .read_state
        .residency
        .lane_diag_rebuilds
        .load(std::sync::atomic::Ordering::Relaxed);

    engine
        .execute_text(4, "UPDATE capacity_idx SET status = 3 WHERE id = 1")
        .unwrap();
    for id in 3..=63_i32 {
        engine
            .execute_text(
                100 + id as u64,
                &format!("INSERT INTO capacity_idx VALUES ({id}, 11, 1)"),
            )
            .unwrap();
    }

    let (index_at_capacity, _, _, rows_at_capacity) =
        resident_named_index_cache_entry(&engine, "capacity_idx", shard_id, secondary_key_id);
    assert!(Arc::ptr_eq(&index_before, &index_at_capacity));
    assert_eq!(rows_at_capacity, 64);
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id, status FROM capacity_idx WHERE id = 1")
            .unwrap()
            .rows
            .row(0),
        &[SqlValue::Int4(1), SqlValue::Int2(3)],
        "the full-lifetime posting chain must resolve the MVCC-visible version"
    );
    assert_eq!(
        engine
            .read_state
            .residency
            .lane_diag_rebuilds
            .load(std::sync::atomic::Ordering::Relaxed),
        rebuilds_before,
        "the resident index must not purge before physical capacity is consumed"
    );
}

/// Resident index validity descriptors omit NULL-bearing keys without folding the physical zero
/// placeholder. Publication and mandatory mutation coverage therefore preserve PostgreSQL
/// NULLS-DISTINCT uniqueness entirely on-device.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_named_index_publication_and_mutation_omit_indexed_nulls() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(
            1,
            "CREATE TABLE nullable_idx (id INT PRIMARY KEY, code INT)",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "INSERT INTO nullable_idx VALUES (1, NULL), (2, NULL), (3, 0)",
        )
        .expect("PostgreSQL unique indexes admit multiple NULLs");
    engine
        .execute_text(
            3,
            "CREATE UNIQUE INDEX nullable_idx_code ON nullable_idx (code)",
        )
        .expect("non-enrolled DDL preserves PostgreSQL NULL-distinct semantics");
    let report = engine
        .publish_relational_resident_indexes("nullable_idx")
        .expect("explicit indexed NULL publication uses device validity predicates");
    assert_eq!(report.indexed_rows, 3);
    assert_eq!(report.indexes.len(), 2);

    let mutation = Engine::new_local();
    mutation.set_shard_residency_enabled(true);
    mutation.set_auto_admit_on_commit(true);
    mutation
        .execute_text(
            1,
            "CREATE TABLE nullable_mut (id INT PRIMARY KEY, code INT)",
        )
        .unwrap();
    mutation
        .execute_text(
            2,
            "CREATE UNIQUE INDEX nullable_mut_code ON nullable_mut (code)",
        )
        .unwrap();
    mutation
        .execute_text(3, "INSERT INTO nullable_mut VALUES (1, 5)")
        .unwrap();
    mutation
        .publish_relational_resident_indexes("nullable_mut")
        .unwrap();
    mutation
        .execute_text(4, "INSERT INTO nullable_mut VALUES (2, NULL)")
        .expect("enrolled index omits the first NULL posting");
    mutation
        .execute_text(5, "INSERT INTO nullable_mut VALUES (3, NULL)")
        .expect("enrolled unique index admits a second NULL posting");
    assert!(
        mutation
            .execute_text(6, "INSERT INTO nullable_mut VALUES (4, 5)")
            .is_err()
    );
    assert_eq!(
        mutation
            .execute_relational_select_text("SELECT id, code FROM nullable_mut ORDER BY id")
            .unwrap()
            .rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(5)],
            vec![SqlValue::Int4(2), SqlValue::Null],
            vec![SqlValue::Int4(3), SqlValue::Null],
        ]
    );

    let lazy = Engine::new_local();
    lazy.set_shard_residency_enabled(true);
    lazy.set_auto_admit_on_commit(true);
    lazy.set_shard_size_target(64);
    lazy.set_device_write_locate_wave_batch_enabled(true);
    lazy.execute_text(
        1,
        "CREATE TABLE nullable_lazy (id INT PRIMARY KEY, code BIGINT UNIQUE)",
    )
    .unwrap();
    lazy.execute_dml_concurrent(2, "INSERT INTO nullable_lazy VALUES (1, 5)")
        .unwrap();
    lazy.execute_dml_concurrent(3, "INSERT INTO nullable_lazy VALUES (2, 6)")
        .unwrap();
    assert!(
        lazy.execute_dml_concurrent(4, "INSERT INTO nullable_lazy VALUES (9, 5)")
            .is_err(),
        "precondition: the wide unique index is validated on the device"
    );
    let lazy_table = lazy.relational_catalog_table("nullable_lazy").unwrap();
    assert!(
        !lazy
            .read_state
            .residency
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(&lazy_table.oid),
        "ordinary lazy validation must not implicitly enroll the table"
    );
    assert!(
        lazy.read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .any(|(table, _, _)| table == "nullable_lazy"),
        "precondition: unique validation populated a best-effort device index"
    );
    let lazy_shards = lazy.read_residency_shards();
    let lazy_shard = lazy_shards["nullable_lazy"]
        .iter()
        .find(|shard| shard.row_count != 0)
        .expect("resident lazy shard");
    let lazy_memory = lazy_shard
        .device_memory
        .as_ref()
        .expect("resident lazy shard device allocation");
    assert!(
        lazy.extend_shard_fingerprint_device_indexes_on_append(
            "nullable_lazy",
            lazy_shard.shard_id,
            lazy_memory.device_ptr(),
            lazy_shard.row_count,
            &[vec![SqlValue::Int4(3), SqlValue::Null]],
        ),
        "a non-enrolled lazy index retires its pre-rollover cache before the validity bitmap exists"
    );
    assert!(
        lazy.read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .all(|(table, _, _)| table != "nullable_lazy"),
        "the pending nullable rollover retires the stale best-effort cache"
    );
    drop(lazy_shards);
    lazy.execute_dml_concurrent(5, "INSERT INTO nullable_lazy VALUES (3, NULL)")
        .expect("a non-enrolled nullable unique index remains PostgreSQL-compatible");
    assert_eq!(
        lazy.execute_relational_select_text("SELECT id, code FROM nullable_lazy ORDER BY id")
            .unwrap()
            .rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int8(5)],
            vec![SqlValue::Int4(2), SqlValue::Int8(6)],
            vec![SqlValue::Int4(3), SqlValue::Null],
        ]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_named_index_enrollment_survives_rename_and_shape_ddl_but_not_table_recreate() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(
            1,
            "CREATE TABLE shape_idx (id INT PRIMARY KEY, tenant_id INT, status INT)",
        )
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO shape_idx VALUES (1, 7, 2)")
        .unwrap();
    engine
        .publish_relational_resident_indexes("shape_idx")
        .unwrap();
    let original_oid = engine.relational_catalog_table("shape_idx").unwrap().oid;

    engine
        .execute_text(3, "ALTER TABLE shape_idx RENAME TO renamed_shape_idx")
        .unwrap();
    let renamed = engine
        .relational_catalog_table("renamed_shape_idx")
        .unwrap();
    assert_eq!(renamed.oid, original_oid);
    assert!(
        engine.relational_named_index_publication_required(&renamed),
        "same-OID table rename must retain mandatory named-index enrollment"
    );
    engine
        .publish_relational_resident_indexes("renamed_shape_idx")
        .expect("renamed relation republishes under its new cache namespace");
    engine
        .execute_text(4, "INSERT INTO renamed_shape_idx VALUES (2, 8, 3)")
        .expect("renamed enrolled relation keeps mandatory mutation maintenance");

    engine
        .execute_text(
            5,
            "CREATE INDEX shape_idx_tenant_status ON renamed_shape_idx (tenant_id, status)",
        )
        .unwrap();
    let created = engine
        .publish_relational_resident_indexes("renamed_shape_idx")
        .expect("same-OID CREATE INDEX remains mandatory and republishes current shape");
    assert_eq!(created.indexes.len(), 2);

    engine
        .execute_text(6, "DROP INDEX shape_idx_tenant_status")
        .unwrap();
    let dropped = engine
        .publish_relational_resident_indexes("renamed_shape_idx")
        .expect("same-OID DROP INDEX remains mandatory and republishes current shape");
    assert_eq!(dropped.indexes.len(), 1);

    engine
        .execute_text(7, "DROP TABLE renamed_shape_idx")
        .unwrap();
    engine
        .execute_text(
            8,
            "CREATE TABLE shape_idx (id INT PRIMARY KEY, tenant_id INT, status INT)",
        )
        .unwrap();
    let recreated = engine.relational_catalog_table("shape_idx").unwrap();
    assert_ne!(recreated.oid, original_oid);
    assert!(
        !engine.relational_named_index_publication_required(&recreated),
        "a new table OID must not inherit the dropped table's mandatory enrollment"
    );
    engine
        .execute_text(9, "INSERT INTO shape_idx VALUES (1, 7, 2)")
        .unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_named_index_publication_rejects_concurrent_cache_retirement() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine
        .execute_text(
            1,
            "CREATE TABLE publish_race (id INT PRIMARY KEY, tenant_id INT, status INT)",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE INDEX publish_race_status ON publish_race (tenant_id, status)",
        )
        .unwrap();
    engine
        .execute_text(3, "INSERT INTO publish_race VALUES (1, 7, 2)")
        .unwrap();
    engine
        .publish_relational_resident_indexes("publish_race")
        .unwrap();

    let engine = Arc::new(engine);
    let reached = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    engine
        .set_named_index_publication_pre_linearize_hook(Arc::clone(&reached), Arc::clone(&resume));
    let publisher = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || engine.publish_relational_resident_indexes("publish_race"))
    };
    reached.wait();
    engine
        .read_state
        .residency
        .purge_shard_pk_index_for_table("publish_race");
    resume.wait();
    let error = publisher
        .join()
        .expect("publication thread")
        .expect_err("cache retirement must win over a stale success report");
    assert!(
        error.to_string().contains("cache changed"),
        "unexpected publication race error: {error}"
    );
}

/// The posting chain, rather than the directory collision bound, owns repeated physical versions
/// of one logical key. More than 256 same-key UPDATEs must remain append-only in one allocation and
/// the current visible row must still be found through the named index.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_named_index_survives_more_than_256_same_key_updates() {
    let engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.set_shard_size_target(1024);
    engine
        .execute_text(
            1,
            "CREATE TABLE version_chain (id INT PRIMARY KEY, tenant_id INT, status INT)",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE INDEX version_chain_status ON version_chain (tenant_id, status)",
        )
        .unwrap();
    engine
        .execute_text(3, "INSERT INTO version_chain VALUES (1, 7, 2)")
        .unwrap();
    engine
        .publish_relational_resident_indexes("version_chain")
        .unwrap();

    let table = engine.relational_catalog_table("version_chain").unwrap();
    let secondary_ordinal = table
        .indexes
        .iter()
        .position(|index| index.name == "version_chain_status")
        .unwrap();
    let secondary_key_id = crate::engine_residency::index_probe_key_id(
        &table,
        &table.indexes[secondary_ordinal],
        secondary_ordinal,
    )
    .unwrap();
    let shard_id = engine.read_residency_shards()["version_chain"]
        .iter()
        .find(|shard| shard.row_count != 0)
        .expect("non-empty version-chain shard")
        .shard_id;
    let (before, _, _, _) =
        resident_named_index_cache_entry(&engine, "version_chain", shard_id, secondary_key_id);

    for offset in 0..300_u64 {
        engine
            .execute_text(
                10 + offset,
                "UPDATE version_chain SET status = 2 WHERE id = 1",
            )
            .unwrap();
    }

    let (after, table_mask, hash_shift, row_count) =
        resident_named_index_cache_entry(&engine, "version_chain", shard_id, secondary_key_id);
    assert!(Arc::ptr_eq(&before, &after));
    assert_eq!(row_count, 301);
    let key = crate::engine_residency::compound_key_fingerprint(&[7, 2]);
    assert_eq!(
        resident_named_index_physical_hit_count(&after, table_mask, hash_shift, row_count, key,),
        301
    );
    let result = engine
        .execute_relational_select_text("SELECT id, status FROM version_chain WHERE id = 1")
        .unwrap();
    assert_eq!(result.rows.row(0), &[SqlValue::Int4(1), SqlValue::Int4(2)]);
}

/// A rollover rejected by the shared residency budget must allocate or publish nothing: the old
/// shard, named-index allocation, and coverage proof remain exact while the commit path fails stop.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_named_index_rollover_budget_denial_is_atomic() {
    let mut engine = Engine::new_local_test_engine();
    engine.set_shard_size_target(4);
    engine.set_shard_residency_enabled(true);
    // Configure the bound before the empty authoritative generation receives its first rows. This
    // suppresses the throughput-oriented 262k-row floor; the two batches below create then fill a
    // 16-row open shard. The bound is tightened to exact live bytes after publication.
    engine.set_relational_residency_budget_bytes(0, 1 << 30);
    engine
        .execute_text(
            1,
            "CREATE TABLE budget_idx (id INT PRIMARY KEY, tenant_id INT, status INT)",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE INDEX budget_idx_status ON budget_idx (tenant_id, status)",
        )
        .unwrap();
    engine
        .execute_text(
            3,
            "INSERT INTO budget_idx VALUES (1, 7, 2), (2, 7, 2), (3, 7, 2), (4, 7, 2), (5, 7, 2), (6, 7, 2), (7, 7, 2), (8, 7, 2)",
        )
        .unwrap();
    engine
        .execute_text(
            4,
            "INSERT INTO budget_idx VALUES (9, 7, 2), (10, 7, 2), (11, 7, 2), (12, 7, 2), (13, 7, 2), (14, 7, 2), (15, 7, 2), (16, 7, 2)",
        )
        .unwrap();
    let admitted = engine
        .populate_relational_residency_snapshot("budget_idx")
        .unwrap();
    if admitted.device_memory_proof.is_none() {
        return;
    }
    engine
        .publish_relational_resident_indexes("budget_idx")
        .unwrap();
    let table = engine.relational_catalog_table("budget_idx").unwrap();
    let secondary_ordinal = table
        .indexes
        .iter()
        .position(|index| index.name == "budget_idx_status")
        .unwrap();
    let secondary_key_id = crate::engine_residency::index_probe_key_id(
        &table,
        &table.indexes[secondary_ordinal],
        secondary_ordinal,
    )
    .unwrap();
    let shards = engine.read_residency_shards();
    let shard = shards["budget_idx"]
        .iter()
        .rfind(|shard| shard.row_count != 0)
        .expect("non-empty budget shard");
    assert_eq!(
        shard.row_count, shard.capacity,
        "the denial fixture must force the next write through rollover"
    );
    let shard_count_before = shards["budget_idx"].len();
    let shard_id = shard.shard_id;
    let shard_ptr = shard.device_memory.as_ref().unwrap().device_ptr();
    let (index_before, _, _, indexed_rows_before) =
        resident_named_index_cache_entry(&engine, "budget_idx", shard_id, secondary_key_id);
    let live_before = engine.relational_resident_bytes_for_gpu(0);
    let declines_before = engine.rollover_budget_declines();
    engine.set_relational_residency_budget_bytes(0, live_before);
    engine.set_auto_admit_on_commit(true);

    let error = engine
        .execute_text(5, "INSERT INTO budget_idx VALUES (17, 7, 2)")
        .expect_err("mandatory rollover must fail stop at a full residency budget");
    assert!(
        error
            .to_string()
            .contains("could not publish its device generation"),
        "unexpected fail-stop error: {error}"
    );
    assert!(engine.rollover_budget_declines() > declines_before);
    assert_eq!(engine.relational_resident_bytes_for_gpu(0), live_before);
    let shards = engine.read_residency_shards();
    assert_eq!(shards["budget_idx"].len(), shard_count_before);
    let same_shard = shards["budget_idx"]
        .iter()
        .find(|shard| shard.shard_id == shard_id)
        .expect("pre-denial shard remains published");
    assert_eq!(
        same_shard.device_memory.as_ref().unwrap().device_ptr(),
        shard_ptr
    );
    let (index_after, _, _, indexed_rows_after) =
        resident_named_index_cache_entry(&engine, "budget_idx", shard_id, secondary_key_id);
    assert!(Arc::ptr_eq(&index_before, &index_after));
    assert_eq!(indexed_rows_after, indexed_rows_before);
    assert!(engine.is_commit_path_poisoned());
}

/// Publication and mutation both acquire the lane device-apply boundary before the residency
/// budget lock. Starting them together repeatedly is a regression gate against the former inverted
/// order; both workers must finish within a bounded interval even when publications race generation
/// changes.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn resident_named_index_publication_and_mutation_do_not_deadlock() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);
    engine.set_shard_size_target(128);
    engine
        .execute_text(
            1,
            "CREATE TABLE lock_order_idx (id INT PRIMARY KEY, tenant_id INT, status INT)",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE INDEX lock_order_idx_status ON lock_order_idx (tenant_id, status)",
        )
        .unwrap();
    engine
        .execute_text(3, "INSERT INTO lock_order_idx VALUES (1, 7, 2)")
        .unwrap();
    engine
        .publish_relational_resident_indexes("lock_order_idx")
        .unwrap();
    engine.attach_test_intent_lanes(
        crate::tests::test_wal_path("named-index-lock-order").into_path_buf(),
        4,
    );

    let engine = Arc::new(engine);
    let start = Arc::new(std::sync::Barrier::new(3));
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let publisher = {
        let engine = Arc::clone(&engine);
        let start = Arc::clone(&start);
        let done_tx = done_tx.clone();
        std::thread::spawn(move || {
            start.wait();
            for _ in 0..24 {
                let _ = engine.publish_relational_resident_indexes("lock_order_idx");
            }
            done_tx.send("publisher").unwrap();
        })
    };
    let writer = {
        let engine = Arc::clone(&engine);
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            start.wait();
            for id in 2..26_u64 {
                engine
                    .execute_text(
                        100 + id,
                        &format!("INSERT INTO lock_order_idx VALUES ({id}, 7, 2)"),
                    )
                    .unwrap();
            }
            done_tx.send("writer").unwrap();
        })
    };
    start.wait();
    let mut completed = std::collections::BTreeSet::new();
    for _ in 0..2 {
        completed.insert(
            done_rx
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("publication/mutation lock order must make progress"),
        );
    }
    assert_eq!(
        completed,
        std::collections::BTreeSet::from(["publisher", "writer"])
    );
    publisher.join().unwrap();
    writer.join().unwrap();
    let report = engine
        .publish_relational_resident_indexes("lock_order_idx")
        .unwrap();
    assert_eq!(report.indexed_rows, 25);
    assert!(report.indexes.iter().all(|index| index.indexed_rows == 25));
}
