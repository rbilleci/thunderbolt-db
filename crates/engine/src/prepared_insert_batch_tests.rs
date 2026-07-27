use super::*;
use crate::engine_residency::PreparedI32AppendRowIds;

fn accounts_insert(rows: usize) -> Insert {
    Insert {
        table: "accounts".to_string(),
        columns: Vec::new(),
        rows: (0..rows)
            .map(|row| {
                vec![
                    SqlValue::Int4(row as i32),
                    SqlValue::Int4((row as i32) * 10),
                ]
            })
            .collect(),
        returning: Vec::new(),
    }
}

fn prepared_accounts(rows: usize) -> (Engine, Insert, WriteDelta, CatalogSnapshot) {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    let insert = accounts_insert(rows);
    let catalog = (*engine.catalog_snapshot()).clone();
    let delta = engine
        .prepare_insert(
            &insert,
            engine.dml_read_snapshot(engine.committed_seq()),
            None,
            InsertPrepareValidation::WaveOffLock,
        )
        .unwrap();
    (engine, insert, delta, catalog)
}

fn sealed_i32_source(engine: &Engine, rows: Vec<Vec<SqlValue>>) -> PreparedI32AppendSource {
    let insert = Insert {
        table: "accounts".to_string(),
        columns: Vec::new(),
        rows,
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    let delta = engine
        .prepare_insert(
            &insert,
            engine.dml_read_snapshot(engine.committed_seq()),
            None,
            InsertPrepareValidation::WaveOffLock,
        )
        .expect("sealed source uses authoritative off-lock prepare");
    try_prepare_fixed_insert_batch(
        &Command::Insert(insert),
        &delta,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .expect("accounts int4 source is eligible")
    .into_i32_append_source()
}

fn exact_row_ids(ids: impl IntoIterator<Item = u64>) -> PreparedI32AppendRowIds {
    PreparedI32AppendRowIds::exact(ids.into_iter().collect::<Vec<_>>().into_boxed_slice())
}

fn synthetic_no_identity() -> PreparedI32AppendRowIds {
    PreparedI32AppendRowIds::synthetic_no_identity()
}

fn read_device_u64(memory: &CudaResidentDeviceMemory, slot: usize) -> u64 {
    let words = memory
        .read_resident_i32_column((slot * std::mem::size_of::<u64>()) as u64, 2)
        .expect("read device u64 words");
    (words[0] as u32 as u64) | ((words[1] as u32 as u64) << 32)
}

#[test]
fn accounts_i32_batch_binds_exact_columns_values_dependencies_and_statement_memory() {
    let (_engine, insert, delta, catalog) = prepared_accounts(1_000);
    let batch = try_prepare_fixed_insert_batch(
        &Command::Insert(insert),
        &delta,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .expect("exact accounts int4 route is eligible");
    let table = catalog.relational_catalog.get("accounts").unwrap();
    assert_eq!(batch.table.name.as_ref(), "accounts");
    assert_eq!(batch.table.oid, table.oid);
    assert_eq!(
        batch.table.schema_digest,
        crate::engine_transaction_reset::table_schema_digest(table).unwrap()
    );
    assert_eq!(batch.table.prepared_catalog_seq, catalog.commit_seq);
    assert_eq!(batch.row_count, 1_000);
    assert_eq!(batch.columns.len(), 2);
    assert_eq!(batch.columns[0].column_id, table.columns[0].id);
    assert_eq!(batch.columns[1].column_id, table.columns[1].id);
    assert_eq!(batch.columns[0].attnum, table.columns[0].attnum);
    assert_eq!(batch.columns[1].attnum, table.columns[1].attnum);
    assert_eq!(batch.columns[0].ty, SqlType::Int4);
    assert_eq!(batch.columns[1].ty, SqlType::Int4);
    assert_eq!(batch.columns[0].type_oid, table.columns[0].type_oid);
    assert_eq!(batch.columns[1].type_oid, table.columns[1].type_oid);
    assert_eq!(batch.columns[0].type_size, table.columns[0].type_size);
    assert_eq!(batch.columns[1].type_size, table.columns[1].type_size);
    assert_eq!(batch.columns[0].values[0], 0);
    assert_eq!(batch.columns[1].values[0], 0);
    assert_eq!(batch.columns[0].values[999], 999);
    assert_eq!(batch.columns[1].values[999], 9_990);
    assert_eq!(batch.dependencies.len(), 1);
    assert_eq!(batch.dependencies[0].name.as_ref(), "accounts");
    assert_eq!(batch.dependencies[0].oid, table.oid);
    assert_eq!(
        batch.dependencies[0].schema_digest,
        batch.table.schema_digest
    );
    assert_eq!(batch.value_bytes(), 1_000 * 2 * std::mem::size_of::<i32>());

    let (_engine, small_insert, small_delta, small_catalog) = prepared_accounts(3);
    let small = try_prepare_fixed_insert_batch(
        &Command::Insert(small_insert),
        &small_delta,
        &small_catalog,
        small_catalog.commit_seq,
        None,
    )
    .unwrap();
    assert_eq!(small.value_bytes(), 3 * 2 * std::mem::size_of::<i32>());
    assert_eq!(
        batch.value_bytes() / (2 * std::mem::size_of::<i32>()),
        1_000
    );
    assert_eq!(small.value_bytes() / (2 * std::mem::size_of::<i32>()), 3);
    // Each statement owns only its own boxed fixed-width vectors; there is no global store.
    assert_eq!(batch.value_bytes(), 1_000 * 2 * std::mem::size_of::<i32>());
}

#[test]
fn consuming_i32_source_preserves_each_boxed_column_allocation() {
    let (_engine, insert, delta, catalog) = prepared_accounts(3);
    let batch = try_prepare_fixed_insert_batch(
        &Command::Insert(insert),
        &delta,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .expect("accounts shape is eligible");
    let original: Vec<(*const i32, usize)> = batch
        .columns
        .iter()
        .map(|column| (column.values.as_ptr(), column.values.len()))
        .collect();
    let source = batch.into_i32_append_source();
    assert_eq!(source.row_count(), 3);
    assert_eq!(source.columns().len(), original.len());
    for (column, (ptr, len)) in source.columns().iter().zip(original) {
        assert_eq!(column.values().as_ptr(), ptr);
        assert_eq!(column.values().len(), len);
    }
}

#[test]
fn typed_adapter_rejects_identity_schema_and_column_cardinality_drift_before_publish() {
    let (engine, insert, delta, catalog) = prepared_accounts(2);
    let device_cells_before = engine
        .read_state
        .residency
        .shard_device_memory
        .cells
        .load()
        .len();
    let source = || {
        try_prepare_fixed_insert_batch(
            &Command::Insert(insert.clone()),
            &delta,
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .into_i32_append_source()
    };
    let mut oid_drift = source();
    oid_drift.table.oid = oid_drift.table.oid.saturating_add(1);
    assert!(engine
        .prepare_prepared_i32_open_shard_append(oid_drift, synthetic_no_identity())
        .is_err());

    let mut type_drift = source();
    type_drift.columns[1].type_size = 8;
    assert!(engine
        .prepare_prepared_i32_open_shard_append(type_drift, synthetic_no_identity())
        .is_err());

    let mut cardinality_drift = source();
    cardinality_drift.columns[0].values.pop();
    assert!(engine
        .prepare_prepared_i32_open_shard_append(cardinality_drift, synthetic_no_identity())
        .is_err());

    let mut catalog_drift = source();
    catalog_drift.table.prepared_catalog_seq += 1;
    assert!(engine
        .prepare_prepared_i32_open_shard_append(catalog_drift, synthetic_no_identity())
        .is_err());
    assert_eq!(
        engine
            .read_state
            .residency
            .shard_device_memory
            .cells
            .load()
            .len(),
        device_cells_before,
        "CPU validation declines before any device allocation or descriptor publication"
    );
}

#[test]
fn typed_column_encoder_matches_row_major_bytes_with_offset_and_extrema() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    let insert = Insert {
        table: "accounts".to_string(),
        columns: Vec::new(),
        rows: vec![
            vec![SqlValue::Int4(i32::MIN), SqlValue::Int4(-7)],
            vec![SqlValue::Int4(0), SqlValue::Int4(13)],
            vec![SqlValue::Int4(i32::MAX), SqlValue::Int4(-1)],
        ],
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    let delta = engine
        .prepare_insert(
            &insert,
            engine.dml_read_snapshot(engine.committed_seq()),
            None,
            InsertPrepareValidation::WaveOffLock,
        )
        .unwrap();
    let batch = try_prepare_fixed_insert_batch(
        &Command::Insert(insert.clone()),
        &delta,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap();
    let source = batch.into_i32_append_source();
    let columns: Vec<&[i32]> = source
        .columns()
        .iter()
        .map(|column| column.values())
        .collect();
    let row_major = crate::engine_residency::compute_open_shard_int4_append_chunks(
        &[SqlType::Int4, SqlType::Int4],
        11,
        4,
        &insert.rows,
    )
    .unwrap();
    let typed =
        crate::engine_residency::compute_open_shard_i32_column_append_chunks(11, 4, &columns)
            .unwrap();
    assert_eq!(typed, row_major);
    assert_eq!(typed.last().unwrap().byte_offset, 0, "header is final");
    assert_eq!(typed.last().unwrap().bytes, (7_u64).to_le_bytes());
    assert!(
        crate::engine_residency::compute_open_shard_i32_column_append_chunks(6, 4, &columns)
            .is_err()
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn typed_source_enters_the_shared_fused_and_unfused_open_shard_publisher() {
    for lanes_attached in [false, true] {
        for fused in [false, true] {
            let mut engine = Engine::new_local();
            if lanes_attached {
                engine.attach_test_intent_lanes(
                    std::env::temp_dir().join("insert001-residency-gate"),
                    2,
                );
            }
            engine.set_shard_residency_enabled(true);
            engine.set_shard_size_target(8);
            engine.set_fused_apply_enabled(fused);
            engine
                .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
                .unwrap();
            engine
                .execute_text(2, "INSERT INTO accounts VALUES (1, 10), (2, 20)")
                .unwrap();
            engine
                .populate_relational_residency_snapshot("accounts")
                .unwrap();
            let before = engine
                .read_state
                .residency
                .shards
                .load()
                .get("accounts")
                .and_then(|shards| shards.last())
                .cloned()
                .expect("qualified GPU host must publish the initial resident shard");
            assert!(
                before.device_memory.is_some(),
                "initial shard is device-resident"
            );
            assert!(
                before.device_memory_proof.is_some(),
                "initial shard carries a device allocation proof"
            );
            let visible_stamp = engine.committed_seq();
            let fused_hits_before = engine.fused_apply_hits();
            let plan = engine
                .prepare_prepared_i32_open_shard_append(
                    sealed_i32_source(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(-30)]]),
                    exact_row_ids([3]),
                )
                .expect("pre-WAL plan seals the current open descriptor");
            let prepared = engine
                .read_state
                .residency
                .shards
                .load()
                .get("accounts")
                .and_then(|shards| shards.last())
                .cloned()
                .expect("planning must not remove the open descriptor");
            assert_eq!(
                prepared.row_count, before.row_count,
                "planning is pre-publication"
            );
            assert_eq!(prepared.shard_id, before.shard_id);
            assert!(
                engine
                    .read_state
                    .residency
                    .mutation_gate
                    .try_lock()
                    .is_err(),
                "sealed plan holds the always-present residency gate with or without lanes"
            );
            engine
                .apply_prepared_i32_open_shard_append(
                    plan,
                    crate::engine_residency::AppendCreatedBy::InsertUniform(visible_stamp),
                )
                .expect("sealed plan reaches mutation's single publisher");
            assert!(
                engine.read_state.residency.mutation_gate.try_lock().is_ok(),
                "consumed plan releases the residency gate before later tail work"
            );
            let after = engine
                .read_state
                .residency
                .shards
                .load()
                .get("accounts")
                .and_then(|shards| shards.last())
                .cloned()
                .unwrap();
            assert_eq!(after.shard_id, before.shard_id);
            assert_eq!(after.row_count, before.row_count + 1);
            assert_eq!(after.max_created_by, visible_stamp);
            assert_eq!(
                engine.fused_apply_hits() - fused_hits_before,
                u64::from(fused),
                "the fused flag selects exactly the fused publisher branch"
            );
            let memory = after.device_memory.as_ref().unwrap();
            let row_id_region = after
                .row_id_region
                .as_ref()
                .expect("identity-bearing typed append retains its row-id sidecar");
            assert_eq!(
                read_device_u64(row_id_region, before.row_count),
                3,
                "sealed exact row ID is stamped into the appended in-place slot"
            );
            assert_eq!(
                memory
                    .read_resident_i32_column((8 + before.row_count * 4) as u64, 1)
                    .unwrap(),
                [3]
            );
            assert_eq!(
                memory
                    .read_resident_i32_column(
                        (8 + after.capacity * 4 + before.row_count * 4) as u64,
                        1,
                    )
                    .unwrap(),
                [-30]
            );
            assert_eq!(
                engine
                    .execute_relational_select_text("SELECT id, balance FROM accounts ORDER BY id",)
                    .unwrap()
                    .rows,
                vec![
                    vec![SqlValue::Int4(1), SqlValue::Int4(10)],
                    vec![SqlValue::Int4(2), SqlValue::Int4(20)],
                    vec![SqlValue::Int4(3), SqlValue::Int4(-30)],
                ],
            );
        }
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn typed_source_rollover_keeps_pending_build_and_descriptor_publication_in_mutation() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(4);
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO accounts VALUES (1, 10), (2, 20)")
        .unwrap();
    engine
        .populate_relational_residency_snapshot("accounts")
        .unwrap();
    let before_count = engine.resident_shard_count("accounts");
    assert!(
        before_count > 0,
        "qualified GPU host must publish an initial shard"
    );
    let before = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("initial rollover tail exists");
    assert!(
        before.device_memory.is_some(),
        "initial tail is device-resident"
    );
    let unpublished_stamp = engine.committed_seq() + 1;
    let row_ids = [41_u64, 42, 43];
    let plan = engine
        .prepare_prepared_i32_open_shard_append(
            sealed_i32_source(
                &engine,
                vec![
                    vec![SqlValue::Int4(i32::MIN), SqlValue::Int4(i32::MAX)],
                    vec![SqlValue::Int4(0), SqlValue::Int4(-1)],
                    vec![SqlValue::Int4(i32::MAX), SqlValue::Int4(i32::MIN)],
                ],
            ),
            exact_row_ids(row_ids),
        )
        .expect("pre-WAL plan seals the rollover capacity and budget");
    assert_eq!(
        engine.resident_shard_count("accounts"),
        before_count,
        "planning neither allocates nor publishes a shard"
    );
    assert!(
        engine
            .read_state
            .residency
            .budget_allocation_lock
            .try_lock()
            .is_err(),
        "the sealed rollover plan blocks a budget thief through WAL/apply"
    );
    engine
        .apply_prepared_i32_open_shard_append(
            plan,
            crate::engine_residency::AppendCreatedBy::InsertUniform(unpublished_stamp),
        )
        .expect("sealed rollover applies through mutation");
    assert!(
        engine
            .read_state
            .residency
            .budget_allocation_lock
            .try_lock()
            .is_ok(),
        "consuming the plan releases the budget reservation before later tail work"
    );
    let rolled = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .cloned()
        .unwrap();
    assert_eq!(engine.resident_shard_count("accounts"), before_count + 1);
    assert_eq!(rolled.shard_id, before.shard_id + 1);
    assert!(
        rolled.device_memory.is_some(),
        "rollover publishes device memory"
    );
    assert!(
        rolled.device_memory_proof.is_some(),
        "rollover descriptor records the device allocation proof"
    );
    assert_eq!(rolled.row_count, 3);
    assert_eq!(rolled.max_created_by, unpublished_stamp);
    assert!(
        rolled.capacity > rolled.row_count,
        "rollover preserves open headroom"
    );
    assert!(rolled.resident_device_null_columns.is_empty());
    let memory = rolled.device_memory.as_ref().unwrap();
    assert_eq!(read_device_u64(memory, 0), 3, "header is written last");
    assert_eq!(
        memory.read_resident_i32_column(8, 3).unwrap(),
        [i32::MIN, 0, i32::MAX]
    );
    assert_eq!(
        memory
            .read_resident_i32_column((8 + rolled.capacity * 4) as u64, 3)
            .unwrap(),
        [i32::MAX, -1, i32::MIN]
    );
    let created_by = rolled.created_by_region.as_ref().unwrap();
    assert_eq!(read_device_u64(created_by, 0), unpublished_stamp);
    assert_eq!(read_device_u64(created_by, rolled.row_count), 0);
    let row_id_region = rolled.row_id_region.as_ref().unwrap();
    assert_eq!(
        (0..row_ids.len())
            .map(|slot| read_device_u64(row_id_region, slot))
            .collect::<Vec<_>>(),
        row_ids
    );
    assert_eq!(read_device_u64(row_id_region, rolled.row_count), u64::MAX);
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id, balance FROM accounts ORDER BY id")
            .unwrap()
            .rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(10)],
            vec![SqlValue::Int4(2), SqlValue::Int4(20)],
        ],
        "created_by hides rows from the pre-publication snapshot"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn create_bootstrap_sentinel_requires_exact_clean_shape_and_pre_wal_budget() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(8);
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    let bootstrap = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .filter(|shards| shards.len() == 1)
        .map(|shards| shards[0].clone())
        .expect("CREATE auto-admission must publish the sole zero-capacity bootstrap shard");
    assert_eq!(bootstrap.shard_id, 0);
    assert_eq!(bootstrap.row_start, 0);
    assert_eq!(bootstrap.row_count, 0);
    assert_eq!(bootstrap.capacity, 0);
    assert!(bootstrap.created_by_region.is_none());
    assert!(bootstrap.row_id_region.is_none());
    assert!(bootstrap.deleted_by_region.is_none());
    assert!(engine
        .read_state
        .residency
        .shard_deleted_by_memory
        .get(&("accounts".to_string(), 0))
        .is_none());

    assert!(matches!(
        engine.prepare_prepared_i32_open_shard_append(
            sealed_i32_source(&engine, vec![vec![SqlValue::Int4(1), SqlValue::Int4(10)]]),
            synthetic_no_identity(),
        ),
        Err(crate::engine_residency::PreparedI32AppendPrepareError::UnsupportedShape)
    ));

    engine
        .read_state
        .residency
        .with_shards_mut_for_table("accounts", |shards| {
            shards
                .get_mut("accounts")
                .expect("bootstrap descriptor exists for capacity sabotage")[0]
                .capacity = 1;
        });
    assert!(
        matches!(
            engine.prepare_prepared_i32_open_shard_append(
                sealed_i32_source(&engine, vec![vec![SqlValue::Int4(1), SqlValue::Int4(10)]]),
                exact_row_ids([1]),
            ),
            Err(crate::engine_residency::PreparedI32AppendPrepareError::UnsupportedShape)
        ),
        "a sole empty descriptor with positive capacity is never the CREATE sentinel"
    );
    engine
        .read_state
        .residency
        .with_shards_mut_for_table("accounts", |shards| {
            shards
                .get_mut("accounts")
                .expect("capacity sabotage remains the sole bootstrap descriptor")[0]
                .capacity = 0;
        });

    let deleted_sabotage = bootstrap
        .device_memory
        .as_ref()
        .expect("bootstrap descriptor owns its device payload")
        .clone();
    engine
        .read_state
        .residency
        .with_shards_mut_for_table("accounts", |shards| {
            shards
                .get_mut("accounts")
                .expect("bootstrap descriptor exists for deleted-sidecar sabotage")[0]
                .deleted_by_region = Some(std::sync::Arc::clone(&deleted_sabotage));
        });
    engine
        .read_state
        .residency
        .shard_deleted_by_memory
        .insert_shard("accounts", 0, deleted_sabotage);
    assert!(
        matches!(
            engine.prepare_prepared_i32_open_shard_append(
                sealed_i32_source(&engine, vec![vec![SqlValue::Int4(1), SqlValue::Int4(10)]]),
                exact_row_ids([1]),
            ),
            Err(crate::engine_residency::PreparedI32AppendPrepareError::UnsupportedShape)
        ),
        "a deleted-sidecar-bearing empty descriptor is not the CREATE sentinel"
    );
    engine
        .read_state
        .residency
        .with_shards_mut_for_table("accounts", |shards| {
            shards
                .get_mut("accounts")
                .expect("deleted-sidecar sabotage remains the sole bootstrap descriptor")[0]
                .deleted_by_region = None;
        });
    engine
        .read_state
        .residency
        .shard_deleted_by_memory
        .invalidate_shard("accounts", 0);

    engine.set_relational_residency_budget_bytes(0, 0);
    let wal_before = engine.durable_wal_records().len();
    let row_id_before = engine.read_state.mvcc.current_row_id();
    let append_before = engine.open_shard_append_hits();
    let authority_before = engine.device_authoritative_commits();
    assert!(matches!(
        engine.prepare_prepared_i32_open_shard_append(
            sealed_i32_source(
                &engine,
                vec![
                    vec![SqlValue::Int4(1), SqlValue::Int4(10)],
                    vec![SqlValue::Int4(2), SqlValue::Int4(20)],
                ],
            ),
            exact_row_ids([row_id_before, row_id_before + 1]),
        ),
        Err(
            crate::engine_residency::PreparedI32AppendPrepareError::RetryableBoundBootstrapResource
        )
    ));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
    assert_eq!(engine.open_shard_append_hits(), append_before);
    assert_eq!(engine.device_authoritative_commits(), authority_before);
    assert_eq!(
        engine
            .read_state
            .residency
            .shards
            .load()
            .get("accounts")
            .cloned(),
        Some(vec![bootstrap.clone()]),
        "pre-WAL bootstrap budget rejection must not allocate or republish a descriptor"
    );
    engine.set_relational_residency_budget_bytes(0, u64::MAX);
    engine
        .read_state
        .residency
        .with_shards_mut_for_table("accounts", |shards| {
            let table = shards
                .get_mut("accounts")
                .expect("bootstrap table remains published for sole-shard sabotage");
            table.push(table[0].clone());
        });
    assert!(matches!(
        engine.prepare_prepared_i32_open_shard_append(
            sealed_i32_source(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
            exact_row_ids([row_id_before + 2]),
        ),
        Err(crate::engine_residency::PreparedI32AppendPrepareError::UnsupportedShape)
    ));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sealed_plan_budget_declines_before_allocation_or_descriptor_publication() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(8);
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO accounts VALUES (1, 10), (2, 20)")
        .unwrap();
    engine
        .populate_relational_residency_snapshot("accounts")
        .unwrap();
    let before = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("qualified GPU host must publish the initial resident shard");
    assert!(before.device_memory_proof.is_some());
    engine.set_relational_residency_budget_bytes(0, 0);
    let declines_before = engine.rollover_budget_declines();
    let source = sealed_i32_source(
        &engine,
        (3..10)
            .map(|id| vec![SqlValue::Int4(id), SqlValue::Int4(-id)])
            .collect(),
    );
    assert!(
        engine
            .prepare_prepared_i32_open_shard_append(source, exact_row_ids(3..10))
            .is_err(),
        "dense rollover must decline before the first CUDA allocation"
    );
    let after = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("budget decline must retain the original descriptor");
    assert_eq!(after.shard_id, before.shard_id);
    assert_eq!(after.row_count, before.row_count);
    assert_eq!(after.device_memory_proof, before.device_memory_proof);
    assert!(engine.rollover_budget_declines() > declines_before);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn stale_sealed_plan_returns_fatal_drift_without_a_second_publication() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(8);
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO accounts VALUES (1, 10), (2, 20)")
        .unwrap();
    engine
        .populate_relational_residency_snapshot("accounts")
        .unwrap();
    let before = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("qualified GPU host must publish the initial resident shard");
    let plan = engine
        .prepare_prepared_i32_open_shard_append(
            sealed_i32_source(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
            exact_row_ids([3]),
        )
        .expect("plan binds the initial descriptor");
    // Test-only sabotage of the immutable descriptor publication. Normal publishers cannot
    // interleave here: the plan retains the residency mutation gate through apply.
    engine
        .read_state
        .residency
        .with_shards_mut_for_table("accounts", |shards| {
            let open = shards
                .get_mut("accounts")
                .and_then(|shards| shards.last_mut())
                .expect("test descriptor exists");
            open.point_route_generation = std::sync::Arc::new(());
        });
    assert_eq!(
        engine.apply_prepared_i32_open_shard_append(
            plan,
            crate::engine_residency::AppendCreatedBy::InsertUniform(engine.committed_seq(),),
        ),
        Err(crate::engine_residency::PreparedI32AppendApplyError::PlanDrift)
    );
    let after = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("sabotage descriptor remains the sole publication");
    assert_eq!(after.row_count, before.row_count);
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id, balance FROM accounts ORDER BY id")
            .unwrap()
            .rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(10)],
            vec![SqlValue::Int4(2), SqlValue::Int4(20)],
        ],
        "fatal drift must not fall back to a second append publisher"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn sealed_plan_refuses_positive_capacity_nonempty_missing_identity_sidecar_before_wal() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(8);
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO accounts VALUES (1, 10), (2, 20)")
        .unwrap();
    engine
        .populate_relational_residency_snapshot("accounts")
        .unwrap();
    let before = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("qualified GPU host must publish the initial resident shard");
    assert!(before.row_count > 0);
    assert!(before.capacity > 0);
    engine
        .read_state
        .residency
        .with_shards_mut_for_table("accounts", |shards| {
            let open = shards
                .get_mut("accounts")
                .and_then(|shards| shards.last_mut())
                .expect("test descriptor exists");
            open.created_by_region = None;
            open.row_id_region = None;
        });
    engine
        .read_state
        .residency
        .shard_created_by_memory
        .invalidate_shard("accounts", before.shard_id);
    engine
        .read_state
        .residency
        .shard_row_id_memory
        .invalidate_shard("accounts", before.shard_id);
    let sidecar_bytes = (before.capacity * std::mem::size_of::<u64>()) as u64;
    engine.set_relational_residency_budget_bytes(
        0,
        engine
            .relational_resident_bytes_for_gpu(0)
            .saturating_add(sidecar_bytes.saturating_sub(1)),
    );
    assert!(
        engine
            .prepare_prepared_i32_open_shard_append(
                sealed_i32_source(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
                synthetic_no_identity(),
            )
            .is_err(),
        "a first created_by sidecar must be budgeted before WAL"
    );
    engine.set_relational_residency_budget_bytes(0, u64::MAX);
    assert!(
        engine
            .prepare_prepared_i32_open_shard_append(
                sealed_i32_source(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
                exact_row_ids([3]),
            )
            .is_err(),
        "row IDs require the exact bound row-id sidecar; no get-or-skip path is allowed"
    );
    let after = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("pre-WAL refusal cannot remove the sabotaged descriptor");
    assert_eq!(after.row_count, before.row_count);
    assert!(after.created_by_region.is_none());
    assert!(after.row_id_region.is_none());
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn created_by_gc_runs_only_after_a_sealed_plan_releases_the_residency_gate() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(8);
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO accounts VALUES (1, 10), (2, 20)")
        .unwrap();
    engine
        .populate_relational_residency_snapshot("accounts")
        .unwrap();
    let stamp = engine.committed_seq();
    let first = engine
        .prepare_prepared_i32_open_shard_append(
            sealed_i32_source(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
            exact_row_ids([3]),
        )
        .expect("initial plan is eligible");
    engine
        .apply_prepared_i32_open_shard_append(
            first,
            crate::engine_residency::AppendCreatedBy::InsertUniform(stamp),
        )
        .expect("first append installs the created_by sidecar");
    assert!(engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .is_some_and(|shard| shard.created_by_region.is_some()));
    let plan = engine
        .prepare_prepared_i32_open_shard_append(
            sealed_i32_source(&engine, vec![vec![SqlValue::Int4(4), SqlValue::Int4(40)]]),
            exact_row_ids([4]),
        )
        .expect("second plan binds the created_by sidecar");
    assert!(
        engine
            .read_state
            .residency
            .mutation_gate
            .try_lock()
            .is_err(),
        "plan binds the same gate GC must acquire before sidecar retirement"
    );
    assert!(engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .is_some_and(|shard| shard.created_by_region.is_some()));
    drop(plan);
    assert!(
        engine.gc_transaction_created_by_regions() >= 1,
        "GC retires the sidecar only after the sealed plan releases the shared gate"
    );
}

#[test]
fn parsed_1000_row_accounts_workload_prepares_to_the_fixed_width_batch() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    let mut workload = String::from("INSERT INTO accounts VALUES ");
    for row in 0..1_000_i32 {
        if row != 0 {
            workload.push(',');
        }
        workload.push_str(&format!("({row}, {})", row * 10));
    }
    let command = parse_command(&workload).expect("the canonical accounts workload parses");
    let catalog = engine.catalog_snapshot();
    let delta = engine
        .prepare_dml(
            &command,
            engine.dml_read_snapshot(engine.committed_seq()),
            InsertPrepareValidation::WaveOffLock,
        )
        .expect("authoritative off-lock prepare accepts the canonical workload");
    let batch =
        try_prepare_fixed_insert_batch(&command, &delta, &catalog, catalog.commit_seq, None)
            .expect("the exact parsed workload is eligible");
    assert_eq!(batch.row_count, 1_000);
    assert_eq!(batch.columns.len(), 2);
    assert_eq!(batch.columns[0].values[0], 0);
    assert_eq!(batch.columns[1].values[0], 0);
    assert_eq!(batch.columns[0].values[999], 999);
    assert_eq!(batch.columns[1].values[999], 9_990);
}

#[test]
fn direct_builder_needs_no_delta_and_fails_closed_on_shape_or_catalog_mismatch() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    let insert = accounts_insert(2);
    let catalog = engine.catalog_snapshot();
    let command = Command::Insert(insert.clone());
    let batch = try_prepare_direct_fixed_insert_batch(
        &command,
        &catalog,
        catalog.commit_seq,
        Some(
            crate::engine_mutation_admission::CatalogVersionExpectation::Prepared(
                catalog.commit_seq,
            ),
        ),
    )
    .unwrap()
    .expect("direct fixed eligibility uses only parsed values and the pinned catalog");
    assert_eq!(batch.row_count, 2);
    assert_eq!(batch.columns[0].values.as_ref(), &[0, 1]);
    assert_eq!(batch.columns[1].values.as_ref(), &[0, 10]);

    let mut reordered = insert.clone();
    reordered.columns = vec!["balance".to_string(), "id".to_string()];
    assert!(try_prepare_direct_fixed_insert_batch(
        &Command::Insert(reordered),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap()
    .is_none());
    assert!(try_prepare_direct_fixed_insert_batch(
        &command,
        &catalog,
        catalog.commit_seq,
        Some(
            crate::engine_mutation_admission::CatalogVersionExpectation::Prepared(
                catalog.commit_seq + 1,
            ),
        ),
    )
    .is_err());
}

#[test]
fn batch_rejects_stable_oid_schema_catalog_and_delta_proof_mismatches() {
    let (_engine, insert, delta, catalog) = prepared_accounts(2);
    let command = Command::Insert(insert.clone());
    assert!(
        try_prepare_fixed_insert_batch(&command, &delta, &catalog, catalog.commit_seq, None,)
            .is_some()
    );

    let mut oid_drift = catalog.clone();
    oid_drift
        .relational_catalog
        .get_mut("accounts")
        .unwrap()
        .oid += 1;
    assert!(
        try_prepare_fixed_insert_batch(&command, &delta, &oid_drift, catalog.commit_seq, None,)
            .is_none()
    );

    let mut schema_drift = catalog.clone();
    schema_drift
        .relational_catalog
        .get_mut("accounts")
        .unwrap()
        .columns[1]
        .type_size = 8;
    assert!(try_prepare_fixed_insert_batch(
        &command,
        &delta,
        &schema_drift,
        catalog.commit_seq,
        None,
    )
    .is_none());
    assert!(try_prepare_fixed_insert_batch(
        &command,
        &delta,
        &catalog,
        catalog.commit_seq + 1,
        None,
    )
    .is_none());

    let mut wrong_proof = delta.clone();
    wrong_proof.write_set.tables.clear();
    assert!(try_prepare_fixed_insert_batch(
        &command,
        &wrong_proof,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .is_none());
    let mut wrong_string_unique_projection = delta.clone();
    wrong_string_unique_projection
        .write_set
        .unique_slots
        .push(UniqueIndexSlotKey {
            table: "accounts".to_string(),
            column: "id".to_string(),
            value: "1".to_string(),
        });
    assert!(try_prepare_fixed_insert_batch(
        &command,
        &wrong_string_unique_projection,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .is_none());
    let mut wrong_unique_projection = delta.clone();
    wrong_unique_projection
        .write_set
        .unique_slots_i32
        .push((1, 1));
    assert!(try_prepare_fixed_insert_batch(
        &command,
        &wrong_unique_projection,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .is_none());
}

#[test]
fn unsupported_null_text_sequence_fk_and_returning_shapes_fall_back() {
    let (_engine, insert, delta, catalog) = prepared_accounts(1);
    let mut null = insert.clone();
    null.rows[0][1] = SqlValue::Null;
    assert!(try_prepare_fixed_insert_batch(
        &Command::Insert(null),
        &delta,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .is_none());
    let mut text = insert.clone();
    text.rows[0][1] = SqlValue::Text("not-fixed-width".to_string());
    assert!(try_prepare_fixed_insert_batch(
        &Command::Insert(text),
        &delta,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .is_none());
    let mut returning = insert.clone();
    returning.returning.push("id".to_string());
    assert!(try_prepare_fixed_insert_batch(
        &Command::Insert(returning),
        &delta,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .is_none());
    let mut sequence = delta.clone();
    let PreparedMutation::Insert { seq_advances, .. } = &mut sequence.mutation else {
        unreachable!();
    };
    seq_advances.insert("sequence_proof".to_string(), (1, true));
    assert!(try_prepare_fixed_insert_batch(
        &Command::Insert(insert.clone()),
        &sequence,
        &catalog,
        catalog.commit_seq,
        None,
    )
    .is_none());
    let mut fk = catalog.clone();
    fk.relational_catalog
        .get_mut("accounts")
        .unwrap()
        .foreign_keys
        .push(RelationalForeignKey {
            name: "accounts_fk".to_string(),
            column: "id".to_string(),
            referenced_table: "parents".to_string(),
            referenced_column: "id".to_string(),
        });
    assert!(try_prepare_fixed_insert_batch(
        &Command::Insert(insert.clone()),
        &delta,
        &fk,
        catalog.commit_seq,
        None,
    )
    .is_none());
}

#[test]
fn exact_full_column_list_and_catalog_expectation_build_fixed_batch() {
    let (engine, mut insert, _delta, catalog) = prepared_accounts(2);
    insert.columns = vec!["id".to_string(), "balance".to_string()];
    let delta = engine
        .prepare_insert(
            &insert,
            engine.dml_read_snapshot(engine.committed_seq()),
            None,
            InsertPrepareValidation::WaveOffLock,
        )
        .expect("exact named catalog order is a normal INSERT prepare");
    let exact_expectation = Some(
        crate::engine_mutation_admission::CatalogVersionExpectation::Prepared(catalog.commit_seq),
    );
    assert!(try_prepare_fixed_insert_batch(
        &Command::Insert(insert.clone()),
        &delta,
        &catalog,
        catalog.commit_seq,
        exact_expectation,
    )
    .is_some());
    assert!(try_prepare_fixed_insert_batch(
        &Command::Insert(insert.clone()),
        &delta,
        &catalog,
        catalog.commit_seq,
        Some(
            crate::engine_mutation_admission::CatalogVersionExpectation::Prepared(
                catalog
                    .commit_seq
                    .checked_add(1)
                    .expect("test catalog generation has a successor"),
            ),
        ),
    )
    .is_none());

    for columns in [
        vec!["balance".to_string(), "id".to_string()],
        vec!["id".to_string()],
        vec!["id".to_string(), "id".to_string()],
        vec!["id".to_string(), "unknown".to_string()],
    ] {
        let mut unsupported = insert.clone();
        unsupported.columns = columns;
        assert!(try_prepare_fixed_insert_batch(
            &Command::Insert(unsupported),
            &delta,
            &catalog,
            catalog.commit_seq,
            exact_expectation,
        )
        .is_none());
    }
}

#[test]
fn direct_carrier_preserves_concurrent_execution_canonical_wal_and_recovery() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    engine
        .execute_dml_concurrent(2, "INSERT INTO accounts VALUES (7, 70)")
        .unwrap();
    let records = engine.durable_wal_records();
    let envelope = gpu_db_wal::decode_canonical_record_payload(&records[1].payload)
        .unwrap()
        .expect("concurrent INSERT remains canonical WAL");
    assert_eq!(envelope.outcome.affected_rows, 1);
    let recovered = Engine::recover_from_durable_wal(&records).unwrap();
    assert_eq!(
        recovered
            .execute_relational_select_text("SELECT id, balance FROM accounts")
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(7), SqlValue::Int4(70)]]
    );
}
