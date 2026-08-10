use super::*;
use crate::engine_residency::DeviceInsertRowIds;

fn accounts_insert(rows: usize) -> Insert {
    Insert {
        table: "accounts".to_string(),
        columns: Vec::new(),
        rows: Insert::programmatic_rows(
            (0..rows)
                .map(|row| {
                    vec![
                        SqlValue::Int4(row as i32),
                        SqlValue::Int4((row as i32) * 10),
                    ]
                })
                .collect(),
        ),
        returning: Vec::new(),
    }
}

fn prepared_accounts(rows: usize) -> (Engine, Insert, CatalogSnapshot) {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
        .unwrap();
    let insert = accounts_insert(rows);
    let catalog = (*engine.catalog_snapshot()).clone();
    (engine, insert, catalog)
}

fn sealed_i32_batch(engine: &Engine, rows: Vec<Vec<SqlValue>>) -> TypedInsertBatch {
    let insert = Insert {
        table: "accounts".to_string(),
        columns: Vec::new(),
        rows: Insert::programmatic_rows(rows),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    seal_typed_insert_batch_for_test(&Command::Insert(insert), &catalog, catalog.commit_seq, None)
        .expect("sealed batch direct preparation succeeds")
        .expect("accounts int4 source is eligible")
}

fn sealed_fixed_width_batch(
    engine: &Engine,
    table: &str,
    rows: Vec<Vec<SqlValue>>,
) -> TypedInsertBatch {
    let insert = Insert {
        table: table.to_string(),
        columns: Vec::new(),
        rows: Insert::programmatic_rows(rows),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    seal_typed_insert_batch_for_test(&Command::Insert(insert), &catalog, catalog.commit_seq, None)
        .expect("sealed mixed fixed-width batch prepares")
        .expect("mixed NULL-free fixed-width source is eligible")
}

fn sealed_mixed_fixed_width_batch(engine: &Engine, rows: Vec<Vec<SqlValue>>) -> TypedInsertBatch {
    sealed_fixed_width_batch(engine, "mixed", rows)
}

fn build_general_insert(
    insert: &Insert,
    catalog: &CatalogSnapshot,
) -> Result<TypedInsertBatch, ExecuteError> {
    prepare_typed_insert_semantics_at(
        insert,
        catalog,
        catalog.commit_seq,
        None,
        InsertStatementOrdinal::FIRST,
    )?
    .expect("test catalog generation is current")
    .seal(sequence_defaults::SequenceDefaultBindings::empty())
}

fn ready_general_insert(insert: &Insert, catalog: &CatalogSnapshot) -> TypedInsertBatch {
    build_general_insert(insert, catalog).expect("typed semantic preparation succeeds")
}

fn exact_row_ids(ids: impl IntoIterator<Item = u64>) -> DeviceInsertRowIds {
    DeviceInsertRowIds::exact(ids.into_iter().collect::<Vec<_>>().into_boxed_slice())
}

fn synthetic_no_identity() -> DeviceInsertRowIds {
    DeviceInsertRowIds::synthetic_no_identity()
}

fn read_device_u64(memory: &CudaResidentDeviceMemory, slot: usize) -> u64 {
    let words = memory
        .read_resident_i32_column((slot * std::mem::size_of::<u64>()) as u64, 2)
        .expect("read device u64 words");
    (words[0] as u32 as u64) | ((words[1] as u32 as u64) << 32)
}

#[test]
fn typed_batch_binds_exact_columns_values_dependencies_and_statement_memory() {
    let (_engine, insert, catalog) = prepared_accounts(1_000);
    let batch = seal_typed_insert_batch_for_test(
        &Command::Insert(insert),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap()
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
    assert_eq!(batch.columns[0].values.as_i32().unwrap()[0], 0);
    assert_eq!(batch.columns[1].values.as_i32().unwrap()[0], 0);
    assert_eq!(batch.columns[0].values.as_i32().unwrap()[999], 999);
    assert_eq!(batch.columns[1].values.as_i32().unwrap()[999], 9_990);
    assert!(batch
        .columns
        .iter()
        .all(|column| matches!(column.validity, TypedInsertColumnValidity::AllValid)));
    assert_eq!(batch.dependencies.len(), 1);
    assert_eq!(batch.dependencies[0].name.as_ref(), "accounts");
    assert_eq!(batch.dependencies[0].oid, table.oid);
    assert_eq!(
        batch.dependencies[0].schema_digest,
        batch.table.schema_digest
    );
    assert_eq!(batch.value_bytes(), 1_000 * 2 * std::mem::size_of::<i32>());

    let (_engine, small_insert, small_catalog) = prepared_accounts(3);
    let small = seal_typed_insert_batch_for_test(
        &Command::Insert(small_insert),
        &small_catalog,
        small_catalog.commit_seq,
        None,
    )
    .unwrap()
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
fn typed_batch_has_one_nonclone_semantic_values_authority() {
    let (_engine, insert, catalog) = prepared_accounts(3);
    let batch = seal_typed_insert_batch_for_test(
        &Command::Insert(insert),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap()
    .expect("accounts shape is eligible");
    assert_eq!(batch.row_count, 3);
    assert!(batch
        .columns
        .iter()
        .all(|column| column.is_all_valid_i32(3)));
    let source = include_str!("typed_insert_batch.rs");
    assert!(!source.contains(&["struct Prepared", "InsertBatch"].concat()));
    assert!(!source.contains(&["from_", "authoritative_offlock_prepare"].concat()));
    assert!(source.contains("enum TypedInsertColumnValues"));
    assert!(source.contains("enum TypedInsertColumnValidity"));
    for wave_owner in [
        include_str!("engine_dml_concurrent/state.rs"),
        include_str!("engine_dml_concurrent/request.rs"),
        include_str!("engine_dml_concurrent/wave.rs"),
    ] {
        assert!(
            !wave_owner.contains("PreparedFixedWidthAppendSource"),
            "only the residency compiler may name its fixed-width physical source"
        );
        assert!(
            !wave_owner.contains("FixedWidthOpenShardAppendPlan"),
            "wave code must hold only DeviceInsertPlan"
        );
    }
}

#[test]
fn typed_device_compiler_rejects_catalog_drift_before_publish() {
    let (engine, insert, catalog) = prepared_accounts(2);
    let batch = || {
        seal_typed_insert_batch_for_test(
            &Command::Insert(insert.clone()),
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .unwrap()
    };
    let catalog_drift = batch();
    engine
        .execute_text(2, "CREATE TABLE catalog_drift (value int4)")
        .unwrap();
    let device_cells_after_catalog_drift = engine
        .read_state
        .residency
        .shard_device_memory
        .cells
        .load()
        .len();
    assert!(engine
        .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            catalog_drift,
            synthetic_no_identity()
        )
        .is_err());
    assert_eq!(
        engine
            .read_state
            .residency
            .shard_device_memory
            .cells
            .load()
            .len(),
        device_cells_after_catalog_drift,
        "compiler decline does not allocate or publish after the catalog-drift setup"
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
        rows: Insert::programmatic_rows(vec![
            vec![SqlValue::Int4(i32::MIN), SqlValue::Int4(-7)],
            vec![SqlValue::Int4(0), SqlValue::Int4(13)],
            vec![SqlValue::Int4(i32::MAX), SqlValue::Int4(-1)],
        ]),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    let batch = seal_typed_insert_batch_for_test(
        &Command::Insert(insert.clone()),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap()
    .unwrap();
    let row_major = crate::engine_residency::compute_open_shard_int4_append_chunks(
        &[SqlType::Int4, SqlType::Int4],
        11,
        4,
        &[
            vec![SqlValue::Int4(i32::MIN), SqlValue::Int4(-7)],
            vec![SqlValue::Int4(0), SqlValue::Int4(13)],
            vec![SqlValue::Int4(i32::MAX), SqlValue::Int4(-1)],
        ],
    )
    .unwrap();
    let typed = batch
        .into_codec5_resident_append_source_for_test(&catalog)
        .unwrap()
        .checked_append_chunks(11, 4)
        .unwrap();
    assert_eq!(
        typed.chunks.last().unwrap().byte_offset,
        0,
        "header is final"
    );
    assert_eq!(
        typed.chunks.last().unwrap().bytes.as_ref(),
        (7_u64).to_le_bytes()
    );
    let typed = IntoIterator::into_iter(typed.chunks)
        .map(|chunk| gpu_db_execution::CudaOwnedDeviceMemoryChunk {
            byte_offset: chunk.byte_offset,
            bytes: chunk.bytes.into_vec(),
        })
        .collect::<Vec<_>>();
    assert_eq!(typed, row_major);
}

#[test]
fn codec5_resident_source_drops_semantic_only_backing() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE source_retention (id int4, flag bool)")
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let insert = Insert {
        table: "source_retention".to_string(),
        columns: Vec::new(),
        rows: Insert::programmatic_rows(vec![
            vec![SqlValue::Int4(7), SqlValue::Bool(true)],
            vec![SqlValue::Int4(9), SqlValue::Bool(false)],
        ]),
        returning: Vec::new(),
    };
    let batch = ready_general_insert(&insert, &catalog);
    let batch_report = batch.host_retention_report().unwrap();
    let source = batch
        .into_codec5_resident_append_source_for_test(&catalog)
        .unwrap();
    let materialized = source.host_retention_report().unwrap();
    assert!(
        batch_report.retained_bytes() > materialized.retained_bytes(),
        "input state/provenance and canonical semantic-only storage must retire at the move"
    );
    assert!(batch_report.allocation_slots().unwrap() > materialized.allocation_slots().unwrap());
    assert_eq!(materialized.generation_pin_slots().unwrap(), 0);
}

#[test]
fn dense_vector_payload_matches_row_major_layout_for_all_types_and_bitmap_boundaries() {
    for rows in [1_usize, 32, 33] {
        let engine = Engine::new_local();
        let table_name = format!("dense_vectors_{rows}");
        engine
            .execute_text(
                1,
                &format!(
                    "CREATE TABLE {table_name} (small int2, id int4, day date, wide int8, stamp timestamp, amount numeric(10,2), token uuid, flag bool, note text, note_two text)"
                ),
            )
            .unwrap();
        let values = (0..rows)
            .map(|row| {
                let nullable = row == 0 || row == 32;
                vec![
                    if nullable {
                        SqlValue::Null
                    } else {
                        SqlValue::Int2(row as i16 - 2)
                    },
                    SqlValue::Int4(row as i32 - 10),
                    SqlValue::Date(20_000 + row as i32),
                    SqlValue::Int8(9_000_000_000 + row as i64),
                    SqlValue::Timestamp(1_785_168_000_000_000 + row as i64),
                    SqlValue::Numeric(gpu_db_sql::Decimal128::new(250 + row as i128, 2)),
                    SqlValue::Uuid([row as u8; 16]),
                    SqlValue::Bool(row % 2 == 0),
                    if nullable {
                        SqlValue::Null
                    } else {
                        SqlValue::Text(format!("left-{row}"))
                    },
                    SqlValue::Text(format!("right-{row}-alignment")),
                ]
            })
            .collect::<Vec<_>>();
        let insert = Insert {
            table: table_name.clone(),
            columns: Vec::new(),
            rows: Insert::programmatic_rows(values.clone()),
            returning: Vec::new(),
        };
        let catalog = engine.catalog_snapshot();
        let table = catalog.relational_catalog.get(&table_name).unwrap();
        let names = table
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let types = table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect::<Vec<_>>();
        let (expected_bytes, expected_text, expected_bool, expected_stats, _, expected_null) =
            crate::engine_residency::build_relational_device_payload_with_capacity(
                &names, &types, &values, rows,
            )
            .unwrap();
        let batch = ready_general_insert(&insert, &catalog);
        let source = batch
            .into_codec5_resident_append_source_for_test(&catalog)
            .unwrap();
        let mut source = source;
        let mut dense = source.checked_dense_payload(table).unwrap();
        assert!(
            source.dense_vectors_are_consumed(),
            "dense preflight drops logical values and validity before the plan crosses WAL"
        );
        let mut actual_bytes = dense.take_pre_wal_upload().unwrap();
        let descriptor = dense.into_descriptor_parts();
        actual_bytes[..descriptor.final_count_header.len()]
            .copy_from_slice(&descriptor.final_count_header);
        assert_eq!(actual_bytes, expected_bytes, "rows={rows}");
        assert_eq!(descriptor.text_layouts, expected_text, "rows={rows}");
        assert_eq!(descriptor.bool_layouts, expected_bool, "rows={rows}");
        assert_eq!(descriptor.int4_stats, expected_stats, "rows={rows}");
        assert_eq!(descriptor.null_layouts, expected_null, "rows={rows}");
    }
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
                .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
                    sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(-30)]]),
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
            plan.apply_after_typed_wal_claim(
                &engine,
                crate::engine_dml_concurrent::issue_test_only_typed_insert_post_wal_apply_permit(
                    visible_stamp,
                ),
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
fn typed_in_place_missing_created_by_sidecar_stamps_the_pre_wal_arc() {
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
        .expect("resident open shard");
    engine
        .read_state
        .residency
        .shard_created_by_memory
        .invalidate_shard("accounts", before.shard_id);
    engine
        .read_state
        .residency
        .with_shards_mut_for_table("accounts", |shards| {
            let open = shards
                .get_mut("accounts")
                .and_then(|shards| shards.last_mut())
                .expect("open descriptor remains published");
            open.created_by_region = None;
        });
    let plan = engine
        .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
            exact_row_ids([3]),
        )
        .expect("missing created-by sidecar is reserved before WAL");
    let planned = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("private pre-WAL sidecar does not replace the descriptor");
    assert!(planned.created_by_region.is_none());
    assert!(engine
        .read_state
        .residency
        .shard_created_by_memory
        .get(&("accounts".to_string(), before.shard_id))
        .is_none());
    let stamp = engine.committed_seq();
    plan.apply_after_typed_wal_claim(
        &engine,
        crate::engine_dml_concurrent::issue_test_only_typed_insert_post_wal_apply_permit(stamp),
    )
    .expect("post-WAL mutation installs and stamps the sealed sidecar");
    let after = engine
        .read_state
        .residency
        .shards
        .load()
        .get("accounts")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("stamped open descriptor");
    let descriptor_region = after
        .created_by_region
        .as_ref()
        .expect("post-WAL mutation publishes the reserved Arc");
    let side_map_region = engine
        .read_state
        .residency
        .shard_created_by_memory
        .get(&("accounts".to_string(), before.shard_id))
        .expect("side map retains the exact reserved Arc");
    assert!(std::sync::Arc::ptr_eq(descriptor_region, &side_map_region));
    assert_eq!(read_device_u64(descriptor_region, before.row_count), stamp);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn mixed_fixed_width_typed_rollover_preuploads_all_sections_and_exact_row_ids() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(4);
    engine.set_fused_apply_enabled(true);
    engine
        .execute_text(
            1,
            "CREATE TABLE mixed (id int4, stamp int8, amount numeric(10,2), enabled bool)",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "INSERT INTO mixed VALUES (1, 100, 1.00, true), (2, 200, 2.00, false)",
        )
        .unwrap();
    engine
        .populate_relational_residency_snapshot("mixed")
        .unwrap();
    let before = engine
        .read_state
        .residency
        .shards
        .load()
        .get("mixed")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("mixed fixed-width table has a resident open shard");
    let fused_before = engine.fused_apply_hits();
    engine
        .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            sealed_mixed_fixed_width_batch(
                &engine,
                vec![vec![
                    SqlValue::Int4(3),
                    SqlValue::Int8(300),
                    SqlValue::Numeric(gpu_db_sql::Decimal128::new(300, 2)),
                    SqlValue::Bool(true),
                ]],
            ),
            exact_row_ids([3]),
        )
        .unwrap()
        .apply_after_typed_wal_claim(
            &engine,
            crate::engine_dml_concurrent::issue_test_only_typed_insert_post_wal_apply_permit(
                engine.committed_seq(),
            ),
        )
        .expect("mixed typed in-place plan reaches the sole publisher");
    assert_eq!(
        engine.fused_apply_hits(),
        fused_before,
        "mixed plan skips i32-only fusion"
    );
    let in_place = engine
        .read_state
        .residency
        .shards
        .load()
        .get("mixed")
        .and_then(|shards| shards.last())
        .cloned()
        .unwrap();
    assert_eq!(in_place.shard_id, before.shard_id);
    assert_eq!(in_place.row_count, 3);
    assert_eq!(in_place.resident_device_int8_columns, vec!["stamp"]);
    assert_eq!(in_place.resident_device_numeric_columns, vec!["amount"]);
    assert_eq!(in_place.resident_device_bool_columns.len(), 1);

    engine
        .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            sealed_mixed_fixed_width_batch(
                &engine,
                vec![
                    vec![
                        SqlValue::Int4(4),
                        SqlValue::Int8(400),
                        SqlValue::Numeric(gpu_db_sql::Decimal128::new(400, 2)),
                        SqlValue::Bool(false),
                    ],
                    vec![
                        SqlValue::Int4(5),
                        SqlValue::Int8(500),
                        SqlValue::Numeric(gpu_db_sql::Decimal128::new(500, 2)),
                        SqlValue::Bool(true),
                    ],
                    vec![
                        SqlValue::Int4(6),
                        SqlValue::Int8(600),
                        SqlValue::Numeric(gpu_db_sql::Decimal128::new(600, 2)),
                        SqlValue::Bool(false),
                    ],
                ],
            ),
            exact_row_ids([4, 5, 6]),
        )
        .unwrap()
        .apply_after_typed_wal_claim(
            &engine,
            crate::engine_dml_concurrent::issue_test_only_typed_insert_post_wal_apply_permit(
                engine.committed_seq(),
            ),
        )
        .expect("mixed typed rollover remains a private build through the sole publisher");
    let rolled = engine
        .read_state
        .residency
        .shards
        .load()
        .get("mixed")
        .and_then(|shards| shards.last())
        .cloned()
        .unwrap();
    assert_eq!(rolled.shard_id, in_place.shard_id + 1);
    assert_eq!(rolled.row_count, 3);
    assert_eq!(rolled.resident_device_int8_columns, vec!["stamp"]);
    assert_eq!(rolled.resident_device_numeric_columns, vec!["amount"]);
    assert_eq!(rolled.resident_device_bool_columns.len(), 1);
    let row_ids = rolled
        .row_id_region
        .as_ref()
        .expect("typed fixed-width rollover retains exact row identities");
    assert_eq!(
        (0..rolled.row_count)
            .map(|slot| read_device_u64(row_ids, slot))
            .collect::<Vec<_>>(),
        vec![4, 5, 6]
    );
    assert_eq!(
        read_device_u64(row_ids, rolled.row_count),
        u64::MAX,
        "pre-WAL row-id allocation retains the unknown-id headroom fill"
    );
    let visible = engine
        .execute_relational_select_text("SELECT id, stamp, amount, enabled FROM mixed ORDER BY id")
        .unwrap();
    assert_eq!(visible.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(visible.fallback_reason, None);
    assert_eq!(
        visible.rows.into_boxed(),
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Int8(100),
                SqlValue::Numeric(gpu_db_sql::Decimal128::new(100, 2)),
                SqlValue::Bool(true)
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Int8(200),
                SqlValue::Numeric(gpu_db_sql::Decimal128::new(200, 2)),
                SqlValue::Bool(false)
            ],
            vec![
                SqlValue::Int4(3),
                SqlValue::Int8(300),
                SqlValue::Numeric(gpu_db_sql::Decimal128::new(300, 2)),
                SqlValue::Bool(true)
            ],
            vec![
                SqlValue::Int4(4),
                SqlValue::Int8(400),
                SqlValue::Numeric(gpu_db_sql::Decimal128::new(400, 2)),
                SqlValue::Bool(false)
            ],
            vec![
                SqlValue::Int4(5),
                SqlValue::Int8(500),
                SqlValue::Numeric(gpu_db_sql::Decimal128::new(500, 2)),
                SqlValue::Bool(true)
            ],
            vec![
                SqlValue::Int4(6),
                SqlValue::Int8(600),
                SqlValue::Numeric(gpu_db_sql::Decimal128::new(600, 2)),
                SqlValue::Bool(false)
            ],
        ],
        "the rolled generation is GPU-visible with its BoolBits values"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn nullable_text_typed_plan_preallocates_dense_rollover_and_publishes_once() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(4);
    engine
        .execute_text(1, "CREATE TABLE notes (id int4, note text, optional int8)")
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO notes VALUES (1, 'first', 10)")
        .unwrap();
    engine
        .populate_relational_residency_snapshot("notes")
        .unwrap();
    let before = engine
        .read_state
        .residency
        .shards
        .load()
        .get("notes")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("text table has a resident descriptor");
    let plan = engine
        .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            sealed_fixed_width_batch(
                &engine,
                "notes",
                vec![
                    vec![SqlValue::Int4(2), SqlValue::Null, SqlValue::Int8(20)],
                    vec![
                        SqlValue::Int4(3),
                        SqlValue::Text("third".to_string()),
                        SqlValue::Null,
                    ],
                ],
            ),
            exact_row_ids([2, 3]),
        )
        .expect("NULL/TEXT source seals a pre-WAL dense device plan");
    let planned = engine
        .read_state
        .residency
        .shards
        .load()
        .get("notes")
        .and_then(|shards| shards.last())
        .cloned()
        .unwrap();
    assert_eq!(planned.shard_id, before.shard_id, "pre-WAL plan is private");
    plan.apply_after_typed_wal_claim(
        &engine,
        crate::engine_dml_concurrent::issue_test_only_typed_insert_post_wal_apply_permit(
            engine.committed_seq(),
        ),
    )
    .expect("post-WAL plan consumes its private allocations through mutation");
    let after = engine
        .read_state
        .residency
        .shards
        .load()
        .get("notes")
        .and_then(|shards| shards.last())
        .cloned()
        .unwrap();
    assert_eq!(after.shard_id, before.shard_id + 1);
    assert_eq!(after.row_count, 2);
    assert_eq!(after.resident_device_text_columns.len(), 1);
    assert_eq!(after.resident_device_null_columns.len(), 2);
    let ids = after.row_id_region.as_ref().unwrap();
    assert_eq!(read_device_u64(ids, 0), 2);
    assert_eq!(read_device_u64(ids, 1), 3);
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id, note, optional FROM notes ORDER BY id")
            .unwrap()
            .rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("first".to_string()),
                SqlValue::Int8(10),
            ],
            vec![SqlValue::Int4(2), SqlValue::Null, SqlValue::Int8(20)],
            vec![
                SqlValue::Int4(3),
                SqlValue::Text("third".to_string()),
                SqlValue::Null,
            ],
        ],
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn generalized_section_descriptor_sabotage_declines_before_allocation_or_publication() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(8);
    engine
        .execute_text(
            1,
            "CREATE TABLE descriptor_sabotage (id int4, stamp_a int8, stamp_b timestamp, amount numeric(10,2), token uuid, flag bool)",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "INSERT INTO descriptor_sabotage VALUES (1, 10, '2026-07-27 00:00:00', 1.25, '00112233-4455-6677-8899-aabbccddeeff', true)",
        )
        .unwrap();
    engine
        .populate_relational_residency_snapshot("descriptor_sabotage")
        .unwrap();
    let baseline = engine
        .read_state
        .residency
        .shards
        .load()
        .get("descriptor_sabotage")
        .and_then(|shards| shards.last())
        .cloned()
        .expect("mixed fixed-width descriptor is resident before sabotage");
    let resident_bytes = engine.relational_resident_bytes_for_gpu(0);
    let uuid = gpu_db_sql::uuid::parse_uuid("10213243-5465-7687-98a9-bacbdcedfe0f")
        .expect("test UUID is valid");
    let assert_pre_wal_decline = |sabotage: fn(&mut RelationalResidentShard)| {
        engine
            .read_state
            .residency
            .with_shards_mut_for_table("descriptor_sabotage", |shards| {
                let open = shards
                    .get_mut("descriptor_sabotage")
                    .and_then(|shards| shards.last_mut())
                    .expect("sabotage retains one open descriptor");
                *open = baseline.clone();
                sabotage(open);
            });
        assert!(matches!(
            engine.compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
                sealed_fixed_width_batch(
                    &engine,
                    "descriptor_sabotage",
                    vec![vec![
                        SqlValue::Int4(2),
                        SqlValue::Int8(20),
                        SqlValue::Timestamp(1_785_168_000_000_000),
                        SqlValue::Numeric(gpu_db_sql::Decimal128::new(250, 2)),
                        SqlValue::Uuid(uuid),
                        SqlValue::Bool(false),
                    ]],
                ),
                exact_row_ids([2]),
            ),
            Err(crate::engine_residency::DeviceInsertPlanPrepareError::UnsupportedShape)
        ));
        let after = engine
            .read_state
            .residency
            .shards
            .load()
            .get("descriptor_sabotage")
            .and_then(|shards| shards.last())
            .cloned()
            .expect("failed pre-WAL compile must retain the sabotaged descriptor");
        assert_eq!(after.shard_id, baseline.shard_id);
        assert_eq!(after.row_count, baseline.row_count);
        assert_eq!(after.device_memory_proof, baseline.device_memory_proof);
        assert_eq!(engine.relational_resident_bytes_for_gpu(0), resident_bytes);
    };
    assert_pre_wal_decline(|open| open.resident_device_int8_columns.swap(0, 1));
    assert_pre_wal_decline(|open| open.resident_device_numeric_columns.swap(0, 1));
    assert_pre_wal_decline(|open| open.resident_device_bool_columns[0].name = "wrong".to_string());
    assert_pre_wal_decline(|open| open.resident_device_bool_columns[0].bitmap_byte_offset += 4);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn typed_fixed_width_rollover_preallocates_generation_and_publishes_through_mutation() {
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
        .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test_at_commit(
            sealed_i32_batch(
                &engine,
                vec![
                    vec![SqlValue::Int4(i32::MIN), SqlValue::Int4(i32::MAX)],
                    vec![SqlValue::Int4(0), SqlValue::Int4(-1)],
                    vec![SqlValue::Int4(i32::MAX), SqlValue::Int4(i32::MIN)],
                ],
            ),
            exact_row_ids(row_ids),
            unpublished_stamp,
        )
        .expect("pre-WAL plan seals the rollover capacity and budget");
    assert_eq!(
        engine.resident_shard_count("accounts"),
        before_count,
        "planning retains the old descriptor and does not publish a shard"
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
    plan.apply_after_typed_wal_claim(
        &engine,
        crate::engine_dml_concurrent::issue_test_only_typed_insert_post_wal_apply_permit(
            unpublished_stamp,
        ),
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
        engine.compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(1), SqlValue::Int4(10)]]),
            synthetic_no_identity(),
        ),
        Err(crate::engine_residency::DeviceInsertPlanPrepareError::UnsupportedShape)
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
            engine.compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
                sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(1), SqlValue::Int4(10)]]),
                exact_row_ids([1]),
            ),
            Err(crate::engine_residency::DeviceInsertPlanPrepareError::UnsupportedShape)
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
            engine.compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
                sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(1), SqlValue::Int4(10)]]),
                exact_row_ids([1]),
            ),
            Err(crate::engine_residency::DeviceInsertPlanPrepareError::UnsupportedShape)
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
        engine.compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            sealed_i32_batch(
                &engine,
                vec![
                    vec![SqlValue::Int4(1), SqlValue::Int4(10)],
                    vec![SqlValue::Int4(2), SqlValue::Int4(20)],
                ],
            ),
            exact_row_ids([row_id_before, row_id_before + 1]),
        ),
        Err(crate::engine_residency::DeviceInsertPlanPrepareError::RetryableBoundBootstrapResource)
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
        engine.compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
            exact_row_ids([row_id_before + 2]),
        ),
        Err(crate::engine_residency::DeviceInsertPlanPrepareError::UnsupportedShape)
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
    let batch = sealed_i32_batch(
        &engine,
        (3..10)
            .map(|id| vec![SqlValue::Int4(id), SqlValue::Int4(-id)])
            .collect(),
    );
    assert!(
        engine
            .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
                batch,
                exact_row_ids(3..10)
            )
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
        .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
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
        plan.apply_after_typed_wal_claim(
            &engine,
            crate::engine_dml_concurrent::issue_test_only_typed_insert_post_wal_apply_permit(
                engine.committed_seq(),
            ),
        ),
        Err(crate::engine_residency::DeviceInsertPlanApplyError::PlanDrift)
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
            .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
                sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
                synthetic_no_identity(),
            )
            .is_err(),
        "a first created_by sidecar must be budgeted before WAL"
    );
    engine.set_relational_residency_budget_bytes(0, u64::MAX);
    assert!(
        engine
            .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
                sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
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
        .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(3), SqlValue::Int4(30)]]),
            exact_row_ids([3]),
        )
        .expect("initial plan is eligible");
    first
        .apply_after_typed_wal_claim(
            &engine,
            crate::engine_dml_concurrent::issue_test_only_typed_insert_post_wal_apply_permit(stamp),
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
        .compile_transaction_terminal_typed_insert_device_plan_from_batch_for_test(
            sealed_i32_batch(&engine, vec![vec![SqlValue::Int4(4), SqlValue::Int4(40)]]),
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
fn parsed_1000_row_accounts_workload_prepares_to_the_typed_batch() {
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
    let batch = seal_typed_insert_batch_for_test(&command, &catalog, catalog.commit_seq, None)
        .unwrap()
        .expect("the exact parsed workload is eligible");
    assert_eq!(batch.row_count, 1_000);
    assert_eq!(batch.columns.len(), 2);
    assert_eq!(batch.columns[0].values.as_i32().unwrap()[0], 0);
    assert_eq!(batch.columns[1].values.as_i32().unwrap()[0], 0);
    assert_eq!(batch.columns[0].values.as_i32().unwrap()[999], 999);
    assert_eq!(batch.columns[1].values.as_i32().unwrap()[999], 9_990);
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
    let batch = seal_typed_insert_batch_for_test(
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
    assert_eq!(batch.columns[0].values.as_i32().unwrap(), &[0, 1]);
    assert_eq!(batch.columns[1].values.as_i32().unwrap(), &[0, 10]);

    let mut reordered = insert.clone();
    reordered.columns = vec!["balance".to_string(), "id".to_string()];
    assert!(seal_typed_insert_batch_for_test(
        &Command::Insert(reordered),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap()
    .is_some());
    assert!(seal_typed_insert_batch_for_test(
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
fn typed_batch_declines_stale_catalog_generation() {
    let (_engine, insert, catalog) = prepared_accounts(2);
    let command = Command::Insert(insert.clone());
    assert!(
        seal_typed_insert_batch_for_test(&command, &catalog, catalog.commit_seq, None,)
            .unwrap()
            .is_some()
    );

    assert!(
        seal_typed_insert_batch_for_test(&command, &catalog, catalog.commit_seq + 1, None,)
            .unwrap()
            .is_none()
    );
    let source = include_str!("typed_insert_batch.rs");
    assert!(!source.contains(&["from_", "authoritative_offlock_prepare"].concat()));
}

#[test]
fn nullable_defaults_returning_and_foreign_keys_share_one_semantic_batch() {
    let (engine, insert, catalog) = prepared_accounts(1);
    let mut null = insert.clone();
    null.rows[0][1] = InsertCell::programmatic(SqlValue::Null);
    assert!(seal_typed_insert_batch_for_test(
        &Command::Insert(null),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap()
    .is_some());
    let mut text = insert.clone();
    text.rows[0][1] = InsertCell::programmatic(SqlValue::Text("not-fixed-width".to_string()));
    assert!(seal_typed_insert_batch_for_test(
        &Command::Insert(text),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .is_err());
    let mut returning = insert.clone();
    returning.returning.push("id".to_string());
    assert!(seal_typed_insert_batch_for_test(
        &Command::Insert(returning),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap()
    .is_some());
    engine
        .execute_text(2, "CREATE TABLE defaults (id int4 DEFAULT 1, balance int4)")
        .unwrap();
    let defaults = engine.catalog_snapshot();
    assert!(
        seal_typed_insert_batch_for_test(
            &Command::Insert(Insert {
                table: "defaults".to_string(),
                ..insert.clone()
            }),
            &defaults,
            defaults.commit_seq,
            None,
        )
        .unwrap()
        .is_some(),
        "a table may define a default when every cell is supplied"
    );
    engine
        .execute_text(3, "CREATE TABLE parents (id int4 PRIMARY KEY)")
        .unwrap();
    let mut fk = (*engine.catalog_snapshot()).clone();
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
    assert!(seal_typed_insert_batch_for_test(
        &Command::Insert(insert.clone()),
        &fk,
        fk.commit_seq,
        None,
    )
    .unwrap()
    .is_some());
}

#[test]
fn exact_full_column_list_and_catalog_expectation_build_typed_batch() {
    let (_engine, mut insert, catalog) = prepared_accounts(2);
    insert.columns = vec!["id".to_string(), "balance".to_string()];
    let exact_expectation = Some(
        crate::engine_mutation_admission::CatalogVersionExpectation::Prepared(catalog.commit_seq),
    );
    assert!(seal_typed_insert_batch_for_test(
        &Command::Insert(insert.clone()),
        &catalog,
        catalog.commit_seq,
        exact_expectation,
    )
    .unwrap()
    .is_some());
    assert!(seal_typed_insert_batch_for_test(
        &Command::Insert(insert.clone()),
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
    .is_err());

    let mut reordered = insert.clone();
    reordered.columns = vec!["balance".to_string(), "id".to_string()];
    assert!(seal_typed_insert_batch_for_test(
        &Command::Insert(reordered),
        &catalog,
        catalog.commit_seq,
        exact_expectation,
    )
    .unwrap()
    .is_some());
    let mut partial = insert.clone();
    partial.columns = vec!["id".to_string()];
    assert!(seal_typed_insert_batch_for_test(
        &Command::Insert(partial),
        &catalog,
        catalog.commit_seq,
        exact_expectation,
    )
    .is_err());
    for columns in [
        vec!["id".to_string(), "id".to_string()],
        vec!["id".to_string(), "unknown".to_string()],
    ] {
        let mut malformed = insert.clone();
        malformed.columns = columns;
        assert!(seal_typed_insert_batch_for_test(
            &Command::Insert(malformed),
            &catalog,
            catalog.commit_seq,
            exact_expectation,
        )
        .is_err());
    }
}

#[test]
fn general_builder_reorders_columns_and_resolves_absent_defaults_without_collapsing_null() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE general_insert (id int4, note text, total int8, enabled bool)",
        )
        .unwrap();
    let insert = Insert {
        table: "general_insert".to_string(),
        columns: vec!["note".to_string(), "id".to_string()],
        rows: Insert::programmatic_rows(vec![
            vec![SqlValue::Text(String::new()), SqlValue::Int4(7)],
            vec![SqlValue::Null, SqlValue::Int4(8)],
        ]),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    let batch = ready_general_insert(&insert, &catalog);
    let table = catalog.relational_catalog.get("general_insert").unwrap();

    assert_eq!(
        batch
            .columns
            .iter()
            .map(|column| column.column_id)
            .collect::<Vec<_>>(),
        table
            .columns
            .iter()
            .map(|column| column.id)
            .collect::<Vec<_>>(),
        "the input list is reordered once into catalog identity order"
    );
    assert!(matches!(
        batch.columns[0].presence,
        TypedInsertColumnPresence::AllProvided
    ));
    let TypedInsertColumnValues::Text { offsets, bytes } = &batch.columns[1].values else {
        panic!("note must use the text vector arm");
    };
    assert_eq!(offsets.as_ref(), &[0, 0, 0]);
    assert!(bytes.is_empty(), "empty text stores no bytes");
    assert!(matches!(
        batch.columns[1].presence,
        TypedInsertColumnPresence::AllProvided
    ));
    let TypedInsertColumnValidity::Bitmap(note_validity) = &batch.columns[1].validity else {
        panic!("the supplied SQL NULL requires a validity bitmap");
    };
    assert_eq!(note_validity.as_ref(), &[0b1]);
    assert!(matches!(
        batch.columns[2].presence,
        TypedInsertColumnPresence::AllProvided
    ));
    let TypedInsertColumnValidity::Bitmap(total_validity) = &batch.columns[2].validity else {
        panic!("an absent default must materialize as SQL NULL");
    };
    assert_eq!(total_validity.as_ref(), &[0]);
    assert!(matches!(
        &batch.columns[2].values,
        TypedInsertColumnValues::I64(values) if values.as_ref() == [0, 0]
    ));
    assert!(matches!(
        &batch.columns[3].values,
        TypedInsertColumnValues::BoolBits(words) if words.as_ref() == [0]
    ));
    assert!(matches!(
        &batch.columns[2].default_resolution,
        TypedInsertDefaultResolution::Bitmap(words) if words.as_ref() == [0b11]
    ));
    assert!(matches!(
        &batch.columns[3].default_resolution,
        TypedInsertDefaultResolution::Bitmap(words) if words.as_ref() == [0b11]
    ));

    assert!(
        seal_typed_insert_batch_for_test(
            &Command::Insert(insert),
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .is_some(),
        "resolved absent defaults are catalog-order vectors suitable for the live compiler"
    );
}

#[test]
fn semantic_ir_keeps_literal_null_omission_and_explicit_default_distinct() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE semantic_ir (id int4, supplied_null int4, omitted int4 DEFAULT 19, explicit_default int4 DEFAULT 23)",
        )
        .unwrap();
    let Command::Insert(insert) = parse_command(
        "INSERT INTO semantic_ir (id, supplied_null, explicit_default) VALUES (7, NULL, DEFAULT)",
    )
    .unwrap() else {
        panic!("parser must produce an INSERT");
    };
    assert!(matches!(
        insert.rows[0][0],
        InsertCell::Value {
            provenance: gpu_db_sql::InsertValueProvenance::Literal,
            ..
        }
    ));
    assert!(matches!(
        insert.rows[0][2],
        InsertCell::Default {
            provenance: gpu_db_sql::InsertDefaultProvenance::SqlKeyword,
        }
    ));

    let catalog = engine.catalog_snapshot();
    let batch = ready_general_insert(&insert, &catalog);
    assert_eq!(batch.statement_ordinal, InsertStatementOrdinal::FIRST);
    assert_eq!(
        batch.columns[0].input_states.as_ref(),
        &[TypedInsertInputState::Provided]
    );
    assert_eq!(
        batch.columns[1].input_states.as_ref(),
        &[TypedInsertInputState::ProvidedNull]
    );
    assert_eq!(
        batch.columns[2].input_states.as_ref(),
        &[TypedInsertInputState::Omitted]
    );
    assert_eq!(
        batch.columns[3].input_states.as_ref(),
        &[TypedInsertInputState::ExplicitDefault]
    );
    assert_eq!(
        batch.columns[0].source_ordinal(batch.statement_ordinal, 0),
        Some(InsertSourceOrdinal {
            statement: InsertStatementOrdinal::FIRST,
            row: 0,
            column: 0,
        })
    );
    assert_eq!(
        batch.columns[1].source_ordinal(batch.statement_ordinal, 0),
        Some(InsertSourceOrdinal {
            statement: InsertStatementOrdinal::FIRST,
            row: 0,
            column: 1,
        })
    );
    assert_eq!(
        batch.columns[2].input_provenance[0],
        TypedInsertInputProvenance::Omitted
    );
    assert_eq!(
        batch.columns[2].source_ordinal(batch.statement_ordinal, 0),
        None
    );
    assert_eq!(
        batch.columns[3].source_ordinal(batch.statement_ordinal, 0),
        Some(InsertSourceOrdinal {
            statement: InsertStatementOrdinal::FIRST,
            row: 0,
            column: 2,
        })
    );
    assert_eq!(
        batch.columns[3].input_provenance[0],
        TypedInsertInputProvenance::SqlDefault
    );
    assert_eq!(
        batch.columns[0].input_provenance[0],
        TypedInsertInputProvenance::Literal
    );
    assert!(matches!(
        &batch.columns[2].values,
        TypedInsertColumnValues::I32(values) if values.as_ref() == [19]
    ));
    assert!(matches!(
        &batch.columns[3].values,
        TypedInsertColumnValues::I32(values) if values.as_ref() == [23]
    ));
    assert!(
        seal_typed_insert_batch_for_test(
            &Command::Insert(insert),
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .is_some(),
        "literal DEFAULT and omission are resolved before the fixed-width compiler"
    );
}

#[test]
fn semantic_ir_retains_bound_parameter_provenance_without_sql_reparse() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE bound_semantic_ir (id int4, note int4, omitted int4 DEFAULT 19, explicit_default int4 DEFAULT 23)",
        )
        .unwrap();
    let bound = PreparedCommand::parse(
        "INSERT INTO bound_semantic_ir (note, explicit_default, id) VALUES ($1, DEFAULT, $2)",
    )
    .unwrap()
    .bind(&[SqlValue::Null, SqlValue::Int4(7)])
    .unwrap();
    let Command::Insert(insert) = bound.command() else {
        panic!("Bind must retain the parsed INSERT AST");
    };
    assert!(matches!(
        insert.rows[0][0],
        InsertCell::Value {
            provenance: gpu_db_sql::InsertValueProvenance::BoundParameter { index: 1 },
            value: SqlValue::Null,
        }
    ));
    assert!(matches!(
        insert.rows[0][2],
        InsertCell::Value {
            provenance: gpu_db_sql::InsertValueProvenance::BoundParameter { index: 2 },
            value: SqlValue::Int4(7),
        }
    ));

    let catalog = engine.catalog_snapshot();
    let batch = ready_general_insert(insert, &catalog);
    assert_eq!(
        batch
            .columns
            .iter()
            .map(|column| column.column_id)
            .collect::<Vec<_>>(),
        catalog
            .relational_catalog
            .get("bound_semantic_ir")
            .unwrap()
            .columns
            .iter()
            .map(|column| column.id)
            .collect::<Vec<_>>(),
        "bound source order is resolved once into stable catalog order"
    );
    assert_eq!(
        batch.columns[0].input_provenance[0],
        TypedInsertInputProvenance::BoundParameter { index: 2 }
    );
    assert_eq!(
        batch.columns[1].input_provenance[0],
        TypedInsertInputProvenance::BoundParameter { index: 1 }
    );
    assert_eq!(
        batch.columns[2].input_states[0],
        TypedInsertInputState::Omitted
    );
    assert_eq!(
        batch.columns[3].input_states[0],
        TypedInsertInputState::ExplicitDefault
    );
}

#[test]
fn semantic_ir_rejects_fabricated_unbound_parameter_provenance() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE parameter_provenance_ir (id int4)")
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let error_for = |value, provenance| {
        let insert = Insert {
            table: "parameter_provenance_ir".to_string(),
            columns: vec!["id".to_string()],
            rows: vec![vec![InsertCell::Value { value, provenance }]],
            returning: Vec::new(),
        };
        match build_general_insert(&insert, &catalog) {
            Err(error) => error,
            Ok(_) => panic!("fabricated unbound parameter provenance must fail closed"),
        }
    };

    let matched_unbound = error_for(
        SqlValue::Parameter {
            index: 1,
            cast: None,
        },
        gpu_db_sql::InsertValueProvenance::Parameter { index: 1 },
    );
    assert!(matches!(
        matched_unbound,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "unbound INSERT parameter reached semantic preparation"
    ));
    let mismatched_slot = error_for(
        SqlValue::Parameter {
            index: 1,
            cast: None,
        },
        gpu_db_sql::InsertValueProvenance::Parameter { index: 2 },
    );
    assert!(matches!(
        mismatched_slot,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "unbound INSERT parameter provenance does not match its parameter slot"
    ));
    let fabricated_scalar = error_for(
        SqlValue::Int4(1),
        gpu_db_sql::InsertValueProvenance::Parameter { index: 1 },
    );
    assert!(matches!(
        fabricated_scalar,
        ExecuteError::Engine(EngineError::ApplyFailed(message))
            if message == "unbound INSERT parameter provenance requires a parameter slot"
    ));
}

#[test]
fn typed_semantic_metadata_is_compact_and_source_identity_is_column_scoped() {
    assert_eq!(std::mem::size_of::<TypedInsertInputState>(), 1);
    assert!(std::mem::size_of::<TypedInsertInputProvenance>() <= 8);
    assert!(
        std::mem::size_of::<TypedInsertInputState>()
            + std::mem::size_of::<TypedInsertInputProvenance>()
            <= 9,
        "sealed INSERT metadata has a bounded nine-byte per-cell payload"
    );

    let (_engine, insert, catalog) = prepared_accounts(3);
    let batch = ready_general_insert(&insert, &catalog);
    let metadata_bytes = batch
        .columns
        .iter()
        .map(|column| {
            column.input_states.len() * std::mem::size_of::<TypedInsertInputState>()
                + column.input_provenance.len() * std::mem::size_of::<TypedInsertInputProvenance>()
        })
        .sum::<usize>();
    assert_eq!(
        metadata_bytes,
        batch.row_count as usize
            * batch.columns.len()
            * (std::mem::size_of::<TypedInsertInputState>()
                + std::mem::size_of::<TypedInsertInputProvenance>())
    );
    assert_eq!(
        batch
            .columns
            .iter()
            .map(|column| column.source_column_ordinal)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1)]
    );
    assert_eq!(
        batch.columns[1].source_ordinal(batch.statement_ordinal, 2),
        Some(InsertSourceOrdinal {
            statement: InsertStatementOrdinal::FIRST,
            row: 2,
            column: 1,
        })
    );
}

#[test]
fn explicit_default_omission_and_null_keep_distinct_runtime_results() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE default_runtime_ir (id int4, value int4 DEFAULT 9)",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "INSERT INTO default_runtime_ir (id, value) VALUES (1, NULL), (2, DEFAULT)",
        )
        .unwrap();
    engine
        .execute_text(3, "INSERT INTO default_runtime_ir (id) VALUES (3)")
        .unwrap();
    assert_eq!(
        engine
            .execute_relational_select_text("SELECT id, value FROM default_runtime_ir ORDER BY id")
            .unwrap()
            .rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Null],
            vec![SqlValue::Int4(2), SqlValue::Int4(9)],
            vec![SqlValue::Int4(3), SqlValue::Int4(9)],
        ],
        "the compatibility lowering resolves DEFAULT/omission only after semantic preparation"
    );
}

#[test]
fn insert_cell_json_preserves_legacy_value_bytes_and_round_trips_default() {
    let legacy = br#"{"Insert":{"table":"t","columns":["id"],"rows":[[{"Int4":1}]]}}"#;
    let decoded: Command = serde_json::from_slice(legacy).unwrap();
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), legacy);
    let Command::Insert(legacy_insert) = decoded else {
        panic!("legacy shape must decode as INSERT");
    };
    assert!(matches!(
        legacy_insert.rows[0][0],
        InsertCell::Value {
            provenance: gpu_db_sql::InsertValueProvenance::Programmatic,
            value: SqlValue::Int4(1),
        }
    ));

    // Every serializable legacy SqlValue wire arm remains byte-identical when decoded through
    // InsertCell. Numeric deliberately exercises serde's i128 scalar path that untagged
    // deserialization cannot replay into a nested enum.
    for value in [
        SqlValue::Null,
        SqlValue::Int4(-4),
        SqlValue::Int8(5_000_000_000),
        SqlValue::Numeric(gpu_db_sql::Decimal128::new(123_456, 2)),
        SqlValue::Bool(true),
        SqlValue::Text("legacy text".to_string()),
        SqlValue::Date(9_000),
        SqlValue::Timestamp(123_456_789),
        SqlValue::Uuid([0xAB; 16]),
        SqlValue::Int2(-2),
    ] {
        let legacy = serde_json::to_vec(&value).unwrap();
        let decoded: InsertCell = serde_json::from_slice(&legacy).unwrap();
        assert_eq!(decoded, InsertCell::programmatic(value));
        assert_eq!(serde_json::to_vec(&decoded).unwrap(), legacy);
    }

    let Command::Insert(parsed) = parse_command("INSERT INTO t (id) VALUES (DEFAULT)").unwrap()
    else {
        panic!("DEFAULT syntax must parse as INSERT");
    };
    assert_eq!(
        serde_json::to_vec(&InsertCell::sql_default()).unwrap(),
        br#"{"$gpu_db_insert_cell":"default_v1"}"#
    );
    let bytes = serde_json::to_vec(&parsed).unwrap();
    assert!(std::str::from_utf8(&bytes)
        .unwrap()
        .contains("\"$gpu_db_insert_cell\":\"default_v1\""));
    let round_trip: Insert = serde_json::from_slice(&bytes).unwrap();
    assert!(matches!(
        round_trip.rows[0][0],
        InsertCell::Default {
            provenance: gpu_db_sql::InsertDefaultProvenance::SqlKeyword,
        }
    ));

    let programmatic = Insert {
        table: "t".to_string(),
        columns: vec!["id".to_string()],
        rows: vec![vec![InsertCell::programmatic_default()]],
        returning: Vec::new(),
    };
    assert_eq!(
        serde_json::to_vec(&InsertCell::programmatic_default()).unwrap(),
        br#"{"$gpu_db_insert_cell":"programmatic_default_v1"}"#
    );
    let bytes = serde_json::to_vec(&programmatic).unwrap();
    assert!(std::str::from_utf8(&bytes)
        .unwrap()
        .contains("\"$gpu_db_insert_cell\":\"programmatic_default_v1\""));
    let round_trip: Insert = serde_json::from_slice(&bytes).unwrap();
    assert!(matches!(
        round_trip.rows[0][0],
        InsertCell::Default {
            provenance: gpu_db_sql::InsertDefaultProvenance::Programmatic,
        }
    ));
}

#[test]
fn omitted_literal_default_reaches_general_semantic_and_fixed_route_batches() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE defaults_ir (id int4 DEFAULT 9, balance int4)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let omitted = Insert {
        table: "defaults_ir".to_string(),
        columns: vec!["balance".to_string()],
        rows: Insert::programmatic_rows(vec![vec![SqlValue::Int4(2)]]),
        returning: Vec::new(),
    };
    let table = catalog.relational_catalog.get("defaults_ir").unwrap();
    let omitted_batch = ready_general_insert(&omitted, &catalog);
    assert_eq!(
        omitted_batch.columns[0].input_states[0],
        TypedInsertInputState::Omitted
    );
    assert_eq!(omitted_batch.columns[0].column_id, table.columns[0].id);
    assert!(matches!(
        &omitted_batch.columns[0].values,
        TypedInsertColumnValues::I32(values) if values.as_ref() == [9]
    ));
    let supplied = Insert {
        table: "defaults_ir".to_string(),
        columns: vec!["balance".to_string(), "id".to_string()],
        rows: Insert::programmatic_rows(vec![vec![SqlValue::Int4(2), SqlValue::Int4(9)]]),
        returning: Vec::new(),
    };
    let batch = ready_general_insert(&supplied, &catalog);
    assert!(batch
        .columns
        .iter()
        .all(|column| column.presence.all_provided(1)));
    assert!(
        seal_typed_insert_batch_for_test(
            &Command::Insert(supplied),
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .is_some(),
        "a catalog default is irrelevant when every corresponding cell is supplied"
    );
}

#[test]
fn general_builder_preserves_relation_column_arity_and_coercion_precedence() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE precedence_ir (id int4, value int4)")
        .unwrap();
    let catalog = engine.catalog_snapshot();
    let assert_error = |insert: Insert| match build_general_insert(&insert, &catalog) {
        Err(error) => error,
        Ok(_) => panic!("malformed general input must be a semantic error"),
    };

    let relation = assert_error(Insert {
        table: "missing_ir".to_string(),
        columns: vec!["id".to_string(), "id".to_string()],
        rows: Insert::programmatic_rows(vec![vec![SqlValue::Text("bad".to_string())]]),
        returning: Vec::new(),
    });
    assert!(matches!(
        relation,
        ExecuteError::Engine(EngineError::ApplyFailed(message)) if message == "relation \"missing_ir\" does not exist"
    ));
    let duplicate = assert_error(Insert {
        table: "precedence_ir".to_string(),
        columns: vec!["id".to_string(), "id".to_string(), "missing".to_string()],
        rows: Insert::programmatic_rows(vec![vec![
            SqlValue::Int4(1),
            SqlValue::Int4(2),
            SqlValue::Int4(3),
        ]]),
        returning: Vec::new(),
    });
    assert!(matches!(
        duplicate,
        ExecuteError::Engine(EngineError::DuplicateColumn(name)) if name == "id"
    ));
    let undefined = assert_error(Insert {
        table: "precedence_ir".to_string(),
        columns: vec!["missing".to_string()],
        rows: Insert::programmatic_rows(vec![vec![SqlValue::Int4(1)]]),
        returning: Vec::new(),
    });
    assert!(matches!(
        undefined,
        ExecuteError::Engine(EngineError::UndefinedColumn(name)) if name == "missing"
    ));
    let arity = assert_error(Insert {
        table: "precedence_ir".to_string(),
        columns: vec!["id".to_string()],
        rows: vec![vec![]],
        returning: Vec::new(),
    });
    assert!(matches!(
        arity,
        ExecuteError::Engine(EngineError::ApplyFailed(message)) if message == "INSERT value count must match target columns"
    ));
    let coercion = assert_error(Insert {
        table: "precedence_ir".to_string(),
        columns: vec!["id".to_string()],
        rows: Insert::programmatic_rows(vec![vec![SqlValue::Text("bad".to_string())]]),
        returning: Vec::new(),
    });
    assert!(matches!(
        coercion,
        ExecuteError::Engine(EngineError::ApplyFailed(message)) if message == "invalid value for column \"id\""
    ));
    let reordered_coercion = assert_error(Insert {
        table: "precedence_ir".to_string(),
        columns: vec!["value".to_string(), "id".to_string()],
        rows: Insert::programmatic_rows(vec![vec![
            SqlValue::Text("first-source-error".to_string()),
            SqlValue::Text("second-source-error".to_string()),
        ]]),
        returning: Vec::new(),
    });
    assert!(matches!(
        reordered_coercion,
        ExecuteError::Engine(EngineError::ApplyFailed(message)) if message == "invalid value for column \"value\""
    ));
}

#[test]
fn ordinary_typed_insert_carriers_cannot_bypass_finite_temporal_validation() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE temporal_carrier_batch (d date, t timestamp)",
        )
        .unwrap();
    let catalog = engine.catalog_snapshot();
    for (value, ty, row) in [
        (
            SqlValue::Date(gpu_db_sql::datetime::PG_DATE_END_DAYS_EXCLUSIVE),
            SqlType::Date,
            vec![
                SqlValue::Date(gpu_db_sql::datetime::PG_DATE_END_DAYS_EXCLUSIVE),
                SqlValue::Timestamp(0),
            ],
        ),
        (
            SqlValue::Timestamp(gpu_db_sql::datetime::PG_TIMESTAMP_MIN_MICROS - 1),
            SqlType::Timestamp,
            vec![
                SqlValue::Date(0),
                SqlValue::Timestamp(gpu_db_sql::datetime::PG_TIMESTAMP_MIN_MICROS - 1),
            ],
        ),
    ] {
        assert!(matches!(
            coerce_insert_value(value, ty, "temporal"),
            Err(EngineError::DatetimeFieldOverflow(_))
        ));
        let insert = Insert {
            table: "temporal_carrier_batch".to_string(),
            columns: Vec::new(),
            rows: Insert::programmatic_rows(vec![row]),
            returning: Vec::new(),
        };
        assert!(matches!(
            seal_typed_insert_batch_for_test(
                &Command::Insert(insert),
                &catalog,
                catalog.commit_seq,
                None,
            ),
            Err(ExecuteError::Engine(EngineError::DatetimeFieldOverflow(_)))
        ));
    }
}

#[test]
fn general_builder_all_scalar_arms_and_wal_template_are_byte_identical() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE scalar_ir (small int2, integer int4, day date, big int8, moment timestamp, amount numeric(10,2), token uuid, flag bool, note text)",
        )
        .unwrap();
    let uuid_text = "00112233-4455-6677-8899-aabbccddeeff";
    let insert = Insert {
        table: "scalar_ir".to_string(),
        columns: vec![
            "note".to_string(),
            "flag".to_string(),
            "token".to_string(),
            "amount".to_string(),
            "moment".to_string(),
            "big".to_string(),
            "day".to_string(),
            "integer".to_string(),
            "small".to_string(),
        ],
        rows: Insert::programmatic_rows(vec![vec![
            SqlValue::Text("a|b\\c".to_string()),
            SqlValue::Bool(true),
            SqlValue::Text(uuid_text.to_string()),
            SqlValue::Int4(123),
            SqlValue::Text("2026-07-27 12:34:56".to_string()),
            SqlValue::Int4(99),
            SqlValue::Text("2026-07-27".to_string()),
            SqlValue::Int4(-44),
            SqlValue::Int4(-7),
        ]]),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    let batch = ready_general_insert(&insert, &catalog);
    assert!(matches!(
        &batch.columns[0].values,
        TypedInsertColumnValues::I32(values) if values.as_ref() == [-7]
    ));
    assert!(matches!(
        &batch.columns[1].values,
        TypedInsertColumnValues::I32(values) if values.as_ref() == [-44]
    ));
    assert!(matches!(
        &batch.columns[3].values,
        TypedInsertColumnValues::I64(values) if values.as_ref() == [99]
    ));
    assert!(matches!(
        &batch.columns[5].values,
        TypedInsertColumnValues::I128(values) if values.as_ref() == [12_300]
    ));
    assert!(matches!(
        &batch.columns[6].values,
        TypedInsertColumnValues::Bytes16(values)
            if values[0] == [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]
    ));
    assert!(matches!(
        &batch.columns[7].values,
        TypedInsertColumnValues::BoolBits(words) if words.as_ref() == [1]
    ));

    let table = catalog.relational_catalog.get("scalar_ir").unwrap();
    let expected = table
        .columns
        .iter()
        .enumerate()
        .map(|(catalog_index, column)| {
            let source_index = insert
                .columns
                .iter()
                .position(|name| name == &column.name)
                .unwrap();
            coerce_insert_value(
                insert.rows[0][source_index].value().unwrap().clone(),
                column.ty,
                &column.name,
            )
            .unwrap_or_else(|_| panic!("scalar test source {catalog_index} coerces"))
        })
        .collect::<Vec<_>>();
    assert_eq!(expected.len(), batch.columns.len());
}

#[test]
fn live_fixed_width_builder_reorders_all_scalar_vectors_and_matches_row_major_chunks() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE TABLE scalar_live (small int2, integer int4, day date, big int8, moment timestamp, amount numeric(10,2), token uuid, flag bool)",
        )
        .unwrap();
    let insert = Insert {
        table: "scalar_live".to_string(),
        columns: vec![
            "flag".to_string(),
            "token".to_string(),
            "amount".to_string(),
            "moment".to_string(),
            "big".to_string(),
            "day".to_string(),
            "integer".to_string(),
            "small".to_string(),
        ],
        rows: Insert::programmatic_rows(vec![
            vec![
                SqlValue::Bool(true),
                SqlValue::Text("00112233-4455-6677-8899-aabbccddeeff".to_string()),
                SqlValue::Int4(123),
                SqlValue::Text("2026-07-27 12:34:56".to_string()),
                SqlValue::Int4(99),
                SqlValue::Text("2026-07-27".to_string()),
                SqlValue::Int4(-44),
                SqlValue::Int4(-7),
            ],
            vec![
                SqlValue::Bool(false),
                SqlValue::Text("ffeeddcc-bbaa-9988-7766-554433221100".to_string()),
                SqlValue::Int4(-25),
                SqlValue::Text("2026-07-28 00:00:00".to_string()),
                SqlValue::Int4(-9),
                SqlValue::Text("2026-07-28".to_string()),
                SqlValue::Int4(44),
                SqlValue::Int4(7),
            ],
        ]),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    let batch = seal_typed_insert_batch_for_test(
        &Command::Insert(insert.clone()),
        &catalog,
        catalog.commit_seq,
        None,
    )
    .unwrap()
    .expect("all provided NULL-free fixed-width scalar columns are live-route eligible");
    let table = catalog.relational_catalog.get("scalar_live").unwrap();
    let row_major = insert
        .rows
        .iter()
        .map(|row| {
            table
                .columns
                .iter()
                .map(|column| {
                    let source = insert
                        .columns
                        .iter()
                        .position(|name| name == &column.name)
                        .unwrap();
                    coerce_insert_value(
                        row[source].value().unwrap().clone(),
                        column.ty,
                        &column.name,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expected = crate::engine_residency::compute_open_shard_int4_append_chunks(
        &table
            .columns
            .iter()
            .map(|column| column.ty)
            .collect::<Vec<_>>(),
        11,
        4,
        &row_major,
    )
    .unwrap();
    let source = batch
        .into_codec5_resident_append_source_for_test(&catalog)
        .unwrap();
    let chunks = source.checked_append_chunks(11, 4).unwrap().chunks;
    let actual = IntoIterator::into_iter(chunks)
        .map(|chunk| gpu_db_execution::CudaOwnedDeviceMemoryChunk {
            byte_offset: chunk.byte_offset,
            bytes: chunk.bytes.into_vec(),
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    let bool_uploads = source.fixed_bool_uploads(table).unwrap();
    assert_eq!(bool_uploads.len(), 1);
    assert_eq!(bool_uploads[0].values.as_ref(), &[1, 0]);
    assert_eq!(source.int4_min_max().unwrap().len(), 3);
}

#[test]
fn bool_bitmap_tail_and_positions_zero_and_thirty_two_encode_without_host_rows() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE bool_tail_ir (flag bool, optional int4)")
        .unwrap();
    let insert = Insert {
        table: "bool_tail_ir".to_string(),
        columns: Vec::new(),
        rows: Insert::programmatic_rows(
            (0..33)
                .map(|row| {
                    vec![
                        SqlValue::Bool(row == 0),
                        if row == 32 {
                            SqlValue::Null
                        } else {
                            SqlValue::Int4(row)
                        },
                    ]
                })
                .collect(),
        ),
        returning: Vec::new(),
    };
    let catalog = engine.catalog_snapshot();
    let batch = ready_general_insert(&insert, &catalog);
    let TypedInsertColumnValues::BoolBits(bits) = &batch.columns[0].values else {
        panic!("flag uses BoolBits");
    };
    assert_eq!(bits.as_ref(), &[1, 0], "bool tail bits are zero");
    let TypedInsertColumnValidity::Bitmap(validity) = &batch.columns[1].validity else {
        panic!("the NULL at row 32 requires a bitmap");
    };
    assert_eq!(validity.as_ref(), &[u32::MAX, 0]);
    assert!(bitmap_shape_is_exact(validity, 33));
}

#[test]
fn typed_insert_ir_source_guards_keep_vectors_sealed_and_no_row_reconstruction() {
    let source = include_str!("typed_insert_batch.rs");
    assert!(source.contains("enum TypedInsertColumnValues"));
    assert!(source.contains("Bytes16(Box<[[u8; 16]]>)"));
    assert!(source.contains("Text {"));
    assert!(source.contains("offsets: Box<[u64]>"));
    assert!(source.contains("bytes: Box<[u8]>"));
    assert!(source.contains("enum TypedInsertColumnPresence"));
    assert!(!source.contains("pub(crate) fn into_resident_append_source"));
    assert!(source.contains("into_codec5_resident_append_source_for_test"));
    assert!(include_str!("typed_insert_batch/resident_source.rs").contains("checked_dense_payload"));
    assert!(!source.contains("Vec<Vec<SqlValue>>"));
    assert!(!source.contains("impl Clone for TypedInsertBatch"));
    assert!(!source.contains("fn from_raw"));
}

#[test]
fn concurrent_insert_ingress_preserves_codec5_wal_and_recovery() {
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
