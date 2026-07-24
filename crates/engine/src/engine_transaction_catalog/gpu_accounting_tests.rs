use super::*;

fn assert_exact_named_index_enrollment(engine: &Engine, table: &RelationalTable) {
    assert_eq!(
        engine
            .read_state
            .residency
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&table.oid)
            .cloned(),
        Some(table.indexes.clone())
    );
    assert_eq!(
        engine
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&table.name)
            .cloned(),
        Some((table.oid, table.indexes.clone()))
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn create_table_only_reserves_its_empty_gpu_root_before_wal_and_retries() {
    let mut engine = Engine::new_local_test_engine();
    engine.set_shard_residency_enabled(true);
    engine.set_relational_residency_budget_bytes(0, 0);
    engine.submit_transaction(9_300, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            9_300,
            parsed("CREATE TABLE empty_root_budget (id int4 PRIMARY KEY, payload int4)"),
        )
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    let rejected = engine
        .submit_transaction(9_300, parsed("COMMIT"))
        .expect_err("the mandatory allocation-backed count header must be reserved before WAL");
    assert!(
        rejected.to_string().contains("canonical GPU publication"),
        "{rejected}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert!(!engine
        .catalog_snapshot()
        .relational_catalog
        .contains_key("empty_root_budget"));
    let private = engine.transaction_snapshot_handle(9_300).unwrap();
    assert!(private
        .transaction_catalog()
        .relational_catalog
        .contains_key("empty_root_budget"));
    drop(private);

    engine.set_relational_residency_budget_bytes(0, 1024 * 1024);
    engine
        .submit_transaction(9_300, parsed("COMMIT"))
        .expect("the same frozen transaction can retry after budget headroom is restored");
    let table = engine
        .relational_catalog_table("empty_root_budget")
        .unwrap();
    assert!(engine
        .zero_row_resident_generation_boundary(&table)
        .is_some());
    assert_exact_named_index_enrollment(&engine, &table);
    assert!(engine.relational_named_index_publication_required(&table));
    assert_eq!(engine.durable_wal_records().len(), wal_before + 1);

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
    let recovered_table = recovered
        .relational_catalog_table("empty_root_budget")
        .unwrap();
    assert!(recovered
        .zero_row_resident_generation_boundary(&recovered_table)
        .is_some());
    assert_exact_named_index_enrollment(&recovered, &recovered_table);
    assert!(recovered.relational_named_index_publication_required(&recovered_table));

    recovered
        .submit_transaction(
            9_301,
            parsed("INSERT INTO empty_root_budget VALUES (1, 11)"),
        )
        .expect("the first append must preserve mandatory implicit-index enrollment");
    let appended = recovered
        .relational_catalog_table("empty_root_budget")
        .unwrap();
    assert_exact_named_index_enrollment(&recovered, &appended);
    let key_id =
        crate::engine_residency::index_probe_key_id(&appended, &appended.indexes[0], 0).unwrap();
    let non_empty_shards = recovered
        .read_residency_shards()
        .get("empty_root_budget")
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|shard| shard.row_count != 0)
        .map(|shard| shard.shard_id)
        .collect::<BTreeSet<_>>();
    assert!(!non_empty_shards.is_empty());
    assert!(non_empty_shards.iter().all(|shard_id| {
        recovered
            .read_state
            .residency
            .shard_pk_device_index
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(&("empty_root_budget".to_string(), *shard_id, key_id))
    }));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn transaction_named_index_publication_orders_purges_around_its_final_cut() {
    for purge_phase in 0..3 {
        let engine = Arc::new(Engine::new_local());
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine
            .execute_text(9_305, "CREATE TABLE index_purge_cut (id int4, code int4)")
            .unwrap();
        engine
            .execute_text(9_306, "INSERT INTO index_purge_cut VALUES (1, 7)")
            .unwrap();
        engine.submit_transaction(9_307, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                9_307,
                parsed("CREATE INDEX index_purge_cut_code ON index_purge_cut (code)"),
            )
            .unwrap();

        let reached = Arc::new(std::sync::Barrier::new(2));
        let resume = Arc::new(std::sync::Barrier::new(2));
        match purge_phase {
            0 => {
                let hook_reached = Arc::clone(&reached);
                let hook_resume = Arc::clone(&resume);
                engine.set_transaction_post_durable_hook(move || {
                    hook_reached.wait();
                    hook_resume.wait();
                });
            }
            1 => engine.set_named_index_publication_post_publish_hook(
                Arc::clone(&reached),
                Arc::clone(&resume),
            ),
            2 => engine
                .read_state
                .residency
                .set_transaction_named_index_post_apply_hook(
                    Arc::clone(&reached),
                    Arc::clone(&resume),
                ),
            _ => unreachable!(),
        }
        let committer = {
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || engine.submit_transaction(9_307, parsed("COMMIT")))
        };
        reached.wait();
        engine
            .read_state
            .residency
            .purge_shard_pk_index_for_table("index_purge_cut");
        resume.wait();
        committer.join().unwrap().unwrap();

        let table = engine.relational_catalog_table("index_purge_cut").unwrap();
        let key_id =
            crate::engine_residency::index_probe_key_id(&table, &table.indexes[0], 0).unwrap();
        let cache_contains_index = || {
            engine
                .read_state
                .residency
                .shard_pk_device_index
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .keys()
                .any(|(owner, _, cached_key)| owner == "index_purge_cut" && *cached_key == key_id)
        };
        let coverage_is_complete = || {
            engine
                .read_state
                .residency
                .named_index_coverage_complete
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get("index_purge_cut")
                .cloned()
                == Some((table.oid, table.indexes.clone()))
        };
        let purge_must_win = purge_phase != 0;
        assert_eq!(cache_contains_index(), !purge_must_win);
        assert_eq!(coverage_is_complete(), !purge_must_win);
        assert!(
            engine.relational_named_index_publication_required(&table),
            "cache retirement must preserve mandatory OID enrollment"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn single_buffer_empty_generation_accepts_all_index_lifecycle_operations() {
    let mut engine = Engine::new_local();
    engine.set_shard_residency_enabled(false);
    engine.set_auto_admit_on_commit(true);
    engine.set_relational_residency_budget_bytes(0, 1024 * 1024);
    engine
        .execute_text(
            9_310,
            "CREATE TABLE empty_single_buffer_index (id int4, code int4)",
        )
        .unwrap();
    let table = engine
        .relational_catalog_table("empty_single_buffer_index")
        .unwrap();
    assert!(engine
        .read_residency_shards()
        .get("empty_single_buffer_index")
        .is_none_or(Vec::is_empty));
    assert!(engine
        .zero_row_resident_generation_boundary(&table)
        .is_some());

    for (txn_id, sql) in [
        (
            9_311,
            "CREATE INDEX empty_single_buffer_code_idx \
             ON empty_single_buffer_index (code)",
        ),
        (
            9_312,
            "ALTER INDEX empty_single_buffer_code_idx \
             RENAME TO empty_single_buffer_code_renamed",
        ),
        (9_313, "DROP INDEX empty_single_buffer_code_renamed"),
    ] {
        engine.submit_transaction(txn_id, parsed("BEGIN")).unwrap();
        engine.submit_transaction(txn_id, parsed(sql)).unwrap();
        engine.submit_transaction(txn_id, parsed("COMMIT")).unwrap();
        let table = engine
            .relational_catalog_table("empty_single_buffer_index")
            .unwrap();
        assert!(engine
            .zero_row_resident_generation_boundary(&table)
            .is_some());
        assert!(engine
            .read_residency_shards()
            .get("empty_single_buffer_index")
            .is_none_or(Vec::is_empty));
    }
    assert!(engine
        .relational_catalog_table("empty_single_buffer_index")
        .unwrap()
        .indexes
        .is_empty());

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

fn rollover_drop_fixture() -> Engine {
    let mut engine = Engine::new_local_test_engine();
    engine.set_shard_size_target(4);
    engine.set_shard_residency_enabled(true);
    engine.set_relational_residency_budget_bytes(0, 1 << 30);
    engine
        .execute_text(
            9_320,
            "CREATE TABLE rollover_drop_owner (id int4, a int4, b int4)",
        )
        .unwrap();
    engine
        .execute_text(
            9_321,
            "CREATE INDEX rollover_drop_a_idx ON rollover_drop_owner (a)",
        )
        .unwrap();
    engine
        .execute_text(
            9_322,
            "CREATE INDEX rollover_drop_b_idx ON rollover_drop_owner (b)",
        )
        .unwrap();
    let values = (1..=16)
        .map(|id| format!("({id}, {}, {})", id * 10, id * 100))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            9_323,
            &format!("INSERT INTO rollover_drop_owner VALUES {values}"),
        )
        .unwrap();
    engine
        .populate_relational_residency_snapshot("rollover_drop_owner")
        .unwrap();
    engine
        .publish_relational_resident_indexes("rollover_drop_owner")
        .unwrap();
    engine.set_auto_admit_on_commit(true);
    let fill = (17..=32)
        .map(|id| format!("({id}, {}, {})", id * 10, id * 100))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            9_324,
            &format!("INSERT INTO rollover_drop_owner VALUES {fill}"),
        )
        .unwrap();
    let shards = engine.read_residency_shards();
    let open = shards["rollover_drop_owner"]
        .iter()
        .rfind(|shard| shard.row_count != 0)
        .unwrap();
    assert_eq!(
        open.row_count, open.capacity,
        "fixture must force the ordered INSERT through rollover"
    );
    drop(shards);
    engine
}

fn allocated_index_bytes(engine: &Engine, names: &[&str]) -> u64 {
    let table = engine
        .relational_catalog_table("rollover_drop_owner")
        .unwrap();
    let keys = table
        .indexes
        .iter()
        .enumerate()
        .filter(|(_, index)| names.contains(&index.name.as_str()))
        .map(|(ordinal, index)| {
            crate::engine_residency::index_probe_key_id(&table, index, ordinal).unwrap()
        })
        .collect::<BTreeSet<_>>();
    let cache = engine
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let allocations = cache
        .iter()
        .filter(|((table, _, key), _)| table == "rollover_drop_owner" && keys.contains(key))
        .filter_map(|(_, entry)| entry.device_index.as_ref())
        .map(|memory| (memory.device_ptr(), memory.metadata().allocated_bytes))
        .collect::<BTreeMap<_, _>>();
    let bytes = allocations.values().sum();
    assert!(bytes > 0, "dropped index fixture must own physical keys");
    bytes
}

fn newest_shard_allocation_bytes(engine: &Engine) -> u64 {
    let shards = engine.read_residency_shards();
    let shard = shards["rollover_drop_owner"]
        .iter()
        .max_by_key(|shard| shard.shard_id)
        .unwrap();
    assert_eq!(
        shard.row_count, 1,
        "the committed rollover must own one row"
    );
    let mut allocations = [
        shard.device_memory.as_ref(),
        shard.deleted_by_region.as_ref(),
        shard.created_by_region.as_ref(),
        shard.row_id_region.as_ref(),
    ]
    .into_iter()
    .flatten()
    .map(|memory| (memory.device_ptr(), memory.metadata().allocated_bytes))
    .collect::<BTreeMap<_, _>>();
    let cache = engine
        .read_state
        .residency
        .shard_pk_device_index
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for memory in cache
        .iter()
        .filter(|((table, shard_id, _), _)| {
            table == "rollover_drop_owner" && *shard_id == shard.shard_id
        })
        .filter_map(|(_, entry)| entry.device_index.as_ref())
    {
        allocations.insert(memory.device_ptr(), memory.metadata().allocated_bytes);
    }
    allocations.values().sum()
}

fn stage_insert_then_drop(engine: &Engine, txn_id: TxnId, drop_names: &[&str]) {
    engine.submit_transaction(txn_id, parsed("BEGIN")).unwrap();
    engine
        .submit_transaction(
            txn_id,
            parsed("INSERT INTO rollover_drop_owner VALUES (33, 330, 3300)"),
        )
        .unwrap();
    engine
        .submit_transaction(
            txn_id,
            parsed(&format!("DROP INDEX {}", drop_names.join(", "))),
        )
        .unwrap();
}

fn commit_insert_then_drop(engine: &Engine, txn_id: TxnId, drop_names: &[&str]) {
    stage_insert_then_drop(engine, txn_id, drop_names);
    engine.submit_transaction(txn_id, parsed("COMMIT")).unwrap();
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn rollover_insert_uses_final_catalog_after_drop_subset_or_all_at_exact_budget() {
    for (case, drop_names) in [
        (0u64, &["rollover_drop_a_idx"][..]),
        (1u64, &["rollover_drop_a_idx", "rollover_drop_b_idx"][..]),
    ] {
        // An unconstrained twin measures the exact final rollover allocation. Add back the
        // allocations retired by DROP as a non-vacuity check, then sum the new shard's exact
        // allocation-backed payload/identity/final-index set.
        let control = rollover_drop_fixture();
        let retired = allocated_index_bytes(&control, drop_names);
        commit_insert_then_drop(&control, 9_330 + case, drop_names);
        assert!(retired > 0);
        let exact_rollover = newest_shard_allocation_bytes(&control);
        assert!(exact_rollover > 0);

        let mut tight = rollover_drop_fixture();
        let wal_before = tight.durable_wal_records().len();
        stage_insert_then_drop(&tight, 9_340 + case, drop_names);
        let snapshot = tight.transaction_snapshot_handle(9_340 + case).unwrap();
        let private_bytes = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .private_gpu_bytes_by_gpu
            .get(&0)
            .copied()
            .unwrap_or(0);
        assert!(
            private_bytes > 0,
            "ordered INSERT must retain a transaction-private GPU generation"
        );
        let staged_bytes = tight.relational_resident_bytes_for_gpu(0);
        tight.set_relational_residency_budget_bytes(
            0,
            staged_bytes.checked_add(exact_rollover).unwrap(),
        );
        drop(snapshot);
        tight
            .submit_transaction(9_340 + case, parsed("COMMIT"))
            .unwrap();
        assert!(!tight.is_commit_path_poisoned());
        assert_eq!(tight.durable_wal_records().len(), wal_before + 1);
        let final_table = tight
            .relational_catalog_table("rollover_drop_owner")
            .unwrap();
        for name in drop_names {
            assert!(final_table.indexes.iter().all(|index| index.name != *name));
        }
        let rows = tight
            .execute_relational_select(&select("SELECT id FROM rollover_drop_owner WHERE id = 33"))
            .unwrap();
        assert!(matches!(rows.executed_target, DeviceTarget::Gpu(_)));
        assert_eq!(rows.rows, vec![vec![SqlValue::Int4(33)]]);

        let recovered = Engine::recover_from_durable_wal(&tight.durable_wal_records()).unwrap();
        assert!(recovered
            .catalog_snapshot()
            .same_contents(tight.catalog_snapshot().as_ref()));
        assert_eq!(
            recovered
                .execute_relational_select_text("SELECT id FROM rollover_drop_owner WHERE id = 33")
                .unwrap()
                .rows,
            vec![vec![SqlValue::Int4(33)]]
        );
    }
}
