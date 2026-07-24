use super::*;
use std::sync::Arc;

fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
    gpu_db_sql::ParsedCommand::parse(sql).unwrap()
}

fn select(sql: &str) -> Select {
    match parse_command(sql).unwrap() {
        Command::Select(select) => select,
        other => panic!("expected SELECT, got {other:?}"),
    }
}

/// The hook linearizes B after A's statement snapshot exists but before A takes owner access.
/// READ COMMITTED exercises the statement refresh; REPEATABLE READ first performs a read so the
/// retained base is unambiguously older than B. In both cases the exclusive-access generation
/// fence must reject before private catalog/OID/residency publication, then a whole-transaction
/// retry against an uncontested owner must succeed and recover.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn create_unique_fences_owner_dml_between_snapshot_and_exclusive_access() {
    let engine = Arc::new(Engine::new_local());
    engine.set_shard_residency_enabled(true);
    engine.set_auto_admit_on_commit(true);

    let cases = [
        (
            "rc_insert",
            "BEGIN",
            4_300_u64,
            "index_owner_rc_insert",
            "index_owner_rc_insert_code",
            false,
        ),
        (
            "rr_insert",
            "BEGIN ISOLATION LEVEL REPEATABLE READ",
            4_320_u64,
            "index_owner_rr_insert",
            "index_owner_rr_insert_code",
            false,
        ),
        (
            "rc_delete",
            "BEGIN",
            4_340_u64,
            "index_owner_rc_delete",
            "index_owner_rc_delete_code",
            true,
        ),
        (
            "rr_delete",
            "BEGIN ISOLATION LEVEL REPEATABLE READ",
            4_360_u64,
            "index_owner_rr_delete",
            "index_owner_rr_delete_code",
            true,
        ),
    ];
    let mut committed_indexes = Vec::new();

    for (isolation, begin, txn, table, index, delete_race) in cases {
        engine
            .submit_transaction(
                txn,
                parsed(&format!(
                    "CREATE TABLE {table} (id int4 PRIMARY KEY, code int4)"
                )),
            )
            .unwrap();
        engine
            .submit_transaction(
                txn + 1,
                parsed(&if delete_race {
                    format!("INSERT INTO {table} VALUES (1, 7), (2, 7), (3, 99)")
                } else {
                    format!("INSERT INTO {table} VALUES (1, 7)")
                }),
            )
            .unwrap();
        assert!(engine.table_device_authoritative(table));
        if delete_race {
            engine
                .submit_transaction(
                    txn + 2,
                    parsed(&format!("DELETE FROM {table} WHERE id = 3")),
                )
                .unwrap();
            assert!(
                engine.read_state.residency.shards.load_full()[table]
                    .iter()
                    .any(|shard| shard.deleted_by_region.is_some()),
                "the delete-race fixture must preallocate its tombstone sidecar"
            );
        }
        let owner_txn = txn + 3;
        engine.submit_transaction(owner_txn, parsed(begin)).unwrap();
        if isolation.starts_with("rr") {
            let pinned = engine
                .execute_relational_select_in_transaction(
                    owner_txn,
                    &select(&format!("SELECT id, code FROM {table} ORDER BY id")),
                )
                .unwrap();
            assert_eq!(pinned.rows.len(), if delete_race { 2 } else { 1 });
        }

        let allocator_before = engine.catalog_snapshot().relational_next_oid;
        let reached = Arc::new(std::sync::Barrier::new(2));
        let resume = Arc::new(std::sync::Barrier::new(2));
        engine.set_index_owner_pre_acquire_hook(Arc::clone(&reached), Arc::clone(&resume));
        let creator = {
            let engine = Arc::clone(&engine);
            let sql = format!("CREATE UNIQUE INDEX {index} ON {table} (code)");
            std::thread::spawn(move || engine.submit_transaction(owner_txn, parsed(&sql)))
        };
        reached.wait();

        // B completes under shared owner access while A is paused after snapshot capture.
        let shards_before_b = engine
            .read_state
            .residency
            .shards
            .load_full()
            .get(table)
            .cloned();
        engine
            .submit_transaction(
                txn + 4,
                parsed(&if delete_race {
                    format!("DELETE FROM {table} WHERE id = 2")
                } else {
                    format!("INSERT INTO {table} VALUES (2, 7)")
                }),
            )
            .unwrap();
        let catalog_after_b = engine.catalog_snapshot();
        let wal_after_b = engine.durable_wal_records().len();
        let shards_after_b = engine
            .read_state
            .residency
            .shards
            .load_full()
            .get(table)
            .cloned();
        if delete_race {
            assert_eq!(
                shards_after_b, shards_before_b,
                "B must reuse the existing tombstone Arc without a descriptor-generation change"
            );
        }
        let cold_after_b = engine
            .read_state
            .residency
            .streaming_cold_chunks
            .load_full()
            .get(table)
            .cloned();
        let publications_after_b = engine
            .read_state
            .residency
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let coverage_after_b = engine
            .read_state
            .residency
            .named_index_coverage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let coverage_complete_after_b = engine
            .read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();

        resume.wait();
        let error = creator
            .join()
            .expect("CREATE UNIQUE worker")
            .expect_err("stale owner generation must serialize");
        assert!(matches!(error, ExecuteError::Serialization(_)), "{error}");
        assert_eq!(engine.durable_wal_records().len(), wal_after_b);
        assert_eq!(engine.catalog_snapshot().as_ref(), catalog_after_b.as_ref());
        assert_eq!(
            engine.catalog_snapshot().relational_next_oid,
            allocator_before
        );
        assert!(engine
            .relational_catalog_table(table)
            .unwrap()
            .indexes
            .iter()
            .all(|candidate| candidate.name != index));
        assert_eq!(
            engine
                .read_state
                .residency
                .shards
                .load_full()
                .get(table)
                .cloned(),
            shards_after_b
        );
        let cold_after_failure = engine
            .read_state
            .residency
            .streaming_cold_chunks
            .load_full()
            .get(table)
            .cloned();
        assert!(match (&cold_after_b, &cold_after_failure) {
            (None, None) => true,
            (Some(before), Some(after)) => Arc::ptr_eq(before, after),
            _ => false,
        });
        assert_eq!(
            *engine
                .read_state
                .residency
                .named_index_publications
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            publications_after_b
        );
        assert_eq!(
            *engine
                .read_state
                .residency
                .named_index_coverage
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            coverage_after_b
        );
        assert_eq!(
            *engine
                .read_state
                .residency
                .named_index_coverage_complete
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            coverage_complete_after_b
        );
        let failed_snapshot = engine.transaction_snapshot_handle(owner_txn).unwrap();
        let failed_delta = failed_snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(failed_delta.operations.is_empty());
        assert!(failed_delta.catalog_overlay.is_none());
        drop(failed_delta);
        drop(failed_snapshot);

        engine
            .submit_transaction(owner_txn, parsed("ROLLBACK"))
            .unwrap();
        if !delete_race {
            engine
                .submit_transaction(
                    txn + 5,
                    parsed(&format!("DELETE FROM {table} WHERE id = 2")),
                )
                .unwrap();
        }

        // Retry the entire transaction after resolving B's conflicting row.
        engine.submit_transaction(txn + 6, parsed(begin)).unwrap();
        engine
            .submit_transaction(
                txn + 6,
                parsed(&format!("CREATE UNIQUE INDEX {index} ON {table} (code)")),
            )
            .unwrap();
        engine
            .submit_transaction(txn + 6, parsed("COMMIT"))
            .unwrap();
        let committed = engine.relational_catalog_table(table).unwrap();
        let committed_index = committed
            .indexes
            .iter()
            .find(|candidate| candidate.name == index)
            .unwrap();
        committed_indexes.push((table, index, committed_index.oid));
        let duplicate = engine
            .submit_transaction(
                txn + 7,
                parsed(&format!("INSERT INTO {table} VALUES (10, 7)")),
            )
            .expect_err("committed UNIQUE index must reject the duplicate");
        assert!(
            matches!(
                duplicate,
                ExecuteError::Engine(EngineError::UniqueViolation(_))
            ),
            "{duplicate}"
        );
    }

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    for (offset, (table, index, oid)) in committed_indexes.into_iter().enumerate() {
        let recovered_table = recovered.relational_catalog_table(table).unwrap();
        assert_eq!(
            recovered_table
                .indexes
                .iter()
                .find(|candidate| candidate.name == index)
                .unwrap()
                .oid,
            oid
        );
        let duplicate = recovered
            .submit_transaction(
                4_400 + offset as u64,
                parsed(&format!("INSERT INTO {table} VALUES (99, 7)")),
            )
            .expect_err("recovery must preserve UNIQUE enforcement");
        assert!(
            matches!(
                duplicate,
                ExecuteError::Engine(EngineError::UniqueViolation(_))
            ),
            "{duplicate}"
        );
    }
}
