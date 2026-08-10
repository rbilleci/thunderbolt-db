use super::*;
use gpu_db_wal::WalBuffer;

struct WideShape {
    table: &'static str,
    key_type: &'static str,
    sentinel: &'static str,
    fresh: &'static str,
}

fn warm_key(shape: &WideShape, i: u32) -> String {
    match shape.table {
        "wide_i8" => format!("{}", 100_000_000_000_u64 + u64::from(i)),
        "wide_ts" => format!("'2025-01-01 00:{:02}:{:02}'", (i / 60) % 60, i % 60),
        "wide_num" => format!("{}.25", 1_000_000_u64 + u64::from(i)),
        "wide_uuid" => format!("'00000000-0000-0000-0001-{i:012x}'"),
        "wide_text" => format!("'warm-key-{i}'"),
        other => panic!("unknown wide shape {other}"),
    }
}

/// Construct two distinct positive BIGINT values with the same canonical fingerprint. BIGINT folds
/// to `[low32, high32]`; the fingerprint's per-word round is a bijection in the second word, so two
/// first-word states with the same top 12 bits can be completed by XORing their state delta into the
/// second word. The production fold below verifies the construction and makes any algorithm drift
/// fail loudly.
fn colliding_bigints() -> (i64, i64) {
    let step = |h: u32, word: i32| {
        let h = (h ^ word as u32).wrapping_mul(0x0100_0193);
        h.rotate_left(13).wrapping_add(0x9E37_79B1)
    };
    let mut buckets = std::collections::HashMap::<u32, (u32, u32)>::new();
    let mut pair = None;
    for low in 10_000_u32..2_000_000 {
        let state = step(0x811C_9DC5, low as i32);
        if let Some((previous_low, previous_state)) = buckets.insert(state >> 20, (low, state)) {
            let high_a = 1_u32 << 20;
            let high_b = high_a ^ (previous_state ^ state);
            let a = ((u64::from(high_a) << 32) | u64::from(previous_low)) as i64;
            let b = ((u64::from(high_b) << 32) | u64::from(low)) as i64;
            pair = Some((a, b));
            break;
        }
    }
    let (a, b) = pair.expect("construct a BIGINT fingerprint collision");
    let fingerprint = |value| {
        let words =
            crate::engine_residency::sql_value_key_words(SqlType::Int8, &SqlValue::Int8(value))
                .unwrap();
        crate::engine_residency::compound_key_fingerprint(&words)
    };
    assert_ne!(a, b);
    assert_eq!(fingerprint(a), fingerprint(b));
    (a, b)
}

/// R3-002/R3-003 unique conflict-history retirement: a writer prepares an absent TEXT unique key,
/// then another writer claims and moves away from that key before the first commits. Current-row
/// validation alone would see the key as free; the exact device scan must observe the tombstoned
/// physical version's newer stamp and serialize the stale writer without consulting the CPU slot
/// ledger.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_text_unique_key_away_history_conflicts_from_device_stamp() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let mut engine = Engine::new_local_test_engine();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);
    engine
        .execute_text(
            1,
            "CREATE TABLE history_t (id INT PRIMARY KEY, u TEXT UNIQUE, v INT)",
        )
        .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("history_t")
        .expect("populate history table residency");
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    let engine = std::sync::Arc::new(engine);
    let txn_ids = AtomicU64::new(10);
    for i in 0..512_i32 {
        engine
            .execute_dml_concurrent(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                &format!("INSERT INTO history_t VALUES ({i}, 'warm-{i}', {i})"),
            )
            .unwrap();
        if engine.table_device_authoritative("history_t") {
            break;
        }
    }
    assert!(engine.table_device_authoritative("history_t"));

    let (prepared_tx, prepared_rx) = std::sync::mpsc::channel();
    let (continue_tx, continue_rx) = std::sync::mpsc::channel();
    let stale_writer = {
        let engine = std::sync::Arc::clone(&engine);
        std::thread::spawn(move || {
            engine.execute_dml_concurrent_instrumented(
                50_000,
                "INSERT INTO history_t VALUES (9000, 'claim-then-release', 1)",
                move || {
                    prepared_tx.send(()).unwrap();
                    continue_rx.recv().unwrap();
                },
            )
        })
    };
    prepared_rx.recv().unwrap();
    engine
        .execute_dml_concurrent(
            50_001,
            "INSERT INTO history_t VALUES (9001, 'claim-then-release', 2)",
        )
        .unwrap();
    engine
        .execute_dml_concurrent(
            50_002,
            "UPDATE history_t SET u = 'moved-away' WHERE id = 9001",
        )
        .unwrap();
    // Scope non-vacuity to the stale writer itself. The intervening INSERT/UPDATE also run device
    // validation, so sampling before them would let the host-neutral parity ledger obscure a missing
    // stale-writer device verdict while this counter still advanced.
    assert!(
        engine.table_device_authoritative("history_t"),
        "the stale writer must resume while the relation is still device-authoritative"
    );
    let stale_validate_before = engine.dml_device_validate_hits();
    continue_tx.send(()).unwrap();
    let error = stale_writer
        .join()
        .unwrap()
        .expect_err("post-snapshot key history must serialize the stale writer")
        .to_string();
    assert!(error.contains("write-write conflict"), "{error}");
    assert!(
        engine.dml_device_validate_hits() > stale_validate_before,
        "the stale writer itself must run the exact device conflict-history scan"
    );
    let rows = engine
        .execute_relational_select_text("SELECT id, u FROM history_t WHERE id >= 9000 ORDER BY id")
        .unwrap()
        .rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], SqlValue::Int4(9001));
    assert_eq!(rows[0][1], SqlValue::Text("moved-away".to_string()));
}

/// R3-002 single-column wider/text index graduation. INT8, TIMESTAMP, NUMERIC, UUID, and TEXT
/// primary keys all enter host-install elision and use the flagged device fingerprint index for
/// uniqueness plus UPDATE/DELETE locate. Every candidate is materialized and compared as the
/// original typed value, so a deliberately colliding pair is accepted while a true duplicate is
/// rejected. NULL payloads cross insert/update/read/recovery, satisfying the device-slice NULL
/// differential without weakening PRIMARY KEY nullability. The durable replay count is the final
/// parity oracle.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_single_wide_unique_indexes_elide_validate_collisions_and_recover() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let dir = std::env::temp_dir().join(format!(
        "gpu-db-r3-wide-index-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let wal_path = dir.join("db.wal");
    let shapes = [
        WideShape {
            table: "wide_i8",
            key_type: "BIGINT",
            sentinel: "4294967301",
            fresh: "8589934593",
        },
        WideShape {
            table: "wide_ts",
            key_type: "TIMESTAMP",
            sentinel: "'2024-06-01 09:00:00'",
            fresh: "'2024-06-01 10:00:00'",
        },
        WideShape {
            table: "wide_num",
            key_type: "NUMERIC(12,2)",
            sentinel: "10.25",
            fresh: "20.50",
        },
        WideShape {
            table: "wide_uuid",
            key_type: "UUID",
            sentinel: "'ffffffff-0000-0000-0000-000000000001'",
            fresh: "'00000000-0000-0000-0000-000000000007'",
        },
        WideShape {
            table: "wide_text",
            key_type: "TEXT",
            sentinel: "'sentinel-key'",
            fresh: "'fresh-key'",
        },
    ];
    let mut expected_counts = std::collections::BTreeMap::<String, i64>::new();
    {
        let mut engine = Engine::new_local_test_engine();
        engine.commit_state_mut().wal = WalBuffer::with_durable_segment(&wal_path);
        engine.set_auto_admit_on_commit(true);
        engine.set_device_write_locate_wave_batch_enabled(true);
        let txn_ids = AtomicU64::new(1);
        macro_rules! sql {
            ($statement:expr) => {
                engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $statement)
            };
        }

        for shape in &shapes {
            engine
                .execute_text(
                    txn_ids.fetch_add(1, Ordering::Relaxed),
                    &format!(
                        "CREATE TABLE {} (k {} PRIMARY KEY, marker INT, nullable_v INT)",
                        shape.table, shape.key_type
                    ),
                )
                .unwrap();
            sql!(&format!(
                "INSERT INTO {} VALUES ({}, 0, NULL)",
                shape.table, shape.sentinel
            ))
            .unwrap();
            let snapshot = engine
                .populate_relational_residency_snapshot(shape.table)
                .expect("populate wide-key residency");
            if snapshot.device_memory_proof.is_none() {
                return;
            }

            let mut warm_rows = 0_i64;
            for i in 0..512_u32 {
                sql!(&format!(
                    "INSERT INTO {} VALUES ({}, {}, NULL)",
                    shape.table,
                    warm_key(shape, i),
                    i
                ))
                .unwrap();
                warm_rows += 1;
                if engine.table_device_authoritative(shape.table) {
                    break;
                }
            }
            assert!(
                engine.table_device_authoritative(shape.table),
                "{} never entered elision",
                shape.table
            );

            let locate_before = engine.device_write_locate_hits();
            let validate_before = engine.dml_device_validate_hits();
            sql!(&format!(
                "INSERT INTO {} VALUES ({}, 10, NULL)",
                shape.table, shape.fresh
            ))
            .unwrap();
            let duplicate = sql!(&format!(
                "INSERT INTO {} VALUES ({}, 99, NULL)",
                shape.table, shape.fresh
            ))
            .unwrap_err()
            .to_string();
            assert!(duplicate.contains("duplicate key value"), "{duplicate}");
            let rebuilt_duplicate = sql!(&format!(
                "INSERT INTO {} VALUES ({}, 98, NULL)",
                shape.table, shape.sentinel
            ))
            .unwrap_err()
            .to_string();
            assert!(
                rebuilt_duplicate.contains("duplicate key value"),
                "{} device rebuild fold must match its host needle: {rebuilt_duplicate}",
                shape.table
            );
            assert!(engine.device_write_locate_hits() > locate_before);
            assert!(engine.dml_device_validate_hits() > validate_before);

            let resolve_before = engine.dml_device_resolve_hits();
            sql!(&format!(
                "UPDATE {} SET marker = 11, nullable_v = NULL WHERE k = {}",
                shape.table, shape.fresh
            ))
            .unwrap();
            sql!(&format!(
                "DELETE FROM {} WHERE k = {}",
                shape.table, shape.fresh
            ))
            .unwrap();
            sql!(&format!(
                "INSERT INTO {} VALUES ({}, 987654, NULL)",
                shape.table, shape.fresh
            ))
            .unwrap();
            assert!(engine.dml_device_resolve_hits() > resolve_before);
            assert!(
                engine.table_device_authoritative(shape.table),
                "{} wide-key writes must remain device-authoritative",
                shape.table
            );

            let Command::Select(select) = parse_command(&format!(
                "SELECT marker, nullable_v FROM {} WHERE marker = 987654",
                shape.table
            ))
            .unwrap() else {
                unreachable!()
            };
            assert_eq!(
                engine.execute_relational_select(&select).unwrap().rows,
                vec![vec![SqlValue::Int4(987654), SqlValue::Null]],
                "{} mixed read/write result",
                shape.table
            );
            expected_counts.insert(shape.table.to_string(), warm_rows + 2);
        }

        // No-i32-column append: the fingerprint tail count must come from `new_rows`, not the
        // absent i32-section vectors. This keeps a pure TEXT-PK table's cached index incremental.
        engine
            .execute_text(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                "CREATE TABLE wide_only (k TEXT PRIMARY KEY)",
            )
            .unwrap();
        sql!("INSERT INTO wide_only VALUES ('only-seed')").unwrap();
        let snapshot = engine
            .populate_relational_residency_snapshot("wide_only")
            .expect("populate no-i32 wide-key residency");
        assert!(snapshot.device_memory_proof.is_some());
        let mut only_warm = 0_i64;
        for i in 0..512_u32 {
            sql!(&format!("INSERT INTO wide_only VALUES ('only-warm-{i}')")).unwrap();
            only_warm += 1;
            if engine.table_device_authoritative("wide_only") {
                break;
            }
        }
        assert!(engine.table_device_authoritative("wide_only"));
        let only_hits = engine.device_write_locate_hits();
        sql!("INSERT INTO wide_only VALUES ('only-next')").unwrap();
        let duplicate = sql!("INSERT INTO wide_only VALUES ('only-next')")
            .unwrap_err()
            .to_string();
        assert!(duplicate.contains("duplicate key value"), "{duplicate}");
        assert!(engine.device_write_locate_hits() > only_hits);
        assert!(engine.table_device_authoritative("wide_only"));
        expected_counts.insert("wide_only".to_string(), only_warm + 2);

        // Adversarial collision: both distinct BIGINT keys must coexist in the same fingerprint
        // chain; true duplicate detection must still materialize and identify the exact key.
        let (collision_a, collision_b) = colliding_bigints();
        sql!(&format!(
            "INSERT INTO wide_i8 VALUES ({collision_a}, 70, NULL)"
        ))
        .unwrap();
        sql!(&format!(
            "INSERT INTO wide_i8 VALUES ({collision_b}, 71, NULL)"
        ))
        .unwrap_or_else(|err| panic!("distinct colliding BIGINT key rejected: {err:?}"));
        let duplicate = sql!(&format!(
            "INSERT INTO wide_i8 VALUES ({collision_a}, 72, NULL)"
        ))
        .unwrap_err()
        .to_string();
        assert!(duplicate.contains("duplicate key value"), "{duplicate}");
        assert!(engine.table_device_authoritative("wide_i8"));
        *expected_counts.get_mut("wide_i8").unwrap() += 2;

        // Explicit transaction arbitration binds the BEGIN generation for staging, then checks the
        // current flagged BIGINT index at COMMIT. The private duplicate stages successfully, but the
        // intervening committed row wins and COMMIT reports the device unique conflict.
        const EXPLICIT_TXN: u64 = 9_000_000;
        const INTERVENING_TXN: u64 = EXPLICIT_TXN + 1;
        engine.execute_text(EXPLICIT_TXN, "BEGIN").unwrap();
        engine
            .execute_text(
                EXPLICIT_TXN,
                "INSERT INTO wide_i8 VALUES (777777777777, 80, NULL)",
            )
            .unwrap();
        engine
            .execute_dml_concurrent(
                INTERVENING_TXN,
                "INSERT INTO wide_i8 VALUES (777777777777, 81, NULL)",
            )
            .unwrap();
        let conflict = engine.execute_text(EXPLICIT_TXN, "COMMIT").unwrap_err();
        assert!(
            matches!(&conflict, ExecuteError::Serialization(message) if message.contains("device unique conflict")),
            "wide-key COMMIT must arbitrate against the current device index: {conflict:?}"
        );
        engine.execute_text(EXPLICIT_TXN, "ROLLBACK").unwrap();
        *expected_counts.get_mut("wide_i8").unwrap() += 1;
    }

    let recovered = Engine::open_durable_wal_segment(&wal_path).expect("wide-key WAL recovery");
    for (table, expected) in expected_counts {
        let Command::Select(select) =
            parse_command(&format!("SELECT COUNT(*) FROM {table}")).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            recovered.execute_relational_select(&select).unwrap().rows,
            vec![vec![SqlValue::Int8(expected)]],
            "{table} recovery parity"
        );
        if table != "wide_only" {
            let Command::Select(null_row) = parse_command(&format!(
                "SELECT nullable_v FROM {table} WHERE marker = 987654"
            ))
            .unwrap() else {
                unreachable!()
            };
            assert_eq!(
                recovered.execute_relational_select(&null_row).unwrap().rows,
                vec![vec![SqlValue::Null]],
                "{table} NULL recovery parity"
            );
        }
    }
    drop(recovered);
    let _ = std::fs::remove_dir_all(dir);
}

/// R3-002 BOOL key graduation: the index rebuild extracts 0/1 directly from the resident bit-packed
/// bitmap, while host needles use the same canonical word. The PRIMARY KEY path proves true/false
/// uniqueness and device DML locate. The nullable UNIQUE path proves PostgreSQL NULL-distinct
/// semantics without rehydrating to a host value-index probe. Recovery preserves both accepted
/// histories.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_bool_unique_index_elides_validates_null_and_recovers() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let dir = std::env::temp_dir().join(format!(
        "gpu-db-r3-bool-index-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let wal_path = dir.join("db.wal");
    {
        let mut engine = Engine::new_local_test_engine();
        engine.commit_state_mut().wal = WalBuffer::with_durable_segment(&wal_path);
        engine.set_auto_admit_on_commit(true);
        engine.set_device_write_locate_wave_batch_enabled(true);
        let txn_ids = AtomicU64::new(1);
        macro_rules! sql {
            ($statement:expr) => {
                engine.execute_dml_concurrent(txn_ids.fetch_add(1, Ordering::Relaxed), $statement)
            };
        }

        engine
            .execute_text(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                "CREATE TABLE bool_pk (k BOOL PRIMARY KEY, marker INT, nullable_v INT)",
            )
            .unwrap();
        sql!("INSERT INTO bool_pk VALUES (false, 0, NULL)").unwrap();
        let snapshot = engine
            .populate_relational_residency_snapshot("bool_pk")
            .expect("populate bool PK residency");
        if snapshot.device_memory_proof.is_none() {
            return;
        }
        let locate_before = engine.device_write_locate_hits();
        sql!("INSERT INTO bool_pk VALUES (true, 1, NULL)").unwrap();
        assert!(engine.table_device_authoritative("bool_pk"));
        let duplicate = sql!("INSERT INTO bool_pk VALUES (false, 9, NULL)")
            .unwrap_err()
            .to_string();
        assert!(duplicate.contains("duplicate key value"), "{duplicate}");
        assert!(engine.device_write_locate_hits() > locate_before);
        let resolve_before = engine.dml_device_resolve_hits();
        sql!("UPDATE bool_pk SET marker = 7 WHERE k = true").unwrap();
        sql!("DELETE FROM bool_pk WHERE k = false").unwrap();
        sql!("INSERT INTO bool_pk VALUES (false, 8, NULL)").unwrap();
        assert!(engine.dml_device_resolve_hits() > resolve_before);
        assert!(engine.table_device_authoritative("bool_pk"));

        engine
            .execute_text(
                txn_ids.fetch_add(1, Ordering::Relaxed),
                "CREATE TABLE bool_nullable (id INT PRIMARY KEY, k BOOL UNIQUE, marker INT)",
            )
            .unwrap();
        sql!("INSERT INTO bool_nullable VALUES (1, false, 10)").unwrap();
        let snapshot = engine
            .populate_relational_residency_snapshot("bool_nullable")
            .expect("populate nullable bool UNIQUE residency");
        assert!(snapshot.device_memory_proof.is_some());
        sql!("INSERT INTO bool_nullable VALUES (2, true, 20)").unwrap();
        assert!(engine.table_device_authoritative("bool_nullable"));
        let nonnull_validate_before = engine.dml_device_validate_hits();
        let duplicate = sql!("INSERT INTO bool_nullable VALUES (9, false, 90)")
            .expect_err("ordinary non-NULL BOOL uniqueness remains enforced")
            .to_string();
        assert!(duplicate.contains("duplicate key value"), "{duplicate}");
        assert!(engine.dml_device_validate_hits() > nonnull_validate_before);
        assert!(engine.table_device_authoritative("bool_nullable"));
        let validate_before = engine.dml_device_validate_hits();
        sql!("INSERT INTO bool_nullable VALUES (3, NULL, 30)").unwrap();
        assert!(
            engine.table_device_authoritative("bool_nullable"),
            "the first NULL must be decided by the device validity scan"
        );
        let catalog = engine.catalog_snapshot();
        let table = catalog.relational_catalog.get("bool_nullable").unwrap();
        assert_eq!(
            engine.device_visible_row_with_value(
                table,
                StorageVisibility {
                    read_txn_id: engine.committed_seq(),
                },
                1,
                &SqlValue::Null,
                None,
            ),
            Some(true),
            "the resident IS NULL scan must authoritatively see the first NULL"
        );
        assert_eq!(
            engine.device_visible_row_with_value(
                table,
                StorageVisibility {
                    read_txn_id: engine.committed_seq(),
                },
                0,
                &SqlValue::Int4(4),
                None,
            ),
            Some(false),
            "the untouched raw INT4 PK index must authoritatively miss id=4"
        );
        let Command::Insert(null_duplicate) =
            parse_command("INSERT INTO bool_nullable VALUES (4, NULL, 40)").unwrap()
        else {
            unreachable!()
        };
        let prepared = engine.prepare_insert(
            &null_duplicate,
            engine.dml_read_snapshot(engine.committed_seq()),
            None,
        );
        assert!(
            prepared.is_ok(),
            "a second NULL remains distinct for PostgreSQL UNIQUE semantics"
        );
        assert!(
            engine.table_device_authoritative("bool_nullable"),
            "non-deferrable full validation must stay device-native"
        );
        sql!("INSERT INTO bool_nullable VALUES (4, NULL, 40)").unwrap();
        assert!(engine.dml_device_validate_hits() > validate_before);
        assert!(
            engine.table_device_authoritative("bool_nullable"),
            "the second NULL must not rehydrate"
        );

        let Command::Select(select) =
            parse_command("SELECT marker FROM bool_nullable WHERE id = 3").unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            engine.execute_relational_select(&select).unwrap().rows,
            vec![vec![SqlValue::Int4(30)]]
        );
    }

    let recovered = Engine::open_durable_wal_segment(&wal_path).expect("bool-key WAL recovery");
    for (table, expected) in [("bool_pk", 2_i64), ("bool_nullable", 4_i64)] {
        let Command::Select(select) =
            parse_command(&format!("SELECT COUNT(*) FROM {table}")).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            recovered.execute_relational_select(&select).unwrap().rows,
            vec![vec![SqlValue::Int8(expected)]],
            "{table} recovery parity"
        );
    }
    let Command::Select(null_row) =
        parse_command("SELECT marker FROM bool_nullable WHERE id = 3").unwrap()
    else {
        unreachable!()
    };
    assert_eq!(
        recovered.execute_relational_select(&null_row).unwrap().rows,
        vec![vec![SqlValue::Int4(30)]]
    );
    drop(recovered);
    let _ = std::fs::remove_dir_all(dir);
}

/// R3-002 decline ladder: a compound key containing NULL has no complete fingerprint and is distinct
/// from every other PostgreSQL UNIQUE tuple. The device-authoritative table must remain elided while
/// accepting repeated partial-NULL tuples and preserving ordinary non-NULL uniqueness.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_compound_partial_null_unique_scans_exact_tuple_on_device() {
    let mut engine = Engine::new_local();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    engine
        .execute_text(
            1,
            "CREATE TABLE nullable_tuple (id INT PRIMARY KEY, a INT, b TEXT, UNIQUE (a, b))",
        )
        .unwrap();
    engine
        .execute_dml_concurrent(2, "INSERT INTO nullable_tuple VALUES (1, 0, 'warm')")
        .unwrap();
    let snapshot = engine
        .populate_relational_residency_snapshot("nullable_tuple")
        .expect("populate compound nullable residency");
    if snapshot.device_memory_proof.is_none() {
        return;
    }
    engine
        .execute_dml_concurrent(3, "INSERT INTO nullable_tuple VALUES (2, 1, NULL)")
        .unwrap();
    assert!(engine.table_device_authoritative("nullable_tuple"));

    let catalog = engine.catalog_snapshot();
    let table = catalog.relational_catalog.get("nullable_tuple").unwrap();
    let (ordinal, unique) = table
        .indexes
        .iter()
        .enumerate()
        .find(|(_, index)| !index.primary_key)
        .unwrap();
    let key_id = crate::engine_residency::index_probe_key_id(table, unique, ordinal).unwrap();
    let key_cols = [(1, SqlValue::Int4(1)), (2, SqlValue::Null)];
    let validate_before = engine.dml_device_validate_hits();
    assert!(engine
        .visible_row_with_tuple(
            table,
            StorageVisibility {
                read_txn_id: engine.committed_seq(),
            },
            key_id,
            None,
            &key_cols,
            None,
        )
        .unwrap());
    assert!(engine.dml_device_validate_hits() > validate_before);
    assert!(engine.table_device_authoritative("nullable_tuple"));

    engine
        .execute_dml_concurrent(4, "INSERT INTO nullable_tuple VALUES (3, 1, NULL)")
        .expect("a repeated partial-NULL UNIQUE tuple remains distinct");
    assert!(engine.table_device_authoritative("nullable_tuple"));
    engine
        .execute_dml_concurrent(5, "INSERT INTO nullable_tuple VALUES (4, 2, NULL)")
        .unwrap();
    let duplicate = engine
        .execute_dml_concurrent(6, "INSERT INTO nullable_tuple VALUES (7, 0, 'warm')")
        .expect_err("ordinary non-NULL compound uniqueness remains enforced")
        .to_string();
    assert!(duplicate.contains("duplicate key value"), "{duplicate}");

    // Partial-NULL claim/release history is never a unique-key conflict under PostgreSQL semantics,
    // even when another transaction inserts and removes the same visible non-NULL members.
    const STALE_TXN: u64 = 90;
    engine.execute_text(STALE_TXN, "BEGIN").unwrap();
    engine
        .execute_dml_concurrent(STALE_TXN, "INSERT INTO nullable_tuple VALUES (5, 3, NULL)")
        .unwrap();
    engine
        .execute_dml_concurrent(91, "INSERT INTO nullable_tuple VALUES (6, 3, NULL)")
        .unwrap();
    engine
        .execute_dml_concurrent(92, "DELETE FROM nullable_tuple WHERE id = 6")
        .unwrap();
    assert!(
        engine.table_device_authoritative("nullable_tuple"),
        "the partial-NULL stale commit must remain device-authoritative"
    );
    engine
        .execute_text(STALE_TXN, "COMMIT")
        .expect("partial-NULL history cannot manufacture a unique conflict");
    assert!(engine.table_device_authoritative("nullable_tuple"));
}

/// R3-002 nullable mutation coverage: structural NULL equality is device-native not only for a
/// wider compound fingerprint, but also for an all-i32 compound and every raw i32-section unique
/// key type. UPDATE must tombstone+append without rehydrating; DELETE must tombstone without
/// rehydrating. Continued elision is the non-vacuity proof because any locate decline must fail closed before
/// the mutation is acknowledged.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_nullable_i32_unique_mutations_remain_device_authoritative() {
    let mut engine = Engine::new_local();
    engine.set_auto_admit_on_commit(true);
    engine.set_device_write_locate_wave_batch_enabled(true);

    let cases = [
        (
            "nullable_compound_i32",
            "CREATE TABLE nullable_compound_i32 (id INT PRIMARY KEY, a INT, b INT, v INT, UNIQUE (a, b))",
            "INSERT INTO nullable_compound_i32 VALUES (1, 1, 1, 10)",
            "INSERT INTO nullable_compound_i32 VALUES (2, 2, NULL, 20)",
        ),
        (
            "nullable_raw_i32",
            "CREATE TABLE nullable_raw_i32 (id INT PRIMARY KEY, k INT UNIQUE, v INT)",
            "INSERT INTO nullable_raw_i32 VALUES (1, 1, 10)",
            "INSERT INTO nullable_raw_i32 VALUES (2, NULL, 20)",
        ),
        (
            "nullable_raw_int2",
            "CREATE TABLE nullable_raw_int2 (id INT PRIMARY KEY, k SMALLINT UNIQUE, v INT)",
            "INSERT INTO nullable_raw_int2 VALUES (1, 1, 10)",
            "INSERT INTO nullable_raw_int2 VALUES (2, NULL, 20)",
        ),
        (
            "nullable_raw_date",
            "CREATE TABLE nullable_raw_date (id INT PRIMARY KEY, k DATE UNIQUE, v INT)",
            "INSERT INTO nullable_raw_date VALUES (1, '2026-01-01', 10)",
            "INSERT INTO nullable_raw_date VALUES (2, NULL, 20)",
        ),
    ];
    let mut seq = 1u64;
    for (table, ddl, warm, insert_null) in cases {
        engine.execute_text(seq, ddl).unwrap();
        seq += 1;
        engine.execute_dml_concurrent(seq, warm).unwrap();
        seq += 1;
        let snapshot = engine
            .populate_relational_residency_snapshot(table)
            .expect("populate nullable unique residency");
        if snapshot.device_memory_proof.is_none() {
            return;
        }
        engine.execute_dml_concurrent(seq, insert_null).unwrap();
        seq += 1;
        assert!(
            engine.table_device_authoritative(table),
            "{table}: nullable insert must enter device authority"
        );

        engine
            .execute_dml_concurrent(seq, &format!("UPDATE {table} SET v = 21 WHERE id = 2"))
            .unwrap();
        seq += 1;
        assert!(
            engine.table_device_authoritative(table),
            "{table}: nullable UPDATE must not rehydrate"
        );
        let updated = engine
            .execute_relational_select_text(&format!("SELECT v FROM {table} WHERE id = 2"))
            .unwrap();
        assert_eq!(updated.rows, vec![vec![SqlValue::Int4(21)]], "{table}");

        engine
            .execute_dml_concurrent(seq, &format!("DELETE FROM {table} WHERE id = 2"))
            .unwrap();
        seq += 1;
        assert!(
            engine.table_device_authoritative(table),
            "{table}: nullable DELETE must not rehydrate"
        );
        assert!(
            engine
                .execute_relational_select_text(&format!("SELECT id FROM {table} WHERE id = 2"))
                .unwrap()
                .rows
                .is_empty(),
            "{table}: deleted nullable row remained visible"
        );
    }
}
