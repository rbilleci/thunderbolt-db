use super::*;

fn straddled_rich_select(
    engine: &Arc<Engine>,
    sql: &'static str,
    publish: impl FnOnce(&Engine) -> Result<(), ExecuteError>,
) -> RelationalSelectResult {
    let captured = Arc::new(std::sync::Barrier::new(2));
    let resume = Arc::new(std::sync::Barrier::new(2));
    let reader = {
        let engine = Arc::clone(engine);
        let captured = Arc::clone(&captured);
        let resume = Arc::clone(&resume);
        std::thread::spawn(move || {
            engine.execute_resident_expr_select_sql_instrumented(sql, || {
                captured.wait();
                resume.wait();
            })
        })
    };

    captured.wait();
    let publication = publish(engine);
    resume.wait();
    publication.expect("concurrent publication must complete while the rich reader is retained");
    reader
        .join()
        .expect("rich reader thread")
        .expect("retained rich reader")
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn rich_autocommit_select_retains_one_generation_across_publication_and_drop_recreate() {
    let mut publication = Engine::new_local_test_engine();
    publication
        .execute_text(
            1,
            "CREATE TABLE rich_snapshot_publication (id INT PRIMARY KEY)",
        )
        .unwrap();
    publication
        .execute_text(2, "INSERT INTO rich_snapshot_publication VALUES (1)")
        .unwrap();
    publication
        .populate_relational_residency_snapshot("rich_snapshot_publication")
        .unwrap();
    let publication = Arc::new(publication);
    let result = straddled_rich_select(
        &publication,
        "SELECT id FROM rich_snapshot_publication WHERE id + 0 > 0 ORDER BY id",
        |engine| engine.execute_text(3, "INSERT INTO rich_snapshot_publication VALUES (2)"),
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(1)]],
        "a concurrent publication must not replace the retained device generation"
    );

    let mut aba = Engine::new_local_test_engine();
    aba.execute_text(10, "CREATE TABLE rich_snapshot_aba (id INT PRIMARY KEY)")
        .unwrap();
    aba.execute_text(11, "INSERT INTO rich_snapshot_aba VALUES (7)")
        .unwrap();
    aba.populate_relational_residency_snapshot("rich_snapshot_aba")
        .unwrap();
    let aba = Arc::new(aba);
    let result = straddled_rich_select(
        &aba,
        "SELECT id FROM rich_snapshot_aba WHERE id + 0 > 0 ORDER BY id",
        |engine| {
            engine
                .execute_text(12, "DROP TABLE rich_snapshot_aba")
                .and_then(|_| {
                    engine.execute_text(13, "CREATE TABLE rich_snapshot_aba (id INT PRIMARY KEY)")
                })
                .and_then(|_| engine.execute_text(14, "INSERT INTO rich_snapshot_aba VALUES (99)"))
                .and_then(|_| {
                    engine
                        .populate_relational_residency_snapshot_shared("rich_snapshot_aba")
                        .map(|_| ())
                })
        },
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(7)]],
        "a shape-identical DROP/recreate must not replace the retained relation generation"
    );

    for sharded in [false, true] {
        let layout = if sharded { "sharded" } else { "single-buffer" };
        let mut engine = Engine::new_local_test_engine();
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(1);
        engine
            .execute_text(20, "CREATE TABLE retained_join_left (id INT PRIMARY KEY)")
            .unwrap();
        engine
            .execute_text(21, "CREATE TABLE retained_join_right (id INT PRIMARY KEY)")
            .unwrap();
        engine
            .execute_text(22, "INSERT INTO retained_join_left VALUES (1)")
            .unwrap();
        engine
            .execute_text(23, "INSERT INTO retained_join_right VALUES (1)")
            .unwrap();
        if sharded {
            engine
                .populate_relational_residency_snapshot_shared("retained_join_left")
                .unwrap();
            engine
                .populate_relational_residency_snapshot_shared("retained_join_right")
                .unwrap();
            assert!(
                engine
                    .read_residency_shards()
                    .get("retained_join_left")
                    .is_some_and(|shards| !shards.is_empty()),
                "{layout} precondition"
            );
        } else {
            install_test_single_buffer_residency(&mut engine, "retained_join_left");
            install_test_single_buffer_residency(&mut engine, "retained_join_right");
            assert!(
                engine
                    .relational_residency_entry("retained_join_left")
                    .is_some(),
                "{layout} precondition"
            );
        }
        let engine = Arc::new(engine);
        let joined = straddled_rich_select(
            &engine,
            "SELECT l.id, r.id FROM retained_join_left l \
             JOIN retained_join_right r ON l.id = r.id ORDER BY l.id",
            |engine| {
                engine.set_shard_residency_enabled(true);
                let inserted = if sharded {
                    engine.execute_text(24, "INSERT INTO retained_join_left VALUES (2)")
                } else {
                    Ok(())
                };
                inserted
                    .and_then(|_| engine.execute_text(25, "DROP TABLE retained_join_right"))
                    .and_then(|_| {
                        engine.execute_text(
                            26,
                            "CREATE TABLE retained_join_right (id INT PRIMARY KEY)",
                        )
                    })
                    .and_then(|_| {
                        engine.execute_text(27, "INSERT INTO retained_join_right VALUES (9)")
                    })
                    .and_then(|_| {
                        engine
                            .populate_relational_residency_snapshot_shared("retained_join_right")
                            .map(|_| ())
                    })
            },
        );
        assert_eq!(joined.executed_target, DeviceTarget::Gpu(0), "{layout}");
        assert_eq!(
            joined.rows,
            vec![vec![SqlValue::Int4(1), SqlValue::Int4(1)]],
            "{layout} JOIN must use one retained generation across INSERT and DROP/recreate"
        );

        let ranked = straddled_rich_select(
            &engine,
            "SELECT id, row_number() OVER (ORDER BY id) FROM retained_join_left ORDER BY id",
            |engine| {
                engine
                    .execute_text(28, "DROP TABLE retained_join_left")
                    .and_then(|_| {
                        engine.execute_text(
                            29,
                            "CREATE TABLE retained_join_left (id INT PRIMARY KEY)",
                        )
                    })
                    .and_then(|_| {
                        engine.execute_text(30, "INSERT INTO retained_join_left VALUES (8)")
                    })
                    .and_then(|_| {
                        engine
                            .populate_relational_residency_snapshot_shared("retained_join_left")
                            .map(|_| ())
                    })
            },
        );
        assert_eq!(ranked.executed_target, DeviceTarget::Gpu(0), "{layout}");
        let expected_rank = if sharded {
            vec![
                vec![SqlValue::Int4(1), SqlValue::Int8(1)],
                vec![SqlValue::Int4(2), SqlValue::Int8(2)],
            ]
        } else {
            vec![vec![SqlValue::Int4(1), SqlValue::Int8(1)]]
        };
        assert_eq!(
            ranked.rows, expected_rank,
            "{layout} rank must retain the pre-DROP generation"
        );
    }
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn explicit_transaction_oversized_join_and_rank_do_not_repair_global_residency() {
    let mut engine = Engine::new_local_test_engine();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(64);
    engine
        .execute_text(1, "CREATE TABLE scoped_large_left (id INT PRIMARY KEY)")
        .unwrap();
    engine
        .execute_text(2, "CREATE TABLE scoped_large_right (id INT PRIMARY KEY)")
        .unwrap();
    let values = (0..2_000)
        .map(|value| format!("({value})"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(3, &format!("INSERT INTO scoped_large_left VALUES {values}"))
        .unwrap();
    engine
        .execute_text(
            4,
            &format!("INSERT INTO scoped_large_right VALUES {values}"),
        )
        .unwrap();
    engine
        .populate_relational_residency_snapshot_shared("scoped_large_left")
        .unwrap();
    engine
        .populate_relational_residency_snapshot_shared("scoped_large_right")
        .unwrap();

    engine
        .execute_text(90, "BEGIN ISOLATION LEVEL REPEATABLE READ")
        .unwrap();
    let pinned = engine
        .execute_resident_expr_select_sql_in_transaction(
            90,
            "SELECT id FROM scoped_large_left WHERE id + 0 = 0",
        )
        .unwrap();
    assert_eq!(pinned.rows, vec![vec![SqlValue::Int4(0)]]);
    let shards_before = engine.read_state.residency.shards.load_full();
    let cold_before = engine
        .read_state
        .residency
        .streaming_cold_chunks
        .load_full();
    engine.set_relational_residency_budget_bytes(0, 4_096);

    let joined = engine
        .execute_resident_expr_select_sql_in_transaction(
            90,
            "SELECT l.id, r.id FROM scoped_large_left l \
             JOIN scoped_large_right r ON l.id = r.id \
             WHERE l.id = 1999 AND r.id = 1999",
        )
        .expect("the explicit transaction must execute from its retained resident inputs");
    assert_eq!(joined.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        joined.rows,
        vec![vec![SqlValue::Int4(1999), SqlValue::Int4(1999)]]
    );

    let rank_error = engine
        .execute_resident_expr_select_sql_in_transaction(
            90,
            "SELECT id, row_number() OVER (ORDER BY id) \
             FROM scoped_large_left ORDER BY id LIMIT 1",
        )
        .expect_err("an absent retained streaming representation must fail cleanly")
        .to_string();
    assert!(
        rank_error.contains("resident input") && rank_error.contains("exceeds the query budget"),
        "unexpected scoped rank error: {rank_error}"
    );

    let shards_after = engine.read_state.residency.shards.load_full();
    let cold_after = engine
        .read_state
        .residency
        .streaming_cold_chunks
        .load_full();
    assert!(
        Arc::ptr_eq(&shards_before, &shards_after),
        "scoped join/rank must not publish a global shard repair"
    );
    assert!(
        Arc::ptr_eq(&cold_before, &cold_after),
        "scoped join/rank must not publish global cold chunks"
    );
    engine.execute_text(90, "ROLLBACK").unwrap();
}
