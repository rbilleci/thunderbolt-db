/// READ-002 canonical engine-internal route: `(tenant_id int4, account_id int8)` probes a
/// generation-owned table-level GPU directory, exact-rechecks collisions on-device, applies MVCC,
/// and gathers the required fixed-width account projections without a cold or CPU relational path.
///
/// The unreferenced `optional_code` NULLs are a differential guard against accidentally treating a
/// resident NULL bitmap as a whole-table decline. Conversely, selecting that NULL-bearing column is
/// rejected explicitly until the prepared result ABI carries validity bits; raw zero is never exposed
/// as a substitute for SQL NULL.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn compound_i32_i64_prepared_route_is_exact_resident_and_generation_safe() {
    fn compound_collision(tenant: i32) -> (i64, i64) {
        let step = |h: u32, word: i32| {
            let h = (h ^ word as u32).wrapping_mul(0x0100_0193);
            h.rotate_left(13).wrapping_add(0x9E37_79B1)
        };
        let prefix = step(0x811C_9DC5, tenant);
        let mut buckets = std::collections::HashMap::<u32, (u32, u32)>::new();
        let mut pair = None;
        for low in 10_000_u32..2_000_000 {
            let state = step(prefix, low as i32);
            if let Some((previous_low, previous_state)) = buckets.insert(state >> 20, (low, state))
            {
                let high_a = 1_u32 << 20;
                let high_b = high_a ^ (previous_state ^ state);
                pair = Some((
                    ((u64::from(high_a) << 32) | u64::from(previous_low)) as i64,
                    ((u64::from(high_b) << 32) | u64::from(low)) as i64,
                ));
                break;
            }
        }
        let pair = pair.expect("construct a compound `(int4, int8)` fingerprint collision");
        let fingerprint = |account_id: i64| {
            let bits = account_id as u64;
            crate::engine_residency::compound_key_fingerprint(&[
                tenant,
                bits as u32 as i32,
                (bits >> 32) as u32 as i32,
            ])
        };
        assert_ne!(pair.0, pair.1);
        assert_eq!(fingerprint(pair.0), fingerprint(pair.1));
        pair
    }

    let mut e = std::sync::Arc::new(Engine::new_local());
    e.set_shard_residency_enabled(true);
    e.set_shard_int8_section_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(32);
    e.execute_text(
        1,
        "CREATE TABLE accounts_read002 (tenant_id INT, account_id BIGINT, balance_cents BIGINT, version BIGINT, status SMALLINT, optional_code INT, PRIMARY KEY (tenant_id, account_id))",
    )
    .unwrap();
    let (collision_a, collision_b) = compound_collision(7);
    let mut rows = (0_i64..96)
        .map(|row| {
            let optional = if row % 11 == 0 {
                "NULL".to_string()
            } else {
                (row as i32).to_string()
            };
            format!(
                "({}, {}, {}, {}, {}, {optional})",
                (row % 8) + 1,
                10_000 + row,
                5_000_000_000_i64 + row * 17,
                -6_000_000_000_i64 - row,
                (row % 4) + 1,
            )
        })
        .collect::<Vec<_>>();
    rows.push(format!(
        "(7, {collision_a}, 7000000001, -9000000001, 2, NULL)"
    ));
    rows.push(format!(
        "(7, {collision_b}, -7000000002, 9000000002, 3, 42)"
    ));
    rows.push(format!(
        "(8, {collision_a}, 15000000001, -16000000001, 1, 43)"
    ));
    e.execute_text(
        2,
        &format!("INSERT INTO accounts_read002 VALUES {}", rows.join(",")),
    )
    .unwrap();
    let cold_before = e.streaming_cold_hits();
    assert_eq!(cold_before, 0, "fixture construction used no cold tier");

    let template = e
        .prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "accounts_read002",
            ["tenant_id", "account_id"],
            &["balance_cents", "version", "status"],
        )
        .expect("prepare canonical READ-002 route");
    assert!(
        e.prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "accounts_read002",
            ["tenant_id", "account_id"],
            &["optional_code"],
        )
        .is_err(),
        "a referenced NULL-bearing projection must decline rather than expose raw zero"
    );

    let params = vec![
        RelationalCompoundI32I64PointReadParam {
            first: 1,
            second: 10_000,
        },
        RelationalCompoundI32I64PointReadParam {
            first: 7,
            second: collision_a,
        },
        RelationalCompoundI32I64PointReadParam {
            first: 7,
            second: collision_b,
        },
        RelationalCompoundI32I64PointReadParam {
            first: 3,
            second: 10_042,
        },
        RelationalCompoundI32I64PointReadParam {
            first: 1,
            second: 10_000,
        },
        RelationalCompoundI32I64PointReadParam {
            first: 99,
            second: -1,
        },
    ];
    let gpu_before = e.sharded_point_gpu_probe_hits();
    let batch_before = e.sharded_point_batch_hits();
    let got = e
        .execute_relational_compound_i32_i64_point_reads(&template, &params)
        .expect("execute canonical READ-002 route");
    assert_eq!(e.sharded_point_gpu_probe_hits(), gpu_before + 1);
    assert_eq!(e.sharded_point_batch_hits(), batch_before + 1);
    assert_eq!(e.streaming_cold_hits(), cold_before, "zero cold accesses");
    // The independent general executor cannot yet lower the mixed-width compound conjunction; that
    // SQL/type wiring is the next PRODUCT-002 milestone. Force its full resident GPU scan instead,
    // then let this test harness select the exact tuple from device-produced rows. The duplicate
    // `account_id=collision_a` under tenant 8 makes both key components load-bearing.
    let Command::Select(scan) = parse_command(
        "SELECT tenant_id, account_id, balance_cents, version, status FROM accounts_read002",
    )
    .expect("parse independent GPU scan oracle") else {
        panic!("expected SELECT oracle")
    };
    let oracle = e
        .execute_resident_select_via_general(&scan)
        .expect("authoritative full GPU scan");
    assert!(matches!(oracle.executed_target, DeviceTarget::Gpu(_)));
    assert!(oracle.fallback_reason.is_none());
    let oracle_rows = oracle.rows.into_boxed();
    for (param, result) in params.iter().zip(&got) {
        assert!(matches!(result.planned_target, DeviceTarget::Gpu(_)));
        assert!(matches!(result.executed_target, DeviceTarget::Gpu(_)));
        assert!(result.fallback_reason.is_none());
        let expected = oracle_rows
            .iter()
            .filter(|row| {
                row[0] == SqlValue::Int4(param.first) && row[1] == SqlValue::Int8(param.second)
            })
            .map(|row| row[2..].to_vec())
            .collect::<Vec<_>>();
        assert_eq!(
            result.rows.clone().into_boxed(),
            expected,
            "prepared route must be byte-identical to the exact GPU scan for ({}, {})",
            param.first,
            param.second
        );
    }
    assert_eq!(
        got[1].rows.clone().into_boxed(),
        vec![vec![
            SqlValue::Int8(7_000_000_001),
            SqlValue::Int8(-9_000_000_001),
            SqlValue::Int2(2),
        ]]
    );
    assert_eq!(
        got[2].rows.clone().into_boxed(),
        vec![vec![
            SqlValue::Int8(-7_000_000_002),
            SqlValue::Int8(9_000_000_002),
            SqlValue::Int2(3),
        ]]
    );

    let cache_before = e.sharded_point_route_cache_hits();
    let repeated = e
        .execute_relational_compound_i32_i64_point_reads(&template, &params)
        .expect("reuse the exact generation route");
    assert_eq!(e.sharded_point_route_cache_hits(), cache_before + 1);
    assert_eq!(e.streaming_cold_hits(), cold_before);
    assert_eq!(
        repeated
            .iter()
            .map(|result| result.rows.clone().into_boxed())
            .collect::<Vec<_>>(),
        got.iter()
            .map(|result| result.rows.clone().into_boxed())
            .collect::<Vec<_>>()
    );
    let expected = got
        .iter()
        .map(|result| result.rows.clone().into_boxed())
        .collect::<Vec<_>>();
    let concurrent = (0..2)
        .map(|_| {
            let engine = std::sync::Arc::clone(&e);
            let template = template.clone();
            let params = params.clone();
            std::thread::spawn(move || {
                engine
                    .execute_relational_compound_i32_i64_point_reads(&template, &params)
                    .expect("concurrent readers reuse one immutable route")
                    .into_iter()
                    .map(|result| result.rows.into_boxed())
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    for reader in concurrent {
        assert_eq!(reader.join().expect("concurrent READ-002 reader"), expected);
    }
    assert_eq!(e.streaming_cold_hits(), cold_before);

    let e = std::sync::Arc::get_mut(&mut e).expect("all concurrent route readers drained");
    let pressured_gpu_hits = e.sharded_point_gpu_probe_hits();
    let pressured_batch_hits = e.sharded_point_batch_hits();
    assert!(e.table_device_authoritative("accounts_read002"));
    let pressure_generation = std::sync::Arc::clone(
        &e.read_residency_shards()["accounts_read002"][0].point_route_generation,
    );
    let pressure_route_count = e
        .read_state
        .residency
        .compound_point_routes
        .load()
        .len();
    e.mark_gpu_memory_pressured(0);
    assert!(std::sync::Arc::ptr_eq(
        &pressure_generation,
        &e.read_residency_shards()["accounts_read002"][0].point_route_generation
    ));
    assert_eq!(
        e.read_state
            .residency
            .compound_point_routes
            .load()
            .len(),
        pressure_route_count,
        "authoritative pressure leaves the cached route present for the execution-time gate"
    );
    let pressure_error = e
        .execute_relational_compound_i32_i64_point_reads(&template, &params)
        .unwrap_err()
        .to_string();
    assert!(pressure_error.contains("memory pressured"), "{pressure_error}");
    assert_eq!(e.sharded_point_gpu_probe_hits(), pressured_gpu_hits);
    assert_eq!(e.sharded_point_batch_hits(), pressured_batch_hits);
    e.clear_gpu_memory_pressured(0);
    e.execute_relational_compound_i32_i64_point_reads(&template, &params)
        .expect("the exact route is reusable after pressure clears");

    e.execute_text(
        3,
        &format!(
            "UPDATE accounts_read002 SET balance_cents = 11000000001, version = -12000000001, status = 4 WHERE tenant_id = 7 AND account_id = {collision_a}"
        ),
    )
    .unwrap();
    let stale_gpu_hits = e.sharded_point_gpu_probe_hits();
    let stale_batch_hits = e.sharded_point_batch_hits();
    assert!(
        e.execute_relational_compound_i32_i64_point_reads(&template, &params)
            .is_err(),
        "a template cannot cross a resident generation publication"
    );
    assert_eq!(e.sharded_point_gpu_probe_hits(), stale_gpu_hits);
    assert_eq!(e.sharded_point_batch_hits(), stale_batch_hits);
    let updated_template = e
        .prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "accounts_read002",
            ["tenant_id", "account_id"],
            &["balance_cents", "version", "status"],
        )
        .expect("reprepare after generation publication");
    let updated = e
        .execute_relational_compound_i32_i64_point_reads(
            &updated_template,
            &[RelationalCompoundI32I64PointReadParam {
                first: 7,
                second: collision_a,
            }],
        )
        .expect("new generation sees the updated version");
    assert_eq!(
        updated[0].rows.clone().into_boxed(),
        vec![vec![
            SqlValue::Int8(11_000_000_001),
            SqlValue::Int8(-12_000_000_001),
            SqlValue::Int2(4),
        ]]
    );

    e.execute_text(
        4,
        &format!("DELETE FROM accounts_read002 WHERE tenant_id = 7 AND account_id = {collision_b}"),
    )
    .unwrap();
    let delete_cache_hits = e.sharded_point_route_cache_hits();
    let deleted = e
        .execute_relational_compound_i32_i64_point_reads(
            &updated_template,
            &[RelationalCompoundI32I64PointReadParam {
                first: 7,
                second: collision_b,
            }],
        )
        .expect("the retained route applies the in-place tombstone sidecar");
    assert!(deleted[0].rows.is_empty());
    assert_eq!(
        e.sharded_point_route_cache_hits(),
        delete_cache_hits + 1,
        "DELETE visibility reused the exact prepared directory"
    );

    e.execute_text(
        5,
        "INSERT INTO accounts_read002 VALUES (9, 99999999999, -13000000001, 14000000001, 4, NULL)",
    )
    .unwrap();
    let inserted_template = e
        .prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "accounts_read002",
            ["tenant_id", "account_id"],
            &["balance_cents", "version", "status"],
        )
        .expect("reprepare after insert publication");
    let fresh = e
        .execute_relational_compound_i32_i64_point_reads(
            &inserted_template,
            &[RelationalCompoundI32I64PointReadParam {
                first: 9,
                second: 99_999_999_999,
            }],
        )
        .expect("new generation sees the inserted row");
    assert_eq!(
        fresh[0].rows.clone().into_boxed(),
        vec![vec![
            SqlValue::Int8(-13_000_000_001),
            SqlValue::Int8(14_000_000_001),
            SqlValue::Int2(4),
        ]]
    );

    e.execute_text(
        6,
        "CREATE TABLE limits_read002 (tenant_id INT, account_id BIGINT, debit_limit_cents BIGINT, PRIMARY KEY (tenant_id, account_id))",
    )
    .unwrap();
    e.execute_text(
        7,
        "INSERT INTO limits_read002 VALUES (9, 99999999999, -7654321987)",
    )
    .unwrap();
    let limit_template = e
        .prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "limits_read002",
            ["tenant_id", "account_id"],
            &["debit_limit_cents"],
        )
        .expect("prepare the required debit-limit projection route");
    let limit = e
        .execute_relational_compound_i32_i64_point_reads(
            &limit_template,
            &[RelationalCompoundI32I64PointReadParam {
                first: 9,
                second: 99_999_999_999,
            }],
        )
        .unwrap();
    assert_eq!(
        limit[0].rows.clone().into_boxed(),
        vec![vec![SqlValue::Int8(-7_654_321_987)]]
    );
    assert_eq!(e.streaming_cold_hits(), cold_before);
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn compound_route_concurrent_prepare_reuses_one_exact_budgeted_plan() {
    let mut e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_shard_int8_section_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(16);
    e.execute_text(
        1,
        "CREATE TABLE prepare_race_read002 (tenant_id INT, account_id BIGINT, value BIGINT, PRIMARY KEY (tenant_id, account_id))",
    )
    .unwrap();
    let rows = (0_i64..40)
        .map(|row| format!("({}, {}, {})", row % 4, 1000 + row, 5000000000_i64 + row))
        .collect::<Vec<_>>();
    e.execute_text(
        2,
        &format!("INSERT INTO prepare_race_read002 VALUES {}", rows.join(",")),
    )
    .unwrap();
    let shards = e.read_residency_shards();
    let table_shards = &shards["prepare_race_read002"];
    let estimated = CudaI32I64MultiShardProbePlan::estimated_allocated_bytes(
        table_shards.len(),
        table_shards
            .iter()
            .map(|shard| shard.row_count as u64)
            .sum(),
    )
    .unwrap();
    let base = e.relational_resident_bytes_for_gpu(0);
    e.set_relational_residency_budget_bytes(0, base + estimated);
    let e = std::sync::Arc::new(e);
    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    e.set_sharded_point_route_pre_publish_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    let first_engine = std::sync::Arc::clone(&e);
    let first = std::thread::spawn(move || {
        first_engine.prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "prepare_race_read002",
            ["tenant_id", "account_id"],
            &["value"],
        )
    });
    reached.wait();
    let second_engine = std::sync::Arc::clone(&e);
    let second = std::thread::spawn(move || {
        second_engine.prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "prepare_race_read002",
            ["tenant_id", "account_id"],
            &["value"],
        )
    });
    let second_template = second
        .join()
        .expect("second prepare thread")
        .expect("second prepare publishes the one budgeted plan");
    resume.wait();
    let first_template = first
        .join()
        .expect("first prepare thread")
        .expect("first prepare reuses the concurrently published plan");
    assert_eq!(first_template.route_id, second_template.route_id);
    assert!(e.relational_resident_bytes_for_gpu(0) <= base + estimated);
    assert_eq!(
        e.read_state
            .residency
            .compound_point_routes
            .load()
            .len(),
        1,
        "both preparers converge on one retained allocation"
    );
    let result = e
        .execute_relational_compound_i32_i64_point_reads(
            &first_template,
            &[RelationalCompoundI32I64PointReadParam {
                first: 3,
                second: 1_039,
            }],
        )
        .unwrap();
    assert_eq!(
        result[0].rows.clone().into_boxed(),
        vec![vec![SqlValue::Int8(5_000_000_039)]]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn compound_route_build_cannot_publish_after_generation_changes() {
    let e = std::sync::Arc::new(Engine::new_local());
    e.set_shard_residency_enabled(true);
    e.set_shard_int8_section_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE stale_build_read002 (tenant_id INT, account_id BIGINT, value BIGINT, PRIMARY KEY (tenant_id, account_id))",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO stale_build_read002 VALUES (1, 1001, 5000000001)",
    )
    .unwrap();
    let reached = std::sync::Arc::new(std::sync::Barrier::new(2));
    let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
    e.set_sharded_point_route_pre_publish_hook(
        std::sync::Arc::clone(&reached),
        std::sync::Arc::clone(&resume),
    );
    let builder_engine = std::sync::Arc::clone(&e);
    let builder = std::thread::spawn(move || {
        builder_engine.prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "stale_build_read002",
            ["tenant_id", "account_id"],
            &["value"],
        )
    });
    reached.wait();
    e.execute_text(
        3,
        "INSERT INTO stale_build_read002 VALUES (2, 2002, -6000000002)",
    )
    .unwrap();
    resume.wait();
    let stale_error = builder
        .join()
        .expect("stale builder thread")
        .unwrap_err()
        .to_string();
    assert!(
        stale_error.contains("generation changed during preparation"),
        "the exact post-build generation recheck must reject G0: {stale_error}"
    );
    assert!(
        e.read_state
            .residency
            .compound_point_routes
            .load()
            .keys()
            .all(|(table, _, _)| table != "stale_build_read002"),
        "the losing builder published no stale route"
    );
    let current = e
        .prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "stale_build_read002",
            ["tenant_id", "account_id"],
            &["value"],
        )
        .expect("current generation prepares normally");
    let result = e
        .execute_relational_compound_i32_i64_point_reads(
            &current,
            &[RelationalCompoundI32I64PointReadParam {
                first: 2,
                second: 2_002,
            }],
        )
        .unwrap();
    assert_eq!(
        result[0].rows.clone().into_boxed(),
        vec![vec![SqlValue::Int8(-6_000_000_002)]]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn retired_compound_plan_remains_charged_until_last_owner_drains() {
    let mut e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_shard_int8_section_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(
        1,
        "CREATE TABLE retired_charge_read002 (tenant_id INT, account_id BIGINT, value BIGINT, PRIMARY KEY (tenant_id, account_id))",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO retired_charge_read002 VALUES (1, 1001, 5000000001)",
    )
    .unwrap();
    e.prepare_relational_compound_i32_i64_point_read_template(
        "public",
        "retired_charge_read002",
        ["tenant_id", "account_id"],
        &["value"],
    )
    .unwrap();
    let pinned_routes = e.read_state.residency.compound_point_routes.load_full();
    let old_plan_bytes = pinned_routes
        .iter()
        .find(|((table, _, _), _)| table == "retired_charge_read002")
        .map(|(_, route)| route.plan.plan.allocated_bytes())
        .expect("pin the old compound plan through cache retirement");
    e.execute_text(
        3,
        "INSERT INTO retired_charge_read002 VALUES (2, 2002, -6000000002)",
    )
    .unwrap();
    assert!(e
        .read_state
        .residency
        .compound_point_routes
        .load()
        .keys()
        .all(|(table, _, _)| table != "retired_charge_read002"));
    assert_eq!(
        e.read_state
            .residency
            .live_compound_point_route_bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&(0, "retired_charge_read002".to_string()))
            .copied(),
        Some(old_plan_bytes)
    );
    let shards = e.read_residency_shards();
    let current = &shards["retired_charge_read002"];
    let replacement_bytes = CudaI32I64MultiShardProbePlan::estimated_allocated_bytes(
        current.len(),
        current.iter().map(|shard| shard.row_count as u64).sum(),
    )
    .unwrap();
    let base_without_old = e
        .relational_resident_bytes_for_gpu(0)
        .saturating_sub(old_plan_bytes);
    e.set_relational_residency_budget_bytes(0, base_without_old + replacement_bytes);
    let budget_error = e
        .prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "retired_charge_read002",
            ["tenant_id", "account_id"],
            &["value"],
        )
        .unwrap_err()
        .to_string();
    assert!(
        budget_error.contains("residency budget exceeded"),
        "a pinned retired plan must consume the one-plan budget: {budget_error}"
    );
    drop(pinned_routes);
    assert!(e
        .read_state
        .residency
        .live_compound_point_route_bytes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_empty());
    let current_template = e
        .prepare_relational_compound_i32_i64_point_read_template(
            "public",
            "retired_charge_read002",
            ["tenant_id", "account_id"],
            &["value"],
        )
        .expect("replacement fits after the retired owner drains");
    let result = e
        .execute_relational_compound_i32_i64_point_reads(
            &current_template,
            &[RelationalCompoundI32I64PointReadParam {
                first: 2,
                second: 2_002,
            }],
        )
        .unwrap();
    assert_eq!(
        result[0].rows.clone().into_boxed(),
        vec![vec![SqlValue::Int8(-6_000_000_002)]]
    );
}
