use super::*;

#[test]
fn catalog_visibility_and_explicit_text_oid_comparisons_fail_closed_during_binding() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE visibility_binding_probe (id INT)")
        .unwrap();

    for sql in [
        "SELECT relname FROM pg_catalog.pg_class \
         WHERE pg_catalog.pg_table_is_visible(relnamespace)",
        "SELECT relname FROM pg_catalog.pg_class \
         WHERE pg_catalog.pg_table_is_visible(relname)",
        "SELECT relname FROM pg_catalog.pg_class \
         WHERE pg_catalog.pg_table_is_visible('2200')",
        "SELECT typname FROM pg_catalog.pg_type \
         WHERE pg_catalog.pg_type_is_visible(typnamespace)",
        "SELECT proname FROM pg_catalog.pg_proc \
         WHERE pg_catalog.pg_function_is_visible('not-an-oid')",
        "SELECT relname FROM pg_catalog.pg_class WHERE relnamespace = '2200'::text",
        "SELECT relname FROM pg_catalog.pg_class WHERE '2200'::text = relnamespace",
    ] {
        let error = engine
            .execute_resident_expr_select_sql(sql)
            .expect_err("invalid visibility/OID typing must fail during binding")
            .to_string();
        assert!(
            !error.contains("GPU execution is required"),
            "`{sql}` reached device execution instead of failing closed: {error}"
        );
    }
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn gpu_catalog_visibility_accepts_only_corresponding_oid_membership_shapes() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE visibility_gpu_probe (id INT)")
        .unwrap();

    for sql in [
        "SELECT typname FROM pg_catalog.pg_type \
         WHERE pg_catalog.pg_type_is_visible(typnamespace)",
        "SELECT typname FROM pg_catalog.pg_type \
         WHERE pg_catalog.pg_type_is_visible(23)",
        "SELECT typname FROM pg_catalog.pg_type \
         WHERE pg_catalog.pg_table_is_visible(oid)",
        "SELECT t.n FROM pg_catalog.pg_type AS t(x,n,l,k,s) \
         WHERE pg_catalog.pg_type_is_visible(t.s)",
    ] {
        let error = engine
            .execute_resident_expr_select_sql(sql)
            .expect_err("invalid visibility shape must fail during binding")
            .to_string();
        assert!(
            !error.contains("GPU execution is required"),
            "`{sql}` reached device execution instead of failing during binding: {error}"
        );
    }

    let visible = engine
        .execute_resident_expr_select_sql(
            "SELECT c.relname FROM pg_catalog.pg_class c \
             WHERE pg_catalog.pg_table_is_visible(c.oid) \
               AND c.relname = 'visibility_gpu_probe'",
        )
        .unwrap();
    assert_eq!(visible.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        visible.rows,
        vec![vec![SqlValue::Text("visibility_gpu_probe".to_string())]]
    );

    let unknown_literal = engine
        .execute_resident_expr_select_sql(
            "SELECT c.relname FROM pg_catalog.pg_class c \
             WHERE c.relnamespace = '2200' AND c.relname = 'visibility_gpu_probe'",
        )
        .unwrap();
    assert_eq!(unknown_literal.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(unknown_literal.rows, visible.rows);

    engine
        .execute_text(2, "CREATE TABLE public.pg_class (id INT)")
        .unwrap();
    let composed = engine
        .execute_resident_expr_select_sql(
            "SELECT c.relname FROM pg_catalog.pg_class c \
             WHERE pg_catalog.pg_table_is_visible(c.oid) \
               AND (c.relname = 'visibility_gpu_probe' OR c.relname = 'pg_class') \
             ORDER BY c.relname",
        )
        .unwrap();
    assert_eq!(composed.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(composed.rows, visible.rows);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn gpu_pg_type_visibility_honors_pg_catalog_shadowing() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE DOMAIN public.account_id AS int4")
        .unwrap();
    engine
        .execute_text(2, "CREATE DOMAIN public.int4 AS int4")
        .unwrap();

    let builtin = engine
        .execute_resident_expr_select_sql(
            "SELECT oid, typnamespace FROM pg_catalog.pg_type \
             WHERE typname = 'int4' AND pg_catalog.pg_type_is_visible(oid) \
             ORDER BY oid",
        )
        .unwrap();
    assert_eq!(builtin.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        builtin.rows,
        vec![vec![
            SqlValue::Int4(23),
            SqlValue::Int4(PG_CATALOG_NAMESPACE_OID),
        ]]
    );

    let public = engine
        .execute_resident_expr_select_sql(
            "SELECT typname, typnamespace FROM pg_catalog.pg_type \
             WHERE typname = 'account_id' AND pg_catalog.pg_type_is_visible(oid)",
        )
        .unwrap();
    assert_eq!(public.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        public.rows,
        vec![vec![
            SqlValue::Text("account_id".to_string()),
            SqlValue::Int4(PG_PUBLIC_NAMESPACE_OID),
        ]]
    );

    for sql in [
        "SELECT typname FROM pg_catalog.pg_type \
         WHERE pg_catalog.pg_type_is_visible(oid) AND typname = 'account_id'",
        "SELECT typname FROM pg_catalog.pg_type \
         WHERE pg_catalog.pg_type_is_visible(pg_type.oid) AND typname = 'account_id'",
        "SELECT t.typname FROM pg_catalog.pg_type t \
         WHERE pg_catalog.pg_type_is_visible(t.oid) AND t.typname = 'account_id'",
        "SELECT t.n FROM pg_catalog.pg_type AS t(x,n,l,k,s) \
         WHERE pg_catalog.pg_type_is_visible(t.x) AND t.n = 'account_id'",
    ] {
        let aliased = engine.execute_resident_expr_select_sql(sql).unwrap();
        assert_eq!(aliased.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(
            aliased.rows,
            vec![vec![SqlValue::Text("account_id".to_string())]],
            "{sql}"
        );
    }

    let wrong_alias = engine
        .execute_resident_expr_select_sql(
            "SELECT t.n FROM pg_catalog.pg_type AS t(x,n,l,k,s) \
             WHERE pg_catalog.pg_type_is_visible(t.s)",
        )
        .expect_err("an aliased non-OID column must still fail visibility binding")
        .to_string();
    assert!(
        wrong_alias.contains("requires the oid column"),
        "{wrong_alias}"
    );
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn over_budget_mixed_user_catalog_join_keeps_the_user_side_resident() {
    let mut engine = Engine::new_local_test_engine();
    engine.set_shard_residency_enabled(true);
    engine.set_shard_size_target(64);
    engine
        .execute_text(
            1,
            "CREATE TABLE mixed_catalog_join_probe (id INT PRIMARY KEY)",
        )
        .unwrap();
    let relation_oid = i32::try_from(
        engine
            .relational_catalog_table("mixed_catalog_join_probe")
            .unwrap()
            .oid,
    )
    .unwrap();
    let values = (0..2_000)
        .filter(|value| *value != relation_oid)
        .chain(std::iter::once(relation_oid))
        .map(|value| format!("({value})"))
        .collect::<Vec<_>>()
        .join(",");
    engine
        .execute_text(
            2,
            &format!("INSERT INTO mixed_catalog_join_probe VALUES {values}"),
        )
        .unwrap();
    engine
        .populate_relational_residency_snapshot_shared("mixed_catalog_join_probe")
        .unwrap();
    let shards_before = engine.read_state.residency.shards.load_full();
    let cold_before = engine
        .read_state
        .residency
        .streaming_cold_chunks
        .load_full();
    assert!(
        shards_before
            .get("mixed_catalog_join_probe")
            .is_some_and(|shards| !shards.is_empty()),
        "over-budget mixed join fixture must be shard-resident"
    );
    engine.set_relational_residency_budget_bytes(0, 4_096);

    for sql in [
        "SELECT u.id, c.relname FROM mixed_catalog_join_probe u \
         JOIN pg_catalog.pg_class c ON u.id = c.oid \
         WHERE c.relname = 'mixed_catalog_join_probe'",
        "SELECT u.id, c.relname FROM mixed_catalog_join_probe u, pg_catalog.pg_class c \
         WHERE u.id = c.oid AND c.relname = 'mixed_catalog_join_probe'",
    ] {
        let result = engine.execute_resident_expr_select_sql(sql).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert_eq!(
            result.rows,
            vec![vec![
                SqlValue::Int4(relation_oid),
                SqlValue::Text("mixed_catalog_join_probe".to_string()),
            ]],
            "{sql}"
        );
    }

    let shards_after = engine.read_state.residency.shards.load_full();
    let cold_after = engine
        .read_state
        .residency
        .streaming_cold_chunks
        .load_full();
    assert!(
        Arc::ptr_eq(&shards_before, &shards_after),
        "mixed catalog preparation must not transition the user peer"
    );
    assert!(
        Arc::ptr_eq(&cold_before, &cold_after),
        "mixed catalog preparation must not publish an unusable cold-only user peer"
    );
}
