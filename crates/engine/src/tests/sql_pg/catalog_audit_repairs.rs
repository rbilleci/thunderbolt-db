use super::*;

#[test]
fn dropped_public_regnamespace_fails_live_and_after_wal_replay() {
    let engine = Engine::new_local_test_engine();
    engine.execute_text(1, "DROP SCHEMA public").unwrap();

    let assert_missing_public_regnamespace = |candidate: &Engine| {
        let error = candidate
            .execute_resident_expr_select_sql(
                "SELECT nspname FROM pg_catalog.pg_namespace \
                 WHERE oid = 'public'::regnamespace",
            )
            .expect_err("a dropped public schema must not retain regnamespace identity")
            .to_string();
        assert!(
            error.contains("schema \"public\" does not exist for regnamespace cast"),
            "{error}"
        );
        assert!(!error.contains("GPU execution is required"), "{error}");
    };
    assert_missing_public_regnamespace(&engine);

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(!recovered.ddl_catalog().relational_public_schema_exists);
    assert_missing_public_regnamespace(&recovered);
}

#[test]
fn catalog_compatibility_binding_fails_closed_before_empty_or_device_execution() {
    let engine = Engine::new_local_test_engine();

    for sql in [
        "SELECT oid FROM pg_catalog.pg_publication \
         UNION ALL SELECT oid, pubname FROM pg_catalog.pg_publication",
        "SELECT oid FROM pg_catalog.pg_publication \
         UNION ALL SELECT pubname FROM pg_catalog.pg_publication",
        "SELECT CASE WHEN true THEN oid ELSE pubname END \
         FROM pg_catalog.pg_publication",
        "SELECT p.polname FROM pg_catalog.pg_policy p \
         JOIN pg_catalog.pg_trigger t ON p.definitely_missing = t.oid",
        "SELECT p.polname FROM pg_catalog.pg_policy p \
         JOIN pg_catalog.pg_trigger t ON p.polname = t.oid",
        "SELECT polname FROM pg_catalog.pg_policy \
         ORDER BY definitely_missing NULLS FIRST",
        "SELECT polname FROM pg_catalog.pg_policy \
         WHERE definitely_missing = 1",
        "SELECT pg_catalog.format_type(a.atttypid, a.attname) \
         FROM pg_catalog.pg_attribute a",
        "SELECT evil.pg_get_constraintdef(oid) FROM pg_catalog.pg_constraint",
        "SELECT polname + polname FROM pg_catalog.pg_policy",
        "SELECT oid ~ oid FROM pg_catalog.pg_policy",
        "SELECT polname FROM pg_catalog.pg_policy WHERE oid",
        "SELECT polname FROM pg_catalog.pg_policy WHERE oid IS TRUE",
        "SELECT polname FROM pg_catalog.pg_policy WHERE oid AND oid",
        "SELECT polname FROM pg_catalog.pg_policy WHERE oid IN ('not-an-oid')",
        "SELECT count(*) FROM pg_catalog.pg_policy WHERE oid",
        "SELECT DISTINCT ON (definitely_missing) polname FROM pg_catalog.pg_policy",
        "SELECT polname FROM pg_catalog.pg_policy LIMIT 'bad'",
        "SELECT polname FROM pg_catalog.pg_policy OFFSET 'bad'",
        "SELECT oid FROM pg_catalog.pg_publication UNION ALL \
         SELECT oid FROM pg_catalog.pg_publication LIMIT 'bad'",
        "SELECT CASE oid WHEN polname THEN 1 ELSE 2 END FROM pg_catalog.pg_policy",
        "SELECT polname FROM pg_catalog.pg_policy \
         ORDER BY CASE WHEN true THEN 1 ELSE 'incompatible' END",
        "SELECT (VALUES (1), ('incompatible')) FROM pg_catalog.pg_policy",
        "SELECT oid::int4[] FROM pg_catalog.pg_policy",
        "SELECT true::int4 FROM pg_catalog.pg_policy",
        "SELECT 1::bool FROM pg_catalog.pg_policy",
        "SELECT 'bad'::int4 FROM pg_catalog.pg_policy",
        "SELECT 'bad'::bool FROM pg_catalog.pg_policy",
        "SELECT 'definitely_missing'::regclass FROM pg_catalog.pg_policy",
        "SELECT 'definitely_missing'::regnamespace FROM pg_catalog.pg_policy",
        "SELECT 'definitely_missing'::regtype FROM pg_catalog.pg_policy",
        "SELECT 'bad'::text::int4 FROM pg_catalog.pg_policy",
        "SELECT '40000'::text::int2 FROM pg_catalog.pg_policy",
        "SELECT 'definitely_missing'::text::regtype FROM pg_catalog.pg_policy",
        "SELECT array_to_string(polname, ',') FROM pg_catalog.pg_policy",
        "SELECT array_upper(polname, 1) FROM pg_catalog.pg_policy",
        "SELECT array_upper(p.prattrs, 1) FROM pg_catalog.pg_policy AS p(prattrs)",
        "SELECT count(oid) WITHIN GROUP (ORDER BY oid) FROM pg_catalog.pg_policy",
        "SELECT p.oid FROM pg_catalog.pg_policy AS p(policy_oid) \
         JOIN pg_catalog.pg_trigger t ON p.policy_oid = t.oid",
        "SELECT p.oid FROM pg_catalog.pg_policy AS p(policy_oid), pg_catalog.pg_trigger t \
         WHERE p.policy_oid = t.oid",
        "SELECT oid FROM pg_catalog.pg_policy AS p(policy_oid)",
        "SELECT c.oid FROM evil.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
        "SELECT c.oid FROM evil.pg_class c, pg_catalog.pg_namespace n WHERE n.oid = c.relnamespace",
        "SELECT polname FROM pg_catalog.pg_policy \
         WHERE string_agg(polname, ',') = ''",
        "SELECT oid, count(*) FROM pg_catalog.pg_policy",
        "SELECT polname FROM pg_catalog.pg_policy GROUP BY oid",
        "SELECT pg_catalog.format_type(DISTINCT atttypid, atttypmod) \
         FROM pg_catalog.pg_attribute",
        "SELECT pg_catalog.pg_get_userbyid(DISTINCT relowner) \
         FROM pg_catalog.pg_class",
        "SELECT oid FROM pg_catalog.pg_class \
         WHERE pg_catalog.pg_table_is_visible(DISTINCT oid)",
    ] {
        let error = engine
            .execute_resident_expr_select_sql(sql)
            .expect_err("invalid empty-catalog SQL must fail during binding")
            .to_string();
        assert!(
            !error.contains("GPU execution is required"),
            "`{sql}` reached device execution instead of failing closed: {error}"
        );
    }

    for (sql, expected) in [
        (
            "SELECT 'bad'::text::int4 FROM pg_catalog.pg_policy",
            "invalid input syntax for an integer cast",
        ),
        (
            "SELECT 'definitely_missing'::text::regtype FROM pg_catalog.pg_policy",
            "does not exist for regtype cast",
        ),
        (
            "SELECT array_to_string(polname, ',') FROM pg_catalog.pg_policy",
            "not a modeled array source",
        ),
        (
            "SELECT array_upper(p.prattrs, 1) FROM pg_catalog.pg_policy AS p(prattrs)",
            "not a modeled array source",
        ),
        (
            "SELECT count(oid) WITHIN GROUP (ORDER BY oid) FROM pg_catalog.pg_policy",
            "WITHIN GROUP",
        ),
        (
            "SELECT p.policy_oid FROM pg_catalog.pg_policy AS p(policy_oid) \
             JOIN pg_catalog.pg_trigger t ON p.policy_oid = t.oid",
            "column-alias lists",
        ),
    ] {
        let error = engine
            .execute_resident_expr_select_sql(sql)
            .expect_err("focused catalog binder sabotage must reject")
            .to_string();
        assert!(error.contains(expected), "`{sql}`: {error}");
    }

    for sql in [
        "SELECT c.oid FROM public.pg_class c \
         JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace",
        "SELECT c.oid FROM public.pg_class c, pg_catalog.pg_namespace n \
         WHERE n.oid = c.relnamespace",
    ] {
        let error = engine
            .execute_resident_expr_select_sql(sql)
            .expect_err("explicit public lookup must not fall back to pg_catalog")
            .to_string();
        assert!(
            error.contains("relation \"pg_class\" does not exist"),
            "{sql}: {error}"
        );
        assert!(
            !error.contains("GPU execution is required"),
            "{sql}: {error}"
        );
    }

    let regex = engine
        .execute_resident_expr_select_sql(
            "SELECT nspname FROM pg_catalog.pg_namespace \
             WHERE nspname ~ '^(p.blic)$'",
        )
        .expect_err("regex metacharacters must not be lowered to equality")
        .to_string();
    assert!(
        regex.contains("outside the exact/prefix GPU subset"),
        "{regex}"
    );

    engine
        .execute_text(
            1,
            "CREATE TABLE strict_catalog_binding (id INT, note TEXT DEFAULT 'd')",
        )
        .unwrap();
    for sql in [
        "SELECT relname FROM pg_catalog.pg_class \
         WHERE 'strict_catalog_binding'::regclass[] = oid",
        "SELECT nspname FROM pg_catalog.pg_namespace \
         WHERE 'public'::regnamespace[] = oid",
        "SELECT typname FROM pg_catalog.pg_type WHERE oid::int4[] = oid",
        "SELECT typname FROM pg_catalog.pg_type WHERE oid::int4(3) = oid",
    ] {
        let error = engine
            .execute_resident_expr_select_sql(sql)
            .expect_err("complex casts over nonempty catalogs must fail during binding")
            .to_string();
        assert!(error.contains("complex cast target"), "`{sql}`: {error}");
        assert!(
            !error.contains("GPU execution is required"),
            "`{sql}` reached device execution instead of failing closed: {error}"
        );
    }
    for sql in [
        "SELECT pg_catalog.format_type(a.atttypid, a.definitely_missing) \
         FROM pg_catalog.pg_attribute a",
        "SELECT evil.pg_get_userbyid(c.relowner) FROM pg_catalog.pg_class c",
    ] {
        let error = engine
            .execute_resident_expr_select_sql(sql)
            .expect_err("malformed catalog presentation must fail during binding")
            .to_string();
        assert!(
            !error.contains("GPU execution is required"),
            "`{sql}` reached device execution instead of failing closed: {error}"
        );
    }

    let oid = engine
        .relational_catalog_table("strict_catalog_binding")
        .expect("strict binding table descriptor")
        .oid;
    for sql in [
        format!(
            "SELECT (SELECT d.definitely_missing FROM pg_catalog.pg_attrdef d \
             WHERE a.atthasdef) FROM pg_catalog.pg_attribute a \
             WHERE a.attrelid = '{oid}'"
        ),
        format!(
            "SELECT (SELECT 42 FROM pg_catalog.pg_attrdef d \
             WHERE a.atthasdef OR true) FROM pg_catalog.pg_attribute a \
             WHERE a.attrelid = '{oid}'"
        ),
        format!(
            "SELECT a.atttypid::text FROM pg_catalog.pg_attribute a \
             WHERE a.attrelid = '{oid}'"
        ),
        "SELECT oid FROM evil.pg_class".to_string(),
    ] {
        let error = engine
            .execute_resident_expr_select_sql(&sql)
            .expect_err("unmodeled catalog presentation must fail closed")
            .to_string();
        assert!(
            !error.contains("GPU execution is required"),
            "`{sql}` reached device execution instead of failing closed: {error}"
        );
    }
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn gpu_catalog_audit_repairs_preserve_identity_aliases_casts_arrays_and_literal_patterns() {
    let engine = Engine::new_local_test_engine();

    for sql in [
        "SELECT table_name FROM information_schema.tables ORDER BY table_name",
        "SELECT column_name FROM information_schema.columns ORDER BY column_name",
    ] {
        let result = engine.execute_resident_expr_select_sql(sql).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0), "{sql}");
        assert!(result.rows.is_empty(), "{sql}");
    }

    engine
        .execute_text(100, "CREATE TABLE victim_shadow_probe (id INT)")
        .unwrap();
    engine
        .execute_text(101, "CREATE TABLE pg_attribute (bogus INT)")
        .unwrap();
    engine
        .execute_text(102, "INSERT INTO pg_attribute VALUES (7)")
        .unwrap();
    let shadow_safe = engine
        .execute_resident_expr_select_sql(
            "SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) \
             FROM pg_catalog.pg_attribute a \
             WHERE a.attrelid = 'victim_shadow_probe'::regclass ORDER BY a.attnum",
        )
        .unwrap();
    assert_eq!(shadow_safe.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        shadow_safe.rows,
        vec![vec![SqlValue::Text("integer".to_string())]]
    );
    let unaliased_shadow_safe = engine
        .execute_resident_expr_select_sql(
            "SELECT pg_catalog.format_type(pg_attribute.atttypid, pg_attribute.atttypmod) \
             FROM pg_catalog.pg_attribute \
             WHERE pg_attribute.attrelid = 'victim_shadow_probe'::regclass \
             ORDER BY pg_attribute.attnum",
        )
        .unwrap();
    assert_eq!(unaliased_shadow_safe.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        unaliased_shadow_safe.rows,
        vec![vec![SqlValue::Text("integer".to_string())]]
    );

    let namespace_engine = Engine::new_local_test_engine();
    let public_namespace = namespace_engine
        .execute_resident_expr_select_sql(
            "SELECT nspname FROM pg_catalog.pg_namespace \
             WHERE oid = 'public'::regnamespace",
        )
        .unwrap();
    assert_eq!(public_namespace.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        public_namespace.rows,
        vec![vec![SqlValue::Text("public".to_string())]]
    );
    namespace_engine
        .execute_text(200, "DROP SCHEMA public")
        .unwrap();
    let missing_public = namespace_engine
        .execute_resident_expr_select_sql(
            "SELECT nspname FROM pg_catalog.pg_namespace \
             WHERE oid = 'public'::regnamespace",
        )
        .expect_err("regnamespace must reject a public schema dropped from this snapshot")
        .to_string();
    assert!(
        missing_public.contains("schema \"public\" does not exist for regnamespace cast"),
        "{missing_public}"
    );
    assert!(!missing_public.contains("GPU execution is required"));

    for sql in [
        "SELECT relname FROM pg_catalog.pg_class \
         WHERE 'victim_shadow_probe'::regclass[] = oid",
        "SELECT nspname FROM pg_catalog.pg_namespace \
         WHERE 'public'::regnamespace[] = oid",
        "SELECT typname FROM pg_catalog.pg_type WHERE oid::int4[] = oid",
        "SELECT typname FROM pg_catalog.pg_type WHERE oid::int4(3) = oid",
    ] {
        let error = engine
            .execute_resident_expr_select_sql(sql)
            .expect_err("complex casts must fail before a nonempty catalog device launch")
            .to_string();
        assert!(error.contains("complex cast target"), "`{sql}`: {error}");
        assert!(
            !error.contains("GPU execution is required"),
            "`{sql}` reached device execution instead of failing closed: {error}"
        );
    }

    let public_window = engine
        .execute_resident_expr_select_sql(
            "SELECT bogus, row_number() OVER (ORDER BY bogus) FROM public.pg_attribute",
        )
        .unwrap();
    assert_eq!(public_window.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        public_window.rows,
        vec![vec![SqlValue::Int4(7), SqlValue::Int8(1)]]
    );
    for sql in [
        "SELECT attname, row_number() OVER (ORDER BY attname) FROM pg_catalog.pg_attribute",
        "SELECT bogus, row_number() OVER (ORDER BY bogus) FROM evil.pg_attribute",
        "SELECT bogus, row_number() OVER (ORDER BY bogus) \
         FROM public.pg_attribute AS p(alias_bogus)",
    ] {
        assert!(
            engine.execute_resident_expr_select_sql(sql).is_err(),
            "window RangeVar lookup/alias must fail closed: {sql}"
        );
    }
    let user_alias = engine
        .execute_resident_expr_select_sql(
            "SELECT bogus FROM public.pg_attribute AS p(alias_bogus) WHERE bogus + 0 = 7",
        )
        .expect_err("resident user column-alias lists must not expose hidden source names");
    assert!(
        user_alias.to_string().contains("column-alias"),
        "{user_alias}"
    );

    engine
        .execute_text(
            1,
            "CREATE TABLE public_join_left (id INT PRIMARY KEY, value INT)",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "CREATE TABLE public_join_right (id INT PRIMARY KEY, value INT)",
        )
        .unwrap();
    engine
        .execute_text(3, "INSERT INTO public_join_left VALUES (1, 10), (2, 20)")
        .unwrap();
    engine
        .execute_text(4, "INSERT INTO public_join_right VALUES (1, 100), (3, 300)")
        .unwrap();
    for sql in [
        "SELECT l.value, r.value FROM public.public_join_left l \
         JOIN public.public_join_right r ON l.id = r.id",
        "SELECT l.value, r.value FROM public.public_join_left l, public.public_join_right r \
         WHERE l.id = r.id",
    ] {
        let result = engine.execute_resident_expr_select_sql(sql).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(
            result.rows,
            vec![vec![SqlValue::Int4(10), SqlValue::Int4(100)]],
            "public-qualified join identity diverged for {sql}"
        );
    }

    engine
        .execute_text(
            5,
            r#"CREATE TABLE quoted_star_projection ("*" INT, id INT)"#,
        )
        .unwrap();
    engine
        .execute_text(6, "INSERT INTO quoted_star_projection VALUES (11, 1)")
        .unwrap();
    let star = engine
        .execute_resident_expr_select_sql(
            r#"SELECT "*", * FROM quoted_star_projection WHERE id + 0 = 1"#,
        )
        .unwrap();
    assert_eq!(star.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        star.rows,
        vec![vec![
            SqlValue::Int4(11),
            SqlValue::Int4(11),
            SqlValue::Int4(1),
        ]]
    );

    engine
        .execute_text(7, "CREATE TABLE catalog_regex_match (id INT)")
        .unwrap();
    engine
        .execute_text(8, "CREATE TABLE catalogXregexXwrong (id INT)")
        .unwrap();
    let regex = engine
        .execute_resident_expr_select_sql(
            "SELECT relname FROM pg_catalog.pg_class \
             WHERE relname ~ '^(catalog_regex_.*)$' ORDER BY relname",
        )
        .unwrap();
    assert_eq!(regex.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        regex.rows,
        vec![vec![SqlValue::Text("catalog_regex_match".to_string())]]
    );

    let aliased = engine
        .execute_resident_expr_select_sql(
            "SELECT x FROM pg_catalog.pg_class AS c(x) WHERE x > 0 ORDER BY x LIMIT 1",
        )
        .unwrap();
    assert_eq!(aliased.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(aliased.columns[0].name, "x");
    assert_eq!(aliased.rows.len(), 1);
    assert!(matches!(aliased.rows.row(0), [SqlValue::Int4(_)]));

    let empty_alias = engine
        .execute_resident_expr_select_sql(
            "SELECT policy_oid FROM pg_catalog.pg_policy AS p(policy_oid)",
        )
        .unwrap();
    assert_eq!(empty_alias.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(empty_alias.columns[0].name, "policy_oid");
    assert!(empty_alias.rows.is_empty());

    for alias in [
        "bool",
        "boolean",
        "int2",
        "smallint",
        "int4",
        "integer",
        "int",
        "int8",
        "bigint",
        "numeric",
        "decimal",
        "text",
        "date",
        "timestamp",
        "timestamp without time zone",
        "uuid",
    ] {
        let sql = format!(
            "SELECT '{alias}'::pg_catalog.text::pg_catalog.regtype AS builtin_type \
             FROM pg_catalog.pg_policy"
        );
        let casts = engine.execute_resident_expr_select_sql(&sql).unwrap();
        assert_eq!(casts.executed_target, DeviceTarget::Gpu(0));
        assert!(casts.rows.is_empty());
        assert_eq!(casts.columns[0].ty, SqlType::Int4, "{alias}");
    }

    let arrays = engine
        .execute_resident_expr_select_sql(
            "SELECT array_to_string(polroles, ','), array_upper(polroles, 1) \
             FROM pg_catalog.pg_policy",
        )
        .unwrap();
    assert_eq!(arrays.executed_target, DeviceTarget::Gpu(0));
    assert!(arrays.rows.is_empty());
    assert_eq!(arrays.columns[0].ty, SqlType::Text);
    assert_eq!(arrays.columns[1].ty, SqlType::Int4);

    let renamed_array = engine
        .execute_resident_expr_select_sql(
            "SELECT array_upper(p.roles, 1) \
             FROM pg_catalog.pg_policy AS p(policy_oid, policy_name, policy_cmd, roles)",
        )
        .unwrap();
    assert_eq!(renamed_array.executed_target, DeviceTarget::Gpu(0));
    assert!(renamed_array.rows.is_empty());
    assert_eq!(renamed_array.columns[0].ty, SqlType::Int4);

    let psql_array_subquery = engine
        .execute_resident_expr_select_sql(
            "SELECT array_to_string(ARRAY(SELECT rolname FROM pg_catalog.pg_roles \
             WHERE oid = ANY (p.polroles) ORDER BY 1), ',') \
             FROM pg_catalog.pg_policy p",
        )
        .unwrap();
    assert_eq!(psql_array_subquery.executed_target, DeviceTarget::Gpu(0));
    assert!(psql_array_subquery.rows.is_empty());
    assert_eq!(psql_array_subquery.columns[0].ty, SqlType::Text);
}
