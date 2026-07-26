use super::*;

const PG16_DEFAULT_ACL_QUERY: &str = "SELECT oid, tableoid, defaclrole, defaclnamespace, \
    defaclobjtype, defaclacl, CASE WHEN defaclnamespace = 0 THEN \
    acldefault(CASE WHEN defaclobjtype = 'S' THEN 's'::\"char\" ELSE defaclobjtype END, \
    defaclrole) ELSE '{}' END AS acldefault FROM pg_default_acl";

fn column_privilege_query(null_test: &str) -> String {
    format!(
        "SELECT c.relname, pg_catalog.array_to_string(ARRAY( \
         SELECT attname || E':\\n  ' || pg_catalog.array_to_string(attacl, E'\\n  ') \
         FROM pg_catalog.pg_attribute a \
         WHERE attrelid = c.oid AND NOT attisdropped AND attacl {null_test} \
         ), E'\\n') AS column_privileges \
         FROM pg_catalog.pg_class c \
         LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace \
         WHERE c.relname = 'dump_catalog_probe'"
    )
}

#[test]
fn pg16_dump_catalog_matchers_reject_near_misses_before_device_execution() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE dump_catalog_probe (id INT PRIMARY KEY)")
        .unwrap();
    engine
        .execute_text(2, "CREATE DOMAIN dump_domain_probe AS int4")
        .unwrap();

    let default_acl_near_miss = PG16_DEFAULT_ACL_QUERY.replace("ELSE '{}'", "ELSE '{wrong}'");
    let error = engine
        .execute_resident_expr_select_sql(&default_acl_near_miss)
        .expect_err("a changed pg_dump default-ACL program must miss the prepared route")
        .to_string();
    assert!(!error.contains("GPU execution is required"), "{error}");

    let default_acl_literal_case_miss =
        PG16_DEFAULT_ACL_QUERY.replace("defaclobjtype = 'S'", "defaclobjtype = 's'");
    let error = engine
        .execute_resident_expr_select_sql(&default_acl_literal_case_miss)
        .expect_err("a changed case-sensitive ACL-kind literal must miss the prepared route")
        .to_string();
    assert!(!error.contains("GPU execution is required"), "{error}");

    let error = engine
        .execute_resident_expr_select_sql(&column_privilege_query("IS NULL"))
        .expect_err("the psql column-ACL subquery predicate must match exactly")
        .to_string();
    assert!(
        error.contains("array_to_string has no modeled presentation column"),
        "{error}"
    );

    let changed_acl_projection = column_privilege_query("IS NOT NULL")
        .replace("SELECT attname ||", "SELECT upper(attname) ||");
    let error = engine
        .execute_resident_expr_select_sql(&changed_acl_projection)
        .expect_err("the psql column-ACL subquery projection must match exactly")
        .to_string();
    assert!(
        error.contains("array_to_string has no modeled presentation column"),
        "{error}"
    );

    let changed_acl_indentation = column_privilege_query("IS NOT NULL").replace(":\\n  ", ":\\n ");
    let error = engine
        .execute_resident_expr_select_sql(&changed_acl_indentation)
        .expect_err("the PostgreSQL 16 column-ACL indentation literal must match exactly")
        .to_string();
    assert!(
        error.contains("array_to_string has no modeled presentation column"),
        "{error}"
    );

    for sql in [
        "SELECT rolname FROM pg_catalog.pg_roles WHERE rolname !~ '^pg.'",
        "SELECT oid FROM pg_catalog.pg_roles WHERE oid !~ '^pg_'",
        "SELECT spcname FROM pg_catalog.pg_tablespace WHERE spcname !~ '^pgx_'",
    ] {
        let error = engine
            .execute_resident_expr_select_sql(sql)
            .expect_err("negative catalog regex widening must fail closed")
            .to_string();
        assert!(
            error.contains("negative regular-expression predicates")
                || error.contains("outside the exact/prefix GPU subset")
                || error.contains("does not support"),
            "{sql}: {error}"
        );
    }

    let preserved_side = engine
        .execute_resident_expr_select_sql(
            "SELECT t.typname FROM pg_catalog.pg_type t \
             LEFT JOIN pg_catalog.pg_description d \
             ON d.objoid = t.oid AND t.typtype = 'd'",
        )
        .expect_err("a local ON filter on the preserved outer-join side must be rejected")
        .to_string();
    assert!(preserved_side.contains("preserved"), "{preserved_side}");
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_pg16_dump_and_dumpall_catalog_routes_are_nonvacuous_and_wal_neutral() {
    let engine = Engine::new_local_test_engine();
    for (txn_id, sql) in [
        (1, "CREATE ROLE dump_reader WITH LOGIN"),
        (2, "COMMENT ON ROLE dump_reader IS 'dump reader role'"),
        (
            3,
            "CREATE TABLESPACE dump_space LOCATION '/tmp/gpu-db-dump-space'",
        ),
        (4, "COMMENT ON TABLESPACE dump_space IS 'dump tablespace'"),
        (5, "GRANT CREATE ON TABLESPACE dump_space TO dump_reader"),
        (
            6,
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dump_reader",
        ),
        (
            7,
            "CREATE TABLE dump_catalog_probe (id INT PRIMARY KEY, note TEXT)",
        ),
        (8, "CREATE DOMAIN dump_domain_probe AS int4"),
        (
            9,
            "COMMENT ON DOMAIN dump_domain_probe IS 'domain description'",
        ),
    ] {
        engine.execute_text(txn_id, sql).unwrap();
    }
    let wal_before = engine.durable_wal_records().len();
    let visible_before = engine.visible_up_to();

    let roles = engine
        .execute_resident_expr_select_sql(
            "SELECT oid, rolname, rolsuper, rolinherit, rolcreaterole, rolcreatedb, \
             rolcanlogin, rolconnlimit, rolpassword, rolvaliduntil, rolreplication, \
             rolbypassrls, pg_catalog.shobj_description(oid, 'pg_authid') AS rolcomment, \
             rolname = current_user AS is_current_user FROM pg_roles \
             WHERE rolname !~ '^pg_' ORDER BY 2",
        )
        .unwrap();
    assert_eq!(roles.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(roles.rows.len(), 2);
    assert_eq!(roles.rows[0][1], SqlValue::Text("dump_reader".to_string()));
    assert_eq!(
        roles.rows[0][12],
        SqlValue::Text("dump reader role".to_string())
    );
    assert_eq!(roles.rows[0][13], SqlValue::Bool(false));
    assert_eq!(roles.rows[1][1], SqlValue::Text("postgres".to_string()));
    assert_eq!(roles.rows[1][13], SqlValue::Bool(true));

    let tablespaces = engine
        .execute_resident_expr_select_sql(
            "SELECT oid, spcname, pg_catalog.pg_get_userbyid(spcowner) AS spcowner, \
             pg_catalog.pg_tablespace_location(oid), spcacl, \
             acldefault('t', spcowner) AS acldefault, \
             array_to_string(spcoptions, ', '), \
             pg_catalog.shobj_description(oid, 'pg_tablespace') \
             FROM pg_catalog.pg_tablespace WHERE spcname !~ '^pg_' ORDER BY 1",
        )
        .unwrap();
    assert_eq!(tablespaces.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(tablespaces.rows.len(), 1);
    assert_eq!(
        tablespaces.rows[0][1],
        SqlValue::Text("dump_space".to_string())
    );
    assert_eq!(
        tablespaces.rows[0][3],
        SqlValue::Text("/tmp/gpu-db-dump-space".to_string())
    );
    assert_eq!(
        tablespaces.rows[0][4],
        SqlValue::Text("{dump_reader=C/postgres}".to_string())
    );
    assert_eq!(
        tablespaces.rows[0][7],
        SqlValue::Text("dump tablespace".to_string())
    );

    let defaults = engine
        .execute_resident_expr_select_sql(PG16_DEFAULT_ACL_QUERY)
        .unwrap();
    assert_eq!(defaults.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(defaults.rows.len(), 1);
    assert_eq!(
        defaults.rows[0][5],
        SqlValue::Text("{dump_reader=r/postgres}".to_string())
    );
    assert_eq!(defaults.rows[0][6], SqlValue::Text("{}".to_string()));

    let column_privileges = engine
        .execute_resident_expr_select_sql(&column_privilege_query("IS NOT NULL"))
        .unwrap();
    assert_eq!(column_privileges.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(column_privileges.rows.len(), 1);
    assert_eq!(column_privileges.rows[0][1], SqlValue::Null);

    let domain_comment = engine
        .execute_resident_expr_select_sql(
            "SELECT t.typname, n.nspname, d.description \
             FROM pg_catalog.pg_type t \
             LEFT JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace \
             LEFT JOIN pg_catalog.pg_description d \
             ON d.classoid = 1247 \
             AND d.objoid = t.oid AND d.objsubid = 0 \
             WHERE t.typtype = 'd' AND t.typname = 'dump_domain_probe'",
        )
        .unwrap();
    assert_eq!(domain_comment.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(domain_comment.rows.len(), 1);
    assert_eq!(
        domain_comment.rows[0],
        vec![
            SqlValue::Text("dump_domain_probe".to_string()),
            SqlValue::Text("public".to_string()),
            SqlValue::Text("domain description".to_string()),
        ]
    );

    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.visible_up_to(), visible_before);
}
