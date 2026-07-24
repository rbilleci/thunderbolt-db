use super::*;

fn assert_gpu(result: &RelationalSelectResult) {
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn pg16_dump_typed_plans_apply_relational_semantics_on_device() {
    let engine = Engine::new_local_test_engine();
    for (txn_id, sql) in [
        (
            1,
            "CREATE TABLE dump_plan_a (id INT PRIMARY KEY, amount INT DEFAULT 7)",
        ),
        (
            2,
            "CREATE TABLE dump_plan_b (id INT PRIMARY KEY, a_id INT, note INT)",
        ),
        (
            3,
            "ALTER TABLE ONLY dump_plan_b ADD CONSTRAINT dump_plan_b_a_fk \
             FOREIGN KEY (a_id) REFERENCES dump_plan_a(id)",
        ),
        (4, "CREATE INDEX dump_plan_b_note_idx ON dump_plan_b(note)"),
        (
            5,
            "CREATE VIEW dump_plan_view AS SELECT id, amount FROM dump_plan_a",
        ),
        (
            6,
            "CREATE MATERIALIZED VIEW dump_plan_mv AS SELECT * FROM dump_plan_a WITH DATA",
        ),
        (7, "CREATE SEQUENCE dump_plan_seq"),
        (8, "CREATE DOMAIN dump_plan_domain AS int4"),
        (
            9,
            "CREATE FUNCTION dump_plan_function() RETURNS int4 LANGUAGE sql AS 'SELECT 42'",
        ),
        (10, "CREATE PUBLICATION dump_plan_pub FOR TABLE dump_plan_a"),
        (
            11,
            "CREATE SUBSCRIPTION dump_plan_sub CONNECTION 'host=localhost dbname=postgres' \
             PUBLICATION dump_plan_pub WITH (connect = false, enabled = false)",
        ),
        (12, "CREATE DATABASE dump_plan_other_database"),
        (
            13,
            "CREATE SUBSCRIPTION dump_plan_sub_two CONNECTION 'host=localhost dbname=postgres' \
             PUBLICATION dump_plan_pub WITH (connect = false, enabled = false)",
        ),
    ] {
        engine
            .execute_text(txn_id, sql)
            .unwrap_or_else(|error| panic!("fixture statement failed: {sql}: {error}"));
    }
    let catalog = engine.catalog_snapshot();
    let a_oid = catalog.relational_catalog["dump_plan_a"].oid;
    let b_oid = catalog.relational_catalog["dump_plan_b"].oid;
    let sequence_oid = catalog.relational_sequences["dump_plan_seq"].oid;
    let domain_oid = catalog.relational_domains["dump_plan_domain"].oid;
    let function_oid = catalog.relational_functions["dump_plan_function"].oid;

    let classes = engine.execute_pg16_dump_class_metadata().unwrap();
    assert_gpu(&classes);
    assert!(classes
        .rows
        .iter()
        .all(|row| row[4] != SqlValue::Text("i".to_string())));
    assert!(classes
        .rows
        .iter()
        .any(|row| row[1] == SqlValue::Int4(a_oid as i32)));
    assert!(classes
        .rows
        .iter()
        .any(|row| row[1] == SqlValue::Int4(b_oid as i32)));

    let attributes = engine
        .execute_pg16_dump_attribute_metadata(&[a_oid])
        .unwrap();
    assert_gpu(&attributes);
    assert!(!attributes.rows.is_empty());
    assert!(attributes
        .rows
        .iter()
        .all(|row| row[0] == SqlValue::Int4(a_oid as i32)));

    let defaults = engine.execute_pg16_dump_attrdef_metadata(&[a_oid]).unwrap();
    assert_gpu(&defaults);
    assert_eq!(defaults.rows.len(), 1);
    assert_eq!(defaults.rows[0][2], SqlValue::Int4(a_oid as i32));
    let no_defaults = engine.execute_pg16_dump_attrdef_metadata(&[b_oid]).unwrap();
    assert_gpu(&no_defaults);
    assert!(no_defaults.rows.is_empty());

    let indexes = engine.execute_pg16_dump_index_metadata(&[b_oid]).unwrap();
    assert_gpu(&indexes);
    assert!(!indexes.rows.is_empty());
    assert!(indexes
        .rows
        .iter()
        .all(|row| row[2] == SqlValue::Int4(b_oid as i32)));
    assert!(indexes
        .rows
        .iter()
        .all(|row| { row[3] != SqlValue::Text("dump_plan_a_pkey".to_string()) }));

    let foreign_keys = engine
        .execute_pg16_dump_foreign_key_metadata(&[b_oid])
        .unwrap();
    assert_gpu(&foreign_keys);
    assert_eq!(foreign_keys.rows.len(), 1);
    assert_eq!(foreign_keys.rows[0][2], SqlValue::Int4(b_oid as i32));
    let unrelated_foreign_keys = engine
        .execute_pg16_dump_foreign_key_metadata(&[a_oid])
        .unwrap();
    assert_gpu(&unrelated_foreign_keys);
    assert!(unrelated_foreign_keys.rows.is_empty());

    let subscriptions = engine.execute_pg16_dump_subscription_count().unwrap();
    assert_gpu(&subscriptions);
    assert_eq!(subscriptions.rows, vec![vec![SqlValue::Int8(2)]]);

    let empty_engine = Engine::new_local_test_engine();
    let no_subscriptions = empty_engine.execute_pg16_dump_subscription_count().unwrap();
    assert_gpu(&no_subscriptions);
    assert_eq!(no_subscriptions.rows, vec![vec![SqlValue::Int8(0)]]);

    let sequence = engine
        .execute_pg16_dump_sequence_metadata(sequence_oid)
        .unwrap();
    assert_gpu(&sequence);
    assert_eq!(sequence.rows.len(), 1);
    let missing_sequence = engine
        .execute_pg16_dump_sequence_metadata(sequence_oid + 1_000_000)
        .unwrap();
    assert_gpu(&missing_sequence);
    assert!(missing_sequence.rows.is_empty());

    let dependencies = engine.execute_pg16_dump_dependencies().unwrap();
    assert_gpu(&dependencies);
    assert!(dependencies.rows.iter().any(|row| {
        row[1] == SqlValue::Int4(catalog.relational_views["dump_plan_view"].oid as i32)
            && row[3] == SqlValue::Int4(a_oid as i32)
    }));
    assert!(dependencies.rows.iter().any(|row| {
        row[1] == SqlValue::Int4(catalog.relational_materialized_views["dump_plan_mv"].oid as i32)
            && row[3] == SqlValue::Int4(a_oid as i32)
    }));

    let domain = engine
        .execute_prepared_catalog_program(&PreparedCatalogProgram::Pg16DomainDefinition {
            type_oid: SqlValue::Int4(domain_oid as i32),
        })
        .unwrap();
    assert_gpu(&domain);
    assert_eq!(domain.rows.len(), 1);
    let missing_domain = engine
        .execute_prepared_catalog_program(&PreparedCatalogProgram::Pg16DomainDefinition {
            type_oid: SqlValue::Int4(domain_oid as i32 + 1_000_000),
        })
        .unwrap();
    assert_gpu(&missing_domain);
    assert!(missing_domain.rows.is_empty());
    let function = engine
        .execute_prepared_catalog_program(&PreparedCatalogProgram::Pg16FunctionDefinition {
            function_oid: SqlValue::Int4(function_oid as i32),
        })
        .unwrap();
    assert_gpu(&function);
    assert_eq!(function.rows.len(), 1);
    let missing_function = engine
        .execute_prepared_catalog_program(&PreparedCatalogProgram::Pg16FunctionDefinition {
            function_oid: SqlValue::Int4(function_oid as i32 + 1_000_000),
        })
        .unwrap();
    assert_gpu(&missing_function);
    assert!(missing_function.rows.is_empty());

    let materialized_dependencies = engine
        .execute_prepared_catalog_program(&PreparedCatalogProgram::Pg16MaterializedViewDependencies)
        .unwrap();
    assert_gpu(&materialized_dependencies);
    assert!(materialized_dependencies.rows.is_empty());
    let invalid_materialized_dependency = engine
        .execute_text(
            14,
            "CREATE MATERIALIZED VIEW dump_plan_invalid_mv AS \
             SELECT * FROM dump_plan_view WITH DATA",
        )
        .expect_err("DDL must preserve the authoritative-empty materialized dependency invariant");
    assert!(
        invalid_materialized_dependency
            .to_string()
            .contains("materialized views over views are unsupported"),
        "{invalid_materialized_dependency}"
    );

    let types = engine.execute_pg16_dump_type_metadata().unwrap();
    assert_gpu(&types);
    assert!(types
        .rows
        .iter()
        .any(|row| row[1] == SqlValue::Int4(domain_oid as i32)));
    let languages = engine.execute_pg16_dump_language_metadata().unwrap();
    assert_gpu(&languages);
    assert_eq!(languages.rows.len(), 1);
    assert_eq!(languages.rows[0][2], SqlValue::Text("plpgsql".to_string()));

    let database = engine.execute_pg16_dump_database_metadata().unwrap();
    assert_gpu(&database);
    assert_eq!(database.rows.len(), 1);
    assert_eq!(database.rows[0][2], SqlValue::Text("postgres".to_string()));
}
