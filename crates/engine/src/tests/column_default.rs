use super::*;
use crate::engine_transaction_reset::table_schema_digest;

fn assert_numeric_range(error: ExecuteError) {
    assert!(
        matches!(
            error,
            ExecuteError::Engine(EngineError::NumericValueOutOfRange(_))
        ),
        "expected 22003-shaped numeric error, got {error:?}"
    );
}

fn assert_datatype_mismatch(error: ExecuteError) {
    assert!(
        matches!(
            error,
            ExecuteError::Engine(EngineError::DatatypeMismatch(_))
        ),
        "expected 42804-shaped default assignment error, got {error:?}"
    );
}

fn assert_undefined_relation(error: ExecuteError) {
    assert!(
        matches!(
            error,
            ExecuteError::Engine(EngineError::UndefinedRelation(_))
        ),
        "expected 42P01-shaped missing relation error, got {error:?}"
    );
}

fn assert_duplicate_column(error: ExecuteError) {
    assert!(
        matches!(error, ExecuteError::Engine(EngineError::DuplicateColumn(_))),
        "expected 42701-shaped duplicate column error, got {error:?}"
    );
}

fn select_rows(engine: &Engine, sql: &str) -> RowBlock {
    let Command::Select(select) = parse_command(sql).unwrap() else {
        panic!("expected SELECT");
    };
    engine.execute_relational_select(&select).unwrap().rows
}

fn postgres16_default_assignment(source: SqlType, target: SqlType) -> bool {
    source == target
        || matches!(
            (source, target),
            (
                SqlType::Int2 | SqlType::Int4 | SqlType::Int8,
                SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. }
            ) | (
                SqlType::Numeric { .. },
                SqlType::Int2 | SqlType::Int4 | SqlType::Int8 | SqlType::Numeric { .. }
            ) | (SqlType::Date, SqlType::Timestamp)
                | (SqlType::Timestamp, SqlType::Date)
                | (_, SqlType::Text)
        )
}

#[test]
fn default_assignment_matrix_is_owned_by_the_engine_binder() {
    let numeric = SqlType::Numeric {
        precision: 10,
        scale: 2,
    };
    let types = [
        SqlType::Int2,
        SqlType::Int4,
        SqlType::Int8,
        numeric,
        SqlType::Bool,
        SqlType::Text,
        SqlType::Date,
        SqlType::Timestamp,
        SqlType::Uuid,
    ];
    for source in types {
        for target in types {
            let result = bind_to_column(
                ColumnDefault::DeferredScalar {
                    value: SqlValue::Null,
                    input: DefaultInputType::Explicit(source),
                },
                target,
                "matrix",
            );
            let expected = postgres16_default_assignment(source, target);
            assert_eq!(
                result.is_ok(),
                expected,
                "source={source:?} target={target:?} result={result:?}"
            );
            if !expected {
                assert!(matches!(result, Err(EngineError::DatatypeMismatch(_))));
            }
        }
    }
}

#[test]
fn default_assignment_mismatches_reach_the_engine_as_datatype_mismatch() {
    let engine = Engine::new_local_test_engine();
    for (txn_id, sql) in [
        (
            1,
            "CREATE TABLE mismatch_create_bool (value BOOL DEFAULT 1)",
        ),
        (
            2,
            "CREATE TABLE mismatch_create_null (value INT DEFAULT NULL::text)",
        ),
    ] {
        assert_datatype_mismatch(engine.execute_text(txn_id, sql).unwrap_err());
    }

    engine
        .execute_text(
            3,
            "CREATE TABLE mismatch_alter (id INT, value BOOL, day DATE)",
        )
        .unwrap();
    for (txn_id, sql) in [
        (
            4,
            "ALTER TABLE mismatch_alter ADD COLUMN added_bool BOOL DEFAULT 1",
        ),
        (
            5,
            "ALTER TABLE mismatch_alter ADD COLUMN added_null INT DEFAULT NULL::text",
        ),
        (
            6,
            "ALTER TABLE mismatch_alter ALTER COLUMN value SET DEFAULT 1",
        ),
        (
            7,
            "ALTER TABLE mismatch_alter ALTER COLUMN id SET DEFAULT NULL::text",
        ),
    ] {
        assert_datatype_mismatch(engine.execute_text(txn_id, sql).unwrap_err());
    }

    // Unknown literals remain target-directed input errors rather than assignment mismatches.
    assert!(matches!(
        engine.execute_text(8, "CREATE TABLE target_directed (value INT DEFAULT 'nope')"),
        Err(ExecuteError::Parse(
            ParseError::InvalidTextRepresentation { .. }
        ))
    ));

    // ALTER retains an unknown literal until this binder knows the target.  Its typed input
    // diagnostics must survive that engine boundary instead of collapsing to ApplyFailed.
    assert!(matches!(
        engine.execute_text(
            9,
            "ALTER TABLE mismatch_alter ALTER COLUMN id SET DEFAULT 'not-an-int'"
        ),
        Err(ExecuteError::Engine(
            EngineError::InvalidTextRepresentation(_)
        ))
    ));
    assert!(matches!(
        engine.execute_text(
            10,
            "ALTER TABLE mismatch_alter ALTER COLUMN day SET DEFAULT 'not-a-date'"
        ),
        Err(ExecuteError::Engine(EngineError::InvalidDatetimeFormat(_)))
    ));
    assert!(matches!(
        engine.execute_text(
            11,
            "ALTER TABLE mismatch_alter ALTER COLUMN day SET DEFAULT '2024-02-30'"
        ),
        Err(ExecuteError::Engine(EngineError::DatetimeFieldOverflow(_)))
    ));
    assert!(matches!(
        engine.execute_text(
            12,
            "ALTER TABLE mismatch_alter ALTER COLUMN id SET DEFAULT '2147483648'"
        ),
        Err(ExecuteError::Engine(EngineError::NumericValueOutOfRange(_)))
    ));

    engine
        .execute_text(13, "CREATE SEQUENCE mismatch_nextval_sequence")
        .unwrap();
    // Explicit nextval is parsed independently of a target.  Once its regclass target exists,
    // the engine binder, not the parser, owns its integer-only assignment rule.
    assert_datatype_mismatch(
        engine
            .execute_text(
                14,
                "CREATE TABLE mismatch_nextval (value BOOLEAN DEFAULT nextval('mismatch_nextval_sequence'::regclass))",
            )
            .unwrap_err(),
    );
}

#[test]
fn explicit_sequence_default_target_resolution_precedes_binding_across_ddl() {
    const MISSING: &str = "missing_default_sequence";
    const EXISTING: &str = "existing_default_sequence";
    const NON_SEQUENCE: &str = "non_sequence_default_target";

    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, &format!("CREATE SEQUENCE {EXISTING}"))
        .unwrap();
    engine
        .execute_text(2, &format!("CREATE TABLE {NON_SEQUENCE} (id INT)"))
        .unwrap();
    engine
        .execute_text(3, "CREATE TABLE sequence_default_add (id INT)")
        .unwrap();
    engine
        .execute_text(
            4,
            "CREATE TABLE sequence_default_alter (missing_text TEXT, existing_bool BOOLEAN, missing_int INT, non_sequence_bool BOOLEAN)",
        )
        .unwrap();

    for (txn_id, sql) in [
        (
            5,
            format!(
                "CREATE TABLE sequence_create_missing_text (value TEXT DEFAULT nextval('{MISSING}'::regclass))"
            ),
        ),
        (
            6,
            format!(
                "ALTER TABLE sequence_default_add ADD COLUMN missing_text TEXT DEFAULT nextval('{MISSING}'::regclass)"
            ),
        ),
        (
            7,
            format!(
                "ALTER TABLE sequence_default_alter ALTER COLUMN missing_text SET DEFAULT nextval('{MISSING}'::regclass)"
            ),
        ),
        (
            8,
            format!(
                "CREATE TABLE sequence_create_missing_int (value INT DEFAULT nextval('{MISSING}'::regclass))"
            ),
        ),
        (
            9,
            format!(
                "ALTER TABLE sequence_default_add ADD COLUMN missing_int INT DEFAULT nextval('{MISSING}'::regclass)"
            ),
        ),
        (
            10,
            format!(
                "ALTER TABLE sequence_default_alter ALTER COLUMN missing_int SET DEFAULT nextval('{MISSING}'::regclass)"
            ),
        ),
    ] {
        assert_undefined_relation(engine.execute_text(txn_id, &sql).unwrap_err());
    }

    for (txn_id, sql) in [
        (
            11,
            format!(
                "CREATE TABLE sequence_create_existing_bool (value BOOLEAN DEFAULT nextval('{EXISTING}'::regclass))"
            ),
        ),
        (
            12,
            format!(
                "ALTER TABLE sequence_default_add ADD COLUMN existing_bool BOOLEAN DEFAULT nextval('{EXISTING}'::regclass)"
            ),
        ),
        (
            13,
            format!(
                "ALTER TABLE sequence_default_alter ALTER COLUMN existing_bool SET DEFAULT nextval('{EXISTING}'::regclass)"
            ),
        ),
    ] {
        assert_datatype_mismatch(engine.execute_text(txn_id, &sql).unwrap_err());
    }

    // CREATE validates each default as it appears.  A prior assignment mismatch cannot be
    // overtaken by a later missing regclass, but reversing the columns exposes that 42P01.
    for (txn_id, sql) in [
        (
            17,
            format!(
                "CREATE TABLE sequence_default_order_literal_first (a BOOLEAN DEFAULT 1, b TEXT DEFAULT nextval('{MISSING}'::regclass))"
            ),
        ),
        (
            18,
            format!(
                "CREATE TABLE sequence_default_order_sequence_first (a BOOLEAN DEFAULT nextval('{EXISTING}'::regclass), b TEXT DEFAULT nextval('{MISSING}'::regclass))"
            ),
        ),
    ] {
        assert_datatype_mismatch(engine.execute_text(txn_id, &sql).unwrap_err());
    }
    assert_undefined_relation(
        engine
            .execute_text(
                19,
                &format!(
                    "CREATE TABLE sequence_default_order_missing_first (a TEXT DEFAULT nextval('{MISSING}'::regclass), b BOOLEAN DEFAULT 1)"
                ),
            )
            .unwrap_err(),
    );

    for (txn_id, sql) in [
        (
            14,
            format!(
                "CREATE TABLE sequence_create_non_sequence_bool (value BOOLEAN DEFAULT nextval('{NON_SEQUENCE}'::regclass))"
            ),
        ),
        (
            15,
            format!(
                "ALTER TABLE sequence_default_add ADD COLUMN non_sequence_bool BOOLEAN DEFAULT nextval('{NON_SEQUENCE}'::regclass)"
            ),
        ),
        (
            16,
            format!(
                "ALTER TABLE sequence_default_alter ALTER COLUMN non_sequence_bool SET DEFAULT nextval('{NON_SEQUENCE}'::regclass)"
            ),
        ),
    ] {
        assert_datatype_mismatch(engine.execute_text(txn_id, &sql).unwrap_err());
    }
}

#[test]
fn add_column_structural_errors_precede_default_validation() {
    let engine = Engine::new_local_test_engine();
    let wal_before = engine.durable_wal_records().len();

    assert_undefined_relation(
        engine
            .execute_text(
                20,
                "ALTER TABLE missing_add_default_target ADD COLUMN value BOOLEAN DEFAULT 1",
            )
            .unwrap_err(),
    );

    engine
        .execute_text(21, "CREATE TABLE add_default_target (id INT)")
        .unwrap();
    let wal_after_setup = engine.durable_wal_records().len();
    assert_duplicate_column(
        engine
            .execute_text(
                22,
                "ALTER TABLE add_default_target ADD COLUMN id BOOLEAN DEFAULT 1",
            )
            .unwrap_err(),
    );
    assert_duplicate_column(
        engine
            .execute_text(
                23,
                "ALTER TABLE add_default_target ADD COLUMN id INT DEFAULT nextval('missing_duplicate_default_sequence'::regclass)",
            )
            .unwrap_err(),
    );

    assert_eq!(wal_before + 1, wal_after_setup);
    assert_eq!(engine.durable_wal_records().len(), wal_after_setup);
    assert_eq!(
        engine
            .relational_catalog_table("add_default_target")
            .unwrap()
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        vec!["id"],
    );
}

#[test]
fn decimal_default_with_a_scale_40_tiny_mantissa_rounds_to_zero_through_recovery() {
    const TINY: &str = "0.0000000000000000000000000000000000000005::numeric(3,0)";

    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            &format!("CREATE TABLE tiny_create (id INT, value INT DEFAULT {TINY})"),
        )
        .unwrap();
    engine
        .execute_text(2, "INSERT INTO tiny_create (id) VALUES (1)")
        .unwrap();
    assert_eq!(
        select_rows(&engine, "SELECT id, value FROM tiny_create"),
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(0)]]
    );

    engine
        .execute_text(3, "CREATE TABLE tiny_add (id INT)")
        .unwrap();
    engine
        .execute_text(4, "INSERT INTO tiny_add VALUES (1)")
        .unwrap();
    engine
        .execute_text(
            5,
            &format!("ALTER TABLE tiny_add ADD COLUMN value INT DEFAULT {TINY}"),
        )
        .unwrap();
    assert_eq!(
        select_rows(&engine, "SELECT id, value FROM tiny_add"),
        vec![vec![SqlValue::Int4(1), SqlValue::Int4(0)]]
    );

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    recovered
        .execute_text(6, "INSERT INTO tiny_add (id) VALUES (2)")
        .unwrap();
    assert_eq!(
        select_rows(&recovered, "SELECT id, value FROM tiny_add ORDER BY id"),
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(0)],
            vec![SqlValue::Int4(2), SqlValue::Int4(0)],
        ]
    );
}

#[test]
fn legacy_returning_broadcasts_scalar_defaults_once_per_column_and_skips_unused_defaults() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE default_once (id INT, once_value INT DEFAULT 1.5::numeric(10,1))",
        )
        .unwrap();
    reset_scalar_default_evaluation_count("once_value");
    let result = engine
        .execute_dml_concurrent_with_result(
            2,
            "INSERT INTO default_once (id) VALUES (1), (2), (3) RETURNING id, once_value",
        )
        .unwrap();
    assert_eq!(scalar_default_evaluation_count("once_value"), 1);
    assert_eq!(
        result.returning.unwrap().rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(2)],
            vec![SqlValue::Int4(2), SqlValue::Int4(2)],
            vec![SqlValue::Int4(3), SqlValue::Int4(2)],
        ]
    );

    engine
        .execute_text(
            3,
            "CREATE TABLE returning_precedence_default (id INT, bad INT DEFAULT 999.5::numeric(3,0))",
        )
        .unwrap();
    reset_scalar_default_evaluation_count("bad");
    assert!(matches!(
        engine.execute_dml_concurrent_with_result(
            4,
            "INSERT INTO returning_precedence_default (id) VALUES (1) RETURNING missing",
        ),
        Err(ExecuteError::Engine(EngineError::UndefinedColumn(name))) if name == "missing"
    ));
    assert_eq!(
        scalar_default_evaluation_count("bad"),
        0,
        "RETURNING binding must precede default evaluation after the live typed adapter defers"
    );

    engine
        .execute_text(
            5,
            "CREATE TABLE bypass_bad_default (id INT, bypass_value_9f03 INT DEFAULT 999.5::numeric(3,0))",
        )
        .unwrap();
    reset_scalar_default_evaluation_count("bypass_value_9f03");
    engine
        .execute_text(6, "INSERT INTO bypass_bad_default VALUES (1, 7), (2, 8)")
        .unwrap();
    assert_eq!(scalar_default_evaluation_count("bypass_value_9f03"), 0);
    assert_eq!(
        select_rows(
            &engine,
            "SELECT id, bypass_value_9f03 FROM bypass_bad_default ORDER BY id"
        ),
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(7)],
            vec![SqlValue::Int4(2), SqlValue::Int4(8)],
        ]
    );
}

#[test]
fn per_row_default_evaluator_rejects_scalar_authority() {
    let engine = Engine::new_local_test_engine();
    let error = engine
        .evaluate_column_default_pure(
            &ColumnDefault::Literal(SqlValue::Int4(7)),
            SqlType::Int4,
            "scalar",
            &mut BTreeMap::new(),
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("scalar default reached the per-row sequence evaluator"),
        "{error}"
    );
}

#[test]
fn deferred_default_fails_at_insert_before_wal_and_keeps_engine_live() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE deferred_default (id INT, bad NUMERIC(3,0) DEFAULT 999.5::numeric(3,0))",
        )
        .unwrap();
    let table = engine.relational_catalog_table("deferred_default").unwrap();
    let Some(ColumnDefault::DeferredScalar {
        value: SqlValue::Numeric(raw),
        input:
            DefaultInputType::Explicit(SqlType::Numeric {
                precision: 3,
                scale: 0,
            }),
    }) = table.columns[1].default.as_ref()
    else {
        panic!("expected explicit deferred numeric default");
    };
    assert_eq!(*raw, Decimal128::new(9_995, 1));
    assert_eq!(
        render_expression(table.columns[1].default.as_ref().unwrap()).unwrap(),
        "999.5::numeric(3,0)"
    );
    let wal_before = engine.durable_wal_records().len();
    assert_numeric_range(
        engine
            .execute_text(2, "INSERT INTO deferred_default (id) VALUES (1)")
            .unwrap_err(),
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    let Command::Select(select) = parse_command("SELECT id FROM deferred_default").unwrap() else {
        panic!("expected SELECT");
    };
    assert!(engine
        .execute_relational_select(&select)
        .unwrap()
        .rows
        .is_empty());
    engine
        .execute_text(3, "CREATE TABLE default_followup (id INT DEFAULT 7)")
        .unwrap();
    engine
        .execute_text(4, "INSERT INTO default_followup (id) VALUES (DEFAULT)")
        .unwrap();
}

#[test]
fn deferred_default_source_precedes_target_and_typed_batch_broadcasts() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE default_order (id INT, source_fail TEXT DEFAULT 999.5::numeric(3,0), target_fail SMALLINT DEFAULT 32767.5::numeric(10,1), rounded INT DEFAULT 1.5::numeric(10,1))",
        )
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    let source = engine
        .execute_text(2, "INSERT INTO default_order (id) VALUES (1)")
        .unwrap_err();
    assert!(
        source.to_string().contains("numeric field overflow"),
        "{source}"
    );
    assert_numeric_range(source);
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    engine
        .execute_text(
            3,
            "CREATE TABLE target_order (id INT, bad SMALLINT DEFAULT 32767.5::numeric(10,1))",
        )
        .unwrap();
    let target = engine
        .execute_text(4, "INSERT INTO target_order (id) VALUES (1)")
        .unwrap_err();
    assert!(
        target.to_string().contains("smallint out of range"),
        "{target}"
    );
    assert_numeric_range(target);

    engine
        .execute_text(
            5,
            "CREATE TABLE integer_target (id INT DEFAULT 2147483647.5::numeric(11,1))",
        )
        .unwrap();
    let integer_target = engine
        .execute_text(6, "INSERT INTO integer_target (id) VALUES (DEFAULT)")
        .unwrap_err();
    assert!(
        integer_target.to_string().contains("integer out of range"),
        "{integer_target}"
    );
    assert_numeric_range(integer_target);

    engine
        .execute_text(
            7,
            "CREATE TABLE typed_broadcast (id INT, rounded INT DEFAULT 1.5::numeric(10,1))",
        )
        .unwrap();
    engine
        .execute_text(8, "INSERT INTO typed_broadcast (id) VALUES (1), (2), (3)")
        .unwrap();
    let Command::Select(select) =
        parse_command("SELECT rounded FROM typed_broadcast ORDER BY id").unwrap()
    else {
        panic!("expected SELECT");
    };
    assert_eq!(
        engine.execute_relational_select(&select).unwrap().rows,
        vec![
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(2)],
        ]
    );
}

#[test]
fn add_column_default_is_eager_before_wal_and_preserves_expression_for_recovery() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE add_default (id INT)")
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    assert_numeric_range(
        engine
            .execute_text(
                2,
                "ALTER TABLE add_default ADD COLUMN bad NUMERIC(3,0) DEFAULT 999.5",
            )
            .unwrap_err(),
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(
        engine
            .relational_catalog_table("add_default")
            .unwrap()
            .columns
            .len(),
        1
    );

    engine
        .execute_text(3, "CREATE TABLE add_default_nonempty (id INT)")
        .unwrap();
    engine
        .execute_text(4, "INSERT INTO add_default_nonempty (id) VALUES (1)")
        .unwrap();
    let nonempty_wal_before = engine.durable_wal_records().len();
    assert_numeric_range(
        engine
            .execute_text(
                5,
                "ALTER TABLE add_default_nonempty ADD COLUMN bad NUMERIC(3,0) DEFAULT 999.5",
            )
            .unwrap_err(),
    );
    assert_eq!(engine.durable_wal_records().len(), nonempty_wal_before);
    assert_eq!(
        engine
            .relational_catalog_table("add_default_nonempty")
            .unwrap()
            .columns
            .len(),
        1
    );

    engine
        .execute_text(
            6,
            "ALTER TABLE add_default ADD COLUMN rounded INT DEFAULT 1.5::numeric(10,1)",
        )
        .unwrap();
    let durable = engine.durable_wal_records();
    let recovered = Engine::recover_from_durable_wal(&durable).unwrap();
    let default = &recovered
        .relational_catalog_table("add_default")
        .unwrap()
        .columns[1]
        .default;
    assert!(matches!(
        default,
        Some(ColumnDefault::DeferredScalar {
            input: DefaultInputType::Explicit(SqlType::Numeric {
                precision: 10,
                scale: 1
            }),
            ..
        })
    ));
    recovered
        .execute_text(7, "INSERT INTO add_default (id) VALUES (1)")
        .unwrap();
}

#[test]
fn deferred_default_catalog_rendering_and_digest_preserve_explicit_source_type() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE default_catalog (id INT DEFAULT 999.5::numeric(3,0))",
        )
        .unwrap();
    let table = engine.relational_catalog_table("default_catalog").unwrap();
    let default = table.columns[0].default.as_ref().unwrap();
    assert_eq!(render_expression(default).unwrap(), "999.5::numeric(3,0)");
    assert_eq!(
        render_expression(&ColumnDefault::DeferredScalar {
            value: SqlValue::Text("typed".to_string()),
            input: DefaultInputType::Explicit(SqlType::Text),
        })
        .unwrap(),
        "'typed'::text"
    );
    let original = table_schema_digest(&table).unwrap();
    let mut changed = table.clone();
    changed.columns[0].default = Some(ColumnDefault::DeferredScalar {
        value: SqlValue::Numeric(Decimal128::new(9_995, 1)),
        input: DefaultInputType::Explicit(SqlType::Numeric {
            precision: 4,
            scale: 0,
        }),
    });
    assert_ne!(original, table_schema_digest(&changed).unwrap());
}

#[test]
fn alter_default_stays_deferred_through_recovery_and_legacy_literal_json_stays_stable() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(1, "CREATE TABLE alter_default (id INT, value INT)")
        .unwrap();
    engine
        .execute_text(
            2,
            "ALTER TABLE alter_default ALTER COLUMN value SET DEFAULT 999.5::numeric(3,0)",
        )
        .unwrap();
    let table = engine.relational_catalog_table("alter_default").unwrap();
    assert!(matches!(
        table.columns[1].default,
        Some(ColumnDefault::DeferredScalar {
            input: DefaultInputType::Explicit(SqlType::Numeric {
                precision: 3,
                scale: 0
            }),
            ..
        })
    ));
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    let wal_before = recovered.durable_wal_records().len();
    assert_numeric_range(
        recovered
            .execute_text(3, "INSERT INTO alter_default (id) VALUES (1)")
            .unwrap_err(),
    );
    assert_eq!(recovered.durable_wal_records().len(), wal_before);

    let legacy = ColumnDefault::Literal(SqlValue::Int4(7));
    let bytes = serde_json::to_vec(&legacy).unwrap();
    assert_eq!(bytes, br#"{"Literal":{"Int4":7}}"#);
    let decoded: ColumnDefault = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded, legacy);
    assert_eq!(
        evaluate_scalar(&decoded, SqlType::Int4, "legacy").unwrap(),
        SqlValue::Int4(7)
    );
}
