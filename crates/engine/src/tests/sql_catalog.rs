use super::*;

mod relation_lifecycle;

#[test]
fn where_equality_coerces_literal_across_the_numeric_tower() {
    // Blocker regression: `WHERE numeric_col = <int literal>` (and the integral-numeric
    // reverse) must match via PostgreSQL's implicit cross-type coercion, not silently
    // miss — both the in-memory predicate and the equality value-index probe.
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE acct (id INT, bal NUMERIC(10,2))")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO acct (id, bal) VALUES (5, 100.00), (6, 1.50), (7, 2.00)",
    )
    .unwrap();
    let run = |e: &Engine, sql: &str| {
        let Command::Select(s) = parse_command(sql).unwrap() else {
            panic!("expected SELECT");
        };
        e.execute_relational_select(&s).map(|result| result.rows)
    };

    // numeric column = bare-int literal (the reported blocker) and = different-scale numeric.
    assert_eq!(
        run(&e, "SELECT id FROM acct WHERE bal = 100").unwrap(),
        vec![vec![SqlValue::Int4(5)]]
    );
    assert_eq!(
        run(&e, "SELECT id FROM acct WHERE bal = 1.5").unwrap(),
        vec![vec![SqlValue::Int4(6)]]
    );
    // integer column = integral numeric literal matches. The fractional shape is not a supported GPU
    // route, so production declines loudly; the closed-form predicate specification separately owns
    // the semantic result that 7 does not equal 7.5.
    assert_eq!(
        run(&e, "SELECT id FROM acct WHERE id = 7.0").unwrap(),
        vec![vec![SqlValue::Int4(7)]]
    );
    let fallback_before = e.metrics().snapshot().fallback_total;
    let error = run(&e, "SELECT id FROM acct WHERE id = 7.5").unwrap_err();
    assert!(error
        .to_string()
        .contains("GPU execution is required for SELECT on relation \"acct\""));
    assert_eq!(e.metrics().snapshot().fallback_total, fallback_before);
    let fractional =
        coerce_filter_literal(SqlValue::Numeric(Decimal128::new(75, 1)), SqlType::Int4);
    let specification_rows = [SqlValue::Int4(7)]
        .into_iter()
        .filter(|value| select_filter_matches(value, SelectFilterOp::Eq, &fractional))
        .collect::<Vec<_>>();
    assert!(
        specification_rows.is_empty(),
        "the closed-form predicate specification must preserve the empty semantic result"
    );
    // Equality is now consistent with the ordering ops across the int/numeric boundary.
    assert_eq!(
        run(&e, "SELECT id FROM acct WHERE bal >= 2 ORDER BY id").unwrap(),
        vec![vec![SqlValue::Int4(5)], vec![SqlValue::Int4(7)]]
    );
    // DELETE coerces the same way (parity with SELECT, not a type error).
    e.execute_text(3, "DELETE FROM acct WHERE bal = 100")
        .unwrap();
    assert_eq!(
        run(&e, "SELECT id FROM acct ORDER BY id").unwrap(),
        vec![vec![SqlValue::Int4(6)], vec![SqlValue::Int4(7)]]
    );
}

#[test]
fn coerce_filter_literal_spans_the_integer_numeric_tower() {
    let num = SqlType::Numeric {
        precision: 10,
        scale: 2,
    };
    // int -> numeric / int8, and the integral-numeric -> int reverses.
    assert_eq!(
        coerce_filter_literal(SqlValue::Int4(5), num),
        SqlValue::Numeric(Decimal128::new(5, 0))
    );
    assert_eq!(
        coerce_filter_literal(SqlValue::Int4(5), SqlType::Int8),
        SqlValue::Int8(5)
    );
    assert_eq!(
        coerce_filter_literal(SqlValue::Int8(5), num),
        SqlValue::Numeric(Decimal128::new(5, 0))
    );
    assert_eq!(
        coerce_filter_literal(SqlValue::Numeric(Decimal128::new(700, 2)), SqlType::Int4),
        SqlValue::Int4(7)
    );
    // No implicit cast / out of range: returned unchanged (then compares unequal).
    assert_eq!(
        coerce_filter_literal(SqlValue::Numeric(Decimal128::new(75, 1)), SqlType::Int4),
        SqlValue::Numeric(Decimal128::new(75, 1))
    );
    assert_eq!(
        coerce_filter_literal(SqlValue::Int8(5_000_000_000), SqlType::Int4),
        SqlValue::Int8(5_000_000_000)
    );
    // Same-type and unrelated types pass through untouched.
    assert_eq!(
        coerce_filter_literal(SqlValue::Int4(9), SqlType::Int4),
        SqlValue::Int4(9)
    );
    assert_eq!(
        coerce_filter_literal(SqlValue::Text("x".into()), num),
        SqlValue::Text("x".into())
    );
}

#[test]
fn insert_and_update_widen_literals_across_the_numeric_tower() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE acct (id INT, bal NUMERIC(10,2), big BIGINT)",
    )
    .unwrap();
    // Bare-int literals populate the numeric and bigint columns (PG implicit cast);
    // before this fix they errored "invalid value for column".
    e.execute_text(2, "INSERT INTO acct (id, bal, big) VALUES (1, 100, 5)")
        .unwrap();
    let run = |e: &Engine, sql: &str| {
        let Command::Select(s) = parse_command(sql).unwrap() else {
            panic!("expected SELECT");
        };
        e.execute_relational_select(&s).unwrap().rows
    };
    // The numeric is stored at the column scale (100 -> 100.00), the bigint as int8.
    assert_eq!(
        run(&e, "SELECT bal, big FROM acct WHERE id = 1"),
        vec![vec![
            SqlValue::Numeric(Decimal128::parse("100.00").unwrap()),
            SqlValue::Int8(5)
        ]]
    );
    // UPDATE coerces + rescales the same way (was a type error + missing rescale before).
    e.execute_text(3, "UPDATE acct SET bal = 7 WHERE id = 1")
        .unwrap();
    assert_eq!(
        run(&e, "SELECT bal FROM acct WHERE id = 1"),
        vec![vec![SqlValue::Numeric(Decimal128::parse("7.00").unwrap())]]
    );
    // Precision overflow on a widened int still errors (numeric(10,2) holds 8 integer digits).
    assert!(e
        .execute_text(4, "INSERT INTO acct (id, bal) VALUES (2, 123456789)")
        .is_err());
    // A genuinely incompatible type still errors loudly — no silent coercion.
    assert!(e
        .execute_text(5, "INSERT INTO acct (id, bal) VALUES (3, 'x')")
        .is_err());
}

#[test]
fn column_defaults_coerce_cross_type_literals() {
    let e = Engine::new_local_test_engine();
    // Cross-type DEFAULT literals (int -> numeric / int8) are accepted at CREATE and
    // stored at the column type/scale; before this they errored "invalid default".
    e.execute_text(
        1,
        "CREATE TABLE t (id INT, bal NUMERIC(10,2) DEFAULT 0, big BIGINT DEFAULT 7)",
    )
    .unwrap();
    let run = |e: &Engine, sql: &str| {
        let Command::Select(s) = parse_command(sql).unwrap() else {
            panic!("expected SELECT");
        };
        e.execute_relational_select(&s).unwrap().rows
    };
    // INSERT omitting the defaulted columns materializes the defaults at the column type.
    e.execute_text(2, "INSERT INTO t (id) VALUES (1)").unwrap();
    assert_eq!(
        run(&e, "SELECT bal, big FROM t WHERE id = 1"),
        vec![vec![
            SqlValue::Numeric(Decimal128::parse("0.00").unwrap()),
            SqlValue::Int8(7)
        ]]
    );
    // ALTER ... SET DEFAULT with a cross-type literal is accepted and applied.
    e.execute_text(3, "ALTER TABLE t ALTER COLUMN bal SET DEFAULT 5")
        .unwrap();
    e.execute_text(4, "INSERT INTO t (id) VALUES (2)").unwrap();
    assert_eq!(
        run(&e, "SELECT bal FROM t WHERE id = 2"),
        vec![vec![SqlValue::Numeric(Decimal128::parse("5.00").unwrap())]]
    );
    // ADD COLUMN with a cross-type default backfills existing rows at the column type.
    e.execute_text(5, "ALTER TABLE t ADD COLUMN tax NUMERIC(10,2) DEFAULT 1")
        .unwrap();
    assert_eq!(
        run(&e, "SELECT tax FROM t WHERE id = 1"),
        vec![vec![SqlValue::Numeric(Decimal128::parse("1.00").unwrap())]]
    );
    // A genuinely incompatible default still errors loudly — no silent coercion.
    assert!(e
        .execute_text(6, "CREATE TABLE bad (x NUMERIC(10,2) DEFAULT 'oops')")
        .is_err());
}

#[test]
fn engine_answers_single_relation_pg_catalog_queries() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE orders (id INT)").unwrap();
    // The catalog-aware parse entry carries pg_catalog/information_schema qualifiers;
    // the strict parse_command (legacy server) keeps rejecting them.
    let run = |e: &Engine, sql: &str| {
        let Command::Select(s) = parse_command_allowing_catalog(sql).unwrap() else {
            panic!("expected SELECT");
        };
        e.execute_relational_select(&s).unwrap().rows
    };
    // pg_namespace synthesizes the public schema (queried via the normal WHERE path).
    assert_eq!(
        run(
            &e,
            "SELECT nspname FROM pg_namespace WHERE nspname = 'public'"
        ),
        vec![vec![SqlValue::Text("public".to_string())]]
    );
    // pg_class: both tables, resolvable BARE (implicit pg_catalog search path) with
    // WHERE + ORDER BY routed through the standard relational SELECT machinery.
    assert_eq!(
        run(
            &e,
            "SELECT relname FROM pg_class WHERE relkind = 'r' ORDER BY relname"
        ),
        vec![
            vec![SqlValue::Text("orders".to_string())],
            vec![SqlValue::Text("people".to_string())],
        ]
    );
    // ...and QUALIFIED (pg_catalog.pg_class), which the parser now carries to the engine.
    assert_eq!(
        run(
            &e,
            "SELECT relname FROM pg_catalog.pg_class WHERE relname = 'people'"
        ),
        vec![vec![SqlValue::Text("people".to_string())]]
    );
    // relnatts/relkind are synthesized from the live catalog (people has 2 columns).
    assert_eq!(
        run(&e, "SELECT relnatts FROM pg_class WHERE relname = 'people'"),
        vec![vec![SqlValue::Int4(2)]]
    );
    // Aggregates reuse the same path.
    assert_eq!(
        run(&e, "SELECT COUNT(*) FROM pg_class WHERE relkind = 'r'"),
        vec![vec![SqlValue::Int8(2)]]
    );
    // A real user table always shadows a catalog name; an unknown catalog relation errors.
    assert_eq!(
        run(&e, "SELECT id FROM people WHERE id = 0"),
        Vec::<Vec<SqlValue>>::new()
    );
    let Command::Select(bad) =
        parse_command_allowing_catalog("SELECT * FROM pg_catalog.pg_does_not_exist").unwrap()
    else {
        panic!("expected SELECT");
    };
    assert!(e.execute_relational_select(&bad).is_err());
}

#[test]
fn engine_answers_pg_attribute_pg_type_and_information_schema() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE people (id INT, name TEXT, bal NUMERIC(10,2))",
    )
    .unwrap();
    let run = |e: &Engine, sql: &str| {
        let Command::Select(s) = parse_command_allowing_catalog(sql).unwrap() else {
            panic!("expected SELECT");
        };
        e.execute_relational_select(&s).unwrap().rows
    };
    // pg_attribute: the table's columns with their type OIDs (numeric=1700/int4=23/text=25).
    assert_eq!(
        run(
            &e,
            "SELECT attname, atttypid FROM pg_attribute ORDER BY attname"
        ),
        vec![
            vec![SqlValue::Text("bal".to_string()), SqlValue::Int4(1700)],
            vec![SqlValue::Text("id".to_string()), SqlValue::Int4(23)],
            vec![SqlValue::Text("name".to_string()), SqlValue::Int4(25)],
        ]
    );
    // pg_type: the fixed base types (pg_type spells them int8, not bigint).
    assert_eq!(
        run(
            &e,
            "SELECT oid, typlen FROM pg_type WHERE typname = 'numeric'"
        ),
        vec![vec![SqlValue::Int4(1700), SqlValue::Int4(-1)]]
    );
    assert_eq!(
        run(&e, "SELECT oid FROM pg_type WHERE typname = 'int8'"),
        vec![vec![SqlValue::Int4(20)]]
    );
    // information_schema.tables (qualified; not in the implicit search path).
    assert_eq!(
            run(
                &e,
                "SELECT table_schema, table_type FROM information_schema.tables WHERE table_name = 'people'"
            ),
            vec![vec![
                SqlValue::Text("public".to_string()),
                SqlValue::Text("BASE TABLE".to_string())
            ]]
        );
    // information_schema.columns: SQL-standard data_type names, ordered by position.
    assert_eq!(
            run(
                &e,
                "SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = 'people' ORDER BY ordinal_position"
            ),
            vec![
                vec![
                    SqlValue::Text("id".to_string()),
                    SqlValue::Text("integer".to_string()),
                    SqlValue::Text("YES".to_string())
                ],
                vec![
                    SqlValue::Text("name".to_string()),
                    SqlValue::Text("text".to_string()),
                    SqlValue::Text("YES".to_string())
                ],
                vec![
                    SqlValue::Text("bal".to_string()),
                    SqlValue::Text("numeric".to_string()),
                    SqlValue::Text("YES".to_string())
                ],
            ]
        );
    // The text -> rows catalog entry resolves the same relations.
    assert_eq!(
        e.execute_relational_select_text(
            "SELECT table_type FROM information_schema.tables WHERE table_name = 'people'"
        )
        .unwrap()
        .rows,
        vec![vec![SqlValue::Text("BASE TABLE".to_string())]]
    );
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_equality_predicate_and_limit_push_down_to_gpu_bridge() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (2, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT * FROM people WHERE id = 2 LIMIT 1").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())]]
    );
    assert_eq!(result.execution.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "id".to_string(),
            matched_keys: 2,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_range_predicate_uses_filtered_key_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) =
        parse_command("SELECT name FROM people WHERE id >= 3 LIMIT 1").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Text("Grace".to_string())]]
    );
    assert_eq!(result.execution.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::FilteredKeyBatch {
            table: "people".to_string(),
            predicate_column: "id".to_string(),
            predicate_op: SelectFilterOp::Gte,
            matched_keys: 2,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_range_predicate_with_order_uses_ordered_key_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE id > 1 ORDER BY name DESC LIMIT 2").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(4)]]
    );
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("id".to_string()),
            predicate_op: Some(SelectFilterOp::Gt),
            order_column: "name".to_string(),
            descending: true,
            matched_keys: 3,
        }
    );
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_and_predicates_use_conjunctive_key_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE id >= 3 AND name = 'Grace' LIMIT 1").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(result.specification.rows, vec![vec![SqlValue::Int4(3)]]);
    assert_eq!(result.execution.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::ConjunctiveFilteredKeyBatch {
            table: "people".to_string(),
            predicate_count: 2,
            matched_keys: 2,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_and_predicates_with_order_use_ordered_key_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) = parse_command(
        "SELECT id FROM people WHERE id >= 2 AND name = 'Grace' ORDER BY id DESC LIMIT 1",
    )
    .unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(result.specification.rows, vec![vec![SqlValue::Int4(4)]]);
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("<conjunction>".to_string()),
            predicate_op: None,
            order_column: "id".to_string(),
            descending: true,
            matched_keys: 2,
        }
    );
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_or_predicates_use_disjunctive_key_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE id = 2 OR name = 'Grace' LIMIT 2").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(3)]]
    );
    assert_eq!(result.execution.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::DisjunctiveFilteredKeyBatch {
            table: "people".to_string(),
            predicate_group_count: 2,
            matched_keys: 2,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_same_column_or_equality_uses_index_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE id = 2 OR id = 4").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(4)]]
    );
    assert_eq!(result.execution.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "id".to_string(),
            matched_keys: 2,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_same_column_or_equality_with_order_uses_ordered_index_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE id = 1 OR id = 3 ORDER BY id DESC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(1)]]
    );
    assert_eq!(result.execution.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("id".to_string()),
            predicate_op: Some(SelectFilterOp::Eq),
            order_column: "id".to_string(),
            descending: true,
            matched_keys: 2,
        }
    );
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_in_membership_uses_index_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE id IN (1, 3) ORDER BY id DESC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(3)], vec![SqlValue::Int4(1)]]
    );
    assert_eq!(result.execution.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("id".to_string()),
            predicate_op: Some(SelectFilterOp::Eq),
            order_column: "id".to_string(),
            descending: true,
            matched_keys: 2,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_between_predicate_uses_conjunctive_key_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE id BETWEEN 2 AND 4 ORDER BY id DESC LIMIT 2")
            .unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(4)], vec![SqlValue::Int4(3)]]
    );
    assert_eq!(result.execution.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("<conjunction>".to_string()),
            predicate_op: None,
            order_column: "id".to_string(),
            descending: true,
            matched_keys: 3,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_prefix_like_predicate_uses_filtered_key_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Grady')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE name LIKE 'Gra%' ORDER BY id DESC LIMIT 2")
            .unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(4)], vec![SqlValue::Int4(3)]]
    );
    assert_eq!(result.execution.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("name".to_string()),
            predicate_op: Some(SelectFilterOp::LikePrefix),
            order_column: "id".to_string(),
            descending: true,
            matched_keys: 2,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_or_predicates_with_order_use_ordered_key_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) = parse_command(
        "SELECT id FROM people WHERE id <= 2 OR name = 'Grace' ORDER BY name DESC LIMIT 2",
    )
    .unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(3)]]
    );
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("<disjunction>".to_string()),
            predicate_op: None,
            order_column: "name".to_string(),
            descending: true,
            matched_keys: 3,
        }
    );
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_parenthesized_or_predicates_use_disjunctive_key_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE (id = 1) OR (name = 'Grace') ORDER BY id")
            .unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(3)]]
    );
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
#[ignore = "requires CUDA driver"]
fn relational_sql_gpu_bridge_nested_boolean_predicates_use_disjunctive_key_batch() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) = parse_command(
            "SELECT id FROM people WHERE (id = 1 OR id = 3) AND (name = 'Ada' OR name = 'Grace') ORDER BY id",
        )
        .unwrap() else {
            panic!("expected SELECT plan");
        };
    let result = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap();

    assert_eq!(
        result.specification.rows,
        vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(3)]]
    );
    assert_eq!(result.execution.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.execution.fallback_reason, None);
    assert_eq!(
        *result.specification.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("<disjunction>".to_string()),
            predicate_op: None,
            order_column: "id".to_string(),
            descending: false,
            matched_keys: 2,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
fn relational_sql_gpu_bridge_report_summarizes_gpu_only_execution() {
    let result = || RelationalSelectResult {
        columns: Arc::new(Vec::new()),
        rows: Vec::<Vec<SqlValue>>::new().into(),
        planned_target: DeviceTarget::Gpu(0),
        executed_target: DeviceTarget::Gpu(0),
        fallback_reason: None,
        access_path: Arc::new(RelationalAccessPath::FullTableScan),
    };
    let results = vec![result(), result()];
    let report = RelationalSqlGpuBridgeReport::from_results(&results);

    assert_eq!(report.query_count, 2);
    assert_eq!(report.gpu_executed_count, 2);
    assert_eq!(report.cpu_fallback_count, 0);
    assert_eq!(report.gpu_executed_permyriad, 10_000);
    assert_eq!(report.cpu_fallback_permyriad, 0);
}

#[test]
fn relational_sql_cuda_probe_reuses_cached_unavailable_snapshot_and_fails_loud() {
    let mut e = Engine::new_local_test_engine();
    let _ = e
        .cached_cuda_probe_runtime
        .set(CudaDriverRuntime::unavailable());
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();

    let Command::Select(select) = parse_command("SELECT * FROM people").unwrap() else {
        panic!("expected SELECT plan");
    };
    let fallback_before = e.metrics().snapshot().fallback_total;
    let first = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap_err();
    let second = e
        .evaluate_relational_select_specification_with_cuda_driver(&select)
        .unwrap_err();

    for error in [first, second] {
        assert!(
            error
                .to_string()
                .contains("has no retained resident device memory"),
            "unexpected error: {error}"
        );
    }
    assert_eq!(e.metrics().snapshot().fallback_total, fallback_before);
    assert_eq!(
        e.cached_cuda_probe_runtime.get().unwrap().snapshot(),
        CudaDriverRuntime::unavailable().snapshot()
    );
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn relational_sql_select_cuda_driver_reports_gpu_execution() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();

    let Command::Select(select) = parse_command("SELECT * FROM people").unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();

    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
        ]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert!(e.table_device_authoritative("people"));
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_relational_sql_equality_limit_without_fallback() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (2, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT * FROM people WHERE id = 2 LIMIT 1").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();

    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_relational_sql_projection_without_fallback() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (2, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT name FROM people WHERE id = 2 LIMIT 1").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();

    assert_eq!(result.rows, vec![vec![SqlValue::Text("Linus".to_string())]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_relational_sql_order_by_without_fallback() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (2, 'Linus'), (1, 'Ada'), (3, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people ORDER BY name DESC LIMIT 2").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();

    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(3)]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_relational_sql_range_without_fallback() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE id > 1 ORDER BY name DESC LIMIT 2").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();

    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(4)]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_relational_sql_and_predicates_without_fallback() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) = parse_command(
        "SELECT id FROM people WHERE id >= 2 AND name = 'Grace' ORDER BY id DESC LIMIT 1",
    )
    .unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();

    assert_eq!(result.rows, vec![vec![SqlValue::Int4(4)]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
}

#[test]
#[ignore = "requires local NVIDIA driver and CUDA-capable hardware"]
fn execute_mvcc_query_cuda_driver_runs_relational_sql_or_predicates_without_fallback() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) = parse_command(
        "SELECT id FROM people WHERE id <= 2 OR name = 'Grace' ORDER BY name DESC LIMIT 2",
    )
    .unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();

    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(3)]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
}

#[test]
fn relational_catalog_truncates_table_and_replays_from_wal() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE people (id INT PRIMARY KEY, name TEXT UNIQUE)",
    )
    .unwrap();
    e.execute_text(2, "CREATE TABLE teams (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO teams (id, name) VALUES (9, 'Infra')")
        .unwrap();
    e.execute_text(5, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    e.execute_text(6, "COMMENT ON TABLE public.people IS 'people table'")
        .unwrap();
    e.execute_text(7, "COMMENT ON COLUMN public.people.name IS 'display name'")
        .unwrap();
    e.execute_text(8, "COMMENT ON INDEX public.people_name_idx IS 'lookup'")
        .unwrap();
    e.execute_text(
        9,
        "COMMENT ON CONSTRAINT people_pkey ON public.people IS 'identity'",
    )
    .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("people").unwrap();
    assert!(snapshot.is_valid());

    e.execute_text(10, "TRUNCATE TABLE ONLY public.people")
        .unwrap();

    let Command::Select(empty_people) =
        parse_command("SELECT id, name FROM people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    assert!(e
        .execute_relational_select(&empty_people)
        .unwrap()
        .rows
        .is_empty());
    assert!(e.relational_catalog_table("people").is_some());
    assert!(e.relational_catalog_table("teams").is_some());
    assert_eq!(
        e.relational_table_comment("people").as_deref(),
        Some("people table")
    );
    assert_eq!(
        e.relational_column_comment("people", 2).as_deref(),
        Some("display name")
    );
    assert_eq!(
        e.relational_index_comment("people_name_idx").as_deref(),
        Some("lookup")
    );
    assert_eq!(
        e.relational_constraint_comment("people", "people_pkey")
            .as_deref(),
        Some("identity")
    );
    // TYPE-COVERAGE #14 (text): a PK'd text table is SHARD-resident, so there is no single-buffer
    // snapshot to invalidate — the empty ORDER BY select above already proved TRUNCATE serves no stale
    // rows. Accept either an invalidated single-buffer snapshot (legacy) or the shard representation.
    assert!(e
        .relational_residency_snapshot("people")
        .is_none_or(|snapshot| !snapshot.is_valid()));

    e.execute_text(11, "INSERT INTO people (id, name) VALUES (1, 'Ada')")
        .unwrap();
    let result = e.execute_relational_select(&empty_people).unwrap();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_result = recovered.execute_relational_select(&empty_people).unwrap();
    assert_eq!(
        recovered_result.rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]]
    );
    assert_eq!(
        recovered.relational_table_comment("people").as_deref(),
        Some("people table")
    );
    assert_eq!(
        recovered
            .relational_constraint_comment("people", "people_pkey")
            .as_deref(),
        Some("identity")
    );

    e.execute_text(12, "CREATE TABLE restart_people (id SERIAL, name TEXT)")
        .unwrap();
    e.execute_text(
        13,
        "INSERT INTO restart_people (name) VALUES ('Ada'), ('Grace')",
    )
    .unwrap();
    let restart_seq = e
        .relational_catalog_sequence("restart_people_id_seq")
        .unwrap();
    assert_eq!(restart_seq.last_value, 2);
    assert!(restart_seq.is_called);
    e.execute_text(14, "TRUNCATE TABLE public.restart_people RESTART IDENTITY")
        .unwrap();
    let restart_seq = e
        .relational_catalog_sequence("restart_people_id_seq")
        .unwrap();
    assert_eq!(restart_seq.last_value, 1);
    assert!(!restart_seq.is_called);
    e.execute_text(15, "INSERT INTO restart_people (name) VALUES ('Linus')")
        .unwrap();
    let Command::Select(restart_select) =
        parse_command("SELECT id, name FROM restart_people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    assert_eq!(
        e.execute_relational_select(&restart_select).unwrap().rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("Linus".to_string())]]
    );
    let recovered_restart = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered_restart
            .execute_relational_select(&restart_select)
            .unwrap()
            .rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("Linus".to_string())]]
    );
    let recovered_seq = recovered_restart
        .relational_catalog_sequence("restart_people_id_seq")
        .unwrap();
    assert_eq!(recovered_seq.last_value, 1);
    assert!(recovered_seq.is_called);

    let missing_truncate = e
        .execute_text(16, "TRUNCATE TABLE missing_people")
        .unwrap_err()
        .to_string();
    assert!(
        missing_truncate.contains("relation \"missing_people\" does not exist"),
        "{missing_truncate}"
    );

    let with_view = Engine::new_local_test_engine();
    with_view
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    with_view
        .execute_text(2, "CREATE VIEW public.people_view AS SELECT * FROM people")
        .unwrap();
    let view_truncate = with_view
        .execute_text(3, "TRUNCATE people_view")
        .unwrap_err()
        .to_string();
    assert!(
        view_truncate.contains("relation \"people_view\" is not a table"),
        "{view_truncate}"
    );
    assert!(with_view.relational_catalog_view("people_view").is_some());
}

#[test]
fn relational_catalog_records_relation_acl_metadata_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "GRANT SELECT, INSERT ON TABLE public.people TO PUBLIC")
        .unwrap();
    e.execute_text(3, "GRANT ALL PRIVILEGES ON people TO postgres")
        .unwrap();
    e.execute_text(4, "REVOKE INSERT ON people FROM PUBLIC")
        .unwrap();

    let acl = e.relational_table_acl("people").unwrap();
    assert_eq!(
        acl.get("public").unwrap(),
        &BTreeSet::from([TablePrivilege::Select])
    );
    assert_eq!(
        acl.get("postgres").unwrap(),
        &BTreeSet::from([
            TablePrivilege::Select,
            TablePrivilege::Insert,
            TablePrivilege::Update,
            TablePrivilege::Delete,
        ])
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_acl = recovered.relational_table_acl("people").unwrap();
    assert_eq!(recovered_acl, acl);

    e.execute_text(5, "REVOKE SELECT ON people FROM PUBLIC")
        .unwrap();
    assert!(!e
        .relational_table_acl("people")
        .unwrap()
        .contains_key("public"));

    let missing = e
        .execute_text(6, "GRANT SELECT ON missing_people TO PUBLIC")
        .unwrap_err()
        .to_string();
    assert!(
        missing.contains("relation \"missing_people\" does not exist"),
        "{missing}"
    );

    e.execute_text(7, "CREATE VIEW people_view AS SELECT * FROM people")
        .unwrap();
    e.execute_text(
        8,
        "CREATE MATERIALIZED VIEW people_mv AS SELECT * FROM people WITH DATA",
    )
    .unwrap();
    e.execute_text(9, "CREATE SEQUENCE people_seq").unwrap();
    e.execute_text(10, "GRANT SELECT ON VIEW people_view TO PUBLIC")
        .unwrap();
    e.execute_text(11, "GRANT SELECT ON MATERIALIZED VIEW people_mv TO PUBLIC")
        .unwrap();
    e.execute_text(
        12,
        "GRANT SELECT, UPDATE ON SEQUENCE people_seq TO postgres",
    )
    .unwrap();

    assert_eq!(
        e.relational_relation_acl("people_view")
            .unwrap()
            .get("public")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select])
    );
    assert_eq!(
        e.relational_relation_acl("people_mv")
            .unwrap()
            .get("public")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select])
    );
    assert_eq!(
        e.relational_relation_acl("people_seq")
            .unwrap()
            .get("postgres")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select, TablePrivilege::Update])
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_relation_acl("people_view")
            .unwrap()
            .get("public")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select])
    );
    assert_eq!(
        recovered
            .relational_relation_acl("people_mv")
            .unwrap()
            .get("public")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select])
    );
    assert_eq!(
        recovered
            .relational_relation_acl("people_seq")
            .unwrap()
            .get("postgres")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select, TablePrivilege::Update])
    );

    e.execute_text(13, "REVOKE SELECT ON TABLE people_view FROM PUBLIC")
        .unwrap();
    assert!(!e
        .relational_relation_acl("people_view")
        .unwrap()
        .contains_key("public"));

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(!recovered
        .relational_relation_acl("people_view")
        .unwrap()
        .contains_key("public"));
}

#[test]
fn relational_catalog_records_default_table_acl_metadata_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
            1,
            "ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT, INSERT ON TABLES TO PUBLIC",
        )
        .unwrap();
    e.execute_text(2, "CREATE TABLE first_people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "ALTER DEFAULT PRIVILEGES REVOKE INSERT ON TABLES FROM PUBLIC",
    )
    .unwrap();
    e.execute_text(4, "CREATE TABLE second_people (id INT, name TEXT)")
        .unwrap();

    assert_eq!(
        e.relational_table_acl("first_people")
            .unwrap()
            .get("public")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select, TablePrivilege::Insert])
    );
    assert_eq!(
        e.relational_table_acl("second_people")
            .unwrap()
            .get("public")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select])
    );
    assert_eq!(
        e.relational_default_table_acl().get("public").unwrap(),
        &BTreeSet::from([TablePrivilege::Select])
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_table_acl("first_people")
            .unwrap()
            .get("public")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select, TablePrivilege::Insert])
    );
    assert_eq!(
        recovered
            .relational_table_acl("second_people")
            .unwrap()
            .get("public")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select])
    );
    assert_eq!(
        recovered
            .relational_default_table_acl()
            .get("public")
            .unwrap(),
        &BTreeSet::from([TablePrivilege::Select])
    );
}

#[test]
fn relational_catalog_records_schema_acl_metadata_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "GRANT USAGE, CREATE ON SCHEMA public TO PUBLIC")
        .unwrap();
    e.execute_text(2, "GRANT ALL PRIVILEGES ON SCHEMA public TO postgres")
        .unwrap();
    e.execute_text(3, "REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        .unwrap();

    assert_eq!(
        e.relational_schema_acl().get("public").unwrap(),
        &BTreeSet::from([SchemaPrivilege::Usage])
    );
    assert_eq!(
        e.relational_schema_acl().get("postgres").unwrap(),
        &BTreeSet::from([SchemaPrivilege::Usage, SchemaPrivilege::Create])
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(recovered.relational_schema_acl(), e.relational_schema_acl());

    let missing = e
        .execute_text(4, "GRANT USAGE ON SCHEMA private TO PUBLIC")
        .unwrap_err()
        .to_string();
    assert!(
        missing.contains("invalid relational SQL syntax"),
        "{missing}"
    );

    e.execute_text(5, "DROP SCHEMA public").unwrap();
    assert!(e.relational_schema_acl().is_empty());
    let recovered_after_drop = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered_after_drop.relational_schema_acl().is_empty());
}

#[test]
fn relational_catalog_records_function_acl_metadata_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE FUNCTION answer() RETURNS int LANGUAGE sql AS 'SELECT 42'",
    )
    .unwrap();
    e.execute_text(2, "CREATE ROLE app_reader").unwrap();
    e.execute_text(3, "GRANT EXECUTE ON FUNCTION public.answer() TO app_reader")
        .unwrap();
    e.execute_text(4, "GRANT ALL PRIVILEGES ON FUNCTION answer() TO PUBLIC")
        .unwrap();

    let acl = e.relational_function_acl("answer").unwrap();
    assert_eq!(
        acl.get("app_reader").unwrap(),
        &BTreeSet::from([FunctionPrivilege::Execute])
    );
    assert_eq!(
        acl.get("public").unwrap(),
        &BTreeSet::from([FunctionPrivilege::Execute])
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(recovered.relational_function_acl("answer").unwrap(), acl);

    e.execute_text(5, "ALTER ROLE app_reader RENAME TO app_executor")
        .unwrap();
    assert!(!e
        .relational_function_acl("answer")
        .unwrap()
        .contains_key("app_reader"));
    assert_eq!(
        e.relational_function_acl("answer")
            .unwrap()
            .get("app_executor")
            .unwrap(),
        &BTreeSet::from([FunctionPrivilege::Execute])
    );

    e.execute_text(6, "REVOKE EXECUTE ON FUNCTION answer() FROM PUBLIC")
        .unwrap();
    assert!(!e
        .relational_function_acl("answer")
        .unwrap()
        .contains_key("public"));

    let missing = e
        .execute_text(7, "GRANT EXECUTE ON FUNCTION missing_answer() TO PUBLIC")
        .unwrap_err()
        .to_string();
    assert!(
        missing.contains("function \"missing_answer\" does not exist"),
        "{missing}"
    );
}

#[test]
fn relational_catalog_records_publications_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE accounts (id INT, owner TEXT)")
        .unwrap();
    e.execute_text(
        3,
        "CREATE PUBLICATION app_pub FOR TABLE public.people, accounts",
    )
    .unwrap();
    e.execute_text(4, "CREATE PUBLICATION all_pub FOR ALL TABLES")
        .unwrap();

    let app_pub = e.relational_catalog_publication("app_pub").unwrap();
    assert!(!app_pub.all_tables);
    assert_eq!(
        app_pub.tables,
        vec!["people".to_string(), "accounts".to_string()]
    );
    let all_pub = e.relational_catalog_publication("all_pub").unwrap();
    assert!(all_pub.all_tables);
    assert!(all_pub.tables.is_empty());

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_publication("app_pub")
            .unwrap()
            .tables,
        vec!["people".to_string(), "accounts".to_string()]
    );
    assert!(
        recovered
            .relational_catalog_publication("all_pub")
            .unwrap()
            .all_tables
    );

    e.execute_text(5, "DROP PUBLICATION app_pub").unwrap();
    assert!(e.relational_catalog_publication("app_pub").is_none());
    e.execute_text(6, "DROP PUBLICATION IF EXISTS missing_pub")
        .unwrap();

    let duplicate = e
        .execute_text(7, "CREATE PUBLICATION all_pub FOR ALL TABLES")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate.contains("publication \"all_pub\" already exists"),
        "{duplicate}"
    );
    let missing_table = e
        .execute_text(8, "CREATE PUBLICATION missing_pub FOR TABLE missing_people")
        .unwrap_err()
        .to_string();
    assert!(
        missing_table.contains("relation \"missing_people\" does not exist"),
        "{missing_table}"
    );
}

#[test]
fn relational_catalog_records_disabled_subscriptions_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE PUBLICATION app_pub FOR TABLE people")
        .unwrap();
    e.execute_text(3, "CREATE PUBLICATION all_pub FOR ALL TABLES")
        .unwrap();
    e.execute_text(
            4,
            "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost dbname=postgres' PUBLICATION app_pub, all_pub WITH (connect = false, enabled = false)",
        )
        .unwrap();

    let subscription = e.relational_catalog_subscription("app_sub").unwrap();
    assert_eq!(subscription.connection, "host=localhost dbname=postgres");
    assert_eq!(
        subscription.publications,
        vec!["app_pub".to_string(), "all_pub".to_string()]
    );
    assert!(!subscription.enabled);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_subscription("app_sub")
            .unwrap()
            .publications,
        vec!["app_pub".to_string(), "all_pub".to_string()]
    );

    let duplicate = e
            .execute_text(
                5,
                "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost' PUBLICATION app_pub WITH (connect = false, enabled = false)",
            )
            .unwrap_err()
            .to_string();
    assert!(
        duplicate.contains("subscription \"app_sub\" already exists"),
        "{duplicate}"
    );
    let missing_publication = e
            .execute_text(
                6,
                "CREATE SUBSCRIPTION missing_pub_sub CONNECTION 'host=localhost' PUBLICATION missing_pub WITH (connect = false, enabled = false)",
            )
            .unwrap_err()
            .to_string();
    assert!(
        missing_publication.contains("publication \"missing_pub\" does not exist"),
        "{missing_publication}"
    );

    e.execute_text(7, "DROP SUBSCRIPTION app_sub").unwrap();
    assert!(e.relational_catalog_subscription("app_sub").is_none());
    e.execute_text(8, "DROP SUBSCRIPTION IF EXISTS missing_sub")
        .unwrap();
}

#[test]
fn relational_catalog_records_logical_replication_comments_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE PUBLICATION app_pub FOR TABLE people")
        .unwrap();
    e.execute_text(
            3,
            "CREATE SUBSCRIPTION app_sub CONNECTION 'host=localhost dbname=postgres' PUBLICATION app_pub WITH (connect = false, enabled = false)",
        )
        .unwrap();

    e.execute_text(
        4,
        "COMMENT ON PUBLICATION app_pub IS 'publication metadata'",
    )
    .unwrap();
    e.execute_text(
        5,
        "COMMENT ON SUBSCRIPTION app_sub IS 'subscription metadata'",
    )
    .unwrap();
    assert_eq!(
        e.relational_publication_comment("app_pub").as_deref(),
        Some("publication metadata")
    );
    assert_eq!(
        e.relational_subscription_comment("app_sub").as_deref(),
        Some("subscription metadata")
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_publication_comment("app_pub")
            .as_deref(),
        Some("publication metadata")
    );
    assert_eq!(
        recovered
            .relational_subscription_comment("app_sub")
            .as_deref(),
        Some("subscription metadata")
    );

    let missing_pub_engine = Engine::new_local_test_engine();
    missing_pub_engine
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    let missing_publication = missing_pub_engine
        .execute_text(2, "COMMENT ON PUBLICATION missing_pub IS 'missing'")
        .unwrap_err()
        .to_string();
    assert!(
        missing_publication.contains("publication \"missing_pub\" does not exist"),
        "{missing_publication}"
    );
    let missing_sub_engine = Engine::new_local_test_engine();
    missing_sub_engine
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    missing_sub_engine
        .execute_text(2, "CREATE PUBLICATION app_pub FOR TABLE people")
        .unwrap();
    let missing_subscription = missing_sub_engine
        .execute_text(3, "COMMENT ON SUBSCRIPTION missing_sub IS 'missing'")
        .unwrap_err()
        .to_string();
    assert!(
        missing_subscription.contains("subscription \"missing_sub\" does not exist"),
        "{missing_subscription}"
    );

    e.execute_text(8, "COMMENT ON PUBLICATION app_pub IS NULL")
        .unwrap();
    assert_eq!(e.relational_publication_comment("app_pub"), None);
    e.execute_text(
        9,
        "COMMENT ON PUBLICATION app_pub IS 'publication metadata'",
    )
    .unwrap();
    e.execute_text(10, "DROP SUBSCRIPTION app_sub").unwrap();
    e.execute_text(11, "DROP PUBLICATION app_pub").unwrap();
    assert_eq!(e.relational_subscription_comment("app_sub"), None);
    assert_eq!(e.relational_publication_comment("app_pub"), None);
}

#[test]
fn relational_catalog_records_domains_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE DOMAIN public.account_id AS int4")
        .unwrap();
    e.execute_text(2, "COMMENT ON DOMAIN public.account_id IS 'account ids'")
        .unwrap();
    e.execute_text(3, "CREATE TABLE accounts (id account_id, name TEXT)")
        .unwrap();
    e.execute_text(4, "INSERT INTO accounts VALUES (7, 'Ada')")
        .unwrap();

    let domain = e.relational_catalog_domain("account_id").unwrap();
    assert_eq!(domain.name, "account_id");
    assert_eq!(domain.base_type, SqlType::Int4);
    let oid = domain.oid;
    let table = e.relational_catalog_table("accounts").unwrap();
    assert_eq!(table.columns[0].domain.as_deref(), Some("account_id"));
    assert_eq!(table.columns[0].ty, SqlType::Int4);
    assert_eq!(table.columns[0].type_oid, oid);
    assert_eq!(
        e.ddl_catalog()
            .relational_comments
            .get(&RelationalCommentTarget::Domain {
                domain: "account_id".to_string(),
            })
            .map(String::as_str),
        Some("account ids")
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_domain = recovered.relational_catalog_domain("account_id").unwrap();
    assert_eq!(recovered_domain.oid, oid);
    assert_eq!(recovered_domain.base_type, SqlType::Int4);
    let recovered_table = recovered.relational_catalog_table("accounts").unwrap();
    assert_eq!(
        recovered_table.columns[0].domain.as_deref(),
        Some("account_id")
    );
    assert_eq!(recovered_table.columns[0].type_oid, oid);

    let dependent = e.execute_text(5, "DROP DOMAIN account_id").unwrap_err();
    assert!(
        dependent
            .to_string()
            .contains("cannot drop domain \"account_id\" because other objects depend on it"),
        "{dependent}"
    );
    e.execute_text(6, "DROP TABLE accounts").unwrap();
    e.execute_text(7, "DROP DOMAIN account_id").unwrap();
    assert!(e.relational_catalog_domain("account_id").is_none());
    assert!(!e
        .ddl_catalog()
        .relational_comments
        .contains_key(&RelationalCommentTarget::Domain {
            domain: "account_id".to_string(),
        }));
    e.execute_text(8, "DROP DOMAIN IF EXISTS missing_domain")
        .unwrap();

    let duplicate = e
        .execute_text(9, "CREATE DOMAIN label AS text")
        .and_then(|_| e.execute_text(10, "CREATE DOMAIN label AS text"))
        .unwrap_err()
        .to_string();
    assert!(
        duplicate.contains("type \"label\" already exists"),
        "{duplicate}"
    );
    let missing = e
        .execute_text(11, "COMMENT ON DOMAIN missing_domain IS 'nope'")
        .unwrap_err()
        .to_string();
    assert!(
        missing.contains("domain \"missing_domain\" does not exist"),
        "{missing}"
    );
}

#[test]
fn relational_catalog_records_bounded_functions_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE FUNCTION public.answer() RETURNS int4 LANGUAGE sql AS 'SELECT 42'",
    )
    .unwrap();
    e.execute_text(2, "COMMENT ON FUNCTION public.answer() IS 'metadata only'")
        .unwrap();

    let function = e.relational_catalog_function("answer").unwrap();
    assert_eq!(function.name, "answer");
    assert_eq!(function.return_type, SqlType::Int4);
    assert_eq!(function.body, "SELECT 42");
    let oid = function.oid;
    assert_eq!(
        e.relational_function_comment("answer").as_deref(),
        Some("metadata only")
    );
    e.execute_text(
        3,
        "ALTER FUNCTION public.answer() RENAME TO ultimate_answer",
    )
    .unwrap();
    assert!(e.relational_catalog_function("answer").is_none());
    let renamed_function = e.relational_catalog_function("ultimate_answer").unwrap();
    assert_eq!(renamed_function.oid, oid);
    assert_eq!(renamed_function.return_type, SqlType::Int4);
    assert_eq!(renamed_function.body, "SELECT 42");
    assert_eq!(
        e.relational_function_comment("ultimate_answer").as_deref(),
        Some("metadata only")
    );
    assert_eq!(e.relational_function_comment("answer"), None);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_function = recovered
        .relational_catalog_function("ultimate_answer")
        .unwrap();
    assert_eq!(recovered_function.oid, oid);
    assert_eq!(recovered_function.return_type, SqlType::Int4);
    assert_eq!(
        recovered
            .relational_function_comment("ultimate_answer")
            .as_deref(),
        Some("metadata only")
    );
    let result = recovered
        .execute_relational_function(&SelectFunction {
            name: "ultimate_answer".to_string(),
        })
        .unwrap();
    assert_eq!(result.columns[0].name, "ultimate_answer");
    assert_eq!(result.columns[0].ty, SqlType::Int4);
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(42)]]);
    let missing_old = recovered
        .execute_relational_function(&SelectFunction {
            name: "answer".to_string(),
        })
        .unwrap_err()
        .to_string();
    assert!(
        missing_old.contains("function \"answer\" does not exist"),
        "{missing_old}"
    );

    let duplicate = e
        .execute_text(
            4,
            "CREATE FUNCTION public.ultimate_answer() RETURNS text LANGUAGE sql AS 'SELECT ''x'''",
        )
        .unwrap_err()
        .to_string();
    assert!(
        duplicate.contains("function \"ultimate_answer\" already exists"),
        "{duplicate}"
    );
    e.execute_text(
        5,
        "CREATE FUNCTION public.greeting() RETURNS text LANGUAGE sql AS 'SELECT ''hello'''",
    )
    .unwrap();
    let duplicate_rename = e
        .execute_text(
            6,
            "ALTER FUNCTION public.ultimate_answer() RENAME TO greeting",
        )
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_rename.contains("function \"greeting\" already exists"),
        "{duplicate_rename}"
    );
    let missing_rename = e
        .execute_text(7, "ALTER FUNCTION missing() RENAME TO still_missing")
        .unwrap_err()
        .to_string();
    assert!(
        missing_rename.contains("function \"missing\" does not exist"),
        "{missing_rename}"
    );
    let missing_comment = e
        .execute_text(8, "COMMENT ON FUNCTION missing() IS 'missing'")
        .unwrap_err()
        .to_string();
    assert!(
        missing_comment.contains("function \"missing\" does not exist"),
        "{missing_comment}"
    );
    let missing_drop = e
        .execute_text(9, "DROP FUNCTION missing()")
        .unwrap_err()
        .to_string();
    assert!(
        missing_drop.contains("function \"missing\" does not exist"),
        "{missing_drop}"
    );
    e.execute_text(10, "DROP FUNCTION IF EXISTS missing()")
        .unwrap();
    e.execute_text(11, "DROP FUNCTION ultimate_answer()")
        .unwrap();
    assert!(e.relational_catalog_function("ultimate_answer").is_none());
    assert_eq!(e.relational_function_comment("ultimate_answer"), None);

    e.execute_text(
        12,
        "CREATE FUNCTION public.bad_body() RETURNS int4 LANGUAGE sql AS 'SELECT id FROM people'",
    )
    .unwrap();
    let unsupported = e
        .execute_relational_function(&SelectFunction {
            name: "bad_body".to_string(),
        })
        .unwrap_err()
        .to_string();
    assert!(
        unsupported.contains("only literal SELECT bodies are supported"),
        "{unsupported}"
    );
}

#[test]
fn relational_catalog_records_bounded_public_schema_lifecycle() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "COMMENT ON SCHEMA public IS 'application schema'")
        .unwrap();
    let non_empty = e
        .execute_text(2, "CREATE TABLE people (id INT, name TEXT)")
        .and_then(|_| e.execute_text(3, "DROP SCHEMA IF EXISTS public"))
        .unwrap_err()
        .to_string();
    assert!(
        non_empty.contains("cannot drop non-empty schema \"public\""),
        "{non_empty}"
    );
    assert!(e.ddl_catalog().relational_public_schema_exists);
    assert_eq!(
        e.relational_schema_comment("public").as_deref(),
        Some("application schema")
    );

    e.execute_text(4, "DROP TABLE people").unwrap();
    e.execute_text(5, "DROP SCHEMA IF EXISTS public").unwrap();
    assert!(!e.ddl_catalog().relational_public_schema_exists);
    assert_eq!(e.relational_schema_comment("public"), None);

    let missing_schema = e
        .execute_text(6, "CREATE TABLE blocked (id INT)")
        .unwrap_err()
        .to_string();
    assert!(
        missing_schema.contains("schema \"public\" does not exist"),
        "{missing_schema}"
    );

    e.execute_text(7, "CREATE SCHEMA public").unwrap();
    e.execute_text(8, "CREATE SCHEMA IF NOT EXISTS public")
        .unwrap();
    let duplicate = e
        .execute_text(9, "CREATE SCHEMA public")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate.contains("schema \"public\" already exists"),
        "{duplicate}"
    );
    e.execute_text(10, "CREATE TABLE recreated (id INT)")
        .unwrap();

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.ddl_catalog().relational_public_schema_exists);
    assert!(recovered.relational_catalog_table("recreated").is_some());
    assert_eq!(recovered.relational_schema_comment("public"), None);
}

#[test]
fn relational_catalog_records_bounded_tablespace_metadata() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLESPACE appspace LOCATION '/tmp/gpu-db-appspace'",
    )
    .unwrap();
    e.execute_text(2, "COMMENT ON TABLESPACE appspace IS 'application storage'")
        .unwrap();
    e.execute_text(3, "CREATE ROLE app_writer").unwrap();
    e.execute_text(4, "GRANT CREATE ON TABLESPACE appspace TO app_writer")
        .unwrap();

    let tablespace = e.relational_tablespace("appspace").unwrap();
    assert_eq!(tablespace.name, "appspace");
    assert_eq!(tablespace.location, "/tmp/gpu-db-appspace");
    let oid = tablespace.oid;
    assert_eq!(
        e.relational_tablespace_acl("appspace")
            .unwrap()
            .get("app_writer")
            .unwrap(),
        &BTreeSet::from([TablespacePrivilege::Create])
    );
    assert_eq!(
        e.relational_tablespace_comment("appspace").as_deref(),
        Some("application storage")
    );

    e.execute_text(5, "ALTER ROLE app_writer RENAME TO app_loader")
        .unwrap();
    e.execute_text(6, "ALTER TABLESPACE appspace RENAME TO appspace_fast")
        .unwrap();
    let renamed = e.relational_tablespace("appspace_fast").unwrap();
    assert_eq!(renamed.oid, oid);
    assert_eq!(renamed.location, "/tmp/gpu-db-appspace");
    assert!(e
        .relational_tablespace_acl("appspace_fast")
        .unwrap()
        .contains_key("app_loader"));
    assert_eq!(
        e.relational_tablespace_comment("appspace_fast").as_deref(),
        Some("application storage")
    );
    assert_eq!(e.relational_tablespace_comment("appspace"), None);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_tablespace("appspace_fast")
            .unwrap()
            .location,
        "/tmp/gpu-db-appspace"
    );
    assert_eq!(
        recovered
            .relational_tablespace_comment("appspace_fast")
            .as_deref(),
        Some("application storage")
    );

    let duplicate = e
        .execute_text(7, "CREATE TABLESPACE appspace_fast LOCATION '/tmp/other'")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate.contains("tablespace \"appspace_fast\" already exists"),
        "{duplicate}"
    );
    let duplicate_rename = e
        .execute_text(8, "CREATE TABLESPACE appspace LOCATION '/tmp/other'")
        .and_then(|_| e.execute_text(9, "ALTER TABLESPACE appspace_fast RENAME TO appspace"))
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_rename.contains("tablespace \"appspace\" already exists"),
        "{duplicate_rename}"
    );
    let bootstrap = e
        .execute_text(10, "DROP TABLESPACE pg_default")
        .unwrap_err()
        .to_string();
    assert!(
        bootstrap.contains("cannot drop bootstrap tablespace \"pg_default\""),
        "{bootstrap}"
    );
    let bootstrap_rename = e
        .execute_text(11, "ALTER TABLESPACE pg_default RENAME TO appspace_default")
        .unwrap_err()
        .to_string();
    assert!(
        bootstrap_rename.contains("cannot rename bootstrap tablespace \"pg_default\""),
        "{bootstrap_rename}"
    );
    let missing = e
        .execute_text(12, "DROP TABLESPACE missing_space")
        .unwrap_err()
        .to_string();
    assert!(
        missing.contains("tablespace \"missing_space\" does not exist"),
        "{missing}"
    );
    let missing_rename = e
        .execute_text(13, "ALTER TABLESPACE missing_space RENAME TO renamed_space")
        .unwrap_err()
        .to_string();
    assert!(
        missing_rename.contains("tablespace \"missing_space\" does not exist"),
        "{missing_rename}"
    );

    e.execute_text(
        14,
        "DROP TABLESPACE IF EXISTS appspace_fast, appspace, missing_space",
    )
    .unwrap();
    assert!(e.relational_tablespace("appspace_fast").is_none());
    assert_eq!(e.relational_tablespace_comment("appspace_fast"), None);
}

#[test]
fn relational_catalog_records_comments_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "COMMENT ON DATABASE postgres IS 'primary database'")
        .unwrap();
    e.execute_text(3, "COMMENT ON ROLE postgres IS 'bootstrap role'")
        .unwrap();
    e.execute_text(4, "COMMENT ON SCHEMA public IS 'application schema'")
        .unwrap();
    e.execute_text(5, "COMMENT ON TABLESPACE pg_default IS 'default storage'")
        .unwrap();
    e.execute_text(6, "COMMENT ON TABLESPACE pg_global IS 'global storage'")
        .unwrap();
    e.execute_text(7, "COMMENT ON TABLE public.people IS 'lookup people'")
        .unwrap();
    e.execute_text(8, "COMMENT ON COLUMN public.people.name IS 'display name'")
        .unwrap();
    e.execute_text(9, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    e.execute_text(
        10,
        "COMMENT ON INDEX public.people_name_idx IS 'name lookup'",
    )
    .unwrap();
    e.execute_text(
        11,
        "ALTER TABLE ONLY public.people ADD CONSTRAINT people_pkey PRIMARY KEY (id)",
    )
    .unwrap();
    e.execute_text(
        12,
        "COMMENT ON CONSTRAINT people_pkey ON public.people IS 'row identity'",
    )
    .unwrap();
    e.execute_text(
        13,
        "CREATE VIEW public.people_lookup AS SELECT id, name FROM people WHERE id > 0 ORDER BY id",
    )
    .unwrap();
    e.execute_text(14, "COMMENT ON VIEW public.people_lookup IS 'lookup view'")
        .unwrap();
    assert_eq!(
        e.relational_database_comment("postgres").as_deref(),
        Some("primary database")
    );
    assert_eq!(
        e.relational_role_comment("postgres").as_deref(),
        Some("bootstrap role")
    );
    assert_eq!(
        e.relational_schema_comment("public").as_deref(),
        Some("application schema")
    );
    assert_eq!(
        e.relational_tablespace_comment("pg_default").as_deref(),
        Some("default storage")
    );
    assert_eq!(
        e.relational_tablespace_comment("pg_global").as_deref(),
        Some("global storage")
    );
    assert_eq!(
        e.relational_table_comment("people").as_deref(),
        Some("lookup people")
    );
    assert_eq!(
        e.relational_column_comment("people", 2).as_deref(),
        Some("display name")
    );
    assert_eq!(
        e.relational_index_comment("people_name_idx").as_deref(),
        Some("name lookup")
    );
    assert_eq!(
        e.relational_constraint_comment("people", "people_pkey")
            .as_deref(),
        Some("row identity")
    );
    assert_eq!(
        e.relational_view_comment("people_lookup").as_deref(),
        Some("lookup view")
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered.relational_database_comment("postgres").as_deref(),
        Some("primary database")
    );
    assert_eq!(
        recovered.relational_role_comment("postgres").as_deref(),
        Some("bootstrap role")
    );
    assert_eq!(
        recovered.relational_schema_comment("public").as_deref(),
        Some("application schema")
    );
    assert_eq!(
        recovered
            .relational_tablespace_comment("pg_default")
            .as_deref(),
        Some("default storage")
    );
    assert_eq!(
        recovered
            .relational_tablespace_comment("pg_global")
            .as_deref(),
        Some("global storage")
    );
    assert_eq!(
        recovered.relational_table_comment("people").as_deref(),
        Some("lookup people")
    );
    assert_eq!(
        recovered.relational_column_comment("people", 2).as_deref(),
        Some("display name")
    );
    assert_eq!(
        recovered
            .relational_index_comment("people_name_idx")
            .as_deref(),
        Some("name lookup")
    );
    assert_eq!(
        recovered
            .relational_constraint_comment("people", "people_pkey")
            .as_deref(),
        Some("row identity")
    );
    assert_eq!(
        recovered
            .relational_view_comment("people_lookup")
            .as_deref(),
        Some("lookup view")
    );

    e.execute_text(15, "COMMENT ON DATABASE postgres IS NULL")
        .unwrap();
    assert_eq!(e.relational_database_comment("postgres"), None);
    e.execute_text(16, "COMMENT ON ROLE postgres IS NULL")
        .unwrap();
    assert_eq!(e.relational_role_comment("postgres"), None);
    e.execute_text(17, "COMMENT ON SCHEMA public IS NULL")
        .unwrap();
    assert_eq!(e.relational_schema_comment("public"), None);
    e.execute_text(18, "COMMENT ON TABLESPACE pg_default IS NULL")
        .unwrap();
    assert_eq!(e.relational_tablespace_comment("pg_default"), None);
    e.execute_text(19, "COMMENT ON TABLESPACE pg_global IS NULL")
        .unwrap();
    assert_eq!(e.relational_tablespace_comment("pg_global"), None);
    e.execute_text(20, "COMMENT ON COLUMN public.people.name IS NULL")
        .unwrap();
    assert_eq!(e.relational_column_comment("people", 2), None);
    e.execute_text(21, "DROP INDEX people_name_idx").unwrap();
    assert_eq!(e.relational_index_comment("people_name_idx"), None);
    e.execute_text(22, "DROP INDEX people_pkey").unwrap();
    assert_eq!(
        e.relational_constraint_comment("people", "people_pkey"),
        None
    );
    e.execute_text(23, "DROP VIEW people_lookup").unwrap();
    assert_eq!(e.relational_view_comment("people_lookup"), None);

    let missing = Engine::new_local_test_engine();
    missing
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    assert!(missing
        .execute_text(2, "COMMENT ON COLUMN public.people.missing IS 'bad'")
        .unwrap_err()
        .to_string()
        .contains("column \"missing\" does not exist"));

    let missing_index = Engine::new_local_test_engine();
    missing_index
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    assert!(missing_index
        .execute_text(2, "COMMENT ON INDEX public.people_name_idx IS 'bad'")
        .unwrap_err()
        .to_string()
        .contains("index \"people_name_idx\" does not exist"));

    assert!(missing_index
        .execute_text(3, "COMMENT ON VIEW public.people IS 'bad'")
        .is_err());

    let missing_constraint = Engine::new_local_test_engine();
    missing_constraint
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    assert!(missing_constraint
        .execute_text(
            2,
            "COMMENT ON CONSTRAINT people_pkey ON public.people IS 'bad'",
        )
        .unwrap_err()
        .to_string()
        .contains("constraint \"people_pkey\" does not exist"));

    assert!(Engine::new_local_test_engine()
        .execute_text(1, "COMMENT ON SCHEMA private IS 'bad'")
        .unwrap_err()
        .to_string()
        .contains("schema \"private\" does not exist"));
    assert!(Engine::new_local_test_engine()
        .execute_text(1, "COMMENT ON DATABASE template1 IS 'bad'")
        .unwrap_err()
        .to_string()
        .contains("database \"template1\" does not exist"));
    assert!(Engine::new_local_test_engine()
        .execute_text(1, "COMMENT ON ROLE missing_role IS 'bad'")
        .unwrap_err()
        .to_string()
        .contains("role \"missing_role\" does not exist"));
    assert!(Engine::new_local_test_engine()
        .execute_text(1, "COMMENT ON TABLESPACE missing_space IS 'bad'")
        .unwrap_err()
        .to_string()
        .contains("tablespace \"missing_space\" does not exist"));
}

#[test]
fn relational_catalog_select_binding_uses_catalog_descriptors() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    let table = e.relational_catalog_table("people").unwrap();
    let Command::Select(select) =
        parse_command("SELECT name FROM people WHERE id = 1 ORDER BY name DESC").unwrap()
    else {
        panic!("expected SELECT plan");
    };

    let bound = bind_relational_select(&table, &select).unwrap();

    assert_eq!(bound.selected_indexes, vec![1]);
    assert_eq!(bound.selected_columns[0].name, "name");
    assert_eq!(
        bound.selected_columns[0].type_oid,
        SqlType::Text.postgres_oid()
    );
    assert_eq!(
        bound.filter,
        Some((0, SelectFilterOp::Eq, SqlValue::Int4(1)))
    );
    assert_eq!(bound.order, Some((1, true)));

    // SUM(int4) emits an Int8 value and PostgreSQL bigint metadata. Keep all three descriptors in
    // agreement so persisted/transient result relations select the i64 payload section.
    for sql in [
        "SELECT SUM(id) FROM people",
        "SELECT name, SUM(id) FROM people GROUP BY name",
    ] {
        let Command::Select(sum_select) = parse_command(sql).unwrap() else {
            panic!("expected SELECT plan");
        };
        let sum_bound = bind_relational_select(&table, &sum_select).unwrap();
        let sum = sum_bound.selected_columns.last().unwrap();
        assert_eq!(sum.ty, SqlType::Int8, "{sql}");
        assert_eq!(sum.type_oid, 20, "{sql}");
        assert_eq!(sum.type_size, 8, "{sql}");
    }

    let Command::Select(bad_select) =
        parse_command("SELECT missing FROM people ORDER BY name").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    assert!(bind_relational_select(&table, &bad_select)
        .unwrap_err()
        .to_string()
        .contains("column \"missing\" does not exist"));

    let Command::Select(bad_distinct) =
        parse_command("SELECT DISTINCT name FROM people ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    assert!(bind_relational_select(&table, &bad_distinct)
        .unwrap_err()
        .to_string()
        .contains("SELECT DISTINCT ORDER BY must reference a selected column"));
}

#[test]
fn relational_catalog_replays_from_durable_wal_with_table_data() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();

    let durable = e.durable_wal_records().to_vec();
    let recovered = Engine::recover_from_durable_wal(&durable).unwrap();
    let table = recovered.relational_catalog_table("people").unwrap();

    assert_eq!(table.schema, PUBLIC_SCHEMA_NAME);
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);
    assert_eq!(table.columns[0].table_oid, FIRST_USER_RELATION_OID);
    assert_eq!(table.columns[0].attnum, 1);
    assert_eq!(table.columns[1].attnum, 2);
    assert_eq!(recovered.wal_flushed_count(), durable.len());

    let Command::Select(select) = parse_command("SELECT id, name FROM people ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = recovered.execute_relational_select(&select).unwrap();

    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
        ]
    );
    assert_recovered_relational_access_path(
        &result,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: None,
            predicate_op: None,
            order_column: "id".to_string(),
            descending: false,
            matched_keys: 2,
        },
    );
}

#[test]
fn sql_value_key_words_matches_le_section_layout() {
    use crate::engine_residency::{key_column_width_words, sql_value_key_words};
    use gpu_db_sql::SqlType;
    // i32-section types -> 1 word (the value).
    assert_eq!(
        sql_value_key_words(SqlType::Int4, &SqlValue::Int4(-5)),
        Some(vec![-5])
    );
    assert_eq!(key_column_width_words(SqlType::Int4), Some(1));
    // i64 types -> 2 words [low32, high32], matching the section's `value.to_le_bytes()` read as two
    // LE i32 words (this ordering is load-bearing for host/device fold agreement).
    let v: i64 = 5_000_000_000; // 0x1_2A05F200: low = 0x2A05F200, high = 0x1
    let low = 0x2A05_F200_u32 as i32;
    let high = 0x1_i32;
    assert_eq!(
        sql_value_key_words(SqlType::Int8, &SqlValue::Int8(v)),
        Some(vec![low, high])
    );
    assert_eq!(key_column_width_words(SqlType::Int8), Some(2));
    assert_eq!(key_column_width_words(SqlType::Timestamp), Some(2));
    // Reassembling the two words little-endian recovers the i64 exactly.
    let recon = (low as u32 as u64) | ((high as u32 as u64) << 32);
    assert_eq!(recon as i64, v);
    // b128 (Numeric/Uuid) -> 4 words (16 LE bytes = 4 LE i32). Uuid folds its raw bytes; Numeric folds
    // its i128 mantissa (both matching the b128 section's LE byte layout).
    assert_eq!(key_column_width_words(SqlType::Uuid), Some(4));
    assert_eq!(
        key_column_width_words(SqlType::Numeric {
            precision: 20,
            scale: 4
        }),
        Some(4)
    );
    let uuid_bytes: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    assert_eq!(
        sql_value_key_words(SqlType::Uuid, &SqlValue::Uuid(uuid_bytes)),
        Some(vec![
            i32::from_le_bytes([1, 2, 3, 4]),
            i32::from_le_bytes([5, 6, 7, 8]),
            i32::from_le_bytes([9, 10, 11, 12]),
            i32::from_le_bytes([13, 14, 15, 16]),
        ])
    );
    // Text (variable-length, Stage 2d) is a supported compound key column via the TEXT SENTINEL width 0
    // (the device fold reads the row's blob span and hashes it, rather than reading fixed words).
    assert_eq!(key_column_width_words(SqlType::Text), Some(0));
}

#[test]
fn compound_primary_key_over_i64_columns_enforces_tuple_uniqueness() {
    // COMPOUND KEYS (wider types, Stage 2a): a compound PK over i64 (Int8/Timestamp) columns — and a
    // MIXED int4+int8 key — is accepted and enforces TUPLE uniqueness through the authoritative
    // device predicate. The GPU sweep extends the same contract through b128/text keys.
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE ct (a INT8, b INT8, v INT, PRIMARY KEY (a, b))",
    )
    .unwrap();
    e.execute_text(2, "INSERT INTO ct VALUES (5000000000, 1, 10)")
        .unwrap();
    e.execute_text(3, "INSERT INTO ct VALUES (5000000000, 2, 20)")
        .unwrap(); // same a, different b -> OK
    e.execute_text(4, "INSERT INTO ct VALUES (9000000000, 1, 30)")
        .unwrap(); // different a, same b -> OK
    let err = e
        .execute_text(5, "INSERT INTO ct VALUES (5000000000, 1, 99)")
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("duplicate key value"),
        "duplicate i64 compound tuple raises 23505, got {err:?}"
    );

    // MIXED int4 + int8 compound PK.
    e.execute_text(
        6,
        "CREATE TABLE mt (a INT, b INT8, v INT, PRIMARY KEY (a, b))",
    )
    .unwrap();
    e.execute_text(7, "INSERT INTO mt VALUES (1, 8000000000, 0)")
        .unwrap();
    e.execute_text(8, "INSERT INTO mt VALUES (1, 8000000001, 0)")
        .unwrap(); // distinct b -> OK
    assert!(e
        .execute_text(9, "INSERT INTO mt VALUES (1, 8000000000, 5)")
        .is_err());

    // COMPOUND KEYS (wider types, Stage 2c): b128 (Uuid / Numeric) key columns are now ACCEPTED and
    // enforce tuple uniqueness.
    e.execute_text(
        10,
        "CREATE TABLE ut (a INT, u UUID, v INT, PRIMARY KEY (a, u))",
    )
    .unwrap();
    e.execute_text(
        11,
        "INSERT INTO ut VALUES (1, '00000000-0000-0000-0000-000000000001', 0)",
    )
    .unwrap();
    e.execute_text(
        12,
        "INSERT INTO ut VALUES (1, '00000000-0000-0000-0000-000000000002', 0)",
    )
    .unwrap(); // distinct uuid -> OK
    assert!(e
        .execute_text(
            13,
            "INSERT INTO ut VALUES (1, '00000000-0000-0000-0000-000000000001', 9)"
        )
        .is_err()); // duplicate (a, u) tuple -> 23505

    e.execute_text(
        14,
        "CREATE TABLE nt (a INT, n NUMERIC(20,4), v INT, PRIMARY KEY (a, n))",
    )
    .unwrap();
    e.execute_text(15, "INSERT INTO nt VALUES (1, 1.5, 0)")
        .unwrap();
    e.execute_text(16, "INSERT INTO nt VALUES (1, 2.5, 0)")
        .unwrap(); // distinct -> OK
    assert!(e
        .execute_text(17, "INSERT INTO nt VALUES (1, 1.5, 9)")
        .is_err()); // duplicate (a, n) tuple -> 23505

    // COMPOUND KEYS (wider types, Stage 2d): a TEXT (variable-length) key column is now ACCEPTED and
    // enforces tuple uniqueness (each text column folds to one word = the FNV-1a hash of its bytes).
    e.execute_text(
        18,
        "CREATE TABLE tt (a INT, s TEXT, v INT, PRIMARY KEY (a, s))",
    )
    .unwrap();
    e.execute_text(19, "INSERT INTO tt VALUES (1, 'alpha', 0)")
        .unwrap();
    e.execute_text(20, "INSERT INTO tt VALUES (1, 'beta', 0)")
        .unwrap(); // distinct text -> OK
    assert!(e
        .execute_text(21, "INSERT INTO tt VALUES (1, 'alpha', 9)")
        .is_err()); // duplicate (a, s) tuple -> 23505
}

#[test]
fn compound_key_fingerprint_is_deterministic_and_order_sensitive() {
    use crate::engine_residency::compound_key_fingerprint as fp;
    // Deterministic: same tuple -> same fingerprint (host builder, needle, and append fold must agree).
    assert_eq!(fp(&[1, 2, 3]), fp(&[1, 2, 3]));
    // Order-sensitive: the key-column ORDER is part of the tuple identity.
    assert_ne!(fp(&[1, 2]), fp(&[2, 1]));
    // Distinct tuples differing in ONE column produce (near-always) distinct fingerprints; the
    // authoritative recheck restores exactness regardless, but a good mix keeps rechecks rare.
    assert_ne!(fp(&[1, 2]), fp(&[1, 3]));
    assert_ne!(fp(&[1, 2]), fp(&[5, 2]));
    // A single-element tuple is still folded (NOT the raw key) — compound keys never reuse the
    // single-column raw-key path, so there is no aliasing between the two representations.
    assert_ne!(fp(&[7]), 7);
}

#[test]
fn compound_primary_key_over_i32_section_columns_enforces_tuple_uniqueness() {
    // TYPE-COVERAGE #14 Track 3 (compound keys): a compound PRIMARY KEY / UNIQUE over i32-SECTION
    // columns (Int4/Date/Int2) is now DEVICE-NATIVE — the ordered key tuple folds to a surrogate
    // fingerprint that rides the i32 index; the count>0 recheck compares the FULL tuple, so
    // uniqueness is exact. `new_local_test_engine` uses the same device-authoritative validation
    // contract as production; the ignored GPU sweep adds wider-type coverage.

    // (1) The parser captures BOTH key columns (not just the first).
    let Command::CreateTable(create) =
        parse_command("CREATE TABLE t (a INT, b INT, PRIMARY KEY (a, b))").unwrap()
    else {
        panic!("expected CREATE TABLE");
    };
    let pk = create.primary_key.expect("primary key parsed");
    assert_eq!(pk.columns, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(pk.column, "a"); // the first key column (single-column back-compat)

    // (2) A compound PK over i32-section columns is ACCEPTED and the catalog records both columns.
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE t (a INT, b INT, PRIMARY KEY (a, b))")
        .unwrap();
    let table = e.relational_catalog_table("t").unwrap();
    let idx = table.indexes.iter().find(|i| i.primary_key).unwrap();
    assert_eq!(idx.key_columns, vec!["a".to_string(), "b".to_string()]);

    // (3) TUPLE uniqueness — not first-column-only, not second-column-only.
    e.execute_text(2, "INSERT INTO t (a, b) VALUES (1, 2)")
        .unwrap();
    // Same first column, different second -> DISTINCT tuple, allowed.
    e.execute_text(3, "INSERT INTO t (a, b) VALUES (1, 3)")
        .unwrap();
    // Same second column, different first -> DISTINCT tuple, allowed.
    e.execute_text(4, "INSERT INTO t (a, b) VALUES (5, 2)")
        .unwrap();
    // ORDER matters: (2,1) is a distinct tuple from (1,2), allowed (order-sensitive fingerprint).
    e.execute_text(5, "INSERT INTO t (a, b) VALUES (2, 1)")
        .unwrap();
    // Exact tuple repeat -> 23505.
    let err = e
        .execute_text(6, "INSERT INTO t (a, b) VALUES (1, 2)")
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("duplicate key value"),
        "duplicate compound tuple raises 23505, got {err:?}"
    );

    // (4) BOOL has a canonical one-bit resident fold too. Distinct boolean members remain
    // distinct tuples and an exact repeat is rejected by the device-authoritative validator.
    e.execute_text(7, "CREATE TABLE w (a INT, f BOOL, PRIMARY KEY (a, f))")
        .unwrap();
    e.execute_text(8, "INSERT INTO w (a, f) VALUES (1, TRUE), (1, FALSE)")
        .unwrap();
    let err = e
        .execute_text(9, "INSERT INTO w (a, f) VALUES (1, TRUE)")
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("duplicate key value"),
        "duplicate compound BOOL tuple raises 23505, got {err:?}"
    );

    // (5) The other compound entry points also work for i32-section keys (ADD PK / ADD UNIQUE /
    // CREATE UNIQUE INDEX) — and each is preflight-checked, so a rejection never poisons the engine.
    let e2 = Engine::new_local_test_engine();
    e2.execute_text(1, "CREATE TABLE k (a INT, b INT, c INT)")
        .unwrap();
    e2.execute_text(
        2,
        "ALTER TABLE ONLY public.k ADD CONSTRAINT k_pkey PRIMARY KEY (a, b)",
    )
    .unwrap();
    e2.execute_text(
        3,
        "ALTER TABLE ONLY public.k ADD CONSTRAINT k_bc_key UNIQUE (b, c)",
    )
    .unwrap();
    e2.execute_text(4, "INSERT INTO k (a, b, c) VALUES (1, 2, 3)")
        .unwrap();
    // Violates the (a,b) PK.
    assert!(e2
        .execute_text(5, "INSERT INTO k (a, b, c) VALUES (1, 2, 9)")
        .is_err());
    // Violates the (b,c) UNIQUE index (distinct (a,b)).
    assert!(e2
        .execute_text(6, "INSERT INTO k (a, b, c) VALUES (7, 2, 3)")
        .is_err());
    // Distinct on both compound keys -> allowed.
    e2.execute_text(7, "INSERT INTO k (a, b, c) VALUES (7, 8, 9)")
        .unwrap();

    // (6) DDL integrity: a column that participates in a compound key cannot be dropped.
    assert!(e2
        .execute_text(8, "ALTER TABLE ONLY public.k DROP COLUMN b")
        .is_err());

    // (7) A single-column PRIMARY KEY still works end-to-end (no regression).
    e.execute_text(10, "CREATE TABLE s (id INT PRIMARY KEY, v INT)")
        .unwrap();
    e.execute_text(11, "INSERT INTO s (id, v) VALUES (1, 10), (2, 20)")
        .unwrap();
    assert!(e
        .execute_text(12, "INSERT INTO s (id, v) VALUES (1, 99)")
        .is_err());
    let table = e.relational_catalog_table("s").unwrap();
    let pk = table.indexes.iter().find(|i| i.primary_key).unwrap();
    assert_eq!(pk.key_columns, vec!["id".to_string()]);
}
