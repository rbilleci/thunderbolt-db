use super::*;

#[test]
fn relational_sql_create_insert_select_uses_mvcc_execution_path() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT name, id FROM people WHERE id = 2 ORDER BY name ASC LIMIT 1")
            .unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();

    assert_eq!(
        result
            .columns
            .iter()
            .map(RelationalColumn::as_column_def)
            .collect::<Vec<_>>(),
        vec![
            ColumnDef {
                name: "name".to_string(),
                ty: SqlType::Text,
                domain: None,
                default: None,
            },
            ColumnDef {
                name: "id".to_string(),
                ty: SqlType::Int4,
                domain: None,
                default: None,
            },
        ]
    );
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Text("Linus".to_string()), SqlValue::Int4(2)]]
    );
    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(
        result.fallback_reason,
        Some(FallbackReason::GpuMvccReadParityGap)
    );
    assert_eq!(
        *result.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("id".to_string()),
            predicate_op: Some(SelectFilterOp::Eq),
            order_column: "name".to_string(),
            descending: false,
            matched_keys: 1,
        }
    );
}

#[test]
fn relational_copy_rows_commit_through_engine_wal_mvcc() {
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE people (id INT PRIMARY KEY, name TEXT DEFAULT 'unknown'::text)",
    )
    .unwrap();

    let copy =
        gpu_db_sql::parse_copy_from_stdin("COPY people (id, name) FROM STDIN WITH (FORMAT csv)")
            .unwrap();
    let copy_columns = e.relational_copy_columns(&copy.table).unwrap();
    let rows = ["1,Ada", "2,O'Brien"]
        .into_iter()
        .map(|line| {
            gpu_db_sql::parse_copy_row(
                &copy_columns,
                copy.columns.as_deref().unwrap(),
                copy.options,
                line,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let (copied, profile) = e
        .execute_relational_copy_rows_profiled(2, &copy, rows)
        .unwrap();
    assert_eq!(copied, 2);
    assert_eq!(profile.rows, 2);
    assert!(profile.commit_total_micros >= profile.current_apply_total_micros);

    let default_copy = gpu_db_sql::parse_copy_from_stdin("COPY people (id) FROM STDIN").unwrap();
    let default_rows = ["3"]
        .into_iter()
        .map(|line| {
            gpu_db_sql::parse_copy_row(
                &copy_columns,
                default_copy.columns.as_deref().unwrap(),
                default_copy.options,
                line,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        e.execute_relational_copy_rows(3, &default_copy, default_rows)
            .unwrap(),
        1
    );

    let Command::Select(select) =
        parse_command("SELECT id, name FROM people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("O'Brien".to_string())],
            vec![SqlValue::Int4(3), SqlValue::Text("unknown".to_string())],
        ]
    );

    let Command::Select(indexed_select) =
        parse_command("SELECT id, name FROM people WHERE id = 2").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let indexed_result = e.execute_relational_select(&indexed_select).unwrap();
    assert_eq!(
        indexed_result.rows,
        vec![vec![
            SqlValue::Int4(2),
            SqlValue::Text("O'Brien".to_string())
        ]]
    );
    assert_eq!(
        *indexed_result.access_path,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "id".to_string(),
            matched_keys: 1,
        }
    );

    let err = e
        .execute_relational_copy_rows(
            4,
            &copy,
            vec![vec![
                SqlValue::Int4(1),
                SqlValue::Text("duplicate".to_string()),
            ]],
        )
        .unwrap_err();
    assert!(err.to_string().contains("duplicate key value"));
    let after_reject = e.execute_relational_select(&select).unwrap();
    assert_eq!(after_reject.rows, result.rows);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);
    let recovered_indexed = recovered
        .execute_relational_select(&indexed_select)
        .unwrap();
    assert_eq!(recovered_indexed.rows, indexed_result.rows);
    assert_eq!(recovered_indexed.access_path, indexed_result.access_path);
}

#[test]
fn relational_copy_ingests_null_marker_and_selects_back_null() {
    // M3 (doc 21) Slice G: the COPY NULL marker ingests as a SQL NULL. TEXT format: the unquoted `\N`.
    // CSV format: an UNQUOTED empty field (a QUOTED empty field is the empty STRING, not NULL).
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE t (id INT, name TEXT)")
        .unwrap();

    // TEXT format: row 1's name is `\N` (NULL); row 2's name is a real value.
    let copy = gpu_db_sql::parse_copy_from_stdin("COPY t (id, name) FROM STDIN").unwrap();
    let cols = e.relational_copy_columns(&copy.table).unwrap();
    let text_rows = ["1\t\\N", "2\tAda"]
        .into_iter()
        .map(|line| {
            gpu_db_sql::parse_copy_row(&cols, copy.columns.as_deref().unwrap(), copy.options, line)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        text_rows[0],
        vec![SqlValue::Int4(1), SqlValue::Null],
        "COPY TEXT \\N ingests as NULL"
    );
    assert_eq!(
        text_rows[1],
        vec![SqlValue::Int4(2), SqlValue::Text("Ada".to_string())]
    );
    assert_eq!(
        e.execute_relational_copy_rows(2, &copy, text_rows).unwrap(),
        2
    );

    // CSV format: `3,` -> unquoted empty name -> NULL; `4,""` -> quoted empty name -> the empty string.
    let csv = gpu_db_sql::parse_copy_from_stdin("COPY t (id, name) FROM STDIN WITH (FORMAT csv)")
        .unwrap();
    // `3,` unquoted empty -> NULL; `4,""` quoted empty -> empty string; `"5",` a QUOTED first field then
    // an UNQUOTED empty -> NULL (regression: the per-field `quoted` flag must reset across the delimiter,
    // else the empty field after a quoted one is mis-read as a quoted empty string).
    let csv_rows = ["3,", "4,\"\"", "\"5\","]
        .into_iter()
        .map(|line| {
            gpu_db_sql::parse_copy_row(&cols, csv.columns.as_deref().unwrap(), csv.options, line)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        csv_rows[0],
        vec![SqlValue::Int4(3), SqlValue::Null],
        "an UNQUOTED empty CSV field is NULL"
    );
    assert_eq!(
        csv_rows[1],
        vec![SqlValue::Int4(4), SqlValue::Text(String::new())],
        "a QUOTED empty CSV field is the empty string, not NULL"
    );
    assert_eq!(
        csv_rows[2],
        vec![SqlValue::Int4(5), SqlValue::Null],
        "an UNQUOTED empty field after a QUOTED field is still NULL (per-field quoted reset)"
    );
    assert_eq!(
        e.execute_relational_copy_rows(3, &csv, csv_rows).unwrap(),
        3
    );

    // The \N-ingested row selects back as NULL (the store round-trips it).
    let Command::Select(select) = parse_command("SELECT name FROM t WHERE id = 1").unwrap() else {
        panic!("expected SELECT plan");
    };
    assert_eq!(
        e.execute_relational_select(&select).unwrap().rows,
        vec![vec![SqlValue::Null]],
        "COPY \\N stored and selects back as NULL"
    );
}

#[test]
fn relational_copy_round_trips_all_column_types_and_null() {
    // M3 (doc 21) Track A.6: COPY into a table with int8/numeric/bool/date/timestamp/uuid columns
    // round-trips every value AND a `\N` NULL per type through the COPY-to-engine bridge
    // (render_sql_value_literal -> re-parsed INSERT -> store). Previously the render errored on any
    // non-int4/text/Null column ("supports int4/text rows only").
    let mut e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE tt (id INT, big BIGINT, amt NUMERIC(12,2), flag BOOL, d DATE, ts TIMESTAMP, u UUID)",
    )
    .unwrap();
    let copy =
        gpu_db_sql::parse_copy_from_stdin("COPY tt (id, big, amt, flag, d, ts, u) FROM STDIN")
            .unwrap();
    let cols = e.relational_copy_columns(&copy.table).unwrap();
    // Row 1: real values for every type (big > i32 to prove the int8 round-trip; numeric scale 2;
    // timestamp with sub-second precision). Row 2: `\N` (NULL) for every typed column.
    let lines = [
        "1\t5000000000\t1234.56\tt\t2024-01-15\t2024-01-15 10:30:00.123456\t00000000-0000-0000-0000-000000000001",
        "2\t\\N\t\\N\t\\N\t\\N\t\\N\t\\N",
    ];
    let rows = lines
        .into_iter()
        .map(|line| {
            gpu_db_sql::parse_copy_row(&cols, copy.columns.as_deref().unwrap(), copy.options, line)
                .unwrap()
        })
        .collect::<Vec<_>>();
    // The COPY parse produced the expected typed values (row 1) and a NULL per typed column (row 2).
    let parsed_row1 = rows[0].clone();
    assert_eq!(
        parsed_row1[1],
        SqlValue::Int8(5_000_000_000),
        "big parses as int8 (> i32)"
    );
    assert_eq!(
        parsed_row1[2],
        SqlValue::Numeric(Decimal128::new(123_456, 2)),
        "amt parses as numeric(.,2)"
    );
    assert_eq!(parsed_row1[3], SqlValue::Bool(true), "flag parses as bool");
    let all_null_row2 = vec![
        SqlValue::Int4(2),
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Null,
        SqlValue::Null,
    ];
    assert_eq!(
        rows[1], all_null_row2,
        "row 2 \\N -> NULL for every typed column"
    );

    // Ingest through the engine (render_relational_insert -> render_sql_value_literal per cell).
    assert_eq!(e.execute_relational_copy_rows(3, &copy, rows).unwrap(), 2);

    // SELECT back: the render -> re-parse -> store round-trip preserves every value and every NULL.
    let select_row = |id: i32| {
        let Command::Select(select) = parse_command(&format!(
            "SELECT id, big, amt, flag, d, ts, u FROM tt WHERE id = {id}"
        ))
        .unwrap() else {
            panic!("expected SELECT plan");
        };
        e.execute_relational_select(&select).unwrap().rows
    };
    assert_eq!(
        select_row(1),
        vec![parsed_row1.clone()],
        "row 1 round-trips through COPY render -> INSERT -> store (every type)"
    );
    assert_eq!(
        select_row(2),
        vec![all_null_row2.clone()],
        "row 2 round-trips as all-NULL (a \\N per type)"
    );

    // WAL-REPLAY round trip — the path this slice actually feeds. The COPY's WAL payload is the RENDERED
    // INSERT (render_sql_value_literal per cell); recovery RE-PARSES it (parse_sql_value + coerce). So a
    // persist+recover proves the rendered literal for every type re-parses to the identical stored value
    // (the live apply uses the original typed row and would mask a wrong render).
    let path = test_wal_path("copy_all_types_roundtrip");
    e.persist_durable_wal_to_file(&path).unwrap();
    let recovered = Engine::recover_from_durable_wal_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let recovered_row = |id: i32| {
        let Command::Select(select) = parse_command(&format!(
            "SELECT id, big, amt, flag, d, ts, u FROM tt WHERE id = {id}"
        ))
        .unwrap() else {
            panic!("expected SELECT plan");
        };
        recovered.execute_relational_select(&select).unwrap().rows
    };
    assert_eq!(
        recovered_row(1),
        vec![parsed_row1],
        "WAL replay re-parses the rendered literals to the identical values (every type)"
    );
    assert_eq!(
        recovered_row(2),
        vec![all_null_row2],
        "WAL replay re-parses the rendered \\N to NULL for every type"
    );
}

#[test]
fn relational_column_defaults_fill_omitted_insert_columns_and_replay() {
    let e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE default_people (id INT, name TEXT DEFAULT 'unknown'::text, bucket INT DEFAULT 7)",
        )
        .unwrap();
    e.execute_text(2, "INSERT INTO default_people (id) VALUES (1)")
        .unwrap();
    e.execute_text(
        3,
        "ALTER TABLE ONLY public.default_people ALTER COLUMN name SET DEFAULT 'changed'::text",
    )
    .unwrap();
    e.execute_text(
        4,
        "INSERT INTO default_people (id, name, bucket) VALUES (2, 'Ada', 9)",
    )
    .unwrap();
    e.execute_text(5, "INSERT INTO default_people (id) VALUES (3)")
        .unwrap();
    e.execute_text(
        6,
        "ALTER TABLE ONLY public.default_people ALTER COLUMN name DROP DEFAULT",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id, name, bucket FROM default_people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("unknown".to_string()),
                SqlValue::Int4(7),
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Ada".to_string()),
                SqlValue::Int4(9),
            ],
            vec![
                SqlValue::Int4(3),
                SqlValue::Text("changed".to_string()),
                SqlValue::Int4(7),
            ],
        ]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let table = recovered
        .relational_catalog_table("default_people")
        .unwrap();
    assert_eq!(table.columns[1].default, None);
    assert_eq!(
        table.columns[2].default,
        Some(ColumnDefault::Literal(SqlValue::Int4(7)))
    );
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);

    let err = e
        .execute_text(7, "INSERT INTO default_people (id, bucket) VALUES (4, 8)")
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("INSERT must provide every column without a default"));
}

#[test]
fn relational_add_column_default_rewrites_rows_and_replays() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE default_people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO default_people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    e.execute_text(
        3,
        "ALTER TABLE ONLY public.default_people ADD COLUMN bucket INT DEFAULT 7",
    )
    .unwrap();
    e.execute_text(
        4,
        "INSERT INTO default_people (id, name) VALUES (3, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id, name, bucket FROM default_people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("Ada".to_string()),
                SqlValue::Int4(7),
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Linus".to_string()),
                SqlValue::Int4(7),
            ],
            vec![
                SqlValue::Int4(3),
                SqlValue::Text("Grace".to_string()),
                SqlValue::Int4(7),
            ],
        ]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let table = recovered
        .relational_catalog_table("default_people")
        .unwrap();
    assert_eq!(table.columns.len(), 3);
    assert_eq!(table.columns[2].name, "bucket");
    assert_eq!(table.columns[2].attnum, 3);
    assert_eq!(
        table.columns[2].default,
        Some(ColumnDefault::Literal(SqlValue::Int4(7)))
    );
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);

    let duplicate = e
        .execute_text(
            5,
            "ALTER TABLE public.default_people ADD COLUMN bucket INT DEFAULT 9",
        )
        .unwrap_err();
    assert!(duplicate.to_string().contains("already exists"));
    let no_default = Engine::new_local();
    no_default
        .execute_text(1, "CREATE TABLE default_people (id INT)")
        .unwrap();
    let unsupported = no_default
        .execute_text(2, "ALTER TABLE public.default_people ADD COLUMN note TEXT")
        .unwrap_err();
    assert!(
        unsupported
            .to_string()
            .contains("ADD COLUMN requires a supported DEFAULT"),
        "{unsupported}"
    );
}

#[test]
fn relational_add_column_sequence_default_rewrites_rows_and_replays() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE default_people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO default_people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    e.execute_text(3, "CREATE SEQUENCE public.default_bucket_seq")
        .unwrap();
    e.execute_text(
            4,
            "ALTER TABLE ONLY public.default_people ADD COLUMN bucket INT DEFAULT nextval('public.default_bucket_seq'::regclass)",
        )
        .unwrap();
    e.execute_text(
        5,
        "INSERT INTO default_people (id, name) VALUES (3, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id, name, bucket FROM default_people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![
                SqlValue::Int4(1),
                SqlValue::Text("Ada".to_string()),
                SqlValue::Int4(1),
            ],
            vec![
                SqlValue::Int4(2),
                SqlValue::Text("Linus".to_string()),
                SqlValue::Int4(2),
            ],
            vec![
                SqlValue::Int4(3),
                SqlValue::Text("Grace".to_string()),
                SqlValue::Int4(3),
            ],
        ]
    );
    let sequence = e.relational_catalog_sequence("default_bucket_seq").unwrap();
    assert_eq!(sequence.last_value, 3);
    assert!(sequence.is_called);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_table = recovered
        .relational_catalog_table("default_people")
        .unwrap();
    assert_eq!(
        recovered_table.columns[2].default,
        Some(ColumnDefault::SequenceNextVal {
            sequence: "default_bucket_seq".to_string(),
            create_if_missing: false,
        })
    );
    assert_eq!(
        recovered
            .relational_catalog_sequence("default_bucket_seq")
            .unwrap()
            .last_value,
        3
    );
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);

    e.execute_text(
        6,
        "ALTER TABLE ONLY public.default_people ALTER COLUMN bucket DROP DEFAULT",
    )
    .unwrap();
    let table = e.relational_catalog_table("default_people").unwrap();
    assert_eq!(table.columns[2].default, None);
    let sequence = e.relational_catalog_sequence("default_bucket_seq").unwrap();
    assert_eq!(sequence.last_value, 3);
    assert!(sequence.is_called);
    let missing_default = e
        .execute_text(
            7,
            "INSERT INTO default_people (id, name) VALUES (4, 'Barbara')",
        )
        .unwrap_err();
    assert!(missing_default
        .to_string()
        .contains("INSERT must provide every column without a default"));

    let recovered_after_drop = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_table = recovered_after_drop
        .relational_catalog_table("default_people")
        .unwrap();
    assert_eq!(recovered_table.columns[2].default, None);
    let recovered_sequence = recovered_after_drop
        .relational_catalog_sequence("default_bucket_seq")
        .unwrap();
    assert_eq!(recovered_sequence.last_value, 3);
    assert!(recovered_sequence.is_called);

    let missing = e
            .execute_text(
                8,
                "ALTER TABLE default_people ADD COLUMN missing_bucket INT DEFAULT nextval('missing_bucket_seq'::regclass)",
            )
            .unwrap_err();
    assert!(missing
        .to_string()
        .contains("sequence \"missing_bucket_seq\" does not exist"));
    assert!(e
        .relational_catalog_table("default_people")
        .unwrap()
        .columns
        .iter()
        .all(|column| column.name != "missing_bucket"));

    let table_target = e
            .execute_text(9, "CREATE TABLE default_target_table (id INT)")
            .and_then(|_| {
                e.execute_text(
                    10,
                    "ALTER TABLE default_people ADD COLUMN bad_bucket INT DEFAULT nextval('default_target_table'::regclass)",
                )
            })
            .unwrap_err();
    assert!(table_target
        .to_string()
        .contains("relation \"default_target_table\" is not a sequence"));
}

#[test]
fn relational_drop_column_rewrites_rows_and_replays() {
    let e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE drop_column_people (id INT, name TEXT, bucket INT DEFAULT 7)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO drop_column_people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    e.execute_text(
        3,
        "COMMENT ON COLUMN public.drop_column_people.name IS 'drop me'",
    )
    .unwrap();
    e.execute_text(
        4,
        "COMMENT ON COLUMN public.drop_column_people.bucket IS 'keep me'",
    )
    .unwrap();
    e.execute_text(
        5,
        "ALTER TABLE ONLY public.drop_column_people DROP COLUMN name",
    )
    .unwrap();
    e.execute_text(6, "INSERT INTO drop_column_people (id) VALUES (3)")
        .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id, bucket FROM drop_column_people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Int4(7)],
            vec![SqlValue::Int4(2), SqlValue::Int4(7)],
            vec![SqlValue::Int4(3), SqlValue::Int4(7)],
        ]
    );
    let table = e.relational_catalog_table("drop_column_people").unwrap();
    assert_eq!(
        table
            .columns
            .iter()
            .map(|column| (column.name.as_str(), column.attnum))
            .collect::<Vec<_>>(),
        vec![("id", 1), ("bucket", 2)]
    );
    assert_eq!(
        e.relational_column_comment("drop_column_people", 2)
            .as_deref(),
        Some("keep me")
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);
    assert_eq!(
        recovered
            .relational_column_comment("drop_column_people", 2)
            .as_deref(),
        Some("keep me")
    );

    let missing_column = e
        .execute_text(
            7,
            "ALTER TABLE public.drop_column_people DROP COLUMN missing_name",
        )
        .unwrap_err();
    assert!(missing_column.to_string().contains("does not exist"));

    let constrained = Engine::new_local();
    constrained
        .execute_text(
            1,
            "CREATE TABLE constrained_people (id INT PRIMARY KEY, name TEXT)",
        )
        .unwrap();
    let dependency = constrained
        .execute_text(2, "ALTER TABLE constrained_people DROP COLUMN id")
        .unwrap_err();
    assert!(dependency.to_string().contains("depends on it"));
}

#[test]
fn relational_rename_table_rewrites_rows_catalog_comments_and_replays() {
    let e = Engine::new_local();
    e.execute_text(
            1,
            "CREATE TABLE rename_table_people (id INT PRIMARY KEY, name TEXT UNIQUE, bucket INT DEFAULT 7)",
        )
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO rename_table_people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    e.execute_text(
        3,
        "COMMENT ON TABLE public.rename_table_people IS 'old table'",
    )
    .unwrap();
    e.execute_text(
        4,
        "COMMENT ON COLUMN public.rename_table_people.name IS 'person name'",
    )
    .unwrap();
    e.execute_text(
            5,
            "COMMENT ON CONSTRAINT rename_table_people_pkey ON public.rename_table_people IS 'primary id'",
        )
        .unwrap();
    e.execute_text(
        6,
        "ALTER TABLE ONLY public.rename_table_people RENAME TO renamed_table_people",
    )
    .unwrap();
    e.execute_text(
        7,
        "INSERT INTO renamed_table_people (id, name) VALUES (3, 'Grace')",
    )
    .unwrap();

    assert!(e.relational_catalog_table("rename_table_people").is_none());
    let table = e.relational_catalog_table("renamed_table_people").unwrap();
    assert_eq!(table.name, "renamed_table_people");
    assert_eq!(
        table
            .indexes
            .iter()
            .map(|index| (index.name.as_str(), index.table.as_str()))
            .collect::<Vec<_>>(),
        vec![
            ("rename_table_people_pkey", "renamed_table_people"),
            ("rename_table_people_name_key", "renamed_table_people"),
        ]
    );
    assert_eq!(
        e.relational_table_comment("renamed_table_people")
            .as_deref(),
        Some("old table")
    );
    assert_eq!(
        e.relational_column_comment("renamed_table_people", 2)
            .as_deref(),
        Some("person name")
    );
    assert_eq!(
        e.relational_constraint_comment("renamed_table_people", "rename_table_people_pkey")
            .as_deref(),
        Some("primary id")
    );

    let Command::Select(select) =
        parse_command("SELECT id, name, bucket FROM renamed_table_people WHERE id = 2").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![vec![
            SqlValue::Int4(2),
            SqlValue::Text("Linus".to_string()),
            SqlValue::Int4(7),
        ]]
    );
    assert_eq!(
        *result.access_path,
        RelationalAccessPath::EqualityIndex {
            table: "renamed_table_people".to_string(),
            column: "id".to_string(),
            matched_keys: 1,
        }
    );

    let old_select = parse_command("SELECT id FROM rename_table_people WHERE id = 1").unwrap();
    let Command::Select(old_select) = old_select else {
        panic!("expected SELECT plan");
    };
    assert!(e.execute_relational_select(&old_select).is_err());

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);
    assert_eq!(
        recovered
            .relational_table_comment("renamed_table_people")
            .as_deref(),
        Some("old table")
    );
    assert_eq!(
        recovered
            .relational_constraint_comment("renamed_table_people", "rename_table_people_pkey")
            .as_deref(),
        Some("primary id")
    );

    let duplicate = e
        .execute_text(
            8,
            "ALTER TABLE renamed_table_people RENAME TO renamed_table_people",
        )
        .unwrap_err();
    assert!(duplicate.to_string().contains("already exists"));

    let view_engine = Engine::new_local();
    view_engine
        .execute_text(1, "CREATE TABLE rename_table_base (id INT, name TEXT)")
        .unwrap();
    view_engine
        .execute_text(
            2,
            "CREATE VIEW rename_table_view AS SELECT id, name FROM rename_table_base",
        )
        .unwrap();
    let dependency = view_engine
        .execute_text(3, "ALTER TABLE rename_table_base RENAME TO renamed_base")
        .unwrap_err();
    assert!(dependency.to_string().contains("view depends on it"));
    let view_err = view_engine
        .execute_text(4, "ALTER TABLE rename_table_view RENAME TO renamed_view")
        .unwrap_err();
    assert!(view_err.to_string().contains("is not a table"));
}

#[test]
fn relational_rename_column_updates_catalog_indexes_and_replays() {
    let e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE rename_column_people (id INT PRIMARY KEY, name TEXT DEFAULT 'unknown')",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO rename_column_people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    e.execute_text(
        3,
        "COMMENT ON COLUMN public.rename_column_people.name IS 'display name'",
    )
    .unwrap();
    e.execute_text(
        4,
        "ALTER TABLE ONLY public.rename_column_people RENAME COLUMN name TO display_name",
    )
    .unwrap();
    e.execute_text(
        5,
        "ALTER TABLE public.rename_column_people RENAME COLUMN id TO person_id",
    )
    .unwrap();
    e.execute_text(6, "INSERT INTO rename_column_people (person_id) VALUES (3)")
        .unwrap();

    let Command::Select(select) = parse_command(
        "SELECT person_id, display_name FROM rename_column_people WHERE person_id = 2",
    )
    .unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())]]
    );
    assert_eq!(
        *result.access_path,
        RelationalAccessPath::EqualityIndex {
            table: "rename_column_people".to_string(),
            column: "person_id".to_string(),
            matched_keys: 1,
        }
    );
    let table = e.relational_catalog_table("rename_column_people").unwrap();
    assert_eq!(
        table
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        vec!["person_id", "display_name"]
    );
    assert_eq!(table.indexes[0].column, "person_id");
    assert_eq!(
        e.relational_column_comment("rename_column_people", 2)
            .as_deref(),
        Some("display name")
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);
    assert_eq!(
        recovered
            .relational_catalog_table("rename_column_people")
            .unwrap()
            .indexes[0]
            .column,
        "person_id"
    );
    assert_eq!(
        recovered
            .relational_column_comment("rename_column_people", 2)
            .as_deref(),
        Some("display name")
    );

    let duplicate = e
        .execute_text(
            7,
            "ALTER TABLE public.rename_column_people RENAME COLUMN display_name TO person_id",
        )
        .unwrap_err();
    assert!(duplicate.to_string().contains("already exists"));

    let view_engine = Engine::new_local();
    view_engine
        .execute_text(1, "CREATE TABLE rename_base (id INT, name TEXT)")
        .unwrap();
    view_engine
        .execute_text(
            2,
            "CREATE VIEW rename_view AS SELECT id, name FROM rename_base",
        )
        .unwrap();
    let view_err = view_engine
        .execute_text(
            3,
            "ALTER TABLE public.rename_view RENAME COLUMN name TO display_name",
        )
        .unwrap_err();
    assert!(view_err.to_string().contains("is not a table"));
}

#[test]
fn relational_rename_constraint_updates_index_comments_and_replays() {
    let e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE rename_constraint_people (id INT PRIMARY KEY, name TEXT UNIQUE)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO rename_constraint_people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    e.execute_text(
            3,
            "COMMENT ON CONSTRAINT rename_constraint_people_pkey ON public.rename_constraint_people IS 'old primary key'",
        )
        .unwrap();
    e.execute_text(
        4,
        "COMMENT ON INDEX public.rename_constraint_people_name_key IS 'old unique index'",
    )
    .unwrap();
    e.execute_text(
            5,
            "ALTER TABLE ONLY public.rename_constraint_people RENAME CONSTRAINT rename_constraint_people_pkey TO rename_constraint_people_id_pkey",
        )
        .unwrap();
    e.execute_text(
            6,
            "ALTER TABLE public.rename_constraint_people RENAME CONSTRAINT rename_constraint_people_name_key TO rename_constraint_people_display_name_key",
        )
        .unwrap();

    let indexes = e
        .relational_catalog_table("rename_constraint_people")
        .unwrap()
        .indexes
        .clone();
    assert_eq!(
        indexes
            .iter()
            .map(|index| index.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "rename_constraint_people_id_pkey",
            "rename_constraint_people_display_name_key"
        ]
    );
    assert_eq!(
        e.relational_constraint_comment(
            "rename_constraint_people",
            "rename_constraint_people_id_pkey"
        )
        .as_deref(),
        Some("old primary key")
    );
    assert_eq!(
        e.relational_index_comment("rename_constraint_people_display_name_key")
            .as_deref(),
        Some("old unique index")
    );
    assert_eq!(
        e.relational_constraint_comment(
            "rename_constraint_people",
            "rename_constraint_people_pkey"
        ),
        None
    );
    let duplicate_insert = e
        .execute_text(
            7,
            "INSERT INTO rename_constraint_people (id, name) VALUES (3, 'Ada')",
        )
        .unwrap_err();
    assert!(
        duplicate_insert
            .to_string()
            .contains("rename_constraint_people_display_name_key"),
        "{duplicate_insert}"
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_indexes = &recovered
        .relational_catalog_table("rename_constraint_people")
        .unwrap()
        .indexes;
    assert_eq!(recovered_indexes, &indexes);
    assert_eq!(
        recovered
            .relational_constraint_comment(
                "rename_constraint_people",
                "rename_constraint_people_id_pkey"
            )
            .as_deref(),
        Some("old primary key")
    );
    assert_eq!(
        recovered
            .relational_index_comment("rename_constraint_people_display_name_key")
            .as_deref(),
        Some("old unique index")
    );

    let duplicate_target = e
            .execute_text(
                8,
                "ALTER TABLE public.rename_constraint_people RENAME CONSTRAINT rename_constraint_people_id_pkey TO rename_constraint_people_display_name_key",
            )
            .unwrap_err();
    assert!(duplicate_target.to_string().contains("already exists"));
    let missing_constraint = e
            .execute_text(
                9,
                "ALTER TABLE public.rename_constraint_people RENAME CONSTRAINT missing_constraint TO renamed_missing",
            )
            .unwrap_err();
    assert!(missing_constraint.to_string().contains("does not exist"));

    let view_engine = Engine::new_local();
    view_engine
        .execute_text(1, "CREATE TABLE rename_constraint_base (id INT, name TEXT)")
        .unwrap();
    view_engine
        .execute_text(
            2,
            "CREATE VIEW rename_constraint_view AS SELECT id, name FROM rename_constraint_base",
        )
        .unwrap();
    let view_err = view_engine
            .execute_text(
                3,
                "ALTER TABLE public.rename_constraint_view RENAME CONSTRAINT missing_constraint TO renamed_missing",
            )
            .unwrap_err();
    assert!(view_err.to_string().contains("is not a table"));
}

#[test]
fn relational_sql_delete_uses_wal_before_visibility_and_rebuilds_from_wal() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Ada Lovelace')",
        )
        .unwrap();
    e.execute_text(3, "DELETE FROM people WHERE id = 2 OR name LIKE 'Ada%'")
        .unwrap();

    let Command::Select(select) = parse_command("SELECT id, name FROM people ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();

    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())]]
    );
    assert_eq!(e.durable_wal_records().len(), 3);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);

    let Command::Select(index_select) =
        parse_command("SELECT id FROM people WHERE id = 2").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let index_result = recovered.execute_relational_select(&index_select).unwrap();
    assert!(index_result.rows.is_empty());
    assert_eq!(
        *index_result.access_path,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "id".to_string(),
            matched_keys: 1,
        }
    );
}

#[test]
fn relational_sql_update_uses_wal_before_visibility_and_rebuilds_from_wal() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Ada Lovelace')",
        )
        .unwrap();
    e.execute_text(
        3,
        "UPDATE people SET name = 'Updated' WHERE id = 2 OR name LIKE 'Ada%'",
    )
    .unwrap();

    let Command::Select(select) = parse_command("SELECT id, name FROM people ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();

    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Updated".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("Updated".to_string())],
            vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            vec![SqlValue::Int4(4), SqlValue::Text("Updated".to_string())],
        ]
    );
    assert_eq!(e.durable_wal_records().len(), 3);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);

    let Command::Select(index_select) =
        parse_command("SELECT id FROM people WHERE name = 'Updated' ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let index_result = recovered.execute_relational_select(&index_select).unwrap();
    assert_eq!(
        index_result.rows,
        vec![
            vec![SqlValue::Int4(1)],
            vec![SqlValue::Int4(2)],
            vec![SqlValue::Int4(4)],
        ]
    );
    assert_eq!(
        *index_result.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("name".to_string()),
            predicate_op: Some(SelectFilterOp::Eq),
            order_column: "id".to_string(),
            descending: false,
            matched_keys: 3,
        }
    );
}

#[test]
fn relational_sql_views_select_and_replay_from_wal() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace')",
    )
    .unwrap();
    e.execute_text(
        3,
        "CREATE VIEW public.active_people AS SELECT id, name FROM people WHERE id > 1 ORDER BY id",
    )
    .unwrap();

    let view = e.relational_catalog_view("active_people").unwrap();
    assert_eq!(view.name, "active_people");
    assert_eq!(
        view.definition,
        "SELECT id, name FROM people WHERE id > 1 ORDER BY id"
    );

    let Command::Select(select) = parse_command("SELECT * FROM active_people").unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
        ]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_view("active_people")
            .unwrap()
            .definition,
        "SELECT id, name FROM people WHERE id > 1 ORDER BY id"
    );
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);

    let Command::Select(filtered_view_select) =
        parse_command("SELECT id FROM active_people WHERE id = 2").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    assert!(recovered
        .execute_relational_select(&filtered_view_select)
        .is_err());
}

#[test]
fn relational_sql_create_or_replace_view_replays_from_wal() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace')",
    )
    .unwrap();
    e.execute_text(
        3,
        "CREATE VIEW public.active_people AS SELECT id, name FROM people WHERE id > 1 ORDER BY id",
    )
    .unwrap();
    e.execute_text(
        4,
        "COMMENT ON VIEW public.active_people IS 'active people view'",
    )
    .unwrap();
    let oid = e.relational_catalog_view("active_people").unwrap().oid;
    e.execute_text(
            5,
            "CREATE OR REPLACE VIEW public.active_people AS SELECT id, name FROM people WHERE id > 2 ORDER BY id",
        )
        .unwrap();

    let view = e.relational_catalog_view("active_people").unwrap();
    assert_eq!(view.oid, oid);
    assert_eq!(
        view.definition,
        "SELECT id, name FROM people WHERE id > 2 ORDER BY id"
    );
    assert_eq!(
        e.relational_view_comment("active_people").as_deref(),
        Some("active people view")
    );

    let Command::Select(select) = parse_command("SELECT * FROM active_people").unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())]]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_view("active_people")
            .unwrap()
            .definition,
        "SELECT id, name FROM people WHERE id > 2 ORDER BY id"
    );
    assert_eq!(
        recovered
            .relational_view_comment("active_people")
            .as_deref(),
        Some("active people view")
    );
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);

    let replaced_err = e
        .execute_text(
            6,
            "CREATE OR REPLACE VIEW public.active_people AS SELECT * FROM missing_people",
        )
        .unwrap_err()
        .to_string();
    assert!(
        replaced_err.contains("relation \"missing_people\" does not exist"),
        "{replaced_err}"
    );
    assert_eq!(
        e.relational_catalog_view("active_people")
            .unwrap()
            .definition,
        "SELECT id, name FROM people WHERE id > 2 ORDER BY id"
    );
}

#[test]
fn relational_sql_layered_views_select_and_replay_from_wal() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace')",
    )
    .unwrap();
    e.execute_text(
        3,
        "CREATE VIEW public.active_people AS SELECT id, name FROM people WHERE id > 1 ORDER BY id",
    )
    .unwrap();
    e.execute_text(
        4,
        "CREATE VIEW public.active_people_names AS SELECT * FROM active_people",
    )
    .unwrap();
    e.execute_text(
        5,
        "COMMENT ON VIEW public.active_people_names IS 'layered active people'",
    )
    .unwrap();

    let Command::Select(select) = parse_command("SELECT * FROM active_people_names").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
        ]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_view("active_people_names")
            .unwrap()
            .definition,
        "SELECT * FROM active_people"
    );
    assert_eq!(
        recovered
            .relational_view_comment("active_people_names")
            .as_deref(),
        Some("layered active people")
    );
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);

    let replace_parent_err = recovered
            .execute_text(
                6,
                "CREATE OR REPLACE VIEW public.active_people AS SELECT id, name FROM people WHERE id > 2 ORDER BY id",
            )
            .unwrap_err()
            .to_string();
    assert!(
        replace_parent_err.contains("cannot replace view because another view depends on it"),
        "{replace_parent_err}"
    );

    let rename_parent_err = recovered
        .execute_text(
            7,
            "ALTER VIEW public.active_people RENAME TO active_people_base",
        )
        .unwrap_err()
        .to_string();
    assert!(
        rename_parent_err.contains("cannot rename view"),
        "{rename_parent_err}"
    );

    let drop_parent_err = recovered
        .execute_text(8, "DROP VIEW public.active_people")
        .unwrap_err()
        .to_string();
    assert!(
        drop_parent_err.contains("cannot drop view"),
        "{drop_parent_err}"
    );

    recovered
        .execute_text(
            9,
            "DROP VIEW public.active_people_names, public.active_people",
        )
        .unwrap();
    assert!(recovered
        .relational_catalog_view("active_people_names")
        .is_none());
    assert!(recovered.relational_catalog_view("active_people").is_none());
}

#[test]
fn relational_sql_rename_view_replays_from_wal() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace')",
    )
    .unwrap();
    e.execute_text(
        3,
        "CREATE VIEW public.active_people AS SELECT id, name FROM people WHERE id > 1 ORDER BY id",
    )
    .unwrap();
    e.execute_text(
        4,
        "COMMENT ON VIEW public.active_people IS 'active people view'",
    )
    .unwrap();
    let oid = e.relational_catalog_view("active_people").unwrap().oid;
    e.execute_text(
        5,
        "ALTER VIEW public.active_people RENAME TO renamed_people",
    )
    .unwrap();

    assert!(e.relational_catalog_view("active_people").is_none());
    let view = e.relational_catalog_view("renamed_people").unwrap();
    assert_eq!(view.oid, oid);
    assert_eq!(view.name, "renamed_people");
    assert_eq!(
        view.definition,
        "SELECT id, name FROM people WHERE id > 1 ORDER BY id"
    );
    assert_eq!(
        e.relational_view_comment("renamed_people").as_deref(),
        Some("active people view")
    );
    assert_eq!(e.relational_view_comment("active_people"), None);

    let Command::Select(select) = parse_command("SELECT * FROM renamed_people").unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
        ]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_catalog_view("active_people").is_none());
    assert_eq!(
        recovered
            .relational_catalog_view("renamed_people")
            .unwrap()
            .oid,
        oid
    );
    assert_eq!(
        recovered
            .relational_view_comment("renamed_people")
            .as_deref(),
        Some("active people view")
    );
    let recovered_result = recovered.execute_relational_select(&select).unwrap();
    assert_eq!(recovered_result.rows, result.rows);

    let missing = e
        .execute_text(6, "ALTER VIEW active_people RENAME TO missing_rename")
        .unwrap_err()
        .to_string();
    assert!(missing.contains("view \"active_people\" does not exist"));
    let duplicate = e
        .execute_text(7, "ALTER VIEW renamed_people RENAME TO people")
        .unwrap_err()
        .to_string();
    assert!(duplicate.contains("relation \"people\" already exists"));

    let table_target = e
        .execute_text(8, "ALTER VIEW people RENAME TO people_view")
        .unwrap_err()
        .to_string();
    assert!(table_target.contains("relation \"people\" is not a view"));
}

#[test]
fn relational_sql_drop_view_replays_from_wal() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    e.execute_text(
        3,
        "CREATE VIEW public.active_people AS SELECT id, name FROM people ORDER BY id",
    )
    .unwrap();
    e.execute_text(
        4,
        "CREATE VIEW public.other_people AS SELECT id, name FROM people WHERE id = 2",
    )
    .unwrap();
    e.execute_text(5, "DROP VIEW public.active_people, public.other_people")
        .unwrap();

    assert!(e.relational_catalog_view("active_people").is_none());
    assert!(e.relational_catalog_view("other_people").is_none());
    assert!(e.relational_catalog_table("people").is_some());

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_catalog_view("active_people").is_none());
    assert!(recovered.relational_catalog_view("other_people").is_none());

    let Command::Select(table_select) =
        parse_command("SELECT id, name FROM people ORDER BY id").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let table_result = recovered.execute_relational_select(&table_select).unwrap();
    assert_eq!(
        table_result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
        ]
    );

    e.execute_text(6, "DROP VIEW IF EXISTS active_people")
        .unwrap();
    let missing = e.execute_text(7, "DROP VIEW active_people").unwrap_err();
    assert!(missing
        .to_string()
        .contains("view \"active_people\" does not exist"));

    let table_target_engine = Engine::new_local();
    table_target_engine
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    let table_target = table_target_engine
        .execute_text(2, "DROP VIEW people")
        .unwrap_err();
    assert!(table_target.to_string().contains("not a view"));

    let preflight_engine = Engine::new_local();
    preflight_engine
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    preflight_engine
        .execute_text(
            2,
            "CREATE VIEW public.active_people AS SELECT id, name FROM people ORDER BY id",
        )
        .unwrap();
    preflight_engine
        .execute_text(
            3,
            "CREATE VIEW public.other_people AS SELECT id, name FROM people WHERE id = 2",
        )
        .unwrap();
    let missing_batch = preflight_engine
        .execute_text(4, "DROP VIEW active_people, missing_people")
        .unwrap_err();
    assert!(missing_batch
        .to_string()
        .contains("view \"missing_people\" does not exist"));
    assert!(preflight_engine
        .relational_catalog_view("active_people")
        .is_some());
    assert!(preflight_engine
        .relational_catalog_view("other_people")
        .is_some());

    let duplicate = preflight_engine
        .execute_text(5, "DROP VIEW active_people, active_people")
        .unwrap_err();
    assert!(duplicate
        .to_string()
        .contains("view \"active_people\" specified more than once"));
}

#[test]
fn relational_sql_sequence_catalog_objects_replay_from_wal() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE SEQUENCE public.people_seq")
        .unwrap();
    e.execute_text(3, "COMMENT ON SEQUENCE public.people_seq IS 'people ids'")
        .unwrap();

    let sequence = e.relational_catalog_sequence("people_seq").unwrap();
    assert_eq!(sequence.name, "people_seq");
    let oid = sequence.oid;
    assert_eq!(
        e.relational_sequence_comment("people_seq").as_deref(),
        Some("people ids")
    );
    e.execute_text(
        4,
        "ALTER SEQUENCE public.people_seq RENAME TO people_id_seq",
    )
    .unwrap();

    assert!(e.relational_catalog_sequence("people_seq").is_none());
    let renamed_sequence = e.relational_catalog_sequence("people_id_seq").unwrap();
    assert_eq!(renamed_sequence.name, "people_id_seq");
    assert_eq!(renamed_sequence.oid, oid);
    assert_eq!(
        e.relational_sequence_comment("people_id_seq").as_deref(),
        Some("people ids")
    );
    assert_eq!(e.relational_sequence_comment("people_seq"), None);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered
        .relational_catalog_sequence("people_seq")
        .is_none());
    assert_eq!(
        recovered
            .relational_catalog_sequence("people_id_seq")
            .unwrap()
            .oid,
        oid
    );
    assert_eq!(
        recovered
            .relational_sequence_comment("people_id_seq")
            .as_deref(),
        Some("people ids")
    );

    e.execute_text(5, "DROP SEQUENCE IF EXISTS missing_seq, people_id_seq")
        .unwrap();
    assert!(e.relational_catalog_sequence("people_id_seq").is_none());
    assert_eq!(e.relational_sequence_comment("people_id_seq"), None);

    let recovered_after_drop = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered_after_drop
        .relational_catalog_sequence("people_id_seq")
        .is_none());

    let boundary = Engine::new_local();
    boundary
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    let duplicate = boundary
        .execute_text(2, "CREATE SEQUENCE people")
        .unwrap_err();
    assert!(duplicate
        .to_string()
        .contains("relation \"people\" already exists"));
    let table_target = boundary
        .execute_text(3, "DROP SEQUENCE people")
        .unwrap_err();
    assert!(table_target.to_string().contains("not a sequence"));
    let missing = boundary
        .execute_text(4, "DROP SEQUENCE missing_seq")
        .unwrap_err();
    assert!(missing
        .to_string()
        .contains("sequence \"missing_seq\" does not exist"));

    let rename_boundary = Engine::new_local();
    rename_boundary
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    rename_boundary
        .execute_text(2, "CREATE SEQUENCE people_seq")
        .unwrap();
    let duplicate_rename = rename_boundary
        .execute_text(3, "ALTER SEQUENCE people_seq RENAME TO people_seq")
        .unwrap_err();
    assert!(duplicate_rename
        .to_string()
        .contains("relation \"people_seq\" already exists"));
    let table_rename_target = rename_boundary
        .execute_text(4, "ALTER SEQUENCE people RENAME TO people_seq_renamed")
        .unwrap_err();
    assert!(table_rename_target.to_string().contains("not a sequence"));
}

#[test]
fn relational_sql_sequence_values_replay_from_wal() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE SEQUENCE public.people_seq")
        .unwrap();
    let sequence = e.relational_catalog_sequence("people_seq").unwrap();
    assert_eq!(sequence.last_value, 1);
    assert!(!sequence.is_called);

    e.execute_text(2, "SELECT nextval('public.people_seq'::regclass)")
        .unwrap();
    let sequence = e.relational_catalog_sequence("people_seq").unwrap();
    assert_eq!(sequence.last_value, 1);
    assert!(sequence.is_called);

    e.execute_text(3, "SELECT nextval('people_seq'::regclass)")
        .unwrap();
    let sequence = e.relational_catalog_sequence("people_seq").unwrap();
    assert_eq!(sequence.last_value, 2);
    assert!(sequence.is_called);

    e.execute_text(4, "SELECT setval('public.people_seq', 10, false)")
        .unwrap();
    let sequence = e.relational_catalog_sequence("people_seq").unwrap();
    assert_eq!(sequence.last_value, 10);
    assert!(!sequence.is_called);

    e.execute_text(5, "SELECT nextval('people_seq'::regclass)")
        .unwrap();
    let sequence = e.relational_catalog_sequence("people_seq").unwrap();
    assert_eq!(sequence.last_value, 10);
    assert!(sequence.is_called);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let sequence = recovered.relational_catalog_sequence("people_seq").unwrap();
    assert_eq!(sequence.last_value, 10);
    assert!(sequence.is_called);

    let table_target = e
        .execute_text(6, "CREATE TABLE table_target (id INT)")
        .and_then(|_| e.execute_text(7, "SELECT nextval('table_target'::regclass)"))
        .unwrap_err();
    assert!(table_target.to_string().contains("not a sequence"));
    let missing = e
        .execute_text(8, "SELECT setval('missing_seq'::regclass, 1)")
        .unwrap_err();
    assert!(missing
        .to_string()
        .contains("sequence \"missing_seq\" does not exist"));
}

#[test]
fn relational_sequence_defaults_fill_omitted_columns_and_replay() {
    let e = Engine::new_local();
    e.execute_text(
        1,
        "CREATE TABLE serial_people (id SERIAL PRIMARY KEY, name TEXT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO serial_people (name) VALUES ('Ada'), ('Linus')",
    )
    .unwrap();
    e.execute_text(3, "CREATE SEQUENCE public.manual_people_seq")
        .unwrap();
    e.execute_text(
            4,
            "CREATE TABLE manual_people (id INT DEFAULT nextval('public.manual_people_seq'::regclass), name TEXT)",
        )
        .unwrap();
    e.execute_text(
        5,
        "INSERT INTO manual_people (name) VALUES ('Grace'), ('Barbara')",
    )
    .unwrap();
    e.execute_text(6, "CREATE TABLE after_serial_oid_check (id INT)")
        .unwrap();

    let serial_people_oid = e.relational_catalog_table("serial_people").unwrap().oid;
    let serial_sequence_oid = e
        .relational_catalog_sequence("serial_people_id_seq")
        .unwrap()
        .oid;
    let after_serial_oid = e
        .relational_catalog_table("after_serial_oid_check")
        .unwrap()
        .oid;
    assert_ne!(serial_people_oid, serial_sequence_oid);
    assert_ne!(after_serial_oid, serial_sequence_oid);
    assert!(after_serial_oid > serial_sequence_oid);

    let Command::Select(serial_select) =
        parse_command("SELECT id, name FROM serial_people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let serial_result = e.execute_relational_select(&serial_select).unwrap();
    assert_eq!(
        serial_result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
        ]
    );
    let serial_sequence = e
        .relational_catalog_sequence("serial_people_id_seq")
        .unwrap();
    assert_eq!(serial_sequence.last_value, 2);
    assert!(serial_sequence.is_called);

    let Command::Select(manual_select) =
        parse_command("SELECT id, name FROM manual_people ORDER BY id ASC").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let manual_result = e.execute_relational_select(&manual_select).unwrap();
    assert_eq!(
        manual_result.rows,
        vec![
            vec![SqlValue::Int4(1), SqlValue::Text("Grace".to_string())],
            vec![SqlValue::Int4(2), SqlValue::Text("Barbara".to_string())],
        ]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let recovered_serial = recovered.execute_relational_select(&serial_select).unwrap();
    assert_eq!(recovered_serial.rows, serial_result.rows);
    let recovered_manual = recovered.execute_relational_select(&manual_select).unwrap();
    assert_eq!(recovered_manual.rows, manual_result.rows);
    let recovered_sequence = recovered
        .relational_catalog_sequence("manual_people_seq")
        .unwrap();
    assert_eq!(recovered_sequence.last_value, 2);
    assert!(recovered_sequence.is_called);

    let missing = e
            .execute_text(
                7,
                "CREATE TABLE missing_default (id INT DEFAULT nextval('missing_seq'::regclass), name TEXT)",
            )
            .unwrap_err();
    assert!(missing
        .to_string()
        .contains("sequence \"missing_seq\" does not exist"));
    assert!(e.relational_catalog_table("missing_default").is_none());

    let table_target = e
            .execute_text(
                8,
                "CREATE TABLE bad_default (id INT DEFAULT nextval('serial_people'::regclass), name TEXT)",
            )
            .unwrap_err();
    assert!(table_target
        .to_string()
        .contains("relation \"serial_people\" is not a sequence"));

    let missing_alter = e
            .execute_text(
                9,
                "ALTER TABLE manual_people ALTER COLUMN id SET DEFAULT nextval('still_missing_seq'::regclass)",
            )
            .unwrap_err();
    assert!(missing_alter
        .to_string()
        .contains("sequence \"still_missing_seq\" does not exist"));
}

#[test]
fn relational_sql_materialized_view_lifecycle_replays_from_wal() {
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace')",
    )
    .unwrap();
    e.execute_text(
            3,
            "CREATE MATERIALIZED VIEW public.mv_people AS SELECT id, name FROM people WHERE id > 1 ORDER BY id",
        )
        .unwrap();
    e.execute_text(
        4,
        "COMMENT ON MATERIALIZED VIEW public.mv_people IS 'people snapshot'",
    )
    .unwrap();

    let view = e.relational_catalog_materialized_view("mv_people").unwrap();
    assert_eq!(view.name, "mv_people");
    let oid = view.oid;
    assert_eq!(view.rows.len(), 2);
    assert_eq!(
        e.relational_materialized_view_comment("mv_people")
            .as_deref(),
        Some("people snapshot")
    );

    e.execute_text(5, "INSERT INTO people (id, name) VALUES (4, 'Barbara')")
        .unwrap();
    let Command::Select(select) = parse_command("SELECT * FROM mv_people").unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
        ]
    );
    e.execute_text(6, "REFRESH MATERIALIZED VIEW public.mv_people")
        .unwrap();
    let refreshed_result = e.execute_relational_select(&select).unwrap();
    assert_eq!(
        refreshed_result.rows,
        vec![
            vec![SqlValue::Int4(2), SqlValue::Text("Linus".to_string())],
            vec![SqlValue::Int4(3), SqlValue::Text("Grace".to_string())],
            vec![SqlValue::Int4(4), SqlValue::Text("Barbara".to_string())],
        ]
    );
    assert_eq!(
        e.relational_catalog_materialized_view("mv_people")
            .unwrap()
            .oid,
        oid
    );

    e.execute_text(
        7,
        "ALTER MATERIALIZED VIEW public.mv_people RENAME TO mv_people_snapshot",
    )
    .unwrap();
    assert!(e
        .relational_catalog_materialized_view("mv_people")
        .is_none());
    assert_eq!(
        e.relational_catalog_materialized_view("mv_people_snapshot")
            .unwrap()
            .oid,
        oid
    );
    assert_eq!(
        e.relational_materialized_view_comment("mv_people_snapshot")
            .as_deref(),
        Some("people snapshot")
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let Command::Select(renamed_select) =
        parse_command("SELECT * FROM mv_people_snapshot").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let recovered_result = recovered
        .execute_relational_select(&renamed_select)
        .unwrap();
    assert_eq!(recovered_result.rows, refreshed_result.rows);
    assert_eq!(
        recovered
            .relational_materialized_view_comment("mv_people_snapshot")
            .as_deref(),
        Some("people snapshot")
    );

    e.execute_text(
        8,
        "DROP MATERIALIZED VIEW IF EXISTS missing_mv, mv_people_snapshot",
    )
    .unwrap();
    assert!(e
        .relational_catalog_materialized_view("mv_people_snapshot")
        .is_none());
    assert_eq!(
        e.relational_materialized_view_comment("mv_people_snapshot"),
        None
    );

    let boundary = Engine::new_local();
    boundary
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    boundary
        .execute_text(2, "CREATE TABLE other_people (id INT, name TEXT)")
        .unwrap();
    let table_target = boundary
        .execute_text(3, "DROP MATERIALIZED VIEW people")
        .unwrap_err();
    assert!(table_target.to_string().contains("not a materialized view"));
    let refresh_table_target = boundary
        .execute_text(4, "REFRESH MATERIALIZED VIEW people")
        .unwrap_err();
    assert!(refresh_table_target
        .to_string()
        .contains("not a materialized view"));
    let refresh_missing = boundary
        .execute_text(5, "REFRESH MATERIALIZED VIEW missing_mv")
        .unwrap_err();
    assert!(refresh_missing
        .to_string()
        .contains("materialized view \"missing_mv\" does not exist"));
    let duplicate = boundary
        .execute_text(
            6,
            "CREATE MATERIALIZED VIEW people AS SELECT id, name FROM other_people",
        )
        .unwrap_err();
    assert!(duplicate
        .to_string()
        .contains("relation \"people\" already exists"));
}

#[test]
fn relational_sql_select_gpu_bridge_matches_cpu_results_at_sql_level() {
    let cpu = Engine::new_local();
    cpu.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    cpu.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    let durable = cpu.durable_wal_records().to_vec();
    let mut gpu = Engine::recover_from_durable_wal(&durable).unwrap();

    let Command::Select(select) = parse_command("SELECT * FROM people").unwrap() else {
        panic!("expected SELECT plan");
    };
    let cpu_result = cpu.execute_relational_select(&select).unwrap();
    let gpu_result = gpu
        .execute_relational_select_with_backend(&select, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(gpu_result.columns, cpu_result.columns);
    assert_eq!(gpu_result.rows, cpu_result.rows);
    assert_eq!(gpu_result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(gpu_result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(gpu_result.fallback_reason, None);
}

#[test]
fn relational_sql_gpu_bridge_projection_result_shaping_does_not_report_gpu_fallback() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT name FROM people WHERE id = 2 LIMIT 1").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .execute_relational_select_with_backend(&select, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(result.rows, vec![vec![SqlValue::Text("Linus".to_string())]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
fn relational_sql_gpu_bridge_order_by_decoded_column_uses_ordered_key_batch() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (2, 'Grace')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT name FROM people WHERE id = 2 ORDER BY name LIMIT 1").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .execute_relational_select_with_backend(&select, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(result.rows, vec![vec![SqlValue::Text("Grace".to_string())]]);
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        *result.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("id".to_string()),
            predicate_op: Some(SelectFilterOp::Eq),
            order_column: "name".to_string(),
            descending: false,
            matched_keys: 2,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
fn relational_sql_gpu_bridge_full_scan_order_by_uses_ordered_key_batch() {
    let mut e = Engine::new_local();
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
    let result = e
        .execute_relational_select_with_backend(&select, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(3)]]
    );
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        *result.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: None,
            predicate_op: None,
            order_column: "name".to_string(),
            descending: true,
            matched_keys: 3,
        }
    );
}

#[test]
fn relational_sql_gpu_bridge_ordered_limit_offset_uses_ordered_key_batch() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Katherine')",
        )
        .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people ORDER BY id LIMIT 2 OFFSET 1").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .execute_relational_select_with_backend(&select, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Int4(2)], vec![SqlValue::Int4(3)]]
    );
    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        *result.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: None,
            predicate_op: None,
            order_column: "id".to_string(),
            descending: false,
            matched_keys: 4,
        }
    );
    assert_eq!(e.status_snapshot().latest_fallback_reason(), None);
}

#[test]
fn relational_sql_gpu_bridge_filtered_offset_without_limit_skips_after_order() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace'), (4, 'Grady')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE name LIKE 'Gra%' ORDER BY id OFFSET 1").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .execute_relational_select_with_backend(&select, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(result.rows, vec![vec![SqlValue::Int4(4)]]);
    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        *result.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: Some("name".to_string()),
            predicate_op: Some(SelectFilterOp::LikePrefix),
            order_column: "id".to_string(),
            descending: false,
            matched_keys: 2,
        }
    );
}

#[test]
fn relational_sql_gpu_bridge_distinct_projection_keeps_gpu_row_fetch() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Grace'), (4, 'Linus')",
    )
    .unwrap();

    let Command::Select(select) =
        parse_command("SELECT DISTINCT name FROM people ORDER BY name DESC LIMIT 2 OFFSET 1")
            .unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e
        .execute_relational_select_with_backend(&select, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Text("Grace".to_string())],
            vec![SqlValue::Text("Ada".to_string())],
        ]
    );
    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        *result.access_path,
        RelationalAccessPath::OrderedKeyBatch {
            table: "people".to_string(),
            predicate_column: None,
            predicate_op: None,
            order_column: "name".to_string(),
            descending: true,
            matched_keys: 4,
        }
    );
}

#[test]
fn relational_sql_gpu_bridge_count_group_by_keeps_gpu_row_fetch() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Grace'), (4, 'Linus')",
    )
    .unwrap();

    let Command::Select(select) = parse_command(
        "SELECT name, COUNT(*) FROM people WHERE id >= 2 GROUP BY name ORDER BY count DESC LIMIT 1",
    )
    .unwrap() else {
        panic!("expected SELECT plan");
    };
    let result = e
        .execute_relational_select_with_backend(&select, &FirstCudaSliceParityBackend)
        .unwrap();

    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Text("Grace".to_string()), SqlValue::Int8(2)]]
    );
    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        *result.access_path,
        RelationalAccessPath::FilteredKeyBatch {
            table: "people".to_string(),
            predicate_column: "id".to_string(),
            predicate_op: SelectFilterOp::Gte,
            matched_keys: 3,
        }
    );

    let Command::Select(count_select) =
        parse_command("SELECT COUNT(*) FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let count_result = e
        .execute_relational_select_with_backend(&count_select, &FirstCudaSliceParityBackend)
        .unwrap();
    assert_eq!(count_result.rows, vec![vec![SqlValue::Int8(2)]]);
    assert_eq!(count_result.fallback_reason, None);

    let Command::Select(sum_select) = parse_command(
        "SELECT name, SUM(id) FROM people WHERE id >= 2 GROUP BY name ORDER BY sum DESC LIMIT 1",
    )
    .unwrap() else {
        panic!("expected SELECT plan");
    };
    let sum_result = e
        .execute_relational_select_with_backend(&sum_select, &FirstCudaSliceParityBackend)
        .unwrap();
    assert_eq!(
        sum_result.rows,
        vec![vec![SqlValue::Text("Grace".to_string()), SqlValue::Int8(5)]]
    );
    assert_eq!(sum_result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(sum_result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(sum_result.fallback_reason, None);

    let Command::Select(avg_select) = parse_command(
        "SELECT name, AVG(id) FROM people WHERE id >= 2 GROUP BY name ORDER BY avg DESC LIMIT 1",
    )
    .unwrap() else {
        panic!("expected SELECT plan");
    };
    let avg_result = e
        .execute_relational_select_with_backend(&avg_select, &FirstCudaSliceParityBackend)
        .unwrap();
    assert_eq!(
        avg_result.rows,
        vec![vec![
            SqlValue::Text("Linus".to_string()),
            SqlValue::Numeric(Decimal128::parse("4.0000000000000000").unwrap())
        ]]
    );
    assert_eq!(avg_result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(avg_result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(avg_result.fallback_reason, None);

    let Command::Select(avg_scalar_select) =
        parse_command("SELECT AVG(id) FROM people WHERE name = 'Grace'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let avg_scalar_result = e
        .execute_relational_select_with_backend(&avg_scalar_select, &FirstCudaSliceParityBackend)
        .unwrap();
    assert_eq!(
        avg_scalar_result.rows,
        vec![vec![SqlValue::Numeric(
            Decimal128::parse("2.5000000000000000").unwrap()
        )]]
    );
    assert_eq!(avg_scalar_result.fallback_reason, None);

    let Command::Select(min_select) = parse_command(
        "SELECT name, MIN(id) FROM people WHERE id >= 2 GROUP BY name ORDER BY min DESC LIMIT 1",
    )
    .unwrap() else {
        panic!("expected SELECT plan");
    };
    let min_result = e
        .execute_relational_select_with_backend(&min_select, &FirstCudaSliceParityBackend)
        .unwrap();
    assert_eq!(
        min_result.rows,
        vec![vec![SqlValue::Text("Linus".to_string()), SqlValue::Int4(4)]]
    );
    assert_eq!(min_result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(min_result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(min_result.fallback_reason, None);

    let Command::Select(max_select) =
        parse_command("SELECT MAX(name) FROM people WHERE id <= 2").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let max_result = e
        .execute_relational_select_with_backend(&max_select, &FirstCudaSliceParityBackend)
        .unwrap();
    assert_eq!(
        max_result.rows,
        vec![vec![SqlValue::Text("Grace".to_string())]]
    );
    assert_eq!(max_result.fallback_reason, None);
}

#[test]
fn primary_key_rejects_null_on_insert_and_update() {
    // PG: PRIMARY KEY implies NOT NULL (23502) — a NULL may never enter a PK column. Previously the
    // PK was validated only as a unique index whose BTreeSet collides NULLs, so exactly ONE NULL row
    // could slip in (and would then poison the resident PK index routes + the NULL-blind aggregate
    // fast paths). Both validator arms (index-driven + scan fallback) must reject it identically.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE pk_nn (id INT PRIMARY KEY, v INT)")
        .unwrap();

    // INSERT of an explicit NULL PK: rejected, nothing written.
    let err = e
        .execute_text(2, "INSERT INTO pk_nn (id, v) VALUES (NULL, 1)")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("null value in column \"id\" of relation \"pk_nn\" violates not-null constraint"),
        "{err}"
    );
    // Multi-row INSERT with one NULL among valid rows: the whole statement fails (atomicity).
    let err = e
        .execute_text(3, "INSERT INTO pk_nn (id, v) VALUES (7, 7), (NULL, 8)")
        .unwrap_err()
        .to_string();
    assert!(err.contains("violates not-null constraint"), "{err}");
    let Command::Select(count) = parse_command("SELECT COUNT(*) FROM pk_nn").unwrap() else {
        panic!("expected SELECT plan");
    };
    assert_eq!(
        e.execute_relational_select(&count).unwrap().rows,
        vec![vec![SqlValue::Int8(0)]],
        "failed inserts must write nothing"
    );

    // Valid rows land; UPDATE to NULL is rejected on BOTH validator arms (the index-driven default
    // and the scan fallback behind the kill switch), byte-identical message; rows stay intact.
    e.execute_text(4, "INSERT INTO pk_nn (id, v) VALUES (1, 1), (2, 2)")
        .unwrap();
    let mut txn = 5;
    for index_arm in [true, false] {
        e.set_dml_value_index_resolve_enabled(index_arm);
        let err = e
            .execute_text(txn, "UPDATE pk_nn SET id = NULL WHERE v = 1")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(
                "null value in column \"id\" of relation \"pk_nn\" violates not-null constraint"
            ),
            "index_arm={index_arm}: {err}"
        );
        txn += 1;
    }
    e.set_dml_value_index_resolve_enabled(true);
    assert_eq!(
        e.execute_relational_select(&count).unwrap().rows,
        vec![vec![SqlValue::Int8(2)]],
        "failed updates must change nothing"
    );
    // A valid UPDATE of the PK still works (the check is NULL-only, not immutability).
    e.execute_text(txn, "UPDATE pk_nn SET id = 3 WHERE v = 1").unwrap();

    // Control: a plain (non-PK) UNIQUE column keeps this engine's existing NULL semantics unchanged
    // (one NULL admitted; a second collides) — the not-null check is scoped to the PK.
    e.execute_text(txn + 1, "CREATE TABLE uq_ctl (id INT PRIMARY KEY, u INT UNIQUE)")
        .unwrap();
    e.execute_text(txn + 2, "INSERT INTO uq_ctl (id, u) VALUES (1, NULL)")
        .unwrap();
    let err = e
        .execute_text(txn + 3, "INSERT INTO uq_ctl (id, u) VALUES (2, NULL)")
        .unwrap_err()
        .to_string();
    assert!(err.contains("duplicate key value"), "{err}");
}

#[test]
fn alter_add_primary_key_rejects_null_bearing_column() {
    // PG: promoting a null-bearing column to PRIMARY KEY fails (the creation-time half of the PK
    // NOT NULL invariant the DML validators rely on). After the NULL row is gone the promotion
    // succeeds, and the promoted PK then enforces not-null on subsequent writes.
    let e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE promote (id INT, v INT)").unwrap();
    e.execute_text(2, "INSERT INTO promote (id, v) VALUES (NULL, 1), (2, 2)")
        .unwrap();
    let err = e
        .execute_text(
            3,
            "ALTER TABLE ONLY public.promote ADD CONSTRAINT promote_pkey PRIMARY KEY (id)",
        )
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("column \"id\" of relation \"promote\" contains null values"),
        "{err}"
    );
    e.execute_text(4, "DELETE FROM promote WHERE v = 1").unwrap();
    e.execute_text(
        5,
        "ALTER TABLE ONLY public.promote ADD CONSTRAINT promote_pkey PRIMARY KEY (id)",
    )
    .unwrap();
    let err = e
        .execute_text(6, "INSERT INTO promote (id, v) VALUES (NULL, 3)")
        .unwrap_err()
        .to_string();
    assert!(err.contains("violates not-null constraint"), "{err}");
}
