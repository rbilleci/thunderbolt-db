use crate::tests::assert_recovered_relational_access_path;
use crate::{
    CheckOperandIdentityVersion, Engine, EngineError, ExecuteError, RelationalAccessPath,
    RelationalCheckConstraint, RelationalColumn, RelationalIndex, FIRST_USER_COLUMN_ID,
    FIRST_USER_RELATION_OID, PUBLIC_SCHEMA_NAME,
};
use gpu_db_sql::{
    parse_command, CheckLiteralProvenance, Command, Decimal128, SelectFilterOp, SqlType, SqlValue,
    NUMERIC_DEFAULT_PRECISION,
};

const FIRST_TABLE_INDEX_OID: u32 = FIRST_USER_RELATION_OID + 1;

#[test]
fn relational_catalog_assigns_stable_public_schema_and_type_metadata() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();

    let table = e.relational_catalog_table("people").unwrap();
    assert_eq!(table.schema, PUBLIC_SCHEMA_NAME);
    assert_eq!(table.name, "people");
    assert_eq!(table.oid, FIRST_USER_RELATION_OID);
    assert_eq!(table.columns.len(), 2);
    assert_eq!(
        table.columns[0],
        RelationalColumn {
            id: FIRST_USER_COLUMN_ID,
            table_oid: FIRST_USER_RELATION_OID,
            attnum: 1,
            name: "id".to_string(),
            ty: SqlType::Int4,
            domain: None,
            default: None,
            type_oid: SqlType::Int4.postgres_oid(),
            type_size: SqlType::Int4.type_size(),
        }
    );
    assert_eq!(
        table.columns[1],
        RelationalColumn {
            id: FIRST_USER_COLUMN_ID + 1,
            table_oid: FIRST_USER_RELATION_OID,
            attnum: 2,
            name: "name".to_string(),
            ty: SqlType::Text,
            domain: None,
            default: None,
            type_oid: SqlType::Text.postgres_oid(),
            type_size: SqlType::Text.type_size(),
        }
    );

    e.execute_text(2, "CREATE TABLE teams (id INT)").unwrap();
    assert_eq!(
        e.relational_catalog_table("teams").unwrap().oid,
        FIRST_USER_RELATION_OID + 1
    );
}

#[test]
fn relational_catalog_records_create_index_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    let table = e.relational_catalog_table("people").unwrap();
    assert_eq!(
        table.indexes,
        vec![RelationalIndex {
            oid: FIRST_TABLE_INDEX_OID,
            name: "people_name_idx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            key_columns: vec!["name".to_string()],
            unique: false,
            primary_key: false,
            unique_constraint: false,
        }]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        vec![RelationalIndex {
            oid: FIRST_TABLE_INDEX_OID,
            name: "people_name_idx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            key_columns: vec!["name".to_string()],
            unique: false,
            primary_key: false,
            unique_constraint: false,
        }]
    );

    let duplicate_err = e
        .execute_text(3, "CREATE INDEX people_name_idx ON people (id)")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_err.contains("relation \"people_name_idx\" already exists"),
        "{duplicate_err}"
    );

    let missing = Engine::new_local_test_engine();
    missing
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    assert!(missing
        .execute_text(2, "CREATE INDEX people_missing_idx ON people (missing)")
        .unwrap_err()
        .to_string()
        .contains("column \"missing\" does not exist"));
}

#[test]
fn relational_catalog_preserves_compound_secondary_index_order_across_recovery() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE accounts (tenant_id INT, account_id BIGINT, status SMALLINT)",
    )
    .unwrap();
    e.execute_text(
        2,
        "CREATE INDEX accounts_by_status ON accounts (tenant_id, status, account_id)",
    )
    .unwrap();

    let expected = RelationalIndex {
        oid: FIRST_TABLE_INDEX_OID,
        name: "accounts_by_status".to_string(),
        table: "accounts".to_string(),
        column: "tenant_id".to_string(),
        key_columns: vec![
            "tenant_id".to_string(),
            "status".to_string(),
            "account_id".to_string(),
        ],
        unique: false,
        primary_key: false,
        unique_constraint: false,
    };
    assert_eq!(
        e.relational_catalog_table("accounts").unwrap().indexes,
        vec![expected.clone()]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("accounts")
            .unwrap()
            .indexes,
        vec![expected]
    );
}

#[test]
fn relational_unique_index_rejects_duplicate_create_insert_update_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();
    e.execute_text(3, "CREATE UNIQUE INDEX people_name_uidx ON people (name)")
        .unwrap();

    let indexes = e
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .clone();
    assert_eq!(
        indexes,
        vec![RelationalIndex {
            oid: FIRST_TABLE_INDEX_OID,
            name: "people_name_uidx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            key_columns: vec!["name".to_string()],
            unique: true,
            primary_key: false,
            unique_constraint: false,
        }]
    );

    let duplicate_insert = e
        .execute_text(4, "INSERT INTO people (id, name) VALUES (3, 'Ada')")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_insert.contains("duplicate key value violates unique index"),
        "{duplicate_insert}"
    );
    let duplicate_update = e
        .execute_text(5, "UPDATE people SET name = 'Ada' WHERE id = 2")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_update.contains("duplicate key value violates unique index"),
        "{duplicate_update}"
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
            vec![SqlValue::Int4(2), SqlValue::Text("Grace".to_string())],
        ]
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        indexes
    );

    let duplicate_existing = Engine::new_local_test_engine();
    duplicate_existing
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    duplicate_existing
        .execute_text(
            2,
            "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Ada')",
        )
        .unwrap();
    let create_err = duplicate_existing
        .execute_text(3, "CREATE UNIQUE INDEX people_name_uidx ON people (name)")
        .unwrap_err()
        .to_string();
    assert!(
        create_err.contains("duplicate key value violates unique index"),
        "{create_err}"
    );
}

#[test]
fn relational_unique_constraints_reject_duplicates_and_replay_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE people (id INT, name TEXT UNIQUE, CONSTRAINT people_id_key UNIQUE (id))",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();

    let indexes = e
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .clone();
    assert_eq!(
        indexes,
        vec![
            RelationalIndex {
                oid: FIRST_TABLE_INDEX_OID,
                name: "people_name_key".to_string(),
                table: "people".to_string(),
                column: "name".to_string(),
                key_columns: vec!["name".to_string()],
                unique: true,
                primary_key: false,
                unique_constraint: true,
            },
            RelationalIndex {
                oid: FIRST_TABLE_INDEX_OID + 1,
                name: "people_id_key".to_string(),
                table: "people".to_string(),
                column: "id".to_string(),
                key_columns: vec!["id".to_string()],
                unique: true,
                primary_key: false,
                unique_constraint: true,
            },
        ]
    );

    let duplicate_insert = e
        .execute_text(3, "INSERT INTO people (id, name) VALUES (3, 'Ada')")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_insert.contains("duplicate key value violates unique index"),
        "{duplicate_insert}"
    );
    let duplicate_update = e
        .execute_text(4, "UPDATE people SET id = 1 WHERE name = 'Grace'")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_update.contains("duplicate key value violates unique index"),
        "{duplicate_update}"
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        indexes
    );

    let alter = Engine::new_local_test_engine();
    alter
        .execute_text(1, "CREATE TABLE teams (id INT, name TEXT)")
        .unwrap();
    alter
        .execute_text(
            2,
            "INSERT INTO teams (id, name) VALUES (1, 'core'), (2, 'db')",
        )
        .unwrap();
    alter
        .execute_text(
            3,
            "ALTER TABLE ONLY public.teams ADD CONSTRAINT teams_name_key UNIQUE (name)",
        )
        .unwrap();
    assert!(alter
        .execute_text(4, "INSERT INTO teams (id, name) VALUES (3, 'core')")
        .unwrap_err()
        .to_string()
        .contains("duplicate key value violates unique index"));
}

#[test]
fn relational_check_constraints_enforce_and_replay_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE people (id INT, name TEXT, CONSTRAINT people_id_positive CHECK (id > 0))",
    )
    .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();

    let checks = e
        .relational_catalog_table("people")
        .unwrap()
        .check_constraints
        .clone();
    assert_eq!(
        checks,
        vec![RelationalCheckConstraint {
            name: "people_id_positive".to_string(),
            column: "id".to_string(),
            op: SelectFilterOp::Gt,
            value: SqlValue::Int4(0),
            resolved_input_type: SqlType::Int4,
            identity_version: CheckOperandIdentityVersion::ResolvedV2,
        }]
    );

    let invalid_insert = e
        .execute_text(3, "INSERT INTO people (id, name) VALUES (-1, 'Bad')")
        .unwrap_err()
        .to_string();
    assert!(
        invalid_insert.contains("violates check constraint"),
        "{invalid_insert}"
    );
    let invalid_update = e
        .execute_text(4, "UPDATE people SET id = -2 WHERE name = 'Grace'")
        .unwrap_err()
        .to_string();
    assert!(
        invalid_update.contains("violates check constraint"),
        "{invalid_update}"
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
            vec![SqlValue::Int4(2), SqlValue::Text("Grace".to_string())],
        ]
    );

    e.execute_text(
        5,
        "ALTER TABLE ONLY public.people RENAME CONSTRAINT people_id_positive TO people_id_gt_zero",
    )
    .unwrap();
    e.execute_text(6, "ALTER TABLE public.people RENAME COLUMN id TO person_id")
        .unwrap();
    let table = e.relational_catalog_table("people").unwrap();
    assert_eq!(table.check_constraints[0].name, "people_id_gt_zero");
    assert_eq!(table.check_constraints[0].column, "person_id");
    let renamed_checks = table.check_constraints.clone();
    assert!(e
        .execute_text(7, "ALTER TABLE public.people DROP COLUMN person_id")
        .unwrap_err()
        .to_string()
        .contains("index or constraint depends on it"));

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .check_constraints,
        renamed_checks
    );

    let alter = Engine::new_local_test_engine();
    alter
        .execute_text(1, "CREATE TABLE teams (id INT, name TEXT)")
        .unwrap();
    alter
        .execute_text(
            2,
            "INSERT INTO teams (id, name) VALUES (1, 'core'), (-1, 'bad')",
        )
        .unwrap();
    let existing_rows = alter
        .execute_text(
            3,
            "ALTER TABLE ONLY public.teams ADD CONSTRAINT teams_id_positive CHECK (id > 0)",
        )
        .unwrap_err()
        .to_string();
    assert!(
        existing_rows.contains("violates check constraint")
            || existing_rows.contains("violated by some row"),
        "{existing_rows}"
    );
    alter
        .execute_text(4, "CREATE TABLE valid_teams (id INT, name TEXT)")
        .unwrap();
    alter
            .execute_text(
                5,
                "ALTER TABLE ONLY public.valid_teams ADD CONSTRAINT valid_teams_id_positive CHECK (id > 0)",
            )
            .unwrap();
    assert!(alter
        .execute_text(6, "INSERT INTO valid_teams (id, name) VALUES (-2, 'bad')")
        .unwrap_err()
        .to_string()
        .contains("violates check constraint"));
}

fn normalized_check_literal_values() -> Vec<SqlValue> {
    vec![
        SqlValue::Int4(7),
        SqlValue::Int4(7),
        SqlValue::Int4(7),
        SqlValue::Numeric(Decimal128::new(12, 0)),
        SqlValue::Bool(true),
        SqlValue::Text("open".to_string()),
        SqlValue::Date(gpu_db_sql::datetime::parse_date("2024-02-03").unwrap()),
        SqlValue::Timestamp(gpu_db_sql::datetime::parse_timestamp("2024-02-03 04:05:06").unwrap()),
        SqlValue::Uuid(
            gpu_db_sql::uuid::parse_uuid("00112233-4455-6677-8899-aabbccddeeff").unwrap(),
        ),
    ]
}

fn assert_normalized_check_literals(engine: &Engine, table: &str) {
    assert_eq!(
        engine
            .relational_catalog_table(table)
            .unwrap()
            .check_constraints
            .iter()
            .map(|constraint| constraint.value.clone())
            .collect::<Vec<_>>(),
        normalized_check_literal_values(),
    );
}

/// CHECK literals use comparison semantics rather than INSERT assignment casts. This covers every
/// scalar in `SUPPORTED_SQL_TYPES` for both CREATE and ADD, and verifies replay re-binds the same
/// resolved catalog values from the raw WAL command.
#[test]
fn check_literals_use_comparison_resolution_for_create_add_and_wal_replay() {
    let engine = Engine::new_local_test_engine();
    engine.execute_text(1, "CREATE TABLE check_create (s SMALLINT, i INT, b BIGINT, n NUMERIC(8,2), flag BOOLEAN, note TEXT, d DATE, ts TIMESTAMP, u UUID, CONSTRAINT s_check CHECK (s >= 7), CONSTRAINT i_check CHECK (i >= 7), CONSTRAINT b_check CHECK (b >= 7), CONSTRAINT n_check CHECK (n >= 12), CONSTRAINT flag_check CHECK (flag = true), CONSTRAINT note_check CHECK (note = 'open'), CONSTRAINT d_check CHECK (d >= '2024-02-03'), CONSTRAINT ts_check CHECK (ts >= '2024-02-03 04:05:06'), CONSTRAINT u_check CHECK (u = '00112233-4455-6677-8899-aabbccddeeff'))").unwrap();
    assert_normalized_check_literals(&engine, "check_create");

    engine
        .execute_text(2, "CREATE DOMAIN check_code AS BIGINT")
        .unwrap();
    engine
        .execute_text(
            3,
            "CREATE TABLE check_domain (code check_code, CHECK (code >= 7))",
        )
        .unwrap();
    assert_eq!(
        engine
            .relational_catalog_table("check_domain")
            .unwrap()
            .check_constraints[0]
            .value,
        SqlValue::Int4(7),
    );

    engine.execute_text(4, "CREATE TABLE check_add (s SMALLINT, i INT, b BIGINT, n NUMERIC(8,2), flag BOOLEAN, note TEXT, d DATE, ts TIMESTAMP, u UUID)").unwrap();
    engine.execute_text(5, "INSERT INTO check_add VALUES (7, 7, 7, 12, true, 'open', '2024-02-03', '2024-02-03 04:05:06', '00112233-4455-6677-8899-aabbccddeeff')").unwrap();
    for (seq, sql) in [
        "ALTER TABLE check_add ADD CONSTRAINT add_s_check CHECK (s >= 7)",
        "ALTER TABLE check_add ADD CONSTRAINT add_i_check CHECK (i >= 7)",
        "ALTER TABLE check_add ADD CONSTRAINT add_b_check CHECK (b >= 7)",
        "ALTER TABLE check_add ADD CONSTRAINT add_n_check CHECK (n >= 12)",
        "ALTER TABLE check_add ADD CONSTRAINT add_flag_check CHECK (flag = true)",
        "ALTER TABLE check_add ADD CONSTRAINT add_note_check CHECK (note = 'open')",
        "ALTER TABLE check_add ADD CONSTRAINT add_d_check CHECK (d >= '2024-02-03')",
        "ALTER TABLE check_add ADD CONSTRAINT add_ts_check CHECK (ts >= '2024-02-03 04:05:06')",
        "ALTER TABLE check_add ADD CONSTRAINT add_u_check CHECK (u = '00112233-4455-6677-8899-aabbccddeeff')",
    ].into_iter().enumerate() {
        engine.execute_text(6 + seq as u64, sql).unwrap();
    }
    assert_normalized_check_literals(&engine, "check_add");

    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert_normalized_check_literals(&recovered, "check_create");
    assert_eq!(
        recovered
            .relational_catalog_table("check_domain")
            .unwrap()
            .check_constraints[0]
            .value,
        SqlValue::Int4(7),
    );
    assert_normalized_check_literals(&recovered, "check_add");
}

#[test]
fn check_comparison_literals_preserve_out_of_range_and_cross_scale_values() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE check_cross_create (s SMALLINT, i INT, n NUMERIC(8,2), CONSTRAINT cross_create_s CHECK (s < 32768), CONSTRAINT cross_create_i CHECK (i < 2147483648), CONSTRAINT cross_create_n CHECK (n < 12.345))",
        )
        .unwrap();
    let expected = vec![
        SqlValue::Int4(32_768),
        SqlValue::Int8(2_147_483_648),
        SqlValue::Numeric(Decimal128::new(12_345, 3)),
    ];
    assert_eq!(
        engine
            .relational_catalog_table("check_cross_create")
            .unwrap()
            .check_constraints
            .iter()
            .map(|constraint| constraint.value.clone())
            .collect::<Vec<_>>(),
        expected,
    );

    engine
        .execute_text(
            2,
            "CREATE TABLE check_cross_add (s SMALLINT, i INT, n NUMERIC(8,2))",
        )
        .unwrap();
    engine
        .execute_text(
            3,
            "INSERT INTO check_cross_add VALUES (-32768, -2147483648, 12.34)",
        )
        .unwrap();
    for (seq, sql) in [
        "ALTER TABLE check_cross_add ADD CONSTRAINT cross_add_s CHECK (s < 32768)",
        "ALTER TABLE check_cross_add ADD CONSTRAINT cross_add_i CHECK (i < 2147483648)",
        "ALTER TABLE check_cross_add ADD CONSTRAINT cross_add_n CHECK (n < 12.345)",
    ]
    .into_iter()
    .enumerate()
    {
        engine.execute_text(4 + seq as u64, sql).unwrap();
    }
    assert_eq!(
        engine
            .relational_catalog_table("check_cross_add")
            .unwrap()
            .check_constraints
            .iter()
            .map(|constraint| constraint.value.clone())
            .collect::<Vec<_>>(),
        expected,
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

#[test]
fn invalid_check_comparison_literal_rejects_atomically_before_wal() {
    let engine = Engine::new_local_test_engine();
    let create_catalog_before = engine.catalog_snapshot();
    let create_wal_before = engine.durable_wal_records().len();
    let create_error = engine
        .execute_text(
            1,
            "CREATE TABLE bad_check_create (d DATE, CHECK (d > 'not-a-date'))",
        )
        .unwrap_err();
    assert!(matches!(
        create_error,
        ExecuteError::Engine(EngineError::InvalidDatetimeFormat(message))
            if message.contains("invalid input syntax for type date")
    ));
    assert!(engine
        .catalog_snapshot()
        .same_contents(create_catalog_before.as_ref()));
    assert_eq!(engine.durable_wal_records().len(), create_wal_before);

    engine
        .execute_text(2, "CREATE TABLE bad_check_add (d DATE)")
        .unwrap();
    let add_catalog_before = engine.catalog_snapshot();
    let add_wal_before = engine.durable_wal_records().len();
    let add_error = engine
        .execute_text(
            3,
            "ALTER TABLE bad_check_add ADD CONSTRAINT bad_date CHECK (d > 'not-a-date')",
        )
        .unwrap_err();
    assert!(matches!(
        add_error,
        ExecuteError::Engine(EngineError::InvalidDatetimeFormat(message))
            if message.contains("invalid input syntax for type date")
    ));
    assert!(engine
        .catalog_snapshot()
        .same_contents(add_catalog_before.as_ref()));
    assert_eq!(engine.durable_wal_records().len(), add_wal_before);

    let overflow_catalog_before = engine.catalog_snapshot();
    let overflow_wal_before = engine.durable_wal_records().len();
    let overflow_error = engine
        .execute_text(
            4,
            "ALTER TABLE bad_check_add ADD CONSTRAINT date_field_retry CHECK (d > '2024-02-30')",
        )
        .unwrap_err();
    assert!(matches!(
        overflow_error,
        ExecuteError::Engine(EngineError::DatetimeFieldOverflow(message))
            if message.contains("invalid input syntax for type date")
    ));
    assert!(engine
        .catalog_snapshot()
        .same_contents(overflow_catalog_before.as_ref()));
    assert_eq!(engine.durable_wal_records().len(), overflow_wal_before);
    engine
        .execute_text(
            5,
            "ALTER TABLE bad_check_add ADD CONSTRAINT date_field_retry CHECK (d > '2024-01-01')",
        )
        .expect("overflowing ALTER must leave the same constraint name reusable");
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

#[test]
fn check_literal_provenance_preserves_unknown_explicit_and_legacy_replay_semantics() {
    let explicit =
        parse_command("CREATE TABLE explicit_provenance (s SMALLINT, CHECK (s > '7'::int4))")
            .unwrap();
    let Command::CreateTable(create) = &explicit else {
        panic!("expected CREATE TABLE");
    };
    assert_eq!(
        create.check_constraints[0].literal_provenance,
        CheckLiteralProvenance::Known(SqlType::Int4)
    );
    let explicit_bytes = serde_json::to_vec(&explicit).unwrap();
    assert!(std::str::from_utf8(&explicit_bytes)
        .unwrap()
        .contains("literal_provenance"),);
    let explicit_replayed: Command = serde_json::from_slice(&explicit_bytes).unwrap();
    assert_eq!(explicit_replayed, explicit);

    // Old typed catalog payloads omitted the field. Their serialization remains byte-identical;
    // decoding deliberately preserves the ambiguity instead of pretending this was a new parse.
    let mut legacy_source =
        parse_command("CREATE TABLE legacy_provenance (s SMALLINT, CHECK (s > 7))").unwrap();
    let Command::CreateTable(legacy_create) = &mut legacy_source else {
        panic!("expected CREATE TABLE");
    };
    legacy_create.check_constraints[0].literal_provenance = CheckLiteralProvenance::LegacyAmbiguous;
    let legacy_bytes = serde_json::to_vec(&legacy_source).unwrap();
    let legacy_replayed: Command = serde_json::from_slice(&legacy_bytes).unwrap();
    let Command::CreateTable(legacy_create) = &legacy_replayed else {
        panic!("expected CREATE TABLE");
    };
    assert_eq!(
        legacy_create.check_constraints[0].literal_provenance,
        CheckLiteralProvenance::LegacyAmbiguous
    );
    assert_eq!(serde_json::to_vec(&legacy_replayed).unwrap(), legacy_bytes);

    let engine = Engine::new_local_test_engine();
    let wal_before = engine.durable_wal_records().len();
    let quoted_smallint = engine
        .execute_text(
            1,
            "CREATE TABLE rejected_unknown_smallint (s SMALLINT, CHECK (s > '32768'))",
        )
        .unwrap_err()
        .to_string();
    assert!(
        quoted_smallint.contains("smallint out of range"),
        "{quoted_smallint}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    let explicit_text = engine
        .execute_text(
            2,
            "CREATE TABLE rejected_explicit_text (s SMALLINT, CHECK (s > '32768'::text))",
        )
        .unwrap_err()
        .to_string();
    assert!(
        explicit_text.contains("operator does not exist"),
        "{explicit_text}"
    );
    assert_eq!(engine.durable_wal_records().len(), wal_before);

    engine
        .execute_text(
            3,
            "CREATE TABLE comparison_provenance (s SMALLINT, t TEXT, n NUMERIC(8,2), CHECK (s > 32768), CHECK (t = 'x'), CHECK (n < '12.345'))",
        )
        .unwrap();
    assert_eq!(
        engine
            .relational_catalog_table("comparison_provenance")
            .unwrap()
            .check_constraints
            .iter()
            .map(|check| check.value.clone())
            .collect::<Vec<_>>(),
        vec![
            SqlValue::Int4(32_768),
            SqlValue::Text("x".to_string()),
            SqlValue::Numeric(Decimal128::new(12_345, 3)),
        ]
    );
    engine
        .execute_text(
            4,
            "CREATE TABLE explicit_recovery (s SMALLINT, CHECK (s > '7'::int4))",
        )
        .unwrap();
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
    assert_eq!(
        recovered
            .relational_catalog_table("explicit_recovery")
            .unwrap()
            .check_constraints[0]
            .value,
        SqlValue::Int4(7),
    );
}

#[test]
fn legacy_check_numeric_operand_keeps_its_natural_type_after_catalog_resolution() {
    let numeric_column = SqlType::Numeric {
        precision: 3,
        scale: 0,
    };
    let (value, resolved_input_type) = crate::resolve_check_comparison_operand(
        SqlValue::Numeric(Decimal128::new(1_000, 0)),
        CheckLiteralProvenance::LegacyAmbiguous,
        numeric_column,
        SelectFilterOp::Gt,
        "n",
    )
    .expect("legacy numeric literal remains a compatible numeric operand");
    assert_eq!(value, SqlValue::Numeric(Decimal128::new(1_000, 0)));
    assert_eq!(
        resolved_input_type,
        SqlType::Numeric {
            precision: NUMERIC_DEFAULT_PRECISION,
            scale: 0,
        }
    );
    let legacy_constraint = RelationalCheckConstraint {
        name: "legacy_numeric".to_string(),
        column: "n".to_string(),
        op: SelectFilterOp::Gt,
        value,
        resolved_input_type,
        identity_version: CheckOperandIdentityVersion::LegacyV1,
    };
    crate::check_violation_expr::validate_check_literal_at_evaluation(&legacy_constraint)
        .expect("legacy numeric must not inherit the column numeric typmod");

    let (_, null_input_type) = crate::resolve_check_comparison_operand(
        SqlValue::Null,
        CheckLiteralProvenance::LegacyAmbiguous,
        SqlType::Timestamp,
        SelectFilterOp::Eq,
        "created_at",
    )
    .expect("legacy NULL binds to the comparison target");
    assert_eq!(null_input_type, SqlType::Timestamp);
}

#[test]
fn check_schema_digest_keeps_legacy_v1_and_new_check_ddl_upgrades_to_v2() {
    let mut legacy_engine = Engine::new_local_test_engine();
    let mut historical_catalog = legacy_engine.ddl_catalog_mut().clone();
    let mut legacy_command =
        parse_command("CREATE TABLE legacy_digest (id int4, CHECK (id > 0))").unwrap();
    let Command::CreateTable(legacy_create) = &mut legacy_command else {
        panic!("expected CREATE TABLE");
    };
    legacy_create.check_constraints[0].literal_provenance = CheckLiteralProvenance::LegacyAmbiguous;
    let Command::CreateTable(legacy_create) = legacy_command else {
        unreachable!("CREATE command was matched above");
    };
    legacy_engine
        .apply_create_table(&mut historical_catalog, legacy_create)
        .expect("historical CREATE replay");
    let historical = historical_catalog.relational_catalog["legacy_digest"].clone();
    assert_eq!(
        crate::engine_transaction_reset::table_schema_digest_version(&historical),
        1
    );
    let historical_digest = crate::engine_transaction_reset::table_schema_digest(&historical)
        .expect("legacy schema digest");
    assert_eq!(
        historical_digest,
        crate::engine_transaction_reset::table_schema_digest(&historical)
            .expect("legacy digest is deterministic")
    );

    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(2, "CREATE TABLE digest_upgrade (id int4)")
        .unwrap();
    let before = engine.relational_catalog_table("digest_upgrade").unwrap();
    assert_eq!(
        crate::engine_transaction_reset::table_schema_digest_version(&before),
        1
    );
    engine
        .execute_text(
            3,
            "ALTER TABLE digest_upgrade ADD CONSTRAINT digest_upgrade_check CHECK (id > 0)",
        )
        .unwrap();
    let upgraded = engine.relational_catalog_table("digest_upgrade").unwrap();
    assert_eq!(
        crate::engine_transaction_reset::table_schema_digest_version(&upgraded),
        2
    );
    assert!(upgraded
        .check_constraints
        .iter()
        .all(|constraint| constraint.identity_version == CheckOperandIdentityVersion::ResolvedV2));
    let upgraded_digest = crate::engine_transaction_reset::table_schema_digest(&upgraded)
        .expect("upgraded schema digest");
    assert_ne!(historical_digest, upgraded_digest);
    assert_eq!(
        upgraded_digest,
        crate::engine_transaction_reset::table_schema_digest(&upgraded)
            .expect("upgraded digest is deterministic")
    );
}

#[test]
fn temporal_check_casts_cover_create_add_null_recovery_and_schema_proof() {
    let engine = Engine::new_local_test_engine();
    engine
        .execute_text(
            1,
            "CREATE TABLE temporal_checks (ts timestamp, d date, \
             CONSTRAINT ts_before_day CHECK (ts < '2000-01-01'::date), \
             CONSTRAINT day_after_pre_epoch CHECK (d > '1999-12-31 23:59:59.999999'::timestamp))",
        )
        .unwrap();
    engine
        .execute_text(
            2,
            "INSERT INTO temporal_checks VALUES ('1999-12-31 23:59:59.999999'::timestamp, '2000-01-01'::date)",
        )
        .unwrap();
    engine
        .execute_text(
            3,
            "INSERT INTO temporal_checks VALUES ('1999-12-31 23:59:59.999999'::timestamp, NULL)",
        )
        .unwrap();
    let wal_before = engine.durable_wal_records().len();
    let row_id_before = engine.read_state.mvcc.current_row_id();
    let error = engine
        .execute_text(
            4,
            "INSERT INTO temporal_checks VALUES ('2000-01-01'::timestamp, '2000-01-01'::date)",
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ExecuteError::Engine(EngineError::CheckViolation(message)) if message.contains("ts_before_day")
    ));
    assert_eq!(engine.durable_wal_records().len(), wal_before);
    assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);

    engine
        .execute_text(5, "CREATE TABLE temporal_add (d date, ts timestamp)")
        .unwrap();
    engine
        .execute_text(
            6,
            "INSERT INTO temporal_add VALUES ('2000-01-01'::date, '2000-01-01'::timestamp)",
        )
        .unwrap();
    engine
        .execute_text(
            7,
            "ALTER TABLE temporal_add ADD CONSTRAINT day_before_plus_one CHECK (d < '2000-01-01 00:00:00.000001'::timestamp)",
        )
        .unwrap();
    let add_wal_before = engine.durable_wal_records().len();
    let error = engine
        .execute_text(
            8,
            "ALTER TABLE temporal_add ADD CONSTRAINT timestamp_before_day CHECK (ts < '2000-01-01'::date)",
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ExecuteError::Engine(EngineError::CheckViolation(_))
    ));
    assert_eq!(engine.durable_wal_records().len(), add_wal_before);
    engine
        .execute_text(
            9,
            "ALTER TABLE temporal_add ADD CONSTRAINT timestamp_before_day CHECK (ts <= '2000-01-01'::date)",
        )
        .unwrap();

    let table = engine.relational_catalog_table("temporal_checks").unwrap();
    assert_eq!(
        table.check_constraints[0].resolved_input_type,
        SqlType::Date
    );
    assert_eq!(
        table.check_constraints[1].resolved_input_type,
        SqlType::Timestamp
    );
    let digest = crate::engine_transaction_reset::table_schema_digest(&table).unwrap();
    let mut sabotaged = table.clone();
    sabotaged.check_constraints[0].resolved_input_type = SqlType::Timestamp;
    assert_ne!(
        digest,
        crate::engine_transaction_reset::table_schema_digest(&sabotaged).unwrap()
    );
    let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(engine.catalog_snapshot().as_ref()));
}

/// PG 3VL: a CHECK constraint is violated only when its predicate evaluates to FALSE — a NULL
/// operand makes it UNKNOWN, which SATISFIES the constraint ("the check expression should ...
/// yield true or the null value"). Pins: (a) an INSERT with a NULL checked value succeeds; (b) an
/// UPDATE setting the checked column to NULL succeeds; (c) ADD CHECK over existing NULL rows
/// succeeds; (d) a FALSE value still rejects everywhere. Was: NULL wrongly treated as a violation
/// (`select_filter_matches` returns false on NULL — correct for WHERE, wrong for CHECK).
#[test]
fn check_constraint_null_is_satisfied_pg_semantics() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE m (id INT PRIMARY KEY, v INT, CONSTRAINT v_pos CHECK (v > 0))",
    )
    .unwrap();
    // (a) NULL passes the CHECK on INSERT.
    e.execute_text(2, "INSERT INTO m (id, v) VALUES (1, NULL)")
        .unwrap();
    // (d) FALSE still rejects.
    assert!(e
        .execute_text(3, "INSERT INTO m (id, v) VALUES (2, -1)")
        .unwrap_err()
        .to_string()
        .contains("violates check constraint"));
    e.execute_text(4, "INSERT INTO m (id, v) VALUES (3, 5)")
        .unwrap();
    // (b) an UPDATE to NULL passes; an UPDATE to a FALSE value rejects.
    e.execute_text(5, "UPDATE m SET v = NULL WHERE id = 3")
        .unwrap();
    assert!(e
        .execute_text(6, "UPDATE m SET v = -7 WHERE id = 1")
        .unwrap_err()
        .to_string()
        .contains("violates check constraint"));
    // (c) ADD CHECK over existing NULL rows succeeds (both rows now hold v = NULL).
    e.execute_text(7, "ALTER TABLE m ADD CONSTRAINT v_cap CHECK (v < 1000)")
        .unwrap();
    // And ADD CHECK still rejects when a NON-NULL row violates.
    e.execute_text(8, "INSERT INTO m (id, v) VALUES (4, 500)")
        .unwrap();
    assert!(e
        .execute_text(9, "ALTER TABLE m ADD CONSTRAINT v_tiny CHECK (v < 100)")
        .unwrap_err()
        .to_string()
        .to_lowercase()
        .contains("violated"));
}

/// PG 3VL (MATCH SIMPLE): a NULL foreign-key value SATISFIES the constraint — it references
/// nothing, so no provider is required. Pins INSERT-NULL-fk passes, UPDATE-to-NULL-fk passes,
/// and a real missing key still rejects (both validator families share the rule).
#[test]
fn foreign_key_null_is_satisfied_pg_semantics() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE p (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE TABLE c (id INT PRIMARY KEY, pid INT)")
        .unwrap();
    e.execute_text(
        3,
        "ALTER TABLE ONLY c ADD CONSTRAINT c_fk FOREIGN KEY (pid) REFERENCES p(id)",
    )
    .unwrap();
    e.execute_text(4, "INSERT INTO p VALUES (1, 'a')").unwrap();
    // NULL fk passes (references nothing).
    e.execute_text(5, "INSERT INTO c (id, pid) VALUES (10, NULL)")
        .unwrap();
    // A real missing key still rejects.
    assert!(e
        .execute_text(6, "INSERT INTO c VALUES (11, 999)")
        .unwrap_err()
        .to_string()
        .contains("foreign key"));
    e.execute_text(7, "INSERT INTO c VALUES (12, 1)").unwrap();
    // UPDATE a valid fk to NULL passes.
    e.execute_text(8, "UPDATE c SET pid = NULL WHERE id = 12")
        .unwrap();
    // Deleting the now-unreferenced parent succeeds (NULL fks never pin a provider).
    e.execute_text(9, "DELETE FROM p WHERE id = 1").unwrap();
}

#[test]
fn relational_foreign_keys_enforce_and_replay_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE customers (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT)",
    )
    .unwrap();
    e.execute_text(3, "INSERT INTO customers (id, name) VALUES (1, 'Ada')")
        .unwrap();
    e.execute_text(4, "INSERT INTO orders (id, customer_id) VALUES (10, 1)")
        .unwrap();
    e.execute_text(
            5,
            "ALTER TABLE ONLY orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES customers(id)",
        )
        .unwrap();

    let invalid_insert = e
        .execute_text(6, "INSERT INTO orders (id, customer_id) VALUES (11, 99)")
        .unwrap_err()
        .to_string();
    assert!(invalid_insert.contains("violates foreign key constraint"));
    let invalid_child_update = e
        .execute_text(7, "UPDATE orders SET customer_id = 99 WHERE id = 10")
        .unwrap_err()
        .to_string();
    assert!(invalid_child_update.contains("violates foreign key constraint"));
    let invalid_parent_delete = e
        .execute_text(8, "DELETE FROM customers WHERE id = 1")
        .unwrap_err()
        .to_string();
    assert!(invalid_parent_delete.contains("violates foreign key constraint"));

    e.execute_text(
        9,
        "ALTER TABLE ONLY orders RENAME CONSTRAINT orders_customer_fk TO orders_customer_ref_fk",
    )
    .unwrap();
    e.execute_text(
        10,
        "ALTER TABLE ONLY customers RENAME COLUMN id TO customer_id",
    )
    .unwrap();
    e.execute_text(
        11,
        "ALTER TABLE IF EXISTS ONLY orders DROP CONSTRAINT orders_customer_ref_fk",
    )
    .unwrap();
    e.execute_text(12, "DELETE FROM customers WHERE customer_id = 1")
        .unwrap();

    let replayed = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    let orders = replayed.relational_catalog_table("orders").unwrap();
    assert!(orders.foreign_keys.is_empty());
    let customers = replayed.relational_catalog_table("customers").unwrap();
    assert_eq!(customers.columns[0].name, "customer_id");
}

#[test]
fn relational_primary_key_rejects_duplicates_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();

    let indexes = e
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .clone();
    assert_eq!(
        indexes,
        vec![RelationalIndex {
            oid: FIRST_TABLE_INDEX_OID,
            name: "people_pkey".to_string(),
            table: "people".to_string(),
            column: "id".to_string(),
            key_columns: vec!["id".to_string()],
            unique: true,
            primary_key: true,
            unique_constraint: false,
        }]
    );

    let duplicate_insert = e
        .execute_text(3, "INSERT INTO people (id, name) VALUES (1, 'Edsger')")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_insert.contains("duplicate key value violates unique index"),
        "{duplicate_insert}"
    );
    let duplicate_update = e
        .execute_text(4, "UPDATE people SET id = 1 WHERE name = 'Grace'")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_update.contains("duplicate key value violates unique index"),
        "{duplicate_update}"
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        indexes
    );

    let alter = Engine::new_local_test_engine();
    alter
        .execute_text(1, "CREATE TABLE teams (id INT, name TEXT)")
        .unwrap();
    alter
        .execute_text(
            2,
            "INSERT INTO teams (id, name) VALUES (1, 'core'), (2, 'db')",
        )
        .unwrap();
    alter
        .execute_text(
            3,
            "ALTER TABLE ONLY public.teams ADD CONSTRAINT teams_pkey PRIMARY KEY (id)",
        )
        .unwrap();
    assert!(alter
        .execute_text(4, "INSERT INTO teams (id, name) VALUES (1, 'dup')")
        .unwrap_err()
        .to_string()
        .contains("duplicate key value violates unique index"));

    let duplicate_existing = Engine::new_local_test_engine();
    duplicate_existing
        .execute_text(1, "CREATE TABLE dupes (id INT, name TEXT)")
        .unwrap();
    duplicate_existing
        .execute_text(2, "INSERT INTO dupes (id, name) VALUES (1, 'a'), (1, 'b')")
        .unwrap();
    let add_err = duplicate_existing
        .execute_text(
            3,
            "ALTER TABLE ONLY public.dupes ADD CONSTRAINT dupes_pkey PRIMARY KEY (id)",
        )
        .unwrap_err()
        .to_string();
    assert!(
        add_err.contains("duplicate key value violates unique index"),
        "{add_err}"
    );
}

#[test]
fn relational_catalog_drops_index_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(2, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    e.execute_text(3, "DROP INDEX public.people_name_idx")
        .unwrap();
    assert!(e
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .is_empty());

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .is_empty());

    e.execute_text(4, "DROP INDEX IF EXISTS people_name_idx")
        .unwrap();
    let missing_err = e
        .execute_text(5, "DROP INDEX people_name_idx")
        .unwrap_err()
        .to_string();
    assert!(
        missing_err.contains("index \"people_name_idx\" does not exist"),
        "{missing_err}"
    );

    let multi = Engine::new_local_test_engine();
    multi
        .execute_text(
            1,
            "CREATE TABLE people (id INT PRIMARY KEY, name TEXT, city TEXT)",
        )
        .unwrap();
    multi
        .execute_text(
            2,
            "INSERT INTO people (id, name, city) VALUES (1, 'Ada', 'London')",
        )
        .unwrap();
    multi
        .execute_text(3, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    multi
        .execute_text(4, "CREATE INDEX people_city_idx ON people (city)")
        .unwrap();
    multi
        .execute_text(
            5,
            "COMMENT ON INDEX public.people_name_idx IS 'name lookup'",
        )
        .unwrap();
    multi
        .execute_text(
            6,
            "COMMENT ON INDEX public.people_city_idx IS 'city lookup'",
        )
        .unwrap();
    let partial_err = multi
        .execute_text(7, "DROP INDEX public.people_name_idx, public.missing_idx")
        .unwrap_err()
        .to_string();
    assert!(
        partial_err.contains("index \"missing_idx\" does not exist"),
        "{partial_err}"
    );
    assert_eq!(
        multi.relational_index_comment("people_name_idx").as_deref(),
        Some("name lookup")
    );
    multi
        .execute_text(
            8,
            "DROP INDEX public.people_pkey, public.people_name_idx, public.people_city_idx",
        )
        .unwrap();
    let table_indexes = multi
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .clone();
    assert_eq!(table_indexes, Vec::<RelationalIndex>::new());
    assert_eq!(multi.relational_index_comment("people_name_idx"), None);
    assert_eq!(multi.relational_index_comment("people_city_idx"), None);
    let Command::Select(select) =
        parse_command("SELECT id, name FROM people WHERE id = 1").unwrap()
    else {
        panic!("expected SELECT");
    };
    let rows = multi.execute_relational_select(&select).unwrap();
    assert_eq!(
        rows.rows,
        vec![vec![SqlValue::Int4(1), SqlValue::Text("Ada".to_string())]]
    );

    let recovered = Engine::recover_from_durable_wal(&multi.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        table_indexes
    );
}

#[test]
fn relational_catalog_renames_index_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (2, 'Linus')",
    )
    .unwrap();
    e.execute_text(3, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    e.execute_text(4, "COMMENT ON INDEX public.people_name_idx IS 'lookup'")
        .unwrap();
    e.execute_text(
        5,
        "ALTER INDEX public.people_name_idx RENAME TO people_lookup_idx",
    )
    .unwrap();

    let table = e.relational_catalog_table("people").unwrap();
    let renamed_indexes = table.indexes.clone();
    assert_eq!(
        renamed_indexes,
        vec![RelationalIndex {
            oid: FIRST_TABLE_INDEX_OID,
            name: "people_lookup_idx".to_string(),
            table: "people".to_string(),
            column: "name".to_string(),
            key_columns: vec!["name".to_string()],
            unique: false,
            primary_key: false,
            unique_constraint: false,
        }]
    );
    assert_eq!(
        e.relational_index_comment("people_lookup_idx").as_deref(),
        Some("lookup")
    );
    assert_eq!(e.relational_index_comment("people_name_idx"), None);

    let Command::Select(select) =
        parse_command("SELECT id FROM people WHERE name = 'Linus'").unwrap()
    else {
        panic!("expected SELECT plan");
    };
    let result = e.execute_relational_select(&select).unwrap();
    assert_recovered_relational_access_path(
        &result,
        RelationalAccessPath::EqualityIndex {
            table: "people".to_string(),
            column: "name".to_string(),
            matched_keys: 1,
        },
    );
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(2)]]);

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert_eq!(
        recovered
            .relational_catalog_table("people")
            .unwrap()
            .indexes,
        renamed_indexes
    );
    assert_eq!(
        recovered
            .relational_index_comment("people_lookup_idx")
            .as_deref(),
        Some("lookup")
    );

    let duplicate_err = e
        .execute_text(
            6,
            "ALTER INDEX people_lookup_idx RENAME TO people_lookup_idx",
        )
        .unwrap_err()
        .to_string();
    assert!(
        duplicate_err.contains("relation \"people_lookup_idx\" already exists"),
        "{duplicate_err}"
    );
    let missing_err = e
        .execute_text(7, "ALTER INDEX people_name_idx RENAME TO people_old_idx")
        .unwrap_err()
        .to_string();
    assert!(
        missing_err.contains("index \"people_name_idx\" does not exist"),
        "{missing_err}"
    );

    let constrained = Engine::new_local_test_engine();
    constrained
        .execute_text(1, "CREATE TABLE keyed_people (id INT PRIMARY KEY)")
        .unwrap();
    constrained
        .execute_text(2, "COMMENT ON INDEX keyed_people_pkey IS 'primary lookup'")
        .unwrap();
    constrained
        .execute_text(
            3,
            "COMMENT ON CONSTRAINT keyed_people_pkey ON keyed_people IS 'primary identity'",
        )
        .unwrap();
    let before = constrained.catalog_snapshot();
    let wal_before = constrained.durable_wal_records().len();
    let error = constrained
        .execute_text(
            4,
            "ALTER INDEX keyed_people_pkey RENAME TO keyed_people_id_idx",
        )
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("cannot rename constraint-backed index with ALTER INDEX"),
        "{error}"
    );
    assert_eq!(constrained.durable_wal_records().len(), wal_before);
    assert!(constrained
        .catalog_snapshot()
        .same_contents(before.as_ref()));
    assert_eq!(
        constrained
            .relational_index_comment("keyed_people_pkey")
            .as_deref(),
        Some("primary lookup")
    );
    assert_eq!(
        constrained
            .relational_constraint_comment("keyed_people", "keyed_people_pkey")
            .as_deref(),
        Some("primary identity")
    );
    let recovered = Engine::recover_from_durable_wal(&constrained.durable_wal_records()).unwrap();
    assert!(recovered
        .catalog_snapshot()
        .same_contents(constrained.catalog_snapshot().as_ref()));
}

#[test]
fn relational_catalog_drops_constraints_and_replays_from_wal() {
    let e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE people (id INT PRIMARY KEY, name TEXT UNIQUE)",
    )
    .unwrap();
    e.execute_text(
        2,
        "COMMENT ON INDEX public.people_name_key IS 'name uniqueness'",
    )
    .unwrap();
    e.execute_text(
        3,
        "COMMENT ON CONSTRAINT people_pkey ON public.people IS 'row identity'",
    )
    .unwrap();
    e.execute_text(
        4,
        "ALTER TABLE IF EXISTS ONLY public.people DROP CONSTRAINT IF EXISTS people_pkey",
    )
    .unwrap();
    e.execute_text(
        5,
        "ALTER TABLE ONLY public.people DROP CONSTRAINT people_name_key",
    )
    .unwrap();

    let table = e.relational_catalog_table("people").unwrap();
    assert!(table.indexes.is_empty());
    assert_eq!(e.relational_index_comment("people_name_key"), None);
    assert_eq!(
        e.relational_constraint_comment("people", "people_pkey"),
        None
    );
    e.execute_text(
        6,
        "INSERT INTO people (id, name) VALUES (1, 'Ada'), (1, 'Ada')",
    )
    .unwrap();

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered
        .relational_catalog_table("people")
        .unwrap()
        .indexes
        .is_empty());
    assert_eq!(recovered.relational_index_comment("people_name_key"), None);
    assert_eq!(
        recovered.relational_constraint_comment("people", "people_pkey"),
        None
    );

    e.execute_text(
        7,
        "ALTER TABLE ONLY public.people DROP CONSTRAINT IF EXISTS people_pkey",
    )
    .unwrap();
    let missing_constraint = e
        .execute_text(
            8,
            "ALTER TABLE ONLY public.people DROP CONSTRAINT people_pkey",
        )
        .unwrap_err()
        .to_string();
    assert!(
        missing_constraint.contains("constraint \"people_pkey\" does not exist"),
        "{missing_constraint}"
    );
    let missing_table = e
        .execute_text(
            9,
            "ALTER TABLE ONLY public.missing_people DROP CONSTRAINT people_pkey",
        )
        .unwrap_err()
        .to_string();
    assert!(
        missing_table.contains("relation \"missing_people\" does not exist"),
        "{missing_table}"
    );
}

#[test]
fn relational_catalog_drops_table_and_replays_from_wal() {
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
    e.execute_text(4, "CREATE INDEX people_name_idx ON people (name)")
        .unwrap();
    e.execute_text(5, "COMMENT ON TABLE public.people IS 'people table'")
        .unwrap();
    e.execute_text(6, "COMMENT ON COLUMN public.people.name IS 'display name'")
        .unwrap();
    e.execute_text(7, "COMMENT ON INDEX public.people_name_idx IS 'lookup'")
        .unwrap();
    e.execute_text(
        8,
        "COMMENT ON CONSTRAINT people_pkey ON public.people IS 'identity'",
    )
    .unwrap();
    e.execute_text(9, "COMMENT ON ROLE postgres IS 'bootstrap role'")
        .unwrap();
    let snapshot = e.populate_relational_residency_snapshot("people").unwrap();
    assert!(snapshot.is_valid());
    // TYPE-COVERAGE #14 (text): a PK'd text table is SHARD-resident (device-authoritative), not
    // single-buffer, so check either representation.
    assert!(
        e.relational_residency_snapshot("people").is_some() || e.resident_shard_count("people") > 0
    );
    e.execute_text(10, "DROP TABLE public.people").unwrap();

    assert!(e.relational_catalog_table("people").is_none());
    assert!(e.relational_catalog_table("teams").is_some());
    assert!(
        e.relational_residency_snapshot("people").is_none()
            && e.resident_shard_count("people") == 0
    );
    assert!(!e.read_state.residency.device_memory.contains_key("people"));
    assert_eq!(e.relational_table_comment("people"), None);
    assert_eq!(e.relational_column_comment("people", 2), None);
    assert_eq!(e.relational_index_comment("people_name_idx"), None);
    assert_eq!(
        e.relational_constraint_comment("people", "people_pkey"),
        None
    );
    assert_eq!(
        e.relational_role_comment("postgres").as_deref(),
        Some("bootstrap role")
    );
    let missing_select = e
        .execute_text(11, "INSERT INTO people (id, name) VALUES (3, 'Edsger')")
        .unwrap_err()
        .to_string();
    assert!(
        missing_select.contains("relation \"people\" does not exist"),
        "{missing_select}"
    );

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_catalog_table("people").is_none());
    assert!(recovered.relational_catalog_table("teams").is_some());
    assert_eq!(recovered.relational_table_comment("people"), None);
    assert_eq!(
        recovered.relational_role_comment("postgres").as_deref(),
        Some("bootstrap role")
    );

    e.execute_text(12, "DROP TABLE IF EXISTS people").unwrap();
    let missing_drop = e
        .execute_text(13, "DROP TABLE people")
        .unwrap_err()
        .to_string();
    assert!(
        missing_drop.contains("relation \"people\" does not exist"),
        "{missing_drop}"
    );

    let with_view = Engine::new_local_test_engine();
    with_view
        .execute_text(1, "CREATE TABLE people (id INT, name TEXT)")
        .unwrap();
    with_view
        .execute_text(2, "CREATE VIEW public.people_view AS SELECT * FROM people")
        .unwrap();
    let view_drop = with_view
        .execute_text(3, "DROP TABLE people_view")
        .unwrap_err()
        .to_string();
    assert!(
        view_drop.contains("relation \"people_view\" is not a table"),
        "{view_drop}"
    );
    assert!(with_view.relational_catalog_view("people_view").is_some());
}

#[test]
fn relational_catalog_drops_table_batches_atomically_and_replays() {
    let mut e = Engine::new_local_test_engine();
    e.execute_text(
        1,
        "CREATE TABLE batch_people (id INT PRIMARY KEY, name TEXT)",
    )
    .unwrap();
    e.execute_text(2, "CREATE TABLE batch_teams (id INT, name TEXT UNIQUE)")
        .unwrap();
    e.execute_text(3, "CREATE TABLE batch_keep (id INT, name TEXT)")
        .unwrap();
    e.execute_text(
        4,
        "INSERT INTO batch_people (id, name) VALUES (1, 'Ada'), (2, 'Grace')",
    )
    .unwrap();
    e.execute_text(
        5,
        "INSERT INTO batch_teams (id, name) VALUES (10, 'Compiler')",
    )
    .unwrap();
    e.execute_text(
        6,
        "CREATE INDEX batch_people_name_idx ON batch_people (name)",
    )
    .unwrap();
    e.execute_text(7, "COMMENT ON TABLE public.batch_people IS 'people table'")
        .unwrap();
    e.execute_text(
        8,
        "COMMENT ON COLUMN public.batch_teams.name IS 'team name'",
    )
    .unwrap();
    e.execute_text(
        9,
        "COMMENT ON INDEX public.batch_people_name_idx IS 'lookup'",
    )
    .unwrap();
    assert!(e
        .populate_relational_residency_snapshot("batch_people")
        .unwrap()
        .is_valid());

    e.execute_text(10, "DROP TABLE public.batch_people, public.batch_teams")
        .unwrap();

    assert!(e.relational_catalog_table("batch_people").is_none());
    assert!(e.relational_catalog_table("batch_teams").is_none());
    assert!(e.relational_catalog_table("batch_keep").is_some());
    assert_eq!(e.relational_table_comment("batch_people"), None);
    assert_eq!(e.relational_column_comment("batch_teams", 2), None);
    assert_eq!(e.relational_index_comment("batch_people_name_idx"), None);
    assert!(e.relational_residency_snapshot("batch_people").is_none());

    let recovered = Engine::recover_from_durable_wal(&e.durable_wal_records()).unwrap();
    assert!(recovered.relational_catalog_table("batch_people").is_none());
    assert!(recovered.relational_catalog_table("batch_teams").is_none());
    assert!(recovered.relational_catalog_table("batch_keep").is_some());

    let atomic = Engine::new_local_test_engine();
    atomic
        .execute_text(1, "CREATE TABLE atomic_people (id INT, name TEXT)")
        .unwrap();
    atomic
        .execute_text(2, "CREATE TABLE atomic_teams (id INT, name TEXT)")
        .unwrap();
    atomic
        .execute_text(
            3,
            "CREATE VIEW public.atomic_view AS SELECT id, name FROM atomic_people",
        )
        .unwrap();
    let missing = atomic
        .execute_text(4, "DROP TABLE atomic_people, missing_atomic")
        .unwrap_err()
        .to_string();
    assert!(
        missing.contains("relation \"missing_atomic\" does not exist"),
        "{missing}"
    );
    assert!(atomic.relational_catalog_table("atomic_people").is_some());
    assert!(atomic.relational_catalog_table("atomic_teams").is_some());

    let duplicate = atomic
        .execute_text(5, "DROP TABLE atomic_people, atomic_people")
        .unwrap_err()
        .to_string();
    assert!(
        duplicate.contains("table \"atomic_people\" specified more than once"),
        "{duplicate}"
    );
    assert!(atomic.relational_catalog_table("atomic_people").is_some());

    let view_target = atomic
        .execute_text(6, "DROP TABLE IF EXISTS missing_atomic, atomic_view")
        .unwrap_err()
        .to_string();
    assert!(
        view_target.contains("relation \"atomic_view\" is not a table"),
        "{view_target}"
    );
    assert!(atomic.relational_catalog_view("atomic_view").is_some());
    assert!(atomic.relational_catalog_table("atomic_people").is_some());

    atomic
        .execute_text(7, "DROP TABLE IF EXISTS missing_atomic, atomic_people")
        .unwrap();
    assert!(atomic.relational_catalog_table("atomic_people").is_none());
    assert!(atomic.relational_catalog_table("atomic_teams").is_some());
}
