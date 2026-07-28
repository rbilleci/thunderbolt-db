//! Direct-façade temporal SQLSTATE coverage; pgwire Bind coverage stays with the wire codec.

use super::*;

#[test]
fn default_assignment_mismatches_are_42804_at_the_facade_boundary() {
    let shared = SharedEngine::new();
    for sql in [
        "CREATE TABLE facade_default_bool (value BOOL DEFAULT 1)",
        "CREATE TABLE facade_default_null (value INT DEFAULT NULL::text)",
    ] {
        let error = submit_ephemeral_text(&shared, sql).expect_err("default assignment mismatch");
        assert_eq!(error.category, ErrorCategory::DatatypeMismatch, "{sql}");
        assert_eq!(pg_adapter::error_sqlstate(error.category), "42804", "{sql}");
    }

    submit_ephemeral_text(
        &shared,
        "CREATE TABLE facade_default_alter (id INT, value BOOL)",
    )
    .unwrap();
    for sql in [
        "ALTER TABLE facade_default_alter ADD COLUMN added BOOL DEFAULT 1",
        "ALTER TABLE facade_default_alter ADD COLUMN added_null INT DEFAULT NULL::text",
        "ALTER TABLE facade_default_alter ALTER COLUMN value SET DEFAULT 1",
        "ALTER TABLE facade_default_alter ALTER COLUMN id SET DEFAULT NULL::text",
    ] {
        let error = submit_ephemeral_text(&shared, sql).expect_err("default assignment mismatch");
        assert_eq!(error.category, ErrorCategory::DatatypeMismatch, "{sql}");
        assert_eq!(pg_adapter::error_sqlstate(error.category), "42804", "{sql}");
    }

    submit_ephemeral_text(&shared, "CREATE SEQUENCE facade_default_nextval_sequence").unwrap();
    let error = submit_ephemeral_text(
        &shared,
        "CREATE TABLE facade_default_nextval (value BOOLEAN DEFAULT nextval('facade_default_nextval_sequence'::regclass))",
    )
    .expect_err("nextval target mismatch");
    assert_eq!(error.category, ErrorCategory::DatatypeMismatch);
    assert_eq!(pg_adapter::error_sqlstate(error.category), "42804");
}

#[test]
fn sequence_default_target_resolution_preserves_facade_sqlstates_across_ddl() {
    const MISSING: &str = "facade_missing_default_sequence";
    const EXISTING: &str = "facade_existing_default_sequence";
    const NON_SEQUENCE: &str = "facade_non_sequence_default_target";

    let shared = SharedEngine::new();
    submit_ephemeral_text(&shared, &format!("CREATE SEQUENCE {EXISTING}")).unwrap();
    submit_ephemeral_text(&shared, &format!("CREATE TABLE {NON_SEQUENCE} (id INT)")).unwrap();
    submit_ephemeral_text(&shared, "CREATE TABLE facade_sequence_add (id INT)").unwrap();
    submit_ephemeral_text(
        &shared,
        "CREATE TABLE facade_sequence_alter (missing_text TEXT, existing_bool BOOLEAN, missing_int INT, non_sequence_bool BOOLEAN)",
    )
    .unwrap();

    for (sql, expected) in [
        (
            format!(
                "CREATE TABLE facade_sequence_create_missing_text (value TEXT DEFAULT nextval('{MISSING}'::regclass))"
            ),
            "42P01",
        ),
        (
            format!(
                "ALTER TABLE facade_sequence_add ADD COLUMN missing_text TEXT DEFAULT nextval('{MISSING}'::regclass)"
            ),
            "42P01",
        ),
        (
            format!(
                "ALTER TABLE facade_sequence_alter ALTER COLUMN missing_text SET DEFAULT nextval('{MISSING}'::regclass)"
            ),
            "42P01",
        ),
        (
            format!(
                "CREATE TABLE facade_sequence_create_missing_int (value INT DEFAULT nextval('{MISSING}'::regclass))"
            ),
            "42P01",
        ),
        (
            format!(
                "ALTER TABLE facade_sequence_add ADD COLUMN missing_int INT DEFAULT nextval('{MISSING}'::regclass)"
            ),
            "42P01",
        ),
        (
            format!(
                "ALTER TABLE facade_sequence_alter ALTER COLUMN missing_int SET DEFAULT nextval('{MISSING}'::regclass)"
            ),
            "42P01",
        ),
        (
            format!(
                "CREATE TABLE facade_sequence_create_existing_bool (value BOOLEAN DEFAULT nextval('{EXISTING}'::regclass))"
            ),
            "42804",
        ),
        (
            format!(
                "ALTER TABLE facade_sequence_add ADD COLUMN existing_bool BOOLEAN DEFAULT nextval('{EXISTING}'::regclass)"
            ),
            "42804",
        ),
        (
            format!(
                "ALTER TABLE facade_sequence_alter ALTER COLUMN existing_bool SET DEFAULT nextval('{EXISTING}'::regclass)"
            ),
            "42804",
        ),
        (
            format!(
                "CREATE TABLE facade_sequence_create_non_sequence_bool (value BOOLEAN DEFAULT nextval('{NON_SEQUENCE}'::regclass))"
            ),
            "42804",
        ),
        (
            format!(
                "ALTER TABLE facade_sequence_add ADD COLUMN non_sequence_bool BOOLEAN DEFAULT nextval('{NON_SEQUENCE}'::regclass)"
            ),
            "42804",
        ),
        (
            format!(
                "ALTER TABLE facade_sequence_alter ALTER COLUMN non_sequence_bool SET DEFAULT nextval('{NON_SEQUENCE}'::regclass)"
            ),
            "42804",
        ),
        (
            format!(
                "CREATE TABLE facade_sequence_order_mismatch_first (a BOOLEAN DEFAULT 1, b TEXT DEFAULT nextval('{MISSING}'::regclass))"
            ),
            "42804",
        ),
        (
            format!(
                "CREATE TABLE facade_sequence_order_missing_first (a TEXT DEFAULT nextval('{MISSING}'::regclass), b BOOLEAN DEFAULT 1)"
            ),
            "42P01",
        ),
    ] {
        let error = submit_ephemeral_text(&shared, &sql).expect_err("sequence default target");
        assert_eq!(pg_adapter::error_sqlstate(error.category), expected, "{sql}");
    }
}

#[test]
fn add_column_target_errors_precede_default_sqlstates_at_the_facade_boundary() {
    let shared = SharedEngine::new();
    submit_ephemeral_text(&shared, "CREATE TABLE facade_add_default_target (id INT)").unwrap();

    for (sql, expected) in [
        (
            "ALTER TABLE facade_missing_add_default_target ADD COLUMN value BOOLEAN DEFAULT 1",
            "42P01",
        ),
        (
            "ALTER TABLE facade_add_default_target ADD COLUMN id BOOLEAN DEFAULT 1",
            "42701",
        ),
        (
            "ALTER TABLE facade_add_default_target ADD COLUMN id INT DEFAULT nextval('facade_missing_duplicate_default_sequence'::regclass)",
            "42701",
        ),
    ] {
        let error = submit_ephemeral_text(&shared, sql).expect_err("ADD COLUMN target error");
        assert_eq!(pg_adapter::error_sqlstate(error.category), expected, "{sql}");
    }
}

#[test]
fn alter_default_target_directed_input_errors_preserve_facade_sqlstates() {
    let shared = SharedEngine::new();
    submit_ephemeral_text(
        &shared,
        "CREATE TABLE facade_default_targeted (id INT, day DATE)",
    )
    .unwrap();

    for (sql, expected) in [
        (
            "ALTER TABLE facade_default_targeted ALTER COLUMN id SET DEFAULT 'not-an-int'",
            "22P02",
        ),
        (
            "ALTER TABLE facade_default_targeted ALTER COLUMN day SET DEFAULT 'not-a-date'",
            "22007",
        ),
        (
            "ALTER TABLE facade_default_targeted ALTER COLUMN day SET DEFAULT '2024-02-30'",
            "22008",
        ),
        (
            "ALTER TABLE facade_default_targeted ALTER COLUMN id SET DEFAULT '2147483648'",
            "22003",
        ),
    ] {
        let error = submit_ephemeral_text(&shared, sql).expect_err("target-directed default");
        assert_eq!(
            pg_adapter::error_sqlstate(error.category),
            expected,
            "{sql}"
        );
    }
}

#[test]
fn temporal_check_and_insert_coercions_preserve_postgresql_taxonomy() {
    let shared = SharedEngine::new();
    for sql in [
        "CREATE TABLE field_date (d date, CHECK (d > '2024-02-30'::date))",
        "CREATE TABLE field_timestamp (t timestamp, CHECK ('294277-12-31 00:00:00'::timestamp < t))",
    ] {
        let error = submit_ephemeral_text(&shared, sql).expect_err("explicit overflow cast");
        assert_eq!(error.category, ErrorCategory::DatetimeFieldOverflow, "{sql}");
    }
    for (sql, expected) in [
        (
            "CREATE TABLE dt_day (d date, CHECK (d > '2024-02-30'))",
            "22008",
        ),
        (
            "CREATE TABLE dt_month (d date, CHECK (d > '2024-13-01'))",
            "22008",
        ),
        (
            "CREATE TABLE dt_year (d date, CHECK (d > '0000-01-01'))",
            "22008",
        ),
        (
            "CREATE TABLE dt_ts_day (t timestamp, CHECK (t > '2024-02-30 00:00:00'))",
            "22008",
        ),
        (
            "CREATE TABLE dt_ts_hour (t timestamp, CHECK (t > '2024-01-01 25:00:00'))",
            "22008",
        ),
        (
            "CREATE TABLE dt_ts_minute (t timestamp, CHECK (t > '2024-01-01 10:60:00'))",
            "22008",
        ),
        (
            "CREATE TABLE dt_ts_range (t timestamp, CHECK (t > '294277-12-31 00:00:00'))",
            "22008",
        ),
        (
            "CREATE TABLE dt_cross_date (d date, CHECK (d > '2024-02-30 00:00:00'::timestamp))",
            "22008",
        ),
        (
            "CREATE TABLE dt_cross_timestamp (t timestamp, CHECK (t > '2024-02-30'::date))",
            "22008",
        ),
    ] {
        let error = submit_ephemeral_text(&shared, sql).expect_err("temporal CHECK field overflow");
        assert_eq!(
            pg_adapter::error_sqlstate(error.category),
            expected,
            "{sql}: {error:?}"
        );
    }

    submit_ephemeral_text(
        &shared,
        "CREATE TABLE datetime_insert_error (d date, t timestamp)",
    )
    .unwrap();
    for sql in [
        "INSERT INTO datetime_insert_error VALUES ('2024-02-30', '2024-01-01 00:00:00')",
        "INSERT INTO datetime_insert_error VALUES ('2024-01-01', '2024-01-01 25:00:00')",
    ] {
        let error = submit_ephemeral_text(&shared, sql).expect_err("invalid temporal INSERT");
        assert_eq!(pg_adapter::error_sqlstate(error.category), "22008", "{sql}");
    }
}

#[test]
fn datetime_field_overflow_parse_error_keeps_direct_and_execute_categories() {
    assert_eq!(
        map_parse_error(ParseError::DatetimeFieldOverflow {
            input: "2024-02-30".to_string(),
        })
        .category,
        ErrorCategory::DatetimeFieldOverflow
    );
    assert_eq!(
        map_execute_error(ExecuteError::Parse(ParseError::DatetimeFieldOverflow {
            input: "2024-02-30".to_string(),
        }))
        .category,
        ErrorCategory::DatetimeFieldOverflow
    );
}

#[test]
fn temporal_result_text_codec_emits_postgresql_bc_at_finite_lower_boundaries() {
    assert_eq!(
        pg_adapter::db_value_text(&DbValue::Date(gpu_db_sql::datetime::PG_DATE_MIN_DAYS)),
        "4714-11-24 BC"
    );
    assert_eq!(
        pg_adapter::db_value_text(&DbValue::Timestamp(
            gpu_db_sql::datetime::PG_TIMESTAMP_MIN_MICROS
        )),
        "4714-11-24 00:00:00 BC"
    );
}

#[test]
fn temporal_text_bind_respects_pg16_workspace_and_fractional_two_field_time() {
    let date = format!("{}2000-01-01", "0".repeat(118));
    assert_eq!(
        pg_adapter::decode_parameter(1082, 0, Some(date.as_bytes())),
        Ok(DbValue::Date(0))
    );
    let date_over = format!("{}2000-01-01", "0".repeat(119));
    assert!(matches!(
        pg_adapter::decode_parameter(1082, 0, Some(date_over.as_bytes())),
        Err(pg_adapter::PgValueCodecError::InvalidValue { format: 0, .. })
    ));

    let timestamp = format!("2000-01-01 00:00:00.5{}", "0".repeat(131));
    assert_eq!(
        pg_adapter::decode_parameter(1114, 0, Some(timestamp.as_bytes())),
        Ok(DbValue::Timestamp(500_000))
    );
    let timestamp_over = format!("2000-01-01 00:00:00.5{}", "0".repeat(132));
    assert!(matches!(
        pg_adapter::decode_parameter(1114, 0, Some(timestamp_over.as_bytes())),
        Err(pg_adapter::PgValueCodecError::InvalidValue { format: 0, .. })
    ));

    for (text, expected) in [
        ("2000-01-01 00:00:00.", 0),
        ("2000-01-01 10:20.5", 620_500_000),
        ("2000-01-01 23:59.5", 1_439_500_000),
    ] {
        assert_eq!(
            pg_adapter::decode_parameter(1114, 0, Some(text.as_bytes())),
            Ok(DbValue::Timestamp(expected)),
            "{text}"
        );
    }
    assert!(matches!(
        pg_adapter::decode_parameter(1114, 0, Some(b"2000-01-01 60:00.5")),
        Err(pg_adapter::PgValueCodecError::DatetimeFieldOverflow { format: 0, .. })
    ));
}

#[test]
fn temporal_text_bind_preserves_pg16_time_field_and_later_token_precedence() {
    for text in [
        b"2000-01-01 25:00:00.abc".as_slice(),
        b"2000-02-30 25:00:00",
        b"2000-02-30 00:00:00",
    ] {
        assert!(matches!(
            pg_adapter::decode_parameter(1114, 0, Some(text)),
            Err(pg_adapter::PgValueCodecError::DatetimeFieldOverflow { format: 0, .. })
        ));
    }
    for text in [
        b"2000-02-30 00:00:00.abc".as_slice(),
        b"2000-01-01 00:00:00.abc",
    ] {
        assert!(matches!(
            pg_adapter::decode_parameter(1114, 0, Some(text)),
            Err(pg_adapter::PgValueCodecError::InvalidValue { format: 0, .. })
        ));
    }
}

#[test]
fn temporal_text_bind_decodes_date_numeric_overflow_before_an_adjacent_trailing_t() {
    for text in [
        b"999999999999999999999-01-01T".as_slice(),
        b"2000-999999999999999999999-01T",
        b"2000-01-999999999999999999999T",
    ] {
        assert!(matches!(
            pg_adapter::decode_parameter(1114, 0, Some(text)),
            Err(pg_adapter::PgValueCodecError::DatetimeFieldOverflow { format: 0, .. })
        ));
    }
}
