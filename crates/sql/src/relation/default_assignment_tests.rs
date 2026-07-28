use super::*;

fn parse_relational(sql: &str) -> Command {
    parse_relational_command(sql, false)
        .expect("relational shape")
        .expect("supported relational command")
}

#[test]
fn inferred_defaults_use_assignment_casts_while_unknown_literals_remain_target_directed() {
    let Command::CreateTable(create) = parse_relational(
        "CREATE TABLE inferred_defaults (unknown_int INT DEFAULT '7', unknown_bool BOOL DEFAULT 'true', inferred_int_to_numeric NUMERIC(8,2) DEFAULT 5, inferred_int_to_text TEXT DEFAULT 5, inferred_bool_to_text TEXT DEFAULT true)",
    ) else {
        panic!("expected CREATE TABLE");
    };
    assert_eq!(
        create.columns[0].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Int4(7),
            input: DefaultInputType::TargetTyped,
        })
    );
    assert_eq!(
        create.columns[1].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Bool(true),
            input: DefaultInputType::TargetTyped,
        })
    );
    assert_eq!(
        create.columns[2].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Int4(5),
            input: DefaultInputType::Inferred(SqlType::Int4),
        })
    );
    assert_eq!(
        create.columns[3].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Int4(5),
            input: DefaultInputType::Inferred(SqlType::Int4),
        })
    );
    assert_eq!(
        create.columns[4].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Bool(true),
            input: DefaultInputType::Inferred(SqlType::Bool),
        })
    );

    let Command::AddColumn(int_to_numeric) = parse_relational(
        "ALTER TABLE inferred_defaults ADD COLUMN inferred_int_to_numeric NUMERIC(8,2) DEFAULT 5",
    ) else {
        panic!("expected ADD COLUMN");
    };
    assert_eq!(
        int_to_numeric.column.default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Int4(5),
            input: DefaultInputType::Inferred(SqlType::Int4),
        })
    );
    let Command::AddColumn(int_to_text) = parse_relational(
        "ALTER TABLE inferred_defaults ADD COLUMN inferred_int_to_text TEXT DEFAULT 5",
    ) else {
        panic!("expected ADD COLUMN");
    };
    assert!(matches!(
        int_to_text.column.default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Int4(5),
            input: DefaultInputType::Inferred(SqlType::Int4),
        })
    ));
    let Command::AddColumn(bool_to_text) = parse_relational(
        "ALTER TABLE inferred_defaults ADD COLUMN inferred_bool_to_text TEXT DEFAULT true",
    ) else {
        panic!("expected ADD COLUMN");
    };
    assert!(matches!(
        bool_to_text.column.default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Bool(true),
            input: DefaultInputType::Inferred(SqlType::Bool),
        })
    ));

    for sql in [
        "CREATE TABLE rejected_inferred_bool (flag BOOL DEFAULT 1)",
        "CREATE TABLE rejected_inferred_bool_zero (flag BOOL DEFAULT 0)",
        "ALTER TABLE inferred_defaults ADD COLUMN rejected_inferred_bool BOOL DEFAULT 1",
        "ALTER TABLE inferred_defaults ADD COLUMN rejected_inferred_bool_zero BOOL DEFAULT 0",
    ] {
        assert!(matches!(
            parse_relational_command(sql, false),
            Some(Ok(Command::CreateTable(_) | Command::AddColumn(_)))
        ));
    }
}

#[test]
fn typed_defaults_apply_postgresql_assignment_casts_and_target_validation() {
    let Command::CreateTable(create) = parse_relational(
        "CREATE TABLE typed_defaults (unknown_int INT DEFAULT '7', exact_text TEXT DEFAULT 'ok'::text, int_to_numeric NUMERIC(8,2) DEFAULT 5::int4, int_to_bigint INT8 DEFAULT 5::int4, numeric_to_int INT DEFAULT 1.5::numeric(10,1), int_to_text TEXT DEFAULT 5::int4, numeric_to_text TEXT DEFAULT 1.5::numeric(10,1), bool_to_text TEXT DEFAULT true::bool, date_to_text TEXT DEFAULT '2024-01-02'::date, timestamp_to_text TEXT DEFAULT '2024-01-02 03:04:05'::timestamp, uuid_to_text TEXT DEFAULT '00000000-0000-0000-0000-000000000001'::uuid, timestamp_to_date DATE DEFAULT '2024-01-02 03:04:05'::timestamp, date_to_timestamp TIMESTAMP DEFAULT '2024-01-02'::date)",
    ) else {
        panic!("expected CREATE TABLE");
    };
    assert!(matches!(
        create.columns[0].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Int4(7),
            input: DefaultInputType::TargetTyped,
        })
    ));
    assert!(matches!(
        create.columns[1].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Text(_),
            input: DefaultInputType::Explicit(SqlType::Text),
        })
    ));
    assert!(matches!(
        create.columns[2].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Int4(5),
            input: DefaultInputType::Explicit(SqlType::Int4),
        })
    ));
    assert!(matches!(
        create.columns[4].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Numeric(_),
            input: DefaultInputType::Explicit(SqlType::Numeric {
                precision: 10,
                scale: 1
            }),
        })
    ));
    for column in 5..=10 {
        assert!(matches!(
            create.columns[column].default,
            Some(ColumnDefault::DeferredScalar {
                input: DefaultInputType::Explicit(_),
                ..
            })
        ));
    }
    assert!(matches!(
        create.columns[11].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Timestamp(_),
            input: DefaultInputType::Explicit(SqlType::Timestamp),
        })
    ));
    assert!(matches!(
        create.columns[12].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Date(_),
            input: DefaultInputType::Explicit(SqlType::Date),
        })
    ));
}

#[test]
fn typed_add_column_defaults_share_assignment_cast_matrix_and_rejections() {
    let Command::AddColumn(numeric_to_int) =
        parse_relational("ALTER TABLE defaults ADD COLUMN rounded INT DEFAULT 1.5::numeric(10,1)")
    else {
        panic!("expected ADD COLUMN");
    };
    assert!(matches!(
        numeric_to_int.column.default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Numeric(_),
            input: DefaultInputType::Explicit(SqlType::Numeric {
                precision: 10,
                scale: 1
            }),
        })
    ));
    let Command::AddColumn(scalar_to_text) =
        parse_relational("ALTER TABLE defaults ADD COLUMN rendered TEXT DEFAULT true::bool")
    else {
        panic!("expected ADD COLUMN");
    };
    assert!(matches!(
        scalar_to_text.column.default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Bool(true),
            input: DefaultInputType::Explicit(SqlType::Bool),
        })
    ));
    for sql in [
        "CREATE TABLE invalid_default (id INT DEFAULT '7'::text)",
        "CREATE TABLE invalid_bool_default (id INT DEFAULT true::bool)",
        "ALTER TABLE defaults ADD COLUMN rejected INT DEFAULT '7'::text",
        "ALTER TABLE defaults ADD COLUMN rejected_date INT DEFAULT '2024-01-02'::date",
    ] {
        assert!(matches!(
            parse_relational_command(sql, false),
            Some(Ok(Command::CreateTable(_) | Command::AddColumn(_)))
        ));
    }
}

#[test]
fn typed_null_defaults_follow_the_same_assignment_matrix() {
    let Command::CreateTable(create) = parse_relational(
        "CREATE TABLE typed_null_defaults (unknown_null INT DEFAULT NULL, exact_null INT DEFAULT NULL::int4, numeric_to_int INT DEFAULT NULL::numeric(10,1), bool_to_text TEXT DEFAULT NULL::bool, timestamp_to_date DATE DEFAULT NULL::timestamp)",
    ) else {
        panic!("expected CREATE TABLE");
    };
    assert!(matches!(
        create.columns[0].default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Null,
            input: DefaultInputType::TargetTyped,
        })
    ));
    assert!(create.columns[1..].iter().all(|column| matches!(
        column.default,
        Some(ColumnDefault::DeferredScalar {
            value: SqlValue::Null,
            ..
        })
    )));
    for sql in [
        "ALTER TABLE typed_null_defaults ADD COLUMN numeric_to_int INT DEFAULT NULL::numeric(10,1)",
        "ALTER TABLE typed_null_defaults ADD COLUMN uuid_to_text TEXT DEFAULT NULL::uuid",
    ] {
        let Command::AddColumn(add) = parse_relational(sql) else {
            panic!("expected ADD COLUMN");
        };
        assert!(matches!(
            add.column.default,
            Some(ColumnDefault::DeferredScalar {
                value: SqlValue::Null,
                ..
            })
        ));
    }
    for sql in [
        "CREATE TABLE invalid_typed_null (id INT DEFAULT NULL::text)",
        "ALTER TABLE typed_null_defaults ADD COLUMN rejected INT DEFAULT NULL::text",
    ] {
        assert!(matches!(
            parse_relational_command(sql, false),
            Some(Ok(Command::CreateTable(_) | Command::AddColumn(_)))
        ));
    }
}
