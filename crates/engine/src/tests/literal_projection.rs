use super::*;

/// Non-vacuity sabotage for the no-FROM literal route. An impossible device ordinal must fail the
/// transient GPU relation before it can return a row; a host-fabricated scalar result would make
/// this test succeed and therefore fail the assertion.
#[test]
fn literal_projection_invalid_gpu_sabotage_fails_closed() {
    let engine = Engine::with_planner_config(PlannerConfig {
        default_gpu_id: u16::MAX,
    });
    let literal = SelectLiteral {
        column_name: "missing".to_string(),
        ty: SqlType::Int4,
        value: SqlValue::Null,
        add_int4: None,
    };
    let error = engine
        .execute_relational_literal(&literal)
        .expect_err("an unavailable GPU must not produce a fabricated NULL row");
    let message = error.to_string().to_ascii_lowercase();
    assert!(
        message.contains("gpu") || message.contains("device") || message.contains("cuda"),
        "sabotaged literal route must report a device failure: {error}"
    );
}

/// A prepared overflow must reach the device route.  A host-side checked add (or a host-fabricated
/// completed scalar) would instead return `integer out of range` before this impossible GPU can
/// reject the launch.
#[test]
fn prepared_int4_addition_invalid_gpu_sabotage_fails_closed_before_overflow_result() {
    let engine = Engine::with_planner_config(PlannerConfig {
        default_gpu_id: u16::MAX,
    });
    let prepared = PreparedCommand::parse("SELECT $1 + 1 AS plus_one").unwrap();
    let bound = prepared.bind(&[SqlValue::Int4(i32::MAX)]).unwrap();
    let Command::SelectLiteral(literal) = bound.command() else {
        panic!("bounded prepared scalar must retain its typed literal route");
    };
    let error = engine
        .execute_relational_literal(literal)
        .expect_err("sabotaged device arithmetic must fail closed");
    assert!(!error.is_numeric_value_out_of_range());
    let message = error.to_string().to_ascii_lowercase();
    assert!(
        message.contains("gpu") || message.contains("device") || message.contains("cuda"),
        "prepared arithmetic must attempt the device route before exposing a result: {error}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn prepared_int4_addition_runs_checked_on_device_and_surfaces_range_error() {
    let engine = Engine::new_local_test_engine();
    let prepared = PreparedCommand::parse("SELECT $1 + 10 AS plus_ten").unwrap();

    for (input, expected) in [(5, 15), (-7, 3)] {
        let bound = prepared.bind(&[SqlValue::Int4(input)]).unwrap();
        let Command::SelectLiteral(literal) = bound.command() else {
            panic!("bounded prepared scalar must retain its typed literal route");
        };
        assert_eq!(literal.add_int4, Some(10));
        let result = engine.execute_relational_literal(literal).unwrap();
        assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
        assert_eq!(result.rows, vec![vec![SqlValue::Int4(expected)]]);
    }

    let zero = PreparedCommand::parse("SELECT $1 + 0 AS unchanged").unwrap();
    let bound = zero.bind(&[SqlValue::Int4(i32::MAX)]).unwrap();
    let Command::SelectLiteral(literal) = bound.command() else {
        panic!("bounded prepared scalar must retain its typed literal route");
    };
    let result = engine.execute_relational_literal(literal).unwrap();
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(i32::MAX)]]);

    let negative = PreparedCommand::parse("SELECT $1 + -1 AS minus_one").unwrap();
    let bound = negative.bind(&[SqlValue::Int4(0)]).unwrap();
    let Command::SelectLiteral(literal) = bound.command() else {
        panic!("bounded prepared scalar must retain its typed literal route");
    };
    let result = engine.execute_relational_literal(literal).unwrap();
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(-1)]]);

    let null = prepared.bind(&[SqlValue::Null]).unwrap();
    let Command::SelectLiteral(literal) = null.command() else {
        panic!("bounded prepared scalar must retain its typed literal route");
    };
    let result = engine.execute_relational_literal(literal).unwrap();
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.rows, vec![vec![SqlValue::Null]]);

    for (overflowing, input) in [
        (
            PreparedCommand::parse("SELECT $1 + 1 AS plus_one").unwrap(),
            i32::MAX,
        ),
        (
            PreparedCommand::parse("SELECT $1 + -1 AS minus_one").unwrap(),
            i32::MIN,
        ),
    ] {
        // Binding carries the operand and VM operator; range validation is an Execute-time GPU
        // effect, not a host bind-time fold.
        let bound = overflowing.bind(&[SqlValue::Int4(input)]).unwrap();
        let Command::SelectLiteral(literal) = bound.command() else {
            panic!("bounded prepared scalar must retain its typed literal route");
        };
        let error = engine
            .execute_relational_literal(literal)
            .expect_err("device checked arithmetic must reject int4 overflow");
        assert!(error.is_numeric_value_out_of_range());
        assert!(error.to_string().contains("integer out of range"));
    }
}
