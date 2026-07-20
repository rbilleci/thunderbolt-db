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
