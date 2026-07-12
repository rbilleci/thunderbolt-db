use gpu_db_engine::Engine;
use gpu_db_execution::DeviceTarget;
use gpu_db_sql::{parse_command, Command, SelectFunction, SqlValue};

#[test]
fn production_constructor_enables_strata_auto_admission_by_default() {
    assert!(Engine::new_local().auto_admit_on_commit_enabled());
}

#[test]
fn production_relational_select_never_falls_back_to_the_host_executor() {
    let engine = Engine::new_local();
    engine.set_auto_admit_on_commit(false);
    let mut engine = engine;
    engine.set_relational_residency_budget_bytes(0, 0);
    engine.execute_text(1, "CREATE TABLE t (id INT)").unwrap();
    engine.execute_text(2, "INSERT INTO t VALUES (1)").unwrap();
    let Command::Select(select) = parse_command("SELECT id FROM t").unwrap() else {
        unreachable!()
    };

    let err = engine.execute_relational_select(&select).unwrap_err();
    assert!(
        err.to_string()
            .contains("GPU execution is required for SELECT on relation \"t\""),
        "production reads must fail loud instead of executing relational work on the host: {err}"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn production_catalog_select_executes_from_a_transient_gpu_relation() {
    let engine = Engine::new_local();
    engine
        .execute_text(1, "CREATE TABLE catalog_gpu_witness (id INT)")
        .unwrap();
    let result = engine
        .execute_relational_select_text(
            "SELECT relname FROM pg_catalog.pg_class WHERE relname = 'catalog_gpu_witness'",
        )
        .unwrap();
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(
        result.rows,
        vec![vec![SqlValue::Text("catalog_gpu_witness".to_string())]]
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn production_bounded_function_result_is_materialized_by_the_gpu() {
    let engine = Engine::new_local();
    engine
        .execute_text(
            1,
            "CREATE FUNCTION public.answer() RETURNS int4 LANGUAGE sql AS 'SELECT 42'",
        )
        .unwrap();
    let result = engine
        .execute_relational_function(&SelectFunction {
            name: "answer".to_string(),
        })
        .unwrap();
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(result.fallback_reason, None);
    assert_eq!(result.rows, vec![vec![SqlValue::Int4(42)]]);
}
