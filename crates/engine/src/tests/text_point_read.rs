use super::*;

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn text_only_point_projection_runs_on_gpu_without_numeric_projection_slots() {
    let mut e = Engine::new_local_cpu_oracle();
    e.execute_text(1, "CREATE TABLE labels (k INT, label TEXT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO labels (k, label) VALUES (7, 'first'), (8, 'skip'), (7, ''), (7, 'last'), (0, 'real-zero'), (NULL, 'null-key')",
    )
    .unwrap();
    e.populate_relational_residency_snapshot("labels").unwrap();

    let Command::Select(present) = parse_command("SELECT label FROM labels WHERE k = 7").unwrap()
    else {
        unreachable!()
    };
    let route = e.plan_relational_resident_route(&present);
    assert!(
        route.accepted,
        "TEXT-only point route must be GPU-eligible: {route:?}"
    );
    assert_eq!(route.query_shape, "int4_equality_mixed_column_projection");

    let result = e.execute_relational_select(&present).unwrap();
    assert_eq!(result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        result.rows,
        vec![
            vec![SqlValue::Text("first".to_string())],
            vec![SqlValue::Text(String::new())],
            vec![SqlValue::Text("last".to_string())],
        ]
    );

    let Command::Select(absent) = parse_command("SELECT label FROM labels WHERE k = 999").unwrap()
    else {
        unreachable!()
    };
    let absent_result = e.execute_relational_select(&absent).unwrap();
    assert_eq!(absent_result.executed_target, DeviceTarget::Gpu(0));
    assert!(absent_result.rows.is_empty());

    let Command::Select(zero) = parse_command("SELECT label FROM labels WHERE k = 0").unwrap()
    else {
        unreachable!()
    };
    let zero_result = e.execute_relational_select(&zero).unwrap();
    assert_eq!(zero_result.executed_target, DeviceTarget::Gpu(0));
    assert_eq!(
        zero_result.rows,
        vec![vec![SqlValue::Text("real-zero".to_string())]],
        "the nullable int4 filter validity bitmap must exclude the NULL placeholder row"
    );

    let present_job = e.prepare_relational_retained_read_job(&present).unwrap();
    let zero_job = e.prepare_relational_retained_read_job(&zero).unwrap();
    let submission = e
        .submit_relational_retained_read_jobs_with_resident_device_memory_probe(&[
            present_job,
            zero_job,
        ])
        .unwrap();
    let batched = e
        .complete_relational_retained_read_submission(submission)
        .unwrap();
    assert_eq!(batched.len(), 2);
    assert_eq!(batched[0].executed_target, DeviceTarget::Gpu(0));
    assert_eq!(batched[0].rows, result.rows);
    assert_eq!(batched[1].executed_target, DeviceTarget::Gpu(0));
    assert_eq!(batched[1].rows, zero_result.rows);
}
