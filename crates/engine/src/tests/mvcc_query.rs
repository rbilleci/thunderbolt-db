use super::*;

#[test]
fn generic_kv_mvcc_query_fails_before_source_resolution_or_gpu_telemetry() {
    let e = Engine::new_local_test_engine();
    e.execute_text(1, "SET acct:1=open").unwrap();
    let before = e.metrics().snapshot();

    let error = e
        .execute_mvcc_query(&MvccReadQuery {
            source: MvccReadSource::FollowValueChain {
                keys: vec!["missing:seed".to_string()],
                plan: MvccValueChainPlan {
                    value_key_hops: 7,
                    terminal: MvccValueChainTerminal::CurrentRow,
                },
                provenance: MvccSourceProvenance::Seed,
            },
            visibility: StorageVisibility { read_txn_id: 1 },
            filter: Some(MvccReadFilter::ValueEquals("never".to_string())),
            order: Some(MvccReadOrder::ValueDesc),
            projection: MvccProjection::ValueOnly,
            limit: Some(1),
        })
        .unwrap_err();

    assert_gpu_mvcc_execution_required(&e, error);
    let after = e.metrics().snapshot();
    assert_eq!(after.h2d_bytes_total, before.h2d_bytes_total);
    assert_eq!(after.d2h_bytes_total, before.d2h_bytes_total);
    assert_eq!(after.kernel_exec_samples, before.kernel_exec_samples);
    assert_eq!(after.fallback_total, before.fallback_total);
}

#[test]
fn mvcc_benchmark_report_summarizes_gpu_only_coverage() {
    let result = || MvccReadResult {
        planned_target: DeviceTarget::Gpu(0),
        executed_target: DeviceTarget::Gpu(0),
        fallback_reason: None,
        rows: Vec::new(),
    };
    let results = vec![result(), result(), result()];
    let metrics = RuntimeMetrics::default();
    metrics.observe_h2d_bytes(128);
    metrics.observe_d2h_bytes(64);
    metrics.observe_kernel_exec_ms(7);
    metrics.observe_batch_wait_ms(3);
    let report = MvccBenchmarkReport::from_results(&results, &metrics.snapshot());

    assert_eq!(report.workload_count, 3);
    assert_eq!(report.gpu_executed_count, 3);
    assert_eq!(report.cpu_fallback_count, 0);
    assert_eq!(report.gpu_executed_permyriad, 10_000);
    assert_eq!(report.cpu_fallback_permyriad, 0);
    assert_eq!(report.h2d_bytes_total, 128);
    assert_eq!(report.d2h_bytes_total, 64);
    assert_eq!(report.kernel_exec_samples, 1);
    assert_eq!(report.kernel_exec_total_ms, 7);
    assert_eq!(report.batch_wait_samples, 1);
    assert_eq!(report.batch_wait_total_ms, 3);
}
