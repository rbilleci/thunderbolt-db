use super::*;

/// Shared actual-CUDA backend for MVCC-query and relational-SQL tests.
///
/// Its consumers are CUDA-gated and keep closed-form result assertions. This backend must never
/// relabel CPU execution as GPU work.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CudaDriverMvccBackend;

impl MvccExecutionBackend for CudaDriverMvccBackend {
    fn execute(&self, query: &MvccReadQuery, rows: Vec<ResolvedMvccRow>) -> MvccBackendDispatch {
        let runtime =
            CudaDriverRuntime::probe().unwrap_or_else(|_| CudaDriverRuntime::unavailable());
        CudaMvccExecutionBackend::new(runtime, 0).execute(query, rows)
    }
}

pub(crate) fn assert_gpu_mvcc_execution_required(engine: &Engine, error: ExecuteError) {
    assert!(
        error
            .to_string()
            .contains("GPU execution is required for MVCC reads"),
        "unexpected error: {error}"
    );
    assert_eq!(engine.metrics().snapshot().fallback_total, 0);
}

pub(crate) fn assert_gpu_relational_execution_required(
    engine: &Engine,
    error: ExecuteError,
    table: &str,
    fallback_before: u64,
) {
    assert!(
        error.to_string().contains(&format!(
            "GPU execution is required for SELECT on relation \"{table}\""
        )),
        "unexpected error: {error}"
    );
    assert_eq!(
        engine.metrics().snapshot().fallback_total,
        fallback_before,
        "a loud GPU decline must not manufacture fallback telemetry"
    );
}

/// Recovery now bulk-admits after WAL replay (STRATA S-F). A successful relational read must be GPU
/// executed without fallback; the resident route may report either its device scan framing or the
/// independently expected index metadata.
pub(crate) fn assert_recovered_relational_access_path(
    result: &RelationalSelectResult,
    expected: RelationalAccessPath,
) {
    assert!(matches!(result.planned_target, DeviceTarget::Gpu(_)));
    assert!(matches!(result.executed_target, DeviceTarget::Gpu(_)));
    assert!(
        matches!(
            result.access_path.as_ref(),
            RelationalAccessPath::FullTableScan
        ) || *result.access_path == expected
    );
    assert_eq!(result.fallback_reason, None);
}

pub(crate) fn test_wal_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "gpu-db-engine-{name}-{}-{}.segment",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Remove a mandatory DML generation when a residency-control test needs to construct a synthetic
/// cold/absent starting state. This is test-state setup only; no relational statement executes
/// against the absent generation.
pub(crate) fn forget_test_relational_residency(engine: &Engine, table: &str) {
    // R3-004: dropping a mandatory generation without first preserving its rows would destroy the
    // sole relational data copy. Residency-control tests that need an artificial cold state cross
    // the explicit RETIRE-002 repair boundary first; production has no corresponding cold fallback.
    repair_test_relational_host_copy(engine, table);
    let catalog = engine.ddl_catalog();
    catalog.relational_resident_cache.remove_table(
        table,
        &engine.read_state.residency,
        &engine.read_state.route_telemetry,
    );
}

pub(crate) fn repair_test_relational_host_copy(engine: &Engine, table: &str) {
    // A non-authoritative test generation already has a complete host tuple-store image, so
    // there is nothing to reverse-gather. This also covers the legacy single-buffer layouts used
    // by read-kernel tests: the production repair gather intentionally accepts shard generations
    // only.
    if !engine.table_device_authoritative(table) {
        return;
    }
    let catalog_table = engine
        .relational_catalog_table(table)
        .expect("test repair table must exist");
    let boundary = engine.committed_seq();
    engine
        .rehydrate_elided_table(
            &catalog_table,
            boundary,
            &Default::default(),
            &Default::default(),
            boundary,
        )
        .expect("test repair must gather the live device generation");
}

/// Rebuild a device-current table into the legacy single-buffer layout for tests whose subject is
/// that read layout. Normal DML remains shard-authoritative; this explicitly crosses the existing
/// test/repair boundary and must not be used to model a production write path.
pub(crate) fn install_test_single_buffer_residency(
    engine: &mut Engine,
    table: &str,
) -> RelationalResidencySnapshot {
    repair_test_relational_host_copy(engine, table);
    invalidate_test_relational_residency(engine, table);
    engine.set_shard_residency_enabled(false);
    engine
        .populate_relational_residency_snapshot(table)
        .expect("single-buffer test admission must succeed")
}

/// Publish an explicit invalid descriptor for route-planning tests. Production DML now maintains
/// its device generation, so invalidation scenarios must be injected rather than inferred from a
/// successful mutation.
pub(crate) fn invalidate_test_relational_residency(engine: &Engine, table: &str) {
    let current = engine.committed_seq();
    engine.invalidate_relational_residency_tables_concurrent(
        &BTreeSet::from([table.to_string()]),
        current,
        current,
    );
}
