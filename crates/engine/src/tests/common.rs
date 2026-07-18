use super::*;

/// Shared test backend: a GPU MvccExecutionBackend that mirrors the CPU reference but
/// reports a GPU target, and falls back exactly on the first-cuda-slice parity gap. Used
/// by both the MVCC-query and relational-SQL test groups.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FirstCudaSliceParityBackend;

impl MvccExecutionBackend for FirstCudaSliceParityBackend {
    fn execute(&self, query: &MvccReadQuery, rows: Vec<ResolvedMvccRow>) -> MvccBackendDispatch {
        if let Some(_gap) = first_cuda_slice_query_gap(query) {
            return MvccBackendDispatch::Fallback {
                reason: FallbackReason::GpuMvccReadParityGap,
                rows,
            };
        }

        match CpuMvccExecutionBackend.execute(query, rows) {
            MvccBackendDispatch::Executed(mut executed) => {
                executed.executed_target = DeviceTarget::Gpu(0);
                MvccBackendDispatch::Executed(executed)
            }
            MvccBackendDispatch::Fallback { .. } => {
                unreachable!("CPU reference backend must execute")
            }
        }
    }
}

pub(crate) fn assert_mvcc_query_uses_tracked_cpu_fallback(
    engine: &Engine,
    result: &MvccReadResult,
    expected_total_fallbacks: u64,
) {
    assert_eq!(result.planned_target, DeviceTarget::Gpu(0));
    assert_eq!(result.executed_target, DeviceTarget::Cpu);
    assert_eq!(
        result.fallback_reason,
        Some(FallbackReason::GpuMvccReadParityGap)
    );
    assert_eq!(
        engine
            .metrics()
            .fallback_for(FallbackReason::GpuMvccReadParityGap),
        expected_total_fallbacks
    );
}

/// Recovery now bulk-admits after WAL replay (STRATA S-F). Off-GPU, the parity-oracle CPU route
/// retains its exact index metadata; on a GPU host, the resident route is authoritative and may
/// report its device scan framing instead. Keep ordinary recovery tests hardware-independent while
/// still proving that a GPU result did not fall back.
pub(crate) fn assert_recovered_relational_access_path(
    result: &RelationalSelectResult,
    expected_cpu: RelationalAccessPath,
) {
    match result.executed_target {
        DeviceTarget::Cpu => assert_eq!(*result.access_path, expected_cpu),
        DeviceTarget::Gpu(_) => {
            assert!(
                matches!(
                    result.access_path.as_ref(),
                    RelationalAccessPath::FullTableScan
                ) || *result.access_path == expected_cpu
            );
            assert_eq!(result.fallback_reason, None);
        }
    }
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
    // A non-authoritative test generation was built from the already-complete host oracle, so
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
