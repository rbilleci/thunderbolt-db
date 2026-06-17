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

pub(crate) fn test_wal_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "gpu-db-engine-{name}-{}-{}.segment",
        std::process::id(),
        NEXT_TEST_WAL_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ))
}
