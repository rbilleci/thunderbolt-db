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

