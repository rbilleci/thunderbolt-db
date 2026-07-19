//! Typed result-column decoding for streaming LAG and LEAD windows.

use super::*;

impl Engine {
    pub(crate) fn decode_materialized_column(
        &self,
        run: &gpu_db_execution::CudaMaterializedRelation,
        column_index: usize,
        ty: SqlType,
        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32,
    ) -> Result<Vec<SqlValue>, ExecuteError> {
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        run.columns().get(column_index).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "materialized column index is outside the run schema".to_string(),
            ))
        })?;
        let specs = Engine::materialized_join_run_specs(run);
        let terminal = run
            .memory()
            .materialize_join_coordinates(coordinates, &specs[column_index..=column_index])
            .map_err(map_err)?;
        let frame = terminal.read_result_frame().map_err(map_err)?;
        self.decode_materialized_result_column_frame(&frame, ty)
    }
}
