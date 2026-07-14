//! Test-only GROUP BY kernel benchmark facades. Production query execution is intentionally
//! outside this module.

use crate::rel_exec_helpers::relational_column_index;
use crate::relational_model::resident_device_int4_column_offset;
use crate::{Engine, ExecuteError};
use gpu_db_execution::{CudaGroupByInput, GroupByI32Row};
use gpu_db_types::EngineError;

impl Engine {
    /// Benchmark helper (tests only): run GROUP BY on a resident table with the SINGLE-LEVEL or
    /// TWO-LEVEL kernel selected explicitly, over a full-table scan. Returns the result rows so the
    /// caller can confirm both kernels agree; the caller times repeated calls. Not on the query path.
    #[cfg(test)]
    pub(crate) fn group_by_i32_bench(
        &self,
        table_name: &str,
        key_col: &str,
        value_col: &str,
        two_level: bool,
    ) -> Result<Vec<GroupByI32Row>, ExecuteError> {
        let table = self.relational_catalog_table(table_name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!("no table {table_name}")))
        })?;
        let snapshot = self
            .relational_residency_snapshot_ref(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no resident snapshot".to_string()))
            })?;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no device memory".to_string()))
            })?;
        let key_idx = relational_column_index(&table, key_col)?;
        let val_idx = relational_column_index(&table, value_col)?;
        let key_off = resident_device_int4_column_offset(&snapshot, &table, key_idx)?;
        let val_off = resident_device_int4_column_offset(&snapshot, &table, val_idx)?;
        let n = u32::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed("rows exceed u32".to_string()))
        })?;
        let indices: Vec<u32> = (0..n).collect();
        device_memory
            .group_by_i32_count_sum_bench(
                CudaGroupByInput::resident_i32(key_off, val_off, u64::from(n)),
                &indices,
                two_level,
            )
            .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))
    }

    /// Benchmark helper (tests only): time JUST the GROUP BY kernel (CUDA events, min of `runs`),
    /// isolating it from the alloc/H2D/D2H/host-compact overhead. Returns the min kernel milliseconds.
    #[cfg(test)]
    // Benchmark-only facade: explicit dimensions keep call sites auditable without creating a
    // product/runtime parameter object for a test-only kernel probe.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn group_by_i32_bench_kernel_ms(
        &self,
        table_name: &str,
        key_col: &str,
        value_col: &str,
        two_level: bool,
        runs: u32,
        rows_limit: usize,
        // Query-aware aggregate-selection mask (`grouped_agg_mask`): `COUNT` times the pruned path,
        // `ALL` the full-compute path, isolating the per-row atomic savings.
        agg_mask: u32,
    ) -> Result<f32, ExecuteError> {
        let table = self.relational_catalog_table(table_name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!("no table {table_name}")))
        })?;
        let snapshot = self
            .relational_residency_snapshot_ref(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no resident snapshot".to_string()))
            })?;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(table_name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed("no device memory".to_string()))
            })?;
        let key_idx = relational_column_index(&table, key_col)?;
        let val_idx = relational_column_index(&table, value_col)?;
        let key_off = resident_device_int4_column_offset(&snapshot, &table, key_idx)?;
        let val_off = resident_device_int4_column_offset(&snapshot, &table, val_idx)?;
        let n = u32::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed("rows exceed u32".to_string()))
        })?;
        // rows_limit == 0 means the whole table; otherwise group only the first `rows_limit` rows (to
        // sweep data size at fixed cardinality from one residency).
        let m = if rows_limit == 0 {
            n
        } else {
            (rows_limit as u32).min(n)
        };
        let indices: Vec<u32> = (0..m).collect();
        device_memory
            .group_by_i32_count_sum_kernel_timed(
                CudaGroupByInput::resident_i32(key_off, val_off, u64::from(n)),
                &indices,
                two_level,
                runs,
                agg_mask,
            )
            .map(|(_, ms)| ms)
            .map_err(|e| ExecuteError::Engine(EngineError::ApplyFailed(e.to_string())))
    }
}
