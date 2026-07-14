//! Shared GPU COUNT(DISTINCT) sort, mark, and grouped reduction.

use crate::relational_model::{
    resident_device_int4_column_offset, resident_device_int8_column_offset,
    resident_device_numeric_column_offset, resident_device_text_column_layout,
    RelationalResidencySnapshot, RelationalTable,
};
use crate::ExecuteError;
use gpu_db_execution::{CudaResidentDeviceMemory, GroupByI32Row};
use gpu_db_sql::SqlType;
use gpu_db_types::EngineError;

// COUNT(DISTINCT v) over a (group key, value) tuple: GPU-sort (g, value) ASC + mark the first
// row of each distinct tuple -> per-group SUM of the new-distinct flags = the per-group distinct
// count. Shared by the grouped branch (g = the real group key) AND the scalar form below (g = a
// constant 0 -> ONE group whose count = the total distinct). A FIXED-WIDTH value packs into an
// i64 multikey matrix (int = k2 (g, v); numeric/uuid = k3 (g, v_hi, v_lo)); a varlen TEXT value
// routes through the hetero sort + the text-aware mark (reads the value text on-device). The
// sort/mark/SUM run ENTIRELY on the GPU; the host marshals only the control-plane g / index
// arrays. `g_vals`/`idx_u64` are parallel, length n (the surviving positions). Empty -> caller
// guards n == 0.
pub(super) fn count_distinct_groups(
    value_idx: usize,
    g_vals: &[i64],
    idx_u64: &[u64],
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
    device_memory: &CudaResidentDeviceMemory,
    row_count: u64,
) -> Result<Vec<GroupByI32Row>, ExecuteError> {
    let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
    };
    let n = idx_u64.len();
    let value_ty = table.columns[value_idx].ty;
    let (g_sorted, new_distinct) = match value_ty {
        SqlType::Int2
        | SqlType::Int4
        | SqlType::Int8
        | SqlType::Date
        | SqlType::Timestamp
        | SqlType::Numeric { .. }
        | SqlType::Uuid => {
            let (matrix, k) = if matches!(value_ty, SqlType::Numeric { .. } | SqlType::Uuid) {
                let off = resident_device_numeric_column_offset(snapshot, table, value_idx)?;
                let v128 = device_memory
                    .project_i128_rows_from_payload(off, idx_u64)
                    .map_err(map_err)?;
                let mut m = Vec::with_capacity(n * 3);
                for i in 0..n {
                    m.push(g_vals[i]);
                    m.push((v128[i] >> 64) as i64);
                    m.push((v128[i] as u64) as i64);
                }
                (m, 3usize)
            } else {
                let v_vals: Vec<i64> = match value_ty {
                    SqlType::Int8 | SqlType::Timestamp => device_memory
                        .project_i64_rows_from_payload(
                            resident_device_int8_column_offset(snapshot, table, value_idx)?,
                            idx_u64,
                        )
                        .map_err(map_err)?,
                    _ => device_memory
                        .project_i32_rows_from_payload(
                            resident_device_int4_column_offset(snapshot, table, value_idx)?,
                            idx_u64,
                        )
                        .map_err(map_err)?
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                };
                let mut m = Vec::with_capacity(n * 2);
                for i in 0..n {
                    m.push(g_vals[i]);
                    m.push(v_vals[i]);
                }
                (m, 2usize)
            };
            let perm = device_memory
                .bitonic_sort_multikey(&matrix, n, k, 0)
                .map_err(map_err)?;
            device_memory
                .mark_new_distinct_device(&matrix, &perm, n as u64, k)
                .map_err(map_err)?
        }
        SqlType::Text => {
            let layout = resident_device_text_column_layout(snapshot, table, value_idx)?;
            let key_plan: Vec<u32> = vec![0, 0x4000_0000_u32];
            let perm = device_memory
                .bitonic_sort_hetero(
                    idx_u64,
                    g_vals,
                    1,
                    &[(layout.offsets_byte_offset, layout.bytes_byte_offset)],
                    &[],
                    &key_plan,
                    0,
                    // COUNT(DISTINCT) (g, text_v) reps: both keys are non-NULL by construction, so
                    // there is no NULL bitmap and the nulls_first mask is irrelevant.
                    &[],
                    0,
                )
                .map_err(map_err)?;
            device_memory
                .mark_new_distinct_text_device(
                    &perm,
                    idx_u64,
                    g_vals,
                    gpu_db_execution::CudaGroupTextSource {
                        offsets_byte_offset: layout.offsets_byte_offset,
                        bytes_byte_offset: layout.bytes_byte_offset,
                        bytes_len: layout.bytes_len,
                        row_count,
                    },
                    n as u64,
                )
                .map_err(map_err)?
        }
        _ => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "COUNT(DISTINCT) value column must be int/numeric/uuid/text on the \
             GPU path"
                    .to_string(),
            )));
        }
    };
    // SUM(new_distinct) grouped by g_sorted = the per-group distinct count. indices = 0..n (the
    // sorted positions); the kernel reads g_sorted / new_distinct (both i64) via the
    // key/value_base_override. The leases live across this synced call.
    let scan_indices: Vec<u32> = (0..n as u32).collect();
    let cd_groups = device_memory
        .group_by_i32_count_sum_minmax_from_payload(
            gpu_db_execution::CudaGroupByInput {
                key: gpu_db_execution::CudaGroupKeySource::Fixed(
                    gpu_db_execution::CudaGroupFixedSource::Derived {
                        buffer: g_sorted.group_view(),
                        width: 8,
                        row_count: n as u64,
                    },
                ),
                value: gpu_db_execution::CudaGroupValueSource::Fixed(
                    gpu_db_execution::CudaGroupFixedSource::Derived {
                        buffer: new_distinct.group_view(),
                        width: 8,
                        row_count: n as u64,
                    },
                ),
                key_validity_bitmap_offset: None,
                value_validity_bitmap_offset: None,
            },
            &scan_indices,
            // COUNT(DISTINCT) SUM-of-new-distinct pass: the result builder reads this pass's
            // `.sum`, so it must compute SUM (and COUNT). ALL is correct + behavior-preserving.
            gpu_db_execution::grouped_agg_mask::ALL,
        )
        .map_err(map_err)?;
    drop((g_sorted, new_distinct));
    Ok(cd_groups)
}
