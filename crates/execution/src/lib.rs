#[cfg(test)]
use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::Arc;

#[cfg(test)]
use libloading::Library;
pub mod probe;
pub use probe::Probe;
mod reference_operators;
pub use reference_operators::{
    CpuNoop, FilterOperator, LimitOperator, Operator, ProjectOperator, ScanOperator, SortOperator,
    VecOperator,
};
mod mvcc_batch;
pub use mvcc_batch::CudaMvccRowBatch;
mod routing;
pub use routing::{
    DeviceRouter, DeviceTarget, GpuFallbackReason, GpuRuntime, GpuRuntimeSnapshot, MockGpuRuntime,
    PlannedOp, RouteDecision,
};
mod runtime_contract;
pub use runtime_contract::{
    CudaDeviceMemoryProof, CudaDeviceSnapshot, CudaRuntimeProbeError, CudaRuntimeSnapshot,
};

mod cuda_context;
use cuda_context::{
    check_cuda, GpuPrimaryContext, PinnedHostLease, PooledBufferLease, PooledDeviceBufferOwned,
    PooledStream, PooledStreamOwned, POOLED_STREAM_SCRATCH_BYTES,
};
#[cfg(test)]
use cuda_context::{
    gpu_primary_context, CudaDeviceAllocationGuard, CudaEventGuard, CudaModuleGuard,
};
pub use cuda_context::{
    CudaAllocationScope, CudaExternalAllocationReservation, PendingCudaResidentDeviceCopy,
};
mod cuda_driver;
use cuda_driver::launch_cuda_resident_device_memory;
pub use cuda_driver::CudaDriverRuntime;
mod staged_filter;
mod staged_mvcc;

mod resident_memory;
use resident_memory::CudaResidentReadSource;
pub use resident_memory::{
    CudaDeviceMemoryChunk, CudaOwnedDeviceMemoryChunk, CudaResidentDeviceMemory,
    CudaResidentDeviceMemoryReadView, RecompactFill, RecompactSegment,
};
mod resident_header;
use resident_header::launch_cuda_resident_row_count;
mod resident_sort;
use resident_sort::{
    launch_cuda_bitonic_sort_hetero, launch_cuda_bitonic_sort_i64,
    launch_cuda_bitonic_sort_multikey, launch_cuda_bitonic_sort_text,
    launch_cuda_order_by_sort_i64_radix,
};
mod resident_count;
use resident_count::{
    launch_cuda_resident_i32_compare_count, launch_cuda_resident_i32_equal_count,
};
#[cfg(test)]
use resident_count::{
    launch_cuda_resident_i32_compare_count_serial, launch_cuda_resident_i32_equal_count_serial,
};
mod resident_window;
pub use resident_window::CudaGroupTextSource;
mod resident_compare_ordered;
use resident_compare_ordered::{
    launch_cuda_buffer_i32_compare_indices_ordered,
    launch_cuda_resident_i32_compare_buffers_indices_ordered,
    launch_cuda_resident_i32_compare_indices_ordered,
    launch_cuda_resident_i32_compare_project, validate_ordered_i32_comparison,
};
#[cfg(test)]
use resident_compare_ordered::{
    COMPARE_ORDERED_PTX, i32_bits_into_u32, validate_ordered_i32_context_identity,
    validate_ordered_i32_index_domain, validate_ordered_i32_input_window,
};
mod resident_text;
mod resident_scalar;
pub use resident_scalar::CudaI32Stats;
mod expression_vm;
use expression_vm::{ExprTerminal, run_resident_arith_program};
pub use expression_vm::{ExprStep, ResidentElemType};
mod derived_column;
use derived_column::{
    launch_cuda_arith_value_column_device, launch_cuda_bool_to_int4_column_device,
    launch_cuda_build_wide_key_device, launch_cuda_mark_new_distinct_device,
    launch_cuda_mark_new_distinct_text_device, launch_cuda_pack_two_cols_i128_device,
    launch_cuda_pack_two_int4_cols_device, launch_cuda_upload_u64_device,
    launch_cuda_widen_col_to_i64_device,
};
pub use derived_column::{
    CudaGroupDeviceView, CudaWideKeyDescriptor, CudaWideKeySource, CudaWideKeyValidity,
    DeviceArithBuffer,
};
mod predicate_mask;
use predicate_mask::compact_mask_i32_to_indices;
pub use predicate_mask::CudaPredicateMaskI32;
mod resident_gather;
use resident_gather::{
    copy_cuda_resident_bool_rows, copy_cuda_resident_i128_rows, copy_cuda_resident_i32_rows,
    copy_cuda_resident_i64_rows,
};
mod resident_filter;
use resident_filter::{
    launch_cuda_resident_bool_to_mask_filter, launch_cuda_resident_i128_compare_columns_filter,
    launch_cuda_resident_i128_compare_scalar_filter,
    launch_cuda_resident_i64_compare_columns_filter, launch_cuda_resident_i64_compare_scalar_filter,
    launch_cuda_resident_text_compare_scalar_filter, launch_cuda_resident_text_eq_scalar_filter,
    launch_cuda_resident_text_like_scalar_filter, launch_cuda_resident_uuid_compare_columns_filter,
    launch_cuda_resident_uuid_compare_scalar_filter,
};
mod resident_aggregate;
use resident_aggregate::{
    launch_cuda_resident_i128_minmax_partials_at_indices,
    launch_cuda_resident_i128_sum_partials_at_indices,
    launch_cuda_resident_i32_minmax_at_indices,
    launch_cuda_resident_i32_sum_at_indices,
    launch_cuda_resident_i64_minmax_at_indices,
    launch_cuda_resident_i64_sum_at_indices_i128,
};
mod resident_group;
use resident_group::{launch_cuda_group_by_i32_count_sum, launch_cuda_group_by_kernel_timed};
pub use resident_group::GroupByI32Row;
mod group_input;
use group_input::{ValidatedGroupInput, validate_group_input};
pub use group_input::{
    CudaGroupByInput, CudaGroupFixedSource, CudaGroupKeySource, CudaGroupTextDescriptorBuffer,
    CudaGroupTextDescriptors, CudaGroupValueSource, CudaGroupWideSource,
};
mod join_contract;
pub use join_contract::{CudaJoinCoordinatesU32, CudaJoinOrderKey, CudaJoinPayloadKey};
mod join_filter;
mod join_fixed;
mod join_materialize;
pub use join_materialize::{
    CudaMaterializeJoinColumn, CudaMaterializedColumnKind, CudaMaterializedColumnLayout,
    CudaMaterializedRelation,
};
mod join_outer;
pub use join_outer::CudaMatchBitmapU32;
mod join_projection;
mod join_sort;
mod join_window;
pub use join_window::{CudaWindowRankKind, CudaWindowRanksU64};
mod staged_hash_join;
pub use staged_hash_join::HashJoinOutcome;
mod write_locate;
pub use write_locate::{
    VisibleLocateResult, VisibleLocateShard, WriteLocateResult, WriteLocateShard,
};
mod write_apply;
pub use write_apply::{
    CudaCompoundFoldColumn, CudaWriteDestination, CudaWriteIndex, FusedApplyRequest,
};
mod resident_sidecar;
pub use resident_sidecar::{CudaSidecarSource, CudaTextOffsetSource};
mod unique_coordinate;
use unique_coordinate::launch_cuda_unique_coordinate_threshold;
mod point_read_submit;
use point_read_submit::{
    launch_cuda_resident_i32_equal_project, submit_cuda_resident_i32_equal_any_project,
    submit_cuda_resident_i32_index_probe,
};
mod point_read_dense;
use point_read_dense::{
    submit_cuda_resident_i32_index_probe_dense,
    submit_cuda_resident_i32_multi_shard_index_probe_dense,
};
pub use point_read_dense::{CudaI32IndexProbeDenseSubmission, MultiShardProbeShard};
mod point_read_bloom;
use point_read_bloom::probe_cuda_chunk_blooms;
pub use point_read_bloom::ChunkBloomProbeShard;
mod point_read_text;
use point_read_text::launch_cuda_resident_i32_equal_any_project_text;
mod point_read_rows;
use point_read_rows::launch_cuda_resident_i32_equal_row_indices;
mod point_read_submission;
pub use point_read_submission::{
    CudaI32BatchProjectionColumns, CudaI32BatchProjectionRow,
    CudaI32EqualAnyProjectSubmission, CudaI32TextBatchProjectionRow,
};
impl CudaResidentDeviceMemoryReadView {
    pub fn submit_match_project_i32_equal_any_from_payload(
        &self,
        filter_offset: u64,
        needles: &[i32],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<CudaI32EqualAnyProjectSubmission, CudaRuntimeProbeError> {
        submit_cuda_resident_i32_equal_any_project(
            self,
            filter_offset,
            needles,
            projection_offsets,
            row_count,
        )
    }

    /// R1a — GPU-index point-lookup variant of [`Self::submit_match_project_i32_equal_any_from_payload`]:
    /// hash-probes the device-resident `index_ptr` (built over the key column on admission) instead of
    /// scanning, producing the SAME `CudaI32EqualAnyProjectSubmission` (so `complete`/`complete_detached`
    /// are unchanged). `index_table_mask` = table_size-1, `index_hash_shift` = 32 - log2(table_size).
    pub fn submit_match_project_i32_index_probe_from_payload(
        &self,
        index: &Arc<CudaResidentDeviceMemory>,
        index_table_mask: u32,
        index_hash_shift: u32,
        needles: &[i32],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<CudaI32EqualAnyProjectSubmission, CudaRuntimeProbeError> {
        submit_cuda_resident_i32_index_probe(
            self,
            index,
            index_table_mask,
            index_hash_shift,
            needles,
            projection_offsets,
            row_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn match_project_i32_equal_any_text_from_payload(
        &self,
        filter_offset: u64,
        filter_validity_bitmap_offset: Option<u64>,
        needles: &[i32],
        projection_offsets: &[u64],
        text_offsets_byte_offset: u64,
        text_bytes_byte_offset: u64,
        text_bytes_len: u64,
        text_validity_bitmap_offset: Option<u64>,
        row_count: u64,
    ) -> Result<Vec<CudaI32TextBatchProjectionRow>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_equal_any_project_text(
            self,
            filter_offset,
            filter_validity_bitmap_offset,
            needles,
            projection_offsets,
            text_offsets_byte_offset,
            text_bytes_byte_offset,
            text_bytes_len,
            text_validity_bitmap_offset,
            row_count,
        )
    }
}

fn launch_validated_group_by(
    resident: &CudaResidentDeviceMemory,
    input: CudaGroupByInput<'_>,
    indices: &[u32],
    two_level: bool,
    agg_mask: u32,
) -> Result<Vec<GroupByI32Row>, CudaRuntimeProbeError> {
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    let ValidatedGroupInput {
        key_byte_offset, value_byte_offset, value_is_int8, key_is_int8, value_is_numeric,
        value_is_uuid, key_is_i128, key_is_text, key_offsets_off, key_bytes_off, key_bytes_len,
        value_is_text, value_offsets_off, value_bytes_off, value_bytes_len, key_base_override,
        value_base_override, comp_w, n_text, text_desc_ptr, value_null_off, key_null_off,
    } = validate_group_input(resident, input, indices)?;
    if matches!(input.value, CudaGroupValueSource::Unused { .. }) && (agg_mask & !grouped_agg_mask::COUNT) != 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(agg_mask as usize));
    }
    if two_level
        && (value_is_int8 || key_is_int8 || value_is_numeric || value_is_uuid || key_is_i128
            || key_is_text || value_is_text || key_base_override != 0 || value_base_override != 0
            || comp_w != 0 || n_text != 0 || value_null_off.is_some() || key_null_off.is_some())
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    launch_cuda_group_by_i32_count_sum(
        resident, key_byte_offset, value_byte_offset, indices,
        if two_level { c"gpu_db_group_by_i32_count_sum_twolevel" } else { c"gpu_db_group_by_i32_count_sum" },
        value_is_int8, key_is_int8, value_is_numeric, value_is_uuid, key_is_i128, key_is_text,
        key_offsets_off, key_bytes_off, key_bytes_len, value_is_text, value_offsets_off,
        value_bytes_off, value_bytes_len, key_base_override, value_base_override, comp_w, n_text,
        text_desc_ptr, value_null_off, key_null_off, agg_mask,
    )
}

impl CudaResidentDeviceMemory {
    /// Sort `keys[0..n]` on the GPU (bitonic), returning the row positions in ascending (or, with
    /// `descending`, descending) key order. The foundation of the charter-native GPU ORDER BY.
    pub fn bitonic_sort_i64(
        &self,
        keys: &[i64],
        descending: bool,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_bitonic_sort_i64(self, keys, descending)
    }

    /// Sort `keys[0..n]` on the GPU for ORDER BY, dispatching by size at the adaptive crossover:
    /// BITONIC (`bitonic_sort_i64`, O(n log^2 n)) below it, the proven resident LSD-RADIX argsort
    /// (O(n), signed->unsigned transform + direction handling) at/above it. Returns the row positions
    /// in ascending (or, with `descending`, descending) key order. Both arms produce a correct ordering
    /// (equal keys' relative order is unspecified, as for SQL ORDER BY without a tie-breaker). The radix
    /// arm uploads `keys` synchronously to a device buffer + reuses `launch_cuda_resident_i64_argsort_radix`.
    pub fn order_by_sort_i64(
        &self,
        keys: &[i64],
        descending: bool,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        if keys.len() as u64 >= ADAPTIVE_SORT_CROSSOVER_ROWS {
            launch_cuda_order_by_sort_i64_radix(self, keys, descending)
        } else {
            launch_cuda_bitonic_sort_i64(self, keys, descending)
        }
    }

    /// Sort the surviving rows `indices` on the GPU by a resident TEXT column (lexicographic, unsigned
    /// bytes, a prefix sorts smaller), returning positions into `indices` in ascending (or, with
    /// `descending`, descending) text order. `offsets_byte_offset`/`bytes_byte_offset` locate the
    /// resident text column's offsets + bytes sections. The text leg of the charter-native GPU ORDER BY.
    pub fn bitonic_sort_text(
        &self,
        indices: &[u64],
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        descending: bool,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_bitonic_sort_text(
            self,
            indices,
            offsets_byte_offset,
            bytes_byte_offset,
            descending,
        )
    }

    /// Multi-key GPU bitonic sort. `keys` is a row-major `n x k` matrix of i64 (`keys[row*k + key]`,
    /// key 0 most significant); `desc_mask` bit j set => key j sorts descending. Returns the row
    /// positions `0..n` in the multi-key order. The general multi-key ORDER BY core
    /// (`ORDER BY a ASC, b DESC, ...`) -- ties on key 0 break on key 1, then key 2, ...
    pub fn bitonic_sort_multikey(
        &self,
        keys: &[i64],
        n: usize,
        k: usize,
        desc_mask: u64,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_bitonic_sort_multikey(self, keys, n, k, desc_mask)
    }

    /// GPU heterogeneous multi-key bitonic sort over a tuple MIXING int + text keys. `indices` =
    /// the surviving row ids (n); `int_keys` = a row-major `n x num_int` i64 matrix by position (the
    /// int keys only); `text_cols[t]` = (offsets_byte_offset, bytes_byte_offset) of text key t;
    /// `key_plan[k]` selects key k (bit31=is_text, low bits=idx into the int columns / text_cols);
    /// `desc_mask` bit k => DESC. Returns the positions 0..n in the tuple order. Completes the canonical
    /// `ORDER BY <text>, <int>, <int>` on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn bitonic_sort_hetero(
        &self,
        indices: &[u64],
        int_keys: &[i64],
        num_int: usize,
        text_cols: &[(u64, u64)],
        b128_cols: &[u64],
        key_plan: &[u32],
        desc_mask: u64,
        null_offs: &[u64],
        nulls_first_mask: u64,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_bitonic_sort_hetero(
            self,
            indices,
            int_keys,
            num_int,
            text_cols,
            b128_cols,
            key_plan,
            desc_mask,
            null_offs,
            nulls_first_mask,
            None,
        )
    }

    /// As [`Self::bitonic_sort_hetero`] but the TEXT/NUMERIC/UUID legs read from `payload` -- a
    /// resident-LIKE columnar buffer built (via build_relational_device_payload) from a NON-resident
    /// result, e.g. a GROUP BY result. `self` supplies only the CUDA context/stream; text_cols/b128_cols
    /// offsets index into `payload`. Lets the GPU sort a host-materialized grouped result on-device.
    #[allow(clippy::too_many_arguments)]
    pub fn bitonic_sort_hetero_on_payload(
        &self,
        payload: &[u8],
        indices: &[u64],
        int_keys: &[i64],
        num_int: usize,
        text_cols: &[(u64, u64)],
        b128_cols: &[u64],
        key_plan: &[u32],
        desc_mask: u64,
        null_offs: &[u64],
        nulls_first_mask: u64,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_bitonic_sort_hetero(
            self,
            indices,
            int_keys,
            num_int,
            text_cols,
            b128_cols,
            key_plan,
            desc_mask,
            null_offs,
            nulls_first_mask,
            Some(payload),
        )
    }

    /// Evaluate an arithmetic ORDER BY expression (`a+b`, `a*2`, ...) over all `n_rows` on the GPU via
    /// the device Expr interpreter, then GATHER the result (i32, sign-extended to i64) at the survivor
    /// `indices` -- an ORDER BY-expression key column that feeds the GPU sort like any materialized int
    /// key. Checked int4 overflow surfaces as PG `IntegerOutOfRange` (no wrap, no CPU re-execution).
    pub fn arith_value_column_at_indices(
        &self,
        program: &[ExprStep],
        n_rows: u64,
        indices: &[u32],
        elem: ResidentElemType,
    ) -> Result<Vec<i64>, CudaRuntimeProbeError> {
        launch_cuda_arith_value_column_at_indices(self, program, n_rows, indices, elem)
    }

    /// Like [`Self::arith_value_column_at_indices`] (I32 arith only) but the expression is NULLABLE: an
    /// extra `validity_program` (a mask VM program that yields 1 where every operand column is non-NULL,
    /// 0 otherwise) is run on the GPU, and the i64 sort key is `i64::MAX` (PG's default-end sentinel, which
    /// no widened int4 value can equal) wherever the expression is NULL, else the sign-extended value --
    /// blended ON-DEVICE (M3 -- doc 21). Returns the gathered i64 keys at `indices`.
    pub fn arith_value_column_at_indices_nullable(
        &self,
        program: &[ExprStep],
        validity_program: &[ExprStep],
        n_rows: u64,
        indices: &[u32],
    ) -> Result<Vec<i64>, CudaRuntimeProbeError> {
        launch_cuda_arith_value_column_at_indices_nullable(
            self,
            program,
            validity_program,
            n_rows,
            indices,
        )
    }

    /// Like [`Self::arith_value_column_at_indices`] but keeps the arith result RESIDENT: runs the
    /// program over all `n_rows`, returns the device value buffer (+ its pooled lease) instead of
    /// D2H-gathering. cuCtxSynchronize'd so a later GROUP BY launch bound through the buffer's typed
    /// `group_view()` sees valid keys, not a racing/stale buffer. The returned owner MUST outlive every
    /// launch that borrows that view.
    /// Checked int4/int8 overflow -> PG error is inherited from the arith VM.
    pub fn arith_value_column_device(
        &self,
        program: &[ExprStep],
        n_rows: u64,
        elem: ResidentElemType,
    ) -> Result<DeviceArithBuffer<'_>, CudaRuntimeProbeError> {
        launch_cuda_arith_value_column_device(self, program, n_rows, elem)
    }

    /// Materialize a BOOL column (1-bit-per-row bitmap) into a derived int4 (0/1) device column. The
    /// GROUP BY kernel reads it through a typed derived-key/value source (a bool GROUP BY key or
    /// MIN/MAX over a bool value) -- grouped/aggregated on the AUDITED int4 path, which avoids a bool
    /// GROUP BY kernel and the shared-state concurrency hazard that blocked it. Reuses
    /// `gpu_db_resident_bool_to_mask` (it already writes int4 0/1). The caller owns the returned buffer;
    /// it MUST outlive every GROUP BY launch that borrows its `group_view()`.
    pub fn bool_to_int4_column_device(
        &self,
        bitmap_byte_offset: u64,
        n_rows: u64,
    ) -> Result<DeviceArithBuffer<'_>, CudaRuntimeProbeError> {
        launch_cuda_bool_to_int4_column_device(self, bitmap_byte_offset, n_rows)
    }

    /// Pack two int4-section columns (at byte offsets `off0`, `off1`) into one i64 derived key per row
    /// (col0 in the high 32 bits, col1 in the low 32 -- bijective) and return it RESIDENT
    /// (cuCtxSynchronize'd) for a COMPOSITE GROUP BY key through its typed derived view. The executor
    /// unpacks the result slot key back into the two column values.
    pub fn pack_two_int4_cols_device(
        &self,
        off0: u64,
        off1: u64,
        n_rows: u64,
    ) -> Result<DeviceArithBuffer<'_>, CudaRuntimeProbeError> {
        launch_cuda_pack_two_int4_cols_device(self, off0, off1, n_rows)
    }

    /// Composite GROUP BY key whose combined width exceeds 64 bits (an int8/timestamp member): pack two
    /// fixed-width int columns into one i128 derived key per row (col0 = HIGH 64 bits, col1 = LOW 64).
    /// `w0`/`w1` are each member's read width in bytes (4 = int4 section, 8 = int8 section). Returns a
    /// leased [i128; n] device buffer (16 bytes/row) the caller holds alive; the b128 GROUP BY claim
    /// receives it through a typed derived-key view. cuCtxSynchronize'd so the GROUP BY launch reads
    /// the completed buffer.
    pub fn pack_two_cols_i128_device(
        &self,
        off0: u64,
        w0: u64,
        off1: u64,
        w1: u64,
        n_rows: u64,
    ) -> Result<DeviceArithBuffer<'_>, CudaRuntimeProbeError> {
        launch_cuda_pack_two_cols_i128_device(self, off0, w0, off1, w1, n_rows)
    }

    /// Widen a fixed-width int column (`w` = 4 or 8 bytes) to a per-row i64 derived buffer
    /// (sign-extended). A typed derived-key view supplies the fixed member of a composite
    /// (fixed-width, text) GROUP BY key. cuCtxSynchronize'd.
    pub fn widen_col_to_i64_device(
        &self,
        off: u64,
        w: u64,
        n_rows: u64,
    ) -> Result<DeviceArithBuffer<'_>, CudaRuntimeProbeError> {
        launch_cuda_widen_col_to_i64_device(self, off, w, n_rows)
    }

    /// Build a fixed-width WIDE KEY buffer for a general all-fixed composite GROUP BY key. Typed
    /// descriptors bind every resident/derived source to its width and CUDA context. Each destination
    /// is 8-byte aligned, in-bounds, and non-overlapping. A non-empty `validity` slice must contain one
    /// entry per descriptor (at most 64) and reserves the final u64 of each output row.
    pub fn build_wide_key_device<'a>(
        &self,
        descriptors: &[CudaWideKeyDescriptor<'a>],
        wbytes: u64,
        n_rows: u64,
        validity: &[CudaWideKeyValidity],
    ) -> Result<DeviceArithBuffer<'_>, CudaRuntimeProbeError> {
        launch_cuda_build_wide_key_device(self, descriptors, wbytes, n_rows, validity)
    }

    /// Upload a small generic u64 array to device. Grouped TEXT descriptors must instead use
    /// [`Self::upload_group_text_descriptors`], which owns their typed sources and byte limits.
    /// Blocking H2D: `data` is fully on-device when this method returns.
    pub fn upload_u64_device(
        &self,
        data: &[u64],
    ) -> Result<DeviceArithBuffer<'_>, CudaRuntimeProbeError> {
        launch_cuda_upload_u64_device(self, data)
    }

    /// Upload owned `(offsets, bytes, bytes_len)` descriptors for composite text group keys.
    pub fn upload_group_text_descriptors(
        &self,
        sources: &[CudaGroupTextSource],
    ) -> Result<CudaGroupTextDescriptorBuffer<'_>, CudaRuntimeProbeError> {
        if sources.is_empty() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
        let mut flat = Vec::with_capacity(sources.len() * 3);
        for source in sources {
            flat.extend_from_slice(&[
                source.offsets_byte_offset,
                source.bytes_byte_offset,
                source.bytes_len,
            ]);
        }
        Ok(CudaGroupTextDescriptorBuffer {
            buffer: launch_cuda_upload_u64_device(self, &flat)?,
            sources: sources.to_vec(),
        })
    }

    /// COUNT(DISTINCT v) mark pass: `keys` is the (key0, key1, ..) i64 tuple matrix (row-major, `k`
    /// values/row) ALREADY sorted via `perm` (from [`Self::bitonic_sort_multikey`]). key0 is the group
    /// key; the remaining keys are the value's fixed-width i64 representation (`k`=2 for an int value
    /// `(g, v)`, `k`=3 for a numeric/uuid value `(g, v_hi, v_lo)`). Runs `gpu_db_mark_new_distinct` and
    /// returns `(g_sorted, new_distinct)` RESIDENT (cuCtxSynchronize'd) -- `g_sorted[i]` = key0 at
    /// sorted position i, `new_distinct[i]` = 1 at each first-seen tuple else 0. SUM(new_distinct)
    /// grouped by g_sorted (via the GROUP BY kernel's key/value_base_override) = the per-group distinct
    /// count. Both typed buffer owners live in the result.
    pub fn mark_new_distinct_device(
        &self,
        keys: &[i64],
        perm: &[u32],
        n: u64,
        k: usize,
    ) -> Result<(DeviceArithBuffer<'_>, DeviceArithBuffer<'_>), CudaRuntimeProbeError> {
        launch_cuda_mark_new_distinct_device(self, keys, perm, n, k)
    }

    /// COUNT(DISTINCT v) mark pass for a TEXT value (varlen -> cannot pack into i64 keys). `perm` is
    /// the positions 0..n from [`Self::bitonic_sort_hetero`] over the `(g, text_v)` tuple; `indices`
    /// the surviving absolute resident rows (by position); `g_keys` the int group key (by position,
    /// like the hetero sort's `int_keys`). `text` owns the value column's resident windows, blob
    /// length, and logical row count. Runs `gpu_db_mark_new_distinct_text` and returns
    /// `(g_sorted, new_distinct)` RESIDENT (cuCtxSynchronize'd). SUM(new_distinct) grouped by g_sorted
    /// through typed derived-key/value views = the per-group distinct count.
    pub fn mark_new_distinct_text_device(
        &self,
        perm: &[u32],
        indices: &[u64],
        g_keys: &[i64],
        text: CudaGroupTextSource,
        n: u64,
    ) -> Result<(DeviceArithBuffer<'_>, DeviceArithBuffer<'_>), CudaRuntimeProbeError> {
        launch_cuda_mark_new_distinct_text_device(self, perm, indices, g_keys, text, n)
    }

    pub fn count_i32_equal_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        // M3 (doc 21): the filter column's NULL validity bitmap byte offset, or `None` if the column
        // has no NULLs (all rows valid). NULL rows never match the needle (three-valued logic).
        null_bitmap_offset: Option<u64>,
    ) -> Result<u64, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_equal_count(
            self,
            byte_offset,
            row_count,
            needle,
            null_bitmap_offset,
        )
    }

    pub fn count_i32_in_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needles: &[i32],
        null_bitmap_offset: Option<u64>,
    ) -> Result<u64, CudaRuntimeProbeError> {
        let mut total = 0_u64;
        for needle in needles {
            total = total
                .checked_add(launch_cuda_resident_i32_equal_count(
                    self,
                    byte_offset,
                    row_count,
                    *needle,
                    null_bitmap_offset,
                )?)
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        }
        Ok(total)
    }

    pub fn count_i32_compare_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
    ) -> Result<u64, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_compare_count(self, byte_offset, row_count, needle, comparison)
    }

    /// P5 uniqueness-verdict primitive: count candidate coordinates that are not present in the
    /// UPDATE self-exclusion set and return whether `reject_at` is reached. The host may marshal
    /// coordinates produced by preceding device predicates, but it does not filter, count, or
    /// decide the constraint outcome; the only readback is this device-computed status bit.
    pub fn unique_coordinate_threshold_reached(
        &self,
        candidates: &[u64],
        exclusions: &[u64],
        reject_at: u32,
    ) -> Result<bool, CudaRuntimeProbeError> {
        launch_cuda_unique_coordinate_threshold(self, candidates, exclusions, reject_at)
    }

    pub fn count_i32_between_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        lower_inclusive: i32,
        upper_inclusive: i32,
    ) -> Result<u64, CudaRuntimeProbeError> {
        if lower_inclusive > upper_inclusive {
            return Ok(0);
        }
        let greater_or_equal_lower_count = launch_cuda_resident_i32_compare_count(
            self,
            byte_offset,
            row_count,
            lower_inclusive,
            CudaI32Comparison::Gte,
        )?;
        let greater_than_upper_count = launch_cuda_resident_i32_compare_count(
            self,
            byte_offset,
            row_count,
            upper_inclusive,
            CudaI32Comparison::Gt,
        )?;
        Ok(greater_or_equal_lower_count.saturating_sub(greater_than_upper_count))
    }


    /// `SUM` of a resident int4 column over a FILTERED set of row indices (the operator axis, doc 19):
    /// gather `col[indices[k]]` and reduce on the GPU (each thread sums its strided slice locally, then
    /// one `atom.add.u64` -> a single i64), returned as bigint. The caller must pass a NON-empty
    /// `indices`; the engine maps an empty SQL aggregate to NULL before this low-level API.
    pub fn sum_i32_at_indices_from_payload(
        &self,
        byte_offset: u64,
        indices: &[u32],
    ) -> Result<i64, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_sum_at_indices(self, byte_offset, indices)
    }

    /// `SUM` of a resident INT8 column over a FILTERED set of row indices, as i128 (the operator axis,
    /// doc 19): a sum of i64 can exceed i64, so PG returns numeric. Each thread reduces its slice into
    /// a local i128, then a two-64-bit-atomic carry add into a single i128. `indices` must be non-empty.
    pub fn sum_i64_at_indices_i128_from_payload(
        &self,
        byte_offset: u64,
        indices: &[u32],
    ) -> Result<i128, CudaRuntimeProbeError> {
        launch_cuda_resident_i64_sum_at_indices_i128(self, byte_offset, indices)
    }

    /// `MIN` (the type matrix / operator axis, doc 19) of a resident int4 column over a FILTERED set
    /// of row indices: a GPU reduction (local min per thread + one `atom.min.s32`). `indices` must be
    /// non-empty; the engine maps an empty SQL MIN to NULL before this low-level API.
    pub fn min_i32_at_indices_from_payload(
        &self,
        byte_offset: u64,
        indices: &[u32],
    ) -> Result<i32, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_minmax_at_indices(self, byte_offset, indices, false)
    }

    /// `MAX` of a resident int4 column over a FILTERED set of row indices (the operator axis, doc 19);
    /// a GPU reduction (local max per thread + one `atom.max.s32`). `indices` must be non-empty.
    pub fn max_i32_at_indices_from_payload(
        &self,
        byte_offset: u64,
        indices: &[u32],
    ) -> Result<i32, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_minmax_at_indices(self, byte_offset, indices, true)
    }

    /// `MIN`/`MAX` of a resident INT8 column over a FILTERED set of row indices (the operator axis,
    /// doc 19); a GPU reduction (each thread reads its i64 as 2x4-byte loads, local min/max, then one
    /// `atom.min/max.s64`). `indices` must be non-empty; empty SQL MIN/MAX maps to NULL upstream.
    pub fn min_i64_at_indices_from_payload(
        &self,
        byte_offset: u64,
        indices: &[u32],
    ) -> Result<i64, CudaRuntimeProbeError> {
        launch_cuda_resident_i64_minmax_at_indices(self, byte_offset, indices, false)
    }

    pub fn max_i64_at_indices_from_payload(
        &self,
        byte_offset: u64,
        indices: &[u32],
    ) -> Result<i64, CudaRuntimeProbeError> {
        launch_cuda_resident_i64_minmax_at_indices(self, byte_offset, indices, true)
    }

    /// `MIN` of a resident NUMERIC column over a FILTERED set of row indices, as the i128 mantissa (the
    /// operator axis, doc 19); a GPU partials reduction + host combine. `indices` must be non-empty.
    pub fn min_i128_at_indices_from_payload(
        &self,
        byte_offset: u64,
        indices: &[u32],
    ) -> Result<i128, CudaRuntimeProbeError> {
        launch_cuda_resident_i128_minmax_partials_at_indices(self, byte_offset, indices, false)
    }

    /// `MAX` of a resident NUMERIC column over a FILTERED set of row indices, as the i128 mantissa.
    pub fn max_i128_at_indices_from_payload(
        &self,
        byte_offset: u64,
        indices: &[u32],
    ) -> Result<i128, CudaRuntimeProbeError> {
        launch_cuda_resident_i128_minmax_partials_at_indices(self, byte_offset, indices, true)
    }

    /// `SUM` of a resident NUMERIC column over a FILTERED set of row indices, as the i128 mantissa with
    /// CHECKED i128 overflow (the operator axis, doc 19); a bounded-grid partials reduction + host
    /// checked-combine. Overflow -> `NumericFieldOverflow`. `indices` must be non-empty.
    pub fn sum_i128_at_indices_from_payload(
        &self,
        byte_offset: u64,
        indices: &[u32],
    ) -> Result<i128, CudaRuntimeProbeError> {
        launch_cuda_resident_i128_sum_partials_at_indices(self, byte_offset, indices)
    }

    /// GROUP BY a resident int4 `key` column over a FILTERED set of row indices, aggregating COUNT(*)
    /// and SUM(int4 `sum`) per group via GPU hash aggregation (the operator axis, doc 19). Returns one
    /// [`GroupByI32Row`] per distinct key (unordered). For COUNT(*) pass `sum_byte_offset =
    /// key_byte_offset`. `indices` may be empty (-> no groups).
    pub fn group_by_i32_count_sum_from_payload(
        &self,
        input: CudaGroupByInput<'_>,
        indices: &[u32],
        // Query-aware aggregate-selection mask (`grouped_agg_mask`). The two-level kernel serves only
        // COUNT/SUM (its MIN/MAX bits are no-ops); the executor passes the query-wide mask so a
        // COUNT(*)-only query prunes the SUM atomics.
        agg_mask: u32,
    ) -> Result<Vec<GroupByI32Row>, CudaRuntimeProbeError> {
        launch_validated_group_by(self, input, indices, true, agg_mask)
    }

    /// GROUP BY with per-group MIN/MAX of the value (the returned [`GroupByI32Row`] carries
    /// `min`/`max`). Uses the SINGLE-LEVEL kernel, which computes min/max (the two-level kernel does
    /// not); count/sum are also valid. A two-level min/max kernel is a perf follow-on for low
    /// cardinality. `value_is_int8` selects the int8 (8-byte) vs int4 (4-byte) value read;
    /// `sum_byte_offset` = the value column (= key column for shapes that ignore it).
    pub fn group_by_i32_count_sum_minmax_from_payload(
        &self,
        input: CudaGroupByInput<'_>,
        indices: &[u32],
        // Query-aware aggregate-selection mask (`grouped_agg_mask`): only the masked-in count/sum/min/max
        // fields' per-row atomics run; the executor reads only what it requested.
        agg_mask: u32,
    ) -> Result<Vec<GroupByI32Row>, CudaRuntimeProbeError> {
        launch_validated_group_by(self, input, indices, false, agg_mask)
    }

    /// Benchmark entry: run GROUP BY with the chosen kernel (`two_level` selects the shared-mem
    /// two-level kernel vs the single-level global-atomic one). For perf comparison only; the engine
    /// always uses the two-level kernel via [`Self::group_by_i32_count_sum_from_payload`].
    pub fn group_by_i32_count_sum_bench(
        &self,
        input: CudaGroupByInput<'_>,
        indices: &[u32],
        two_level: bool,
    ) -> Result<Vec<GroupByI32Row>, CudaRuntimeProbeError> {
        launch_validated_group_by(self, input, indices, two_level, grouped_agg_mask::ALL)
    }

    /// Benchmark entry: time JUST the GROUP BY kernel (CUDA events, min of `runs`), returning the
    /// per-group rows + the min kernel milliseconds. Isolates the kernel from the alloc/H2D/D2H/compact
    /// overhead, so the two-level vs single-level difference is visible. Perf comparison only.
    pub fn group_by_i32_count_sum_kernel_timed(
        &self,
        input: CudaGroupByInput<'_>,
        indices: &[u32],
        two_level: bool,
        runs: u32,
        // Query-aware aggregate-selection mask (`grouped_agg_mask`): pass `COUNT` to time the pruned
        // path vs `ALL` for the full-compute path, isolating the per-row atomic savings.
        agg_mask: u32,
    ) -> Result<(Vec<GroupByI32Row>, f32), CudaRuntimeProbeError> {
        if runs == 0 {
            return Err(CudaRuntimeProbeError::InvalidInputLength(runs as usize));
        }
        if indices.is_empty() {
            return Ok((Vec::new(), 0.0));
        }
        if matches!(input.value, CudaGroupValueSource::Unused { .. }) {
            return Err(CudaRuntimeProbeError::InvalidInputLength(agg_mask as usize));
        }
        let validated = validate_group_input(self, input, indices)?;
        if validated.value_is_int8 || validated.key_is_int8 || validated.value_is_numeric
            || validated.value_is_uuid || validated.key_is_i128 || validated.key_is_text
            || validated.value_is_text || validated.key_base_override != 0
            || validated.value_base_override != 0 || validated.comp_w != 0 || validated.n_text != 0
            || validated.value_null_off.is_some() || validated.key_null_off.is_some()
        {
            return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
        }
        let kernel = if two_level {
            c"gpu_db_group_by_i32_count_sum_twolevel"
        } else {
            c"gpu_db_group_by_i32_count_sum"
        };
        launch_cuda_group_by_kernel_timed(
            self,
            validated.key_byte_offset,
            validated.value_byte_offset,
            indices,
            kernel,
            runs,
            agg_mask,
        )
    }

    pub fn project_i32_rows_from_payload(
        &self,
        byte_offset: u64,
        row_indices: &[u64],
    ) -> Result<Vec<i32>, CudaRuntimeProbeError> {
        copy_cuda_resident_i32_rows(self, byte_offset, row_indices)
    }

    /// Gather a resident BOOL column's bits at `row_indices` into host `bool` values (the type matrix,
    /// doc 19): the column is a 1-bit-per-row bitmap at `bitmap_byte_offset`; for each row index read
    /// the u32 word holding its bit and extract bit `idx & 31`. Like the i32/i64 gather, one device
    /// kernel gathers all selected rows and one bounded bulk D2H materializes the result.
    pub fn project_bool_rows_from_payload(
        &self,
        bitmap_byte_offset: u64,
        row_indices: &[u64],
    ) -> Result<Vec<bool>, CudaRuntimeProbeError> {
        copy_cuda_resident_bool_rows(self, bitmap_byte_offset, row_indices)
    }

    /// int8 (s64) analog of [`Self::project_i32_rows_from_payload`] (the type matrix, doc 19): gather
    /// the int8 column at `byte_offset` for the given row indices into host `i64` values.
    pub fn project_i64_rows_from_payload(
        &self,
        byte_offset: u64,
        row_indices: &[u64],
    ) -> Result<Vec<i64>, CudaRuntimeProbeError> {
        copy_cuda_resident_i64_rows(self, byte_offset, row_indices)
    }

    /// Surviving row indices of `col <cmp> scalar` (`scalar_on_left` flips it to `scalar <cmp> col`,
    /// for `K < big`) over a resident int8 column (the type matrix, doc 19). `comparison`
    /// 0=eq/1=lt/2=le/3=gt/4=ge/5=ne.
    pub fn expr_i64_compare_scalar_filter(
        &self,
        byte_offset: u64,
        scalar: i64,
        scalar_on_left: bool,
        comparison: u32,
        row_count: u64,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_i64_compare_scalar_filter(
            self,
            byte_offset,
            scalar,
            scalar_on_left,
            comparison,
            row_count,
        )
    }

    /// Surviving row indices of `a <cmp> b` over two resident int8 columns (the type matrix, doc 19).
    pub fn expr_i64_compare_columns_filter(
        &self,
        a_byte_offset: u64,
        b_byte_offset: u64,
        comparison: u32,
        row_count: u64,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_i64_compare_columns_filter(
            self,
            a_byte_offset,
            b_byte_offset,
            comparison,
            row_count,
        )
    }

    pub fn project_i128_rows_from_payload(
        &self,
        byte_offset: u64,
        row_indices: &[u64],
    ) -> Result<Vec<i128>, CudaRuntimeProbeError> {
        copy_cuda_resident_i128_rows(self, byte_offset, row_indices)
    }

    /// Surviving row indices of `col <cmp> scalar` (`scalar_on_left` flips it to `scalar <cmp> col`)
    /// over a resident numeric (i128) column (the type matrix, doc 19). `scalar` is the literal
    /// mantissa rescaled to the column scale on the host. `comparison` 0=eq/1=lt/2=le/3=gt/4=ge/5=ne.
    pub fn expr_i128_compare_scalar_filter(
        &self,
        byte_offset: u64,
        scalar: i128,
        scalar_on_left: bool,
        comparison: u32,
        row_count: u64,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_i128_compare_scalar_filter(
            self,
            byte_offset,
            scalar,
            scalar_on_left,
            comparison,
            row_count,
        )
    }

    /// Surviving row indices of `text[i] == needle` (or `<>` when `negate`) over a resident TEXT column
    /// (the type matrix, doc 19). The column is its offsets-array + byte-blob device offsets; `needle`
    /// is the literal's raw bytes (byte-wise = PG deterministic-collation equality).
    pub fn expr_text_eq_scalar_filter(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        needle: &[u8],
        negate: bool,
        row_count: u64,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_text_eq_scalar_filter(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            needle,
            negate,
            row_count,
        )
    }

    /// Surviving row indices of `text[i] <cmp> needle` over a resident TEXT column (the type matrix,
    /// doc 19): LEXICOGRAPHIC unsigned byte compare (memcmp of the common prefix; the shorter string
    /// sorts first), matching the host `compare_sql_values` Text order (Rust `str::cmp`). `cmp`:
    /// 0=eq/1=lt/2=le/3=gt/4=ge/5=ne; `scalar_on_left` reverses the operand order. `validity_offsets`
    /// (M3 — doc 21) holds the text column's validity bitmap offset when nullable (a NULL operand is
    /// UNKNOWN ⇒ excluded); empty ⇒ byte-identical to the no-NULL path.
    #[allow(clippy::too_many_arguments)]
    pub fn expr_text_compare_scalar_filter(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        needle: &[u8],
        scalar_on_left: bool,
        comparison: u32,
        row_count: u64,
        validity_offsets: &[u64],
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_text_compare_scalar_filter(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            needle,
            scalar_on_left,
            comparison,
            row_count,
            validity_offsets,
        )
    }

    /// Surviving row indices of `text[i] LIKE pattern` over a resident TEXT column (the type matrix,
    /// doc 19). `tokens` is the pattern compiled to the kernel ABI -- one u32 per token,
    /// `(op << 8) | literal_byte`, op 0 = literal byte, 1 = any-one (`_`), 2 = any-run (`%`), with `\`
    /// escapes already resolved by the caller. Matches with full UTF-8 character semantics for `_`/`%`.
    pub fn expr_text_like_scalar_filter(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        tokens: &[u32],
        row_count: u64,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_text_like_scalar_filter(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            tokens,
            row_count,
        )
    }

    /// Surviving row indices of `uuid[i] <cmp> needle` over a resident UUID column (the type matrix,
    /// doc 19). The column is 16 raw bytes/row in the i128 section; `needle` is the literal's 16 bytes.
    /// Comparison is an unsigned big-endian 16-byte memcmp (PG's uuid order). `cmp`: 0=eq/1=lt/2=le/
    /// 3=gt/4=ge/5=ne; `scalar_on_left` reverses the operand order.
    /// `validity_offsets` (M3 — doc 21): the NULL validity bitmap byte offset of the uuid column when it
    /// is nullable (empty when not). A NULL operand is UNKNOWN ⇒ excluded (the mask is AND'd with the
    /// validity mask before compaction). Empty ⇒ byte-identical to the no-NULL path.
    pub fn expr_uuid_compare_scalar_filter(
        &self,
        byte_offset: u64,
        needle: &[u8; 16],
        scalar_on_left: bool,
        comparison: u32,
        row_count: u64,
        validity_offsets: &[u64],
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_uuid_compare_scalar_filter(
            self,
            byte_offset,
            needle,
            scalar_on_left,
            comparison,
            row_count,
            validity_offsets,
        )
    }

    /// Surviving row indices of `a <cmp> b` over two resident UUID columns (the type matrix, doc 19),
    /// each 16 raw bytes/row; unsigned big-endian 16-byte memcmp. `validity_offsets` (M3 — doc 21) holds
    /// the validity bitmap offset of each NULLABLE operand (a NULL in either ⇒ the row is excluded).
    pub fn expr_uuid_compare_columns_filter(
        &self,
        a_byte_offset: u64,
        b_byte_offset: u64,
        comparison: u32,
        row_count: u64,
        validity_offsets: &[u64],
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_uuid_compare_columns_filter(
            self,
            a_byte_offset,
            b_byte_offset,
            comparison,
            row_count,
            validity_offsets,
        )
    }

    /// Surviving row indices of a `WHERE bool_col` predicate (the type matrix, doc 19): the bool
    /// column's 1-bit-per-row bitmap at `bitmap_byte_offset` is expanded to an i32 0/1 mask (bit i ->
    /// row i, XOR `negate` for `NOT flag` / `flag = false`), then the shared compactor selects the set
    /// rows. This primitive expands only the supplied value bitmap; nullable SQL lowering composes
    /// the separate validity bitmap before selection.
    pub fn expr_bool_to_mask_filter(
        &self,
        bitmap_byte_offset: u64,
        negate: bool,
        row_count: u64,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_bool_to_mask_filter(self, bitmap_byte_offset, negate, row_count)
    }

    /// Surviving row indices of `a <cmp> b` over two resident numeric (i128) columns of the SAME scale
    /// (the type matrix, doc 19).
    pub fn expr_i128_compare_columns_filter(
        &self,
        a_byte_offset: u64,
        b_byte_offset: u64,
        comparison: u32,
        row_count: u64,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_i128_compare_columns_filter(
            self,
            a_byte_offset,
            b_byte_offset,
            comparison,
            row_count,
        )
    }

    pub fn match_project_i32_equal_from_payload(
        &self,
        filters: &[(u64, i32)],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<Vec<Vec<i32>>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_equal_project(self, filters, projection_offsets, row_count)
    }

    pub fn match_project_i32_equal_any_from_payload(
        &self,
        filter_offset: u64,
        needles: &[i32],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<Vec<CudaI32BatchProjectionRow>, CudaRuntimeProbeError> {
        if row_count == 0 {
            return Ok(Vec::new());
        }
        self.submit_match_project_i32_equal_any_from_payload(
            filter_offset,
            needles,
            projection_offsets,
            row_count,
        )?
        .complete(self)
    }

    pub fn submit_match_project_i32_equal_any_from_payload(
        &self,
        filter_offset: u64,
        needles: &[i32],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<CudaI32EqualAnyProjectSubmission, CudaRuntimeProbeError> {
        submit_cuda_resident_i32_equal_any_project(
            self,
            filter_offset,
            needles,
            projection_offsets,
            row_count,
        )
    }

    /// R1a — GPU-index point-lookup variant of [`Self::submit_match_project_i32_equal_any_from_payload`]:
    /// hash-probes the device-resident `index_ptr` (built over the key column on admission) instead of
    /// scanning, producing the SAME `CudaI32EqualAnyProjectSubmission` (so `complete`/`complete_detached`
    /// are unchanged). `index_table_mask` = table_size-1, `index_hash_shift` = 32 - log2(table_size).
    pub fn submit_match_project_i32_index_probe_from_payload(
        &self,
        index: &Arc<CudaResidentDeviceMemory>,
        index_table_mask: u32,
        index_hash_shift: u32,
        needles: &[i32],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<CudaI32EqualAnyProjectSubmission, CudaRuntimeProbeError> {
        submit_cuda_resident_i32_index_probe(
            self,
            index,
            index_table_mask,
            index_hash_shift,
            needles,
            projection_offsets,
            row_count,
        )
    }

    /// DENSE-emit unique index probe (DECISIONS "lpb read levers" #1) — same args as the atomic variant.
    pub fn submit_match_project_i32_index_probe_dense_from_payload(
        &self,
        index: &Arc<CudaResidentDeviceMemory>,
        index_table_mask: u32,
        index_hash_shift: u32,
        needles: &[i32],
        projection_offsets: &[u64],
        row_count: u64,
    ) -> Result<CudaI32IndexProbeDenseSubmission, CudaRuntimeProbeError> {
        submit_cuda_resident_i32_index_probe_dense(
            self,
            index,
            index_table_mask,
            index_hash_shift,
            needles,
            projection_offsets,
            row_count,
        )
    }

    /// Sub-slice 8 v2 (charter-faithful multi-shard probe): probe a BATCH of needles against ALL `shards`'
    /// device indexes in ONE kernel launch, gathering + dense-emitting on the GPU (1xN output, no host merge).
    /// `self` is only the allocation/launch context (any shard's device buffer on the same GPU works).
    pub fn submit_multi_shard_i32_index_probe_dense(
        &self,
        shards: &[MultiShardProbeShard],
        needles: &[i32],
        read_snapshot: u64,
    ) -> Result<CudaI32IndexProbeDenseSubmission, CudaRuntimeProbeError> {
        submit_cuda_resident_i32_multi_shard_index_probe_dense(
            self,
            shards,
            needles,
            read_snapshot,
        )
    }

    /// P5-later: probe compact per-chunk Bloom filters on-device and return candidate chunk indexes per needle.
    /// False positives are resolved by the caller's exact device predicate; false negatives are forbidden.
    pub fn probe_chunk_blooms(
        &self,
        blooms: &[ChunkBloomProbeShard],
        needles: &[i32],
    ) -> Result<Vec<Vec<u32>>, CudaRuntimeProbeError> {
        probe_cuda_chunk_blooms(self, blooms, needles)
    }

    pub fn match_i32_equal_row_indices_from_payload(
        &self,
        filters: &[(u64, i32)],
        row_count: u64,
    ) -> Result<Vec<u64>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_equal_row_indices(self, filters, row_count)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn match_project_i32_equal_any_text_from_payload(
        &self,
        filter_offset: u64,
        filter_validity_bitmap_offset: Option<u64>,
        needles: &[i32],
        projection_offsets: &[u64],
        text_offsets_byte_offset: u64,
        text_bytes_byte_offset: u64,
        text_bytes_len: u64,
        text_validity_bitmap_offset: Option<u64>,
        row_count: u64,
    ) -> Result<Vec<CudaI32TextBatchProjectionRow>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_equal_any_project_text(
            self,
            filter_offset,
            filter_validity_bitmap_offset,
            needles,
            projection_offsets,
            text_offsets_byte_offset,
            text_bytes_byte_offset,
            text_bytes_len,
            text_validity_bitmap_offset,
            row_count,
        )
    }


    pub fn project_i32_compare_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
    ) -> Result<Vec<i32>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_compare_project(self, byte_offset, row_count, needle, comparison)
    }

    /// PROTOTYPE for the general GPU executor (docs/architecture/17 §5): evaluate the predicate
    /// `(a <op> b) <cmp> needle` over two resident int4 columns by composing two buffer->buffer
    /// primitives (elementwise binary into an intermediate buffer, then compare->matching-row-
    /// indices). Returns the matching row indices, host-sorted ascending. `op_code` 0=add/1=sub/2=mul;
    /// `comparison` 0=eq/1=lt/2=le/3=gt/4=ge. Demonstrates the vectorized-interpreter model (an Expr
    /// tree lowered to a pipeline of primitives over intermediates), not a new shape kernel.
    pub fn expr_filter_two_col_compare_from_payload(
        &self,
        a_byte_offset: u64,
        b_byte_offset: u64,
        op_code: u32,
        row_count: u64,
        needle: i32,
        comparison: u32,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_expr_two_col_filter(
            self,
            a_byte_offset,
            b_byte_offset,
            op_code,
            row_count,
            needle,
            comparison,
        )
    }

    /// Run an arithmetic bytecode `program` (docs/architecture/17 section 2.3) over the resident
    /// columns to evaluate an arbitrary int4 arithmetic tree on device, then compare the result to
    /// `needle` and return the matching row indices (host-sorted ascending). The general recursive
    /// interpreter behind the engine's `ResidentExpr` lowering; `comparison` 0=eq/1=lt/2=le/3=gt/4=ge.
    pub fn run_expr_arith_filter(
        &self,
        program: &[ExprStep],
        row_count: u64,
        comparison: u32,
        needle: i32,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_expr_arith_filter(self, program, row_count, comparison, needle)
    }

    /// Run an arithmetic bytecode `program` that leaves TWO value buffers (compiled lhs then rhs),
    /// then compare them elementwise (`lhs <cmp> rhs`) and return the matching row indices. Behind the
    /// engine's column-vs-column / expr-vs-expr predicate lowering; `comparison` 0=eq/1=lt/2=le/3=gt/
    /// 4=ge.
    pub fn run_expr_compare_buffers_filter(
        &self,
        program: &[ExprStep],
        row_count: u64,
        comparison: u32,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_expr_compare_buffers_filter(self, program, row_count, comparison)
    }

    /// Evaluate a SIMPLE `int4col <cmp> needle` predicate over a resident int4 column and return the
    /// surviving ROW INDICES in ASCENDING ORDER, with NO host sort. Runs the same ordered parallel
    /// compaction as the value path (`project_i32_compare_*`), but the scatter kernel stores each
    /// match's row index instead of its value (`out_is_index = 1`). The ascending order is guaranteed
    /// by construction (the contiguous block partition + ordered intra-block prefix sum), so this is a
    /// drop-in replacement for the atomic-append + host-`sort_unstable` index path. `comparison` is the
    /// raw kernel code: 0=eq, 1=lt, 2=lte, 3=gt, 4=gte, 5=ne.
    pub fn compare_indices_ordered_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: u32,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_compare_indices_ordered(
            self,
            byte_offset,
            row_count,
            needle,
            comparison,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaI32Comparison {
    Lt,
    Lte,
    Gt,
    Gte,
}

impl CudaI32Comparison {
    fn code(self) -> u32 {
        match self {
            Self::Lt => 1,
            Self::Lte => 2,
            Self::Gt => 3,
            Self::Gte => 4,
        }
    }
}

/// Aggregate-selection mask for the LIVE per-group two-level i32 GROUP BY kernel
/// (`gpu_db_group_by_i32_count_sum_twolevel`, via `group_by_i32_count_sum_minmax_from_payload`). Each
/// bit selects which per-row update atomic runs at the matched slot: a query needing only one
/// aggregate (e.g. `SELECT MAX(id)`) passes only that bit and skips the other three atomics.
/// Masked-out fields stay at their init sentinels (count/sum=0, min=`i32::MAX`, max=`i32::MIN`); the
/// caller must read ONLY the field(s) whose bit it set. The slot-claim + probe always run regardless
/// (group identity is needed for every row), so the compacted group SET is unchanged — only the
/// unrequested statistics are left at their sentinels. `ALL` = full count+sum+min+max.
pub mod grouped_agg_mask {
    pub const COUNT: u32 = 1;
    pub const SUM: u32 = 2;
    pub const MIN: u32 = 4;
    pub const MAX: u32 = 8;
    pub const ALL: u32 = COUNT | SUM | MIN | MAX;
}


/// Queue ONE stream-ordered (async) D2H of `dst.len() * size_of::<T>()` bytes from `device_ptr`
/// into a pooled pinned host staging buffer if one can be leased (truly-async + DMA-fast), else
/// directly into `dst` (still async, just from pageable memory). Returns the pinned lease (kept
/// alive by the caller until after the stream sync, then drained by `copy_pinned_into`), or
/// `None` when the copy went straight to `dst`. An empty `dst` queues nothing and returns `None`.
fn stage_result_dtoh_async<'a, T>(
    primary: &'a GpuPrimaryContext,
    dtoh_async: unsafe extern "C" fn(*mut c_void, u64, usize, *mut c_void) -> i32,
    stream: *mut c_void,
    device_ptr: u64,
    dst: &mut [T],
) -> Result<Option<PinnedHostLease<'a>>, CudaRuntimeProbeError> {
    let bytes = std::mem::size_of_val(dst);
    if bytes == 0 {
        return Ok(None);
    }
    match primary.lease_pinned_host_buffer(bytes) {
        Some(pinned) => {
            check_cuda(unsafe { dtoh_async(pinned.ptr, device_ptr, bytes, stream) })?;
            Ok(Some(pinned))
        }
        None => {
            check_cuda(unsafe {
                dtoh_async(dst.as_mut_ptr().cast::<c_void>(), device_ptr, bytes, stream)
            })?;
            Ok(None)
        }
    }
}

/// Copy a completed pinned-host staging buffer into its owned `dst` Vec by typed pointer (no-op
/// when the staged copy went straight to `dst`, i.e. `pinned` is `None`). Must be called only
/// after the stream sync that completed the D2H into the pinned region.
fn copy_pinned_into<T>(pinned: &Option<PinnedHostLease<'_>>, dst: &mut [T]) {
    if let Some(pinned) = pinned {
        // SAFETY: the matching `stage_result_dtoh_async` leased `pinned` with capacity ≥
        // size_of_val(dst) and the stream sync completed the D2H of exactly that many bytes;
        // pinned host memory is page-aligned, so the typed read is well-aligned for `T`.
        unsafe {
            std::ptr::copy_nonoverlapping(pinned.ptr.cast::<T>(), dst.as_mut_ptr(), dst.len());
        }
    }
}

// PROTOTYPE — general GPU executor (docs/architecture/17-general-gpu-executor.md §5).
// Evaluates the predicate `(a <op> b) <cmp> needle` over two resident int4 columns by COMPOSING two
// buffer->buffer primitives on one pooled stream — the vectorized-interpreter model that replaces
// hand-coded per-shape kernels:
//   (1) `gpu_db_resident_i32_binary_elementwise` writes the INTERMEDIATE `t = a <op> b` into a leased
//       device buffer (the key generalization: `t` is not a resident payload column);
//   (2) `gpu_db_buffer_i32_compare_to_indices` scans the intermediate `t` and atomic-appends the
//       matching row indices.
// Returns the matching row indices, host-sorted ascending for determinism (the atomic-append order is
// the non-deterministic GPU schedule; same pattern as `..._equal_row_indices`). The caller gathers
// the projected column at these indices via the existing `project_i32_rows_from_payload`. This is a
// proof of the buffer-intermediate + Expr-lowering + primitive-composition models, NOT a new shape
// method — the predicate is interpreted, and arbitrary trees lower to longer pipelines of the same
// primitives.
fn launch_cuda_resident_expr_two_col_filter(
    resident: &CudaResidentDeviceMemory,
    a_byte_offset: u64,
    b_byte_offset: u64,
    op_code: u32,
    n: u64,
    needle: i32,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuLaunchKernel = unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    const PTX: &[u8] = include_bytes!("expression_i32.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let byte_len = n_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;

    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let elementwise_fn =
        primary.cached_function(c"gpu_db_resident_i32_binary_elementwise", &ptx)?;
    let compare_fn = primary.cached_function(c"gpu_db_buffer_i32_compare_to_indices", &ptx)?;

    // Intermediate `t` buffer, the matching-index output buffer, and the atomic counter.
    let intermediate = primary.lease_device_buffer(byte_len)?;
    let indices_buf = primary.lease_device_buffer(byte_len)?;
    let count_buf = primary.lease_device_buffer(std::mem::size_of::<u32>())?;
    // CHECKED int4 arithmetic (Charter rule 2 PG-fidelity): the elementwise kernel ORs 1 into this
    // flag if `a <op> b` overflows int32; the host raises `integer out of range` after the launches.
    let overflow_buf = primary.lease_device_buffer(std::mem::size_of::<u32>())?;
    // Zero the atomic counter AND the overflow flag with blocking HtoDs before the kernels
    // (synchronous, so they complete before the pooled-stream launches read/write them).
    let zero = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_htod(
            count_buf.ptr,
            (&zero as *const u32).cast::<c_void>(),
            std::mem::size_of::<u32>(),
        )
    })?;
    check_cuda(unsafe {
        cu_memcpy_htod(
            overflow_buf.ptr,
            (&zero as *const u32).cast::<c_void>(),
            std::mem::size_of::<u32>(),
        )
    })?;

    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;

    let mut res_arg = resident.device_ptr();
    let mut a_arg = a_byte_offset;
    let mut b_arg = b_byte_offset;
    let mut n_arg = n;
    let mut op_arg = op_code;
    let mut t_arg = intermediate.ptr;
    let mut ovf_arg = overflow_buf.ptr;
    let mut ew_args = [
        (&mut res_arg as *mut u64).cast::<c_void>(),
        (&mut a_arg as *mut u64).cast::<c_void>(),
        (&mut b_arg as *mut u64).cast::<c_void>(),
        (&mut n_arg as *mut u64).cast::<c_void>(),
        (&mut op_arg as *mut u32).cast::<c_void>(),
        (&mut t_arg as *mut u64).cast::<c_void>(),
        (&mut ovf_arg as *mut u64).cast::<c_void>(),
    ];

    let mut in_arg = intermediate.ptr;
    let mut n2_arg = n;
    let mut needle_arg = needle;
    let mut cmp_arg = comparison;
    let mut count_arg = count_buf.ptr;
    let mut idx_arg = indices_buf.ptr;
    let mut cmp_args = [
        (&mut in_arg as *mut u64).cast::<c_void>(),
        (&mut n2_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut cmp_arg as *mut u32).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
        (&mut idx_arg as *mut u64).cast::<c_void>(),
    ];

    // Both kernels on ONE pooled stream: stream order makes the compare read `t` only after the
    // elementwise write completes. `launch_on_pooled_stream` syncs (and drains on error) before
    // returning, so the blocking readback below sees finished work.
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let first = unsafe {
            cu_launch_kernel(
                elementwise_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                ew_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if first != 0 {
            return first;
        }
        unsafe {
            cu_launch_kernel(
                compare_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                cmp_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;

    // PG-fidelity: if any row's `a <op> b` overflowed int32, the elementwise kernel set the flag.
    // Raise `integer out of range` and discard the (now-untrustworthy) comparison result, exactly as
    // Postgres aborts the statement. Read before the count so an overflow never returns rows.
    let mut overflow = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut overflow as *mut u32).cast::<c_void>(),
            overflow_buf.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;
    if overflow != 0 {
        return Err(CudaRuntimeProbeError::IntegerOutOfRange);
    }

    let mut match_count = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut match_count as *mut u32).cast::<c_void>(),
            count_buf.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;
    let match_count = usize::try_from(match_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?
        .min(n_usize);
    let mut indices = vec![0_u32; match_count];
    if match_count > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                indices.as_mut_ptr().cast::<c_void>(),
                indices_buf.ptr,
                match_count * std::mem::size_of::<u32>(),
            )
        })?;
    }
    // Atomic-append order is the GPU schedule; sort for a deterministic result (matches the engine's
    // stable row order and the `..._equal_row_indices` host-sort).
    indices.sort_unstable();
    Ok(indices)
}

/// Compare an int4 value-buffer lease to `needle` and return the matching row indices ascending.
/// Shared compare-and-compact tail for the arithmetic VM (the 2-col fast-path launcher keeps its
/// own fused copy). `comparison`: 0=eq, 1=lt, 2=le, 3=gt, 4=ge.
fn compact_buffer_i32_compare_to_indices(
    resident: &CudaResidentDeviceMemory,
    value: &PooledBufferLease<'_>,
    n: u64,
    needle: i32,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    // The scalar compact kernel switched on codes 0=eq..4=ge only; reject 5=ne (mask-path only) so a
    // caller gets an error, not a silently-empty result. Preserved byte-identically across the
    // re-route to the ordered compaction (no production caller passes 5 here — `ne` lowers to a mask).
    validate_ordered_i32_comparison(comparison, 4)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    // probe-timing (VM lever): the int4 simple-comparison compaction (`col <cmp> scalar` -> indices). The
    // legacy path was a fused compare+atomic-append kernel + count/index D2H + a host `sort_unstable` of
    // the surviving indices; the ~93%-of-cost host sort is what the ordered compaction eliminates.
    let _compact_scope = Probe::scope("compact");

    // Re-route to the ORDERED parallel compaction (`COMPARE_ORDERED_PTX`) in INDEX-emit mode: read the
    // leased VALUE buffer as the contiguous i32 input. The typed wrapper verifies that this lease
    // belongs to the resident context and covers every `value[idx]` read before either kernel launch.
    // Surviving row indices are emitted ASCENDING BY CONSTRUCTION (contiguous block partition +
    // ordered intra-block prefix sum), replacing the host `sort_unstable`.
    //
    // Lease lifetime: the ordered core does count -> host-scan -> scatter (TWO launches reading
    // value lease). The borrow spans this whole call, so the input outlives both reads.
    launch_cuda_buffer_i32_compare_indices_ordered(resident, value, n, needle, comparison)
}

/// PROTOTYPE — the resident arithmetic bytecode VM (docs/architecture/17 section 2.3). Runs a postfix
/// `program` over a stack of leased device buffers to evaluate an ARBITRARY int4 arithmetic tree on
/// device (each step launches one buffer->buffer primitive), then compares the single result buffer
/// to `needle` and returns the matching row indices. This is the general recursive interpreter that
/// generalizes the fixed 2-col fast-path: arbitrary depth lowers to a longer program of the same
/// primitives. Correctness-first: each step runs on a pooled stream that syncs, so an intermediate
/// is valid before the next step reads it (pipelining/fusion is a later perf lever).
/// Compare two int4 value-buffer leases elementwise and return the matching row indices in ASCENDING
/// order. The col-vs-col / expr-vs-expr analogue of
/// `compact_buffer_i32_compare_to_indices`. `comparison`: 0=eq, 1=lt, 2=le, 3=gt, 4=ge.
fn compact_buffers_i32_compare_to_indices(
    resident: &CudaResidentDeviceMemory,
    lhs: &PooledBufferLease<'_>,
    rhs: &PooledBufferLease<'_>,
    n: u64,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    // The buffer compact kernel switches on codes 0=eq..4=ge only; reject 5=ne (mask-path only) so a
    // caller gets an error, not a silently-empty result. Preserved byte-identically across the re-route
    // to the ordered two-input compaction (no production caller passes 5 here - `ne` lowers to a mask).
    validate_ordered_i32_comparison(comparison, 4)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    // Re-route to the TWO-INPUT ORDERED parallel compaction (`COMPARE_ORDERED_PTX`,
    // `gpu_db_buffers_i32_compare_*_blocks`): compare `lhs[i] <cmp> rhs[i]` over the two leased value
    // buffers (each contiguous i32 at `base + idx*4` - the same layout the legacy
    // `gpu_db_buffer_i32_compare_buffers_to_indices` atomic kernel read). The typed wrapper verifies
    // both exact lease capacities and context ownership before launch. Surviving row indices are
    // IDENTICAL (operand order lhs,rhs; codes 0..4), but emitted ASCENDING BY CONSTRUCTION.
    //
    // Lease lifetime: the ordered launch does count -> host-scan -> scatter (TWO launches, each reading
    // BOTH buffers). The caller leases both value buffers and holds those leases across this whole call
    // (borrowed for the duration), so both inputs outlive both reads - unchanged from the legacy
    // single-launch path, which also read the same two buffers.
    launch_cuda_resident_i32_compare_buffers_indices_ordered(
        resident,
        lhs,
        rhs,
        n,
        comparison,
    )
}

/// Evaluate an arithmetic `program` to one value buffer, then compare it to `needle` -> row indices.
fn launch_cuda_resident_expr_arith_filter(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    n: u64,
    compare_code: u32,
    needle: i32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut stack = run_resident_arith_program(
        resident,
        program,
        &[],
        n,
        ResidentElemType::I32,
        ExprTerminal::Value,
    )?;
    let value = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        // A well-formed arithmetic program leaves exactly one value on the stack.
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    compact_buffer_i32_compare_to_indices(resident, &value, n, needle, compare_code)
}

fn launch_cuda_arith_value_column_at_indices(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    n_rows: u64,
    indices: &[u32],
    elem: ResidentElemType,
) -> Result<Vec<i64>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    let n = usize::try_from(n_rows).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    // Run over ALL rows at the program's element width (I32 for int4 exprs, I64 for int8 -- the arith
    // VM checks the matching overflow bounds internally -> PG error, no wrap); the value buffer (lease)
    // is the single result + must stay alive through the D2H. Read it at that width, gather at the
    // survivor indices, widen to i64 (the universal sort-key width).
    let mut stack = run_resident_arith_program(
        resident,
        program,
        &[],
        n_rows,
        elem,
        ExprTerminal::Value,
    )?;
    let value = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        // A well-formed arithmetic program leaves exactly one value on the stack.
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    let mut out = Vec::with_capacity(indices.len());
    match elem {
        ResidentElemType::I32 => {
            let byte_len = n
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
            let mut host = vec![0i32; n];
            check_cuda(unsafe {
                cu_memcpy_dtoh(host.as_mut_ptr().cast::<c_void>(), value.ptr, byte_len)
            })?;
            drop(value);
            for &i in indices {
                let v = *host
                    .get(i as usize)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(i as usize))?;
                out.push(i64::from(v));
            }
        }
        ResidentElemType::I64 => {
            let byte_len = n
                .checked_mul(std::mem::size_of::<i64>())
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
            let mut host = vec![0i64; n];
            check_cuda(unsafe {
                cu_memcpy_dtoh(host.as_mut_ptr().cast::<c_void>(), value.ptr, byte_len)
            })?;
            drop(value);
            for &i in indices {
                let v = *host
                    .get(i as usize)
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(i as usize))?;
                out.push(v);
            }
        }
        // i128 (numeric) sort expressions are not yet supported here.
        ResidentElemType::I128 => {
            drop(value);
            return Err(CudaRuntimeProbeError::InvalidInputLength(0));
        }
    }
    Ok(out)
}

/// As [`launch_cuda_arith_value_column_at_indices`] (I32 arith) but the expression is NULLABLE: run the
/// arith program (value buffer) AND a `validity_program` (a mask VM program yielding 1 where every
/// operand column is non-NULL), then blend ON-DEVICE into an i64 key buffer -- `i64::MAX` (PG's
/// default-end sentinel; no widened int4 value can equal it) where NULL, else the sign-extended value.
/// D2H the i64 keys and gather at `indices` (M3 -- doc 21).
fn launch_cuda_arith_value_column_at_indices_nullable(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    validity_program: &[ExprStep],
    n_rows: u64,
    indices: &[u32],
) -> Result<Vec<i64>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuLaunchKernel = unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32;
    const PTX: &[u8] = include_bytes!("device_fill.ptx");
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    let n = usize::try_from(n_rows).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let primary = resident.primary();
    primary.set_current()?;
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    // The arith VALUE (I32) and the VALIDITY mask (I32, 0/1) -- two independent VM runs over all rows; each
    // top-of-stack buffer is the single result and must stay alive through the blend.
    let mut value_stack = run_resident_arith_program(
        resident,
        program,
        &[],
        n_rows,
        ResidentElemType::I32,
        ExprTerminal::Value,
    )?;
    let value = value_stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !value_stack.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    let mut mask_stack = run_resident_arith_program(
        resident,
        validity_program,
        &[],
        n_rows,
        ResidentElemType::I32,
        ExprTerminal::Mask,
    )?;
    let mask = mask_stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !mask_stack.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            validity_program.len(),
        ));
    }
    // Blend ON-DEVICE: out_i64[i] = mask[i] ? sign_extend(value[i]) : i64::MAX.
    let out_bytes = n
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    let out_buf = primary.lease_device_buffer(out_bytes)?;
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let blend_fn = primary.cached_function(c"gpu_db_blend_widen_null_sentinel", &ptx)?;
    const BLOCK: u32 = 256;
    let grid = n_rows.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = value.ptr;
    let mut a1 = mask.ptr;
    let mut a2 = out_buf.ptr;
    let mut a3 = n_rows;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            blend_fn,
            grid,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    let mut host = vec![0i64; n];
    check_cuda(unsafe {
        cu_memcpy_dtoh(host.as_mut_ptr().cast::<c_void>(), out_buf.ptr, out_bytes)
    })?;
    drop(out_buf);
    drop(mask);
    drop(value);
    let mut out = Vec::with_capacity(indices.len());
    for &i in indices {
        let v = *host
            .get(i as usize)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(i as usize))?;
        out.push(v);
    }
    Ok(out)
}

/// Evaluate a `program` that leaves TWO value buffers (the compiled lhs then rhs of a comparison),
/// then compare them elementwise (`lhs <cmp> rhs`) -> row indices. The col-vs-col / expr-vs-expr
/// filter behind the engine's `Compare(expr, expr)` lowering.
fn launch_cuda_resident_expr_compare_buffers_filter(
    resident: &CudaResidentDeviceMemory,
    program: &[ExprStep],
    n: u64,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut stack = run_resident_arith_program(
        resident,
        program,
        &[],
        n,
        ResidentElemType::I32,
        ExprTerminal::TwoValues,
    )?;
    let rhs = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    let lhs = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    compact_buffers_i32_compare_to_indices(resident, &lhs, &rhs, n, comparison)
}

// P2 §9.5/S2 — stable bitonic argsort primitive (the small-result branch of the adaptive GPU sort
// operator). Sorts N i64 keys (resident, at `keys_byte_offset`) by (key, original index) and returns
// the permutation indices, ascending or descending. Two kernels on one pooled stream:
//   - `..._init`: grid-stride fill keys_work[i] = (i<N) ? key[i] : i64::MAX (pad sorts to the end),
//     idx_work[i] = i.
//   - `..._step`: one compare-exchange stage of the bitonic network; the host loops k (subsequence
//     size) and j (compare distance) over O(log²N) launches.
// STABLE: equal keys break the tie by ASCENDING original index regardless of direction, matching the
// engine's stable CPU ORDER BY. Host-looped steps handle any N; the per-step launch overhead is
// exactly why radix (S3) wins above the crossover and the adaptive dispatch (S4) chooses.
#[allow(dead_code)] // S2 primitive: wired into the production adaptive sort operator in S4.
fn launch_cuda_resident_i64_argsort_bitonic(
    resident: &CudaResidentDeviceMemory,
    keys_device_ptr: u64,
    n: u64,
    descending: bool,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuLaunchKernel = unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_bitonic_argsort_init(
    .param .u64 keys_src_ptr,
    .param .u64 n,
    .param .u64 n_pad,
    .param .u64 keys_work_ptr,
    .param .u64 idx_work_ptr,
    .param .u32 descending
)
{
    .reg .pred %p_done;
    .reg .pred %p_real;
    .reg .pred %p_pdesc;
    .reg .u32 %desc;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %gdim;
    .reg .u32 %tmp32;
    .reg .u32 %idxv;
    .reg .u64 %src;
    .reg .u64 %n;
    .reg .u64 %npad;
    .reg .u64 %kw;
    .reg .u64 %iw;
    .reg .u64 %i;
    .reg .u64 %stride;
    .reg .u64 %off8;
    .reg .u64 %off4;
    .reg .u64 %addr;
    .reg .s64 %keyv;

    ld.param.u64 %src, [keys_src_ptr];
    ld.param.u64 %n, [n];
    ld.param.u64 %npad, [n_pad];
    ld.param.u64 %kw, [keys_work_ptr];
    ld.param.u64 %iw, [idx_work_ptr];
    ld.param.u32 %desc, [descending];
    setp.eq.u32 %p_pdesc, %desc, 1;

    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %tmp32, %bid, %bdim, %lane;
    cvt.u64.u32 %i, %tmp32;
    mul.lo.u32 %tmp32, %gdim, %bdim;
    cvt.u64.u32 %stride, %tmp32;

iloop:
    setp.ge.u64 %p_done, %i, %npad;
    @%p_done bra idone;
    mul.lo.u64 %off8, %i, 8;
    mul.lo.u64 %off4, %i, 4;
    cvt.u32.u64 %idxv, %i;
    add.u64 %addr, %iw, %off4;
    st.global.u32 [%addr], %idxv;
    setp.lt.u64 %p_real, %i, %n;
    @!%p_real bra ipad;
    add.u64 %addr, %src, %off8;
    ld.global.s64 %keyv, [%addr];
    bra istore;
ipad:
    // Padding sentinel must sink to the UNREAD end so idx[0..n) holds only real rows.
    // Ascending: pad = i64::MAX (sorts to the high end). Descending: pad = i64::MIN
    // (sorts to the low end). MIN is computed as MAX+1 (two's-complement wrap) to avoid
    // the most-negative-literal PTX parse pitfall.
    mov.s64 %keyv, 9223372036854775807;
    @%p_pdesc add.s64 %keyv, %keyv, 1;
istore:
    add.u64 %addr, %kw, %off8;
    st.global.s64 [%addr], %keyv;
    add.u64 %i, %i, %stride;
    bra iloop;
idone:
    ret;
}

.visible .entry gpu_db_bitonic_argsort_step(
    .param .u64 keys_work_ptr,
    .param .u64 idx_work_ptr,
    .param .u64 n_pad,
    .param .u64 j,
    .param .u64 k,
    .param .u32 descending
)
{
    .reg .pred %p_done;
    .reg .pred %p_pair;
    .reg .pred %p_asc;
    .reg .pred %p_nasc;
    .reg .pred %p_desc;
    .reg .pred %p_ndesc;
    .reg .pred %p_kgt;
    .reg .pred %p_klt;
    .reg .pred %p_keq;
    .reg .pred %p_igt;
    .reg .pred %p_ilt;
    .reg .pred %p_ka;
    .reg .pred %p_kb;
    .reg .pred %p_ta;
    .reg .pred %p_tb;
    .reg .pred %p_iaft;
    .reg .pred %p_ibef;
    .reg .pred %p_s1;
    .reg .pred %p_s2;
    .reg .pred %p_swap;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %gdim;
    .reg .u32 %tmp32;
    .reg .u32 %desc;
    .reg .u32 %idxi;
    .reg .u32 %idxl;
    .reg .u64 %kw;
    .reg .u64 %iw;
    .reg .u64 %npad;
    .reg .u64 %j;
    .reg .u64 %k;
    .reg .u64 %i;
    .reg .u64 %l;
    .reg .u64 %stride;
    .reg .u64 %band;
    .reg .u64 %ai8;
    .reg .u64 %al8;
    .reg .u64 %ai4;
    .reg .u64 %al4;
    .reg .u64 %addr;
    .reg .s64 %ki;
    .reg .s64 %kl;

    ld.param.u64 %kw, [keys_work_ptr];
    ld.param.u64 %iw, [idx_work_ptr];
    ld.param.u64 %npad, [n_pad];
    ld.param.u64 %j, [j];
    ld.param.u64 %k, [k];
    ld.param.u32 %desc, [descending];

    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %tmp32, %bid, %bdim, %lane;
    cvt.u64.u32 %i, %tmp32;
    mul.lo.u32 %tmp32, %gdim, %bdim;
    cvt.u64.u32 %stride, %tmp32;

    setp.eq.u32 %p_desc, %desc, 1;
    not.pred %p_ndesc, %p_desc;

sloop:
    setp.ge.u64 %p_done, %i, %npad;
    @%p_done bra sdone;
    xor.b64 %l, %i, %j;
    setp.gt.u64 %p_pair, %l, %i;
    @!%p_pair bra snext;

    mul.lo.u64 %ai8, %i, 8;
    mul.lo.u64 %al8, %l, 8;
    mul.lo.u64 %ai4, %i, 4;
    mul.lo.u64 %al4, %l, 4;
    add.u64 %addr, %kw, %ai8;
    ld.global.s64 %ki, [%addr];
    add.u64 %addr, %kw, %al8;
    ld.global.s64 %kl, [%addr];
    add.u64 %addr, %iw, %ai4;
    ld.global.u32 %idxi, [%addr];
    add.u64 %addr, %iw, %al4;
    ld.global.u32 %idxl, [%addr];

    setp.gt.s64 %p_kgt, %ki, %kl;
    setp.lt.s64 %p_klt, %ki, %kl;
    setp.eq.s64 %p_keq, %ki, %kl;
    setp.gt.u32 %p_igt, %idxi, %idxl;
    setp.lt.u32 %p_ilt, %idxi, %idxl;
    and.pred %p_s1, %p_desc, %p_klt;
    and.pred %p_s2, %p_ndesc, %p_kgt;
    or.pred %p_ka, %p_s1, %p_s2;
    and.pred %p_s1, %p_desc, %p_kgt;
    and.pred %p_s2, %p_ndesc, %p_klt;
    or.pred %p_kb, %p_s1, %p_s2;
    and.pred %p_ta, %p_keq, %p_igt;
    or.pred %p_iaft, %p_ka, %p_ta;
    and.pred %p_tb, %p_keq, %p_ilt;
    or.pred %p_ibef, %p_kb, %p_tb;
    and.b64 %band, %i, %k;
    setp.eq.u64 %p_asc, %band, 0;
    not.pred %p_nasc, %p_asc;
    and.pred %p_s1, %p_asc, %p_iaft;
    and.pred %p_s2, %p_nasc, %p_ibef;
    or.pred %p_swap, %p_s1, %p_s2;
    @!%p_swap bra snext;
    add.u64 %addr, %kw, %ai8;
    st.global.s64 [%addr], %kl;
    add.u64 %addr, %kw, %al8;
    st.global.s64 [%addr], %ki;
    add.u64 %addr, %iw, %ai4;
    st.global.u32 [%addr], %idxl;
    add.u64 %addr, %iw, %al4;
    st.global.u32 [%addr], %idxi;

snext:
    add.u64 %i, %i, %stride;
    bra sloop;
sdone:
    ret;
}
"#;

    if n == 0 {
        return Ok(Vec::new());
    }
    // Argsort indices are materialized as u32 on-device (idx buffer = n_pad*4, `cvt.u32.u64`).
    // Guard the precondition so >4B rows fail loud instead of silently truncating the index.
    if n > u64::from(u32::MAX) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    // `keys_device_ptr` is an absolute device address; the caller owns its [ptr, ptr+n*8) sizing.
    let n_pad = n
        .checked_next_power_of_two()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let n_usize =
        usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let n_pad_usize = usize::try_from(n_pad)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let init_fn = resident
        .primary()
        .cached_function(c"gpu_db_bitonic_argsort_init", &ptx)?;
    let step_fn = resident
        .primary()
        .cached_function(c"gpu_db_bitonic_argsort_step", &ptx)?;

    let primary = resident.primary();
    let keys_work = primary.lease_device_buffer(
        n_pad_usize
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )?;
    let idx_work = primary.lease_device_buffer(
        n_pad_usize
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )?;

    primary.set_current()?;
    struct StreamLease<'a> {
        primary: &'a GpuPrimaryContext,
        pooled: Option<PooledStream>,
    }
    impl Drop for StreamLease<'_> {
        fn drop(&mut self) {
            if let Some(pooled) = self.pooled.take() {
                self.primary.release_pooled_stream(pooled);
            }
        }
    }
    let lease = StreamLease {
        primary,
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let stream = lease
        .pooled
        .as_ref()
        .expect("pooled stream just set")
        .stream;
    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        // SAFETY: best-effort drain on an already-failing path before the leases unwind.
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    const BLOCK: u32 = 256;
    let grid = n_pad.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;

    let mut src_arg = keys_device_ptr;
    let mut n_arg = n;
    let mut n_pad_arg = n_pad;
    let mut kw_arg = keys_work.ptr;
    let mut iw_arg = idx_work.ptr;
    let mut desc_arg: u32 = u32::from(descending);
    let mut init_args = [
        (&mut src_arg as *mut u64).cast::<c_void>(),
        (&mut n_arg as *mut u64).cast::<c_void>(),
        (&mut n_pad_arg as *mut u64).cast::<c_void>(),
        (&mut kw_arg as *mut u64).cast::<c_void>(),
        (&mut iw_arg as *mut u64).cast::<c_void>(),
        (&mut desc_arg as *mut u32).cast::<c_void>(),
    ];
    check_cuda(unsafe {
        cu_launch_kernel(
            init_fn,
            grid,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            init_args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })
    .map_err(drain_err)?;

    let mut k = 2_u64;
    while k <= n_pad {
        let mut j = k >> 1;
        while j > 0 {
            let mut j_arg = j;
            let mut k_arg = k;
            let mut step_args = [
                (&mut kw_arg as *mut u64).cast::<c_void>(),
                (&mut iw_arg as *mut u64).cast::<c_void>(),
                (&mut n_pad_arg as *mut u64).cast::<c_void>(),
                (&mut j_arg as *mut u64).cast::<c_void>(),
                (&mut k_arg as *mut u64).cast::<c_void>(),
                (&mut desc_arg as *mut u32).cast::<c_void>(),
            ];
            check_cuda(unsafe {
                cu_launch_kernel(
                    step_fn,
                    grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    stream,
                    step_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            })
            .map_err(drain_err)?;
            j >>= 1;
        }
        k <<= 1;
    }

    check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;
    resident.record_kernel_event_elapsed_us(None);

    let mut indices = vec![0_u32; n_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            indices.as_mut_ptr().cast::<c_void>(),
            idx_work.ptr,
            n_usize * std::mem::size_of::<u32>(),
        )
    })
    .map_err(drain_err)?;
    drop(lease);
    Ok(indices)
}

/// Parallel GPU LSD-radix argsort over a resident i64 key column — the LARGE-result arm of the
/// adaptive sort operator (S3). Returns a Vec<u32> permutation of 0..n ordering the rows by key
/// (ascending when `descending=false`, descending when true), STABLE (equal keys keep ascending
/// original index) in both directions. O(n), constant 16 LSD passes of 4 bits — beats the bitonic
/// arm's O(n log^2 n) launch count at large n.
///
/// Keys are mapped signed->unsigned-order by XOR with a direction mask (0x8000…0 ascending so i64
/// order == u64 order; 0x7FFF…F descending = the complement, so one ascending radix yields
/// descending keys with the SAME ascending-index tie-break). Each pass is three kernels on one
/// pooled stream: (1) `radix_histogram` — each block counts its contiguous chunk's 4-bit digits
/// into a bucket-major block_hist[d*G+b] via a shared per-digit histogram; (2) `radix_scan` —
/// exclusive prefix sum over the whole 16*G matrix, so block_hist[d*G+b] becomes the global output
/// offset where block b's digit-d run begins; (3) `radix_scatter` — each block STABLY scatters its
/// chunk to keys_dst/idx_dst at base + per-digit-running + within-block rank, the within-block rank
/// computed per grid-stride wave by `match.any.sync` warp ranking + a cross-warp per-digit combine,
/// with a per-digit running offset carried across waves. Ping-pong (keys/idx a<->b) over 16 passes.
#[allow(dead_code)] // wired into the engine grouped/projection ORDER BY path in S4/S5.
fn launch_cuda_resident_i64_argsort_radix(
    resident: &CudaResidentDeviceMemory,
    keys_device_ptr: u64,
    n: u64,
    descending: bool,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuLaunchKernel = unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32;

    const PTX: &[u8] = br#"
.version 6.3
.target sm_70
.address_size 64

.visible .entry gpu_db_radix_init(
    .param .u64 keys_src_ptr,
    .param .u64 n,
    .param .u64 mask,
    .param .u64 keys_a_ptr,
    .param .u64 idx_a_ptr
)
{
    .reg .pred %p;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %gdim;
    .reg .u32 %t32;
    .reg .u32 %idxv;
    .reg .u64 %src;
    .reg .u64 %n;
    .reg .u64 %mask;
    .reg .u64 %ka;
    .reg .u64 %ia;
    .reg .u64 %i;
    .reg .u64 %stride;
    .reg .u64 %o8;
    .reg .u64 %o4;
    .reg .u64 %addr;
    .reg .u64 %key;

    ld.param.u64 %src, [keys_src_ptr];
    ld.param.u64 %n, [n];
    ld.param.u64 %mask, [mask];
    ld.param.u64 %ka, [keys_a_ptr];
    ld.param.u64 %ia, [idx_a_ptr];
    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %t32, %bid, %bdim, %lane;
    cvt.u64.u32 %i, %t32;
    mul.lo.u32 %t32, %gdim, %bdim;
    cvt.u64.u32 %stride, %t32;
init_l:
    setp.ge.u64 %p, %i, %n;
    @%p bra init_d;
    mul.lo.u64 %o8, %i, 8;
    add.u64 %addr, %src, %o8;
    ld.global.u64 %key, [%addr];
    xor.b64 %key, %key, %mask;
    add.u64 %addr, %ka, %o8;
    st.global.u64 [%addr], %key;
    mul.lo.u64 %o4, %i, 4;
    add.u64 %addr, %ia, %o4;
    cvt.u32.u64 %idxv, %i;
    st.global.u32 [%addr], %idxv;
    add.u64 %i, %i, %stride;
    bra init_l;
init_d:
    ret;
}

.visible .entry gpu_db_radix_histogram(
    .param .u64 keys_ptr,
    .param .u64 n,
    .param .u32 shift,
    .param .u64 chunk,
    .param .u64 ndiv,
    .param .u64 block_hist_ptr
)
{
    .shared .align 4 .b32 s_hist[16];
    .reg .pred %p;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %shift;
    .reg .u32 %d;
    .reg .u32 %c;
    .reg .u32 %t32;
    .reg .u64 %keys;
    .reg .u64 %n;
    .reg .u64 %chunk;
    .reg .u64 %G;
    .reg .u64 %bh;
    .reg .u64 %bid64;
    .reg .u64 %start;
    .reg .u64 %end;
    .reg .u64 %row;
    .reg .u64 %iter;
    .reg .u64 %o8;
    .reg .u64 %o4;
    .reg .u64 %addr;
    .reg .u64 %key;
    .reg .u64 %dig;

    ld.param.u64 %keys, [keys_ptr];
    ld.param.u64 %n, [n];
    ld.param.u32 %shift, [shift];
    ld.param.u64 %chunk, [chunk];
    ld.param.u64 %G, [ndiv];
    ld.param.u64 %bh, [block_hist_ptr];
    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;

    setp.ge.u32 %p, %lane, 16;
    @%p bra zskip;
    mul.wide.u32 %o4, %lane, 4;
    mov.u64 %addr, s_hist;
    add.u64 %addr, %addr, %o4;
    mov.u32 %t32, 0;
    st.shared.u32 [%addr], %t32;
zskip:
    bar.sync 0;

    cvt.u64.u32 %bid64, %bid;
    mul.lo.u64 %start, %bid64, %chunk;
    add.u64 %end, %start, %chunk;
    setp.gt.u64 %p, %end, %n;
    @%p mov.u64 %end, %n;

    mov.u64 %iter, %start;
hloop:
    setp.ge.u64 %p, %iter, %end;
    @%p bra hdone;
    cvt.u64.u32 %o8, %lane;
    add.u64 %row, %iter, %o8;
    setp.ge.u64 %p, %row, %end;
    @%p bra hskip;
    mul.lo.u64 %o8, %row, 8;
    add.u64 %addr, %keys, %o8;
    ld.global.u64 %key, [%addr];
    shr.u64 %dig, %key, %shift;
    and.b64 %dig, %dig, 15;
    cvt.u32.u64 %d, %dig;
    mul.wide.u32 %o4, %d, 4;
    mov.u64 %addr, s_hist;
    add.u64 %addr, %addr, %o4;
    atom.shared.add.u32 %t32, [%addr], 1;
hskip:
    cvt.u64.u32 %o8, %bdim;
    add.u64 %iter, %iter, %o8;
    bra hloop;
hdone:
    bar.sync 0;

    setp.ge.u32 %p, %lane, 16;
    @%p bra wskip;
    mul.wide.u32 %o4, %lane, 4;
    mov.u64 %addr, s_hist;
    add.u64 %addr, %addr, %o4;
    ld.shared.u32 %c, [%addr];
    cvt.u64.u32 %o8, %lane;
    mul.lo.u64 %o8, %o8, %G;
    add.u64 %o8, %o8, %bid64;
    mul.lo.u64 %o8, %o8, 4;
    add.u64 %addr, %bh, %o8;
    st.global.u32 [%addr], %c;
wskip:
    ret;
}

// Single-block parallel exclusive prefix sum over block_hist[0..total): each tile of `bdim`
// elements is inclusive-scanned in shared memory (Hillis-Steele), written back as carry +
// (inclusive - own), and the running `carry` chains across tiles. Replaces the old single-thread
// serial scan (a sequential dependent-global-load bottleneck).
.visible .entry gpu_db_radix_scan(
    .param .u64 block_hist_ptr,
    .param .u64 total
)
{
    .shared .align 4 .b32 s[1024];
    .reg .pred %p;
    .reg .pred %p_in;
    .reg .u32 %thr;
    .reg .u32 %bdim;
    .reg .u32 %d;
    .reg .u32 %v;
    .reg .u32 %orig;
    .reg .u32 %recv;
    .reg .u32 %carry;
    .reg .u32 %tot;
    .reg .u32 %t32;
    .reg .u64 %bh;
    .reg .u64 %total;
    .reg .u64 %base;
    .reg .u64 %i;
    .reg .u64 %o4;
    .reg .u64 %addr;

    mov.u32 %thr, %tid.x;
    mov.u32 %bdim, %ntid.x;
    ld.param.u64 %bh, [block_hist_ptr];
    ld.param.u64 %total, [total];
    mov.u32 %carry, 0;
    mov.u64 %base, 0;
tile_loop:
    setp.ge.u64 %p, %base, %total;
    @%p bra tile_done;
    cvt.u64.u32 %o4, %thr;
    add.u64 %i, %base, %o4;
    setp.lt.u64 %p_in, %i, %total;
    mov.u32 %v, 0;
    @!%p_in bra loaded;
    mul.lo.u64 %o4, %i, 4;
    add.u64 %addr, %bh, %o4;
    ld.global.u32 %v, [%addr];
loaded:
    mov.u32 %orig, %v;
    mul.wide.u32 %o4, %thr, 4;
    mov.u64 %addr, s;
    add.u64 %addr, %addr, %o4;
    st.shared.u32 [%addr], %v;
    bar.sync 0;
    mov.u32 %d, 1;
hs_loop:
    setp.ge.u32 %p, %d, %bdim;
    @%p bra hs_done;
    mov.u32 %recv, 0;
    setp.lt.u32 %p, %thr, %d;
    @%p bra hs_noread;
    sub.u32 %t32, %thr, %d;
    mul.wide.u32 %o4, %t32, 4;
    mov.u64 %addr, s;
    add.u64 %addr, %addr, %o4;
    ld.shared.u32 %recv, [%addr];
hs_noread:
    bar.sync 0;
    setp.lt.u32 %p, %thr, %d;
    @%p bra hs_nowrite;
    mul.wide.u32 %o4, %thr, 4;
    mov.u64 %addr, s;
    add.u64 %addr, %addr, %o4;
    ld.shared.u32 %v, [%addr];
    add.u32 %v, %v, %recv;
    st.shared.u32 [%addr], %v;
hs_nowrite:
    bar.sync 0;
    shl.b32 %d, %d, 1;
    bra hs_loop;
hs_done:
    sub.u32 %t32, %bdim, 1;
    mul.wide.u32 %o4, %t32, 4;
    mov.u64 %addr, s;
    add.u64 %addr, %addr, %o4;
    ld.shared.u32 %tot, [%addr];
    @!%p_in bra skip_write;
    mul.wide.u32 %o4, %thr, 4;
    mov.u64 %addr, s;
    add.u64 %addr, %addr, %o4;
    ld.shared.u32 %v, [%addr];
    sub.u32 %v, %v, %orig;
    add.u32 %v, %v, %carry;
    mul.lo.u64 %o4, %i, 4;
    add.u64 %addr, %bh, %o4;
    st.global.u32 [%addr], %v;
skip_write:
    add.u32 %carry, %carry, %tot;
    bar.sync 0;
    cvt.u64.u32 %o4, %bdim;
    add.u64 %base, %base, %o4;
    bra tile_loop;
tile_done:
    ret;
}

.visible .entry gpu_db_radix_scatter(
    .param .u64 keys_src_ptr,
    .param .u64 idx_src_ptr,
    .param .u64 n,
    .param .u32 shift,
    .param .u64 chunk,
    .param .u64 ndiv,
    .param .u64 block_base_ptr,
    .param .u64 keys_dst_ptr,
    .param .u64 idx_dst_ptr
)
{
    .shared .align 4 .b32 s_wh[128];
    .shared .align 4 .b32 s_wb[128];
    .shared .align 4 .b32 s_run[16];
    .shared .align 4 .b32 s_tot[16];
    .reg .pred %p;
    .reg .pred %p_leader;
    .reg .pred %p_valid;
    .reg .pred %p_d16;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %warp;
    .reg .u32 %nwarps;
    .reg .u32 %lid;
    .reg .u32 %lmlt;
    .reg .u32 %shift;
    .reg .u32 %d;
    .reg .u32 %mask;
    .reg .u32 %tmp;
    .reg .u32 %wrank;
    .reg .u32 %wcount;
    .reg .u32 %run;
    .reg .u32 %c;
    .reg .u32 %base;
    .reg .u32 %t32;
    .reg .u32 %idxv;
    .reg .u32 %bid;
    .reg .u32 %w;
    .reg .u64 %ks;
    .reg .u64 %is;
    .reg .u64 %n;
    .reg .u64 %chunk;
    .reg .u64 %G;
    .reg .u64 %bb;
    .reg .u64 %kd;
    .reg .u64 %id;
    .reg .u64 %bid64;
    .reg .u64 %start;
    .reg .u64 %end;
    .reg .u64 %iter;
    .reg .u64 %row;
    .reg .u64 %stride;
    .reg .u64 %o8;
    .reg .u64 %o4;
    .reg .u64 %addr;
    .reg .u64 %key;
    .reg .u64 %dig;
    .reg .u64 %pos;
    .reg .u64 %digd;

    ld.param.u64 %ks, [keys_src_ptr];
    ld.param.u64 %is, [idx_src_ptr];
    ld.param.u64 %n, [n];
    ld.param.u32 %shift, [shift];
    ld.param.u64 %chunk, [chunk];
    ld.param.u64 %G, [ndiv];
    ld.param.u64 %bb, [block_base_ptr];
    ld.param.u64 %kd, [keys_dst_ptr];
    ld.param.u64 %id, [idx_dst_ptr];

    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %lid, %laneid;
    mov.u32 %lmlt, %lanemask_lt;
    shr.u32 %warp, %lane, 5;
    add.u32 %tmp, %bdim, 31;
    shr.u32 %nwarps, %tmp, 5;
    cvt.u64.u32 %bid64, %bid;

    setp.ge.u32 %p, %lane, 16;
    @%p bra runzskip;
    mul.wide.u32 %o4, %lane, 4;
    mov.u64 %addr, s_run;
    add.u64 %addr, %addr, %o4;
    mov.u32 %t32, 0;
    st.shared.u32 [%addr], %t32;
runzskip:
    bar.sync 0;

    mul.lo.u64 %start, %bid64, %chunk;
    add.u64 %end, %start, %chunk;
    setp.gt.u64 %p, %end, %n;
    @%p mov.u64 %end, %n;
    cvt.u64.u32 %stride, %bdim;
    mov.u64 %iter, %start;

wave_loop:
    setp.ge.u64 %p, %iter, %end;
    @%p bra wave_done;
    cvt.u64.u32 %o8, %lane;
    add.u64 %row, %iter, %o8;
    setp.lt.u64 %p_valid, %row, %end;

    mov.u32 %d, 16;
    @!%p_valid bra have_digit;
    mul.lo.u64 %o8, %row, 8;
    add.u64 %addr, %ks, %o8;
    ld.global.u64 %key, [%addr];
    shr.u64 %dig, %key, %shift;
    and.b64 %dig, %dig, 15;
    cvt.u32.u64 %d, %dig;
have_digit:

    setp.ge.u32 %p, %lid, 16;
    @%p bra whz_skip;
    mul.lo.u32 %tmp, %warp, 16;
    add.u32 %tmp, %tmp, %lid;
    mul.wide.u32 %o4, %tmp, 4;
    mov.u64 %addr, s_wh;
    add.u64 %addr, %addr, %o4;
    mov.u32 %t32, 0;
    st.shared.u32 [%addr], %t32;
whz_skip:
    bar.sync 0;

    match.any.sync.b32 %mask, %d, 0xffffffff;
    and.b32 %tmp, %mask, %lmlt;
    popc.b32 %wrank, %tmp;
    popc.b32 %wcount, %mask;
    setp.eq.u32 %p_leader, %wrank, 0;
    setp.lt.u32 %p_d16, %d, 16;
    and.pred %p_leader, %p_leader, %p_d16;
    @!%p_leader bra leader_skip;
    mul.lo.u32 %tmp, %warp, 16;
    add.u32 %tmp, %tmp, %d;
    mul.wide.u32 %o4, %tmp, 4;
    mov.u64 %addr, s_wh;
    add.u64 %addr, %addr, %o4;
    st.shared.u32 [%addr], %wcount;
leader_skip:
    bar.sync 0;

    setp.ge.u32 %p, %lane, 16;
    @%p bra combine_skip;
    mov.u32 %run, 0;
    mov.u32 %w, 0;
ccol_loop:
    setp.ge.u32 %p, %w, %nwarps;
    @%p bra ccol_done;
    mul.lo.u32 %t32, %w, 16;
    add.u32 %t32, %t32, %lane;
    mul.wide.u32 %o4, %t32, 4;
    mov.u64 %addr, s_wh;
    add.u64 %addr, %addr, %o4;
    ld.shared.u32 %c, [%addr];
    mov.u64 %addr, s_wb;
    add.u64 %addr, %addr, %o4;
    st.shared.u32 [%addr], %run;
    add.u32 %run, %run, %c;
    add.u32 %w, %w, 1;
    bra ccol_loop;
ccol_done:
    mul.wide.u32 %o4, %lane, 4;
    mov.u64 %addr, s_tot;
    add.u64 %addr, %addr, %o4;
    st.shared.u32 [%addr], %run;
combine_skip:
    bar.sync 0;

    @!%p_valid bra after_scatter;
    cvt.u64.u32 %digd, %d;
    mul.lo.u64 %o8, %digd, %G;
    add.u64 %o8, %o8, %bid64;
    mul.lo.u64 %o8, %o8, 4;
    add.u64 %addr, %bb, %o8;
    ld.global.u32 %base, [%addr];
    mul.wide.u32 %o4, %d, 4;
    mov.u64 %addr, s_run;
    add.u64 %addr, %addr, %o4;
    ld.shared.u32 %run, [%addr];
    mul.lo.u32 %tmp, %warp, 16;
    add.u32 %tmp, %tmp, %d;
    mul.wide.u32 %o4, %tmp, 4;
    mov.u64 %addr, s_wb;
    add.u64 %addr, %addr, %o4;
    ld.shared.u32 %c, [%addr];
    add.u32 %base, %base, %run;
    add.u32 %base, %base, %c;
    add.u32 %base, %base, %wrank;
    cvt.u64.u32 %pos, %base;
    mul.lo.u64 %o8, %pos, 8;
    add.u64 %addr, %kd, %o8;
    st.global.u64 [%addr], %key;
    mul.lo.u64 %o8, %row, 4;
    add.u64 %addr, %is, %o8;
    ld.global.u32 %idxv, [%addr];
    mul.lo.u64 %o8, %pos, 4;
    add.u64 %addr, %id, %o8;
    st.global.u32 [%addr], %idxv;
after_scatter:
    bar.sync 0;

    setp.ge.u32 %p, %lane, 16;
    @%p bra runupd_skip;
    mul.wide.u32 %o4, %lane, 4;
    mov.u64 %addr, s_tot;
    add.u64 %addr, %addr, %o4;
    ld.shared.u32 %c, [%addr];
    mov.u64 %addr, s_run;
    add.u64 %addr, %addr, %o4;
    ld.shared.u32 %run, [%addr];
    add.u32 %run, %run, %c;
    st.shared.u32 [%addr], %run;
runupd_skip:
    bar.sync 0;
    add.u64 %iter, %iter, %stride;
    bra wave_loop;
wave_done:
    ret;
}
"#;

    if n == 0 {
        return Ok(Vec::new());
    }
    if n > u64::from(u32::MAX) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    // `keys_device_ptr` is an absolute device address; the caller owns its [ptr, ptr+n*8) sizing.
    let n_usize =
        usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    // Contiguous partition: block `b` owns rows [b*chunk, min(b*chunk+chunk, n)). `chunk` is sized
    // so the block count G stays within the grid-x max (65_535) for any n.
    const BLOCK: u32 = 256;
    // The scatter kernel's shared arrays s_wh[128]/s_wb[128] are sized for nwarps = BLOCK/32 = 8
    // (128 = 16 digits * 8 warps), and the scan kernel's s[1024] + its 1024-thread launch assume
    // that block. Raising BLOCK without resizing those hard-coded PTX shared arrays would corrupt
    // shared memory, so pin it at compile time.
    const _: () = assert!(
        BLOCK == 256,
        "radix PTX shared-memory sizes are hard-coded for BLOCK=256"
    );
    const CHUNK_ROWS: u64 = 2_048;
    const MAX_GRID: u64 = 65_535;
    let chunk = CHUNK_ROWS.max(n.div_ceil(MAX_GRID));
    let grid_g_u64 = n.div_ceil(chunk);
    let grid_g = u32::try_from(grid_g_u64)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let hist_len = grid_g_u64
        .checked_mul(16)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let hist_len_usize = usize::try_from(hist_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let init_fn = resident
        .primary()
        .cached_function(c"gpu_db_radix_init", &ptx)?;
    let hist_fn = resident
        .primary()
        .cached_function(c"gpu_db_radix_histogram", &ptx)?;
    let scan_fn = resident
        .primary()
        .cached_function(c"gpu_db_radix_scan", &ptx)?;
    let scatter_fn = resident
        .primary()
        .cached_function(c"gpu_db_radix_scatter", &ptx)?;

    let primary = resident.primary();
    let bytes8 = n_usize
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes4 = n_usize
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let hist_bytes = hist_len_usize
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let keys_a = primary.lease_device_buffer(bytes8)?;
    let keys_b = primary.lease_device_buffer(bytes8)?;
    let idx_a = primary.lease_device_buffer(bytes4)?;
    let idx_b = primary.lease_device_buffer(bytes4)?;
    let block_hist = primary.lease_device_buffer(hist_bytes)?;

    primary.set_current()?;
    struct StreamLease<'a> {
        primary: &'a GpuPrimaryContext,
        pooled: Option<PooledStream>,
    }
    impl Drop for StreamLease<'_> {
        fn drop(&mut self) {
            if let Some(pooled) = self.pooled.take() {
                self.primary.release_pooled_stream(pooled);
            }
        }
    }
    let lease = StreamLease {
        primary,
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let stream = lease
        .pooled
        .as_ref()
        .expect("pooled stream just set")
        .stream;
    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    let src_keys_base = keys_device_ptr;
    let mask: u64 = if descending {
        0x7FFF_FFFF_FFFF_FFFF
    } else {
        0x8000_0000_0000_0000
    };

    // init: transform source keys into keys_a, idx_a = identity.
    {
        let mut src_arg = src_keys_base;
        let mut n_arg = n;
        let mut mask_arg = mask;
        let mut ka_arg = keys_a.ptr;
        let mut ia_arg = idx_a.ptr;
        let mut init_args = [
            (&mut src_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
            (&mut mask_arg as *mut u64).cast::<c_void>(),
            (&mut ka_arg as *mut u64).cast::<c_void>(),
            (&mut ia_arg as *mut u64).cast::<c_void>(),
        ];
        let init_grid = n.div_ceil(u64::from(BLOCK)).clamp(1, MAX_GRID) as u32;
        check_cuda(unsafe {
            cu_launch_kernel(
                init_fn,
                init_grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                init_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;
    }

    // 16 LSD passes of 4 bits, ping-ponging (keys/idx) a<->b.
    let mut keys_src = keys_a.ptr;
    let mut keys_dst = keys_b.ptr;
    let mut idx_src = idx_a.ptr;
    let mut idx_dst = idx_b.ptr;
    for pass in 0u32..16 {
        let mut shift_arg = pass * 4;
        let mut n_arg = n;
        let mut chunk_arg = chunk;
        let mut g_arg = grid_g_u64;
        let mut bh_arg = block_hist.ptr;
        let mut total_arg = hist_len;

        // histogram (reads keys_src)
        let mut ksrc_arg = keys_src;
        let mut hist_args = [
            (&mut ksrc_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
            (&mut shift_arg as *mut u32).cast::<c_void>(),
            (&mut chunk_arg as *mut u64).cast::<c_void>(),
            (&mut g_arg as *mut u64).cast::<c_void>(),
            (&mut bh_arg as *mut u64).cast::<c_void>(),
        ];
        check_cuda(unsafe {
            cu_launch_kernel(
                hist_fn,
                grid_g,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                hist_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;

        // scan (single block, exclusive prefix over the 16*G matrix)
        let mut scan_args = [
            (&mut bh_arg as *mut u64).cast::<c_void>(),
            (&mut total_arg as *mut u64).cast::<c_void>(),
        ];
        check_cuda(unsafe {
            cu_launch_kernel(
                scan_fn,
                1,
                1,
                1,
                1_024,
                1,
                1,
                0,
                stream,
                scan_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;

        // scatter (reads keys_src/idx_src, writes keys_dst/idx_dst)
        let mut isrc_arg = idx_src;
        let mut kdst_arg = keys_dst;
        let mut idst_arg = idx_dst;
        let mut scatter_args = [
            (&mut ksrc_arg as *mut u64).cast::<c_void>(),
            (&mut isrc_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
            (&mut shift_arg as *mut u32).cast::<c_void>(),
            (&mut chunk_arg as *mut u64).cast::<c_void>(),
            (&mut g_arg as *mut u64).cast::<c_void>(),
            (&mut bh_arg as *mut u64).cast::<c_void>(),
            (&mut kdst_arg as *mut u64).cast::<c_void>(),
            (&mut idst_arg as *mut u64).cast::<c_void>(),
        ];
        check_cuda(unsafe {
            cu_launch_kernel(
                scatter_fn,
                grid_g,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                scatter_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;

        std::mem::swap(&mut keys_src, &mut keys_dst);
        std::mem::swap(&mut idx_src, &mut idx_dst);
    }

    check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;
    resident.record_kernel_event_elapsed_us(None);

    // After 16 (even) passes the final result is in the buffer `idx_src` now points at.
    let mut indices = vec![0_u32; n_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            indices.as_mut_ptr().cast::<c_void>(),
            idx_src,
            n_usize * std::mem::size_of::<u32>(),
        )
    })
    .map_err(drain_err)?;
    drop(lease);
    Ok(indices)
}

/// Benchmark-calibrated bitonic<->radix crossover for the adaptive GPU argsort (S2/S3, measured on
/// RTX PRO 6000 Blackwell): at ~10k rows the two arms tie (radix 0.99x bitonic); below it bitonic's
/// single-kernel-per-stage simplicity wins, above it radix's O(n) beats bitonic's O(n log^2 n)
/// launch count and the gap widens (radix 1.27x @100k -> 3.10x @10M).
const ADAPTIVE_SORT_CROSSOVER_ROWS: u64 = 10_000;

/// Adaptive GPU argsort over a resident i64 key column — the unified ORDER BY sort primitive (S4).
/// Dispatches to the bitonic arm (S2) below the crossover and the radix arm (S3) at/above it, so the
/// faster algorithm runs for the result size. Returns a stable Vec<u32> permutation ordering rows by
/// key (ascending when `descending=false`, else descending; equal keys keep ascending original index
/// in both directions). Both arms produce byte-identical permutations, so the choice is invisible to
/// callers — the operator's result is deterministic regardless of which arm runs.
#[allow(dead_code)] // wired into the engine grouped/projection ORDER BY path in S5.
fn launch_cuda_resident_i64_argsort_adaptive(
    resident: &CudaResidentDeviceMemory,
    keys_device_ptr: u64,
    n: u64,
    descending: bool,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    if n < ADAPTIVE_SORT_CROSSOVER_ROWS {
        launch_cuda_resident_i64_argsort_bitonic(resident, keys_device_ptr, n, descending)
    } else {
        launch_cuda_resident_i64_argsort_radix(resident, keys_device_ptr, n, descending)
    }
}

/// GPU ORDER BY ... LIMIT/OFFSET operator (S4): adaptive-argsort the resident i64 key column, then
/// take the ordered window `[offset, offset+limit)` of original row indices. This is the complete,
/// correct LIMIT/OFFSET primitive — the sort (the GPU-worthy work) runs on-device; the window is a
/// host slice of the returned permutation. `offset`/`limit` are clamped to the available rows;
/// `limit = None` returns everything from `offset`.
///
/// NOTE: for a LARGE result with a SMALL limit this still does a full sort. A partial top-K
/// (block-local top-K + merge, or radix-select) that avoids the full sort is a tracked perf
/// follow-up — deferred until S5 wires a large-result ORDER BY caller so the optimization can be
/// benchmark-justified rather than built speculatively.
#[allow(dead_code)] // wired into the engine ORDER BY/LIMIT path in S5.
fn launch_cuda_resident_i64_order_by_limit(
    resident: &CudaResidentDeviceMemory,
    keys_device_ptr: u64,
    n: u64,
    descending: bool,
    offset: u64,
    limit: Option<u64>,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    let perm = launch_cuda_resident_i64_argsort_adaptive(resident, keys_device_ptr, n, descending)?;
    let start = usize::try_from(offset)
        .unwrap_or(usize::MAX)
        .min(perm.len());
    let end = match limit {
        Some(lim) => start
            .saturating_add(usize::try_from(lim).unwrap_or(usize::MAX))
            .min(perm.len()),
        None => perm.len(),
    };
    Ok(perm[start..end].to_vec())
}

/// Serial single-thread GPU LSD-radix argsort — the GPU-native parity ORACLE + benchmark
/// baseline for the parallel `launch_cuda_resident_i64_argsort_radix` (S3). One device thread
/// runs a textbook stable counting sort: 16 LSD passes of 4 bits each over a signed→unsigned
/// key transform (XOR `mask`: 0x8000…0 ascending so i64 order == u64 order; 0x7FFF…F descending
/// = the complement, so one ascending radix yields descending keys with the SAME ascending-index
/// tie-break — stable in both directions). Obviously correct + stable (serial in-order scatter),
/// so the parallel version is validated against THIS (same algorithm → isolates parallelization
/// bugs) as well as the independently-verified bitonic arm. Test-only; never on the hot path.
#[cfg(test)]
fn launch_cuda_resident_i64_argsort_radix_serial(
    resident: &CudaResidentDeviceMemory,
    keys_device_ptr: u64,
    n: u64,
    descending: bool,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuLaunchKernel = unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_radix_argsort_serial(
    .param .u64 keys_src_ptr,
    .param .u64 n,
    .param .u64 mask,
    .param .u64 keys_a_ptr,
    .param .u64 keys_b_ptr,
    .param .u64 idx_a_ptr,
    .param .u64 idx_b_ptr,
    .param .u64 hist_ptr
)
{
    .reg .pred %p;
    .reg .u32 %thr;
    .reg .u32 %bid;
    .reg .u32 %lane32;
    .reg .u32 %pass;
    .reg .u32 %shift;
    .reg .u32 %d;
    .reg .u32 %cnt;
    .reg .u32 %run;
    .reg .u32 %tmp;
    .reg .u32 %idxv;
    .reg .u32 %pos;
    .reg .u64 %src;
    .reg .u64 %n;
    .reg .u64 %mask;
    .reg .u64 %ka;
    .reg .u64 %kb;
    .reg .u64 %ia;
    .reg .u64 %ib;
    .reg .u64 %hist;
    .reg .u64 %i;
    .reg .u64 %o8;
    .reg .u64 %o4;
    .reg .u64 %addr;
    .reg .u64 %key;
    .reg .u64 %dig;

    // Single-thread oracle: only (block 0, thread 0) executes; the rest return.
    mov.u32 %thr, %tid.x;
    mov.u32 %bid, %ctaid.x;
    or.b32 %lane32, %thr, %bid;
    setp.ne.u32 %p, %lane32, 0;
    @%p bra done;

    ld.param.u64 %src, [keys_src_ptr];
    ld.param.u64 %n, [n];
    ld.param.u64 %mask, [mask];
    ld.param.u64 %ka, [keys_a_ptr];
    ld.param.u64 %kb, [keys_b_ptr];
    ld.param.u64 %ia, [idx_a_ptr];
    ld.param.u64 %ib, [idx_b_ptr];
    ld.param.u64 %hist, [hist_ptr];

    // init: keys_a[i] = key[i] XOR mask ; idx_a[i] = i
    mov.u64 %i, 0;
init_loop:
    setp.ge.u64 %p, %i, %n;
    @%p bra init_done;
    mul.lo.u64 %o8, %i, 8;
    add.u64 %addr, %src, %o8;
    ld.global.u64 %key, [%addr];
    xor.b64 %key, %key, %mask;
    add.u64 %addr, %ka, %o8;
    st.global.u64 [%addr], %key;
    mul.lo.u64 %o4, %i, 4;
    add.u64 %addr, %ia, %o4;
    cvt.u32.u64 %idxv, %i;
    st.global.u32 [%addr], %idxv;
    add.u64 %i, %i, 1;
    bra init_loop;
init_done:

    mov.u32 %pass, 0;
pass_loop:
    setp.ge.u32 %p, %pass, 16;
    @%p bra pass_done;
    mul.lo.u32 %shift, %pass, 4;

    // zero hist[0..16)
    mov.u32 %d, 0;
hz_loop:
    setp.ge.u32 %p, %d, 16;
    @%p bra hz_done;
    mul.wide.u32 %o4, %d, 4;
    add.u64 %addr, %hist, %o4;
    mov.u32 %tmp, 0;
    st.global.u32 [%addr], %tmp;
    add.u32 %d, %d, 1;
    bra hz_loop;
hz_done:

    // count: hist[digit(keys_a[i])]++
    mov.u64 %i, 0;
count_loop:
    setp.ge.u64 %p, %i, %n;
    @%p bra count_done;
    mul.lo.u64 %o8, %i, 8;
    add.u64 %addr, %ka, %o8;
    ld.global.u64 %key, [%addr];
    shr.u64 %dig, %key, %shift;
    and.b64 %dig, %dig, 15;
    cvt.u32.u64 %d, %dig;
    mul.wide.u32 %o4, %d, 4;
    add.u64 %addr, %hist, %o4;
    ld.global.u32 %cnt, [%addr];
    add.u32 %cnt, %cnt, 1;
    st.global.u32 [%addr], %cnt;
    add.u64 %i, %i, 1;
    bra count_loop;
count_done:

    // exclusive scan: run=0; for d: t=hist[d]; hist[d]=run; run+=t
    mov.u32 %run, 0;
    mov.u32 %d, 0;
scan_loop:
    setp.ge.u32 %p, %d, 16;
    @%p bra scan_done;
    mul.wide.u32 %o4, %d, 4;
    add.u64 %addr, %hist, %o4;
    ld.global.u32 %cnt, [%addr];
    st.global.u32 [%addr], %run;
    add.u32 %run, %run, %cnt;
    add.u32 %d, %d, 1;
    bra scan_loop;
scan_done:

    // stable scatter (in input order): pos = hist[d]++ ; keys_b[pos]=key ; idx_b[pos]=idx_a[i]
    mov.u64 %i, 0;
scatter_loop:
    setp.ge.u64 %p, %i, %n;
    @%p bra scatter_done;
    mul.lo.u64 %o8, %i, 8;
    add.u64 %addr, %ka, %o8;
    ld.global.u64 %key, [%addr];
    shr.u64 %dig, %key, %shift;
    and.b64 %dig, %dig, 15;
    cvt.u32.u64 %d, %dig;
    mul.wide.u32 %o4, %d, 4;
    add.u64 %addr, %hist, %o4;
    ld.global.u32 %pos, [%addr];
    add.u32 %tmp, %pos, 1;
    st.global.u32 [%addr], %tmp;
    // keys_b[pos] = key
    mul.wide.u32 %o8, %pos, 8;
    add.u64 %addr, %kb, %o8;
    st.global.u64 [%addr], %key;
    // idx_b[pos] = idx_a[i]
    mul.lo.u64 %o4, %i, 4;
    add.u64 %addr, %ia, %o4;
    ld.global.u32 %idxv, [%addr];
    mul.wide.u32 %o4, %pos, 4;
    add.u64 %addr, %ib, %o4;
    st.global.u32 [%addr], %idxv;
    add.u64 %i, %i, 1;
    bra scatter_loop;
scatter_done:

    // copy b -> a (keys + idx) so the next pass reads from a
    mov.u64 %i, 0;
copy_loop:
    setp.ge.u64 %p, %i, %n;
    @%p bra copy_done;
    mul.lo.u64 %o8, %i, 8;
    add.u64 %addr, %kb, %o8;
    ld.global.u64 %key, [%addr];
    add.u64 %addr, %ka, %o8;
    st.global.u64 [%addr], %key;
    mul.lo.u64 %o4, %i, 4;
    add.u64 %addr, %ib, %o4;
    ld.global.u32 %idxv, [%addr];
    add.u64 %addr, %ia, %o4;
    st.global.u32 [%addr], %idxv;
    add.u64 %i, %i, 1;
    bra copy_loop;
copy_done:

    add.u32 %pass, %pass, 1;
    bra pass_loop;
pass_done:
done:
    ret;
}
"#;

    if n == 0 {
        return Ok(Vec::new());
    }
    if n > u64::from(u32::MAX) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    // `keys_device_ptr` is an absolute device address; the caller owns its [ptr, ptr+n*8) sizing.
    let n_usize =
        usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = resident
        .primary()
        .cached_function(c"gpu_db_radix_argsort_serial", &ptx)?;

    let primary = resident.primary();
    let bytes8 = n_usize
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes4 = n_usize
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let keys_a = primary.lease_device_buffer(bytes8)?;
    let keys_b = primary.lease_device_buffer(bytes8)?;
    let idx_a = primary.lease_device_buffer(bytes4)?;
    let idx_b = primary.lease_device_buffer(bytes4)?;
    let hist = primary.lease_device_buffer(16 * std::mem::size_of::<u32>())?;

    primary.set_current()?;
    struct StreamLease<'a> {
        primary: &'a GpuPrimaryContext,
        pooled: Option<PooledStream>,
    }
    impl Drop for StreamLease<'_> {
        fn drop(&mut self) {
            if let Some(pooled) = self.pooled.take() {
                self.primary.release_pooled_stream(pooled);
            }
        }
    }
    let lease = StreamLease {
        primary,
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let stream = lease
        .pooled
        .as_ref()
        .expect("pooled stream just set")
        .stream;
    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    let mut src_arg = keys_device_ptr;
    let mut n_arg = n;
    let mut mask_arg: u64 = if descending {
        0x7FFF_FFFF_FFFF_FFFF
    } else {
        0x8000_0000_0000_0000
    };
    let mut ka_arg = keys_a.ptr;
    let mut kb_arg = keys_b.ptr;
    let mut ia_arg = idx_a.ptr;
    let mut ib_arg = idx_b.ptr;
    let mut hist_arg = hist.ptr;
    let mut args = [
        (&mut src_arg as *mut u64).cast::<c_void>(),
        (&mut n_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut ka_arg as *mut u64).cast::<c_void>(),
        (&mut kb_arg as *mut u64).cast::<c_void>(),
        (&mut ia_arg as *mut u64).cast::<c_void>(),
        (&mut ib_arg as *mut u64).cast::<c_void>(),
        (&mut hist_arg as *mut u64).cast::<c_void>(),
    ];
    check_cuda(unsafe {
        cu_launch_kernel(
            kernel_fn,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })
    .map_err(drain_err)?;
    check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;
    resident.record_kernel_event_elapsed_us(None);

    let mut indices = vec![0_u32; n_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            indices.as_mut_ptr().cast::<c_void>(),
            idx_a.ptr,
            n_usize * std::mem::size_of::<u32>(),
        )
    })
    .map_err(drain_err)?;
    drop(lease);
    Ok(indices)
}

/// GPU HAVING filter over a grouped result (S4) — evaluates a DNF predicate (OR of AND-clauses)
/// per group row on the resident i64 group-value and i64 aggregate columns, returning the surviving
/// ORIGINAL row indices in ascending order. A filter is `(col, op, val)`: `col` 0 selects the group
/// value, 1 the aggregate; `op` is 0=Eq 1=Lt 2=Lte 3=Gt 4=Gte (matching the engine's numeric
/// `select_filter_matches` over i64); `val` is the i64 constant. A clause matches when ALL its
/// filters hold (vacuous AND, an empty clause, = match); the row survives when ANY clause matches.
///
/// CONTRACT for S5 wiring: empty `clauses` => NO survivors (vacuous OR). This deliberately DIFFERS
/// from SQL's *absent* HAVING (which means ALL rows, per the engine's `grouped_row_matches_having`),
/// so the caller MUST keep this kernel behind the engine's existing `!having_groups.is_empty()`
/// guard (or skip the kernel and pass everything through when HAVING is absent).
///
/// Single-block fused kernel: each thread evaluates the DNF for its grid-stride rows -> keep flag,
/// then the block ordered-compacts survivors ascending via a two-level warp-shuffle prefix sum with
/// a running offset carried across waves (same stable-compaction backbone as compare_scatter_blocks).
/// The DNF is uploaded as flat device arrays. Sized for the (small-to-moderate) grouped result; a
/// multi-block variant for very high group cardinality is a tracked follow-up.
#[allow(dead_code)] // wired into the engine grouped HAVING path in S5.
fn launch_cuda_resident_having_filter(
    resident: &CudaResidentDeviceMemory,
    group_device_ptr: u64,
    agg_device_ptr: u64,
    n: u64,
    clauses: &[Vec<(u32, u32, i64)>],
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuLaunchKernel = unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32;

    const PTX: &[u8] = include_bytes!("having.ptx");

    if n == 0 {
        return Ok(Vec::new());
    }
    if n > u64::from(u32::MAX) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    // `group_device_ptr`/`agg_device_ptr` are absolute device addresses (i64 columns); the caller
    // owns their [ptr, ptr+n*8) sizing.
    let n_usize =
        usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    // Encode the DNF as flat arrays: clause_off[c..c+1] bounds clause c's filters in col/op/val.
    let n_clauses = clauses.len() as u64;
    let mut clause_off: Vec<u32> = Vec::with_capacity(clauses.len() + 1);
    let mut filter_col: Vec<u32> = Vec::new();
    let mut filter_op: Vec<u32> = Vec::new();
    let mut filter_val: Vec<i64> = Vec::new();
    let mut acc: u32 = 0;
    clause_off.push(0);
    for clause in clauses {
        for &(col, op, val) in clause {
            filter_col.push(col);
            filter_op.push(op);
            filter_val.push(val);
        }
        acc = acc
            .checked_add(u32::try_from(clause.len()).unwrap_or(u32::MAX))
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        clause_off.push(acc);
    }

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = resident
        .primary()
        .cached_function(c"gpu_db_having_filter", &ptx)?;

    let primary = resident.primary();
    // Lease + upload a host slice into a device buffer; pads zero-length to a 1-element lease so the
    // (unread) device pointer is always valid.
    let upload_u32 = |data: &[u32]| {
        let buf = primary.lease_device_buffer(data.len().max(1) * std::mem::size_of::<u32>())?;
        if !data.is_empty() {
            check_cuda(unsafe {
                cu_memcpy_htod(
                    buf.ptr,
                    data.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(data),
                )
            })?;
        }
        Ok::<_, CudaRuntimeProbeError>(buf)
    };
    let clause_off_buf = upload_u32(&clause_off)?;
    let filter_col_buf = upload_u32(&filter_col)?;
    let filter_op_buf = upload_u32(&filter_op)?;
    let filter_val_buf =
        primary.lease_device_buffer(filter_val.len().max(1) * std::mem::size_of::<i64>())?;
    if !filter_val.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                filter_val_buf.ptr,
                filter_val.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(filter_val.as_slice()),
            )
        })?;
    }
    let out_idx = primary.lease_device_buffer(
        n_usize
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )?;
    let out_count = primary.lease_device_buffer(std::mem::size_of::<u32>())?;

    primary.set_current()?;
    struct StreamLease<'a> {
        primary: &'a GpuPrimaryContext,
        pooled: Option<PooledStream>,
    }
    impl Drop for StreamLease<'_> {
        fn drop(&mut self) {
            if let Some(pooled) = self.pooled.take() {
                self.primary.release_pooled_stream(pooled);
            }
        }
    }
    let lease = StreamLease {
        primary,
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let stream = lease
        .pooled
        .as_ref()
        .expect("pooled stream just set")
        .stream;
    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    const BLOCK: u32 = 256;
    let mut group_arg = group_device_ptr;
    let mut agg_arg = agg_device_ptr;
    let mut n_arg = n;
    let mut nclauses_arg = n_clauses;
    let mut coff_arg = clause_off_buf.ptr;
    let mut fcol_arg = filter_col_buf.ptr;
    let mut fop_arg = filter_op_buf.ptr;
    let mut fval_arg = filter_val_buf.ptr;
    let mut outidx_arg = out_idx.ptr;
    let mut outcnt_arg = out_count.ptr;
    let mut args = [
        (&mut group_arg as *mut u64).cast::<c_void>(),
        (&mut agg_arg as *mut u64).cast::<c_void>(),
        (&mut n_arg as *mut u64).cast::<c_void>(),
        (&mut nclauses_arg as *mut u64).cast::<c_void>(),
        (&mut coff_arg as *mut u64).cast::<c_void>(),
        (&mut fcol_arg as *mut u64).cast::<c_void>(),
        (&mut fop_arg as *mut u64).cast::<c_void>(),
        (&mut fval_arg as *mut u64).cast::<c_void>(),
        (&mut outidx_arg as *mut u64).cast::<c_void>(),
        (&mut outcnt_arg as *mut u64).cast::<c_void>(),
    ];
    check_cuda(unsafe {
        cu_launch_kernel(
            kernel_fn,
            1,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })
    .map_err(drain_err)?;
    check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;
    resident.record_kernel_event_elapsed_us(None);

    let mut count: u32 = 0;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut count as *mut u32).cast::<c_void>(),
            out_count.ptr,
            std::mem::size_of::<u32>(),
        )
    })
    .map_err(drain_err)?;
    let count_usize = count as usize;
    if count_usize > n_usize {
        return Err(drain_err(CudaRuntimeProbeError::InvalidInputLength(
            count_usize,
        )));
    }
    let mut indices = vec![0_u32; count_usize];
    if count_usize > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                indices.as_mut_ptr().cast::<c_void>(),
                out_idx.ptr,
                count_usize * std::mem::size_of::<u32>(),
            )
        })
        .map_err(drain_err)?;
    }
    drop(lease);
    Ok(indices)
}

#[cfg(test)]
fn launch_with_optional_cuda_event_timing<R, F>(
    resident: &R,
    cu_ctx_synchronize: unsafe extern "C" fn() -> i32,
    launch: F,
) -> Result<(), CudaRuntimeProbeError>
where
    R: CudaResidentReadSource,
    F: FnOnce() -> i32,
{
    type CuEventCreate = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
    type CuEventDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuEventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
    type CuEventSynchronize = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuEventElapsedTime = unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> i32;

    let event_symbols = unsafe {
        let create = resident.lib().get::<CuEventCreate>(b"cuEventCreate\0");
        let destroy = resident
            .lib()
            .get::<CuEventDestroy>(b"cuEventDestroy_v2\0")
            .or_else(|_| resident.lib().get::<CuEventDestroy>(b"cuEventDestroy\0"));
        let record = resident.lib().get::<CuEventRecord>(b"cuEventRecord\0");
        let synchronize = resident
            .lib()
            .get::<CuEventSynchronize>(b"cuEventSynchronize\0");
        let elapsed = resident
            .lib()
            .get::<CuEventElapsedTime>(b"cuEventElapsedTime\0");
        match (create, destroy, record, synchronize, elapsed) {
            (Ok(create), Ok(destroy), Ok(record), Ok(synchronize), Ok(elapsed)) => {
                Some((*create, *destroy, *record, *synchronize, *elapsed))
            }
            _ => None,
        }
    };

    if let Some((
        cu_event_create,
        cu_event_destroy,
        cu_event_record,
        cu_event_synchronize,
        cu_event_elapsed_time,
    )) = event_symbols
    {
        let mut start = std::ptr::null_mut();
        check_cuda(unsafe { cu_event_create(&mut start, 0) })?;
        let start_guard = CudaEventGuard {
            event: start,
            destroy: cu_event_destroy,
        };
        let mut stop = std::ptr::null_mut();
        check_cuda(unsafe { cu_event_create(&mut stop, 0) })?;
        let stop_guard = CudaEventGuard {
            event: stop,
            destroy: cu_event_destroy,
        };

        check_cuda(unsafe { cu_event_record(start_guard.event, std::ptr::null_mut()) })?;
        check_cuda(launch())?;
        check_cuda(unsafe { cu_event_record(stop_guard.event, std::ptr::null_mut()) })?;
        check_cuda(unsafe { cu_event_synchronize(stop_guard.event) })?;

        let mut elapsed_ms = 0.0_f32;
        check_cuda(unsafe {
            cu_event_elapsed_time(&mut elapsed_ms, start_guard.event, stop_guard.event)
        })?;
        resident
            .record_kernel_event_elapsed_us(Some((f64::from(elapsed_ms) * 1_000.0).ceil() as u64));
        return Ok(());
    }

    check_cuda(launch())?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    resident.record_kernel_event_elapsed_us(None);
    Ok(())
}

/// Run a kernel launch on a **pooled private stream** synced individually
/// (`cuStreamSynchronize`, not a whole-context `cuCtxSynchronize`) with CUDA-event timing —
/// the concurrency-friendly launch path (P2-M1). The closure receives `(stream,
/// pooled_scratch_ptr)`.
///
/// `scratch_out`:
/// - `Some(buf)` — scalar routes (e.g. COUNT): the kernel writes its small result to the
///   pooled scratch (`buf.len() <= POOLED_STREAM_SCRATCH_BYTES`), and it is copied back here
///   after the stream syncs — no per-call output `cuMemAlloc`/`cuEvent*`.
/// - `None` — multi-buffer routes (projection/gather) that manage their own large device
///   output buffers; they ignore the scratch ptr and do their own D2H after this returns
///   (the stream is already synced, so the data is ready). They still get the cached-module +
///   pooled-stream + per-stream-sync win.
fn launch_on_pooled_stream<R, F>(
    resident: &R,
    scratch_out: Option<&mut [u8]>,
    launch: F,
) -> Result<(), CudaRuntimeProbeError>
where
    R: CudaResidentReadSource,
    F: FnOnce(*mut c_void, u64) -> i32,
{
    let primary = resident.primary();
    // Bind the shared primary context on this thread before creating/using the stream, so
    // the migrated path is self-contained (a reader thread that hasn't bound yet would
    // otherwise hit INVALID_CONTEXT). Idempotent + cheap with one shared context.
    primary.set_current()?;
    if let Some(buf) = &scratch_out {
        assert!(
            buf.len() <= POOLED_STREAM_SCRATCH_BYTES,
            "scratch output ({}) exceeds pooled stream scratch ({POOLED_STREAM_SCRATCH_BYTES})",
            buf.len()
        );
    }

    // RAII: return the pooled stream (and its scratch) on every exit (success/error/panic).
    struct StreamLease<'a> {
        primary: &'a GpuPrimaryContext,
        pooled: Option<PooledStream>,
    }
    impl Drop for StreamLease<'_> {
        fn drop(&mut self) {
            if let Some(pooled) = self.pooled.take() {
                self.primary.release_pooled_stream(pooled);
            }
        }
    }
    let lease = StreamLease {
        primary,
        pooled: Some(primary.acquire_pooled_stream()?),
    };
    let pooled = lease.pooled.as_ref().expect("pooled stream just set");
    let stream = pooled.stream;
    let timed = !pooled.start_event.is_null() && !pooled.stop_event.is_null();

    // Centralized error-path stream drain for ALL callers of this helper. Each step below ENQUEUES
    // work on `stream` (start-event record, the caller's `launch` closure — kernel and possibly
    // memset/HtoD — the stop-event record). An early `?` from any of them would return `Err` with
    // that work still in flight, then unwind the caller's pooled device/pinned leases — returning
    // their buffers to the shared pool while the GPU is still reading/writing them (a cross-thread
    // use-after-free for the next leaser). `drain_err` blocking-syncs the stream FIRST (before the
    // error propagates and any caller lease Drops), then yields the original error. This gives the
    // un-migrated direct callers (row_count / equal_count / equal_project) and the migrated routes'
    // blocking fallbacks the same drain-before-release guarantee the per-op `drain_err` callers
    // already have. Success-path cost is zero (`map_err` skips the closure on `Ok`, so the single
    // covering sync below stays the only success-path sync — no redundant drain).
    let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
        // SAFETY: `stream` is the live pooled stream and the shared primary context is current
        // (bound above via `set_current`). The result is intentionally ignored — best-effort drain
        // on an already-failing path, mirroring the per-route `drain_err` style.
        unsafe {
            let _ = (primary.cu_stream_synchronize)(stream);
        }
        err
    };

    if timed {
        check_cuda(unsafe { (primary.cu_event_record)(pooled.start_event, stream) })
            .map_err(drain_err)?;
    }
    check_cuda(launch(stream, pooled.output)).map_err(drain_err)?;
    if timed {
        check_cuda(unsafe { (primary.cu_event_record)(pooled.stop_event, stream) })
            .map_err(drain_err)?;
    }
    // Covering sync: drains on its own error too (in-flight work above may remain if the sync call
    // itself failed). Past this point the stream is already drained, so the post-sync steps keep a
    // plain `?` — adding a drain there would be a redundant no-op on an idle stream.
    check_cuda(unsafe { (primary.cu_stream_synchronize)(stream) }).map_err(drain_err)?;

    if timed {
        let mut elapsed_ms = 0.0_f32;
        check_cuda(unsafe {
            (primary.cu_event_elapsed_time)(&mut elapsed_ms, pooled.start_event, pooled.stop_event)
        })?;
        resident
            .record_kernel_event_elapsed_us(Some((f64::from(elapsed_ms) * 1_000.0).ceil() as u64));
    } else {
        resident.record_kernel_event_elapsed_us(None);
    }

    if let Some(buf) = scratch_out {
        check_cuda(unsafe {
            (primary.cu_memcpy_dtoh)(buf.as_mut_ptr().cast::<c_void>(), pooled.output, buf.len())
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
