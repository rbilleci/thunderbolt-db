#[cfg(test)]
use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::Arc;

#[cfg(test)]
use libloading::Library;
pub mod probe;
pub use probe::Probe;
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

mod resident_memory;
use resident_memory::CudaResidentReadSource;
pub use resident_memory::{
    CudaDeviceMemoryChunk, CudaOwnedDeviceMemoryChunk, CudaResidentDeviceMemory,
    CudaResidentDeviceMemoryReadView, RecompactFill, RecompactSegment,
};
mod resident_header;
use resident_header::launch_cuda_resident_row_count;
mod resident_index_build;
mod resident_sort;
use resident_sort::{
    launch_cuda_bitonic_sort_hetero, launch_cuda_bitonic_sort_i64,
    launch_cuda_bitonic_sort_multikey, launch_cuda_bitonic_sort_text,
    launch_cuda_order_by_sort_i64,
};
#[cfg(test)]
use resident_sort::{
    launch_cuda_order_by_sort_i64_radix, validate_i64_argsort_host_len, validate_i64_argsort_input,
};
#[cfg(test)]
mod argsort_test_support;
#[cfg(test)]
use argsort_test_support::{
    launch_cuda_resident_i64_argsort_bitonic, launch_cuda_resident_i64_argsort_radix_serial,
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
#[cfg(test)]
use resident_compare_ordered::{
    i32_bits_into_u32, launch_cuda_resident_i32_compare_buffers_indices_ordered,
    validate_ordered_i32_comparison, validate_ordered_i32_context_identity,
    validate_ordered_i32_index_domain, validate_ordered_i32_input_window, COMPARE_ORDERED_PTX,
};
use resident_compare_ordered::{
    launch_cuda_buffer_i32_compare_indices_ordered, launch_cuda_owned_i32_compare_indices_ordered,
    launch_cuda_owned_i32_compare_indices_ordered_device,
    launch_cuda_resident_i32_compare_indices_ordered, launch_cuda_resident_i32_compare_project,
};
mod resident_scalar;
mod resident_text;
pub use resident_scalar::CudaI32Stats;
mod expression_vm;
use expression_vm::{
    run_resident_arith_program, run_resident_arith_program_at_indices, ExprTerminal,
};
pub use expression_vm::{ExprStep, ResidentElemType};
mod expression_filter;
use expression_filter::{
    launch_cuda_arith_value_column_at_indices, launch_cuda_arith_value_column_at_indices_nullable,
    launch_cuda_resident_expr_arith_filter, launch_cuda_resident_expr_compare_buffers_filter,
    launch_cuda_resident_expr_two_col_filter,
};
mod derived_column;
use derived_column::{
    launch_cuda_arith_value_column_device, launch_cuda_arith_value_column_device_at_coordinates,
    launch_cuda_arith_value_column_device_at_coordinates_nullable,
    launch_cuda_bool_to_int4_column_device, launch_cuda_build_wide_key_device,
    launch_cuda_mark_new_distinct_device, launch_cuda_mark_new_distinct_text_device,
    launch_cuda_pack_two_cols_i128_device, launch_cuda_pack_two_int4_cols_device,
    launch_cuda_upload_u64_device, launch_cuda_widen_col_to_i64_device,
};
pub use derived_column::{
    CudaGroupDeviceView, CudaWideKeyDescriptor, CudaWideKeySource, CudaWideKeyValidity,
    DeviceArithBuffer,
};
mod predicate_mask;
pub use predicate_mask::CudaPredicateMaskI32;
use predicate_mask::{compact_mask_i32_to_indices, retain_predicate_mask_i32};
mod version_conflict;
pub use version_conflict::CudaVersionConflictVerdict;
mod resident_gather;
use resident_gather::{
    copy_cuda_resident_bool_rows, copy_cuda_resident_i128_rows, copy_cuda_resident_i32_rows,
    copy_cuda_resident_i64_rows,
};
mod resident_filter;
use resident_filter::{
    launch_cuda_resident_bool_to_mask_filter, launch_cuda_resident_i128_compare_columns_filter,
    launch_cuda_resident_i128_compare_scalar_filter,
    launch_cuda_resident_i64_compare_columns_filter,
    launch_cuda_resident_i64_compare_scalar_filter,
    launch_cuda_resident_text_compare_scalar_filter, launch_cuda_resident_text_eq_scalar_filter,
    launch_cuda_resident_text_like_scalar_filter, launch_cuda_resident_uuid_compare_columns_filter,
    launch_cuda_resident_uuid_compare_scalar_filter,
};
mod resident_aggregate;
use resident_aggregate::{
    launch_cuda_resident_i128_minmax_partials_at_indices,
    launch_cuda_resident_i128_sum_partials_at_indices, launch_cuda_resident_i32_minmax_at_indices,
    launch_cuda_resident_i32_sum_at_indices, launch_cuda_resident_i64_minmax_at_indices,
    launch_cuda_resident_i64_sum_at_indices_i128,
};
mod resident_group;
pub use resident_group::GroupByI32Row;
use resident_group::{launch_cuda_group_by_i32_count_sum, launch_cuda_group_by_kernel_timed};
mod group_input;
use group_input::{validate_group_input, ValidatedGroupInput};
pub use group_input::{
    CudaGroupByInput, CudaGroupFixedSource, CudaGroupKeySource, CudaGroupTextDescriptorBuffer,
    CudaGroupTextDescriptors, CudaGroupValueSource, CudaGroupWideSource,
};
mod join_contract;
pub use join_contract::{
    CudaJoinCoordinatesU32, CudaJoinOrderKey, CudaJoinPayloadKey, CudaJoinSortKey,
};
mod join_filter;
mod join_fixed;
mod join_materialize;
pub use join_materialize::{
    CudaMaterializeJoinColumn, CudaMaterializedColumnKind, CudaMaterializedColumnLayout,
    CudaMaterializedRelation, CudaMaterializedResultFrame,
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
    submit_cuda_resident_i32_equal_any_project, submit_cuda_resident_i32_index_probe,
};
mod point_read_dense;
use point_read_dense::{
    prepare_cuda_resident_i32_multi_shard_index_probe_dense,
    submit_cuda_resident_i32_index_probe_dense,
    submit_cuda_resident_i32_multi_shard_index_probe_dense,
    submit_cuda_resident_i32_multi_shard_index_probe_dense_prepared,
};
pub use point_read_dense::{
    CudaI32DenseBatchProjection, CudaI32IndexProbeDenseSubmission, CudaI32MultiShardProbePlan,
    MultiShardProbeShard,
};
mod point_read_compound;
use point_read_compound::{
    execute_cuda_compound_i32_i64_multi_shard_probe,
    prepare_cuda_compound_i32_i64_multi_shard_probe,
};
pub use point_read_compound::{
    CompoundI32I64ProbeShard, CudaFixedPointProjection, CudaFixedPointProjectionKind,
    CudaI32I64MultiShardProbePlan, CudaI32I64PointBatchProjection, CudaI32I64PointKey,
};
mod point_read_bloom;
use point_read_bloom::probe_cuda_chunk_blooms;
pub use point_read_bloom::ChunkBloomProbeShard;
mod point_read_text;
use point_read_text::launch_cuda_resident_i32_equal_any_project_text;
mod point_read_submission;
use point_read_submission::{validate_i32_index_geometry, I32NeedlesHostGuard};
pub use point_read_submission::{
    CudaI32BatchProjectionColumns, CudaI32BatchProjectionRow, CudaI32EqualAnyProjectSubmission,
    CudaI32TextBatchProjectionRow,
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
        key_byte_offset,
        value_byte_offset,
        value_is_int8,
        key_is_int8,
        value_is_numeric,
        value_is_uuid,
        key_is_i128,
        key_is_text,
        key_offsets_off,
        key_bytes_off,
        key_bytes_len,
        value_is_text,
        value_offsets_off,
        value_bytes_off,
        value_bytes_len,
        key_base_override,
        value_base_override,
        comp_w,
        n_text,
        text_desc_ptr,
        value_null_off,
        key_null_off,
    } = validate_group_input(resident, input, indices)?;
    if matches!(input.value, CudaGroupValueSource::Unused { .. })
        && (agg_mask & !grouped_agg_mask::COUNT) != 0
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(agg_mask as usize));
    }
    if two_level
        && (value_is_int8
            || key_is_int8
            || value_is_numeric
            || value_is_uuid
            || key_is_i128
            || key_is_text
            || value_is_text
            || key_base_override != 0
            || value_base_override != 0
            || comp_w != 0
            || n_text != 0
            || value_null_off.is_some()
            || key_null_off.is_some())
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    launch_cuda_group_by_i32_count_sum(
        resident,
        key_byte_offset,
        value_byte_offset,
        indices,
        if two_level {
            c"gpu_db_group_by_i32_count_sum_twolevel"
        } else {
            c"gpu_db_group_by_i32_count_sum"
        },
        value_is_int8,
        key_is_int8,
        value_is_numeric,
        value_is_uuid,
        key_is_i128,
        key_is_text,
        key_offsets_off,
        key_bytes_off,
        key_bytes_len,
        value_is_text,
        value_offsets_off,
        value_bytes_off,
        value_bytes_len,
        key_base_override,
        value_base_override,
        comp_w,
        n_text,
        text_desc_ptr,
        value_null_off,
        key_null_off,
        agg_mask,
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

    /// Sort `keys[0..n]` on the GPU for ORDER BY, dispatching by size at the measured crossover:
    /// BITONIC (`bitonic_sort_i64`, O(n log^2 n)) below it, the proven resident LSD-RADIX argsort
    /// (O(n), signed->unsigned transform + direction handling) at/above it. Returns the row positions
    /// in ascending (or, with `descending`, descending) key order. Both arms produce a correct ordering
    /// (equal keys' relative order is unspecified, as for SQL ORDER BY without a tie-breaker). The radix
    /// arm uploads `keys` synchronously to a typed same-context device lease before stable radix execution.
    pub fn order_by_sort_i64(
        &self,
        keys: &[i64],
        descending: bool,
    ) -> Result<Vec<u32>, CudaRuntimeProbeError> {
        launch_cuda_order_by_sort_i64(self, keys, descending)
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

    /// Evaluate a checked arithmetic sort key only for the one-relation coordinates that survived
    /// filtering, then retain a source-row-addressable device key for coordinate sorting.
    pub fn arith_value_column_device_at_coordinates<'a>(
        &'a self,
        program: &[ExprStep],
        coordinates: &'a CudaJoinCoordinatesU32,
        source_row_count: u32,
        elem: ResidentElemType,
    ) -> Result<DeviceArithBuffer<'a>, CudaRuntimeProbeError> {
        launch_cuda_arith_value_column_device_at_coordinates(
            self,
            program,
            coordinates,
            source_row_count,
            elem,
        )
    }

    /// Nullable-int4 counterpart of [`Self::arith_value_column_device_at_coordinates`].
    pub fn arith_value_column_device_at_coordinates_nullable<'a>(
        &'a self,
        program: &[ExprStep],
        validity_program: &[ExprStep],
        coordinates: &'a CudaJoinCoordinatesU32,
        source_row_count: u32,
    ) -> Result<DeviceArithBuffer<'a>, CudaRuntimeProbeError> {
        launch_cuda_arith_value_column_device_at_coordinates_nullable(
            self,
            program,
            validity_program,
            coordinates,
            source_row_count,
        )
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

    /// Benchmark entry: time JUST the GROUP BY kernel (CUDA events, p50 of `runs`), returning the
    /// per-group rows + the p50 kernel milliseconds. Isolates the kernel from the alloc/H2D/D2H/compact
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
        if validated.value_is_int8
            || validated.key_is_int8
            || validated.value_is_numeric
            || validated.value_is_uuid
            || validated.key_is_i128
            || validated.key_is_text
            || validated.value_is_text
            || validated.key_base_override != 0
            || validated.value_base_override != 0
            || validated.comp_w != 0
            || validated.n_text != 0
            || validated.value_null_off.is_some()
            || validated.key_null_off.is_some()
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
        submit_cuda_resident_i32_multi_shard_index_probe_dense(self, shards, needles, read_snapshot)
    }

    /// Prepare one immutable, generation-owned multi-shard point route. The returned plan keeps its
    /// GPU descriptor table and every referenced shard/index/MVCC resource resident for reuse.
    pub fn prepare_multi_shard_i32_index_probe_dense(
        &self,
        shards: &[MultiShardProbeShard],
    ) -> Result<Arc<CudaI32MultiShardProbePlan>, CudaRuntimeProbeError> {
        Ok(Arc::new(
            prepare_cuda_resident_i32_multi_shard_index_probe_dense(self, shards)?,
        ))
    }

    /// Submit needles through a previously prepared generation-owned multi-shard point route.
    pub fn submit_prepared_multi_shard_i32_index_probe_dense(
        &self,
        plan: &Arc<CudaI32MultiShardProbePlan>,
        needles: &[i32],
        read_snapshot: u64,
    ) -> Result<CudaI32IndexProbeDenseSubmission, CudaRuntimeProbeError> {
        submit_cuda_resident_i32_multi_shard_index_probe_dense_prepared(
            self,
            Arc::clone(plan),
            needles,
            read_snapshot,
        )
    }

    /// Prepare one immutable table-level `(int4, int8)` compound point route. Index construction,
    /// fingerprint derivation, and descriptor publication stay on the GPU; the returned owner pins
    /// every exact shard generation and visibility sidecar it names.
    pub fn prepare_compound_i32_i64_multi_shard_probe(
        &self,
        shards: &[CompoundI32I64ProbeShard],
        gc_boundary: u64,
    ) -> Result<Arc<CudaI32I64MultiShardProbePlan>, CudaRuntimeProbeError> {
        Ok(Arc::new(prepare_cuda_compound_i32_i64_multi_shard_probe(
            self,
            shards,
            gc_boundary,
        )?))
    }

    /// Execute typed keys through a previously prepared compound point route. The candidate hash,
    /// exact tuple verification, MVCC visibility, and fixed-width gather all execute in one GPU
    /// probe kernel before the bounded terminal result readback.
    pub fn execute_prepared_compound_i32_i64_multi_shard_probe(
        &self,
        plan: &Arc<CudaI32I64MultiShardProbePlan>,
        keys: &[CudaI32I64PointKey],
        read_snapshot: u64,
    ) -> Result<CudaI32I64PointBatchProjection, CudaRuntimeProbeError> {
        execute_cuda_compound_i32_i64_multi_shard_probe(self, plan, keys, read_snapshot)
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
    /// primitives (typed column loads and buffer binary into an intermediate, then ordered
    /// compare-compaction). Returns matching row indices ascending by construction. `op_code`
    /// 0=add/1=sub/2=mul; `comparison` 0=eq/1=lt/2=le/3=gt/4=ge. Demonstrates the vectorized-
    /// interpreter model (an Expr tree lowered to a pipeline over intermediates), not a shape kernel.
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
    /// `needle` and return matching row indices ascending by ordered compaction. The general recursive
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
    // un-migrated direct callers (row_count / equal_count) and the migrated routes'
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
