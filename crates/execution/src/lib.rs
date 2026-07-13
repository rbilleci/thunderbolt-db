#[cfg(test)]
use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::Arc;

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
    check_cuda, CudaContextGuard, CudaDeviceAllocationGuard, CudaModuleGuard, GpuPrimaryContext,
    PinnedHostLease, PooledBufferLease, PooledDeviceBufferOwned, PooledStream, PooledStreamOwned,
    POOLED_STREAM_SCRATCH_BYTES,
};
#[cfg(test)]
use cuda_context::{gpu_primary_context, CudaEventGuard};
pub use cuda_context::{
    CudaAllocationScope, CudaExternalAllocationReservation, PendingCudaResidentDeviceCopy,
};
mod cuda_driver;
use cuda_driver::launch_cuda_resident_device_memory;
pub use cuda_driver::CudaDriverRuntime;

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
    validity_bitmap_kernel_arg,
};
#[cfg(test)]
use resident_count::{
    launch_cuda_resident_i32_compare_count_serial, launch_cuda_resident_i32_equal_count_serial,
};
mod resident_window;
mod expression_vm;
use expression_vm::run_resident_arith_program;
pub use expression_vm::{ExprStep, ResidentElemType};
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
    CudaGroupByInput, CudaGroupDeviceView, CudaGroupFixedSource, CudaGroupKeySource,
    CudaGroupTextDescriptorBuffer, CudaGroupTextDescriptors, CudaGroupTextSource,
    CudaGroupValueSource, CudaGroupWideSource,
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
    /// D2H-gathering. cuCtxSynchronize'd so a LATER kernel launch (the GROUP BY group-key read via
    /// `key_base_override`) sees valid keys, not a racing/stale buffer. The caller owns the returned
    /// `DeviceArithBuffer`'s lifetime -- it MUST outlive every launch that reads `device_ptr()`.
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
    /// GROUP BY kernel reads it via `key_base_override` (a bool GROUP BY key) or `value_base_override`
    /// (MIN/MAX over a bool value) -- grouped/aggregated on the AUDITED int4 path, which avoids a bool
    /// GROUP BY kernel and the shared-state concurrency hazard that blocked it. Reuses
    /// `gpu_db_resident_bool_to_mask` (it already writes int4 0/1). The caller owns the returned buffer;
    /// it MUST outlive every GROUP BY launch that reads `device_ptr()`.
    pub fn bool_to_int4_column_device(
        &self,
        bitmap_byte_offset: u64,
        n_rows: u64,
    ) -> Result<DeviceArithBuffer<'_>, CudaRuntimeProbeError> {
        launch_cuda_bool_to_int4_column_device(self, bitmap_byte_offset, n_rows)
    }

    /// Pack two int4-section columns (at byte offsets `off0`, `off1`) into one i64 derived key per row
    /// (col0 in the high 32 bits, col1 in the low 32 -- bijective) and return it RESIDENT
    /// (cuCtxSynchronize'd) for a COMPOSITE GROUP BY key via key_base_override. The executor unpacks the
    /// result slot key back into the two column values.
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
    /// (key_is_i128 + key_base_override) groups by it. cuCtxSynchronize'd so the GROUP BY launch reads
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
    /// (sign-extended). The fixed member of a composite (fixed-width, text) GROUP BY key, passed as
    /// key_base_override so the text-key claim folds it into the hash + verifies it. cuCtxSynchronize'd.
    pub fn widen_col_to_i64_device(
        &self,
        off: u64,
        w: u64,
        n_rows: u64,
    ) -> Result<DeviceArithBuffer<'_>, CudaRuntimeProbeError> {
        launch_cuda_widen_col_to_i64_device(self, off, w, n_rows)
    }

    /// Build a fixed-width WIDE KEY buffer for a general all-fixed composite GROUP BY key (>2 columns,
    /// or a numeric/uuid member, or a tuple wider than 128 bits). `descriptors` is one (kind, src_off,
    /// dst_off) per member (kind: 0=int4-section, 1=int8-section, 2=numeric/uuid 16B); `wbytes` is the
    /// per-row width (each int member 8 bytes, each numeric/uuid 16). Returns a leased [u8; wbytes*n]
    /// buffer the caller holds alive + passes as key_base_override with comp_w=wbytes to the GROUP BY
    /// kernel (which groups via a (rep_idx, hash) b128 claim that memcmps the wbytes). cuCtxSynchronize'd.
    pub fn build_wide_key_device(
        &self,
        descriptors: &[(u64, u64, u64)],
        wbytes: u64,
        n_rows: u64,
        // A separate per-row buffer for a DERIVED wide-key member (the expression group key), read by
        // descriptor kinds 4 (i32) / 5 (i64). 0 when the wide key has only column members.
        derived_ptr: u64,
        // M3 (doc 21) per-member NULL validity (nullable composite GROUP BY key). EMPTY = no validity (the
        // wide key has no trailing validity word; byte-identical to the pre-M3 path). Else one u64 per
        // member: u64::MAX = a non-nullable member; otherwise the member's validity-bitmap byte offset.
        // The caller must size `wbytes` to include the trailing 8-byte validity word when this is non-empty.
        validity_descs: &[u64],
    ) -> Result<DeviceArithBuffer<'_>, CudaRuntimeProbeError> {
        launch_cuda_build_wide_key_device(
            self,
            descriptors,
            wbytes,
            n_rows,
            derived_ptr,
            validity_descs,
        )
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
    /// count. Both leases live in the result.
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
    /// like the hetero sort's `int_keys`). `text_off`/`text_bytes` are the value column's offsets/bytes
    /// section byte offsets into the resident payload. Runs `gpu_db_mark_new_distinct_text` and returns
    /// `(g_sorted, new_distinct)` RESIDENT (cuCtxSynchronize'd). SUM(new_distinct) grouped by g_sorted
    /// (via the GROUP BY kernel's key/value_base_override) = the per-group distinct count.
    #[allow(clippy::too_many_arguments)]
    pub fn mark_new_distinct_text_device(
        &self,
        perm: &[u32],
        indices: &[u64],
        g_keys: &[i64],
        text_off: u64,
        text_bytes: u64,
        n: u64,
    ) -> Result<(DeviceArithBuffer<'_>, DeviceArithBuffer<'_>), CudaRuntimeProbeError> {
        launch_cuda_mark_new_distinct_text_device(
            self, perm, indices, g_keys, text_off, text_bytes, n,
        )
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

    pub fn count_text_prefix_from_payload(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        row_count: u64,
        prefix: &[u8],
    ) -> Result<u64, CudaRuntimeProbeError> {
        launch_cuda_resident_text_prefix_count(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            row_count,
            prefix,
        )
    }

    pub fn sum_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
    ) -> Result<i64, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_sum(self, byte_offset, row_count)
    }

    /// DIRECT scalar (count, sum, min, max) over a resident, non-nullable, UNFILTERED int4 column in
    /// ONE streaming pass — the MIN/MAX/AVG analogue of [`sum_i32_from_payload`]. Replaced the
    /// (now-removed) self-grouped hash kernel with group==value, which built an O(distinct)-entry hash
    /// table just to reduce; over a high-distinct
    /// column that hash table dominates (~182 Melem/s and falling at 8M distinct). This is a grid-stride
    /// scan + a `bar.sync` shared-memory block tree reduction of all four partials + ONE set of four
    /// global atomics per block (the audited `gpu_db_resident_i32_sum` pattern), so it runs at the same
    /// memory-bound roofline regardless of distinctness. Returns `(count, sum, min, max)`; `count == 0`
    /// (empty input) leaves min/max at their `INT_MAX`/`INT_MIN` init sentinels and the caller maps it
    /// to SQL NULL.
    pub fn scalar_stats_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
    ) -> Result<(u64, i64, i32, i32), CudaRuntimeProbeError> {
        launch_cuda_resident_i32_scalar_stats(self, byte_offset, row_count, None, None)
    }

    /// NULL-aware DIRECT scalar (count, sum, min, max) — the unfiltered nullable analogue of
    /// [`scalar_stats_i32_from_payload`]. A NULL value (per `null_bitmap_offset`, 1 = valid) contributes
    /// to NO statistic and is NOT counted, modelled EXACTLY on the grouped hash kernel's validity-bitmap
    /// logic (sentinel `0xFFFF...` = no bitmap = every row valid). Replaced the (now-removed) self-grouped
    /// NULL-aware hash kernel with group==value on the scalar arm. `count` is the SURVIVING (non-NULL) row
    /// count; the caller maps `count == 0` (all-NULL column) to SQL NULL.
    pub fn nullable_scalar_stats_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        null_bitmap_offset: Option<u64>,
    ) -> Result<(u64, i64, i32, i32), CudaRuntimeProbeError> {
        launch_cuda_resident_i32_scalar_stats(
            self,
            byte_offset,
            row_count,
            None,
            null_bitmap_offset,
        )
    }

    /// FILTERED (+ optionally NULL-aware) DIRECT scalar (count, sum, min, max) — the filtered analogue of
    /// [`scalar_stats_i32_from_payload`]. The filter `<col> <cmp> needle` runs ON-DEVICE (per-row
    /// predication, comparison codes 1=lt/2=lte/3=gt/4=gte matching the grouped hash kernel EXACTLY) and
    /// non-matching rows are SKIPPED; a NULL value (per `null_bitmap_offset`, 1 = valid; `None` = no
    /// bitmap) is ALSO skipped (3VL — a NULL contributes to no statistic). `count` is the count of
    /// SURVIVING (matching, non-NULL) rows; the caller maps `count == 0` (zero matches / all-NULL
    /// survivors) to SQL NULL. Replaced the (now-removed) self-grouped filtered hash kernel and the
    /// gather-to-host `filtered_stats_i32_compare_from_payload` path. Byte-identical to those for
    /// non-empty results.
    pub fn filtered_scalar_stats_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
        null_bitmap_offset: Option<u64>,
    ) -> Result<(u64, i64, i32, i32), CudaRuntimeProbeError> {
        launch_cuda_resident_i32_scalar_stats(
            self,
            byte_offset,
            row_count,
            Some((byte_offset, needle, comparison)),
            null_bitmap_offset,
        )
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

    pub fn project_i32_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
    ) -> Result<Vec<i32>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_project(self, byte_offset, row_count)
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

    pub fn match_i32_between_row_indices_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        lower_inclusive: i32,
        upper_inclusive: i32,
    ) -> Result<Vec<u64>, CudaRuntimeProbeError> {
        launch_cuda_resident_i32_between_row_indices(
            self,
            byte_offset,
            row_count,
            lower_inclusive,
            upper_inclusive,
        )
    }

    pub fn project_text_from_payload(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        row_count: u64,
    ) -> Result<Vec<String>, CudaRuntimeProbeError> {
        launch_cuda_resident_text_project(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            row_count,
        )
    }

    pub fn project_text_rows_from_payload(
        &self,
        offsets_byte_offset: u64,
        bytes_byte_offset: u64,
        bytes_len: u64,
        row_indices: &[u64],
    ) -> Result<Vec<String>, CudaRuntimeProbeError> {
        copy_cuda_resident_text_rows(
            self,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len,
            row_indices,
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

    pub fn filtered_stats_i32_compare_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
    ) -> Result<CudaI32Stats, CudaRuntimeProbeError> {
        let values = launch_cuda_resident_i32_compare_project(
            self,
            byte_offset,
            row_count,
            needle,
            comparison,
        )?;
        Ok(CudaI32Stats::from_values(&values))
    }

    pub fn stats_i32_between_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        lower_inclusive: i32,
        upper_inclusive: i32,
    ) -> Result<CudaI32Stats, CudaRuntimeProbeError> {
        if lower_inclusive > upper_inclusive {
            return Ok(CudaI32Stats::from_values(&[]));
        }
        launch_cuda_resident_i32_between_stats(
            self,
            byte_offset,
            row_count,
            lower_inclusive,
            upper_inclusive,
            None,
        )
    }

    /// NULL-aware BETWEEN stats (M3 — doc 21): like [`Self::stats_i32_between_from_payload`] but a NULL
    /// value (per `null_bitmap_offset`, 1 = valid) never satisfies the range, so it is excluded from
    /// count/sum/min/max. `None` reduces to the plain path. The caller maps a zero `count` to SQL NULL.
    pub fn stats_i32_between_nullable_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        lower_inclusive: i32,
        upper_inclusive: i32,
        null_bitmap_offset: Option<u64>,
    ) -> Result<CudaI32Stats, CudaRuntimeProbeError> {
        if lower_inclusive > upper_inclusive {
            return Ok(CudaI32Stats::from_values(&[]));
        }
        launch_cuda_resident_i32_between_stats(
            self,
            byte_offset,
            row_count,
            lower_inclusive,
            upper_inclusive,
            null_bitmap_offset,
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

    pub fn project_i32_compare_ordered_from_payload(
        &self,
        byte_offset: u64,
        row_count: u64,
        needle: i32,
        comparison: CudaI32Comparison,
        descending: bool,
        window: (u64, u64),
    ) -> Result<Vec<i32>, CudaRuntimeProbeError> {
        let mut values = launch_cuda_resident_i32_compare_project(
            self,
            byte_offset,
            row_count,
            needle,
            comparison,
        )?;
        if descending {
            values.sort_unstable_by(|left, right| right.cmp(left));
        } else {
            values.sort_unstable();
        }
        let (offset, limit) = window;
        let offset = usize::try_from(offset)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let limit = usize::try_from(limit)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        Ok(values.into_iter().skip(offset).take(limit).collect())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaI32Stats {
    pub count: u64,
    pub sum: i64,
    pub min: Option<i32>,
    pub max: Option<i32>,
}

impl CudaI32Stats {
    fn from_values(values: &[i32]) -> Self {
        Self {
            count: values.len() as u64,
            sum: values.iter().map(|value| i64::from(*value)).sum(),
            min: values.iter().copied().min(),
            max: values.iter().copied().max(),
        }
    }
}

fn launch_cuda_resident_text_prefix_count(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    row_count: u64,
    prefix: &[u8],
) -> Result<u64, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    let offsets_len = row_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let offsets_end = offsets_byte_offset
        .checked_add(offsets_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_end = bytes_byte_offset
        .checked_add(bytes_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if offsets_end > resident.metadata().allocated_bytes
        || bytes_end > resident.metadata().allocated_bytes
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            offsets_end.max(bytes_end) as usize,
        ));
    }
    let offsets_len_usize = usize::try_from(offsets_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_len_usize = usize::try_from(bytes_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let row_count_usize = usize::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut raw_offsets = vec![0_u8; offsets_len_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            raw_offsets.as_mut_ptr().cast::<c_void>(),
            resident.device_ptr() + offsets_byte_offset,
            offsets_len_usize,
        )
    })?;
    let mut bytes = vec![0_u8; bytes_len_usize];
    if bytes_len_usize > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                bytes.as_mut_ptr().cast::<c_void>(),
                resident.device_ptr() + bytes_byte_offset,
                bytes_len_usize,
            )
        })?;
    }

    let mut offsets = Vec::with_capacity(row_count_usize + 1);
    for chunk in raw_offsets.chunks_exact(std::mem::size_of::<u64>()) {
        offsets.push(u64::from_le_bytes(chunk.try_into().map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(raw_offsets.len())
        })?));
    }
    if offsets.len() != row_count_usize + 1 || offsets.first().copied() != Some(0) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(offsets.len()));
    }
    let mut matches = 0_u64;
    for pair in offsets.windows(2) {
        let start = usize::try_from(pair[0])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let end = usize::try_from(pair[1])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if start > end || end > bytes.len() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(end));
        }
        if bytes[start..end].starts_with(prefix) {
            matches = matches.saturating_add(1);
        }
    }
    Ok(matches)
}

fn launch_cuda_resident_text_project(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    row_count: u64,
) -> Result<Vec<String>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    let offsets_len = row_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let offsets_end = offsets_byte_offset
        .checked_add(offsets_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_end = bytes_byte_offset
        .checked_add(bytes_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if offsets_end > resident.metadata().allocated_bytes
        || bytes_end > resident.metadata().allocated_bytes
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            offsets_end.max(bytes_end) as usize,
        ));
    }
    let offsets_len_usize = usize::try_from(offsets_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_len_usize = usize::try_from(bytes_len)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let row_count_usize = usize::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut raw_offsets = vec![0_u8; offsets_len_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            raw_offsets.as_mut_ptr().cast::<c_void>(),
            resident.device_ptr() + offsets_byte_offset,
            offsets_len_usize,
        )
    })?;
    let mut bytes = vec![0_u8; bytes_len_usize];
    if bytes_len_usize > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                bytes.as_mut_ptr().cast::<c_void>(),
                resident.device_ptr() + bytes_byte_offset,
                bytes_len_usize,
            )
        })?;
    }

    let mut offsets = Vec::with_capacity(row_count_usize + 1);
    for chunk in raw_offsets.chunks_exact(std::mem::size_of::<u64>()) {
        offsets.push(u64::from_le_bytes(chunk.try_into().map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(raw_offsets.len())
        })?));
    }
    if offsets.len() != row_count_usize + 1 || offsets.first().copied() != Some(0) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(offsets.len()));
    }
    let mut values = Vec::with_capacity(row_count_usize);
    for pair in offsets.windows(2) {
        let start = usize::try_from(pair[0])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let end = usize::try_from(pair[1])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        if start > end || end > bytes.len() {
            return Err(CudaRuntimeProbeError::InvalidInputLength(end));
        }
        let value = std::str::from_utf8(&bytes[start..end])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(end))?;
        values.push(value.to_string());
    }
    Ok(values)
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




fn launch_cuda_resident_i32_between_row_indices(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    lower_inclusive: i32,
    upper_inclusive: i32,
) -> Result<Vec<u64>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    if lower_inclusive > upper_inclusive || row_count == 0 {
        return Ok(Vec::new());
    }
    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }
    let bytes_usize = usize::try_from(bytes - byte_offset)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut raw_values = vec![0_u8; bytes_usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            raw_values.as_mut_ptr().cast::<c_void>(),
            resident.device_ptr() + byte_offset,
            bytes_usize,
        )
    })?;

    let mut indices = Vec::new();
    for (idx, chunk) in raw_values
        .chunks_exact(std::mem::size_of::<i32>())
        .enumerate()
    {
        let value = i32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(raw_values.len()))?,
        );
        if value >= lower_inclusive && value <= upper_inclusive {
            indices.push(
                u64::try_from(idx)
                    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
            );
        }
    }
    Ok(indices)
}

fn copy_cuda_resident_text_rows(
    resident: &CudaResidentDeviceMemory,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    row_indices: &[u64],
) -> Result<Vec<String>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    if row_indices.is_empty() {
        return Ok(Vec::new());
    }

    let min_row_idx = row_indices.iter().copied().min().unwrap_or(0);
    let max_row_idx = row_indices.iter().copied().max().unwrap_or(0);
    let offset_count = max_row_idx
        .checked_sub(min_row_idx)
        .and_then(|span| span.checked_add(2))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let offsets_offset = min_row_idx
        .checked_mul(std::mem::size_of::<u64>() as u64)
        .and_then(|offset| offsets_byte_offset.checked_add(offset))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let offsets_end = offsets_byte_offset
        .checked_add(
            max_row_idx
                .checked_add(2)
                .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        )
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let bytes_end = bytes_byte_offset
        .checked_add(bytes_len)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if offsets_end > resident.metadata().allocated_bytes
        || bytes_end > resident.metadata().allocated_bytes
    {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            offsets_end.max(bytes_end) as usize,
        ));
    }

    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut offsets = vec![
        0_u64;
        usize::try_from(offset_count).map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
        })?
    ];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            offsets.as_mut_ptr().cast::<c_void>(),
            resident.device_ptr() + offsets_offset,
            offsets
                .len()
                .checked_mul(std::mem::size_of::<u64>())
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
        )
    })?;

    let mut spans = Vec::with_capacity(row_indices.len());
    let mut min_text_start = u64::MAX;
    let mut max_text_end = 0_u64;
    for row_idx in row_indices {
        let offset_idx = row_idx
            .checked_sub(min_row_idx)
            .and_then(|idx| usize::try_from(idx).ok())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let start = offsets[offset_idx];
        let end = offsets[offset_idx + 1];
        if start > end || end > bytes_len {
            return Err(CudaRuntimeProbeError::InvalidInputLength(end as usize));
        }
        min_text_start = min_text_start.min(start);
        max_text_end = max_text_end.max(end);
        spans.push((start, end));
    }

    let text_span_len = max_text_end
        .checked_sub(min_text_start)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut text_bytes = vec![
        0_u8;
        usize::try_from(text_span_len).map_err(|_| {
            CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
        })?
    ];
    if text_span_len > 0 {
        check_cuda(unsafe {
            cu_memcpy_dtoh(
                text_bytes.as_mut_ptr().cast::<c_void>(),
                resident.device_ptr() + bytes_byte_offset + min_text_start,
                text_bytes.len(),
            )
        })?;
    }

    let mut values = Vec::with_capacity(row_indices.len());
    for (start, end) in spans {
        let value_len = usize::try_from(end - start)
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let value_start = start
            .checked_sub(min_text_start)
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let value_end = value_start
            .checked_add(value_len)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
        let value = std::str::from_utf8(&text_bytes[value_start..value_end])
            .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(value_len))?;
        values.push(value.to_string());
    }
    Ok(values)
}

fn launch_cuda_resident_i32_sum(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
) -> Result<i64, CudaRuntimeProbeError> {
    type CuMemsetD8Async = unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> i32;
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

    // P2-M2 — the i32-sum kernel is already a parallel grid-stride reduction (each thread sums a
    // strided slice into an s64, then `atom.global.add.u64`s it into one output). This migrates the
    // LAUNCH off the default/null stream + per-call cuModuleLoadData (re-JIT) + per-call cuMemAlloc
    // onto a pooled private stream with a cached module + async-memset scratch via
    // `launch_on_pooled_stream` (event-timed, covering-synced, drained-on-error). Kernel unchanged.

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_sum(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .u64 out_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_active;
    .reg .pred %p_isthr0;
    .shared .align 8 .b64 s_part[1024];
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %addr;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %thread;
    .reg .u32 %grid_dim;
    .reg .u64 %wide_block;
    .reg .u64 %wide_thread;
    .reg .u64 %wide_block_dim;
    .reg .u64 %wide_grid_dim;
    .reg .u64 %sum_bits;
    .reg .u64 %ignored;
    .reg .s64 %sum;
    .reg .s64 %wide;
    .reg .s32 %r_value;
    .reg .u32 %rstride;
    .reg .u32 %peer;
    .reg .u64 %sh_base;
    .reg .u64 %sh_self;
    .reg .u64 %sh_peer;
    .reg .u64 %peer_val;
    .reg .u64 %block_tot;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %out, [out_ptr];

    add.u64 %base, %resident, %offset;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mov.u32 %thread, %tid.x;
    mov.u32 %grid_dim, %nctaid.x;
    cvt.u64.u32 %wide_block, %r_block;
    cvt.u64.u32 %wide_thread, %thread;
    cvt.u64.u32 %wide_block_dim, %r_block_dim;
    cvt.u64.u32 %wide_grid_dim, %grid_dim;
    mul.lo.u64 %idx, %wide_block, %wide_block_dim;
    add.u64 %idx, %idx, %wide_thread;
    mul.lo.u64 %stride, %wide_grid_dim, %wide_block_dim;
    mov.s64 %sum, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %addr, %idx, 4;
    add.u64 %addr, %base, %addr;
    ld.global.s32 %r_value, [%addr];
    cvt.s64.s32 %wide, %r_value;
    add.s64 %sum, %sum, %wide;
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    // ---- per-block reduction: sum every thread's i64 partial, then ONE atomic per block ----
    // The grid is clamped to a saturating constant (<=1024 blocks), so each thread accumulates a REAL
    // i64 partial over its grid-stride rows. Summing the per-thread partials in a barrier-synchronized
    // SHARED-MEMORY tree, then a single atom.add per block, replaces the old one-atomic-per-thread storm
    // (grid*BLOCK atomics on one address). Byte-identical to the old per-thread accumulation: two's-
    // complement (u64) addition is associative AND commutative, so the i64 sum is independent of the
    // grouping/order of the adds (mod 2^64). A shfl.sync would be WRONG here -- the grid-stride loop
    // exits per-thread (idx >= rows), so lanes within a warp run a DIFFERENT number of iterations and are
    // NOT converged at done:; bar.sync synchronizes the WHOLE block regardless, and every barrier below
    // is on the straight-line path (outside the @!%p_active guard) so all threads reach it. %thread =
    // %tid.x (in-block id); %r_block_dim = %ntid.x (block width, a power of two so the tree terminates).
    cvt.u64.s64 %sum_bits, %sum;
    mov.u64 %sh_base, s_part;
    mul.wide.u32 %sh_self, %thread, 8;
    add.u64 %sh_self, %sh_base, %sh_self;
    st.shared.u64 [%sh_self], %sum_bits;
    bar.sync 0;

    // tree reduce: for rstride = bdim/2, bdim/4, ..., 1: s_part[t] += s_part[t + rstride] for t < rstride.
    shr.u32 %rstride, %r_block_dim, 1;
red_loop:
    setp.eq.u32 %p_done, %rstride, 0;
    @%p_done bra red_done;
    setp.lt.u32 %p_active, %thread, %rstride;
    @!%p_active bra red_skip;
    add.u32 %peer, %thread, %rstride;
    mul.wide.u32 %sh_peer, %peer, 8;
    add.u64 %sh_peer, %sh_base, %sh_peer;
    ld.shared.u64 %peer_val, [%sh_peer];
    ld.shared.u64 %sum_bits, [%sh_self];
    add.u64 %sum_bits, %sum_bits, %peer_val;
    st.shared.u64 [%sh_self], %sum_bits;
red_skip:
    bar.sync 0;
    shr.u32 %rstride, %rstride, 1;
    bra red_loop;
red_done:
    // thread 0 holds the block total at s_part[0]; ONE global atomic add for the whole block.
    setp.eq.u32 %p_isthr0, %thread, 0;
    @!%p_isthr0 bra block_done;
    ld.shared.u64 %block_tot, [%sh_base];
    atom.global.add.u64 %ignored, [%out], %block_tot;
block_done:
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memset_d8_async = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8Async>(b"cuMemsetD8Async\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_sum", &ptx)?;

    let block_dim = 256_u32;
    let grid_dim = if row_count == 0 {
        1
    } else {
        row_count.div_ceil(u64::from(block_dim)).min(1024) as u32
    };

    let mut output_bytes = [0_u8; std::mem::size_of::<i64>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        // Zero the 8-byte scratch on the stream (the kernel atom-adds into it), ordered before the
        // kernel launch on the same stream.
        let memset_rc =
            unsafe { cu_memset_d8_async(output_ptr, 0, std::mem::size_of::<i64>(), stream) };
        if memset_rc != 0 {
            return memset_rc;
        }
        let mut resident_arg = resident.device_ptr();
        let mut offset_arg = byte_offset;
        let mut rows_arg = row_count;
        let mut output_arg = output_ptr;
        let mut args = [
            (&mut resident_arg as *mut u64).cast::<c_void>(),
            (&mut offset_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                function,
                grid_dim,
                1,
                1,
                block_dim,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;

    Ok(i64::from_le_bytes(output_bytes))
}

fn launch_cuda_resident_i32_scalar_stats(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    // `Some((filter_byte_offset, needle, comparison))` = an on-device per-row filter `<col> <cmp>
    // needle` (non-matches skipped); `None` = no filter (the unfiltered fast path). The scalar arms
    // always pass `filter_byte_offset == byte_offset` (the filter and aggregate are the SAME column).
    filter: Option<(u64, i32, CudaI32Comparison)>,
    // M3 (doc 21): `Some(off)` = the value column's NULL validity bitmap byte offset (1 = valid); `None`
    // = no bitmap ⇒ every row valid (the no-NULL fast path). A NULL value contributes to NO statistic.
    null_bitmap_offset: Option<u64>,
) -> Result<(u64, i64, i32, i32), CudaRuntimeProbeError> {
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

    // DIRECT scalar-stats reduction — the MIN/MAX/AVG analogue of `gpu_db_resident_i32_sum`, with an
    // OPTIONAL on-device per-row filter and an OPTIONAL NULL-skip. Each thread accumulates (count, i64
    // sum, i32 min, i32 max) over its grid-stride slice; per row it (1) optionally evaluates the filter
    // `<col> <cmp> needle` (comparison 0=none/1=lt/2=lte/3=gt/4=gte) and SKIPS non-matches, then (2)
    // optionally reads the validity bit (sentinel `0xFFFF...` =
    // no bitmap ⇒ valid; the standard per-row validity-bitmap convention) and
    // SKIPS NULL rows. Surviving rows do count++, sum+=v, min/max. The block then reduces all four
    // partials in a `bar.sync` SHARED-MEMORY tree (NOT shfl — the grid-stride loop exits per-thread at
    // `done:` so warp lanes run a DIFFERENT iteration count and are NOT converged; bar.sync synchronizes
    // the WHOLE block and every barrier below is on the straight-line path), and thread 0 issues ONE set
    // of four global atomics for the block: add count, add sum (u64 two's-complement), min.s32, max.s32.
    // count/sum atomic-adds are order-independent (mod 2^64), min/max are associative/commutative, so the
    // result is byte-identical to the self-grouped path's reduced (count, sum, min, max). The UNFILTERED
    // NON-NULLABLE path (comparison==0, null_off==sentinel) takes a fast straight-line branch with NO
    // per-row filter/bitmap load — byte- and speed-identical to the slice-a kernel. Host inits the
    // 24-byte out struct count=0, sum=0, min=INT_MAX, max=INT_MIN (the min/max sentinels can't be a
    // plain memset, so we H2D the init struct on-stream before the kernel). `.target sm_60` for the
    // global min/max atomics. Saturating grid (.min) like sum; row_count==0 (or zero survivors) leaves
    // the out struct at its init (count 0 => caller maps to SQL NULL).
    const PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_resident_i32_scalar_stats(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .u64 out_ptr,
    .param .u64 filter_byte_offset,
    .param .s32 needle,
    .param .u32 comparison,
    .param .u64 value_null_bitmap_offset
)
{
    .reg .pred %p_done;
    .reg .pred %p_active;
    .reg .pred %p_isthr0;
    .reg .pred %p_check;
    .reg .pred %p_match;
    .reg .pred %p_no_bitmap;
    .reg .pred %p_valid;
    .shared .align 8 .b64 s_count[1024];
    .shared .align 8 .b64 s_sum[1024];
    .shared .align 4 .b32 s_min[1024];
    .shared .align 4 .b32 s_max[1024];
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %filter_base;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %addr;
    .reg .u64 %roff;
    .reg .u32 %comparison;
    .reg .s32 %needle;
    .reg .s32 %filter_value;
    .reg .u64 %val_null_off;
    .reg .u64 %sentinel;
    .reg .u64 %word_byte;
    .reg .u64 %bitmap_addr;
    .reg .u32 %bitmap_word;
    .reg .u32 %bit_pos;
    .reg .u32 %valid_bit;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %thread;
    .reg .u32 %grid_dim;
    .reg .u64 %wide_block;
    .reg .u64 %wide_thread;
    .reg .u64 %wide_block_dim;
    .reg .u64 %wide_grid_dim;
    .reg .u64 %count;
    .reg .u64 %sum_bits;
    .reg .u64 %ignored64;
    .reg .s32 %ignored32;
    .reg .s64 %sum;
    .reg .s64 %wide;
    .reg .s32 %r_value;
    .reg .s32 %min;
    .reg .s32 %max;
    .reg .u32 %rstride;
    .reg .u32 %peer;
    .reg .u64 %sh_count_base;
    .reg .u64 %sh_sum_base;
    .reg .u64 %sh_min_base;
    .reg .u64 %sh_max_base;
    .reg .u64 %sh_count_self;
    .reg .u64 %sh_sum_self;
    .reg .u64 %sh_min_self;
    .reg .u64 %sh_max_self;
    .reg .u64 %sh_count_peer;
    .reg .u64 %sh_sum_peer;
    .reg .u64 %sh_min_peer;
    .reg .u64 %sh_max_peer;
    .reg .u64 %off8;
    .reg .u64 %off4;
    .reg .u64 %peer_count;
    .reg .u64 %peer_sum;
    .reg .s32 %peer_min;
    .reg .s32 %peer_max;
    .reg .u64 %count_addr;
    .reg .u64 %sum_addr;
    .reg .u64 %min_addr;
    .reg .u64 %max_addr;
    .reg .u64 %block_count;
    .reg .u64 %block_sum;
    .reg .s32 %block_min;
    .reg .s32 %block_max;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %filter_base, [filter_byte_offset];
    ld.param.s32 %needle, [needle];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %val_null_off, [value_null_bitmap_offset];

    add.u64 %base, %resident, %offset;
    add.u64 %filter_base, %resident, %filter_base;
    mov.u64 %sentinel, 0xFFFFFFFFFFFFFFFF;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mov.u32 %thread, %tid.x;
    mov.u32 %grid_dim, %nctaid.x;
    cvt.u64.u32 %wide_block, %r_block;
    cvt.u64.u32 %wide_thread, %thread;
    cvt.u64.u32 %wide_block_dim, %r_block_dim;
    cvt.u64.u32 %wide_grid_dim, %grid_dim;
    mul.lo.u64 %idx, %wide_block, %wide_block_dim;
    add.u64 %idx, %idx, %wide_thread;
    mul.lo.u64 %stride, %wide_grid_dim, %wide_block_dim;
    mov.u64 %count, 0;
    mov.s64 %sum, 0;
    mov.s32 %min, 2147483647;
    mov.s32 %max, -2147483648;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %roff, %idx, 4;

    // Optional on-device filter `<col> <cmp> needle` (comparison 0=none/1=lt/2=lte/3=gt/4=gte).
    // comparison==0 => no filter, fall
    // straight through (the unfiltered fast branch, no filter load). Non-matches skip this row.
    setp.eq.u32 %p_check, %comparison, 0;
    @%p_check bra after_filter;
    add.u64 %addr, %filter_base, %roff;
    ld.global.s32 %filter_value, [%addr];
    mov.pred %p_match, 0;
    setp.eq.u32 %p_check, %comparison, 1;
    @%p_check bra f_lt;
    setp.eq.u32 %p_check, %comparison, 2;
    @%p_check bra f_lte;
    setp.eq.u32 %p_check, %comparison, 3;
    @%p_check bra f_gt;
    setp.eq.u32 %p_check, %comparison, 4;
    @%p_check bra f_gte;
    bra next_row;
f_lt:
    setp.lt.s32 %p_match, %filter_value, %needle;
    bra f_done;
f_lte:
    setp.le.s32 %p_match, %filter_value, %needle;
    bra f_done;
f_gt:
    setp.gt.s32 %p_match, %filter_value, %needle;
    bra f_done;
f_gte:
    setp.ge.s32 %p_match, %filter_value, %needle;
f_done:
    @!%p_match bra next_row;

after_filter:
    // Optional NULL-skip (M3 3VL): a NULL value contributes to no statistic. val_null_off == sentinel
    // (0xFFFF...) => no validity bitmap => every row valid (skip the load). Modelled EXACTLY on the
    // grouped hash kernel's validity-bitmap logic (1 = valid/present, 0 = NULL).
    setp.eq.u64 %p_no_bitmap, %val_null_off, %sentinel;
    @%p_no_bitmap bra accumulate;
    shr.u64 %word_byte, %idx, 5;          // idx / 32 (the validity word index)
    mul.lo.u64 %word_byte, %word_byte, 4; // * 4 bytes per u32 word
    add.u64 %bitmap_addr, %resident, %val_null_off;
    add.u64 %bitmap_addr, %bitmap_addr, %word_byte;
    ld.global.u32 %bitmap_word, [%bitmap_addr];
    cvt.u32.u64 %bit_pos, %idx;
    and.b32 %bit_pos, %bit_pos, 31;       // idx % 32
    bfe.u32 %valid_bit, %bitmap_word, %bit_pos, 1;
    setp.eq.u32 %p_valid, %valid_bit, 1;  // 1 = valid/present, 0 = NULL
    @!%p_valid bra next_row;              // NULL value => skip this row

accumulate:
    add.u64 %addr, %base, %roff;
    ld.global.s32 %r_value, [%addr];
    cvt.s64.s32 %wide, %r_value;
    add.s64 %sum, %sum, %wide;
    add.u64 %count, %count, 1;
    min.s32 %min, %min, %r_value;
    max.s32 %max, %max, %r_value;

next_row:
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    // ---- per-block reduction: tree-reduce all four partials in shared memory, then ONE set of four
    // atomics per block. Grid is saturating-clamped (<=1024 blocks) so each thread holds a REAL partial
    // over its grid-stride rows. count/sum (u64 two's-complement add) are associative+commutative mod
    // 2^64; min/max are associative+commutative; so the tree grouping is byte-identical to a flat
    // accumulation. %thread = %tid.x (in-block id); %r_block_dim = %ntid.x (a power of two so the tree
    // terminates). Every barrier below is on the straight-line path (outside the @!%p_active guard) so
    // all threads in the block reach it even though the grid-stride loop exited per-thread above.
    mov.u64 %sh_count_base, s_count;
    mov.u64 %sh_sum_base, s_sum;
    mov.u64 %sh_min_base, s_min;
    mov.u64 %sh_max_base, s_max;
    mul.wide.u32 %off8, %thread, 8;
    mul.wide.u32 %off4, %thread, 4;
    add.u64 %sh_count_self, %sh_count_base, %off8;
    add.u64 %sh_sum_self, %sh_sum_base, %off8;
    add.u64 %sh_min_self, %sh_min_base, %off4;
    add.u64 %sh_max_self, %sh_max_base, %off4;
    cvt.u64.s64 %sum_bits, %sum;
    st.shared.u64 [%sh_count_self], %count;
    st.shared.u64 [%sh_sum_self], %sum_bits;
    st.shared.s32 [%sh_min_self], %min;
    st.shared.s32 [%sh_max_self], %max;
    bar.sync 0;

    // tree reduce: for rstride = bdim/2, bdim/4, ..., 1: combine s[t] with s[t + rstride] for t < rstride.
    shr.u32 %rstride, %r_block_dim, 1;
red_loop:
    setp.eq.u32 %p_done, %rstride, 0;
    @%p_done bra red_done;
    setp.lt.u32 %p_active, %thread, %rstride;
    @!%p_active bra red_skip;
    add.u32 %peer, %thread, %rstride;
    mul.wide.u32 %off8, %peer, 8;
    mul.wide.u32 %off4, %peer, 4;
    add.u64 %sh_count_peer, %sh_count_base, %off8;
    add.u64 %sh_sum_peer, %sh_sum_base, %off8;
    add.u64 %sh_min_peer, %sh_min_base, %off4;
    add.u64 %sh_max_peer, %sh_max_base, %off4;
    ld.shared.u64 %peer_count, [%sh_count_peer];
    ld.shared.u64 %peer_sum, [%sh_sum_peer];
    ld.shared.s32 %peer_min, [%sh_min_peer];
    ld.shared.s32 %peer_max, [%sh_max_peer];
    ld.shared.u64 %count, [%sh_count_self];
    ld.shared.u64 %sum_bits, [%sh_sum_self];
    ld.shared.s32 %min, [%sh_min_self];
    ld.shared.s32 %max, [%sh_max_self];
    add.u64 %count, %count, %peer_count;
    add.u64 %sum_bits, %sum_bits, %peer_sum;
    min.s32 %min, %min, %peer_min;
    max.s32 %max, %max, %peer_max;
    st.shared.u64 [%sh_count_self], %count;
    st.shared.u64 [%sh_sum_self], %sum_bits;
    st.shared.s32 [%sh_min_self], %min;
    st.shared.s32 [%sh_max_self], %max;
red_skip:
    bar.sync 0;
    shr.u32 %rstride, %rstride, 1;
    bra red_loop;
red_done:
    // thread 0 holds the block totals at index 0; ONE set of four global atomics for the whole block.
    setp.eq.u32 %p_isthr0, %thread, 0;
    @!%p_isthr0 bra block_done;
    ld.shared.u64 %block_count, [%sh_count_base];
    ld.shared.u64 %block_sum, [%sh_sum_base];
    ld.shared.s32 %block_min, [%sh_min_base];
    ld.shared.s32 %block_max, [%sh_max_base];
    mov.u64 %count_addr, %out;
    atom.global.add.u64 %ignored64, [%count_addr], %block_count;
    add.u64 %sum_addr, %out, 8;
    atom.global.add.u64 %ignored64, [%sum_addr], %block_sum;
    add.u64 %min_addr, %out, 16;
    atom.global.min.s32 %ignored32, [%min_addr], %block_min;
    add.u64 %max_addr, %out, 20;
    atom.global.max.s32 %ignored32, [%max_addr], %block_max;
block_done:
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }

    // Filter args: `None` => comparison 0 (no filter; filter_base unused, point it at the value column).
    // `Some` => the bounds-checked filter column + needle + the comparison `.code()` (1..4, the SAME
    // mapping the grouped hash kernel uses).
    let (filter_byte_offset, needle, comparison_code) = match filter {
        Some((filter_byte_offset, needle, comparison)) => {
            let filter_bytes = row_count
                .checked_mul(std::mem::size_of::<i32>() as u64)
                .and_then(|bytes| filter_byte_offset.checked_add(bytes))
                .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
            if filter_bytes > resident.metadata().allocated_bytes {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    filter_bytes as usize,
                ));
            }
            (filter_byte_offset, needle, comparison.code())
        }
        None => (byte_offset, 0, 0),
    };
    // u64::MAX sentinel when there is no validity bitmap; otherwise the bounds-checked byte offset.
    let null_off_value = validity_bitmap_kernel_arg(null_bitmap_offset, row_count, resident)?;

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = resident
        .primary()
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_scalar_stats", &ptx)?;

    let block_dim = 256_u32;
    let grid_dim = if row_count == 0 {
        1
    } else {
        row_count.div_ceil(u64::from(block_dim)).min(1024) as u32
    };

    // Init struct H2D'd into the scratch on-stream BEFORE the kernel: count/sum start at 0 (atomic
    // add), min/max at the INT sentinels (atomic min/max — they can't be a plain memset). `initial`
    // outlives the helper's covering sync, so the async source stays valid until the copy completes.
    let initial = CudaI32StatsRaw {
        count: 0,
        sum: 0,
        min: i32::MAX,
        max: i32::MIN,
    };
    let mut output_bytes = [0_u8; std::mem::size_of::<CudaI32StatsRaw>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        let htod_rc = unsafe {
            htod_async(
                output_ptr,
                (&initial as *const CudaI32StatsRaw).cast::<c_void>(),
                std::mem::size_of::<CudaI32StatsRaw>(),
                stream,
            )
        };
        if htod_rc != 0 {
            return htod_rc;
        }
        let mut resident_arg = resident.device_ptr();
        let mut offset_arg = byte_offset;
        let mut rows_arg = row_count;
        let mut output_arg = output_ptr;
        let mut filter_off_arg = filter_byte_offset;
        let mut needle_arg = needle;
        let mut comparison_arg = comparison_code;
        let mut null_off_arg = null_off_value;
        let mut args = [
            (&mut resident_arg as *mut u64).cast::<c_void>(),
            (&mut offset_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut filter_off_arg as *mut u64).cast::<c_void>(),
            (&mut needle_arg as *mut i32).cast::<c_void>(),
            (&mut comparison_arg as *mut u32).cast::<c_void>(),
            (&mut null_off_arg as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                function,
                grid_dim,
                1,
                1,
                block_dim,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;

    let count = u64::from_le_bytes(output_bytes[0..8].try_into().unwrap());
    let sum = i64::from_le_bytes(output_bytes[8..16].try_into().unwrap());
    let min = i32::from_le_bytes(output_bytes[16..20].try_into().unwrap());
    let max = i32::from_le_bytes(output_bytes[20..24].try_into().unwrap());
    Ok((count, sum, min, max))
}

#[repr(C)]
struct CudaI32StatsRaw {
    count: u64,
    sum: i64,
    min: i32,
    max: i32,
}

fn launch_cuda_resident_i32_between_stats(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
    lower_inclusive: i32,
    upper_inclusive: i32,
    // M3 (doc 21): `Some(off)` = the column's NULL validity bitmap byte offset (1 = valid); `None` = no
    // bitmap ⇒ every row valid. A NULL value never satisfies BETWEEN (3VL).
    null_bitmap_offset: Option<u64>,
) -> Result<CudaI32Stats, CudaRuntimeProbeError> {
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

    // P2-M2 — the between-stats kernel is already a parallel grid-stride reduction (each thread
    // computes count/sum/min/max for the [lower,upper] predicate, then atomic add count/sum +
    // atomic min/max into one 24-byte output struct). Migrate the LAUNCH off the default/null stream
    // + per-call cuModuleLoadData (re-JIT) + per-call cuMemAlloc + blocking H2D/D2H onto a pooled
    // private stream with a cached module and an ASYNC H2D of the init struct
    // (count=0,sum=0,min=INT_MAX,max=INT_MIN — the min/max sentinels can't be memset like sum's plain
    // zero) into the pooled scratch BEFORE the kernel, via launch_on_pooled_stream (event-timed,
    // covering-synced, drained-on-error). Kernel unchanged.

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_between_stats(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .s32 lower_inclusive,
    .param .s32 upper_inclusive,
    .param .u64 out_ptr,
    .param .u64 null_bitmap_offset
)
{
    .reg .pred %p_done;
    .reg .pred %p_ge_lower;
    .reg .pred %p_le_upper;
    .reg .pred %p_match;
    .reg .pred %p_no_bitmap;
    .reg .pred %p_valid;
    .reg .u64 %null_off;
    .reg .u64 %sentinel;
    .reg .u64 %word_byte;
    .reg .u64 %bitmap_addr;
    .reg .u32 %bitmap_word;
    .reg .u32 %bit_pos;
    .reg .u32 %valid_bit;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %stride;
    .reg .u64 %addr;
    .reg .u64 %count_addr;
    .reg .u64 %sum_addr;
    .reg .u64 %min_addr;
    .reg .u64 %max_addr;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %thread;
    .reg .u32 %grid_dim;
    .reg .u64 %wide_block;
    .reg .u64 %wide_thread;
    .reg .u64 %wide_block_dim;
    .reg .u64 %wide_grid_dim;
    .reg .u64 %count;
    .reg .u64 %sum_bits;
    .reg .u64 %ignored64;
    .reg .s32 %ignored32;
    .reg .s64 %sum;
    .reg .s64 %wide;
    .reg .s32 %r_value;
    .reg .s32 %lower;
    .reg .s32 %upper;
    .reg .s32 %min;
    .reg .s32 %max;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.s32 %lower, [lower_inclusive];
    ld.param.s32 %upper, [upper_inclusive];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %null_off, [null_bitmap_offset];

    add.u64 %base, %resident, %offset;
    mov.u64 %sentinel, 0xFFFFFFFFFFFFFFFF;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mov.u32 %thread, %tid.x;
    mov.u32 %grid_dim, %nctaid.x;
    cvt.u64.u32 %wide_block, %r_block;
    cvt.u64.u32 %wide_thread, %thread;
    cvt.u64.u32 %wide_block_dim, %r_block_dim;
    cvt.u64.u32 %wide_grid_dim, %grid_dim;
    mul.lo.u64 %idx, %wide_block, %wide_block_dim;
    add.u64 %idx, %idx, %wide_thread;
    mul.lo.u64 %stride, %wide_grid_dim, %wide_block_dim;
    mov.u64 %count, 0;
    mov.s64 %sum, 0;
    mov.s32 %min, 2147483647;
    mov.s32 %max, -2147483648;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %addr, %idx, 4;
    add.u64 %addr, %base, %addr;
    ld.global.s32 %r_value, [%addr];
    setp.ge.s32 %p_ge_lower, %r_value, %lower;
    setp.le.s32 %p_le_upper, %r_value, %upper;
    and.pred %p_match, %p_ge_lower, %p_le_upper;
    @!%p_match bra next;
    // M3 3VL: a NULL value never satisfies BETWEEN (its bytes are a 0 placeholder). null_off ==
    // sentinel (0xFFFF...) => no validity bitmap => every row valid (skip the load).
    setp.eq.u64 %p_no_bitmap, %null_off, %sentinel;
    @%p_no_bitmap bra accumulate;
    shr.u64 %word_byte, %idx, 5;          // idx / 32 (validity word index)
    mul.lo.u64 %word_byte, %word_byte, 4; // * 4 bytes per u32 word
    add.u64 %bitmap_addr, %resident, %null_off;
    add.u64 %bitmap_addr, %bitmap_addr, %word_byte;
    ld.global.u32 %bitmap_word, [%bitmap_addr];
    cvt.u32.u64 %bit_pos, %idx;
    and.b32 %bit_pos, %bit_pos, 31;       // idx % 32
    bfe.u32 %valid_bit, %bitmap_word, %bit_pos, 1;
    setp.eq.u32 %p_valid, %valid_bit, 1;  // 1 = valid/present, 0 = NULL
    @!%p_valid bra next;                  // NULL => not a match, skip

accumulate:
    cvt.s64.s32 %wide, %r_value;
    add.s64 %sum, %sum, %wide;
    add.u64 %count, %count, 1;
    min.s32 %min, %min, %r_value;
    max.s32 %max, %max, %r_value;

next:
    add.u64 %idx, %idx, %stride;
    bra loop;

done:
    setp.eq.u64 %p_done, %count, 0;
    @%p_done bra ret_done;
    mov.u64 %count_addr, %out;
    atom.global.add.u64 %ignored64, [%count_addr], %count;
    add.u64 %sum_addr, %out, 8;
    cvt.u64.s64 %sum_bits, %sum;
    atom.global.add.u64 %ignored64, [%sum_addr], %sum_bits;
    add.u64 %min_addr, %out, 16;
    atom.global.min.s32 %ignored32, [%min_addr], %min;
    add.u64 %max_addr, %out, 20;
    atom.global.max.s32 %ignored32, [%max_addr], %max;

ret_done:
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }

    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = resident
        .primary()
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let function = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_between_stats", &ptx)?;

    // u64::MAX sentinel when there is no validity bitmap; otherwise the bounds-checked byte offset.
    let null_off_value = validity_bitmap_kernel_arg(null_bitmap_offset, row_count, resident)?;

    let block_dim = 256_u32;
    let grid_dim = if row_count == 0 {
        1
    } else {
        row_count.div_ceil(u64::from(block_dim)).min(1024) as u32
    };

    // Init struct H2D'd into the scratch on-stream BEFORE the kernel: count/sum start at 0 (atomic
    // add), min/max at the INT sentinels (atomic min/max). `initial` outlives the helper's covering
    // sync, so the async source stays valid until the copy completes.
    let initial = CudaI32StatsRaw {
        count: 0,
        sum: 0,
        min: i32::MAX,
        max: i32::MIN,
    };
    let mut output_bytes = [0_u8; std::mem::size_of::<CudaI32StatsRaw>()];
    launch_on_pooled_stream(resident, Some(&mut output_bytes), |stream, output_ptr| {
        let htod_rc = unsafe {
            htod_async(
                output_ptr,
                (&initial as *const CudaI32StatsRaw).cast::<c_void>(),
                std::mem::size_of::<CudaI32StatsRaw>(),
                stream,
            )
        };
        if htod_rc != 0 {
            return htod_rc;
        }
        let mut resident_arg = resident.device_ptr();
        let mut offset_arg = byte_offset;
        let mut rows_arg = row_count;
        let mut lower_arg = lower_inclusive;
        let mut upper_arg = upper_inclusive;
        let mut output_arg = output_ptr;
        let mut null_off_arg = null_off_value;
        let mut args = [
            (&mut resident_arg as *mut u64).cast::<c_void>(),
            (&mut offset_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut lower_arg as *mut i32).cast::<c_void>(),
            (&mut upper_arg as *mut i32).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut null_off_arg as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                function,
                grid_dim,
                1,
                1,
                block_dim,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    })?;

    let raw = CudaI32StatsRaw {
        count: u64::from_le_bytes(output_bytes[0..8].try_into().unwrap()),
        sum: i64::from_le_bytes(output_bytes[8..16].try_into().unwrap()),
        min: i32::from_le_bytes(output_bytes[16..20].try_into().unwrap()),
        max: i32::from_le_bytes(output_bytes[20..24].try_into().unwrap()),
    };

    Ok(CudaI32Stats {
        count: raw.count,
        sum: raw.sum,
        min: (raw.count > 0).then_some(raw.min),
        max: (raw.count > 0).then_some(raw.max),
    })
}

fn launch_cuda_resident_i32_project(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
) -> Result<Vec<i32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;

    // P2-M2 — projecting a resident int4 column is a pure column read: the values returned ARE the
    // resident column bytes. The old route ran a single-thread `(1,1,1)` "identity copy" kernel
    // (resident -> a fresh device buffer) and THEN D2H'd that buffer — a redundant device->device
    // copy (2x memory traffic) behind a per-call `cuModuleLoadData` (re-JIT) + two `cuMemAlloc`. We
    // drop the kernel entirely and D2H the resident column straight into the host result on a pooled
    // private stream: `cu_memcpy_dtoh_async` keeps the copy OFF the synchronizing null stream, and
    // `launch_on_pooled_stream` event-times it (telemetry), covering-syncs, and drains the stream
    // before any error propagates. Blocking fallback for drivers lacking the async symbol. Net: 1x
    // traffic, no kernel, no JIT, no output alloc; the public signature and callers are unchanged.
    let end_offset = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if end_offset > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            end_offset as usize,
        ));
    }
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let len = usize::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let copy_bytes = len
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let src = resident
        .device_ptr()
        .checked_add(byte_offset)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut values = vec![0_i32; len];
    let dst = values.as_mut_ptr().cast::<c_void>();

    if let Some(dtoh_async) = resident.primary().cu_memcpy_dtoh_async {
        // Whole-column D2H staged on a pooled private stream. `dst` is a raw pointer into `values`
        // (no live borrow); the helper covering-syncs before it returns, so the GPU copy completes
        // before `values` is read below — no use-after-free, no observation of partial data.
        launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
            dtoh_async(dst, src, copy_bytes, stream)
        })?;
    } else {
        // Old-driver fallback (async D2H symbol absent): one blocking copy.
        let cu_memcpy_dtoh = unsafe {
            resident
                .lib()
                .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
                .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
                .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
        };
        check_cuda(unsafe { cu_memcpy_dtoh(dst, src, copy_bytes) })?;
    }
    Ok(values)
}

/// Single-thread `(1,1,1)` serial identity-copy projection — retained ONLY (under `#[cfg(test)]`)
/// as the on-GPU A/B parity + perf baseline for the P2-M2 kernel-less project migration. The
/// production route is the kernel-less `launch_cuda_resident_i32_project` (a direct pooled-stream
/// D2H of the resident column); this is the previously-shipped device->device copy kernel kept as a
/// device-side reference oracle (no CPU operator re-implementation).
#[cfg(test)]
fn launch_cuda_resident_i32_project_serial(
    resident: &CudaResidentDeviceMemory,
    byte_offset: u64,
    row_count: u64,
) -> Result<Vec<i32>, CudaRuntimeProbeError> {
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_resident_i32_project(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .u64 out_values_ptr,
    .param .u64 out_count_ptr
)
{
    .reg .pred %p_done;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %out_values;
    .reg .u64 %out_count;
    .reg .u64 %base;
    .reg .u64 %idx;
    .reg .u64 %input_addr;
    .reg .u64 %output_addr;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %out_values, [out_values_ptr];
    ld.param.u64 %out_count, [out_count_ptr];

    add.u64 %base, %resident, %offset;
    mov.u64 %idx, 0;

loop:
    setp.ge.u64 %p_done, %idx, %rows;
    @%p_done bra done;
    mul.lo.u64 %input_addr, %idx, 4;
    add.u64 %input_addr, %base, %input_addr;
    ld.global.s32 %r_value, [%input_addr];
    mul.lo.u64 %output_addr, %idx, 4;
    add.u64 %output_addr, %out_values, %output_addr;
    st.global.s32 [%output_addr], %r_value;
    add.u64 %idx, %idx, 1;
    bra loop;

done:
    st.global.u64 [%out_count], %rows;
    ret;
}
"#;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let cu_mem_alloc = unsafe {
        resident
            .lib()
            .get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| resident.lib().get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        resident
            .lib()
            .get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| resident.lib().get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        resident
            .lib()
            .get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        resident
            .lib()
            .get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        resident
            .lib()
            .get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        resident
            .lib()
            .get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let value_bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let mut device_values = 0_u64;
    let mut device_count = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_values, value_bytes) })?;
    let values_guard = CudaDeviceAllocationGuard {
        ptr: device_values,
        free: *cu_mem_free,
    };
    check_cuda(unsafe { cu_mem_alloc(&mut device_count, std::mem::size_of::<u64>()) })?;
    let count_guard = CudaDeviceAllocationGuard {
        ptr: device_count,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_resident_i32_project".as_ptr(),
        )
    })?;

    let mut resident_arg = resident.device_ptr();
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut values_arg = values_guard.ptr;
    let mut count_arg = count_guard.ptr;
    let mut args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut values_arg as *mut u64).cast::<c_void>(),
        (&mut count_arg as *mut u64).cast::<c_void>(),
    ];
    launch_with_optional_cuda_event_timing(resident, *cu_ctx_synchronize, || unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;

    let mut copied_count = 0_u64;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut copied_count as *mut u64).cast::<c_void>(),
            count_guard.ptr,
            std::mem::size_of::<u64>(),
        )
    })?;
    let copied_len = usize::try_from(copied_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if copied_count > row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(copied_len));
    }
    let mut values = vec![0_i32; copied_len];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            values.as_mut_ptr().cast::<c_void>(),
            values_guard.ptr,
            copied_len * std::mem::size_of::<i32>(),
        )
    })?;

    drop(module_guard);
    drop(count_guard);
    drop(values_guard);
    Ok(values)
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");

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

/// Compare an int4 value buffer (absolute device ptr) to `needle` and return the matching row
/// indices, host-sorted ascending. Shared compare-and-compact tail for the arithmetic VM (the 2-col
/// fast-path launcher keeps its own fused copy). `comparison`: 0=eq, 1=lt, 2=le, 3=gt, 4=ge.
fn compact_buffer_i32_compare_to_indices(
    resident: &CudaResidentDeviceMemory,
    value_device_ptr: u64,
    n: u64,
    needle: i32,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    // The scalar compact kernel switched on codes 0=eq..4=ge only; reject 5=ne (mask-path only) so a
    // caller gets an error, not a silently-empty result. Preserved byte-identically across the
    // re-route to the ordered compaction (no production caller passes 5 here — `ne` lowers to a mask).
    if comparison > 4 {
        return Err(CudaRuntimeProbeError::UnsupportedComparison(comparison));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    // probe-timing (VM lever): the int4 simple-comparison compaction (`col <cmp> scalar` -> indices). The
    // legacy path was a fused compare+atomic-append kernel + count/index D2H + a host `sort_unstable` of
    // the surviving indices; the ~93%-of-cost host sort is what the ordered compaction eliminates.
    let _compact_scope = Probe::scope("compact");

    // Re-route to the ORDERED parallel compaction (`COMPARE_ORDERED_PTX`) in INDEX-emit mode: read the
    // leased VALUE buffer as the contiguous i32 input (`input_base = value_device_ptr`, `byte_offset =
    // 0`). The kernels read one i32 per row at `value_device_ptr + idx*4` (the same layout the legacy
    // compact kernel read), so the surviving row indices are identical — but emitted ASCENDING BY
    // CONSTRUCTION (contiguous block partition + ordered intra-block prefix sum), replacing the host
    // `sort_unstable`. Call the shared core directly (not the column-only index wrapper, which always
    // reads `resident.device_ptr()`) so the input base is the value buffer; reinterpret each u32 row
    // index from its i32 slot bits (always `< n`, non-negative — bit-exact) exactly as that wrapper
    // does.
    //
    // Lease lifetime: the ordered core does count -> host-scan -> scatter (TWO launches reading
    // `value_device_ptr`). The caller leases the value buffer and holds that lease across this whole
    // call (it is borrowed for the duration), so the input outlives both reads — unchanged from the
    // legacy single-launch path, which also read the same buffer.
    let slots = launch_cuda_resident_i32_compare_ordered_core(
        resident,
        value_device_ptr,
        0,
        n,
        needle,
        comparison,
        1,
    )?;
    Ok(slots.into_iter().map(|slot| slot as u32).collect())
}

/// PROTOTYPE — the resident arithmetic bytecode VM (docs/architecture/17 section 2.3). Runs a postfix
/// `program` over a stack of leased device buffers to evaluate an ARBITRARY int4 arithmetic tree on
/// device (each step launches one buffer->buffer primitive), then compares the single result buffer
/// to `needle` and returns the matching row indices. This is the general recursive interpreter that
/// generalizes the fixed 2-col fast-path: arbitrary depth lowers to a longer program of the same
/// primitives. Correctness-first: each step runs on a pooled stream that syncs, so an intermediate
/// is valid before the next step reads it (pipelining/fusion is a later perf lever).
/// Compare two int4 value buffers (absolute device ptrs) elementwise and return the matching row
/// indices in ASCENDING order. The col-vs-col / expr-vs-expr analogue of
/// `compact_buffer_i32_compare_to_indices`. `comparison`: 0=eq, 1=lt, 2=le, 3=gt, 4=ge.
fn compact_buffers_i32_compare_to_indices(
    resident: &CudaResidentDeviceMemory,
    lhs_device_ptr: u64,
    rhs_device_ptr: u64,
    n: u64,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    // The buffer compact kernel switches on codes 0=eq..4=ge only; reject 5=ne (mask-path only) so a
    // caller gets an error, not a silently-empty result. Preserved byte-identically across the re-route
    // to the ordered two-input compaction (no production caller passes 5 here - `ne` lowers to a mask).
    if comparison > 4 {
        return Err(CudaRuntimeProbeError::UnsupportedComparison(comparison));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    // Re-route to the TWO-INPUT ORDERED parallel compaction (`COMPARE_ORDERED_PTX`,
    // `gpu_db_buffers_i32_compare_*_blocks`): compare `lhs[i] <cmp> rhs[i]` over the two leased value
    // buffers (`lhs_device_ptr`, `rhs_device_ptr`, each contiguous i32 at `base + idx*4` - the same
    // layout the legacy `gpu_db_buffer_i32_compare_buffers_to_indices` atomic kernel read), so the
    // surviving row indices are IDENTICAL (operand order lhs,rhs; codes 0..4) - but emitted ASCENDING
    // BY CONSTRUCTION (contiguous block partition + ordered intra-block prefix sum), replacing the
    // atomic-append + host `sort_unstable`.
    //
    // Lease lifetime: the ordered launch does count -> host-scan -> scatter (TWO launches, each reading
    // BOTH buffers). The caller leases both value buffers and holds those leases across this whole call
    // (borrowed for the duration), so both inputs outlive both reads - unchanged from the legacy
    // single-launch path, which also read the same two buffers.
    launch_cuda_resident_i32_compare_buffers_indices_ordered(
        resident,
        lhs_device_ptr,
        rhs_device_ptr,
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
    let mut stack = run_resident_arith_program(resident, program, &[], n, ResidentElemType::I32)?;
    let value = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        // A well-formed arithmetic program leaves exactly one value on the stack.
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    compact_buffer_i32_compare_to_indices(resident, value.ptr, n, needle, compare_code)
}

/// Evaluate an arithmetic `program` over all `n_rows`, then GATHER the resulting i32 value column at
/// the survivor `indices` (sign-extended to i64). The ORDER BY-expression key column: the device Expr
/// interpreter computes `a+b` etc. with CHECKED int4 arithmetic (overflow -> `IntegerOutOfRange`,
/// inherited from `run_resident_arith_program` -- no wrap, no CPU), and the result feeds the GPU sort
/// exactly like a materialized int key.
/// A resident arith-program result kept ON-DEVICE (the value buffer, one element per row) + its pooled
/// lease. Returned by [`CudaResidentDeviceMemory::arith_value_column_device`]; `device_ptr()` is read by
/// a LATER kernel launch (the GROUP BY group-key via `key_base_override`), so this MUST be kept alive
/// across every such launch -- dropping it returns the buffer to the pool (a UAF under reuse).
pub struct DeviceArithBuffer<'a> {
    _lease: PooledBufferLease<'a>,
    ptr: u64,
    initialized_bytes: u64,
}

impl DeviceArithBuffer<'_> {
    fn new(lease: PooledBufferLease<'_>, initialized_bytes: usize) -> DeviceArithBuffer<'_> {
        debug_assert!(initialized_bytes <= lease.capacity);
        let ptr = lease.ptr;
        DeviceArithBuffer {
            _lease: lease,
            ptr,
            initialized_bytes: initialized_bytes as u64,
        }
    }

    /// Device address of the value buffer (one element per row; width = the program's element type).
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Borrow this allocation with its exact initialized extent and originating CUDA context.
    pub fn group_view(&self) -> CudaGroupDeviceView<'_> {
        CudaGroupDeviceView::new(
            self.ptr,
            self.initialized_bytes,
            self._lease.primary_identity(),
        )
    }
}

/// Run an arith program over all `n_rows` and return the result value buffer RESIDENT (no D2H). See
/// [`CudaResidentDeviceMemory::arith_value_column_device`]. cuCtxSynchronize'd so a later launch reads
/// valid keys. Checked overflow -> PG error inherited from `run_resident_arith_program`.
fn launch_cuda_arith_value_column_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    program: &[ExprStep],
    n_rows: u64,
    elem: ResidentElemType,
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    let n = usize::try_from(n_rows).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let mut stack = run_resident_arith_program(resident, program, &[], n_rows, elem)?;
    let value = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        // A well-formed arithmetic program leaves exactly one value on the stack.
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    // Block until the arith program finishes, so the SEPARATE GROUP BY launch that reads this buffer
    // sees the completed keys (not a racing/stale buffer).
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    let initialized_bytes = n
        .checked_mul(elem.elem_size())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n))?;
    Ok(DeviceArithBuffer::new(value, initialized_bytes))
}

/// Run `gpu_db_resident_bool_to_mask` (negate=0) into a leased int4 buffer (it writes 0/1 per row) and
/// return it as a `DeviceArithBuffer` -- the derived int4 column for a bool GROUP BY key / bool MIN/MAX
/// value. Synchronizes so the SEPARATE GROUP BY launch (which reads it via key/value_base_override) sees
/// the completed buffer, not a racing/stale one. The lease lives in the returned buffer (caller-owned).
fn launch_cuda_bool_to_int4_column_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    bitmap_byte_offset: u64,
    n: u64,
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let out_bytes = n_usize
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
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_resident_bool_to_mask", &ptx)?;
    let out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = bitmap_byte_offset;
    let mut a2 = 0u32; // negate = false: bool true -> int4 1, false -> 0
    let mut a3 = n;
    let mut a4 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u32).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            kernel_fn,
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
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok(DeviceArithBuffer::new(out, out_bytes))
}

/// Run `gpu_db_pack_two_int4_cols` into a leased i64 buffer (col0<<32 | col1 per row) and return it as a
/// `DeviceArithBuffer` -- the derived composite GROUP BY key. cuCtxSynchronize'd so the SEPARATE GROUP BY
/// launch (which reads it via key_base_override) sees the completed buffer, not a racing/stale one.
fn launch_cuda_pack_two_int4_cols_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    off0: u64,
    off1: u64,
    n: u64,
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let out_bytes = n_usize
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_pack_two_int4_cols", &ptx)?;
    let out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = off0;
    let mut a2 = off1;
    let mut a3 = n;
    let mut a4 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            kernel_fn,
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
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok(DeviceArithBuffer::new(out, out_bytes))
}

/// Run `gpu_db_pack_two_cols_i128` into a leased [i128; n] buffer (col0 high 64 bits, col1 low 64) and
/// return it as a `DeviceArithBuffer` -- the derived composite GROUP BY key for a wider (int8/timestamp
/// member) composite. `w0`/`w1` are each member's read width (4 or 8). cuCtxSynchronize'd so the
/// SEPARATE b128 GROUP BY launch (key_is_i128 + key_base_override) sees the completed buffer.
fn launch_cuda_pack_two_cols_i128_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    off0: u64,
    w0: u64,
    off1: u64,
    w1: u64,
    n: u64,
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if (w0 != 4 && w0 != 8) || (w1 != 4 && w1 != 8) {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let out_bytes = n_usize
        .checked_mul(std::mem::size_of::<i128>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_pack_two_cols_i128", &ptx)?;
    let out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = off0;
    let mut a2 = w0;
    let mut a3 = off1;
    let mut a4 = w1;
    let mut a5 = n;
    let mut a6 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
        (&mut a5 as *mut u64).cast::<c_void>(),
        (&mut a6 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            kernel_fn,
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
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok(DeviceArithBuffer::new(out, out_bytes))
}

/// Run `gpu_db_widen_col_to_i64` into a leased [i64; n] buffer (the column sign-extended to i64) for the
/// fixed member of a composite (fixed-width, text) GROUP BY key. `w` = 4 or 8. cuCtxSynchronize'd so the
/// SEPARATE text-key GROUP BY launch reads the completed buffer via key_base_override.
fn launch_cuda_widen_col_to_i64_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    off: u64,
    w: u64,
    n: u64,
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    if n == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if w != 4 && w != 8 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let out_bytes = n_usize
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_widen_col_to_i64", &ptx)?;
    let out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    let mut a0 = resident.device_ptr();
    let mut a1 = off;
    let mut a2 = w;
    let mut a3 = n;
    let mut a4 = out.ptr;
    let mut args = [
        (&mut a0 as *mut u64).cast::<c_void>(),
        (&mut a1 as *mut u64).cast::<c_void>(),
        (&mut a2 as *mut u64).cast::<c_void>(),
        (&mut a3 as *mut u64).cast::<c_void>(),
        (&mut a4 as *mut u64).cast::<c_void>(),
    ];
    launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
        cu_launch_kernel(
            kernel_fn,
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
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok(DeviceArithBuffer::new(out, out_bytes))
}

/// Run `gpu_db_build_wide_key` into a leased [u8; wbytes*n] buffer (the all-fixed composite wide key per
/// row) and return it. `descriptors` = (kind, src_off, dst_off) per member, uploaded as 3 u64 each.
/// cuCtxSynchronize'd so the SEPARATE GROUP BY launch reads the completed buffer via key_base_override.
fn launch_cuda_upload_u64_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    data: &[u64],
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    if data.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let bytes = std::mem::size_of_val(data);
    let primary = resident.primary();
    primary.set_current()?;
    let buf = primary.lease_device_buffer(bytes)?;
    let cu_memcpy_htod = unsafe {
        *primary
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| primary.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    // cuMemcpyHtoD is host-synchronous: the data is fully on device when it returns, so the GROUP BY
    // kernel (on the pooled stream) sees it without a further sync.
    check_cuda(unsafe { cu_memcpy_htod(buf.ptr, data.as_ptr().cast::<c_void>(), bytes) })?;
    Ok(DeviceArithBuffer::new(buf, bytes))
}

fn launch_cuda_build_wide_key_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    descriptors: &[(u64, u64, u64)],
    wbytes: u64,
    n: u64,
    // A separate per-row buffer for a DERIVED member (the expression group key); read by descriptor
    // kinds 4 (i32) / 5 (i64). 0 when the wide key has only column members.
    derived_ptr: u64,
    // M3 (doc 21): per-member NULL validity offsets (EMPTY = none). See the wrapper's doc.
    validity_descs: &[u64],
) -> Result<DeviceArithBuffer<'r>, CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    if n == 0 || descriptors.is_empty() || wbytes == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let wbytes_usize =
        usize::try_from(wbytes).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    let out_bytes = n_usize
        .checked_mul(wbytes_usize)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    // Flatten the descriptors to a [u64] (kind, src_off, dst_off per member) for the device upload.
    let mut desc_flat: Vec<u64> = Vec::with_capacity(descriptors.len() * 3);
    for &(kind, src_off, dst_off) in descriptors {
        desc_flat.push(kind);
        desc_flat.push(src_off);
        desc_flat.push(dst_off);
    }
    let desc_bytes = std::mem::size_of_val(desc_flat.as_slice());
    let n_members = descriptors.len() as u64;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_build_wide_key", &ptx)?;
    let desc_dev = primary.lease_device_buffer(desc_bytes)?;
    // M3 (doc 21): the per-member validity offsets (one u64 per member). Leased + uploaded only when
    // present (nullable composite); else `vdesc_ptr` stays 0 (the kernel skips all validity handling).
    let vdesc_bytes = std::mem::size_of_val(validity_descs);
    let vdesc_dev = if vdesc_bytes > 0 {
        Some(primary.lease_device_buffer(vdesc_bytes)?)
    } else {
        None
    };
    let vdesc_ptr = vdesc_dev.as_ref().map_or(0, |b| b.ptr);
    let out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535) as u32;
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                desc_dev.ptr,
                desc_flat.as_ptr().cast::<c_void>(),
                desc_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        if let Some(vdev) = &vdesc_dev {
            let rc = unsafe {
                htod_async(
                    vdev.ptr,
                    validity_descs.as_ptr().cast::<c_void>(),
                    vdesc_bytes,
                    stream,
                )
            };
            if rc != 0 {
                return rc;
            }
        }
        let mut a0 = resident.device_ptr();
        let mut a1 = desc_dev.ptr;
        let mut a2 = n_members;
        let mut a3 = wbytes;
        let mut a4 = n;
        let mut a5 = out.ptr;
        let mut a6 = derived_ptr;
        let mut a7 = vdesc_ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast::<c_void>(),
            (&mut a1 as *mut u64).cast::<c_void>(),
            (&mut a2 as *mut u64).cast::<c_void>(),
            (&mut a3 as *mut u64).cast::<c_void>(),
            (&mut a4 as *mut u64).cast::<c_void>(),
            (&mut a5 as *mut u64).cast::<c_void>(),
            (&mut a6 as *mut u64).cast::<c_void>(),
            (&mut a7 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                kernel_fn,
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
        }
    })?;
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok(DeviceArithBuffer::new(out, out_bytes))
}

/// COUNT(DISTINCT v) mark pass (see [`CudaResidentDeviceMemory::mark_new_distinct_device`]). Uploads
/// the sorted `k`-wide i64 tuple matrix (key0 = group key) + the permutation, runs
/// `gpu_db_mark_new_distinct`, and returns
/// the two derived i64 device columns `(g_sorted, new_distinct)`. cuCtxSynchronize'd so the SEPARATE
/// GROUP BY launch reads completed buffers (a fully-drained launch, off the bool-GROUP-BY hazard). The
/// `keys`/`perm` upload buffers are temporary -- the synchronize guarantees the kernel read them before
/// they return to the pool at function end; the output leases live in the returned buffers.
fn launch_cuda_mark_new_distinct_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    keys: &[i64],
    perm: &[u32],
    n: u64,
    k: usize,
) -> Result<(DeviceArithBuffer<'r>, DeviceArithBuffer<'r>), CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n_usize == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if k < 2 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(k));
    }
    let expected_keys = n_usize
        .checked_mul(k)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    if keys.len() != expected_keys {
        return Err(CudaRuntimeProbeError::InvalidInputLength(keys.len()));
    }
    if perm.len() != n_usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(perm.len()));
    }
    let keys_bytes = keys
        .len()
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(keys.len()))?;
    let perm_bytes = n_usize
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let out_bytes = n_usize
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_mark_new_distinct", &ptx)?;
    let keys_dev = primary.lease_device_buffer(keys_bytes)?;
    let perm_dev = primary.lease_device_buffer(perm_bytes)?;
    let g_out = primary.lease_device_buffer(out_bytes)?;
    let nd_out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = (n_usize.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                keys_dev.ptr,
                keys.as_ptr().cast::<c_void>(),
                keys_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                perm_dev.ptr,
                perm.as_ptr().cast::<c_void>(),
                perm_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let mut a0 = keys_dev.ptr;
        let mut a1 = perm_dev.ptr;
        let mut a2 = n;
        let mut a3 = k as u64;
        let mut a4 = g_out.ptr;
        let mut a5 = nd_out.ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast::<c_void>(),
            (&mut a1 as *mut u64).cast::<c_void>(),
            (&mut a2 as *mut u64).cast::<c_void>(),
            (&mut a3 as *mut u64).cast::<c_void>(),
            (&mut a4 as *mut u64).cast::<c_void>(),
            (&mut a5 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                kernel_fn,
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
        }
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok((
        DeviceArithBuffer::new(g_out, out_bytes),
        DeviceArithBuffer::new(nd_out, out_bytes),
    ))
}

/// COUNT(DISTINCT v) mark pass for a TEXT value (see
/// [`CudaResidentDeviceMemory::mark_new_distinct_text_device`]). Uploads the hetero-sort permutation,
/// the surviving absolute rows, and the per-position group key; runs `gpu_db_mark_new_distinct_text`
/// (which reads the value text from the resident payload via text_off/text_bytes), and returns the two
/// derived i64 device columns `(g_sorted, new_distinct)`. cuCtxSynchronize'd so the SEPARATE GROUP BY
/// launch reads completed buffers (a fully-drained launch, off the bool-GROUP-BY hazard).
#[allow(clippy::too_many_arguments)]
fn launch_cuda_mark_new_distinct_text_device<'r>(
    resident: &'r CudaResidentDeviceMemory,
    perm: &[u32],
    indices: &[u64],
    g_keys: &[i64],
    text_off: u64,
    text_bytes: u64,
    n: u64,
) -> Result<(DeviceArithBuffer<'r>, DeviceArithBuffer<'r>), CudaRuntimeProbeError> {
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
    let n_usize = usize::try_from(n).map_err(|_| CudaRuntimeProbeError::InvalidInputLength(0))?;
    if n_usize == 0 {
        return Err(CudaRuntimeProbeError::InvalidInputLength(0));
    }
    if perm.len() != n_usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(perm.len()));
    }
    if indices.len() != n_usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(indices.len()));
    }
    if g_keys.len() != n_usize {
        return Err(CudaRuntimeProbeError::InvalidInputLength(g_keys.len()));
    }
    let perm_bytes = n_usize
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let indices_bytes = n_usize
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let gkeys_bytes = n_usize
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(n_usize))?;
    let out_bytes = gkeys_bytes;
    let primary = resident.primary();
    primary.set_current()?;
    let cu_launch_kernel = unsafe {
        resident
            .lib()
            .get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let htod_async = primary
        .cu_memcpy_htod_async
        .ok_or(CudaRuntimeProbeError::DriverLibraryUnavailable)?;
    let cu_ctx_synchronize = unsafe {
        resident
            .lib()
            .get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let kernel_fn = primary.cached_function(c"gpu_db_mark_new_distinct_text", &ptx)?;
    let resident_base = resident.device_ptr();
    let perm_dev = primary.lease_device_buffer(perm_bytes)?;
    let indices_dev = primary.lease_device_buffer(indices_bytes)?;
    let gkeys_dev = primary.lease_device_buffer(gkeys_bytes)?;
    let g_out = primary.lease_device_buffer(out_bytes)?;
    let nd_out = primary.lease_device_buffer(out_bytes)?;
    const BLOCK: u32 = 256;
    let grid = (n_usize.div_ceil(BLOCK as usize) as u32).clamp(1, 65_535);
    launch_on_pooled_stream(resident, None, |stream, _scratch| {
        let rc = unsafe {
            htod_async(
                perm_dev.ptr,
                perm.as_ptr().cast::<c_void>(),
                perm_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                indices_dev.ptr,
                indices.as_ptr().cast::<c_void>(),
                indices_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let rc = unsafe {
            htod_async(
                gkeys_dev.ptr,
                g_keys.as_ptr().cast::<c_void>(),
                gkeys_bytes,
                stream,
            )
        };
        if rc != 0 {
            return rc;
        }
        let mut a0 = perm_dev.ptr;
        let mut a1 = indices_dev.ptr;
        let mut a2 = gkeys_dev.ptr;
        let mut a3 = resident_base;
        let mut a4 = text_off;
        let mut a5 = text_bytes;
        let mut a6 = n;
        let mut a7 = g_out.ptr;
        let mut a8 = nd_out.ptr;
        let mut args = [
            (&mut a0 as *mut u64).cast::<c_void>(),
            (&mut a1 as *mut u64).cast::<c_void>(),
            (&mut a2 as *mut u64).cast::<c_void>(),
            (&mut a3 as *mut u64).cast::<c_void>(),
            (&mut a4 as *mut u64).cast::<c_void>(),
            (&mut a5 as *mut u64).cast::<c_void>(),
            (&mut a6 as *mut u64).cast::<c_void>(),
            (&mut a7 as *mut u64).cast::<c_void>(),
            (&mut a8 as *mut u64).cast::<c_void>(),
        ];
        unsafe {
            cu_launch_kernel(
                kernel_fn,
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
        }
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;
    Ok((
        DeviceArithBuffer::new(g_out, out_bytes),
        DeviceArithBuffer::new(nd_out, out_bytes),
    ))
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
    let mut stack = run_resident_arith_program(resident, program, &[], n_rows, elem)?;
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
    const PTX: &[u8] = include_bytes!("expr_proto.ptx");
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
    let mut value_stack =
        run_resident_arith_program(resident, program, &[], n_rows, ResidentElemType::I32)?;
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
    let mut stack = run_resident_arith_program(resident, program, &[], n, ResidentElemType::I32)?;
    let rhs = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    let lhs = stack
        .pop()
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?;
    if !stack.is_empty() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(program.len()));
    }
    compact_buffers_i32_compare_to_indices(resident, lhs.ptr, rhs.ptr, n, comparison)
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

/// Ordered parallel-compaction PTX shared by the VALUE-emit launch
/// (`launch_cuda_resident_i32_compare_project`) and the INDEX-emit launch
/// (`launch_cuda_resident_i32_compare_indices_ordered`). One module, two `.entry`
/// kernels (`..._count_blocks` Pass A, `..._scatter_blocks` Pass B). The scatter kernel's
/// trailing `out_is_index` param selects what it stores at each ascending output slot: the
/// matching i32 VALUE (mode 0) or the surviving ROW INDEX as u32 (mode != 0). Both kernels
/// support comparison codes 0=eq, 1=lt, 2=lte, 3=gt, 4=gte, 5=ne (eq/ne are folded into
/// `p_match` IDENTICALLY in count and scatter, so Pass A's per-block count equals Pass B's
/// per-block scatter count for every predicate).
const COMPARE_ORDERED_PTX: &[u8] = br#"
.version 6.0
.target sm_60
.address_size 64

.visible .entry gpu_db_resident_i32_compare_count_blocks(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .u64 chunk_rows,
    .param .s32 needle,
    .param .u32 comparison,
    .param .u64 out_block_counts_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_lt;
    .reg .pred %p_lte;
    .reg .pred %p_gt;
    .reg .pred %p_gte;
    .reg .pred %p_eq;
    .reg .pred %p_ne;
    .reg .pred %p_code_lt;
    .reg .pred %p_code_lte;
    .reg .pred %p_code_gt;
    .reg .pred %p_code_gte;
    .reg .pred %p_code_eq;
    .reg .pred %p_code_ne;
    .reg .pred %p_match;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %comparison;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %chunk;
    .reg .u64 %base;
    .reg .u64 %block_counts;
    .reg .u64 %start;
    .reg .u64 %end;
    .reg .u64 %idx;
    .reg .u64 %tmp64;
    .reg .u64 %input_addr;
    .reg .u64 %matches;
    .reg .u64 %count_addr;
    .reg .s32 %needle;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %chunk, [chunk_rows];
    ld.param.s32 %needle, [needle];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %block_counts, [out_block_counts_ptr];

    add.u64 %base, %resident, %offset;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;

    // start = bid * chunk; end = min(start + chunk, rows)
    cvt.u64.u32 %tmp64, %bid;
    mul.lo.u64 %start, %tmp64, %chunk;
    add.u64 %end, %start, %chunk;
    setp.gt.u64 %p_done, %end, %rows;
    @%p_done mov.u64 %end, %rows;

    // idx = start + lane; stride = blockDim (grid-stride WITHIN this block's range)
    cvt.u64.u32 %tmp64, %lane;
    add.u64 %idx, %start, %tmp64;
    cvt.u64.u32 %tmp64, %bdim;

    mov.u64 %matches, 0;

loop:
    setp.ge.u64 %p_done, %idx, %end;
    @%p_done bra done;
    mul.lo.u64 %input_addr, %idx, 4;
    add.u64 %input_addr, %base, %input_addr;
    ld.global.s32 %r_value, [%input_addr];
    setp.lt.s32 %p_lt, %r_value, %needle;
    setp.le.s32 %p_lte, %r_value, %needle;
    setp.gt.s32 %p_gt, %r_value, %needle;
    setp.ge.s32 %p_gte, %r_value, %needle;
    setp.eq.s32 %p_eq, %r_value, %needle;
    setp.ne.s32 %p_ne, %r_value, %needle;
    setp.eq.u32 %p_code_eq, %comparison, 0;
    setp.eq.u32 %p_code_lt, %comparison, 1;
    setp.eq.u32 %p_code_lte, %comparison, 2;
    setp.eq.u32 %p_code_gt, %comparison, 3;
    setp.eq.u32 %p_code_gte, %comparison, 4;
    setp.eq.u32 %p_code_ne, %comparison, 5;
    mov.pred %p_match, 0;
    and.pred %p_eq, %p_eq, %p_code_eq;
    or.pred %p_match, %p_match, %p_eq;
    and.pred %p_lt, %p_lt, %p_code_lt;
    or.pred %p_match, %p_match, %p_lt;
    and.pred %p_lte, %p_lte, %p_code_lte;
    or.pred %p_match, %p_match, %p_lte;
    and.pred %p_gt, %p_gt, %p_code_gt;
    or.pred %p_match, %p_match, %p_gt;
    and.pred %p_gte, %p_gte, %p_code_gte;
    or.pred %p_match, %p_match, %p_gte;
    and.pred %p_ne, %p_ne, %p_code_ne;
    or.pred %p_match, %p_match, %p_ne;
    @!%p_match bra next;
    add.u64 %matches, %matches, 1;

next:
    add.u64 %idx, %idx, %tmp64;
    bra loop;

done:
    // Accumulate this thread's local matches into the block's slot. Threads of a block race here,
    // but ADDITION is commutative so the per-block TOTAL is order-independent (the ordering is
    // established by the contiguous partition + pass B, never by this reduction).
    cvt.u64.u32 %tmp64, %bid;
    mul.lo.u64 %count_addr, %tmp64, 8;
    add.u64 %count_addr, %block_counts, %count_addr;
    red.global.add.u64 [%count_addr], %matches;
    ret;
}

// Pass B (PARALLEL intra-block ordered compaction). Block `b` owns the contiguous row range
// `[b*chunk, min(b*chunk+chunk, rows))` and ALL `blockDim` threads cooperate (the prior version used
// ONE thread per block, a serial re-scan ~= the kernel floor). Per iteration of the within-block
// grid-stride (one row per thread per iteration), every thread computes a 0/1 match `%flag`, then an
// ORDERED intra-block EXCLUSIVE prefix-sum of the flags assigns each matching row its within-block
// rank; the value is scattered to `block_base[b] + running + rank`. The scan is monotonic in row
// index, so the output is ASCENDING by row - byte-identical to the serial append for ANY shape.
//
// The intra-block scan is two-level (warp-shuffle + a tiny shared cross-warp combine): (1) a warp
// inclusive scan of `%flag` via `shfl.sync.up.b32` over the 32 lanes; warp-local exclusive =
// inclusive - own flag. (2) lane 31 of each warp writes its warp total to `s_scan[warp]`; after a
// barrier WARP 0 exclusive-scans the warp totals `s_scan[0..nwarps)` (nwarps = blockDim/32 <= 32,
// since blockDim <= 1024, one warp suffices) and publishes per-warp exclusive prefixes to
// `s_scan[32+w]` plus the block total to `s_scan[64]`. (3) each thread's within-iteration offset =
// (warp-local exclusive) + (per-warp prefix). `%running` accumulates the per-iteration block total
// (matches from earlier stride iterations = strictly lower rows) and stays uniform across the block,
// so the multi-iteration `chunk > blockDim` path also stays ascending. Loop trip count is
// block-uniform (iter_base/end are uniform), so every `bar.sync` is reached by the whole block; each
// `shfl.sync` runs with the full warp converged (the only per-lane branch reconverges before it).
// Pass A and Pass B test the SAME predicate over the SAME range, so the per-block scatter count
// equals Pass A's per-block count (count <-> scatter consistency preserved).
.visible .entry gpu_db_resident_i32_compare_scatter_blocks(
    .param .u64 resident_ptr,
    .param .u64 byte_offset,
    .param .u64 row_count,
    .param .u64 chunk_rows,
    .param .s32 needle,
    .param .u32 comparison,
    .param .u64 block_base_ptr,
    .param .u64 out_values_ptr,
    .param .u32 out_is_index
)
{
    // shared scratch: [0..32) warp totals, [32..64) per-warp exclusive prefixes, [64] block total.
    .shared .align 4 .b32 s_scan[65];

    .reg .pred %p_done;
    .reg .pred %p_lt;
    .reg .pred %p_lte;
    .reg .pred %p_gt;
    .reg .pred %p_gte;
    .reg .pred %p_eq;
    .reg .pred %p_ne;
    .reg .pred %p_code_lt;
    .reg .pred %p_code_lte;
    .reg .pred %p_code_gt;
    .reg .pred %p_code_gte;
    .reg .pred %p_code_eq;
    .reg .pred %p_code_ne;
    .reg .pred %p_match;
    .reg .pred %p_recv;
    .reg .pred %p_inrange;
    .reg .pred %p_islast;
    .reg .pred %p_warp0;
    .reg .pred %p_lane_in;
    .reg .pred %p_is_index;
    .reg .u32 %thr;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %lane;
    .reg .u32 %warp;
    .reg .u32 %nwarps;
    .reg .u32 %comparison;
    .reg .u32 %flag;
    .reg .u32 %incl;
    .reg .u32 %recv;
    .reg .u32 %wexcl;
    .reg .u32 %wtot;
    .reg .u32 %prefix;
    .reg .u32 %btot;
    .reg .u32 %off32;
    .reg .u32 %tmp32;
    .reg .u32 %row_u32;
    .reg .u32 %store_val;
    .reg .u64 %resident;
    .reg .u64 %offset;
    .reg .u64 %rows;
    .reg .u64 %chunk;
    .reg .u64 %base;
    .reg .u64 %block_base;
    .reg .u64 %out_values;
    .reg .u64 %start;
    .reg .u64 %end;
    .reg .u64 %row;
    .reg .u64 %iter_base;
    .reg .u64 %stride;
    .reg .u64 %tmp64;
    .reg .u64 %input_addr;
    .reg .u64 %slot;
    .reg .u64 %output_addr;
    .reg .u64 %base_addr;
    .reg .u64 %running;
    .reg .u64 %sh_addr;
    .reg .s32 %needle;
    .reg .s32 %r_value;

    ld.param.u64 %resident, [resident_ptr];
    ld.param.u64 %offset, [byte_offset];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %chunk, [chunk_rows];
    ld.param.s32 %needle, [needle];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %block_base, [block_base_ptr];
    ld.param.u64 %out_values, [out_values_ptr];
    ld.param.u32 %tmp32, [out_is_index];
    setp.ne.u32 %p_is_index, %tmp32, 0;

    add.u64 %base, %resident, %offset;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %thr, %tid.x;
    mov.u32 %bdim, %ntid.x;

    // lane = thr & 31; warp = thr >> 5; nwarps = (bdim + 31) >> 5
    and.b32 %lane, %thr, 31;
    shr.u32 %warp, %thr, 5;
    add.u32 %tmp32, %bdim, 31;
    shr.u32 %nwarps, %tmp32, 5;

    // start = bid * chunk; end = min(start + chunk, rows)
    cvt.u64.u32 %tmp64, %bid;
    mul.lo.u64 %start, %tmp64, %chunk;
    add.u64 %end, %start, %chunk;
    setp.gt.u64 %p_done, %end, %rows;
    @%p_done mov.u64 %end, %rows;

    // slot = block_base[bid]  (this block's exclusive-prefix base output index)
    mul.lo.u64 %base_addr, %tmp64, 8;
    add.u64 %base_addr, %block_base, %base_addr;
    ld.global.u64 %slot, [%base_addr];

    cvt.u64.u32 %stride, %bdim;
    // iter_base walks start, start+bdim, start+2*bdim, ...; row = iter_base + thr.
    mov.u64 %iter_base, %start;
    mov.u64 %running, 0;
    setp.eq.u32 %p_warp0, %warp, 0;

iter_loop:
    // Continue while the block still has rows to cover: iter_base < end (block-uniform trip count).
    setp.ge.u64 %p_done, %iter_base, %end;
    @%p_done bra iter_done;

    // row = iter_base + thr ; in-range = row < end
    cvt.u64.u32 %tmp64, %thr;
    add.u64 %row, %iter_base, %tmp64;
    setp.lt.u64 %p_inrange, %row, %end;

    mov.u32 %flag, 0;
    mov.s32 %r_value, 0;
    @!%p_inrange bra after_pred;

    // value = input[row]; flag = 1 iff the value matches the comparison predicate.
    mul.lo.u64 %input_addr, %row, 4;
    add.u64 %input_addr, %base, %input_addr;
    ld.global.s32 %r_value, [%input_addr];
    setp.lt.s32 %p_lt, %r_value, %needle;
    setp.le.s32 %p_lte, %r_value, %needle;
    setp.gt.s32 %p_gt, %r_value, %needle;
    setp.ge.s32 %p_gte, %r_value, %needle;
    setp.eq.s32 %p_eq, %r_value, %needle;
    setp.ne.s32 %p_ne, %r_value, %needle;
    setp.eq.u32 %p_code_eq, %comparison, 0;
    setp.eq.u32 %p_code_lt, %comparison, 1;
    setp.eq.u32 %p_code_lte, %comparison, 2;
    setp.eq.u32 %p_code_gt, %comparison, 3;
    setp.eq.u32 %p_code_gte, %comparison, 4;
    setp.eq.u32 %p_code_ne, %comparison, 5;
    mov.pred %p_match, 0;
    and.pred %p_eq, %p_eq, %p_code_eq;
    or.pred %p_match, %p_match, %p_eq;
    and.pred %p_lt, %p_lt, %p_code_lt;
    or.pred %p_match, %p_match, %p_lt;
    and.pred %p_lte, %p_lte, %p_code_lte;
    or.pred %p_match, %p_match, %p_lte;
    and.pred %p_gt, %p_gt, %p_code_gt;
    or.pred %p_match, %p_match, %p_gt;
    and.pred %p_gte, %p_gte, %p_code_gte;
    or.pred %p_match, %p_match, %p_gte;
    and.pred %p_ne, %p_ne, %p_code_ne;
    or.pred %p_match, %p_match, %p_ne;
    selp.u32 %flag, 1, 0, %p_match;

after_pred:
    // ---- warp inclusive scan of %flag over 32 lanes (Hillis-Steele via shfl.sync.up.b32) ----
    // All 32 lanes participate (the in-range branch reconverged at after_pred); membermask = full.
    mov.u32 %incl, %flag;
    shfl.sync.up.b32 %recv|%p_recv, %incl, 1, 0, 0xffffffff;
    @%p_recv add.u32 %incl, %incl, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %incl, 2, 0, 0xffffffff;
    @%p_recv add.u32 %incl, %incl, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %incl, 4, 0, 0xffffffff;
    @%p_recv add.u32 %incl, %incl, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %incl, 8, 0, 0xffffffff;
    @%p_recv add.u32 %incl, %incl, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %incl, 16, 0, 0xffffffff;
    @%p_recv add.u32 %incl, %incl, %recv;
    // warp-local exclusive = inclusive - own flag
    sub.u32 %wexcl, %incl, %flag;

    // lane 31 writes the warp total (= inclusive at the top lane) to s_scan[warp].
    setp.eq.u32 %p_islast, %lane, 31;
    @!%p_islast bra skip_wtot_write;
    mul.wide.u32 %tmp64, %warp, 4;
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, %tmp64;
    st.shared.u32 [%sh_addr], %incl;
skip_wtot_write:
    bar.sync 0;

    // ---- warp 0 exclusive-scans the warp totals s_scan[0..nwarps) ----
    @!%p_warp0 bra skip_combine;
    // each lane of warp 0 loads s_scan[lane] if lane < nwarps else 0
    setp.lt.u32 %p_lane_in, %lane, %nwarps;
    mov.u32 %wtot, 0;
    @!%p_lane_in bra have_wtot;
    mul.wide.u32 %tmp64, %lane, 4;
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, %tmp64;
    ld.shared.u32 %wtot, [%sh_addr];
have_wtot:
    // inclusive scan of %wtot over the warp (nwarps <= 32, so one warp covers every warp slot)
    mov.u32 %prefix, %wtot;
    shfl.sync.up.b32 %recv|%p_recv, %prefix, 1, 0, 0xffffffff;
    @%p_recv add.u32 %prefix, %prefix, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %prefix, 2, 0, 0xffffffff;
    @%p_recv add.u32 %prefix, %prefix, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %prefix, 4, 0, 0xffffffff;
    @%p_recv add.u32 %prefix, %prefix, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %prefix, 8, 0, 0xffffffff;
    @%p_recv add.u32 %prefix, %prefix, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %prefix, 16, 0, 0xffffffff;
    @%p_recv add.u32 %prefix, %prefix, %recv;
    // exclusive per-warp prefix = inclusive - own total; write to s_scan[32 + lane] (byte 128 + 4*lane).
    sub.u32 %tmp32, %prefix, %wtot;
    @!%p_lane_in bra skip_excl_write;
    mul.wide.u32 %tmp64, %lane, 4;
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, %tmp64;
    add.u64 %sh_addr, %sh_addr, 128;
    st.shared.u32 [%sh_addr], %tmp32;
skip_excl_write:
    // lane 31 holds the inclusive scan of ALL warp totals (lanes >= nwarps loaded 0) = block total;
    // store it to s_scan[64] (byte 256).
    setp.eq.u32 %p_islast, %lane, 31;
    @!%p_islast bra skip_combine;
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, 256;
    st.shared.u32 [%sh_addr], %prefix;
skip_combine:
    bar.sync 0;

    // ---- scatter: out[slot + running + per-warp-prefix + warp-local-exclusive] = value ----
    mul.wide.u32 %tmp64, %warp, 4;
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, %tmp64;
    add.u64 %sh_addr, %sh_addr, 128;
    ld.shared.u32 %prefix, [%sh_addr];
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, 256;
    ld.shared.u32 %btot, [%sh_addr];

    @!%p_inrange bra after_scatter;
    setp.eq.u32 %p_match, %flag, 1;
    @!%p_match bra after_scatter;
    add.u32 %off32, %wexcl, %prefix;
    cvt.u64.u32 %tmp64, %off32;
    add.u64 %tmp64, %tmp64, %running;
    add.u64 %tmp64, %tmp64, %slot;
    mul.lo.u64 %output_addr, %tmp64, 4;
    add.u64 %output_addr, %out_values, %output_addr;
    // In INDEX mode (out_is_index != 0) store the surviving ROW INDEX (the row u64 truncated to u32 -
    // the 4-byte out slot holds an index; row < row_count which fits u32); in VALUE mode store the
    // matching i32 value (unchanged). The ascending order is identical for both - the scatter slot is
    // the same row's rank, only the payload written differs.
    cvt.u32.u64 %row_u32, %row;
    selp.b32 %store_val, %row_u32, %r_value, %p_is_index;
    st.global.b32 [%output_addr], %store_val;

after_scatter:
    // running += block total (uniform across the block); advance one stride window; fence the
    // shared scratch before the next iteration's lane-31 writes overwrite it.
    cvt.u64.u32 %tmp64, %btot;
    add.u64 %running, %running, %tmp64;
    add.u64 %iter_base, %iter_base, %stride;
    bar.sync 0;
    bra iter_loop;

iter_done:
    ret;
}

// ---- TWO-INPUT (col-vs-col) ordered compaction: compare lhs[i] <cmp> rhs[i], emit ASCENDING row
// indices, no host sort. COPIED VERBATIM from the single-input `..._compare_count_blocks` /
// `..._compare_scatter_blocks` above (the block partition, grid-stride, the entire two-level
// warp-shuffle prefix-sum scan, the ordered scatter, all scaffolding); the ONLY change is the
// match-test fold: replace the one `input[idx] <cmp> needle` load+setp with TWO loads
// (`%a = lhs_base[idx]`, `%b = rhs_base[idx]`) and `%a <cmp> %b` (operand order lhs,rhs - identical
// to the legacy atomic kernel `gpu_db_buffer_i32_compare_buffers_to_indices`). Codes 0..4 only
// (eq/lt/lte/gt/gte); ne (5) is rejected at the wrapper, so the ne term is omitted. The scatter
// ALWAYS stores the surviving ROW INDEX (col-vs-col is a predicate), so there is no `out_is_index`
// param - the store is hardcoded to the row index (`cvt.u32.u64` of `%row`), exactly the single-input
// index path. The match-test fold (the two loads + the setp/and/or composition) is byte-identical
// between the two kernels below, so Pass A's per-block count equals Pass B's per-block scatter count
// for every predicate (count <-> scatter symmetry).
.visible .entry gpu_db_buffers_i32_compare_count_blocks(
    .param .u64 lhs_base,
    .param .u64 rhs_base,
    .param .u64 row_count,
    .param .u64 chunk_rows,
    .param .u32 comparison,
    .param .u64 out_block_counts_ptr
)
{
    .reg .pred %p_done;
    .reg .pred %p_lt;
    .reg .pred %p_lte;
    .reg .pred %p_gt;
    .reg .pred %p_gte;
    .reg .pred %p_eq;
    .reg .pred %p_code_lt;
    .reg .pred %p_code_lte;
    .reg .pred %p_code_gt;
    .reg .pred %p_code_gte;
    .reg .pred %p_code_eq;
    .reg .pred %p_match;
    .reg .u32 %lane;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %comparison;
    .reg .u64 %lhs;
    .reg .u64 %rhs;
    .reg .u64 %rows;
    .reg .u64 %chunk;
    .reg .u64 %block_counts;
    .reg .u64 %start;
    .reg .u64 %end;
    .reg .u64 %idx;
    .reg .u64 %tmp64;
    .reg .u64 %lhs_addr;
    .reg .u64 %rhs_addr;
    .reg .u64 %off4;
    .reg .u64 %matches;
    .reg .u64 %count_addr;
    .reg .s32 %a;
    .reg .s32 %b;

    ld.param.u64 %lhs, [lhs_base];
    ld.param.u64 %rhs, [rhs_base];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %chunk, [chunk_rows];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %block_counts, [out_block_counts_ptr];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %lane, %tid.x;
    mov.u32 %bdim, %ntid.x;

    // start = bid * chunk; end = min(start + chunk, rows)
    cvt.u64.u32 %tmp64, %bid;
    mul.lo.u64 %start, %tmp64, %chunk;
    add.u64 %end, %start, %chunk;
    setp.gt.u64 %p_done, %end, %rows;
    @%p_done mov.u64 %end, %rows;

    // idx = start + lane; stride = blockDim (grid-stride WITHIN this block's range)
    cvt.u64.u32 %tmp64, %lane;
    add.u64 %idx, %start, %tmp64;
    cvt.u64.u32 %tmp64, %bdim;

    mov.u64 %matches, 0;

loop_buf:
    setp.ge.u64 %p_done, %idx, %end;
    @%p_done bra done_buf;
    // value a = lhs[idx]; value b = rhs[idx]; flag iff a <cmp> b (operand order lhs,rhs).
    mul.lo.u64 %off4, %idx, 4;
    add.u64 %lhs_addr, %lhs, %off4;
    ld.global.s32 %a, [%lhs_addr];
    add.u64 %rhs_addr, %rhs, %off4;
    ld.global.s32 %b, [%rhs_addr];
    setp.lt.s32 %p_lt, %a, %b;
    setp.le.s32 %p_lte, %a, %b;
    setp.gt.s32 %p_gt, %a, %b;
    setp.ge.s32 %p_gte, %a, %b;
    setp.eq.s32 %p_eq, %a, %b;
    setp.eq.u32 %p_code_eq, %comparison, 0;
    setp.eq.u32 %p_code_lt, %comparison, 1;
    setp.eq.u32 %p_code_lte, %comparison, 2;
    setp.eq.u32 %p_code_gt, %comparison, 3;
    setp.eq.u32 %p_code_gte, %comparison, 4;
    mov.pred %p_match, 0;
    and.pred %p_eq, %p_eq, %p_code_eq;
    or.pred %p_match, %p_match, %p_eq;
    and.pred %p_lt, %p_lt, %p_code_lt;
    or.pred %p_match, %p_match, %p_lt;
    and.pred %p_lte, %p_lte, %p_code_lte;
    or.pred %p_match, %p_match, %p_lte;
    and.pred %p_gt, %p_gt, %p_code_gt;
    or.pred %p_match, %p_match, %p_gt;
    and.pred %p_gte, %p_gte, %p_code_gte;
    or.pred %p_match, %p_match, %p_gte;
    @!%p_match bra next_buf;
    add.u64 %matches, %matches, 1;

next_buf:
    add.u64 %idx, %idx, %tmp64;
    bra loop_buf;

done_buf:
    // Accumulate this thread's local matches into the block's slot (commutative add; the ordering is
    // established by the contiguous partition + pass B, never by this reduction).
    cvt.u64.u32 %tmp64, %bid;
    mul.lo.u64 %count_addr, %tmp64, 8;
    add.u64 %count_addr, %block_counts, %count_addr;
    red.global.add.u64 [%count_addr], %matches;
    ret;
}

// Pass B for the two-input (col-vs-col) ordered compaction. Identical structure to the single-input
// `gpu_db_resident_i32_compare_scatter_blocks` above (contiguous block partition + two-level
// warp-shuffle ordered intra-block prefix sum -> ascending-by-construction scatter); the ONLY changes
// vs that kernel are: (1) TWO input bases (lhs/rhs) read at `base + idx*4` and the `%a <cmp> %b`
// match-test fold (codes 0..4, no ne) - byte-identical to the count kernel above; (2) the store is
// hardcoded to the surviving ROW INDEX (`cvt.u32.u64` of %row) with no `out_is_index` param, since
// col-vs-col is a predicate. Pass A and Pass B test the SAME predicate over the SAME range, so the
// per-block scatter count equals Pass A's per-block count.
.visible .entry gpu_db_buffers_i32_compare_scatter_blocks(
    .param .u64 lhs_base,
    .param .u64 rhs_base,
    .param .u64 row_count,
    .param .u64 chunk_rows,
    .param .u32 comparison,
    .param .u64 block_base_ptr,
    .param .u64 out_indices_ptr
)
{
    // shared scratch: [0..32) warp totals, [32..64) per-warp exclusive prefixes, [64] block total.
    .shared .align 4 .b32 s_scan[65];

    .reg .pred %p_done;
    .reg .pred %p_lt;
    .reg .pred %p_lte;
    .reg .pred %p_gt;
    .reg .pred %p_gte;
    .reg .pred %p_eq;
    .reg .pred %p_code_lt;
    .reg .pred %p_code_lte;
    .reg .pred %p_code_gt;
    .reg .pred %p_code_gte;
    .reg .pred %p_code_eq;
    .reg .pred %p_match;
    .reg .pred %p_recv;
    .reg .pred %p_inrange;
    .reg .pred %p_islast;
    .reg .pred %p_warp0;
    .reg .pred %p_lane_in;
    .reg .u32 %thr;
    .reg .u32 %bdim;
    .reg .u32 %bid;
    .reg .u32 %lane;
    .reg .u32 %warp;
    .reg .u32 %nwarps;
    .reg .u32 %comparison;
    .reg .u32 %flag;
    .reg .u32 %incl;
    .reg .u32 %recv;
    .reg .u32 %wexcl;
    .reg .u32 %wtot;
    .reg .u32 %prefix;
    .reg .u32 %btot;
    .reg .u32 %off32;
    .reg .u32 %tmp32;
    .reg .u32 %row_u32;
    .reg .u64 %lhs;
    .reg .u64 %rhs;
    .reg .u64 %rows;
    .reg .u64 %chunk;
    .reg .u64 %block_base;
    .reg .u64 %out_indices;
    .reg .u64 %start;
    .reg .u64 %end;
    .reg .u64 %row;
    .reg .u64 %iter_base;
    .reg .u64 %stride;
    .reg .u64 %tmp64;
    .reg .u64 %lhs_addr;
    .reg .u64 %rhs_addr;
    .reg .u64 %off4;
    .reg .u64 %slot;
    .reg .u64 %output_addr;
    .reg .u64 %base_addr;
    .reg .u64 %running;
    .reg .u64 %sh_addr;
    .reg .s32 %a;
    .reg .s32 %b;

    ld.param.u64 %lhs, [lhs_base];
    ld.param.u64 %rhs, [rhs_base];
    ld.param.u64 %rows, [row_count];
    ld.param.u64 %chunk, [chunk_rows];
    ld.param.u32 %comparison, [comparison];
    ld.param.u64 %block_base, [block_base_ptr];
    ld.param.u64 %out_indices, [out_indices_ptr];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %thr, %tid.x;
    mov.u32 %bdim, %ntid.x;

    // lane = thr & 31; warp = thr >> 5; nwarps = (bdim + 31) >> 5
    and.b32 %lane, %thr, 31;
    shr.u32 %warp, %thr, 5;
    add.u32 %tmp32, %bdim, 31;
    shr.u32 %nwarps, %tmp32, 5;

    // start = bid * chunk; end = min(start + chunk, rows)
    cvt.u64.u32 %tmp64, %bid;
    mul.lo.u64 %start, %tmp64, %chunk;
    add.u64 %end, %start, %chunk;
    setp.gt.u64 %p_done, %end, %rows;
    @%p_done mov.u64 %end, %rows;

    // slot = block_base[bid]  (this block's exclusive-prefix base output index)
    mul.lo.u64 %base_addr, %tmp64, 8;
    add.u64 %base_addr, %block_base, %base_addr;
    ld.global.u64 %slot, [%base_addr];

    cvt.u64.u32 %stride, %bdim;
    // iter_base walks start, start+bdim, start+2*bdim, ...; row = iter_base + thr.
    mov.u64 %iter_base, %start;
    mov.u64 %running, 0;
    setp.eq.u32 %p_warp0, %warp, 0;

iter_loop_buf:
    // Continue while the block still has rows to cover: iter_base < end (block-uniform trip count).
    setp.ge.u64 %p_done, %iter_base, %end;
    @%p_done bra iter_done_buf;

    // row = iter_base + thr ; in-range = row < end
    cvt.u64.u32 %tmp64, %thr;
    add.u64 %row, %iter_base, %tmp64;
    setp.lt.u64 %p_inrange, %row, %end;

    mov.u32 %flag, 0;
    @!%p_inrange bra after_pred_buf;

    // a = lhs[row]; b = rhs[row]; flag = 1 iff a <cmp> b (operand order lhs,rhs).
    mul.lo.u64 %off4, %row, 4;
    add.u64 %lhs_addr, %lhs, %off4;
    ld.global.s32 %a, [%lhs_addr];
    add.u64 %rhs_addr, %rhs, %off4;
    ld.global.s32 %b, [%rhs_addr];
    setp.lt.s32 %p_lt, %a, %b;
    setp.le.s32 %p_lte, %a, %b;
    setp.gt.s32 %p_gt, %a, %b;
    setp.ge.s32 %p_gte, %a, %b;
    setp.eq.s32 %p_eq, %a, %b;
    setp.eq.u32 %p_code_eq, %comparison, 0;
    setp.eq.u32 %p_code_lt, %comparison, 1;
    setp.eq.u32 %p_code_lte, %comparison, 2;
    setp.eq.u32 %p_code_gt, %comparison, 3;
    setp.eq.u32 %p_code_gte, %comparison, 4;
    mov.pred %p_match, 0;
    and.pred %p_eq, %p_eq, %p_code_eq;
    or.pred %p_match, %p_match, %p_eq;
    and.pred %p_lt, %p_lt, %p_code_lt;
    or.pred %p_match, %p_match, %p_lt;
    and.pred %p_lte, %p_lte, %p_code_lte;
    or.pred %p_match, %p_match, %p_lte;
    and.pred %p_gt, %p_gt, %p_code_gt;
    or.pred %p_match, %p_match, %p_gt;
    and.pred %p_gte, %p_gte, %p_code_gte;
    or.pred %p_match, %p_match, %p_gte;
    selp.u32 %flag, 1, 0, %p_match;

after_pred_buf:
    // ---- warp inclusive scan of %flag over 32 lanes (Hillis-Steele via shfl.sync.up.b32) ----
    mov.u32 %incl, %flag;
    shfl.sync.up.b32 %recv|%p_recv, %incl, 1, 0, 0xffffffff;
    @%p_recv add.u32 %incl, %incl, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %incl, 2, 0, 0xffffffff;
    @%p_recv add.u32 %incl, %incl, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %incl, 4, 0, 0xffffffff;
    @%p_recv add.u32 %incl, %incl, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %incl, 8, 0, 0xffffffff;
    @%p_recv add.u32 %incl, %incl, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %incl, 16, 0, 0xffffffff;
    @%p_recv add.u32 %incl, %incl, %recv;
    // warp-local exclusive = inclusive - own flag
    sub.u32 %wexcl, %incl, %flag;

    // lane 31 writes the warp total (= inclusive at the top lane) to s_scan[warp].
    setp.eq.u32 %p_islast, %lane, 31;
    @!%p_islast bra skip_wtot_write_buf;
    mul.wide.u32 %tmp64, %warp, 4;
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, %tmp64;
    st.shared.u32 [%sh_addr], %incl;
skip_wtot_write_buf:
    bar.sync 0;

    // ---- warp 0 exclusive-scans the warp totals s_scan[0..nwarps) ----
    @!%p_warp0 bra skip_combine_buf;
    setp.lt.u32 %p_lane_in, %lane, %nwarps;
    mov.u32 %wtot, 0;
    @!%p_lane_in bra have_wtot_buf;
    mul.wide.u32 %tmp64, %lane, 4;
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, %tmp64;
    ld.shared.u32 %wtot, [%sh_addr];
have_wtot_buf:
    mov.u32 %prefix, %wtot;
    shfl.sync.up.b32 %recv|%p_recv, %prefix, 1, 0, 0xffffffff;
    @%p_recv add.u32 %prefix, %prefix, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %prefix, 2, 0, 0xffffffff;
    @%p_recv add.u32 %prefix, %prefix, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %prefix, 4, 0, 0xffffffff;
    @%p_recv add.u32 %prefix, %prefix, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %prefix, 8, 0, 0xffffffff;
    @%p_recv add.u32 %prefix, %prefix, %recv;
    shfl.sync.up.b32 %recv|%p_recv, %prefix, 16, 0, 0xffffffff;
    @%p_recv add.u32 %prefix, %prefix, %recv;
    // exclusive per-warp prefix = inclusive - own total; write to s_scan[32 + lane] (byte 128 + 4*lane).
    sub.u32 %tmp32, %prefix, %wtot;
    @!%p_lane_in bra skip_excl_write_buf;
    mul.wide.u32 %tmp64, %lane, 4;
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, %tmp64;
    add.u64 %sh_addr, %sh_addr, 128;
    st.shared.u32 [%sh_addr], %tmp32;
skip_excl_write_buf:
    // lane 31 holds the inclusive scan of ALL warp totals (lanes >= nwarps loaded 0) = block total;
    // store it to s_scan[64] (byte 256).
    setp.eq.u32 %p_islast, %lane, 31;
    @!%p_islast bra skip_combine_buf;
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, 256;
    st.shared.u32 [%sh_addr], %prefix;
skip_combine_buf:
    bar.sync 0;

    // ---- scatter: out[slot + running + per-warp-prefix + warp-local-exclusive] = row index ----
    mul.wide.u32 %tmp64, %warp, 4;
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, %tmp64;
    add.u64 %sh_addr, %sh_addr, 128;
    ld.shared.u32 %prefix, [%sh_addr];
    mov.u64 %sh_addr, s_scan;
    add.u64 %sh_addr, %sh_addr, 256;
    ld.shared.u32 %btot, [%sh_addr];

    @!%p_inrange bra after_scatter_buf;
    setp.eq.u32 %p_match, %flag, 1;
    @!%p_match bra after_scatter_buf;
    add.u32 %off32, %wexcl, %prefix;
    cvt.u64.u32 %tmp64, %off32;
    add.u64 %tmp64, %tmp64, %running;
    add.u64 %tmp64, %tmp64, %slot;
    mul.lo.u64 %output_addr, %tmp64, 4;
    add.u64 %output_addr, %out_indices, %output_addr;
    // Col-vs-col is a predicate -> store the surviving ROW INDEX (the row u64 truncated to u32; row <
    // row_count which fits u32 - the wrapper asserts row_count <= u32::MAX). Hardcoded index store (no
    // value mode), exactly the single-input index path.
    cvt.u32.u64 %row_u32, %row;
    st.global.b32 [%output_addr], %row_u32;

after_scatter_buf:
    // running += block total (uniform across the block); advance one stride window; fence the shared
    // scratch before the next iteration's lane-31 writes overwrite it.
    cvt.u64.u32 %tmp64, %btot;
    add.u64 %running, %running, %tmp64;
    add.u64 %iter_base, %iter_base, %stride;
    bar.sync 0;
    bra iter_loop_buf;

iter_done_buf:
    ret;
}
"#;

/// VALUE-emit launch: returns the matching i32 VALUES in ASCENDING ROW ORDER (the ordered
/// parallel-compaction backbone, `COMPARE_ORDERED_PTX`). A thin wrapper over the shared core with
/// `out_is_index = 0`; behavior is unchanged from before the index-emit mode was added.
fn launch_cuda_resident_i32_compare_project<R: CudaResidentReadSource>(
    resident: &R,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    comparison: CudaI32Comparison,
) -> Result<Vec<i32>, CudaRuntimeProbeError> {
    launch_cuda_resident_i32_compare_ordered_core(
        resident,
        resident.device_ptr(),
        byte_offset,
        row_count,
        needle,
        comparison.code(),
        0,
    )
}

fn i32_bits_into_u32(values: Vec<i32>) -> Vec<u32> {
    // SAFETY: i32 and u32 have identical size/alignment and every 32-bit pattern is valid for both.
    // ManuallyDrop transfers the allocation exactly once; the reconstructed Vec retains the same
    // pointer, length, and capacity, so the index path does not allocate/copy millions of result slots.
    let mut values = std::mem::ManuallyDrop::new(values);
    let ptr = values.as_mut_ptr().cast::<u32>();
    let len = values.len();
    let capacity = values.capacity();
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

/// INDEX-emit launch: returns the surviving ROW INDICES (`Vec<u32>`) in ASCENDING ORDER via the SAME
/// ordered parallel compaction, with the scatter kernel storing each match's row index (u32) instead
/// of its value (`out_is_index = 1`). The ascending-by-construction guarantee is identical to the
/// value path — the scatter slot is the same row's rank either way, only the stored payload differs —
/// so this replaces the atomic-append + host-`sort_unstable` index path with no host sort. Takes the
/// raw comparison code (0=eq, 1=lt, 2=lte, 3=gt, 4=gte, 5=ne).
fn launch_cuda_resident_i32_compare_indices_ordered<R: CudaResidentReadSource>(
    resident: &R,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    // Load-bearing precondition (audit P3): the scatter kernel stores each surviving ROW INDEX as a u32
    // (`cvt.u32.u64` then `st.global.b32`), so it would silently truncate past u32::MAX. The grid sizing
    // grows `chunk` for arbitrarily large row_count and does NOT cap it, so assert here (~17 GB for one
    // i32 column = unreachable on a single GPU today, but make the invariant explicit, not a comment).
    debug_assert!(
        row_count <= u64::from(u32::MAX),
        "ordered-index compaction stores row indices as u32; row_count {row_count} exceeds u32::MAX"
    );
    let slots = launch_cuda_resident_i32_compare_ordered_core(
        resident,
        resident.device_ptr(),
        byte_offset,
        row_count,
        needle,
        comparison,
        1,
    )?;
    // The scatter kernel wrote each surviving row index as a u32 via `st.global.b32`; the host buffer
    // is `Vec<i32>` 4-byte slots, so reinterpret each slot's bits back to u32 (bit-exact — a row index
    // is `< row_count`, always non-negative, and fits u32 since row_count <= u32::MAX in every sized
    // grid). The order is already ascending by construction (no host sort).
    Ok(i32_bits_into_u32(slots))
}

/// TWO-INPUT (col-vs-col / expr-vs-expr) ordered compare-compaction: compare `lhs[i] <cmp> rhs[i]`
/// elementwise and return the surviving ROW INDICES (`Vec<u32>`) in ASCENDING ORDER, with NO host
/// sort. Mirrors `launch_cuda_resident_i32_compare_indices_ordered` (and shares the orchestration of
/// `launch_cuda_resident_i32_compare_ordered_core`: parallel per-block count -> tiny host
/// exclusive-scan of the per-block counts -> parallel ordered-scatter, the same chunk/grid sizing,
/// lease lifetimes, async-on-pooled-stream lever + blocking fallback, and the `output_count >
/// row_count` guard), but drives the TWO-INPUT kernels `gpu_db_buffers_i32_compare_count_blocks` /
/// `..._scatter_blocks` with two absolute device bases and NO needle/byte_offset. The scatter always
/// stores the surviving row index (col-vs-col is a predicate). `comparison` is the raw code
/// (0=eq,1=lt,2=lte,3=gt,4=gte; ne is rejected at the wrapper).
///
/// SAFETY (load-bearing precondition, audit P3): the buffers at `lhs_base` and `rhs_base` MUST each
/// cover `[0, n*4)` - BOTH the count and the scatter kernel read every `idx in [0, n)` from BOTH
/// bases. A too-small buffer is an out-of-bounds DEVICE read (CUDA 700). The caller leases each value
/// buffer at exactly `n*4` and holds both leases across this whole call (borrowed for the duration),
/// so both inputs outlive both reads - documented exactly as the single-input core's `input_base`.
fn launch_cuda_resident_i32_compare_buffers_indices_ordered<R: CudaResidentReadSource>(
    resident: &R,
    lhs_base: u64,
    rhs_base: u64,
    n: u64,
    comparison: u32,
) -> Result<Vec<u32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
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

    const PTX: &[u8] = COMPARE_ORDERED_PTX;

    if n == 0 {
        return Ok(Vec::new());
    }
    // Load-bearing precondition (audit P3): the scatter kernel stores each surviving ROW INDEX as a u32
    // (`cvt.u32.u64` then `st.global.b32`), so it would silently truncate past u32::MAX. The grid sizing
    // grows `chunk` for arbitrarily large `n` and does NOT cap it, so assert here (~17 GB for one i32
    // input = unreachable on a single GPU today, but make the invariant explicit, not a comment).
    debug_assert!(
        n <= u64::from(u32::MAX),
        "ordered-index compaction stores row indices as u32; n {n} exceeds u32::MAX"
    );

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
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let values_bytes = usize::try_from(
        n.checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    // ---- ordered-compaction grid shape (identical to the single-input core) ----
    const BLOCK: u32 = 256;
    const CHUNK_ROWS: u64 = 256;
    const MAX_GRID: u64 = 65_535;
    let chunk = CHUNK_ROWS.max(n.div_ceil(MAX_GRID));
    let grid_u64 = n.div_ceil(chunk);
    debug_assert!((1..=MAX_GRID).contains(&grid_u64));
    let grid = u32::try_from(grid_u64)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let block_counts_len = grid as usize;
    let block_scratch_bytes = block_counts_len
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    let async_ops = match (
        resident.primary().cu_memcpy_dtoh_async,
        resident.primary().cu_memcpy_htod_async,
        resident.primary().cu_memset_d8_async,
    ) {
        (Some(dtoh), Some(htod), Some(memset)) => Some((dtoh, htod, memset)),
        _ => None,
    };

    let values_guard = resident
        .primary()
        .lease_device_buffer(values_bytes.max(1))?;
    let block_counts_guard = resident
        .primary()
        .lease_device_buffer(block_scratch_bytes)?;
    let block_base_guard = resident
        .primary()
        .lease_device_buffer(block_scratch_bytes)?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let count_fn = resident
        .primary()
        .cached_function(c"gpu_db_buffers_i32_compare_count_blocks", &ptx)?;
    let scatter_fn = resident
        .primary()
        .cached_function(c"gpu_db_buffers_i32_compare_scatter_blocks", &ptx)?;

    // Shared kernel scalar args. The two-input kernels read lhs[idx] / rhs[idx] at `base + idx*4` (no
    // needle / byte_offset). The scatter ALWAYS stores the row index (col-vs-col is a predicate).
    let mut lhs_arg = lhs_base;
    let mut rhs_arg = rhs_base;
    let mut rows_arg = n;
    let mut chunk_arg = chunk;
    let mut comparison_arg = comparison;
    let mut block_counts_arg = block_counts_guard.ptr;
    let mut block_base_arg = block_base_guard.ptr;
    let mut values_arg = values_guard.ptr;
    let mut count_args = [
        (&mut lhs_arg as *mut u64).cast::<c_void>(),
        (&mut rhs_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut chunk_arg as *mut u64).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut block_counts_arg as *mut u64).cast::<c_void>(),
    ];
    let mut scatter_args = [
        (&mut lhs_arg as *mut u64).cast::<c_void>(),
        (&mut rhs_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut chunk_arg as *mut u64).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut block_base_arg as *mut u64).cast::<c_void>(),
        (&mut values_arg as *mut u64).cast::<c_void>(),
    ];

    // Host exclusive scan of the per-block match counts into per-block base output slots; returns
    // (block_base, total_matches). Identical to the single-input core.
    fn exclusive_scan_blocks(counts: &[u64]) -> (Vec<u64>, u64) {
        let mut base = Vec::with_capacity(counts.len());
        let mut running = 0_u64;
        for &c in counts {
            base.push(running);
            running = running.saturating_add(c);
        }
        (base, running)
    }

    if let Some((dtoh_async, htod_async, memset_async)) = async_ops {
        // ---- async-on-pooled-stream path ----
        resident.primary().set_current()?;
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
            primary: resident.primary(),
            pooled: Some(resident.primary().acquire_pooled_stream()?),
        };
        let pooled = lease.pooled.as_ref().expect("pooled stream just set");
        let stream = pooled.stream;
        let timed = !pooled.start_event.is_null() && !pooled.stop_event.is_null();

        // HARDENING (error-path stream drain): drain the private stream before yielding an error so
        // enqueued ops cannot use leases freed by unwinding. Identical to the single-input core.
        let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
            unsafe {
                let _ = (resident.primary().cu_stream_synchronize)(stream);
            }
            err
        };

        if timed {
            check_cuda(unsafe { (resident.primary().cu_event_record)(pooled.start_event, stream) })
                .map_err(drain_err)?;
        }
        check_cuda(unsafe { memset_async(block_counts_guard.ptr, 0, block_scratch_bytes, stream) })
            .map_err(drain_err)?;
        check_cuda(unsafe {
            cu_launch_kernel(
                count_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                count_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;

        let mut block_counts = vec![0_u64; block_counts_len];
        let counts_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            block_counts_guard.ptr,
            &mut block_counts,
        )
        .map_err(drain_err)?;
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        copy_pinned_into(&counts_pinned, &mut block_counts);
        drop(counts_pinned);

        let (block_base, output_count) = exclusive_scan_blocks(&block_counts);
        if output_count > n {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(output_count).unwrap_or(usize::MAX),
            ));
        }

        let base_pinned = resident
            .primary()
            .lease_pinned_host_buffer(block_scratch_bytes);
        if let Some(pinned) = &base_pinned {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    block_base.as_ptr(),
                    pinned.ptr.cast::<u64>(),
                    block_base.len(),
                );
            }
        }
        let base_src: *const c_void = base_pinned
            .as_ref()
            .map(|p| p.ptr.cast_const())
            .unwrap_or_else(|| block_base.as_ptr().cast::<c_void>());
        check_cuda(unsafe {
            htod_async(block_base_guard.ptr, base_src, block_scratch_bytes, stream)
        })
        .map_err(drain_err)?;
        check_cuda(unsafe {
            cu_launch_kernel(
                scatter_fn,
                grid,
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
        if timed {
            check_cuda(unsafe { (resident.primary().cu_event_record)(pooled.stop_event, stream) })
                .map_err(drain_err)?;
        }

        let mut output = vec![
            0_i32;
            usize::try_from(output_count).map_err(|_| {
                CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
            })?
        ];

        let values_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            values_guard.ptr,
            &mut output,
        )
        .map_err(drain_err)?;
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        drop(base_pinned);
        copy_pinned_into(&values_pinned, &mut output);

        if timed {
            let mut elapsed_ms = 0.0_f32;
            check_cuda(unsafe {
                (resident.primary().cu_event_elapsed_time)(
                    &mut elapsed_ms,
                    pooled.start_event,
                    pooled.stop_event,
                )
            })?;
            resident.record_kernel_event_elapsed_us(Some(
                (f64::from(elapsed_ms) * 1_000.0).ceil() as u64
            ));
        } else {
            resident.record_kernel_event_elapsed_us(None);
        }

        drop(lease);
        // The scatter wrote each surviving row index as a u32 via `st.global.b32`; the host buffer is
        // `Vec<i32>` 4-byte slots, so reinterpret each slot's bits back to u32 (bit-exact - a row index
        // is `< n`, non-negative, fits u32 since n <= u32::MAX). Already ascending by construction.
        Ok(output.into_iter().map(|slot| slot as u32).collect())
    } else {
        // ---- legacy blocking fallback (old driver: no async/pinned symbols) ----
        check_cuda(unsafe { cu_memset_d8(block_counts_guard.ptr, 0, block_scratch_bytes) })?;
        launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
            cu_launch_kernel(
                count_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                count_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;

        let mut block_counts = vec![0_u64; block_counts_len];
        if !block_counts.is_empty() {
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    block_counts.as_mut_ptr().cast::<c_void>(),
                    block_counts_guard.ptr,
                    block_scratch_bytes,
                )
            })?;
        }
        let (block_base, output_count) = exclusive_scan_blocks(&block_counts);
        if output_count > n {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(output_count).unwrap_or(usize::MAX),
            ));
        }

        check_cuda(unsafe {
            cu_memcpy_htod(
                block_base_guard.ptr,
                block_base.as_ptr().cast::<c_void>(),
                block_scratch_bytes,
            )
        })?;
        launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
            cu_launch_kernel(
                scatter_fn,
                grid,
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
        })?;

        let mut output = vec![
            0_i32;
            usize::try_from(output_count).map_err(|_| {
                CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
            })?
        ];
        if !output.is_empty() {
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    output.as_mut_ptr().cast::<c_void>(),
                    values_guard.ptr,
                    output.len() * std::mem::size_of::<i32>(),
                )
            })?;
        }
        Ok(output.into_iter().map(|slot| slot as u32).collect())
    }
}

/// Shared core for the ordered i32 compare-compaction (`COMPARE_ORDERED_PTX`). `out_is_index` selects
/// what each ascending output slot stores: the matching i32 VALUE (0) or the surviving ROW INDEX as a
/// u32 (1, returned as `i32` bits the caller reinterprets). `comparison` is the raw kernel code
/// (0=eq, 1=lt, 2=lte, 3=gt, 4=gte, 5=ne). All count/host-scan/scatter structure, chunk/grid sizing,
/// and lease lifetimes are shared, so the value and index paths are byte-identical except the payload.
///
/// `input_base` is the device base the kernels read `row_count` contiguous i32s from, at
/// `input_base + byte_offset + idx*4`. The column path passes `resident.device_ptr()` (input IS the
/// resident column, so the `byte_offset + row_count*4 <= allocated_bytes` bound is checked against the
/// column allocation). A generalized caller passes a leased device buffer's ptr instead (its OWN i32
/// input — e.g. a 0/1 mask or an arithmetic result); the column-allocation bound does NOT apply (the
/// buffer is caller-sized and its lease must outlive both kernel launches), so it is checked only on
/// the column path. `resident` still supplies the CUDA context (lib / primary / streams / leases)
/// regardless of where the input is read.
///
/// SAFETY (load-bearing precondition for a leased-buffer `input_base`, audit P3): the buffer at
/// `input_base` MUST cover `[byte_offset, byte_offset + row_count*4)` — BOTH the count and the scatter
/// kernel read every `idx in [0, row_count)`. A too-small buffer is an out-of-bounds DEVICE read (CUDA
/// 700). The current callers all satisfy this (the mask producers and the I32 arith filter lease exactly
/// `row_count*4`; the I64/I128 mask buffers are larger), verified by the audit. NOTE (follow-up): make
/// this runtime-checked by threading the input buffer's byte length down so the bound applies on the
/// buffer path too, instead of by convention.
fn launch_cuda_resident_i32_compare_ordered_core<R: CudaResidentReadSource>(
    resident: &R,
    input_base: u64,
    byte_offset: u64,
    row_count: u64,
    needle: i32,
    comparison: u32,
    out_is_index: u32,
) -> Result<Vec<i32>, CudaRuntimeProbeError> {
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemsetD8 = unsafe extern "C" fn(u64, u8, usize) -> i32;
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

    // P2-M2 (compare-route parallel-kernel lever): the legacy kernel was a single-thread
    // `<<<1,1,1>>>` ascending scan that ordered-appended every matching i32 VALUE, so its output
    // is the matches in ASCENDING ROW ORDER. That serialized the whole 50k-row scan into one thread
    // (~277 µs kernel; the route plateaued serial-kernel-bound). This route now runs an ORDERED
    // PARALLEL COMPACTION over a CONTIGUOUS block partition — byte-identical ascending output, no
    // atomic-append (which would yield non-deterministic atomic-SCHEDULE order, the hazard fixed in
    // `row_indices`). The partition is the ordering backbone: block `b` owns the contiguous row
    // range `[b*chunk, min(b*chunk+chunk, rows))`, so "block order" == "row order".
    //
    //   Pass A (`..._count_blocks`, parallel: G blocks x BLOCK threads): each block grid-strides
    //   its own range and `red.global.add`s its local match count into `block_counts[b]`. This is
    //   the parallel scan of all rows (analogous to `equal_count`'s parallel reduction).
    //
    //   Host (between passes): exclusive-scan `block_counts[0..G]` -> `block_base[b]` = number of
    //   matches in all blocks `< b` (the base output slot for block `b`); the total match count is
    //   `block_base[G-1] + block_counts[G-1]`. Tiny (G is the block count), so a host scan avoids a
    //   third device scan kernel + a hand-authored shared-memory prefix sum.
    //
    //   Pass B (`..._scatter_blocks`, G blocks x BLOCK threads): block `b` compacts ITS range in
    //   PARALLEL via an ORDERED intra-block prefix-sum — each thread tests its row(s) to a 0/1 flag,
    //   an intra-block EXCLUSIVE scan (warp `shfl` scan + a tiny shared cross-warp combine) gives
    //   each match its within-block rank, and the value is scattered at `block_base[b] + rank`
    //   (+ a per-iteration running base when `chunk > blockDim`). The scan is monotonic in row index,
    //   so the within-block order is ASCENDING BY CONSTRUCTION (no atomics in the ordering path), and
    //   disjoint `block_base` ranges keep blocks independent — so the global output is exactly the
    //   ascending per-row matches, byte-identical to the old serial kernel for ANY row_count and
    //   match pattern. The serial span drops from one `chunk` (the prior one-thread-per-block scan)
    //   to `ceil(chunk / blockDim)` ordered-scan steps.
    //
    // Both kernels loop over `[start, end)` (block grid-stride), so they are correct for any
    // `chunk`/grid; the host sizes `chunk` so `G = ceil(rows/chunk) <= 65535` for every row_count
    // (the CUDA grid-x max), rounding `chunk` up when rows would exceed `65535 * CHUNK_ROWS`.
    const PTX: &[u8] = COMPARE_ORDERED_PTX;

    let bytes = row_count
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    // The `[byte_offset, byte_offset + row_count*4)` read window must fit the INPUT allocation. When the
    // input IS the resident column (`input_base == resident.device_ptr()`) that allocation is the
    // column's `allocated_bytes`, so check it (byte-identical to the pre-generalization column path).
    // A generalized caller reads its OWN leased buffer (a different base), which is sized to exactly
    // `row_count*4` with `byte_offset == 0` and owned by the caller, so the column bound does not apply.
    if input_base == resident.device_ptr() && bytes > resident.metadata().allocated_bytes {
        return Err(CudaRuntimeProbeError::InvalidInputLength(bytes as usize));
    }
    if row_count == 0 {
        return Ok(Vec::new());
    }
    // Load-bearing precondition (audit P3) for INDEX-emit mode: the scatter kernel stores each surviving
    // ROW INDEX as a u32 (`cvt.u32.u64` then `st.global.b32`), so it would silently truncate past
    // u32::MAX. The grid sizing grows `chunk` for arbitrarily large `row_count` and does NOT cap it, so
    // assert here (~17 GB for one i32 input = unreachable on a single GPU today, but make the invariant
    // explicit). Lives in the core so EVERY index-emit caller is covered (the column wrapper and the
    // generalized mask / arithmetic-result compactions alike), not just the original column wrapper.
    debug_assert!(
        out_is_index == 0 || row_count <= u64::from(u32::MAX),
        "ordered-index compaction stores row indices as u32; row_count {row_count} exceeds u32::MAX"
    );

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
    // Blocking memset/HtoD for the fallback path (zero the block-count scratch; upload the
    // host-scanned per-block base offsets).
    let cu_memset_d8 = unsafe {
        resident
            .lib()
            .get::<CuMemsetD8>(b"cuMemsetD8_v2\0")
            .or_else(|_| resident.lib().get::<CuMemsetD8>(b"cuMemsetD8\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        resident
            .lib()
            .get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| resident.lib().get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let values_bytes = usize::try_from(
        row_count
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?,
    )
    .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    // ---- ordered-compaction grid shape ----
    // Contiguous partition: block `b` owns rows `[b*chunk, min(b*chunk+chunk, rows))`. `chunk` is
    // sized so the block count `G = ceil(rows/chunk)` stays within the CUDA grid-x max (65_535) for
    // ANY row_count: start at CHUNK_ROWS rows/block and round `chunk` up if rows would need more
    // than 65_535 blocks. Both kernels loop over their range, so any `chunk` is correct.
    const BLOCK: u32 = 256;
    const CHUNK_ROWS: u64 = 256;
    const MAX_GRID: u64 = 65_535;
    let chunk = CHUNK_ROWS.max(row_count.div_ceil(MAX_GRID));
    let grid_u64 = row_count.div_ceil(chunk);
    debug_assert!((1..=MAX_GRID).contains(&grid_u64));
    let grid = u32::try_from(grid_u64)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    let block_counts_len = grid as usize;
    let block_scratch_bytes = block_counts_len
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;

    // P2-M2 (compare-route parallel-kernel lever): this single-frame range/comparison projection
    // already ran on the pooled-async substrate (private stream, pinned D2H, two covering syncs, no
    // per-call alloc/JIT/whole-context sync), but its kernel was the legacy single-thread
    // `<<<1,1,1>>>` scan, so the route plateaued serial-kernel-bound (~277 µs for a 50k-row scan).
    // The kernel is now an ORDERED PARALLEL COMPACTION (two passes over a contiguous block
    // partition; see the PTX header): a parallel per-block match count, a tiny host exclusive scan
    // of those counts into per-block base offsets, then a parallel scatter that compacts each block's
    // matches ascending at its base via an ordered intra-block prefix-sum (warp `shfl` scan + a tiny
    // shared cross-warp combine). The output is byte-identical ascending-per-row values for any
    // row_count and match pattern (no atomic-append, so no atomic-schedule non-determinism). The
    // device scratch is the `block_counts` / `block_base` arrays (G u64 each), leased from the same
    // `OutputBufferPool` as the values buffer.
    //
    // The whole async path is gated on the optional async + pinned-host driver symbols; on an old
    // driver lacking them the route keeps a blocking path (still cached-module + pooled-buffer +
    // pooled-stream, just blocking memset/HtoD/D2H), so correctness is unconditional and only the
    // acceleration is best-effort.
    let async_ops = match (
        resident.primary().cu_memcpy_dtoh_async,
        resident.primary().cu_memcpy_htod_async,
        resident.primary().cu_memset_d8_async,
    ) {
        (Some(dtoh), Some(htod), Some(memset)) => Some((dtoh, htod, memset)),
        _ => None,
    };

    // Pooled device buffers (no per-call cuMemAlloc/cuMemFree): the values output plus the two
    // block-offset scratch arrays. `block_counts` is zeroed before pass A (the count kernel
    // red-adds into it); `block_base` is overwritten by the HtoD upload; `values` is read back only
    // over the [0, total) prefix, so pooled stale bytes are never observed.
    let values_guard = resident
        .primary()
        .lease_device_buffer(values_bytes.max(1))?;
    let block_counts_guard = resident
        .primary()
        .lease_device_buffer(block_scratch_bytes)?;
    let block_base_guard = resident
        .primary()
        .lease_device_buffer(block_scratch_bytes)?;

    // P2-M2: cached modules (no per-launch cuModuleLoadData) — both kernel entries live in one PTX
    // module; the cache is keyed per entry name and launched concurrently on distinct streams.
    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);
    let count_fn = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_compare_count_blocks", &ptx)?;
    let scatter_fn = resident
        .primary()
        .cached_function(c"gpu_db_resident_i32_compare_scatter_blocks", &ptx)?;

    // Shared kernel scalar args (pointers/needle/comparison/chunk are identical across both passes;
    // each pass binds its own output pointer). The kernels read the input at `input_base + byte_offset
    // + idx*4`; `input_base` is the resident column ptr for the column path and a leased buffer ptr for
    // a generalized caller (mask / arithmetic-result compaction).
    let mut resident_arg = input_base;
    let mut offset_arg = byte_offset;
    let mut rows_arg = row_count;
    let mut chunk_arg = chunk;
    let mut needle_arg = needle;
    let mut comparison_arg = comparison;
    let mut block_counts_arg = block_counts_guard.ptr;
    let mut block_base_arg = block_base_guard.ptr;
    let mut values_arg = values_guard.ptr;
    // out_is_index selects the scatter payload: 0 = matching i32 VALUE, 1 = surviving ROW INDEX (u32).
    // Both produce the SAME ascending output slots; only the stored bytes differ.
    let mut out_is_index_arg: u32 = out_is_index;
    let mut count_args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut chunk_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut block_counts_arg as *mut u64).cast::<c_void>(),
    ];
    let mut scatter_args = [
        (&mut resident_arg as *mut u64).cast::<c_void>(),
        (&mut offset_arg as *mut u64).cast::<c_void>(),
        (&mut rows_arg as *mut u64).cast::<c_void>(),
        (&mut chunk_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut i32).cast::<c_void>(),
        (&mut comparison_arg as *mut u32).cast::<c_void>(),
        (&mut block_base_arg as *mut u64).cast::<c_void>(),
        (&mut values_arg as *mut u64).cast::<c_void>(),
        (&mut out_is_index_arg as *mut u32).cast::<c_void>(),
    ];

    // Host exclusive scan of the per-block match counts into per-block base output slots; returns
    // (block_base, total_matches). The base of block 0 is 0, and total is the sum of all counts —
    // both kernels see the SAME contiguous partition, so prefix-by-block == prefix-by-row.
    fn exclusive_scan_blocks(counts: &[u64]) -> (Vec<u64>, u64) {
        let mut base = Vec::with_capacity(counts.len());
        let mut running = 0_u64;
        for &c in counts {
            base.push(running);
            running = running.saturating_add(c);
        }
        (base, running)
    }

    if let Some((dtoh_async, htod_async, memset_async)) = async_ops {
        // ---- async-on-pooled-stream path (the lever) ----
        // Bind the shared primary context (idempotent) and lease the pooled private stream.
        resident.primary().set_current()?;
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
            primary: resident.primary(),
            pooled: Some(resident.primary().acquire_pooled_stream()?),
        };
        let pooled = lease.pooled.as_ref().expect("pooled stream just set");
        let stream = pooled.stream;
        let timed = !pooled.start_event.is_null() && !pooled.stop_event.is_null();

        // HARDENING (error-path stream drain): once an async op is enqueued on this private
        // stream, an early `?` would unwind the locals — returning the device + pinned buffer
        // leases to their shared pools while enqueued ops may still be in flight, a
        // use-after-free window for whoever leases those buffers next. So every fallible op from
        // the first async enqueue through each covering `cuStreamSynchronize` propagates its error
        // through `drain_err`, a best-effort blocking sync (result ignored) that drains the stream
        // FIRST, *then* yields the original error. `map_err` runs the closure at the error site
        // BEFORE `?` returns and hence before ANY local Drop, so it precedes the release of every
        // lease regardless of scope; on `Ok` the closure is not invoked, so the success path adds
        // nothing beyond the two explicit syncs below.
        let drain_err = |err: CudaRuntimeProbeError| -> CudaRuntimeProbeError {
            // SAFETY: `stream` is the live pooled private stream; a blocking synchronize on it is
            // valid from this thread (the primary context is current). The result is intentionally
            // ignored — this is a best-effort drain on an already-failing path.
            unsafe {
                let _ = (resident.primary().cu_stream_synchronize)(stream);
            }
            err
        };

        // (1) Stream-ordered zero of the block-count scratch, then the parallel COUNT kernel
        // (pass A). Start the timer before pass A so the recorded span covers BOTH kernels — the
        // meaningful "new kernel time" vs the old serial kernel.
        if timed {
            check_cuda(unsafe { (resident.primary().cu_event_record)(pooled.start_event, stream) })
                .map_err(drain_err)?;
        }
        check_cuda(unsafe { memset_async(block_counts_guard.ptr, 0, block_scratch_bytes, stream) })
            .map_err(drain_err)?;
        check_cuda(unsafe {
            cu_launch_kernel(
                count_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                count_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
        .map_err(drain_err)?;

        // (2) Stream-ordered read of the per-block counts, then sync #1: the count kernel is now
        // complete so the host can exclusive-scan the counts into base offsets and size the values
        // read. The counts stage through a pooled pinned host buffer for a truly-async DMA.
        let mut block_counts = vec![0_u64; block_counts_len];
        let counts_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            block_counts_guard.ptr,
            &mut block_counts,
        )
        .map_err(drain_err)?;
        // Covering sync #1: drains on its own error too (the counts D2H is still enqueued).
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        copy_pinned_into(&counts_pinned, &mut block_counts);
        drop(counts_pinned);

        let (block_base, output_count) = exclusive_scan_blocks(&block_counts);
        if output_count > row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(output_count).unwrap_or(usize::MAX),
            ));
        }

        // (3) Upload the host-scanned base offsets (HtoD, staged through a pooled pinned buffer for
        // a truly-async DMA), then launch the parallel SCATTER kernel (pass B: BLOCK threads/block,
        // ordered intra-block compaction — each match scattered ascending at base+rank), then stop
        // the timer.
        let base_pinned = resident
            .primary()
            .lease_pinned_host_buffer(block_scratch_bytes);
        if let Some(pinned) = &base_pinned {
            // SAFETY: leased with capacity >= block_scratch_bytes; copy the base offsets into the
            // pinned region (page-aligned) before the async HtoD reads from it.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    block_base.as_ptr(),
                    pinned.ptr.cast::<u64>(),
                    block_base.len(),
                );
            }
        }
        let base_src: *const c_void = base_pinned
            .as_ref()
            .map(|p| p.ptr.cast_const())
            .unwrap_or_else(|| block_base.as_ptr().cast::<c_void>());
        check_cuda(unsafe {
            htod_async(block_base_guard.ptr, base_src, block_scratch_bytes, stream)
        })
        .map_err(drain_err)?;
        check_cuda(unsafe {
            cu_launch_kernel(
                scatter_fn,
                grid,
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
        if timed {
            check_cuda(unsafe { (resident.primary().cu_event_record)(pooled.stop_event, stream) })
                .map_err(drain_err)?;
        }

        let mut output = vec![
            0_i32;
            usize::try_from(output_count).map_err(|_| {
                CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
            })?
        ];

        // (4) Stream-ordered result D2H of the populated [0, total) values prefix into a pooled
        // pinned host buffer, then ONE sync #2; copy the pinned bytes into the owned Vec. This sync
        // also covers the still-enqueued base HtoD + scatter kernel (both ordered before it).
        let values_pinned = stage_result_dtoh_async(
            resident.primary(),
            dtoh_async,
            stream,
            values_guard.ptr,
            &mut output,
        )
        .map_err(drain_err)?;
        // Covering sync #2: drains on its own error too (the HtoD/scatter/values D2H are enqueued).
        check_cuda(unsafe { (resident.primary().cu_stream_synchronize)(stream) })
            .map_err(drain_err)?;
        // Keep the base staging buffer alive until after the sync that completes its HtoD.
        drop(base_pinned);
        copy_pinned_into(&values_pinned, &mut output);

        if timed {
            let mut elapsed_ms = 0.0_f32;
            check_cuda(unsafe {
                (resident.primary().cu_event_elapsed_time)(
                    &mut elapsed_ms,
                    pooled.start_event,
                    pooled.stop_event,
                )
            })?;
            resident.record_kernel_event_elapsed_us(Some(
                (f64::from(elapsed_ms) * 1_000.0).ceil() as u64
            ));
        } else {
            resident.record_kernel_event_elapsed_us(None);
        }

        drop(lease);
        Ok(output)
    } else {
        // ---- legacy blocking fallback (old driver: no async/pinned symbols) ----
        // Still cached-module + pooled-buffer + pooled-stream (per-stream sync, no whole-context
        // cuCtxSynchronize); just blocking memset/HtoD/D2H. Run the same two passes with a blocking
        // counts D2H + host scan + base HtoD between them.
        check_cuda(unsafe { cu_memset_d8(block_counts_guard.ptr, 0, block_scratch_bytes) })?;
        launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
            cu_launch_kernel(
                count_fn,
                grid,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                stream,
                count_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;

        let mut block_counts = vec![0_u64; block_counts_len];
        if !block_counts.is_empty() {
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    block_counts.as_mut_ptr().cast::<c_void>(),
                    block_counts_guard.ptr,
                    block_scratch_bytes,
                )
            })?;
        }
        let (block_base, output_count) = exclusive_scan_blocks(&block_counts);
        if output_count > row_count {
            return Err(CudaRuntimeProbeError::InvalidInputLength(
                usize::try_from(output_count).unwrap_or(usize::MAX),
            ));
        }

        check_cuda(unsafe {
            cu_memcpy_htod(
                block_base_guard.ptr,
                block_base.as_ptr().cast::<c_void>(),
                block_scratch_bytes,
            )
        })?;
        launch_on_pooled_stream(resident, None, |stream, _scratch| unsafe {
            cu_launch_kernel(
                scatter_fn,
                grid,
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
        })?;

        let mut output = vec![
            0_i32;
            usize::try_from(output_count).map_err(|_| {
                CudaRuntimeProbeError::InvalidInputLength(usize::MAX)
            })?
        ];
        if !output.is_empty() {
            check_cuda(unsafe {
                cu_memcpy_dtoh(
                    output.as_mut_ptr().cast::<c_void>(),
                    values_guard.ptr,
                    output.len() * std::mem::size_of::<i32>(),
                )
            })?;
        }
        Ok(output)
    }
}

fn launch_cuda_smoke_add_one(input: u32) -> Result<u32, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_smoke_add_one(
    .param .u64 out_ptr,
    .param .u32 input
)
{
    .reg .u64 %out;
    .reg .u32 %r_value;
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %r_value, [input];
    add.u32 %r_value, %r_value, 1;
    st.global.u32 [%out], %r_value;
    ret;
}
"#;

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, std::mem::size_of::<u32>()) })?;
    let allocation_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(&mut function, module, c"gpu_db_cuda_smoke_add_one".as_ptr())
    })?;

    let mut output_arg = allocation_guard.ptr;
    let mut input_arg = input;
    let mut args = [
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut input_arg as *mut u32).cast::<c_void>(),
    ];
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut output = 0_u32;
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            (&mut output as *mut u32).cast::<c_void>(),
            allocation_guard.ptr,
            std::mem::size_of::<u32>(),
        )
    })?;

    drop(module_guard);
    drop(allocation_guard);
    drop(context_guard);

    Ok(output)
}

fn launch_cuda_all_mask(row_count: usize) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_all_mask(
    .param .u64 mask_ptr,
    .param .u32 row_count
)
{
    .reg .pred %p_out;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_row_count, [row_count];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset, %r_idx, 4;
    add.u64 %rd_mask_addr, %rd_mask, %rd_offset;
    st.global.u32 [%rd_mask_addr], 1;

DONE:
    ret;
}
"#;

    let row_count = u32::try_from(row_count)
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(row_count))?;
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let mask_len = row_count as usize;
    let mask_bytes = mask_len * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(&mut function, module, c"gpu_db_cuda_all_mask".as_ptr())
    })?;

    let mut mask_arg = mask_guard.ptr;
    let mut row_count_arg = row_count;
    let mut args = [
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; mask_len];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

fn launch_cuda_u32_equal_mask(
    input: &[u32],
    needle: u32,
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_u32_equal_mask(
    .param .u64 input_ptr,
    .param .u64 mask_ptr,
    .param .u32 len,
    .param .u32 needle
)
{
    .reg .pred %p_out;
    .reg .pred %p_match;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_len;
    .reg .u32 %r_needle;
    .reg .u32 %r_value;
    .reg .u32 %r_mask;
    .reg .u64 %rd_input;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset;
    .reg .u64 %rd_input_addr;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_input, [input_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_len, [len];
    ld.param.u32 %r_needle, [needle];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_len;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset, %r_idx, 4;
    add.u64 %rd_input_addr, %rd_input, %rd_offset;
    ld.global.u32 %r_value, [%rd_input_addr];
    setp.eq.u32 %p_match, %r_value, %r_needle;
    selp.u32 %r_mask, 1, 0, %p_match;
    add.u64 %rd_mask_addr, %rd_mask, %rd_offset;
    st.global.u32 [%rd_mask_addr], %r_mask;

DONE:
    ret;
}
"#;

    let len = u32::try_from(input.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(input.len()))?;
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let input_bytes = std::mem::size_of_val(input);
    let mut device_input = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_input, input_bytes) })?;
    let input_guard = CudaDeviceAllocationGuard {
        ptr: device_input,
        free: *cu_mem_free,
    };

    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, input_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    check_cuda(unsafe {
        cu_memcpy_htod(
            input_guard.ptr,
            input.as_ptr().cast::<c_void>(),
            input_bytes,
        )
    })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_u32_equal_mask".as_ptr(),
        )
    })?;

    let mut input_arg = input_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut len_arg = len;
    let mut needle_arg = needle;
    let mut args = [
        (&mut input_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut len_arg as *mut u32).cast::<c_void>(),
        (&mut needle_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = len.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; input.len()];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            input_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(input_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

fn launch_cuda_bytes_equal_mask(
    input: &[&[u8]],
    needle: &[u8],
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_bytes_equal_mask(
    .param .u64 bytes_ptr,
    .param .u64 offsets_ptr,
    .param .u64 needle_ptr,
    .param .u64 mask_ptr,
    .param .u32 row_count,
    .param .u32 needle_len
)
{
    .reg .pred %p_out;
    .reg .pred %p_len_diff;
    .reg .pred %p_loop_done;
    .reg .pred %p_byte_diff;
    .reg .u16 %h_byte;
    .reg .u16 %n_byte;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_needle_len;
    .reg .u32 %r_start;
    .reg .u32 %r_end;
    .reg .u32 %r_len;
    .reg .u32 %r_i;
    .reg .u32 %r_mask;
    .reg .u64 %rd_bytes;
    .reg .u64 %rd_offsets;
    .reg .u64 %rd_needle;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset_addr;
    .reg .u64 %rd_next_offset_addr;
    .reg .u64 %rd_byte_offset;
    .reg .u64 %rd_hay_addr;
    .reg .u64 %rd_needle_addr;
    .reg .u64 %rd_mask_offset;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_bytes, [bytes_ptr];
    ld.param.u64 %rd_offsets, [offsets_ptr];
    ld.param.u64 %rd_needle, [needle_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_row_count, [row_count];
    ld.param.u32 %r_needle_len, [needle_len];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset_addr, %r_idx, 4;
    add.u64 %rd_offset_addr, %rd_offsets, %rd_offset_addr;
    add.u64 %rd_next_offset_addr, %rd_offset_addr, 4;
    ld.global.u32 %r_start, [%rd_offset_addr];
    ld.global.u32 %r_end, [%rd_next_offset_addr];
    sub.u32 %r_len, %r_end, %r_start;
    setp.ne.u32 %p_len_diff, %r_len, %r_needle_len;
    @%p_len_diff bra NO_MATCH;

    mov.u32 %r_i, 0;
LOOP:
    setp.ge.u32 %p_loop_done, %r_i, %r_needle_len;
    @%p_loop_done bra MATCH;
    add.u32 %r_len, %r_start, %r_i;
    cvt.u64.u32 %rd_byte_offset, %r_len;
    add.u64 %rd_hay_addr, %rd_bytes, %rd_byte_offset;
    cvt.u64.u32 %rd_byte_offset, %r_i;
    add.u64 %rd_needle_addr, %rd_needle, %rd_byte_offset;
    ld.global.u8 %h_byte, [%rd_hay_addr];
    ld.global.u8 %n_byte, [%rd_needle_addr];
    setp.ne.u16 %p_byte_diff, %h_byte, %n_byte;
    @%p_byte_diff bra NO_MATCH;
    add.u32 %r_i, %r_i, 1;
    bra LOOP;

MATCH:
    mov.u32 %r_mask, 1;
    bra STORE;

NO_MATCH:
    mov.u32 %r_mask, 0;

STORE:
    mul.wide.u32 %rd_mask_offset, %r_idx, 4;
    add.u64 %rd_mask_addr, %rd_mask, %rd_mask_offset;
    st.global.u32 [%rd_mask_addr], %r_mask;

DONE:
    ret;
}
"#;

    let row_count = u32::try_from(input.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(input.len()))?;
    let needle_len = u32::try_from(needle.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(needle.len()))?;
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut offsets = Vec::with_capacity(input.len() + 1);
    let mut flattened = Vec::new();
    offsets.push(0_u32);
    for value in input {
        flattened.extend_from_slice(value);
        offsets.push(
            u32::try_from(flattened.len())
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(flattened.len()))?,
        );
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let bytes_len = flattened.len().max(1);
    let mut device_bytes = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_bytes, bytes_len) })?;
    let bytes_guard = CudaDeviceAllocationGuard {
        ptr: device_bytes,
        free: *cu_mem_free,
    };

    let offsets_bytes = std::mem::size_of_val(offsets.as_slice());
    let mut device_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_offsets, offsets_bytes) })?;
    let offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_offsets,
        free: *cu_mem_free,
    };

    let needle_bytes = needle.len().max(1);
    let mut device_needle = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_needle, needle_bytes) })?;
    let needle_guard = CudaDeviceAllocationGuard {
        ptr: device_needle,
        free: *cu_mem_free,
    };

    let mask_bytes = input.len() * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    if !flattened.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                bytes_guard.ptr,
                flattened.as_ptr().cast::<c_void>(),
                flattened.len(),
            )
        })?;
    }
    check_cuda(unsafe {
        cu_memcpy_htod(
            offsets_guard.ptr,
            offsets.as_ptr().cast::<c_void>(),
            offsets_bytes,
        )
    })?;
    if !needle.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                needle_guard.ptr,
                needle.as_ptr().cast::<c_void>(),
                needle.len(),
            )
        })?;
    }

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_bytes_equal_mask".as_ptr(),
        )
    })?;

    let mut bytes_arg = bytes_guard.ptr;
    let mut offsets_arg = offsets_guard.ptr;
    let mut needle_arg = needle_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut row_count_arg = row_count;
    let mut needle_len_arg = needle_len;
    let mut args = [
        (&mut bytes_arg as *mut u64).cast::<c_void>(),
        (&mut offsets_arg as *mut u64).cast::<c_void>(),
        (&mut needle_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
        (&mut needle_len_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; input.len()];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(needle_guard);
    drop(offsets_guard);
    drop(bytes_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

fn launch_cuda_bytes_range_mask(
    input: &[&[u8]],
    start_inclusive: &[u8],
    end_exclusive: &[u8],
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_bytes_range_mask(
    .param .u64 bytes_ptr,
    .param .u64 offsets_ptr,
    .param .u64 start_ptr,
    .param .u64 end_ptr,
    .param .u64 mask_ptr,
    .param .u32 row_count,
    .param .u32 start_len,
    .param .u32 end_len
)
{
    .reg .pred %p_out;
    .reg .pred %p_loop_done;
    .reg .pred %p_row_done;
    .reg .pred %p_bound_done;
    .reg .pred %p_lt;
    .reg .pred %p_gt;
    .reg .pred %p_ge_start;
    .reg .pred %p_lt_end;
    .reg .u16 %h_byte;
    .reg .u16 %b_byte;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_start_len;
    .reg .u32 %r_end_len;
    .reg .u32 %r_row_start;
    .reg .u32 %r_row_end;
    .reg .u32 %r_row_len;
    .reg .u32 %r_i;
    .reg .u32 %r_pos;
    .reg .u32 %r_mask;
    .reg .u64 %rd_bytes;
    .reg .u64 %rd_offsets;
    .reg .u64 %rd_start;
    .reg .u64 %rd_end;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_offset_addr;
    .reg .u64 %rd_next_offset_addr;
    .reg .u64 %rd_byte_offset;
    .reg .u64 %rd_hay_addr;
    .reg .u64 %rd_bound_addr;
    .reg .u64 %rd_mask_offset;
    .reg .u64 %rd_mask_addr;

    ld.param.u64 %rd_bytes, [bytes_ptr];
    ld.param.u64 %rd_offsets, [offsets_ptr];
    ld.param.u64 %rd_start, [start_ptr];
    ld.param.u64 %rd_end, [end_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u32 %r_row_count, [row_count];
    ld.param.u32 %r_start_len, [start_len];
    ld.param.u32 %r_end_len, [end_len];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset_addr, %r_idx, 4;
    add.u64 %rd_offset_addr, %rd_offsets, %rd_offset_addr;
    add.u64 %rd_next_offset_addr, %rd_offset_addr, 4;
    ld.global.u32 %r_row_start, [%rd_offset_addr];
    ld.global.u32 %r_row_end, [%rd_next_offset_addr];
    sub.u32 %r_row_len, %r_row_end, %r_row_start;

    mov.u32 %r_i, 0;
START_LOOP:
    setp.ge.u32 %p_row_done, %r_i, %r_row_len;
    setp.ge.u32 %p_bound_done, %r_i, %r_start_len;
    or.pred %p_loop_done, %p_row_done, %p_bound_done;
    @%p_loop_done bra START_PREFIX_DONE;

    add.u32 %r_pos, %r_row_start, %r_i;
    cvt.u64.u32 %rd_byte_offset, %r_pos;
    add.u64 %rd_hay_addr, %rd_bytes, %rd_byte_offset;
    cvt.u64.u32 %rd_byte_offset, %r_i;
    add.u64 %rd_bound_addr, %rd_start, %rd_byte_offset;
    ld.global.u8 %h_byte, [%rd_hay_addr];
    ld.global.u8 %b_byte, [%rd_bound_addr];
    setp.lt.u16 %p_lt, %h_byte, %b_byte;
    @%p_lt bra NO_MATCH;
    setp.gt.u16 %p_gt, %h_byte, %b_byte;
    @%p_gt bra START_MATCH;
    add.u32 %r_i, %r_i, 1;
    bra START_LOOP;

START_PREFIX_DONE:
    setp.ge.u32 %p_ge_start, %r_row_len, %r_start_len;
    @%p_ge_start bra START_MATCH;
    bra NO_MATCH;

START_MATCH:
    mov.u32 %r_i, 0;
END_LOOP:
    setp.ge.u32 %p_row_done, %r_i, %r_row_len;
    setp.ge.u32 %p_bound_done, %r_i, %r_end_len;
    or.pred %p_loop_done, %p_row_done, %p_bound_done;
    @%p_loop_done bra END_PREFIX_DONE;

    add.u32 %r_pos, %r_row_start, %r_i;
    cvt.u64.u32 %rd_byte_offset, %r_pos;
    add.u64 %rd_hay_addr, %rd_bytes, %rd_byte_offset;
    cvt.u64.u32 %rd_byte_offset, %r_i;
    add.u64 %rd_bound_addr, %rd_end, %rd_byte_offset;
    ld.global.u8 %h_byte, [%rd_hay_addr];
    ld.global.u8 %b_byte, [%rd_bound_addr];
    setp.lt.u16 %p_lt, %h_byte, %b_byte;
    @%p_lt bra MATCH;
    setp.gt.u16 %p_gt, %h_byte, %b_byte;
    @%p_gt bra NO_MATCH;
    add.u32 %r_i, %r_i, 1;
    bra END_LOOP;

END_PREFIX_DONE:
    setp.lt.u32 %p_lt_end, %r_row_len, %r_end_len;
    @%p_lt_end bra MATCH;
    bra NO_MATCH;

MATCH:
    mov.u32 %r_mask, 1;
    bra STORE;

NO_MATCH:
    mov.u32 %r_mask, 0;

STORE:
    mul.wide.u32 %rd_mask_offset, %r_idx, 4;
    add.u64 %rd_mask_addr, %rd_mask, %rd_mask_offset;
    st.global.u32 [%rd_mask_addr], %r_mask;

DONE:
    ret;
}
"#;

    let row_count = u32::try_from(input.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(input.len()))?;
    let start_len = u32::try_from(start_inclusive.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(start_inclusive.len()))?;
    let end_len = u32::try_from(end_exclusive.len())
        .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(end_exclusive.len()))?;
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut offsets = Vec::with_capacity(input.len() + 1);
    let mut flattened = Vec::new();
    offsets.push(0_u32);
    for value in input {
        flattened.extend_from_slice(value);
        offsets.push(
            u32::try_from(flattened.len())
                .map_err(|_| CudaRuntimeProbeError::InvalidInputLength(flattened.len()))?,
        );
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let bytes_len = flattened.len().max(1);
    let mut device_bytes = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_bytes, bytes_len) })?;
    let bytes_guard = CudaDeviceAllocationGuard {
        ptr: device_bytes,
        free: *cu_mem_free,
    };

    let offsets_bytes = std::mem::size_of_val(offsets.as_slice());
    let mut device_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_offsets, offsets_bytes) })?;
    let offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_offsets,
        free: *cu_mem_free,
    };

    let start_bytes = start_inclusive.len().max(1);
    let mut device_start = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_start, start_bytes) })?;
    let start_guard = CudaDeviceAllocationGuard {
        ptr: device_start,
        free: *cu_mem_free,
    };

    let end_bytes = end_exclusive.len().max(1);
    let mut device_end = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_end, end_bytes) })?;
    let end_guard = CudaDeviceAllocationGuard {
        ptr: device_end,
        free: *cu_mem_free,
    };

    let mask_bytes = input.len() * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    if !flattened.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                bytes_guard.ptr,
                flattened.as_ptr().cast::<c_void>(),
                flattened.len(),
            )
        })?;
    }
    check_cuda(unsafe {
        cu_memcpy_htod(
            offsets_guard.ptr,
            offsets.as_ptr().cast::<c_void>(),
            offsets_bytes,
        )
    })?;
    if !start_inclusive.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                start_guard.ptr,
                start_inclusive.as_ptr().cast::<c_void>(),
                start_inclusive.len(),
            )
        })?;
    }
    if !end_exclusive.is_empty() {
        check_cuda(unsafe {
            cu_memcpy_htod(
                end_guard.ptr,
                end_exclusive.as_ptr().cast::<c_void>(),
                end_exclusive.len(),
            )
        })?;
    }

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_bytes_range_mask".as_ptr(),
        )
    })?;

    let mut bytes_arg = bytes_guard.ptr;
    let mut offsets_arg = offsets_guard.ptr;
    let mut start_arg = start_guard.ptr;
    let mut end_arg = end_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut row_count_arg = row_count;
    let mut start_len_arg = start_len;
    let mut end_len_arg = end_len;
    let mut args = [
        (&mut bytes_arg as *mut u64).cast::<c_void>(),
        (&mut offsets_arg as *mut u64).cast::<c_void>(),
        (&mut start_arg as *mut u64).cast::<c_void>(),
        (&mut end_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
        (&mut start_len_arg as *mut u32).cast::<c_void>(),
        (&mut end_len_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = row_count.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; input.len()];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(end_guard);
    drop(start_guard);
    drop(offsets_guard);
    drop(bytes_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
}

fn launch_cuda_mvcc_row_batch_lengths(
    batch: &CudaMvccRowBatch,
) -> Result<Vec<(u32, u32)>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_mvcc_row_batch_lengths(
    .param .u64 key_offsets_ptr,
    .param .u64 value_offsets_ptr,
    .param .u64 output_ptr,
    .param .u32 row_count
)
{
    .reg .pred %p_out;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_key_start;
    .reg .u32 %r_key_end;
    .reg .u32 %r_value_start;
    .reg .u32 %r_value_end;
    .reg .u32 %r_key_len;
    .reg .u32 %r_value_len;
    .reg .u64 %rd_key_offsets;
    .reg .u64 %rd_value_offsets;
    .reg .u64 %rd_output;
    .reg .u64 %rd_offset;
    .reg .u64 %rd_next_offset;
    .reg .u64 %rd_output_offset;
    .reg .u64 %addr;

    ld.param.u64 %rd_key_offsets, [key_offsets_ptr];
    ld.param.u64 %rd_value_offsets, [value_offsets_ptr];
    ld.param.u64 %rd_output, [output_ptr];
    ld.param.u32 %r_row_count, [row_count];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset, %r_idx, 4;
    add.u64 %addr, %rd_key_offsets, %rd_offset;
    add.u64 %rd_next_offset, %addr, 4;
    ld.global.u32 %r_key_start, [%addr];
    ld.global.u32 %r_key_end, [%rd_next_offset];
    sub.u32 %r_key_len, %r_key_end, %r_key_start;

    add.u64 %addr, %rd_value_offsets, %rd_offset;
    add.u64 %rd_next_offset, %addr, 4;
    ld.global.u32 %r_value_start, [%addr];
    ld.global.u32 %r_value_end, [%rd_next_offset];
    sub.u32 %r_value_len, %r_value_end, %r_value_start;

    mul.wide.u32 %rd_output_offset, %r_idx, 8;
    add.u64 %addr, %rd_output, %rd_output_offset;
    st.global.u32 [%addr], %r_key_len;
    add.u64 %addr, %addr, 4;
    st.global.u32 [%addr], %r_value_len;

DONE:
    ret;
}
"#;

    batch.validate()?;
    if batch.row_count == 0 {
        return Ok(Vec::new());
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let key_offsets_bytes = std::mem::size_of_val(batch.key_offsets.as_slice());
    let mut device_key_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_key_offsets, key_offsets_bytes) })?;
    let key_offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_key_offsets,
        free: *cu_mem_free,
    };

    let value_offsets_bytes = std::mem::size_of_val(batch.value_offsets.as_slice());
    let mut device_value_offsets = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_value_offsets, value_offsets_bytes) })?;
    let value_offsets_guard = CudaDeviceAllocationGuard {
        ptr: device_value_offsets,
        free: *cu_mem_free,
    };

    let output_words = batch.row_count as usize * 2;
    let output_bytes = output_words * std::mem::size_of::<u32>();
    let mut device_output = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_output, output_bytes) })?;
    let output_guard = CudaDeviceAllocationGuard {
        ptr: device_output,
        free: *cu_mem_free,
    };

    check_cuda(unsafe {
        cu_memcpy_htod(
            key_offsets_guard.ptr,
            batch.key_offsets.as_ptr().cast::<c_void>(),
            key_offsets_bytes,
        )
    })?;
    check_cuda(unsafe {
        cu_memcpy_htod(
            value_offsets_guard.ptr,
            batch.value_offsets.as_ptr().cast::<c_void>(),
            value_offsets_bytes,
        )
    })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_mvcc_row_batch_lengths".as_ptr(),
        )
    })?;

    let mut key_offsets_arg = key_offsets_guard.ptr;
    let mut value_offsets_arg = value_offsets_guard.ptr;
    let mut output_arg = output_guard.ptr;
    let mut row_count_arg = batch.row_count;
    let mut args = [
        (&mut key_offsets_arg as *mut u64).cast::<c_void>(),
        (&mut value_offsets_arg as *mut u64).cast::<c_void>(),
        (&mut output_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = batch.row_count.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut output = vec![0_u32; output_words];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            output.as_mut_ptr().cast::<c_void>(),
            output_guard.ptr,
            output_bytes,
        )
    })?;

    drop(module_guard);
    drop(output_guard);
    drop(value_offsets_guard);
    drop(key_offsets_guard);
    drop(context_guard);

    Ok(output
        .chunks_exact(2)
        .map(|lengths| (lengths[0], lengths[1]))
        .collect())
}

fn launch_cuda_mvcc_visibility_mask(
    batch: &CudaMvccRowBatch,
    read_txn_id: u64,
) -> Result<Vec<bool>, CudaRuntimeProbeError> {
    type CuInit = unsafe extern "C" fn(u32) -> i32;
    type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
    type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
    type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
    type CuMemFree = unsafe extern "C" fn(u64) -> i32;
    type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
    type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;
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
    type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

    const PTX: &[u8] = br#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry gpu_db_cuda_mvcc_visibility_mask(
    .param .u64 begin_txn_ids_ptr,
    .param .u64 end_txn_ids_ptr,
    .param .u64 mask_ptr,
    .param .u64 read_txn_id,
    .param .u32 row_count
)
{
    .reg .pred %p_out;
    .reg .pred %p_created;
    .reg .pred %p_not_deleted;
    .reg .pred %p_visible;
    .reg .u32 %r_tid;
    .reg .u32 %r_block;
    .reg .u32 %r_block_dim;
    .reg .u32 %r_idx;
    .reg .u32 %r_row_count;
    .reg .u32 %r_mask_value;
    .reg .u64 %rd_begin;
    .reg .u64 %rd_end;
    .reg .u64 %rd_mask;
    .reg .u64 %rd_read_txn_id;
    .reg .u64 %rd_offset8;
    .reg .u64 %rd_offset4;
    .reg .u64 %addr;
    .reg .u64 %rd_created_by;
    .reg .u64 %rd_deleted_by;

    ld.param.u64 %rd_begin, [begin_txn_ids_ptr];
    ld.param.u64 %rd_end, [end_txn_ids_ptr];
    ld.param.u64 %rd_mask, [mask_ptr];
    ld.param.u64 %rd_read_txn_id, [read_txn_id];
    ld.param.u32 %r_row_count, [row_count];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_block, %ctaid.x;
    mov.u32 %r_block_dim, %ntid.x;
    mad.lo.u32 %r_idx, %r_block, %r_block_dim, %r_tid;

    setp.ge.u32 %p_out, %r_idx, %r_row_count;
    @%p_out bra DONE;

    mul.wide.u32 %rd_offset8, %r_idx, 8;
    add.u64 %addr, %rd_begin, %rd_offset8;
    ld.global.u64 %rd_created_by, [%addr];
    add.u64 %addr, %rd_end, %rd_offset8;
    ld.global.u64 %rd_deleted_by, [%addr];

    setp.le.u64 %p_created, %rd_created_by, %rd_read_txn_id;
    setp.gt.u64 %p_not_deleted, %rd_deleted_by, %rd_read_txn_id;
    and.pred %p_visible, %p_created, %p_not_deleted;
    selp.u32 %r_mask_value, 1, 0, %p_visible;

    mul.wide.u32 %rd_offset4, %r_idx, 4;
    add.u64 %addr, %rd_mask, %rd_offset4;
    st.global.u32 [%addr], %r_mask_value;

DONE:
    ret;
}
"#;

    batch.validate()?;
    if batch.row_count == 0 {
        return Ok(Vec::new());
    }

    let lib = unsafe {
        Library::new("libcuda.so.1")
            .or_else(|_| Library::new("libcuda.so"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    let cu_init = unsafe {
        lib.get::<CuInit>(b"cuInit\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_device_get = unsafe {
        lib.get::<CuDeviceGet>(b"cuDeviceGet\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_create = unsafe {
        lib.get::<CuCtxCreate>(b"cuCtxCreate_v2\0")
            .or_else(|_| lib.get::<CuCtxCreate>(b"cuCtxCreate\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_destroy = unsafe {
        lib.get::<CuCtxDestroy>(b"cuCtxDestroy_v2\0")
            .or_else(|_| lib.get::<CuCtxDestroy>(b"cuCtxDestroy\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_alloc = unsafe {
        lib.get::<CuMemAlloc>(b"cuMemAlloc_v2\0")
            .or_else(|_| lib.get::<CuMemAlloc>(b"cuMemAlloc\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_mem_free = unsafe {
        lib.get::<CuMemFree>(b"cuMemFree_v2\0")
            .or_else(|_| lib.get::<CuMemFree>(b"cuMemFree\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_htod = unsafe {
        lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD_v2\0")
            .or_else(|_| lib.get::<CuMemcpyHtoD>(b"cuMemcpyHtoD\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_memcpy_dtoh = unsafe {
        lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH_v2\0")
            .or_else(|_| lib.get::<CuMemcpyDtoH>(b"cuMemcpyDtoH\0"))
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_load_data = unsafe {
        lib.get::<CuModuleLoadData>(b"cuModuleLoadData\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_unload = unsafe {
        lib.get::<CuModuleUnload>(b"cuModuleUnload\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_module_get_function = unsafe {
        lib.get::<CuModuleGetFunction>(b"cuModuleGetFunction\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_launch_kernel = unsafe {
        lib.get::<CuLaunchKernel>(b"cuLaunchKernel\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };
    let cu_ctx_synchronize = unsafe {
        lib.get::<CuCtxSynchronize>(b"cuCtxSynchronize\0")
            .map_err(|_| CudaRuntimeProbeError::DriverLibraryUnavailable)?
    };

    check_cuda(unsafe { cu_init(0) })?;

    let mut device = 0;
    check_cuda(unsafe { cu_device_get(&mut device, 0) })?;

    let mut context = std::ptr::null_mut();
    check_cuda(unsafe { cu_ctx_create(&mut context, 0, device) })?;
    let context_guard = CudaContextGuard {
        context,
        destroy: *cu_ctx_destroy,
    };

    let begin_bytes = std::mem::size_of_val(batch.begin_txn_ids.as_slice());
    let mut device_begin = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_begin, begin_bytes) })?;
    let begin_guard = CudaDeviceAllocationGuard {
        ptr: device_begin,
        free: *cu_mem_free,
    };

    let end_bytes = std::mem::size_of_val(batch.end_txn_ids.as_slice());
    let mut device_end = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_end, end_bytes) })?;
    let end_guard = CudaDeviceAllocationGuard {
        ptr: device_end,
        free: *cu_mem_free,
    };

    let mask_bytes = batch.row_count as usize * std::mem::size_of::<u32>();
    let mut device_mask = 0_u64;
    check_cuda(unsafe { cu_mem_alloc(&mut device_mask, mask_bytes) })?;
    let mask_guard = CudaDeviceAllocationGuard {
        ptr: device_mask,
        free: *cu_mem_free,
    };

    check_cuda(unsafe {
        cu_memcpy_htod(
            begin_guard.ptr,
            batch.begin_txn_ids.as_ptr().cast::<c_void>(),
            begin_bytes,
        )
    })?;
    check_cuda(unsafe {
        cu_memcpy_htod(
            end_guard.ptr,
            batch.end_txn_ids.as_ptr().cast::<c_void>(),
            end_bytes,
        )
    })?;

    let mut ptx = Vec::with_capacity(PTX.len() + 1);
    ptx.extend_from_slice(PTX);
    ptx.push(0);

    let mut module = std::ptr::null_mut();
    check_cuda(unsafe { cu_module_load_data(&mut module, ptx.as_ptr().cast::<c_void>()) })?;
    let module_guard = CudaModuleGuard {
        module,
        unload: *cu_module_unload,
    };

    let mut function = std::ptr::null_mut();
    check_cuda(unsafe {
        cu_module_get_function(
            &mut function,
            module,
            c"gpu_db_cuda_mvcc_visibility_mask".as_ptr(),
        )
    })?;

    let mut begin_arg = begin_guard.ptr;
    let mut end_arg = end_guard.ptr;
    let mut mask_arg = mask_guard.ptr;
    let mut read_txn_id_arg = read_txn_id;
    let mut row_count_arg = batch.row_count;
    let mut args = [
        (&mut begin_arg as *mut u64).cast::<c_void>(),
        (&mut end_arg as *mut u64).cast::<c_void>(),
        (&mut mask_arg as *mut u64).cast::<c_void>(),
        (&mut read_txn_id_arg as *mut u64).cast::<c_void>(),
        (&mut row_count_arg as *mut u32).cast::<c_void>(),
    ];
    let threads_per_block = 128;
    let blocks = batch.row_count.div_ceil(threads_per_block);
    check_cuda(unsafe {
        cu_launch_kernel(
            function,
            blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })?;
    check_cuda(unsafe { cu_ctx_synchronize() })?;

    let mut mask = vec![0_u32; batch.row_count as usize];
    check_cuda(unsafe {
        cu_memcpy_dtoh(
            mask.as_mut_ptr().cast::<c_void>(),
            mask_guard.ptr,
            mask_bytes,
        )
    })?;

    drop(module_guard);
    drop(mask_guard);
    drop(end_guard);
    drop(begin_guard);
    drop(context_guard);

    Ok(mask.into_iter().map(|value| value != 0).collect())
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
