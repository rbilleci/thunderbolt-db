//! Build-only, per-engine aggregate timing for bounded INSERT qualification.
//!
//! The counters deliberately do not define a statement wall-clock total: wave work is shared by
//! several statements and may overlap with waiter work. Each populated phase is therefore an
//! independently bounded seam; `unattributed_or_concurrent_overlap_nanos` is explicit rather
//! than pretending that the fields are additive.

use std::sync::atomic::{AtomicU64, Ordering};

use super::Engine;

/// Aggregate INSERT qualification counters for one [`Engine`] instance.
///
/// This is a monotonic snapshot. Consumers that need a session result take two snapshots and call
/// [`Self::delta_since`]. Durations are host-observed nanoseconds at named ownership seams; they
/// are not a GPU-kernel event timeline and are intentionally not summed into a total.  The one
/// explicit exception is `typed_append_generation_kernel_event_nanos`, a separately named CUDA
/// event span for the generation kernels only.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InsertProbeSnapshot {
    pub successful_insert_statements: u64,
    pub successful_insert_rows: u64,
    /// GPU CHECK mask launches from the typed pre-queue row-local proof.
    pub row_local_check_launches: u64,
    /// One-u32 device terminal verdicts read for those typed CHECK masks.
    pub row_local_check_verdicts: u64,
    /// Calls to the sole canonical request-identity digest derivation constructor. This is the
    /// engine's parser-delimited request domain, not raw pgwire Query-message attribution.
    pub raw_request_digest_derivations: u64,
    /// Exact bytes hashed by those canonical request-identity derivations. For simple-query
    /// INSERTs this excludes source terminators the parser has stripped before admission.
    pub raw_request_digest_derivation_bytes: u64,
    /// Exact raw simple-query SQL bytes observed at the server Query-message boundary after one
    /// successful INSERT. This includes `;` and trailing whitespace, but excludes pgwire framing.
    pub successful_insert_source_bytes: u64,
    /// Facade-observed elapsed time from receipt of one INSERT text request through successful
    /// outcome. This is intentionally non-additive with the phase counters below.
    pub successful_insert_end_to_end_service_nanos: u64,
    pub facade_parse_bind_nanos: u64,
    pub engine_authorization_catalog_admission_nanos: u64,
    pub offlock_coercion_default_constraint_prepare_nanos: u64,
    /// Full immutable publication capture used to create a transaction context.
    pub transaction_begin_snapshot_capture_nanos: u64,
    /// Full immutable publication capture used to establish a statement snapshot.
    pub transaction_statement_snapshot_capture_nanos: u64,
    /// Full immutable publication capture while the terminal owns the commit boundary.
    pub transaction_commit_snapshot_capture_nanos: u64,
    /// Commit boundaries that reused the statement's exact immutable publication without recapture.
    pub transaction_commit_snapshot_reuse_count: u64,
    /// Construction and upload of an immutable transaction-private typed INSERT shard.
    pub transaction_private_overlay_materialize_nanos: u64,
    /// One-statement autocommit typed staging, including canonical row/WAL-image formation.
    pub transaction_statement_stage_nanos: u64,
    /// Canonical typed-record sealing from the immutable typed semantic carrier. This is a
    /// contained subphase of statement staging (or off-lock plan preparation), not an additive
    /// service-time phase.
    pub codec5_record_seal_nanos: u64,
    /// Final typed-image sealing from the same immutable typed semantic carrier. This is a
    /// contained subphase of statement staging (or off-lock plan preparation), not an additive
    /// service-time phase.
    pub codec5_final_image_seal_nanos: u64,
    /// Move-only resident append-source materialization from catalog-order typed vectors. This
    /// is a contained statement-stage subphase and remains type-neutral.
    pub resident_append_source_materialize_nanos: u64,
    /// Constructor-only validation of the immutable transaction-private typed payload. This
    /// includes layout/statistics validation before the payload becomes staged state.
    pub typed_stage_constructor_payload_validate_nanos: u64,
    /// Constructor-only transient row decoding used solely to claim UNIQUE slots. It must be
    /// zero for a table with no UNIQUE/primary-key slot to claim.
    pub typed_stage_constructor_unique_slot_nanos: u64,
    /// Hashing the just-validated immutable private payload to bind its later GPU upload.
    pub typed_stage_constructor_payload_digest_nanos: u64,
    /// Exact revalidation of that immutable payload before it becomes the private GPU shard.
    /// This isolates work that should become a construction invariant rather than a second scan.
    pub typed_overlay_payload_revalidate_nanos: u64,
    /// Reservation, row-id serialization, and H2D allocation for the private staged shard.
    pub typed_overlay_payload_upload_nanos: u64,
    /// Admission-time codec-5 selection, including its temporary final-row materialization.
    pub codec5_admission_select_materialize_nanos: u64,
    /// Admission-time resource-geometry counting, including any row re-encoding performed only
    /// to compute bounds.
    pub codec5_admission_geometry_encode_nanos: u64,
    /// Commit-time codec-5 selection, including final-row materialization from staged payloads.
    pub codec5_terminal_select_materialize_nanos: u64,
    /// Strict final-image decode and append-source reconstruction before GPU generation. Recovery
    /// retains its independent decoder; this isolates the live owned-source duplication.
    pub codec5_terminal_final_image_source_nanos: u64,
    /// The complete claimed-autocommit terminal, from its commit boundary through publication.
    /// This contains the canonical-apply subphase below and is not additive with it.
    pub transaction_terminal_nanos: u64,
    /// Commit-boundary validation, canonical record construction, and physical-plan compilation
    /// before the first WAL proposal.
    pub transaction_terminal_pre_wal_nanos: u64,
    /// Build-only partition of the sole generic codec-5 operation before it reaches WAL. These
    /// ownership seams are diagnostic only; they do not alter the single typed lifecycle.
    pub codec5_operation_input_nanos: u64,
    pub codec5_operation_table_prepare_nanos: u64,
    pub codec5_operation_aggregate_nanos: u64,
    pub codec5_operation_plan_compile_nanos: u64,
    pub codec5_operation_finalize_nanos: u64,
    /// Build-only phases of the already-closed codec-5 S7/envelope encoder. These remain
    /// distinct from the retired `typed_append_*` writer counters, which must stay silent.
    pub codec5_live_s7_build_nanos: u64,
    /// Strict S2 model reconstruction inside the live S7 writer. This is contained in the S7
    /// build interval and identifies whether a provenance-sealed feature-free closure warrants
    /// a narrower no-model path; recovery always retains its full strict decoder.
    pub codec5_live_s2_decode_nanos: u64,
    pub codec5_live_aggregate_prepare_nanos: u64,
    pub codec5_live_aggregate_encode_nanos: u64,
    pub codec5_live_outer_reserve_nanos: u64,
    pub codec5_live_outer_encode_nanos: u64,
    pub codec5_live_outer_digest_nanos: u64,
    pub codec5_live_outer_packed_encode_nanos: u64,
    pub codec5_live_outer_checksum_nanos: u64,
    pub codec5_live_outer_serialized_copy_nanos: u64,
    pub codec5_live_outer_seal_nanos: u64,
    /// Host-observed duration of the one generic device-resident typed-append generation call.
    /// It includes descriptor serialization, H2D/D2H, launch, and synchronization; it is not a
    /// CUDA-event kernel duration.
    pub typed_append_generation_host_nanos: u64,
    /// Preparation of the already-sealed columnar source's read-only generation view. This is
    /// deliberately separate from descriptor encoding so a repeated source ownership scan is
    /// never hidden inside transport attribution.
    pub typed_append_generation_source_view_nanos: u64,
    /// Exact pooled host/device/stream reservation for one generic generation. It contains no
    /// row-value materialization or CUDA submission.
    pub typed_append_generation_prepare_reserve_nanos: u64,
    /// Descriptor fill plus the sole H2D/kernel/D2H submission, fence, and opaque proof decode.
    /// Together with the two preceding fields this partitions the host generation call.
    pub typed_append_generation_submit_complete_nanos: u64,
    /// CUDA-event elapsed time for the five generic typed-append generation kernels.  It
    /// excludes descriptor serialization, H2D/D2H, host proof materialization, and the covering
    /// stream fence.  A zero value means CUDA event timing was unavailable for that sample.
    pub typed_append_generation_kernel_event_nanos: u64,
    /// Ordered CUDA-event spans for the generic generation's validation, cell commitment, row
    /// commitment, tree reduction, and finalization kernels.  They partition the total kernel
    /// event only when all five boundaries were successfully recorded.
    pub typed_append_generation_validate_kernel_event_nanos: u64,
    pub typed_append_generation_cells_kernel_event_nanos: u64,
    pub typed_append_generation_rows_kernel_event_nanos: u64,
    pub typed_append_generation_reduce_kernel_event_nanos: u64,
    pub typed_append_generation_finalize_kernel_event_nanos: u64,
    /// Generic typed-append canonical envelope construction interval.
    pub typed_append_envelope_prepare_nanos: u64,
    /// Generic typed-append canonical body/envelope construction.
    pub typed_append_envelope_encode_nanos: u64,
    /// Construction-closed writer phases: S7 materialization, aggregate root preparation,
    /// aggregate body encode, outer-buffer reservation, and outer canonical encode.
    pub typed_append_s7_build_nanos: u64,
    pub typed_append_aggregate_prepare_nanos: u64,
    pub typed_append_aggregate_body_encode_nanos: u64,
    pub typed_append_outer_reserve_nanos: u64,
    pub typed_append_outer_encode_nanos: u64,
    /// Exact outer-encode attribution: canonical digest closure, packed-frame fill, bytewise
    /// storage checksum, serialized payload copy, and immutable authority seal.
    pub typed_append_outer_digest_nanos: u64,
    pub typed_append_outer_packed_encode_nanos: u64,
    pub typed_append_outer_checksum_nanos: u64,
    pub typed_append_outer_serialized_copy_nanos: u64,
    pub typed_append_outer_seal_nanos: u64,
    /// Redundant retained close of live writer output. The construction-closed route records
    /// zero; fresh recovery still uses the independent retained decoder.
    pub typed_append_semantics_close_nanos: u64,
    /// Generic pre-parent retention/allocator authority body construction.
    pub typed_append_authority_prepare_nanos: u64,
    /// Generic device-resident typed-append apply-plan compilation.
    pub typed_append_device_plan_compile_nanos: u64,
    /// Pre-parent authority commit/apply/publication before parent WAL reservation.
    pub typed_append_authority_commit_nanos: u64,
    /// Canonical proposal, WAL/status installation, and commit timestamp assignment.
    pub transaction_terminal_wal_status_nanos: u64,
    /// Transaction-lifetime release and post-publication maintenance after canonical apply.
    pub transaction_terminal_post_canonical_finalize_nanos: u64,
    /// Decode/coalesce of the canonical transaction record plus global GPU apply and publication.
    pub transaction_canonical_apply_publish_nanos: u64,
    /// The canonical terminal's state-machine apply plus binary transaction decode/coalesce.
    /// This is contained within `transaction_canonical_apply_publish_nanos`.
    pub transaction_canonical_record_apply_nanos: u64,
    /// GPU residency and index publication from either the decoded canonical transaction outcome
    /// or the exact live typed terminal's post-WAL physical append.
    /// This is contained within `transaction_canonical_apply_publish_nanos`.
    pub transaction_canonical_residency_publish_nanos: u64,
    pub commit_validation_reresolve_nanos: u64,
    pub canonical_wal_encode_append_claim_nanos: u64,
    pub durability_wait_nanos: u64,
    /// Engine/WAL group-flush begin work (claim, frame preparation, and encoding), not job IO wait.
    pub durability_begin_group_flush_nanos: u64,
    /// Actual durable job wait (`WalGroupFlushJob::commit`) after begin has handed work off.
    pub durability_job_wait_nanos: u64,
    /// Permanent aggregate physical FUA attribution. These are all zero for the in-memory
    /// development profile and are not statement-wall-clock components.
    pub fua_logical_groups: u64,
    pub fua_logical_payload_bytes: u64,
    pub fua_single_frame_padded_baseline_bytes: u64,
    pub fua_publish_turn_wait_nanos: u64,
    pub fua_publish_turn_wait_groups: u64,
    pub fua_published_frames: u64,
    pub fua_fenced_frames: u64,
    pub fua_fence_failures: u64,
    pub fua_payload_bytes: u64,
    pub fua_padded_bytes: u64,
    pub fua_stage_copy_nanos: u64,
    pub fua_stage_copy_frames: u64,
    pub fua_publish_to_claim_nanos: u64,
    pub fua_publish_to_claim_frames: u64,
    pub fua_claim_to_write_done_nanos: u64,
    pub fua_claim_to_write_done_frames: u64,
    pub fua_write_done_to_contiguous_cut_nanos: u64,
    pub fua_write_done_to_contiguous_cut_frames: u64,
    pub fua_contiguous_cut_events: u64,
    pub fua_contiguous_cut_advanced_frames: u64,
    pub fua_contiguous_cut_advance_max_frames: u64,
    pub fua_waiter_cut_to_observe_nanos: u64,
    pub fua_waiter_cut_to_observe_count: u64,
    pub fua_in_flight_depth_max: u64,
    pub fua_in_flight_depth_1: u64,
    pub fua_in_flight_depth_2: u64,
    pub fua_in_flight_depth_3_to_4: u64,
    pub fua_in_flight_depth_5_to_8: u64,
    pub fua_in_flight_depth_9_to_16: u64,
    pub fua_in_flight_depth_17_to_32: u64,
    pub fua_in_flight_depth_33_plus: u64,
    pub fua_controller_sustained_actions: u64,
    pub fua_controller_pending_probe_cover_actions: u64,
    pub fua_controller_qd1_samples: u64,
    pub fua_controller_qd1_sparse_actions: u64,
    pub fua_controller_qd1_verify_actions: u64,
    pub fua_controller_qd1_fast_actions: u64,
    pub fua_controller_unfragmented_actions: u64,
    pub fua_controller_pool_too_narrow: u64,
    pub fua_controller_empty_chunk: u64,
    pub fua_controller_insufficient_free_slots: u64,
    pub fua_controller_natural_depth: u64,
    pub fua_controller_segment_boundary: u64,
    pub fua_controller_amplification_cap: u64,
    pub fua_controller_fast_samples: u64,
    pub fua_controller_nonfast_samples: u64,
    pub fua_controller_transitions_to_verify: u64,
    pub fua_controller_transitions_to_fast: u64,
    pub fua_controller_transitions_to_sustained: u64,
    pub fua_controller_stale_qd1_samples: u64,
    pub fua_controller_unavailable_qd1_samples: u64,
    pub fua_controller_abandoned_qd1_samples: u64,
    pub fua_controller_protocol_faults: u64,
    pub fua_controller_protocol_fallback_actions: u64,
    /// Current controller phase is a gauge: 0=non-FUA, 1=sustained, 2=verify, 3=fast.
    pub fua_controller_phase: u64,
    pub fua_controller_verify_fast_streak: u64,
    pub fua_controller_sustained_remaining: u64,
    pub fua_controller_generation: u64,
    pub fua_controller_pending_qd1_samples: u64,
    pub fua_controller_fast_in_flight: u64,
    pub fua_controller_fast_in_flight_max: u64,
    pub fua_controller_generation_exhausted: u64,
    pub fua_controller_ordinal_exhausted: u64,
    pub fua_controller_action_reconciliation: u64,
    pub fua_controller_sample_reconciliation: u64,
    pub device_validate_nanos: u64,
    pub device_h2d_append_index_apply_nanos: u64,
    pub publication_status_ack_nanos: u64,
    pub wave_count: u64,
    pub wave_item_count: u64,
    pub peak_host_statement_bytes: u64,
    pub peak_device_statement_bytes_estimate: u64,
    /// Completed rollover generations in this session delta.
    pub rollover_count: u64,
    /// Sum and maximum of the capacity rows selected for completed rollover generations.
    pub rollover_capacity_rows_total: u64,
    pub rollover_capacity_rows_max: u64,
    /// Current and peak shard counts for the most recently rolled relation. `current` is a gauge
    /// at session seal; `peak` is monotonic evidence that the route did not collapse topology.
    pub current_shard_count: u64,
    pub peak_shard_count: u64,
    /// Persistent device allocations transferred by completed rollover descriptors.
    pub persistent_allocation_count: u64,
    /// Descriptor map clone + route-token retoken visits from every shard-map publication.
    pub descriptor_clone_visit_count: u64,
    /// Exact resident-accounting entries walked, distinct from capacity-fit candidates below.
    pub budget_scan_entries: u64,
    pub capacity_fit_evaluation_count: u64,
    pub sidecar_fill_bytes: u64,
    pub live_h2d_bytes: u64,
    pub named_index_shard_visits: u64,
    /// Saturating service-time remainder after named phase sums. It intentionally includes queue,
    /// table-access, maintenance, and cross-statement overlap rather than mislabelling them fsync.
    pub unattributed_or_concurrent_overlap_nanos: u64,
}

/// Resolved, non-secret execution configuration bound to an INSERT probe record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InsertProbeConfig {
    pub durability_backend: &'static str,
    pub fua_fence_lanes: u64,
    pub intent_lane_count: u64,
    /// The present product contract: `Off` remains compatibility syntax, not async acknowledgement.
    pub synchronous_commit_gate: &'static str,
    pub auto_admit_on_commit: bool,
    pub device_authoritative_commits: u64,
}

/// Geometry transferred by one completed resident rollover.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct InsertProbeRolloverGeometry {
    pub(crate) capacity_rows: u64,
    pub(crate) current_shards: u64,
    pub(crate) persistent_allocations: u64,
    pub(crate) capacity_fit_evaluations: u64,
    pub(crate) budget_scans: u64,
    pub(crate) sidecar_fills: u64,
    pub(crate) live_h2d: u64,
}

impl InsertProbeSnapshot {
    /// Saturating per-field difference between two snapshots from the same engine instance.
    pub fn delta_since(self, before: Self) -> Self {
        macro_rules! delta {
            ($field:ident) => {
                self.$field.saturating_sub(before.$field)
            };
        }
        let successful_insert_end_to_end_service_nanos =
            delta!(successful_insert_end_to_end_service_nanos);
        let named_phase_nanos = delta!(facade_parse_bind_nanos)
            .saturating_add(delta!(engine_authorization_catalog_admission_nanos))
            .saturating_add(delta!(offlock_coercion_default_constraint_prepare_nanos))
            .saturating_add(delta!(transaction_begin_snapshot_capture_nanos))
            .saturating_add(delta!(transaction_statement_snapshot_capture_nanos))
            .saturating_add(delta!(transaction_commit_snapshot_capture_nanos))
            .saturating_add(delta!(transaction_private_overlay_materialize_nanos))
            .saturating_add(delta!(transaction_statement_stage_nanos))
            .saturating_add(delta!(transaction_terminal_nanos))
            .saturating_add(delta!(transaction_canonical_apply_publish_nanos))
            .saturating_add(delta!(commit_validation_reresolve_nanos))
            .saturating_add(delta!(canonical_wal_encode_append_claim_nanos))
            .saturating_add(delta!(durability_wait_nanos))
            .saturating_add(delta!(device_validate_nanos))
            .saturating_add(delta!(device_h2d_append_index_apply_nanos))
            .saturating_add(delta!(publication_status_ack_nanos));
        Self {
            successful_insert_statements: delta!(successful_insert_statements),
            successful_insert_rows: delta!(successful_insert_rows),
            row_local_check_launches: delta!(row_local_check_launches),
            row_local_check_verdicts: delta!(row_local_check_verdicts),
            raw_request_digest_derivations: delta!(raw_request_digest_derivations),
            raw_request_digest_derivation_bytes: delta!(raw_request_digest_derivation_bytes),
            successful_insert_source_bytes: delta!(successful_insert_source_bytes),
            successful_insert_end_to_end_service_nanos,
            facade_parse_bind_nanos: delta!(facade_parse_bind_nanos),
            engine_authorization_catalog_admission_nanos: delta!(
                engine_authorization_catalog_admission_nanos
            ),
            offlock_coercion_default_constraint_prepare_nanos: delta!(
                offlock_coercion_default_constraint_prepare_nanos
            ),
            transaction_begin_snapshot_capture_nanos: delta!(
                transaction_begin_snapshot_capture_nanos
            ),
            transaction_statement_snapshot_capture_nanos: delta!(
                transaction_statement_snapshot_capture_nanos
            ),
            transaction_commit_snapshot_capture_nanos: delta!(
                transaction_commit_snapshot_capture_nanos
            ),
            transaction_commit_snapshot_reuse_count: delta!(
                transaction_commit_snapshot_reuse_count
            ),
            transaction_private_overlay_materialize_nanos: delta!(
                transaction_private_overlay_materialize_nanos
            ),
            transaction_statement_stage_nanos: delta!(transaction_statement_stage_nanos),
            codec5_record_seal_nanos: delta!(codec5_record_seal_nanos),
            codec5_final_image_seal_nanos: delta!(codec5_final_image_seal_nanos),
            resident_append_source_materialize_nanos: delta!(
                resident_append_source_materialize_nanos
            ),
            typed_stage_constructor_payload_validate_nanos: delta!(
                typed_stage_constructor_payload_validate_nanos
            ),
            typed_stage_constructor_unique_slot_nanos: delta!(
                typed_stage_constructor_unique_slot_nanos
            ),
            typed_stage_constructor_payload_digest_nanos: delta!(
                typed_stage_constructor_payload_digest_nanos
            ),
            typed_overlay_payload_revalidate_nanos: delta!(typed_overlay_payload_revalidate_nanos),
            typed_overlay_payload_upload_nanos: delta!(typed_overlay_payload_upload_nanos),
            codec5_admission_select_materialize_nanos: delta!(
                codec5_admission_select_materialize_nanos
            ),
            codec5_admission_geometry_encode_nanos: delta!(codec5_admission_geometry_encode_nanos),
            codec5_terminal_select_materialize_nanos: delta!(
                codec5_terminal_select_materialize_nanos
            ),
            codec5_terminal_final_image_source_nanos: delta!(
                codec5_terminal_final_image_source_nanos
            ),
            transaction_terminal_nanos: delta!(transaction_terminal_nanos),
            transaction_terminal_pre_wal_nanos: delta!(transaction_terminal_pre_wal_nanos),
            codec5_operation_input_nanos: delta!(codec5_operation_input_nanos),
            codec5_operation_table_prepare_nanos: delta!(codec5_operation_table_prepare_nanos),
            codec5_operation_aggregate_nanos: delta!(codec5_operation_aggregate_nanos),
            codec5_operation_plan_compile_nanos: delta!(codec5_operation_plan_compile_nanos),
            codec5_operation_finalize_nanos: delta!(codec5_operation_finalize_nanos),
            codec5_live_s7_build_nanos: delta!(codec5_live_s7_build_nanos),
            codec5_live_s2_decode_nanos: delta!(codec5_live_s2_decode_nanos),
            codec5_live_aggregate_prepare_nanos: delta!(codec5_live_aggregate_prepare_nanos),
            codec5_live_aggregate_encode_nanos: delta!(codec5_live_aggregate_encode_nanos),
            codec5_live_outer_reserve_nanos: delta!(codec5_live_outer_reserve_nanos),
            codec5_live_outer_encode_nanos: delta!(codec5_live_outer_encode_nanos),
            codec5_live_outer_digest_nanos: delta!(codec5_live_outer_digest_nanos),
            codec5_live_outer_packed_encode_nanos: delta!(codec5_live_outer_packed_encode_nanos),
            codec5_live_outer_checksum_nanos: delta!(codec5_live_outer_checksum_nanos),
            codec5_live_outer_serialized_copy_nanos: delta!(
                codec5_live_outer_serialized_copy_nanos
            ),
            codec5_live_outer_seal_nanos: delta!(codec5_live_outer_seal_nanos),
            typed_append_generation_host_nanos: delta!(typed_append_generation_host_nanos),
            typed_append_generation_source_view_nanos: delta!(
                typed_append_generation_source_view_nanos
            ),
            typed_append_generation_prepare_reserve_nanos: delta!(
                typed_append_generation_prepare_reserve_nanos
            ),
            typed_append_generation_submit_complete_nanos: delta!(
                typed_append_generation_submit_complete_nanos
            ),
            typed_append_generation_kernel_event_nanos: delta!(
                typed_append_generation_kernel_event_nanos
            ),
            typed_append_generation_validate_kernel_event_nanos: delta!(
                typed_append_generation_validate_kernel_event_nanos
            ),
            typed_append_generation_cells_kernel_event_nanos: delta!(
                typed_append_generation_cells_kernel_event_nanos
            ),
            typed_append_generation_rows_kernel_event_nanos: delta!(
                typed_append_generation_rows_kernel_event_nanos
            ),
            typed_append_generation_reduce_kernel_event_nanos: delta!(
                typed_append_generation_reduce_kernel_event_nanos
            ),
            typed_append_generation_finalize_kernel_event_nanos: delta!(
                typed_append_generation_finalize_kernel_event_nanos
            ),
            typed_append_envelope_prepare_nanos: delta!(typed_append_envelope_prepare_nanos),
            typed_append_envelope_encode_nanos: delta!(typed_append_envelope_encode_nanos),
            typed_append_s7_build_nanos: delta!(typed_append_s7_build_nanos),
            typed_append_aggregate_prepare_nanos: delta!(typed_append_aggregate_prepare_nanos),
            typed_append_aggregate_body_encode_nanos: delta!(
                typed_append_aggregate_body_encode_nanos
            ),
            typed_append_outer_reserve_nanos: delta!(typed_append_outer_reserve_nanos),
            typed_append_outer_encode_nanos: delta!(typed_append_outer_encode_nanos),
            typed_append_outer_digest_nanos: delta!(typed_append_outer_digest_nanos),
            typed_append_outer_packed_encode_nanos: delta!(typed_append_outer_packed_encode_nanos),
            typed_append_outer_checksum_nanos: delta!(typed_append_outer_checksum_nanos),
            typed_append_outer_serialized_copy_nanos: delta!(
                typed_append_outer_serialized_copy_nanos
            ),
            typed_append_outer_seal_nanos: delta!(typed_append_outer_seal_nanos),
            typed_append_semantics_close_nanos: delta!(typed_append_semantics_close_nanos),
            typed_append_authority_prepare_nanos: delta!(typed_append_authority_prepare_nanos),
            typed_append_device_plan_compile_nanos: delta!(typed_append_device_plan_compile_nanos),
            typed_append_authority_commit_nanos: delta!(typed_append_authority_commit_nanos),
            transaction_terminal_wal_status_nanos: delta!(transaction_terminal_wal_status_nanos),
            transaction_terminal_post_canonical_finalize_nanos: delta!(
                transaction_terminal_post_canonical_finalize_nanos
            ),
            transaction_canonical_apply_publish_nanos: delta!(
                transaction_canonical_apply_publish_nanos
            ),
            transaction_canonical_record_apply_nanos: delta!(
                transaction_canonical_record_apply_nanos
            ),
            transaction_canonical_residency_publish_nanos: delta!(
                transaction_canonical_residency_publish_nanos
            ),
            commit_validation_reresolve_nanos: delta!(commit_validation_reresolve_nanos),
            canonical_wal_encode_append_claim_nanos: delta!(
                canonical_wal_encode_append_claim_nanos
            ),
            durability_wait_nanos: delta!(durability_wait_nanos),
            durability_begin_group_flush_nanos: delta!(durability_begin_group_flush_nanos),
            durability_job_wait_nanos: delta!(durability_job_wait_nanos),
            fua_logical_groups: delta!(fua_logical_groups),
            fua_logical_payload_bytes: delta!(fua_logical_payload_bytes),
            fua_single_frame_padded_baseline_bytes: delta!(fua_single_frame_padded_baseline_bytes),
            fua_publish_turn_wait_nanos: delta!(fua_publish_turn_wait_nanos),
            fua_publish_turn_wait_groups: delta!(fua_publish_turn_wait_groups),
            fua_published_frames: delta!(fua_published_frames),
            fua_fenced_frames: delta!(fua_fenced_frames),
            fua_fence_failures: delta!(fua_fence_failures),
            fua_payload_bytes: delta!(fua_payload_bytes),
            fua_padded_bytes: delta!(fua_padded_bytes),
            fua_stage_copy_nanos: delta!(fua_stage_copy_nanos),
            fua_stage_copy_frames: delta!(fua_stage_copy_frames),
            fua_publish_to_claim_nanos: delta!(fua_publish_to_claim_nanos),
            fua_publish_to_claim_frames: delta!(fua_publish_to_claim_frames),
            fua_claim_to_write_done_nanos: delta!(fua_claim_to_write_done_nanos),
            fua_claim_to_write_done_frames: delta!(fua_claim_to_write_done_frames),
            fua_write_done_to_contiguous_cut_nanos: delta!(fua_write_done_to_contiguous_cut_nanos),
            fua_write_done_to_contiguous_cut_frames: delta!(
                fua_write_done_to_contiguous_cut_frames
            ),
            fua_contiguous_cut_events: delta!(fua_contiguous_cut_events),
            fua_contiguous_cut_advanced_frames: delta!(fua_contiguous_cut_advanced_frames),
            fua_contiguous_cut_advance_max_frames: if self.fua_contiguous_cut_advance_max_frames
                > before.fua_contiguous_cut_advance_max_frames
            {
                self.fua_contiguous_cut_advance_max_frames
            } else {
                0
            },
            fua_waiter_cut_to_observe_nanos: delta!(fua_waiter_cut_to_observe_nanos),
            fua_waiter_cut_to_observe_count: delta!(fua_waiter_cut_to_observe_count),
            fua_in_flight_depth_max: if self.fua_in_flight_depth_max
                > before.fua_in_flight_depth_max
            {
                self.fua_in_flight_depth_max
            } else {
                0
            },
            fua_in_flight_depth_1: delta!(fua_in_flight_depth_1),
            fua_in_flight_depth_2: delta!(fua_in_flight_depth_2),
            fua_in_flight_depth_3_to_4: delta!(fua_in_flight_depth_3_to_4),
            fua_in_flight_depth_5_to_8: delta!(fua_in_flight_depth_5_to_8),
            fua_in_flight_depth_9_to_16: delta!(fua_in_flight_depth_9_to_16),
            fua_in_flight_depth_17_to_32: delta!(fua_in_flight_depth_17_to_32),
            fua_in_flight_depth_33_plus: delta!(fua_in_flight_depth_33_plus),
            fua_controller_sustained_actions: delta!(fua_controller_sustained_actions),
            fua_controller_pending_probe_cover_actions: delta!(
                fua_controller_pending_probe_cover_actions
            ),
            fua_controller_qd1_samples: delta!(fua_controller_qd1_samples),
            fua_controller_qd1_sparse_actions: delta!(fua_controller_qd1_sparse_actions),
            fua_controller_qd1_verify_actions: delta!(fua_controller_qd1_verify_actions),
            fua_controller_qd1_fast_actions: delta!(fua_controller_qd1_fast_actions),
            fua_controller_unfragmented_actions: delta!(fua_controller_unfragmented_actions),
            fua_controller_pool_too_narrow: delta!(fua_controller_pool_too_narrow),
            fua_controller_empty_chunk: delta!(fua_controller_empty_chunk),
            fua_controller_insufficient_free_slots: delta!(fua_controller_insufficient_free_slots),
            fua_controller_natural_depth: delta!(fua_controller_natural_depth),
            fua_controller_segment_boundary: delta!(fua_controller_segment_boundary),
            fua_controller_amplification_cap: delta!(fua_controller_amplification_cap),
            fua_controller_fast_samples: delta!(fua_controller_fast_samples),
            fua_controller_nonfast_samples: delta!(fua_controller_nonfast_samples),
            fua_controller_transitions_to_verify: delta!(fua_controller_transitions_to_verify),
            fua_controller_transitions_to_fast: delta!(fua_controller_transitions_to_fast),
            fua_controller_transitions_to_sustained: delta!(
                fua_controller_transitions_to_sustained
            ),
            fua_controller_stale_qd1_samples: delta!(fua_controller_stale_qd1_samples),
            fua_controller_unavailable_qd1_samples: delta!(fua_controller_unavailable_qd1_samples),
            fua_controller_abandoned_qd1_samples: delta!(fua_controller_abandoned_qd1_samples),
            fua_controller_protocol_faults: delta!(fua_controller_protocol_faults),
            fua_controller_protocol_fallback_actions: delta!(
                fua_controller_protocol_fallback_actions
            ),
            fua_controller_phase: self.fua_controller_phase,
            fua_controller_verify_fast_streak: self.fua_controller_verify_fast_streak,
            fua_controller_sustained_remaining: self.fua_controller_sustained_remaining,
            fua_controller_generation: self.fua_controller_generation,
            fua_controller_pending_qd1_samples: self.fua_controller_pending_qd1_samples,
            fua_controller_fast_in_flight: self.fua_controller_fast_in_flight,
            fua_controller_fast_in_flight_max: if self.fua_controller_fast_in_flight_max
                > before.fua_controller_fast_in_flight_max
            {
                self.fua_controller_fast_in_flight_max
            } else {
                0
            },
            fua_controller_generation_exhausted: self.fua_controller_generation_exhausted,
            fua_controller_ordinal_exhausted: self.fua_controller_ordinal_exhausted,
            fua_controller_action_reconciliation: self.fua_controller_action_reconciliation,
            fua_controller_sample_reconciliation: self.fua_controller_sample_reconciliation,
            device_validate_nanos: delta!(device_validate_nanos),
            device_h2d_append_index_apply_nanos: delta!(device_h2d_append_index_apply_nanos),
            publication_status_ack_nanos: delta!(publication_status_ack_nanos),
            wave_count: delta!(wave_count),
            wave_item_count: delta!(wave_item_count),
            peak_host_statement_bytes: if self.peak_host_statement_bytes
                > before.peak_host_statement_bytes
            {
                self.peak_host_statement_bytes
            } else {
                0
            },
            peak_device_statement_bytes_estimate: if self.peak_device_statement_bytes_estimate
                > before.peak_device_statement_bytes_estimate
            {
                self.peak_device_statement_bytes_estimate
            } else {
                0
            },
            rollover_count: delta!(rollover_count),
            rollover_capacity_rows_total: delta!(rollover_capacity_rows_total),
            rollover_capacity_rows_max: if self.rollover_capacity_rows_max
                > before.rollover_capacity_rows_max
            {
                self.rollover_capacity_rows_max
            } else {
                0
            },
            current_shard_count: self.current_shard_count,
            peak_shard_count: if self.peak_shard_count > before.peak_shard_count {
                self.peak_shard_count
            } else {
                0
            },
            persistent_allocation_count: delta!(persistent_allocation_count),
            descriptor_clone_visit_count: delta!(descriptor_clone_visit_count),
            budget_scan_entries: delta!(budget_scan_entries),
            capacity_fit_evaluation_count: delta!(capacity_fit_evaluation_count),
            sidecar_fill_bytes: delta!(sidecar_fill_bytes),
            live_h2d_bytes: delta!(live_h2d_bytes),
            named_index_shard_visits: delta!(named_index_shard_visits),
            unattributed_or_concurrent_overlap_nanos: successful_insert_end_to_end_service_nanos
                .saturating_sub(named_phase_nanos),
        }
    }
}

#[derive(Default)]
pub(crate) struct InsertProbeCounters {
    successful_insert_statements: AtomicU64,
    successful_insert_rows: AtomicU64,
    row_local_check_launches: AtomicU64,
    row_local_check_verdicts: AtomicU64,
    raw_request_digest_derivations: AtomicU64,
    raw_request_digest_derivation_bytes: AtomicU64,
    successful_insert_source_bytes: AtomicU64,
    successful_insert_end_to_end_service_nanos: AtomicU64,
    facade_parse_bind_nanos: AtomicU64,
    engine_authorization_catalog_admission_nanos: AtomicU64,
    offlock_coercion_default_constraint_prepare_nanos: AtomicU64,
    transaction_begin_snapshot_capture_nanos: AtomicU64,
    transaction_statement_snapshot_capture_nanos: AtomicU64,
    transaction_commit_snapshot_capture_nanos: AtomicU64,
    transaction_commit_snapshot_reuse_count: AtomicU64,
    transaction_private_overlay_materialize_nanos: AtomicU64,
    transaction_statement_stage_nanos: AtomicU64,
    codec5_record_seal_nanos: AtomicU64,
    codec5_final_image_seal_nanos: AtomicU64,
    resident_append_source_materialize_nanos: AtomicU64,
    typed_stage_constructor_payload_validate_nanos: AtomicU64,
    typed_stage_constructor_unique_slot_nanos: AtomicU64,
    typed_stage_constructor_payload_digest_nanos: AtomicU64,
    typed_overlay_payload_revalidate_nanos: AtomicU64,
    typed_overlay_payload_upload_nanos: AtomicU64,
    codec5_admission_select_materialize_nanos: AtomicU64,
    codec5_admission_geometry_encode_nanos: AtomicU64,
    codec5_terminal_select_materialize_nanos: AtomicU64,
    codec5_terminal_final_image_source_nanos: AtomicU64,
    transaction_terminal_nanos: AtomicU64,
    transaction_terminal_pre_wal_nanos: AtomicU64,
    codec5_operation_input_nanos: AtomicU64,
    codec5_operation_table_prepare_nanos: AtomicU64,
    codec5_operation_aggregate_nanos: AtomicU64,
    codec5_operation_plan_compile_nanos: AtomicU64,
    codec5_operation_finalize_nanos: AtomicU64,
    codec5_live_s7_build_nanos: AtomicU64,
    codec5_live_s2_decode_nanos: AtomicU64,
    codec5_live_aggregate_prepare_nanos: AtomicU64,
    codec5_live_aggregate_encode_nanos: AtomicU64,
    codec5_live_outer_reserve_nanos: AtomicU64,
    codec5_live_outer_encode_nanos: AtomicU64,
    codec5_live_outer_digest_nanos: AtomicU64,
    codec5_live_outer_packed_encode_nanos: AtomicU64,
    codec5_live_outer_checksum_nanos: AtomicU64,
    codec5_live_outer_serialized_copy_nanos: AtomicU64,
    codec5_live_outer_seal_nanos: AtomicU64,
    typed_append_generation_host_nanos: AtomicU64,
    typed_append_generation_source_view_nanos: AtomicU64,
    typed_append_generation_prepare_reserve_nanos: AtomicU64,
    typed_append_generation_submit_complete_nanos: AtomicU64,
    typed_append_generation_kernel_event_nanos: AtomicU64,
    typed_append_generation_validate_kernel_event_nanos: AtomicU64,
    typed_append_generation_cells_kernel_event_nanos: AtomicU64,
    typed_append_generation_rows_kernel_event_nanos: AtomicU64,
    typed_append_generation_reduce_kernel_event_nanos: AtomicU64,
    typed_append_generation_finalize_kernel_event_nanos: AtomicU64,
    typed_append_envelope_prepare_nanos: AtomicU64,
    typed_append_envelope_encode_nanos: AtomicU64,
    typed_append_s7_build_nanos: AtomicU64,
    typed_append_aggregate_prepare_nanos: AtomicU64,
    typed_append_aggregate_body_encode_nanos: AtomicU64,
    typed_append_outer_reserve_nanos: AtomicU64,
    typed_append_outer_encode_nanos: AtomicU64,
    typed_append_outer_digest_nanos: AtomicU64,
    typed_append_outer_packed_encode_nanos: AtomicU64,
    typed_append_outer_checksum_nanos: AtomicU64,
    typed_append_outer_serialized_copy_nanos: AtomicU64,
    typed_append_outer_seal_nanos: AtomicU64,
    typed_append_semantics_close_nanos: AtomicU64,
    typed_append_authority_prepare_nanos: AtomicU64,
    typed_append_device_plan_compile_nanos: AtomicU64,
    typed_append_authority_commit_nanos: AtomicU64,
    transaction_terminal_wal_status_nanos: AtomicU64,
    transaction_terminal_post_canonical_finalize_nanos: AtomicU64,
    transaction_canonical_apply_publish_nanos: AtomicU64,
    transaction_canonical_record_apply_nanos: AtomicU64,
    transaction_canonical_residency_publish_nanos: AtomicU64,
    commit_validation_reresolve_nanos: AtomicU64,
    canonical_wal_encode_append_claim_nanos: AtomicU64,
    durability_wait_nanos: AtomicU64,
    durability_begin_group_flush_nanos: AtomicU64,
    durability_job_wait_nanos: AtomicU64,
    device_validate_nanos: AtomicU64,
    device_h2d_append_index_apply_nanos: AtomicU64,
    publication_status_ack_nanos: AtomicU64,
    wave_count: AtomicU64,
    wave_item_count: AtomicU64,
    peak_host_statement_bytes: AtomicU64,
    peak_device_statement_bytes_estimate: AtomicU64,
    rollover_count: AtomicU64,
    rollover_capacity_rows_total: AtomicU64,
    rollover_capacity_rows_max: AtomicU64,
    current_shard_count: AtomicU64,
    peak_shard_count: AtomicU64,
    persistent_allocation_count: AtomicU64,
    budget_scan_entries: AtomicU64,
    capacity_fit_evaluation_count: AtomicU64,
    sidecar_fill_bytes: AtomicU64,
    live_h2d_bytes: AtomicU64,
    named_index_shard_visits: AtomicU64,
    unattributed_or_concurrent_overlap_nanos: AtomicU64,
}

impl InsertProbeCounters {
    fn snapshot(&self) -> InsertProbeSnapshot {
        macro_rules! load {
            ($field:ident) => {
                self.$field.load(Ordering::Relaxed)
            };
        }
        InsertProbeSnapshot {
            successful_insert_statements: load!(successful_insert_statements),
            successful_insert_rows: load!(successful_insert_rows),
            row_local_check_launches: load!(row_local_check_launches),
            row_local_check_verdicts: load!(row_local_check_verdicts),
            raw_request_digest_derivations: load!(raw_request_digest_derivations),
            raw_request_digest_derivation_bytes: load!(raw_request_digest_derivation_bytes),
            successful_insert_source_bytes: load!(successful_insert_source_bytes),
            successful_insert_end_to_end_service_nanos: load!(
                successful_insert_end_to_end_service_nanos
            ),
            facade_parse_bind_nanos: load!(facade_parse_bind_nanos),
            engine_authorization_catalog_admission_nanos: load!(
                engine_authorization_catalog_admission_nanos
            ),
            offlock_coercion_default_constraint_prepare_nanos: load!(
                offlock_coercion_default_constraint_prepare_nanos
            ),
            transaction_begin_snapshot_capture_nanos: load!(
                transaction_begin_snapshot_capture_nanos
            ),
            transaction_statement_snapshot_capture_nanos: load!(
                transaction_statement_snapshot_capture_nanos
            ),
            transaction_commit_snapshot_capture_nanos: load!(
                transaction_commit_snapshot_capture_nanos
            ),
            transaction_commit_snapshot_reuse_count: load!(transaction_commit_snapshot_reuse_count),
            transaction_private_overlay_materialize_nanos: load!(
                transaction_private_overlay_materialize_nanos
            ),
            transaction_statement_stage_nanos: load!(transaction_statement_stage_nanos),
            codec5_record_seal_nanos: load!(codec5_record_seal_nanos),
            codec5_final_image_seal_nanos: load!(codec5_final_image_seal_nanos),
            resident_append_source_materialize_nanos: load!(
                resident_append_source_materialize_nanos
            ),
            typed_stage_constructor_payload_validate_nanos: load!(
                typed_stage_constructor_payload_validate_nanos
            ),
            typed_stage_constructor_unique_slot_nanos: load!(
                typed_stage_constructor_unique_slot_nanos
            ),
            typed_stage_constructor_payload_digest_nanos: load!(
                typed_stage_constructor_payload_digest_nanos
            ),
            typed_overlay_payload_revalidate_nanos: load!(typed_overlay_payload_revalidate_nanos),
            typed_overlay_payload_upload_nanos: load!(typed_overlay_payload_upload_nanos),
            codec5_admission_select_materialize_nanos: load!(
                codec5_admission_select_materialize_nanos
            ),
            codec5_admission_geometry_encode_nanos: load!(codec5_admission_geometry_encode_nanos),
            codec5_terminal_select_materialize_nanos: load!(
                codec5_terminal_select_materialize_nanos
            ),
            codec5_terminal_final_image_source_nanos: load!(
                codec5_terminal_final_image_source_nanos
            ),
            transaction_terminal_nanos: load!(transaction_terminal_nanos),
            transaction_terminal_pre_wal_nanos: load!(transaction_terminal_pre_wal_nanos),
            codec5_operation_input_nanos: load!(codec5_operation_input_nanos),
            codec5_operation_table_prepare_nanos: load!(codec5_operation_table_prepare_nanos),
            codec5_operation_aggregate_nanos: load!(codec5_operation_aggregate_nanos),
            codec5_operation_plan_compile_nanos: load!(codec5_operation_plan_compile_nanos),
            codec5_operation_finalize_nanos: load!(codec5_operation_finalize_nanos),
            codec5_live_s7_build_nanos: load!(codec5_live_s7_build_nanos),
            codec5_live_s2_decode_nanos: load!(codec5_live_s2_decode_nanos),
            codec5_live_aggregate_prepare_nanos: load!(codec5_live_aggregate_prepare_nanos),
            codec5_live_aggregate_encode_nanos: load!(codec5_live_aggregate_encode_nanos),
            codec5_live_outer_reserve_nanos: load!(codec5_live_outer_reserve_nanos),
            codec5_live_outer_encode_nanos: load!(codec5_live_outer_encode_nanos),
            codec5_live_outer_digest_nanos: load!(codec5_live_outer_digest_nanos),
            codec5_live_outer_packed_encode_nanos: load!(codec5_live_outer_packed_encode_nanos),
            codec5_live_outer_checksum_nanos: load!(codec5_live_outer_checksum_nanos),
            codec5_live_outer_serialized_copy_nanos: load!(codec5_live_outer_serialized_copy_nanos),
            codec5_live_outer_seal_nanos: load!(codec5_live_outer_seal_nanos),
            typed_append_generation_host_nanos: load!(typed_append_generation_host_nanos),
            typed_append_generation_source_view_nanos: load!(
                typed_append_generation_source_view_nanos
            ),
            typed_append_generation_prepare_reserve_nanos: load!(
                typed_append_generation_prepare_reserve_nanos
            ),
            typed_append_generation_submit_complete_nanos: load!(
                typed_append_generation_submit_complete_nanos
            ),
            typed_append_generation_kernel_event_nanos: load!(
                typed_append_generation_kernel_event_nanos
            ),
            typed_append_generation_validate_kernel_event_nanos: load!(
                typed_append_generation_validate_kernel_event_nanos
            ),
            typed_append_generation_cells_kernel_event_nanos: load!(
                typed_append_generation_cells_kernel_event_nanos
            ),
            typed_append_generation_rows_kernel_event_nanos: load!(
                typed_append_generation_rows_kernel_event_nanos
            ),
            typed_append_generation_reduce_kernel_event_nanos: load!(
                typed_append_generation_reduce_kernel_event_nanos
            ),
            typed_append_generation_finalize_kernel_event_nanos: load!(
                typed_append_generation_finalize_kernel_event_nanos
            ),
            typed_append_envelope_prepare_nanos: load!(typed_append_envelope_prepare_nanos),
            typed_append_envelope_encode_nanos: load!(typed_append_envelope_encode_nanos),
            typed_append_s7_build_nanos: load!(typed_append_s7_build_nanos),
            typed_append_aggregate_prepare_nanos: load!(typed_append_aggregate_prepare_nanos),
            typed_append_aggregate_body_encode_nanos: load!(
                typed_append_aggregate_body_encode_nanos
            ),
            typed_append_outer_reserve_nanos: load!(typed_append_outer_reserve_nanos),
            typed_append_outer_encode_nanos: load!(typed_append_outer_encode_nanos),
            typed_append_outer_digest_nanos: load!(typed_append_outer_digest_nanos),
            typed_append_outer_packed_encode_nanos: load!(typed_append_outer_packed_encode_nanos),
            typed_append_outer_checksum_nanos: load!(typed_append_outer_checksum_nanos),
            typed_append_outer_serialized_copy_nanos: load!(
                typed_append_outer_serialized_copy_nanos
            ),
            typed_append_outer_seal_nanos: load!(typed_append_outer_seal_nanos),
            typed_append_semantics_close_nanos: load!(typed_append_semantics_close_nanos),
            typed_append_authority_prepare_nanos: load!(typed_append_authority_prepare_nanos),
            typed_append_device_plan_compile_nanos: load!(typed_append_device_plan_compile_nanos),
            typed_append_authority_commit_nanos: load!(typed_append_authority_commit_nanos),
            transaction_terminal_wal_status_nanos: load!(transaction_terminal_wal_status_nanos),
            transaction_terminal_post_canonical_finalize_nanos: load!(
                transaction_terminal_post_canonical_finalize_nanos
            ),
            transaction_canonical_apply_publish_nanos: load!(
                transaction_canonical_apply_publish_nanos
            ),
            transaction_canonical_record_apply_nanos: load!(
                transaction_canonical_record_apply_nanos
            ),
            transaction_canonical_residency_publish_nanos: load!(
                transaction_canonical_residency_publish_nanos
            ),
            commit_validation_reresolve_nanos: load!(commit_validation_reresolve_nanos),
            canonical_wal_encode_append_claim_nanos: load!(canonical_wal_encode_append_claim_nanos),
            durability_wait_nanos: load!(durability_wait_nanos),
            durability_begin_group_flush_nanos: load!(durability_begin_group_flush_nanos),
            durability_job_wait_nanos: load!(durability_job_wait_nanos),
            device_validate_nanos: load!(device_validate_nanos),
            device_h2d_append_index_apply_nanos: load!(device_h2d_append_index_apply_nanos),
            publication_status_ack_nanos: load!(publication_status_ack_nanos),
            wave_count: load!(wave_count),
            wave_item_count: load!(wave_item_count),
            peak_host_statement_bytes: load!(peak_host_statement_bytes),
            peak_device_statement_bytes_estimate: load!(peak_device_statement_bytes_estimate),
            rollover_count: load!(rollover_count),
            rollover_capacity_rows_total: load!(rollover_capacity_rows_total),
            rollover_capacity_rows_max: load!(rollover_capacity_rows_max),
            current_shard_count: load!(current_shard_count),
            peak_shard_count: load!(peak_shard_count),
            persistent_allocation_count: load!(persistent_allocation_count),
            descriptor_clone_visit_count: 0,
            budget_scan_entries: load!(budget_scan_entries),
            capacity_fit_evaluation_count: load!(capacity_fit_evaluation_count),
            sidecar_fill_bytes: load!(sidecar_fill_bytes),
            live_h2d_bytes: load!(live_h2d_bytes),
            named_index_shard_visits: load!(named_index_shard_visits),
            unattributed_or_concurrent_overlap_nanos: load!(
                unattributed_or_concurrent_overlap_nanos
            ),
            ..InsertProbeSnapshot::default()
        }
    }

    fn add(counter: &AtomicU64, nanos: u64) {
        counter.fetch_add(nanos, Ordering::Relaxed);
    }

    fn record_success(&self, rows: u64) {
        self.successful_insert_statements
            .fetch_add(1, Ordering::Relaxed);
        self.successful_insert_rows
            .fetch_add(rows, Ordering::Relaxed);
    }

    fn record_row_local_check_launch(&self) {
        self.row_local_check_launches
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_row_local_check_verdict(&self) {
        self.row_local_check_verdicts
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_raw_request_digest_derivation(&self, bytes: u64) {
        self.raw_request_digest_derivations
            .fetch_add(1, Ordering::Relaxed);
        self.raw_request_digest_derivation_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_successful_service_nanos(&self, nanos: u64) {
        Self::add(&self.successful_insert_end_to_end_service_nanos, nanos);
    }

    fn record_successful_raw_simple_query_bytes(&self, source_bytes: u64) {
        self.successful_insert_source_bytes
            .fetch_add(source_bytes, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_wave(&self, items: u64) {
        self.wave_count.fetch_add(1, Ordering::Relaxed);
        self.wave_item_count.fetch_add(items, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_peak(&self, host_bytes: u64, device_bytes_estimate: u64) {
        self.peak_host_statement_bytes
            .fetch_max(host_bytes, Ordering::Relaxed);
        self.peak_device_statement_bytes_estimate
            .fetch_max(device_bytes_estimate, Ordering::Relaxed);
    }

    fn record_rollover_geometry(&self, geometry: InsertProbeRolloverGeometry) {
        self.rollover_count.fetch_add(1, Ordering::Relaxed);
        self.rollover_capacity_rows_total
            .fetch_add(geometry.capacity_rows, Ordering::Relaxed);
        self.rollover_capacity_rows_max
            .fetch_max(geometry.capacity_rows, Ordering::Relaxed);
        self.current_shard_count
            .store(geometry.current_shards, Ordering::Relaxed);
        self.peak_shard_count
            .fetch_max(geometry.current_shards, Ordering::Relaxed);
        self.persistent_allocation_count
            .fetch_add(geometry.persistent_allocations, Ordering::Relaxed);
        self.budget_scan_entries
            .fetch_add(geometry.budget_scans, Ordering::Relaxed);
        self.capacity_fit_evaluation_count
            .fetch_add(geometry.capacity_fit_evaluations, Ordering::Relaxed);
        self.sidecar_fill_bytes
            .fetch_add(geometry.sidecar_fills, Ordering::Relaxed);
        self.live_h2d_bytes
            .fetch_add(geometry.live_h2d, Ordering::Relaxed);
    }

    fn record_named_index_shard_visits(&self, visits: u64) {
        self.named_index_shard_visits
            .fetch_add(visits, Ordering::Relaxed);
    }
}

impl Engine {
    /// Return the monotonic, engine-local aggregate INSERT qualification counters.
    pub fn insert_probe_snapshot(&self) -> InsertProbeSnapshot {
        let mut snapshot = self.insert_probe.snapshot();
        snapshot.descriptor_clone_visit_count = self
            .read_state
            .residency
            .insert_probe_descriptor_clone_retoken_visits
            .load(Ordering::Relaxed);
        let commit = self.commit_state();
        let fua = commit.wal.fua_durability_telemetry();
        snapshot.fua_logical_groups = fua.logical_groups;
        snapshot.fua_logical_payload_bytes = fua.logical_payload_bytes;
        snapshot.fua_single_frame_padded_baseline_bytes = fua.single_frame_padded_baseline_bytes;
        snapshot.fua_publish_turn_wait_nanos = fua.publish_turn_wait_nanos;
        snapshot.fua_publish_turn_wait_groups = fua.publish_turn_wait_groups;
        snapshot.fua_published_frames = fua.published_frames;
        snapshot.fua_fenced_frames = fua.fenced_frames;
        snapshot.fua_fence_failures = fua.fence_failures;
        snapshot.fua_payload_bytes = fua.payload_bytes;
        snapshot.fua_padded_bytes = fua.padded_bytes;
        snapshot.fua_stage_copy_nanos = fua.stage_copy_nanos;
        snapshot.fua_stage_copy_frames = fua.stage_copy_frames;
        snapshot.fua_publish_to_claim_nanos = fua.publish_to_claim_nanos;
        snapshot.fua_publish_to_claim_frames = fua.publish_to_claim_frames;
        snapshot.fua_claim_to_write_done_nanos = fua.claim_to_write_done_nanos;
        snapshot.fua_claim_to_write_done_frames = fua.claim_to_write_done_frames;
        snapshot.fua_write_done_to_contiguous_cut_nanos = fua.write_done_to_contiguous_cut_nanos;
        snapshot.fua_write_done_to_contiguous_cut_frames = fua.write_done_to_contiguous_cut_frames;
        snapshot.fua_contiguous_cut_events = fua.contiguous_cut_events;
        snapshot.fua_contiguous_cut_advanced_frames = fua.contiguous_cut_advanced_frames;
        snapshot.fua_contiguous_cut_advance_max_frames = fua.contiguous_cut_advance_max_frames;
        snapshot.fua_waiter_cut_to_observe_nanos = fua.waiter_cut_to_observe_nanos;
        snapshot.fua_waiter_cut_to_observe_count = fua.waiter_cut_to_observe_count;
        snapshot.fua_in_flight_depth_max = fua.in_flight_depth_max;
        snapshot.fua_in_flight_depth_1 = fua.in_flight_depth_histogram[0];
        snapshot.fua_in_flight_depth_2 = fua.in_flight_depth_histogram[1];
        snapshot.fua_in_flight_depth_3_to_4 = fua.in_flight_depth_histogram[2];
        snapshot.fua_in_flight_depth_5_to_8 = fua.in_flight_depth_histogram[3];
        snapshot.fua_in_flight_depth_9_to_16 = fua.in_flight_depth_histogram[4];
        snapshot.fua_in_flight_depth_17_to_32 = fua.in_flight_depth_histogram[5];
        snapshot.fua_in_flight_depth_33_plus = fua.in_flight_depth_histogram[6];
        snapshot.fua_controller_sustained_actions = fua.controller_sustained_actions;
        snapshot.fua_controller_pending_probe_cover_actions =
            fua.controller_pending_probe_cover_actions;
        snapshot.fua_controller_qd1_samples = fua.controller_qd1_samples;
        snapshot.fua_controller_qd1_sparse_actions = fua.controller_qd1_sparse_actions;
        snapshot.fua_controller_qd1_verify_actions = fua.controller_qd1_verify_actions;
        snapshot.fua_controller_qd1_fast_actions = fua.controller_qd1_fast_actions;
        snapshot.fua_controller_unfragmented_actions = fua.controller_unfragmented_actions;
        snapshot.fua_controller_pool_too_narrow = fua.controller_pool_too_narrow;
        snapshot.fua_controller_empty_chunk = fua.controller_empty_chunk;
        snapshot.fua_controller_insufficient_free_slots = fua.controller_insufficient_free_slots;
        snapshot.fua_controller_natural_depth = fua.controller_natural_depth;
        snapshot.fua_controller_segment_boundary = fua.controller_segment_boundary;
        snapshot.fua_controller_amplification_cap = fua.controller_amplification_cap;
        snapshot.fua_controller_fast_samples = fua.controller_fast_samples;
        snapshot.fua_controller_nonfast_samples = fua.controller_nonfast_samples;
        snapshot.fua_controller_transitions_to_verify = fua.controller_transitions_to_verify;
        snapshot.fua_controller_transitions_to_fast = fua.controller_transitions_to_fast;
        snapshot.fua_controller_transitions_to_sustained = fua.controller_transitions_to_sustained;
        snapshot.fua_controller_stale_qd1_samples = fua.controller_stale_qd1_samples;
        snapshot.fua_controller_unavailable_qd1_samples = fua.controller_unavailable_qd1_samples;
        snapshot.fua_controller_abandoned_qd1_samples = fua.controller_abandoned_qd1_samples;
        snapshot.fua_controller_protocol_faults = fua.controller_protocol_faults;
        snapshot.fua_controller_protocol_fallback_actions =
            fua.controller_protocol_fallback_actions;
        snapshot.fua_controller_phase = fua.controller_phase;
        snapshot.fua_controller_verify_fast_streak = fua.controller_verify_fast_streak;
        snapshot.fua_controller_sustained_remaining = fua.controller_sustained_remaining;
        snapshot.fua_controller_generation = fua.controller_generation;
        snapshot.fua_controller_pending_qd1_samples = fua.controller_pending_qd1_samples;
        snapshot.fua_controller_fast_in_flight = fua.controller_fast_in_flight;
        snapshot.fua_controller_fast_in_flight_max = fua.controller_fast_in_flight_max;
        snapshot.fua_controller_generation_exhausted = fua.controller_generation_exhausted;
        snapshot.fua_controller_ordinal_exhausted = fua.controller_ordinal_exhausted;
        snapshot.fua_controller_action_reconciliation = fua.controller_action_reconciliation;
        snapshot.fua_controller_sample_reconciliation = fua.controller_sample_reconciliation;
        snapshot
    }

    /// Return resolved, non-secret execution configuration for a probe record.
    pub fn insert_probe_config(&self) -> InsertProbeConfig {
        let commit = self.commit_state();
        let durability_backend = if !commit.wal.is_durable() {
            "memory"
        } else if commit.wal.is_fua_durable() {
            "fua"
        } else {
            "serial"
        };
        let fua_fence_lanes = commit.wal.fua_durability_telemetry().configured_fence_lanes;
        drop(commit);
        InsertProbeConfig {
            durability_backend,
            fua_fence_lanes,
            intent_lane_count: self
                .intent_lanes
                .as_ref()
                .map_or(0, |lanes| lanes.lane_count as u64),
            synchronous_commit_gate: "strict_rpo0",
            auto_admit_on_commit: self.auto_admit_on_commit_enabled(),
            device_authoritative_commits: self.device_authoritative_commits(),
        }
    }

    /// Feature-forwarded facade seam. Kept public only because the facade owns SQL parsing.
    #[doc(hidden)]
    pub fn record_insert_probe_facade_parse_bind_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.facade_parse_bind_nanos, nanos);
    }

    pub(crate) fn record_insert_probe_authorization_catalog_admission_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(
            &self
                .insert_probe
                .engine_authorization_catalog_admission_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_transaction_begin_snapshot_capture_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(
            &self.insert_probe.transaction_begin_snapshot_capture_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_transaction_statement_snapshot_capture_nanos(
        &self,
        nanos: u64,
    ) {
        InsertProbeCounters::add(
            &self
                .insert_probe
                .transaction_statement_snapshot_capture_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_transaction_commit_snapshot_capture_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(
            &self.insert_probe.transaction_commit_snapshot_capture_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_transaction_commit_snapshot_reuse(&self) {
        self.insert_probe
            .transaction_commit_snapshot_reuse_count
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_insert_probe_transaction_private_overlay_materialize_nanos(
        &self,
        nanos: u64,
    ) {
        InsertProbeCounters::add(
            &self
                .insert_probe
                .transaction_private_overlay_materialize_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_transaction_statement_stage_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.transaction_statement_stage_nanos, nanos);
    }

    pub(crate) fn record_insert_probe_codec5_seal_nanos(
        &self,
        record_nanos: u64,
        final_image_nanos: u64,
        resident_source_nanos: u64,
    ) {
        InsertProbeCounters::add(&self.insert_probe.codec5_record_seal_nanos, record_nanos);
        InsertProbeCounters::add(
            &self.insert_probe.codec5_final_image_seal_nanos,
            final_image_nanos,
        );
        InsertProbeCounters::add(
            &self.insert_probe.resident_append_source_materialize_nanos,
            resident_source_nanos,
        );
    }

    /// Permanent ownership/revalidation partition for the one staged typed INSERT lifecycle.
    /// These are non-additive named seams used to prove that a later optimization deletes
    /// duplicate work rather than moving it behind another carrier or terminal.
    pub(crate) fn record_insert_probe_staged_payload_nanos(&self, phases: [u64; 5]) {
        let [constructor_validate, unique_slots, payload_digest, overlay_revalidate, overlay_upload] =
            phases;
        InsertProbeCounters::add(
            &self
                .insert_probe
                .typed_stage_constructor_payload_validate_nanos,
            constructor_validate,
        );
        InsertProbeCounters::add(
            &self.insert_probe.typed_stage_constructor_unique_slot_nanos,
            unique_slots,
        );
        InsertProbeCounters::add(
            &self
                .insert_probe
                .typed_stage_constructor_payload_digest_nanos,
            payload_digest,
        );
        InsertProbeCounters::add(
            &self.insert_probe.typed_overlay_payload_revalidate_nanos,
            overlay_revalidate,
        );
        InsertProbeCounters::add(
            &self.insert_probe.typed_overlay_payload_upload_nanos,
            overlay_upload,
        );
    }

    /// Permanent admission/terminal materialization partition for the one codec-5 lifecycle.
    pub(crate) fn record_insert_probe_codec5_materialization_nanos(&self, phases: [u64; 4]) {
        let [admission_select, admission_geometry, terminal_select, final_image_source] = phases;
        InsertProbeCounters::add(
            &self.insert_probe.codec5_admission_select_materialize_nanos,
            admission_select,
        );
        InsertProbeCounters::add(
            &self.insert_probe.codec5_admission_geometry_encode_nanos,
            admission_geometry,
        );
        InsertProbeCounters::add(
            &self.insert_probe.codec5_terminal_select_materialize_nanos,
            terminal_select,
        );
        InsertProbeCounters::add(
            &self.insert_probe.codec5_terminal_final_image_source_nanos,
            final_image_source,
        );
    }

    pub(crate) fn record_insert_probe_transaction_terminal_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.transaction_terminal_nanos, nanos);
    }

    pub(crate) fn record_insert_probe_transaction_terminal_pre_wal_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.transaction_terminal_pre_wal_nanos, nanos);
    }

    /// Build-only phase partition for the existing generic codec-5 operation. The finalizer
    /// still owns exactly one typed record, `DeviceInsertPlan`, WAL/status transition, and GPU
    /// publication; this merely localizes the pre-WAL service-time gap.
    pub(crate) fn record_insert_probe_codec5_operation_nanos(&self, phases: [u64; 5]) {
        let [input, table_prepare, aggregate, plan_compile, finalize] = phases;
        InsertProbeCounters::add(&self.insert_probe.codec5_operation_input_nanos, input);
        InsertProbeCounters::add(
            &self.insert_probe.codec5_operation_table_prepare_nanos,
            table_prepare,
        );
        InsertProbeCounters::add(
            &self.insert_probe.codec5_operation_aggregate_nanos,
            aggregate,
        );
        InsertProbeCounters::add(
            &self.insert_probe.codec5_operation_plan_compile_nanos,
            plan_compile,
        );
        InsertProbeCounters::add(&self.insert_probe.codec5_operation_finalize_nanos, finalize);
    }

    /// Build-only timing for the one already-closed codec-5 aggregate/envelope encoder. It is
    /// an attribution seam beneath the same record and never revives the retired writer path.
    pub(crate) fn record_insert_probe_codec5_live_encoder_nanos(&self, phases: [u64; 11]) {
        let [s7, s2_decode, aggregate_prepare, aggregate_encode, outer_reserve, outer_encode, digest, packed, checksum, serialized_copy, seal] =
            phases;
        InsertProbeCounters::add(&self.insert_probe.codec5_live_s7_build_nanos, s7);
        InsertProbeCounters::add(&self.insert_probe.codec5_live_s2_decode_nanos, s2_decode);
        InsertProbeCounters::add(
            &self.insert_probe.codec5_live_aggregate_prepare_nanos,
            aggregate_prepare,
        );
        InsertProbeCounters::add(
            &self.insert_probe.codec5_live_aggregate_encode_nanos,
            aggregate_encode,
        );
        InsertProbeCounters::add(
            &self.insert_probe.codec5_live_outer_reserve_nanos,
            outer_reserve,
        );
        InsertProbeCounters::add(
            &self.insert_probe.codec5_live_outer_encode_nanos,
            outer_encode,
        );
        InsertProbeCounters::add(&self.insert_probe.codec5_live_outer_digest_nanos, digest);
        InsertProbeCounters::add(
            &self.insert_probe.codec5_live_outer_packed_encode_nanos,
            packed,
        );
        InsertProbeCounters::add(
            &self.insert_probe.codec5_live_outer_checksum_nanos,
            checksum,
        );
        InsertProbeCounters::add(
            &self.insert_probe.codec5_live_outer_serialized_copy_nanos,
            serialized_copy,
        );
        InsertProbeCounters::add(&self.insert_probe.codec5_live_outer_seal_nanos, seal);
    }

    /// Build-only attribution for the sole generic typed-generation call.  The host interval
    /// includes its ABI preparation, DMA, launch, synchronization, and proof decoding; CUDA
    /// events isolate only the five generation kernels inside that same call.
    pub(crate) fn record_insert_probe_typed_append_generation_nanos(
        &self,
        host_nanos: u64,
        source_view_nanos: u64,
        prepare_reserve_nanos: u64,
        submit_complete_nanos: u64,
        kernel_event_nanos: Option<u64>,
        kernel_phase_event_nanos: Option<[u64; 5]>,
    ) {
        InsertProbeCounters::add(
            &self.insert_probe.typed_append_generation_host_nanos,
            host_nanos,
        );
        InsertProbeCounters::add(
            &self.insert_probe.typed_append_generation_source_view_nanos,
            source_view_nanos,
        );
        InsertProbeCounters::add(
            &self
                .insert_probe
                .typed_append_generation_prepare_reserve_nanos,
            prepare_reserve_nanos,
        );
        InsertProbeCounters::add(
            &self
                .insert_probe
                .typed_append_generation_submit_complete_nanos,
            submit_complete_nanos,
        );
        if let Some(kernel_nanos) = kernel_event_nanos {
            InsertProbeCounters::add(
                &self.insert_probe.typed_append_generation_kernel_event_nanos,
                kernel_nanos,
            );
        }
        if let Some([validate, cells, rows, reduce, finalize]) = kernel_phase_event_nanos {
            InsertProbeCounters::add(
                &self
                    .insert_probe
                    .typed_append_generation_validate_kernel_event_nanos,
                validate,
            );
            InsertProbeCounters::add(
                &self
                    .insert_probe
                    .typed_append_generation_cells_kernel_event_nanos,
                cells,
            );
            InsertProbeCounters::add(
                &self
                    .insert_probe
                    .typed_append_generation_rows_kernel_event_nanos,
                rows,
            );
            InsertProbeCounters::add(
                &self
                    .insert_probe
                    .typed_append_generation_reduce_kernel_event_nanos,
                reduce,
            );
            InsertProbeCounters::add(
                &self
                    .insert_probe
                    .typed_append_generation_finalize_kernel_event_nanos,
                finalize,
            );
        }
    }

    pub(crate) fn record_insert_probe_transaction_terminal_wal_status_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(
            &self.insert_probe.transaction_terminal_wal_status_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_transaction_terminal_post_canonical_finalize_nanos(
        &self,
        nanos: u64,
    ) {
        InsertProbeCounters::add(
            &self
                .insert_probe
                .transaction_terminal_post_canonical_finalize_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_transaction_canonical_apply_publish_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(
            &self.insert_probe.transaction_canonical_apply_publish_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_transaction_canonical_record_apply_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(
            &self.insert_probe.transaction_canonical_record_apply_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_transaction_canonical_residency_publish_nanos(
        &self,
        nanos: u64,
    ) {
        InsertProbeCounters::add(
            &self
                .insert_probe
                .transaction_canonical_residency_publish_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_durability_wait_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.durability_wait_nanos, nanos);
    }

    pub(crate) fn record_insert_probe_durability_begin_group_flush_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.durability_begin_group_flush_nanos, nanos);
    }

    pub(crate) fn record_insert_probe_durability_job_wait_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.durability_job_wait_nanos, nanos);
    }

    pub(crate) fn record_insert_probe_publication_status_ack_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.publication_status_ack_nanos, nanos);
    }

    pub(crate) fn record_insert_probe_rollover_geometry(
        &self,
        geometry: InsertProbeRolloverGeometry,
    ) {
        self.insert_probe.record_rollover_geometry(geometry);
    }

    pub(crate) fn record_insert_probe_named_index_shard_visits(&self, visits: u64) {
        self.insert_probe.record_named_index_shard_visits(visits);
    }

    pub(crate) fn record_insert_probe_success(&self, rows: u64) {
        self.insert_probe.record_success(rows);
    }

    pub(crate) fn record_insert_probe_row_local_check_launch(&self) {
        self.insert_probe.record_row_local_check_launch();
    }

    pub(crate) fn record_insert_probe_row_local_check_verdict(&self) {
        self.insert_probe.record_row_local_check_verdict();
    }

    /// Build-only accounting for the exact request bytes that canonical identity hashes.
    pub(crate) fn record_insert_probe_raw_request_digest_derivation(&self, bytes: u64) {
        self.insert_probe
            .record_raw_request_digest_derivation(bytes);
    }

    /// Feature-forwarded simple-text service seam. Raw source bytes belong to the server's
    /// Query-message boundary, which alone retains protocol terminators and trailing whitespace.
    #[doc(hidden)]
    pub fn record_insert_probe_successful_simple_text_service_nanos(&self, nanos: u64) {
        self.insert_probe.record_successful_service_nanos(nanos);
    }

    /// Feature-forwarded server seam. Count raw Query SQL bytes only after the server has proved
    /// exactly one successful INSERT in that message.
    #[doc(hidden)]
    pub fn record_insert_probe_successful_raw_simple_query_bytes(&self, source_bytes: u64) {
        self.insert_probe
            .record_successful_raw_simple_query_bytes(source_bytes);
    }

    /// Feature-forwarded prepared-portal seam. It records service time only because a bound
    /// portal owns canonical bound SQL rather than the original Bind-frame request bytes.
    #[doc(hidden)]
    pub fn record_insert_probe_successful_prepared_service_nanos(&self, nanos: u64) {
        self.insert_probe.record_successful_service_nanos(nanos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_are_engine_local_monotonic_and_delta_preserves_peaks() {
        let counters = InsertProbeCounters::default();
        counters.record_success(1_000);
        assert_eq!(counters.snapshot().successful_insert_source_bytes, 0);
        assert_eq!(counters.snapshot().raw_request_digest_derivations, 0);
        counters.record_successful_service_nanos(7);
        counters.record_successful_raw_simple_query_bytes(31_000);
        counters.record_raw_request_digest_derivation(30_998);
        counters.record_wave(1);
        counters.record_peak(31_000, 48_000);
        InsertProbeCounters::add(&counters.durability_wait_nanos, 7);
        let before = counters.snapshot();
        counters.record_success(1_000);
        counters.record_successful_service_nanos(100);
        counters.record_successful_raw_simple_query_bytes(31_000);
        counters.record_raw_request_digest_derivation(30_998);
        counters.record_wave(1);
        counters.record_peak(30_000, 64_000);
        counters.record_rollover_geometry(InsertProbeRolloverGeometry {
            capacity_rows: 4_096,
            current_shards: 2,
            persistent_allocations: 3,
            capacity_fit_evaluations: 13,
            budget_scans: 13,
            sidecar_fills: 32_768,
            live_h2d: 4_120,
        });
        counters.record_named_index_shard_visits(2);
        InsertProbeCounters::add(&counters.durability_wait_nanos, 11);
        let delta = counters.snapshot().delta_since(before);

        assert_eq!(delta.successful_insert_statements, 1);
        assert_eq!(delta.successful_insert_rows, 1_000);
        assert_eq!(delta.raw_request_digest_derivations, 1);
        assert_eq!(delta.raw_request_digest_derivation_bytes, 30_998);
        assert_eq!(delta.successful_insert_source_bytes, 31_000);
        assert_eq!(delta.wave_count, 1);
        assert_eq!(delta.durability_wait_nanos, 11);
        assert_eq!(delta.successful_insert_end_to_end_service_nanos, 100);
        assert_eq!(delta.unattributed_or_concurrent_overlap_nanos, 89);
        assert_eq!(delta.peak_host_statement_bytes, 0);
        assert_eq!(delta.peak_device_statement_bytes_estimate, 64_000);
        assert_eq!(delta.rollover_count, 1);
        assert_eq!(delta.rollover_capacity_rows_total, 4_096);
        assert_eq!(delta.rollover_capacity_rows_max, 4_096);
        assert_eq!(delta.current_shard_count, 2);
        assert_eq!(delta.peak_shard_count, 2);
        assert_eq!(delta.persistent_allocation_count, 3);
        assert_eq!(delta.descriptor_clone_visit_count, 0);
        assert_eq!(delta.budget_scan_entries, 13);
        assert_eq!(delta.capacity_fit_evaluation_count, 13);
        assert_eq!(delta.sidecar_fill_bytes, 32_768);
        assert_eq!(delta.live_h2d_bytes, 4_120);
        assert_eq!(delta.named_index_shard_visits, 2);
    }

    #[test]
    fn engine_snapshot_uses_exact_shard_publication_clone_and_retoken_visits() {
        let engine = Engine::new_local_test_engine();
        let before = engine.insert_probe_snapshot();
        engine
            .read_state
            .residency
            .insert_probe_descriptor_clone_retoken_visits
            .fetch_add(7, Ordering::Relaxed);
        let delta = engine.insert_probe_snapshot().delta_since(before);
        assert_eq!(delta.descriptor_clone_visit_count, 7);
    }
}
