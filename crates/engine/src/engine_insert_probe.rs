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
/// are not a GPU-kernel event timeline and are intentionally not summed into a total.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InsertProbeSnapshot {
    pub successful_insert_statements: u64,
    pub successful_insert_rows: u64,
    /// Successfully published INSERT-001 typed fixed-width statements. Qualification requires
    /// this to equal `successful_insert_statements` for its exact eligible workload.
    pub fixed_insert_typed_commits: u64,
    /// A sealed fixed candidate that restored the legacy wave before WAL.
    pub fixed_insert_legacy_fallbacks: u64,
    /// A device-authoritative typed preflight declined before WAL and returned retryable.
    pub fixed_insert_retryable_declines: u64,
    /// Qualified fixed candidates that entered the old general commit-validation/re-resolve path.
    pub fixed_insert_legacy_commit_validation_reresolves: u64,
    /// Legacy off-lock INSERT preparations that constructed a general `WriteDelta`.
    pub legacy_insert_delta_builds: u64,
    /// Predicted insert row keys materialized by those legacy preparations.
    pub predicted_row_keys_materialized: u64,
    /// Direct fixed-width carriers materialized without a `WriteDelta` or predicted row keys.
    pub direct_fixed_insert_carriers: u64,
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
    pub binary_wal_records_enabled: bool,
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
            .saturating_add(delta!(commit_validation_reresolve_nanos))
            .saturating_add(delta!(canonical_wal_encode_append_claim_nanos))
            .saturating_add(delta!(durability_wait_nanos))
            .saturating_add(delta!(device_validate_nanos))
            .saturating_add(delta!(device_h2d_append_index_apply_nanos))
            .saturating_add(delta!(publication_status_ack_nanos));
        Self {
            successful_insert_statements: delta!(successful_insert_statements),
            successful_insert_rows: delta!(successful_insert_rows),
            fixed_insert_typed_commits: delta!(fixed_insert_typed_commits),
            fixed_insert_legacy_fallbacks: delta!(fixed_insert_legacy_fallbacks),
            fixed_insert_retryable_declines: delta!(fixed_insert_retryable_declines),
            fixed_insert_legacy_commit_validation_reresolves: delta!(
                fixed_insert_legacy_commit_validation_reresolves
            ),
            legacy_insert_delta_builds: delta!(legacy_insert_delta_builds),
            predicted_row_keys_materialized: delta!(predicted_row_keys_materialized),
            direct_fixed_insert_carriers: delta!(direct_fixed_insert_carriers),
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
    fixed_insert_typed_commits: AtomicU64,
    fixed_insert_legacy_fallbacks: AtomicU64,
    fixed_insert_retryable_declines: AtomicU64,
    fixed_insert_legacy_commit_validation_reresolves: AtomicU64,
    legacy_insert_delta_builds: AtomicU64,
    predicted_row_keys_materialized: AtomicU64,
    direct_fixed_insert_carriers: AtomicU64,
    row_local_check_launches: AtomicU64,
    row_local_check_verdicts: AtomicU64,
    raw_request_digest_derivations: AtomicU64,
    raw_request_digest_derivation_bytes: AtomicU64,
    successful_insert_source_bytes: AtomicU64,
    successful_insert_end_to_end_service_nanos: AtomicU64,
    facade_parse_bind_nanos: AtomicU64,
    engine_authorization_catalog_admission_nanos: AtomicU64,
    offlock_coercion_default_constraint_prepare_nanos: AtomicU64,
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
            fixed_insert_typed_commits: load!(fixed_insert_typed_commits),
            fixed_insert_legacy_fallbacks: load!(fixed_insert_legacy_fallbacks),
            fixed_insert_retryable_declines: load!(fixed_insert_retryable_declines),
            fixed_insert_legacy_commit_validation_reresolves: load!(
                fixed_insert_legacy_commit_validation_reresolves
            ),
            legacy_insert_delta_builds: load!(legacy_insert_delta_builds),
            predicted_row_keys_materialized: load!(predicted_row_keys_materialized),
            direct_fixed_insert_carriers: load!(direct_fixed_insert_carriers),
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

    fn record_fixed_insert_typed_commit(&self) {
        self.fixed_insert_typed_commits
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_fixed_insert_legacy_fallback(&self) {
        self.fixed_insert_legacy_fallbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_fixed_insert_retryable_decline(&self) {
        self.fixed_insert_retryable_declines
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_fixed_insert_legacy_commit_validation_reresolve(&self) {
        self.fixed_insert_legacy_commit_validation_reresolves
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_legacy_insert_delta_build(&self, predicted_row_keys: u64) {
        self.legacy_insert_delta_builds
            .fetch_add(1, Ordering::Relaxed);
        self.predicted_row_keys_materialized
            .fetch_add(predicted_row_keys, Ordering::Relaxed);
    }

    fn record_direct_fixed_insert_carrier(&self) {
        self.direct_fixed_insert_carriers
            .fetch_add(1, Ordering::Relaxed);
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

    fn record_wave(&self, items: u64) {
        self.wave_count.fetch_add(1, Ordering::Relaxed);
        self.wave_item_count.fetch_add(items, Ordering::Relaxed);
    }

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
            binary_wal_records_enabled: self.binary_wal_records_enabled(),
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

    pub(crate) fn record_insert_probe_offlock_prepare_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(
            &self
                .insert_probe
                .offlock_coercion_default_constraint_prepare_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_commit_validation_reresolve_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.commit_validation_reresolve_nanos, nanos);
    }

    pub(crate) fn record_insert_probe_canonical_wal_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(
            &self.insert_probe.canonical_wal_encode_append_claim_nanos,
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

    pub(crate) fn record_insert_probe_device_validate_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.device_validate_nanos, nanos);
    }

    pub(crate) fn record_insert_probe_device_append_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(
            &self.insert_probe.device_h2d_append_index_apply_nanos,
            nanos,
        );
    }

    pub(crate) fn record_insert_probe_publication_status_ack_nanos(&self, nanos: u64) {
        InsertProbeCounters::add(&self.insert_probe.publication_status_ack_nanos, nanos);
    }

    pub(crate) fn record_insert_probe_wave(&self, items: u64) {
        self.insert_probe.record_wave(items);
    }

    pub(crate) fn record_insert_probe_peak_statement(
        &self,
        host_bytes: u64,
        device_bytes_estimate: u64,
    ) {
        self.insert_probe
            .record_peak(host_bytes, device_bytes_estimate);
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

    pub(crate) fn record_insert_probe_fixed_insert_typed_commit(&self) {
        self.insert_probe.record_fixed_insert_typed_commit();
    }

    pub(crate) fn record_insert_probe_fixed_insert_legacy_fallback(&self) {
        self.insert_probe.record_fixed_insert_legacy_fallback();
    }

    pub(crate) fn record_insert_probe_fixed_insert_retryable_decline(&self) {
        self.insert_probe.record_fixed_insert_retryable_decline();
    }

    pub(crate) fn record_insert_probe_fixed_insert_legacy_commit_validation_reresolve(&self) {
        self.insert_probe
            .record_fixed_insert_legacy_commit_validation_reresolve();
    }

    pub(crate) fn record_insert_probe_legacy_insert_delta_build(&self, predicted_row_keys: u64) {
        self.insert_probe
            .record_legacy_insert_delta_build(predicted_row_keys);
    }

    pub(crate) fn record_insert_probe_direct_fixed_insert_carrier(&self) {
        self.insert_probe.record_direct_fixed_insert_carrier();
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
        counters.record_fixed_insert_typed_commit();
        counters.record_fixed_insert_legacy_fallback();
        counters.record_fixed_insert_retryable_decline();
        counters.record_fixed_insert_legacy_commit_validation_reresolve();
        counters.record_legacy_insert_delta_build(1_000);
        counters.record_direct_fixed_insert_carrier();
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
        assert_eq!(delta.fixed_insert_typed_commits, 1);
        assert_eq!(delta.fixed_insert_legacy_fallbacks, 1);
        assert_eq!(delta.fixed_insert_retryable_declines, 1);
        assert_eq!(delta.fixed_insert_legacy_commit_validation_reresolves, 1);
        assert_eq!(delta.legacy_insert_delta_builds, 1);
        assert_eq!(delta.predicted_row_keys_materialized, 1_000);
        assert_eq!(delta.direct_fixed_insert_carriers, 1);
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
