//! Server-owned sealing of build-only INSERT qualification evidence.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use gpu_db_facade::{InsertProbeConfig, InsertProbeSnapshot, SharedEngine};

static ACTIVE_WORKLOAD_SESSIONS: AtomicU64 = AtomicU64::new(0);
static OVERLAP_EPOCH: AtomicU64 = AtomicU64::new(0);
/// Serializes only session enter/seal transitions, never query execution or classifier gates.
///
/// The epoch baseline must precede active-session publication, while sealing must capture the
/// process-global classifier counters before withdrawing that publication. Keeping those two
/// transitions mutually exclusive closes both short-overlap observation windows.
static PROBE_SESSION_TRANSITION: Mutex<()> = Mutex::new(());
static COMPAT_CLASSIFIER_GATE_CHECKS: AtomicU64 = AtomicU64::new(0);
static COMPAT_CLASSIFIER_GATE_REJECTIONS: AtomicU64 = AtomicU64::new(0);
static COMPAT_CLASSIFIER_SLOW_PATH_ADMISSIONS: AtomicU64 = AtomicU64::new(0);
static COMPAT_CLASSIFIER_SLOW_PATH_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
struct CompatClassifierProbeSnapshot {
    checks: u64,
    rejections: u64,
    slow_path_admissions: u64,
    slow_path_bytes: u64,
}

impl CompatClassifierProbeSnapshot {
    fn now() -> Self {
        Self {
            checks: COMPAT_CLASSIFIER_GATE_CHECKS.load(Ordering::Relaxed),
            rejections: COMPAT_CLASSIFIER_GATE_REJECTIONS.load(Ordering::Relaxed),
            slow_path_admissions: COMPAT_CLASSIFIER_SLOW_PATH_ADMISSIONS.load(Ordering::Relaxed),
            slow_path_bytes: COMPAT_CLASSIFIER_SLOW_PATH_BYTES.load(Ordering::Relaxed),
        }
    }

    fn delta_since(self, before: Self) -> Self {
        Self {
            checks: self.checks.saturating_sub(before.checks),
            rejections: self.rejections.saturating_sub(before.rejections),
            slow_path_admissions: self
                .slow_path_admissions
                .saturating_sub(before.slow_path_admissions),
            slow_path_bytes: self.slow_path_bytes.saturating_sub(before.slow_path_bytes),
        }
    }
}

/// Record one allocation-free compatibility classifier gate.
///
/// A positive gate enters the pre-existing canonicalizing parser and owns the input-byte
/// attribution. A rejection returns before that parser and has no slow-path byte cost.
pub(crate) fn record_compat_classifier_gate(statement_bytes: u64, admitted: bool) {
    COMPAT_CLASSIFIER_GATE_CHECKS.fetch_add(1, Ordering::Relaxed);
    if admitted {
        COMPAT_CLASSIFIER_SLOW_PATH_ADMISSIONS.fetch_add(1, Ordering::Relaxed);
        COMPAT_CLASSIFIER_SLOW_PATH_BYTES.fetch_add(statement_bytes, Ordering::Relaxed);
    } else {
        COMPAT_CLASSIFIER_GATE_REJECTIONS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Counts protocol sessions that reached the execution loop. Raw TCP readiness probes never
/// construct this guard because they return during startup framing.
pub(super) struct ProbeSessionGuard {
    overlap_epoch_at_enter: u64,
    overlapped_on_enter: bool,
    compat_classifier_before: CompatClassifierProbeSnapshot,
    sealed: AtomicBool,
}

#[derive(Clone, Copy, Debug)]
struct ProbeSessionSeal {
    compat_classifier_delta: CompatClassifierProbeSnapshot,
    active_workload_sessions_at_seal: u64,
    overlap_observed: bool,
}

impl ProbeSessionGuard {
    pub(super) fn enter() -> Self {
        Self::enter_after_active_publication(|| {})
    }

    /// The hook is test-only at its call site. It sits after publication so the regression can
    /// reproduce a complete overlapping session before the first guard finishes entering.
    fn enter_after_active_publication(after_active_publication: impl FnOnce()) -> Self {
        let (overlap_epoch_at_enter, overlapped_on_enter, compat_classifier_before) = {
            let _transition = probe_session_transition_lock();
            // This baseline is intentionally captured *before* publishing this session. If a
            // concurrent entrant races either side of the publication, one entrant increments
            // the durable epoch and the other guard observes the change when it seals.
            let overlap_epoch_at_enter = OVERLAP_EPOCH.load(Ordering::SeqCst);
            let already_active = ACTIVE_WORKLOAD_SESSIONS.fetch_add(1, Ordering::SeqCst);
            if already_active != 0 {
                OVERLAP_EPOCH.fetch_add(1, Ordering::SeqCst);
            }
            (
                overlap_epoch_at_enter,
                already_active != 0,
                CompatClassifierProbeSnapshot::now(),
            )
        };
        after_active_publication();
        Self {
            overlap_epoch_at_enter,
            overlapped_on_enter,
            compat_classifier_before,
            sealed: AtomicBool::new(false),
        }
    }

    /// Atomically seal the session's process-global attribution window before formatting it.
    ///
    /// A new session can therefore either enter before this transition and advance the durable
    /// overlap epoch, or enter after this guard withdraws itself and cannot contaminate this
    /// record's counter snapshot.
    fn seal(&self) -> ProbeSessionSeal {
        let _transition = probe_session_transition_lock();
        if self.sealed.load(Ordering::Acquire) {
            return ProbeSessionSeal {
                compat_classifier_delta: self.compat_classifier_delta(),
                active_workload_sessions_at_seal: 0,
                overlap_observed: true,
            };
        }
        let seal = ProbeSessionSeal {
            compat_classifier_delta: self.compat_classifier_delta(),
            active_workload_sessions_at_seal: ACTIVE_WORKLOAD_SESSIONS.load(Ordering::SeqCst),
            overlap_observed: self.overlapped_on_enter
                || OVERLAP_EPOCH.load(Ordering::SeqCst) != self.overlap_epoch_at_enter,
        };
        ACTIVE_WORKLOAD_SESSIONS.fetch_sub(1, Ordering::SeqCst);
        self.sealed.store(true, Ordering::Release);
        seal
    }

    fn compat_classifier_delta(&self) -> CompatClassifierProbeSnapshot {
        CompatClassifierProbeSnapshot::now().delta_since(self.compat_classifier_before)
    }
}

impl Drop for ProbeSessionGuard {
    fn drop(&mut self) {
        if self.sealed.load(Ordering::Acquire) {
            return;
        }
        let _transition = probe_session_transition_lock();
        if !self.sealed.load(Ordering::Relaxed) {
            ACTIVE_WORKLOAD_SESSIONS.fetch_sub(1, Ordering::SeqCst);
            self.sealed.store(true, Ordering::Release);
        }
    }
}

fn probe_session_transition_lock() -> std::sync::MutexGuard<'static, ()> {
    PROBE_SESSION_TRANSITION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Snapshot raw Query-message attribution before dispatch. The protocol splitter owns neither the
/// semicolon nor trailing newline, so counting must remain at this boundary.
pub(super) fn raw_simple_query_before(engine: &SharedEngine) -> InsertProbeSnapshot {
    engine.insert_probe_snapshot()
}

/// Attribute raw SQL bytes only when exactly one INSERT became successful while the message ran.
/// The qualification stream intentionally sends one INSERT per Query message, so its raw payload
/// has an unambiguous statement-level owner.
pub(super) fn record_raw_simple_query_after(
    engine: &SharedEngine,
    before: InsertProbeSnapshot,
    raw_sql_bytes: u64,
) {
    let after = engine.insert_probe_snapshot();
    if after
        .successful_insert_statements
        .saturating_sub(before.successful_insert_statements)
        == 1
    {
        engine.record_insert_probe_successful_raw_simple_query_bytes(raw_sql_bytes);
    }
}

/// Emit exactly one machine-readable session delta after the facade session has been closed.
///
/// Engine counters are engine-local. Compatibility-classifier counters are process-global server
/// counters snapshotted by the same single-active-session guard, so qualification still accepts
/// the record only for the runner's one active workload session.
pub(super) fn emit_session_delta(
    engine: &SharedEngine,
    before: InsertProbeSnapshot,
    guard: &ProbeSessionGuard,
) {
    let delta = engine.insert_probe_snapshot().delta_since(before);
    if delta.successful_insert_statements == 0 {
        return;
    }
    let seal = guard.seal();
    eprintln!(
        "{}",
        format_session_delta(
            delta,
            engine.insert_probe_config(),
            seal.compat_classifier_delta,
            seal.active_workload_sessions_at_seal,
            seal.overlap_observed,
        )
    );
}

fn format_session_delta(
    delta: InsertProbeSnapshot,
    config: InsertProbeConfig,
    compat: CompatClassifierProbeSnapshot,
    active_workload_sessions_at_seal: u64,
    overlap_observed: bool,
) -> String {
    format!(
        concat!(
            "insert_probe_session_delta=complete version=7 ",
            "attribution=engine_local_plus_process_global_classifier_counters_single_active_workload_session_required ",
            "successful_insert_statements={} successful_insert_rows={} ",
            "fixed_insert_typed_commits={} fixed_insert_legacy_fallbacks={} ",
            "fixed_insert_retryable_declines={} ",
            "fixed_insert_legacy_commit_validation_reresolves={} ",
            "legacy_insert_delta_builds={} predicted_row_keys_materialized={} ",
            "direct_fixed_insert_carriers={} ",
            "raw_request_digest_derivations={} raw_request_digest_derivation_bytes={} ",
            "compat_classifier_gate_checks={} compat_classifier_gate_rejections={} ",
            "compat_classifier_slow_path_admissions={} compat_classifier_slow_path_bytes={} ",
            "successful_insert_source_bytes={} successful_insert_end_to_end_service_nanos={} ",
            "facade_parse_bind_nanos={} ",
            "engine_authorization_catalog_admission_nanos={} ",
            "offlock_coercion_default_constraint_prepare_nanos={} ",
            "commit_validation_reresolve_nanos={} canonical_wal_encode_append_claim_nanos={} ",
            "durability_wait_nanos={} durability_begin_group_flush_nanos={} ",
            "durability_job_wait_nanos={} device_validate_nanos={} ",
            "fua_logical_groups={} fua_logical_payload_bytes={} ",
            "fua_single_frame_padded_baseline_bytes={} ",
            "fua_publish_turn_wait_nanos={} fua_publish_turn_wait_groups={} ",
            "fua_published_frames={} fua_fenced_frames={} fua_fence_failures={} ",
            "fua_payload_bytes={} fua_padded_bytes={} ",
            "fua_stage_copy_nanos={} fua_stage_copy_frames={} ",
            "fua_publish_to_claim_nanos={} fua_publish_to_claim_frames={} ",
            "fua_claim_to_write_done_nanos={} fua_claim_to_write_done_frames={} ",
            "fua_write_done_to_contiguous_cut_nanos={} ",
            "fua_write_done_to_contiguous_cut_frames={} ",
            "fua_contiguous_cut_events={} fua_contiguous_cut_advanced_frames={} ",
            "fua_contiguous_cut_advance_max_frames={} ",
            "fua_waiter_cut_to_observe_nanos={} fua_waiter_cut_to_observe_count={} ",
            "fua_in_flight_depth_max={} fua_in_flight_depth_1={} ",
            "fua_in_flight_depth_2={} fua_in_flight_depth_3_to_4={} ",
            "fua_in_flight_depth_5_to_8={} fua_in_flight_depth_9_to_16={} ",
            "fua_in_flight_depth_17_to_32={} fua_in_flight_depth_33_plus={} ",
            "fua_controller_sustained_actions={} fua_controller_pending_probe_cover_actions={} ",
            "fua_controller_qd1_samples={} fua_controller_qd1_sparse_actions={} ",
            "fua_controller_qd1_verify_actions={} fua_controller_qd1_fast_actions={} ",
            "fua_controller_unfragmented_actions={} fua_controller_pool_too_narrow={} ",
            "fua_controller_empty_chunk={} fua_controller_insufficient_free_slots={} ",
            "fua_controller_natural_depth={} fua_controller_segment_boundary={} ",
            "fua_controller_amplification_cap={} fua_controller_fast_samples={} ",
            "fua_controller_nonfast_samples={} fua_controller_transitions_to_verify={} ",
            "fua_controller_transitions_to_fast={} fua_controller_transitions_to_sustained={} ",
            "fua_controller_stale_qd1_samples={} fua_controller_unavailable_qd1_samples={} ",
            "fua_controller_abandoned_qd1_samples={} fua_controller_protocol_faults={} ",
            "fua_controller_protocol_fallback_actions={} fua_controller_phase={} ",
            "fua_controller_verify_fast_streak={} fua_controller_sustained_remaining={} ",
            "fua_controller_generation={} fua_controller_pending_qd1_samples={} ",
            "fua_controller_fast_in_flight={} fua_controller_fast_in_flight_max={} ",
            "fua_controller_generation_exhausted={} fua_controller_ordinal_exhausted={} ",
            "fua_controller_action_reconciliation={} fua_controller_sample_reconciliation={} ",
            "device_h2d_append_index_apply_nanos={} publication_status_ack_nanos={} ",
            "wave_count={} wave_item_count={} peak_host_statement_bytes={} ",
            "peak_device_statement_bytes_estimate={} ",
            "rollover_count={} rollover_capacity_rows_total={} rollover_capacity_rows_max={} ",
            "current_shard_count={} peak_shard_count={} persistent_allocation_count={} ",
            "descriptor_clone_visit_count={} budget_scan_entries={} ",
            "capacity_fit_evaluation_count={} sidecar_fill_bytes={} ",
            "live_h2d_bytes={} named_index_shard_visits={} ",
            "unattributed_or_concurrent_overlap_nanos={} durability_backend={} fua_fence_lanes={} ",
            "intent_lane_count={} synchronous_commit_gate={} auto_admit_on_commit={} ",
            "binary_wal_records_enabled={} ",
            "device_authoritative_commits={} active_workload_sessions_at_seal={} overlap_observed={}"
        ),
        delta.successful_insert_statements,
        delta.successful_insert_rows,
        delta.fixed_insert_typed_commits,
        delta.fixed_insert_legacy_fallbacks,
        delta.fixed_insert_retryable_declines,
        delta.fixed_insert_legacy_commit_validation_reresolves,
        delta.legacy_insert_delta_builds,
        delta.predicted_row_keys_materialized,
        delta.direct_fixed_insert_carriers,
        delta.raw_request_digest_derivations,
        delta.raw_request_digest_derivation_bytes,
        compat.checks,
        compat.rejections,
        compat.slow_path_admissions,
        compat.slow_path_bytes,
        delta.successful_insert_source_bytes,
        delta.successful_insert_end_to_end_service_nanos,
        delta.facade_parse_bind_nanos,
        delta.engine_authorization_catalog_admission_nanos,
        delta.offlock_coercion_default_constraint_prepare_nanos,
        delta.commit_validation_reresolve_nanos,
        delta.canonical_wal_encode_append_claim_nanos,
        delta.durability_wait_nanos,
        delta.durability_begin_group_flush_nanos,
        delta.durability_job_wait_nanos,
        delta.device_validate_nanos,
        delta.fua_logical_groups,
        delta.fua_logical_payload_bytes,
        delta.fua_single_frame_padded_baseline_bytes,
        delta.fua_publish_turn_wait_nanos,
        delta.fua_publish_turn_wait_groups,
        delta.fua_published_frames,
        delta.fua_fenced_frames,
        delta.fua_fence_failures,
        delta.fua_payload_bytes,
        delta.fua_padded_bytes,
        delta.fua_stage_copy_nanos,
        delta.fua_stage_copy_frames,
        delta.fua_publish_to_claim_nanos,
        delta.fua_publish_to_claim_frames,
        delta.fua_claim_to_write_done_nanos,
        delta.fua_claim_to_write_done_frames,
        delta.fua_write_done_to_contiguous_cut_nanos,
        delta.fua_write_done_to_contiguous_cut_frames,
        delta.fua_contiguous_cut_events,
        delta.fua_contiguous_cut_advanced_frames,
        delta.fua_contiguous_cut_advance_max_frames,
        delta.fua_waiter_cut_to_observe_nanos,
        delta.fua_waiter_cut_to_observe_count,
        delta.fua_in_flight_depth_max,
        delta.fua_in_flight_depth_1,
        delta.fua_in_flight_depth_2,
        delta.fua_in_flight_depth_3_to_4,
        delta.fua_in_flight_depth_5_to_8,
        delta.fua_in_flight_depth_9_to_16,
        delta.fua_in_flight_depth_17_to_32,
        delta.fua_in_flight_depth_33_plus,
        delta.fua_controller_sustained_actions,
        delta.fua_controller_pending_probe_cover_actions,
        delta.fua_controller_qd1_samples,
        delta.fua_controller_qd1_sparse_actions,
        delta.fua_controller_qd1_verify_actions,
        delta.fua_controller_qd1_fast_actions,
        delta.fua_controller_unfragmented_actions,
        delta.fua_controller_pool_too_narrow,
        delta.fua_controller_empty_chunk,
        delta.fua_controller_insufficient_free_slots,
        delta.fua_controller_natural_depth,
        delta.fua_controller_segment_boundary,
        delta.fua_controller_amplification_cap,
        delta.fua_controller_fast_samples,
        delta.fua_controller_nonfast_samples,
        delta.fua_controller_transitions_to_verify,
        delta.fua_controller_transitions_to_fast,
        delta.fua_controller_transitions_to_sustained,
        delta.fua_controller_stale_qd1_samples,
        delta.fua_controller_unavailable_qd1_samples,
        delta.fua_controller_abandoned_qd1_samples,
        delta.fua_controller_protocol_faults,
        delta.fua_controller_protocol_fallback_actions,
        delta.fua_controller_phase,
        delta.fua_controller_verify_fast_streak,
        delta.fua_controller_sustained_remaining,
        delta.fua_controller_generation,
        delta.fua_controller_pending_qd1_samples,
        delta.fua_controller_fast_in_flight,
        delta.fua_controller_fast_in_flight_max,
        delta.fua_controller_generation_exhausted,
        delta.fua_controller_ordinal_exhausted,
        delta.fua_controller_action_reconciliation,
        delta.fua_controller_sample_reconciliation,
        delta.device_h2d_append_index_apply_nanos,
        delta.publication_status_ack_nanos,
        delta.wave_count,
        delta.wave_item_count,
        delta.peak_host_statement_bytes,
        delta.peak_device_statement_bytes_estimate,
        delta.rollover_count,
        delta.rollover_capacity_rows_total,
        delta.rollover_capacity_rows_max,
        delta.current_shard_count,
        delta.peak_shard_count,
        delta.persistent_allocation_count,
        delta.descriptor_clone_visit_count,
        delta.budget_scan_entries,
        delta.capacity_fit_evaluation_count,
        delta.sidecar_fill_bytes,
        delta.live_h2d_bytes,
        delta.named_index_shard_visits,
        delta.unattributed_or_concurrent_overlap_nanos,
        config.durability_backend,
        config.fua_fence_lanes,
        config.intent_lane_count,
        config.synchronous_commit_gate,
        config.auto_admit_on_commit,
        config.binary_wal_records_enabled,
        config.device_authoritative_commits,
        active_workload_sessions_at_seal,
        overlap_observed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    macro_rules! assign_sentinels {
        ($snapshot:ident, $expected:ident, $next:ident; $($field:ident),+ $(,)?) => {
            $(
                $snapshot.$field = $next;
                $expected.push((stringify!($field).to_string(), $next.to_string()));
                $next += 1;
            )+
        };
    }

    #[test]
    fn format_is_one_flat_machine_record_with_resolved_configuration() {
        let line = format_session_delta(
            InsertProbeSnapshot {
                successful_insert_statements: 1,
                successful_insert_rows: 1_000,
                ..Default::default()
            },
            InsertProbeConfig {
                durability_backend: "serial",
                fua_fence_lanes: 0,
                intent_lane_count: 10,
                synchronous_commit_gate: "strict_rpo0",
                auto_admit_on_commit: true,
                binary_wal_records_enabled: true,
                device_authoritative_commits: 1_000,
            },
            CompatClassifierProbeSnapshot::default(),
            1,
            false,
        );
        assert_eq!(
            line.matches("insert_probe_session_delta=complete").count(),
            1
        );
        assert!(line.contains("durability_backend=serial"));
        assert!(line.contains("intent_lane_count=10"));
        assert!(line.contains("version=7"));
        assert!(line.contains("binary_wal_records_enabled=true"));
        assert!(line.contains("fixed_insert_typed_commits=0"));
        assert!(line.contains("direct_fixed_insert_carriers=0"));
        assert!(line.contains("raw_request_digest_derivations=0"));
        assert!(line.contains("raw_request_digest_derivation_bytes=0"));
        assert!(line.contains("compat_classifier_gate_checks=0"));
        assert!(line.contains("rollover_count=0"));
        assert!(line.contains("named_index_shard_visits=0"));
        assert!(line.contains("fua_published_frames=0"));
        assert!(line.contains("fua_controller_action_reconciliation=0"));
        assert!(line.contains("fua_fence_lanes=0"));
        assert!(!line.contains('\n'));
    }

    #[test]
    fn format_preserves_every_sentinel_field_in_machine_record_order() {
        let mut expected = vec![
            (
                "insert_probe_session_delta".to_string(),
                "complete".to_string(),
            ),
            ("version".to_string(), "7".to_string()),
            (
                "attribution".to_string(),
                "engine_local_plus_process_global_classifier_counters_single_active_workload_session_required"
                    .to_string(),
            ),
        ];
        let mut next = 1u64;
        let mut delta = InsertProbeSnapshot::default();
        assign_sentinels!(
            delta, expected, next;
            successful_insert_statements,
            successful_insert_rows,
            fixed_insert_typed_commits,
            fixed_insert_legacy_fallbacks,
            fixed_insert_retryable_declines,
            fixed_insert_legacy_commit_validation_reresolves,
            legacy_insert_delta_builds,
            predicted_row_keys_materialized,
            direct_fixed_insert_carriers,
            raw_request_digest_derivations,
            raw_request_digest_derivation_bytes,
        );

        let checks = next;
        expected.push((
            "compat_classifier_gate_checks".to_string(),
            next.to_string(),
        ));
        next += 1;
        let rejections = next;
        expected.push((
            "compat_classifier_gate_rejections".to_string(),
            next.to_string(),
        ));
        next += 1;
        let slow_path_admissions = next;
        expected.push((
            "compat_classifier_slow_path_admissions".to_string(),
            next.to_string(),
        ));
        next += 1;
        let slow_path_bytes = next;
        expected.push((
            "compat_classifier_slow_path_bytes".to_string(),
            next.to_string(),
        ));
        next += 1;
        let compat = CompatClassifierProbeSnapshot {
            checks,
            rejections,
            slow_path_admissions,
            slow_path_bytes,
        };

        assign_sentinels!(
            delta, expected, next;
            successful_insert_source_bytes,
            successful_insert_end_to_end_service_nanos,
            facade_parse_bind_nanos,
            engine_authorization_catalog_admission_nanos,
            offlock_coercion_default_constraint_prepare_nanos,
            commit_validation_reresolve_nanos,
            canonical_wal_encode_append_claim_nanos,
            durability_wait_nanos,
            durability_begin_group_flush_nanos,
            durability_job_wait_nanos,
            device_validate_nanos,
            fua_logical_groups,
            fua_logical_payload_bytes,
            fua_single_frame_padded_baseline_bytes,
            fua_publish_turn_wait_nanos,
            fua_publish_turn_wait_groups,
            fua_published_frames,
            fua_fenced_frames,
            fua_fence_failures,
            fua_payload_bytes,
            fua_padded_bytes,
            fua_stage_copy_nanos,
            fua_stage_copy_frames,
            fua_publish_to_claim_nanos,
            fua_publish_to_claim_frames,
            fua_claim_to_write_done_nanos,
            fua_claim_to_write_done_frames,
            fua_write_done_to_contiguous_cut_nanos,
            fua_write_done_to_contiguous_cut_frames,
            fua_contiguous_cut_events,
            fua_contiguous_cut_advanced_frames,
            fua_contiguous_cut_advance_max_frames,
            fua_waiter_cut_to_observe_nanos,
            fua_waiter_cut_to_observe_count,
            fua_in_flight_depth_max,
            fua_in_flight_depth_1,
            fua_in_flight_depth_2,
            fua_in_flight_depth_3_to_4,
            fua_in_flight_depth_5_to_8,
            fua_in_flight_depth_9_to_16,
            fua_in_flight_depth_17_to_32,
            fua_in_flight_depth_33_plus,
            fua_controller_sustained_actions,
            fua_controller_pending_probe_cover_actions,
            fua_controller_qd1_samples,
            fua_controller_qd1_sparse_actions,
            fua_controller_qd1_verify_actions,
            fua_controller_qd1_fast_actions,
            fua_controller_unfragmented_actions,
            fua_controller_pool_too_narrow,
            fua_controller_empty_chunk,
            fua_controller_insufficient_free_slots,
            fua_controller_natural_depth,
            fua_controller_segment_boundary,
            fua_controller_amplification_cap,
            fua_controller_fast_samples,
            fua_controller_nonfast_samples,
            fua_controller_transitions_to_verify,
            fua_controller_transitions_to_fast,
            fua_controller_transitions_to_sustained,
            fua_controller_stale_qd1_samples,
            fua_controller_unavailable_qd1_samples,
            fua_controller_abandoned_qd1_samples,
            fua_controller_protocol_faults,
            fua_controller_protocol_fallback_actions,
            fua_controller_phase,
            fua_controller_verify_fast_streak,
            fua_controller_sustained_remaining,
            fua_controller_generation,
            fua_controller_pending_qd1_samples,
            fua_controller_fast_in_flight,
            fua_controller_fast_in_flight_max,
            fua_controller_generation_exhausted,
            fua_controller_ordinal_exhausted,
            fua_controller_action_reconciliation,
            fua_controller_sample_reconciliation,
            device_h2d_append_index_apply_nanos,
            publication_status_ack_nanos,
            wave_count,
            wave_item_count,
            peak_host_statement_bytes,
            peak_device_statement_bytes_estimate,
            rollover_count,
            rollover_capacity_rows_total,
            rollover_capacity_rows_max,
            current_shard_count,
            peak_shard_count,
            persistent_allocation_count,
            descriptor_clone_visit_count,
            budget_scan_entries,
            capacity_fit_evaluation_count,
            sidecar_fill_bytes,
            live_h2d_bytes,
            named_index_shard_visits,
            unattributed_or_concurrent_overlap_nanos,
        );

        let config = InsertProbeConfig {
            durability_backend: "sentinel-backend",
            fua_fence_lanes: next,
            intent_lane_count: next + 1,
            synchronous_commit_gate: "sentinel-gate",
            auto_admit_on_commit: true,
            binary_wal_records_enabled: false,
            device_authoritative_commits: next + 2,
        };
        expected.extend([
            (
                "durability_backend".to_string(),
                config.durability_backend.to_string(),
            ),
            (
                "fua_fence_lanes".to_string(),
                config.fua_fence_lanes.to_string(),
            ),
            (
                "intent_lane_count".to_string(),
                config.intent_lane_count.to_string(),
            ),
            (
                "synchronous_commit_gate".to_string(),
                config.synchronous_commit_gate.to_string(),
            ),
            (
                "auto_admit_on_commit".to_string(),
                config.auto_admit_on_commit.to_string(),
            ),
            (
                "binary_wal_records_enabled".to_string(),
                config.binary_wal_records_enabled.to_string(),
            ),
            (
                "device_authoritative_commits".to_string(),
                config.device_authoritative_commits.to_string(),
            ),
            (
                "active_workload_sessions_at_seal".to_string(),
                "10001".to_string(),
            ),
            ("overlap_observed".to_string(), "true".to_string()),
        ]);

        let actual = format_session_delta(delta, config, compat, 10_001, true)
            .split_ascii_whitespace()
            .map(|field| {
                let (key, value) = field
                    .split_once('=')
                    .expect("every probe record token must be key=value");
                (key.to_string(), value.to_string())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn raw_simple_query_bytes_preserve_statement_terminator_and_newline() {
        let query = "INSERT INTO accounts (id, balance) VALUES (1, 7);\n";
        assert_eq!(query.len(), 50);
        assert_eq!(query.as_bytes()[query.len() - 2..], *b";\n");
    }

    #[test]
    fn short_overlapping_classifier_session_is_observed_before_first_guard_seals() {
        let first_published = Arc::new(Barrier::new(2));
        let release_first = Arc::new(Barrier::new(2));
        let worker_published = Arc::clone(&first_published);
        let worker_release = Arc::clone(&release_first);
        let first = std::thread::spawn(move || {
            ProbeSessionGuard::enter_after_active_publication(|| {
                worker_published.wait();
                worker_release.wait();
            })
        });

        first_published.wait();
        {
            let overlapping = ProbeSessionGuard::enter();
            record_compat_classifier_gate(17, false);
            drop(overlapping);
        }
        release_first.wait();

        let first = first.join().expect("first guard thread must not panic");
        let seal = first.seal();
        assert!(
            seal.overlap_observed,
            "a completed overlapping session must fail closed"
        );
        assert!(
            seal.compat_classifier_delta.checks >= 1
                && seal.compat_classifier_delta.rejections >= 1,
            "the overlapping session's process-global classifier write must be in the first delta"
        );
    }
}
