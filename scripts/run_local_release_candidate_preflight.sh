#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-local-release-candidate.XXXXXX")"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

run_gate() {
  local name="$1"
  local script="$2"
  shift 2
  local output_file="$WORKDIR/${name}.out"

  printf 'local_release_candidate_preflight_step=%s status=running\n' "$name"
  "$script" >"$output_file"
  cat "$output_file"

  local required
  for required in "$@"; do
    if ! grep -Fq "$required" "$output_file"; then
      printf 'local release-candidate preflight gate %s missing evidence line: %s\n' "$name" "$required" >&2
      return 1
    fi
  done
  printf 'local_release_candidate_preflight_step=%s status=passed\n' "$name"
}

run_gate \
  local_validation \
  scripts/run_local_validation_preflight.sh \
  "local_validation_preflight=passed" \
  "local_validation_preflight_scope=fmt_clippy_all_features_psql_golden_scorecard_freshness" \
  "local_validation_preflight_cargo_all_features=passed" \
  "local_validation_preflight_psql_golden=passed" \
  "local_validation_preflight_scorecard_freshness=checked_in"

run_gate \
  postgresql_product \
  scripts/run_local_product_preflight.sh \
  "local_product_preflight=passed" \
  "local_product_preflight_scope=application_drivers_pg_dump_restore_pg_dumpall_globals_local_resilience" \
  "local_product_preflight_drivers=tokio-postgres,sqlx,node-postgres,asyncpg,psycopg,pgx,jdbc,r2dbc" \
  "local_product_preflight_dump_restore=plain_custom_directory_tar_parallel_clean_insert_split_privileges_pg_dumpall_globals" \
  "local_product_preflight_resilience=backup_pitr_dr_plus_replication_deployment" \
  "local_product_preflight_privileges=schema_usage_create_relation_sequence_function_execute_default_table_acls" \
  "local_product_preflight_gap_pg_dumpall_bootstrap_role_restore=filtered_existing_bootstrap_role" \
  "local_product_preflight_gap_pg_dumpall_database_acl_restore=not_emitted_by_globals_only" \
  "local_product_preflight_gap_physical_page_image_backup=missing" \
  "local_product_preflight_gap_production_object_storage=missing" \
  "local_product_preflight_gap_live_background_scheduling=missing" \
  "local_product_preflight_gap_live_systemd_supervision=missing" \
  "local_product_preflight_gap_live_kubernetes_rollout=missing" \
  "local_product_preflight_gap_production_timeline_failover=missing"

run_gate \
  gpu_residency \
  scripts/run_local_gpu_residency_preflight.sh \
  "local_gpu_residency_preflight=passed" \
  "local_gpu_residency_preflight_scope=residency_baseline_warmup_maintenance" \
  "local_gpu_residency_preflight_resident_device_memory=retained_cuda_allocation" \
  "local_gpu_residency_preflight_resident_routes=zero_h2d_supported_kernel_shapes" \
  "local_gpu_residency_preflight_cache_manager=budget_admission_eviction_invalidation_refresh" \
  "local_gpu_residency_preflight_cuda_event_timing=first_accepted_route_samples" \
  "local_gpu_residency_preflight_warmup=operator_triggered_dry_run_apply" \
  "local_gpu_residency_preflight_maintenance=scheduler_friendly_tick" \
  "local_gpu_residency_preflight_gap_durable_gpu_pages=missing" \
  "local_gpu_residency_preflight_gap_autonomous_cache_daemon=missing" \
  "local_gpu_residency_preflight_gap_external_orchestration=missing" \
  "local_gpu_residency_preflight_gap_broad_retained_expressions=missing" \
  "local_gpu_residency_preflight_gap_broad_cuda_event_timing=missing"

run_gate \
  connection_security \
  scripts/run_connection_security_posture_preflight.sh \
  "connection_security_posture_preflight=passed" \
  "connection_security_posture_preflight_scope=local_dev_trust_auth_no_tls_plus_opt_in_production_tls_scram" \
  "connection_security_posture_preflight_local_dev_profile=trust_auth_no_tls_supported" \
  "connection_security_posture_preflight_production_profile_v1=passed" \
  "connection_security_posture_preflight_production_config_validation=passed" \
  "connection_security_posture_preflight_production_tls_required=passed" \
  "connection_security_posture_preflight_production_scram_sha_256_valid_password=passed" \
  "connection_security_posture_preflight_production_scram_sha_256_invalid_password=passed" \
  "connection_security_posture_preflight_production_recovery_after_invalid_password=passed" \
  "connection_security_posture_preflight_non_claim_mtls=not_supported" \
  "connection_security_posture_preflight_non_claim_certificate_rotation=not_supported" \
  "connection_security_posture_preflight_non_claim_audit_hash_chain=not_supported" \
  "connection_security_posture_preflight_non_claim_row_level_security=not_supported" \
  "connection_security_posture_preflight_non_claim_masking=not_supported"

printf 'local_release_candidate_preflight=passed\n'
printf 'local_release_candidate_preflight_scope=validation_postgresql_product_gpu_residency_plus_connection_security\n'
printf 'local_release_candidate_preflight_validation=fmt_clippy_all_features_psql_golden_scorecard_freshness\n'
printf 'local_release_candidate_preflight_postgresql=application_drivers_pg_dump_restore_pg_dumpall_globals_privileges_local_resilience\n'
printf 'local_release_candidate_preflight_privileges=schema_usage_create_relation_sequence_function_execute_default_table_acls\n'
printf 'local_release_candidate_preflight_gpu=residency_baseline_warmup_maintenance\n'
printf 'local_release_candidate_preflight_connection_security=local_dev_trust_auth_no_tls_plus_production_tls_scram_profile_v1\n'
printf 'local_release_candidate_preflight_drivers=tokio-postgres,sqlx,node-postgres,asyncpg,psycopg,pgx,jdbc,r2dbc\n'
printf 'local_release_candidate_preflight_gpu_residency=retained_cuda_allocation_zero_h2d_routes_event_timing_warmup_maintenance\n'
printf 'local_release_candidate_preflight_gap_replication_mtls=missing\n'
printf 'local_release_candidate_preflight_gap_certificate_lifecycle_automation=missing\n'
printf 'local_release_candidate_preflight_gap_enterprise_identity=missing\n'
printf 'local_release_candidate_preflight_gap_kms_hsm_secret_manager=missing\n'
printf 'local_release_candidate_preflight_gap_audit_hash_chain=missing\n'
printf 'local_release_candidate_preflight_gap_row_level_security=missing\n'
printf 'local_release_candidate_preflight_gap_masking=missing\n'
printf 'local_release_candidate_preflight_gap_broad_authorization=missing\n'
printf 'local_release_candidate_preflight_gap_pg_dumpall_bootstrap_role_restore=filtered_existing_bootstrap_role\n'
printf 'local_release_candidate_preflight_gap_pg_dumpall_database_acl_restore=not_emitted_by_globals_only\n'
printf 'local_release_candidate_preflight_gap_physical_page_image_backup=missing\n'
printf 'local_release_candidate_preflight_gap_production_object_storage=missing\n'
printf 'local_release_candidate_preflight_gap_live_background_scheduling=missing\n'
printf 'local_release_candidate_preflight_gap_live_systemd_supervision=missing\n'
printf 'local_release_candidate_preflight_gap_live_kubernetes_rollout=missing\n'
printf 'local_release_candidate_preflight_gap_production_timeline_failover=missing\n'
printf 'local_release_candidate_preflight_gap_durable_gpu_pages=missing\n'
printf 'local_release_candidate_preflight_gap_autonomous_cache_daemon=missing\n'
printf 'local_release_candidate_preflight_gap_external_orchestration=missing\n'
printf 'local_release_candidate_preflight_gap_broad_retained_expressions=missing\n'
printf 'local_release_candidate_preflight_gap_broad_cuda_event_timing=missing\n'
