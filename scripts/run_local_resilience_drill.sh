#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-local-resilience-drill.XXXXXX")"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

run_gate() {
  local name="$1"
  local script="$2"
  shift 2
  local output_file="$WORKDIR/${name}.out"

  printf 'local_resilience_step=%s status=running\n' "$name"
  "$script" >"$output_file"
  cat "$output_file"

  local required
  for required in "$@"; do
    if ! grep -Fq "$required" "$output_file"; then
      printf 'local resilience gate %s missing evidence line: %s\n' "$name" "$required" >&2
      return 1
    fi
  done
  printf 'local_resilience_step=%s status=passed\n' "$name"
}

run_gate \
  backup_pitr_dr \
  scripts/run_backup_pitr_dr_drill.sh \
  "backup_pitr_dr_drill=passed" \
  "backup_pitr_dr_restore_targets=transaction,timestamp" \
  "backup_pitr_dr_local_maintenance=archive_retention_plus_timeline_prune" \
  "backup_pitr_dr_object_bundle=file_backed_export_restore_recover_corrupt_reject" \
  "backup_pitr_dr_gap_production_object_storage=missing" \
  "backup_pitr_dr_gap_live_background_scheduling=missing"

run_gate \
  replication_deployment \
  scripts/run_replication_deployment_preflight.sh \
  "operational_replication_deployment_preflight=passed" \
  "deployment_preflight_scope=packaged_service_systemd_contract_kubernetes_manifest_compose_restart" \
  "deployment_preflight_service_smoke=passed" \
  "deployment_preflight_systemd_verify=passed" \
  "deployment_preflight_kubernetes_verify=passed" \
  "deployment_preflight_compose_restart_smoke=passed" \
  "deployment_gap_live_systemd_supervision=missing" \
  "deployment_gap_live_kubernetes_rollout=missing"

printf 'local_resilience_drill=passed\n'
printf 'local_resilience_backup_pitr_dr=passed\n'
printf 'local_resilience_replication_deployment=passed\n'
printf 'local_resilience_scope=backup_pitr_dr_plus_replication_deployment_preflight\n'
printf 'local_resilience_restore_targets=transaction,timestamp\n'
printf 'local_resilience_replication_scope=packaged_service_systemd_contract_kubernetes_manifest_compose_restart\n'
printf 'local_resilience_gap_physical_page_image_backup=missing\n'
printf 'local_resilience_gap_production_object_storage=missing\n'
printf 'local_resilience_gap_live_background_scheduling=missing\n'
printf 'local_resilience_gap_live_systemd_supervision=missing\n'
printf 'local_resilience_gap_live_kubernetes_rollout=missing\n'
printf 'local_resilience_gap_production_timeline_failover=missing\n'
