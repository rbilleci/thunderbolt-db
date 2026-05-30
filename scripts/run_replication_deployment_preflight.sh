#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-replication-deployment-preflight.XXXXXX")"

cleanup() {
  rm -rf "$workdir"
}
trap cleanup EXIT

run_gate() {
  local name="$1"
  local script="$2"
  shift 2
  local output_file="$workdir/${name}.out"

  "$script" >"$output_file"
  cat "$output_file"

  local required
  for required in "$@"; do
    if ! grep -Fq "$required" "$output_file"; then
      printf 'deployment preflight gate %s missing evidence line: %s\n' "$name" "$required" >&2
      return 1
    fi
  done
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'deployment preflight missing required command: %s\n' "$command_name" >&2
    return 1
  fi
}

require_command docker
docker compose version >/dev/null
docker info >/dev/null

run_gate \
  service \
  scripts/run_replication_service_smoke.sh \
  "operational_replication_service_smoke=passed" \
  "deployment_gap_long_running_service=implemented"

run_gate \
  systemd \
  scripts/run_replication_systemd_verify.sh \
  "operational_replication_systemd_verify=passed" \
  "deployment_gap_production_service_manager=implemented_unit_syntax_and_command_contract" \
  "deployment_gap_live_systemd_supervision=missing"

run_gate \
  kubernetes \
  scripts/run_replication_kubernetes_verify.sh \
  "operational_replication_kubernetes_verify=passed" \
  "deployment_gap_kubernetes_deployment=implemented_manifest_contract" \
  "deployment_gap_live_kubernetes_rollout=missing"

run_gate \
  compose_restart \
  scripts/run_replication_compose_restart_smoke.sh \
  "operational_replication_compose_restart_smoke=host_parent_passed" \
  "deployment_gap_compose_restart_supervision=implemented_bounded_local_smoke"

run_gate \
  channel_security \
  scripts/run_replication_channel_security_preflight.sh \
  "operational_replication_channel_security_smoke=passed" \
  "replication_channel_security_transport=mtls_append_entries" \
  "replication_channel_security_missing_client_cert_rejection=passed" \
  "replication_channel_security_plain_transport_profile=dev_test_only"

printf 'operational_replication_deployment_preflight=passed\n'
printf 'deployment_preflight_scope=packaged_service_systemd_contract_kubernetes_manifest_compose_restart_channel_mtls\n'
printf 'deployment_preflight_service_smoke=passed\n'
printf 'deployment_preflight_systemd_verify=passed\n'
printf 'deployment_preflight_kubernetes_verify=passed\n'
printf 'deployment_preflight_compose_restart_smoke=passed\n'
printf 'deployment_preflight_channel_security=local_mtls_append_entries\n'
printf 'deployment_gap_live_systemd_supervision=missing\n'
printf 'deployment_gap_live_kubernetes_rollout=missing\n'
printf 'deployment_gap_replication_certificate_lifecycle=missing\n'
printf 'deployment_gap_replication_production_trust_distribution=missing\n'
