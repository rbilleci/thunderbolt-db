#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-local-gpu-residency.XXXXXX")"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

run_gate() {
  local name="$1"
  local script="$2"
  shift 2
  local output_file="$WORKDIR/${name}.out"

  printf 'local_gpu_residency_preflight_step=%s status=running\n' "$name"
  "$script" >"$output_file"
  cat "$output_file"

  local required
  for required in "$@"; do
    if ! grep -Fq -- "$required" "$output_file"; then
      printf 'local GPU residency preflight gate %s missing evidence line: %s\n' "$name" "$required" >&2
      return 1
    fi
  done
  printf 'local_gpu_residency_preflight_step=%s status=passed\n' "$name"
}

run_gate \
  residency_baseline \
  scripts/run_p7_gpu_residency_baseline.sh \
  "- resident_device_memory_proof_supported: true" \
  "- resident_device_memory_retained: true" \
  "- resident_device_memory_cuda_event_timing_supported: true" \
  "- p8_cache_manager_component_supported: explicit_relational_resident_cache" \
  "- resident_snapshot_valid_before_mutation: true" \
  "- resident_snapshot_valid_after_mutation: false" \
  "- resident_refresh_cost_recorded: true" \
  "- resident_budget_admission_supported: true" \
  "- resident_budget_decision_accepted: true" \
  "- resident_budget_oversize_rejected: true" \
  "- memory_pressure_fallback_supported: true" \
  "- retained_filter_family_closeout: supported retained int4 filter groups and text prefix LIKE count are closed for the current SQL subset" \
  "### warm_resident_snapshot_probe" \
  "### resident_device_memory_count_kernel_probe" \
  "### default_resident_route_count_kernel_probe" \
  "### resident_device_memory_text_prefix_count_probe" \
  "### resident_device_memory_filtered_grouped_count_kernel_probe" \
  "- h2d_bytes: 0" \
  "decision: current P7 evidence includes bounded resident table-data snapshot SELECT probes"

run_gate \
  resident_warmup \
  scripts/run_p8_resident_warmup_preflight_smoke.sh \
  "p8 resident warmup preflight smoke passed"

run_gate \
  resident_maintenance \
  scripts/run_p8_resident_maintenance_smoke.sh \
  "p8 resident maintenance smoke passed"

run_gate \
  gpu_tests \
  scripts/run_local_gpu_tests.sh \
  "local_gpu_tests=passed"

printf 'local_gpu_residency_preflight=passed\n'
printf 'local_gpu_residency_preflight_scope=residency_baseline_warmup_maintenance_gpu_tests\n'
printf 'local_gpu_residency_preflight_resident_device_memory=retained_cuda_allocation\n'
printf 'local_gpu_residency_preflight_resident_routes=zero_h2d_supported_kernel_shapes\n'
printf 'local_gpu_residency_preflight_cache_manager=budget_admission_eviction_invalidation_refresh\n'
printf 'local_gpu_residency_preflight_cuda_event_timing=first_accepted_route_samples\n'
printf 'local_gpu_residency_preflight_warmup=operator_triggered_dry_run_apply\n'
printf 'local_gpu_residency_preflight_maintenance=scheduler_friendly_tick\n'
printf 'local_gpu_residency_preflight_gpu_tests=ignored_gpu_suite_execution_engine\n'
printf 'local_gpu_residency_preflight_gap_durable_gpu_pages=missing\n'
printf 'local_gpu_residency_preflight_gap_autonomous_cache_daemon=missing\n'
printf 'local_gpu_residency_preflight_gap_external_orchestration=missing\n'
printf 'local_gpu_residency_preflight_gap_broad_retained_expressions=missing\n'
printf 'local_gpu_residency_preflight_gap_broad_cuda_event_timing=missing\n'
