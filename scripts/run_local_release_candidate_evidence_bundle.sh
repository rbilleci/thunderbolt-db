#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PREFLIGHT_CMD="${LOCAL_RELEASE_CANDIDATE_PREFLIGHT_CMD:-scripts/run_local_release_candidate_preflight.sh}"
ALLOW_DIRTY="${LOCAL_RELEASE_CANDIDATE_EVIDENCE_ALLOW_DIRTY:-0}"
STAMP="${LOCAL_RELEASE_CANDIDATE_EVIDENCE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${LOCAL_RELEASE_CANDIDATE_EVIDENCE_ROOT:-target/release-candidate-evidence}"
OUT_DIR="${LOCAL_RELEASE_CANDIDATE_EVIDENCE_DIR:-${OUT_ROOT}/${STAMP}}"

required_lines=(
  "local_release_candidate_preflight=passed"
  "local_release_candidate_preflight_scope=validation_postgresql_product_plus_gpu_residency"
  "local_release_candidate_preflight_postgresql=application_drivers_pg_dump_restore_pg_dumpall_globals_privileges_local_resilience"
  "local_release_candidate_preflight_privileges=schema_usage_create_relation_sequence_function_execute_default_table_acls"
  "local_release_candidate_preflight_gpu_residency=retained_cuda_allocation_zero_h2d_routes_event_timing_warmup_maintenance"
  "local_release_candidate_preflight_gap_pgx=blocked_missing_go"
  "local_release_candidate_preflight_gap_jdbc_r2dbc=blocked_missing_java_build_tooling"
  "local_release_candidate_preflight_gap_pg_dumpall_bootstrap_role_restore=filtered_existing_bootstrap_role"
  "local_release_candidate_preflight_gap_pg_dumpall_shared_object_acl_restore=missing"
  "local_release_candidate_preflight_gap_physical_page_image_backup=missing"
  "local_release_candidate_preflight_gap_production_object_storage=missing"
  "local_release_candidate_preflight_gap_live_background_scheduling=missing"
  "local_release_candidate_preflight_gap_live_systemd_supervision=missing"
  "local_release_candidate_preflight_gap_live_kubernetes_rollout=missing"
  "local_release_candidate_preflight_gap_production_timeline_failover=missing"
  "local_release_candidate_preflight_gap_durable_gpu_pages=missing"
  "local_release_candidate_preflight_gap_autonomous_cache_daemon=missing"
  "local_release_candidate_preflight_gap_external_orchestration=missing"
  "local_release_candidate_preflight_gap_broad_retained_expressions=missing"
  "local_release_candidate_preflight_gap_broad_cuda_event_timing=missing"
)

command_value() {
  local label="$1"
  shift
  local tmp_out="$OUT_DIR/${label}.out"
  local tmp_err="$OUT_DIR/${label}.err"
  if "$@" >"$tmp_out" 2>"$tmp_err"; then
    printf '%s=%s\n' "$label" "$(tr '\n' ' ' <"$tmp_out" | sed 's/[[:space:]]*$//')"
  else
    printf '%s=missing\n' "$label"
  fi
  rm -f "$tmp_out" "$tmp_err"
}

command_path() {
  local label="$1"
  local name="$2"
  if command -v "$name" >/dev/null 2>&1; then
    printf '%s=%s\n' "$label" "$(command -v "$name")"
  else
    printf '%s=missing\n' "$label"
  fi
}

mkdir -p "$OUT_DIR"

HEAD="$(git rev-parse HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
DIRTY=0
if [[ -n "$(git status --porcelain)" ]]; then
  DIRTY=1
fi

if [[ "$DIRTY" -ne 0 && "$ALLOW_DIRTY" != "1" ]]; then
  echo "local release-candidate evidence requires a clean worktree; set LOCAL_RELEASE_CANDIDATE_EVIDENCE_ALLOW_DIRTY=1 for wrapper development smoke tests" >&2
  exit 1
fi

PRELIGHT_LOG="$OUT_DIR/local-release-candidate-preflight.log"
MANIFEST="$OUT_DIR/manifest.env"
GAPS="$OUT_DIR/remaining-gaps.env"

{
  printf 'evidence_stamp=%s\n' "$STAMP"
  printf 'git_head=%s\n' "$HEAD"
  printf 'git_branch=%s\n' "$BRANCH"
  printf 'git_upstream=%s\n' "${UPSTREAM:-none}"
  printf 'git_dirty=%s\n' "$DIRTY"
  command_value cargo_version cargo --version
  command_value rustc_version rustc --version
  command_value psql_version psql --version
  command_path psql_path psql
  command_value python3_version python3 --version
  command_value node_version node --version
  command_value npm_version npm --version
  command_value nvidia_smi_version nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  printf 'preflight_command=%s\n' "$PREFLIGHT_CMD"
} >"$MANIFEST"

printf 'local_release_candidate_evidence_step=preflight status=running\n'
bash -lc "$PREFLIGHT_CMD" 2>&1 | tee "$PRELIGHT_LOG"
printf 'local_release_candidate_evidence_step=preflight status=passed\n'

for required in "${required_lines[@]}"; do
  if ! grep -Fq "$required" "$PRELIGHT_LOG"; then
    printf 'local release-candidate evidence missing preflight line: %s\n' "$required" >&2
    exit 1
  fi
done

grep '^local_release_candidate_preflight_gap_' "$PRELIGHT_LOG" >"$GAPS"

TARBALL="${OUT_DIR}.tar.gz"
tar -czf "$TARBALL" -C "$OUT_DIR" .
CHECKSUM="$(sha256sum "$TARBALL" | awk '{print $1}')"
printf '%s  %s\n' "$CHECKSUM" "$TARBALL" >"${TARBALL}.sha256"

printf 'local_release_candidate_evidence_bundle=passed\n'
printf 'local_release_candidate_evidence_dir=%s\n' "$OUT_DIR"
printf 'local_release_candidate_evidence_manifest=%s\n' "$MANIFEST"
printf 'local_release_candidate_evidence_preflight_log=%s\n' "$PRELIGHT_LOG"
printf 'local_release_candidate_evidence_gaps=%s\n' "$GAPS"
printf 'local_release_candidate_evidence_tarball=%s\n' "$TARBALL"
printf 'local_release_candidate_evidence_sha256=%s\n' "$CHECKSUM"
printf 'local_release_candidate_evidence_git_head=%s\n' "$HEAD"
printf 'local_release_candidate_evidence_git_dirty=%s\n' "$DIRTY"
