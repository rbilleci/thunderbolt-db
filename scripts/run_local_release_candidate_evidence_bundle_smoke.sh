#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-release-evidence-smoke.XXXXXX")"
DIRTY_MARKER=".release-candidate-evidence-smoke-dirty-marker"

cleanup() {
  rm -rf "$WORKDIR"
  rm -f "$DIRTY_MARKER"
}
trap cleanup EXIT

FAKE_PREFLIGHT="$WORKDIR/fake-release-candidate-preflight.sh"
OUT_ROOT="$WORKDIR/evidence"
STAMP="20260520T000000Z"
OUTPUT="$WORKDIR/evidence-bundle.out"

cat >"$FAKE_PREFLIGHT" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf 'local_release_candidate_preflight=passed\n'
printf 'local_release_candidate_preflight_scope=validation_postgresql_product_gpu_residency_plus_connection_security\n'
printf 'local_release_candidate_preflight_validation=fmt_clippy_all_features_psql_golden_scorecard_freshness\n'
printf 'local_release_candidate_preflight_postgresql=application_drivers_pg_dump_restore_pg_dumpall_globals_privileges_local_resilience\n'
printf 'local_release_candidate_preflight_privileges=schema_usage_create_relation_sequence_function_execute_default_table_acls\n'
printf 'local_release_candidate_preflight_gpu=residency_baseline_warmup_maintenance\n'
printf 'local_release_candidate_preflight_gpu_residency=retained_cuda_allocation_zero_h2d_routes_event_timing_warmup_maintenance\n'
printf 'local_release_candidate_preflight_connection_security=local_dev_trust_auth_no_tls_boundary\n'
printf 'local_release_candidate_preflight_gap_scram_sha_256=missing\n'
printf 'local_release_candidate_preflight_gap_password_authentication_storage=missing\n'
printf 'local_release_candidate_preflight_gap_tls_client_connections=missing\n'
printf 'local_release_candidate_preflight_gap_replication_mtls=missing\n'
printf 'local_release_candidate_preflight_gap_certificate_lifecycle=missing\n'
printf 'local_release_candidate_preflight_gap_audit_hash_chain=missing\n'
printf 'local_release_candidate_preflight_gap_row_level_security=missing\n'
printf 'local_release_candidate_preflight_gap_masking=missing\n'
printf 'local_release_candidate_preflight_gap_jdbc_r2dbc=not_configured\n'
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
FAKE
chmod +x "$FAKE_PREFLIGHT"

printf 'evidence bundle smoke dirty marker\n' >"$DIRTY_MARKER"

LOCAL_RELEASE_CANDIDATE_PREFLIGHT_CMD="$FAKE_PREFLIGHT" \
LOCAL_RELEASE_CANDIDATE_EVIDENCE_ROOT="$OUT_ROOT" \
LOCAL_RELEASE_CANDIDATE_EVIDENCE_STAMP="$STAMP" \
LOCAL_RELEASE_CANDIDATE_EVIDENCE_ALLOW_DIRTY=1 \
  scripts/run_local_release_candidate_evidence_bundle.sh >"$OUTPUT"

require_line() {
  local file="$1"
  local expected="$2"
  if ! grep -Fq "$expected" "$file"; then
    printf 'release evidence bundle smoke missing line in %s: %s\n' "$file" "$expected" >&2
    return 1
  fi
}

require_line "$OUTPUT" "local_release_candidate_evidence_bundle=passed"
require_line "$OUTPUT" "local_release_candidate_evidence_git_dirty=1"

EVIDENCE_DIR="$(awk -F= '/^local_release_candidate_evidence_dir=/{print $2}' "$OUTPUT")"
MANIFEST="$(awk -F= '/^local_release_candidate_evidence_manifest=/{print $2}' "$OUTPUT")"
PREFLIGHT_LOG="$(awk -F= '/^local_release_candidate_evidence_preflight_log=/{print $2}' "$OUTPUT")"
GAPS="$(awk -F= '/^local_release_candidate_evidence_gaps=/{print $2}' "$OUTPUT")"
TARBALL="$(awk -F= '/^local_release_candidate_evidence_tarball=/{print $2}' "$OUTPUT")"
CHECKSUM="$(awk -F= '/^local_release_candidate_evidence_sha256=/{print $2}' "$OUTPUT")"

test -d "$EVIDENCE_DIR"
test -f "$MANIFEST"
test -f "$PREFLIGHT_LOG"
test -f "$GAPS"
test -f "$TARBALL"
test -f "${TARBALL}.sha256"

require_line "$MANIFEST" "evidence_stamp=$STAMP"
require_line "$MANIFEST" "git_dirty=1"
require_line "$MANIFEST" "preflight_command=$FAKE_PREFLIGHT"
require_line "$PREFLIGHT_LOG" "local_release_candidate_preflight=passed"
require_line "$GAPS" "local_release_candidate_preflight_gap_jdbc_r2dbc=not_configured"
require_line "$GAPS" "local_release_candidate_preflight_gap_broad_cuda_event_timing=missing"

if [[ "$(wc -l <"$GAPS" | tr -d ' ')" -ne 22 ]]; then
  printf 'release evidence bundle smoke expected 22 remaining-gap lines\n' >&2
  exit 1
fi

if [[ "$(sha256sum "$TARBALL" | awk '{print $1}')" != "$CHECKSUM" ]]; then
  printf 'release evidence bundle smoke checksum output did not match tarball\n' >&2
  exit 1
fi

(cd "$(dirname "$TARBALL")" && sha256sum -c "$(basename "${TARBALL}.sha256")" >/dev/null)
tar -tzf "$TARBALL" | grep -Fxq './manifest.env'
tar -tzf "$TARBALL" | grep -Fxq './local-release-candidate-preflight.log'
tar -tzf "$TARBALL" | grep -Fxq './remaining-gaps.env'

printf 'local_release_candidate_evidence_bundle_smoke=passed\n'
printf 'local_release_candidate_evidence_bundle_smoke_scope=fake_preflight_manifest_log_gaps_tarball_checksum\n'
printf 'local_release_candidate_evidence_bundle_smoke_dirty_override=verified\n'
