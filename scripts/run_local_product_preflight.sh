#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-local-product-preflight.XXXXXX")"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

run_gate() {
  local name="$1"
  local script="$2"
  shift 2
  local output_file="$WORKDIR/${name}.out"

  printf 'local_product_preflight_step=%s status=running\n' "$name"
  "$script" >"$output_file"
  cat "$output_file"

  local required
  for required in "$@"; do
    if ! grep -Fq "$required" "$output_file"; then
      printf 'local product preflight gate %s missing evidence line: %s\n' "$name" "$required" >&2
      return 1
    fi
  done
  printf 'local_product_preflight_step=%s status=passed\n' "$name"
}

run_gate \
  application_drivers \
  scripts/run_application_driver_smokes.sh \
  "application_driver_smoke_tokio_postgres=passed" \
  "application_driver_smoke_sqlx=passed" \
  "application_driver_smoke_node_postgres=passed" \
  "application_driver_smoke_asyncpg=passed" \
  "application_driver_smoke_psycopg=passed" \
  "application_driver_smoke_pgx=blocked_missing_go" \
  "application_driver_smoke_jdbc_r2dbc=blocked_missing_java_build_tooling" \
  "application_driver_smoke_scope=supported_sql_protocol_subset"

run_gate \
  pg_dump_restore \
  tests/compat/pg-dump/run.sh \
  "pg_dump_plain_public_schema_restore=passed" \
  "pg_dump_custom_public_schema_pg_restore=passed" \
  "pg_dump_directory_public_schema_pg_restore=passed" \
  "pg_dump_tar_public_schema_pg_restore=passed" \
  "pg_dump_directory_parallel_public_schema_pg_restore=passed" \
  "pg_dump_custom_clean_if_exists_pg_restore=passed" \
  "pg_dump_plain_insert_style_restore=passed" \
  "pg_dump_plain_split_schema_data_restore=passed" \
  "pg_dump_custom_split_schema_data_restore=passed" \
  "pg_dump_directory_split_schema_data_restore=passed" \
  "pg_dump_tar_split_schema_data_restore=passed" \
  "pg_dump_bounded_view_restore=passed" \
  "pg_dump_bounded_materialized_view_restore=passed" \
  "pg_dump_bounded_sequence_restore=passed" \
  "pg_dump_bounded_domain_restore=passed"

run_gate \
  local_resilience \
  scripts/run_local_resilience_drill.sh \
  "local_resilience_drill=passed" \
  "local_resilience_backup_pitr_dr=passed" \
  "local_resilience_replication_deployment=passed" \
  "local_resilience_scope=backup_pitr_dr_plus_replication_deployment_preflight" \
  "local_resilience_gap_physical_page_image_backup=missing" \
  "local_resilience_gap_production_object_storage=missing" \
  "local_resilience_gap_live_background_scheduling=missing" \
  "local_resilience_gap_live_systemd_supervision=missing" \
  "local_resilience_gap_live_kubernetes_rollout=missing" \
  "local_resilience_gap_production_timeline_failover=missing"

printf 'local_product_preflight=passed\n'
printf 'local_product_preflight_scope=application_drivers_pg_dump_restore_local_resilience\n'
printf 'local_product_preflight_drivers=tokio-postgres,sqlx,node-postgres,asyncpg,psycopg\n'
printf 'local_product_preflight_dump_restore=plain_custom_directory_tar_parallel_clean_insert_split\n'
printf 'local_product_preflight_resilience=backup_pitr_dr_plus_replication_deployment\n'
printf 'local_product_preflight_gap_pgx=blocked_missing_go\n'
printf 'local_product_preflight_gap_jdbc_r2dbc=blocked_missing_java_build_tooling\n'
printf 'local_product_preflight_gap_physical_page_image_backup=missing\n'
printf 'local_product_preflight_gap_production_object_storage=missing\n'
printf 'local_product_preflight_gap_live_background_scheduling=missing\n'
printf 'local_product_preflight_gap_live_systemd_supervision=missing\n'
printf 'local_product_preflight_gap_live_kubernetes_rollout=missing\n'
printf 'local_product_preflight_gap_production_timeline_failover=missing\n'
