#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_step() {
  local name="$1"
  shift
  printf 'backup_pitr_dr_step=%s status=running\n' "$name"
  "$@"
  printf 'backup_pitr_dr_step=%s status=passed\n' "$name"
}

run_step base_plus_archive_txn_restore \
  cargo test -q -p gpu_db_engine \
    relational_state_recovers_from_base_checkpoint_plus_wal_archive_transaction_target \
    -- --color never

run_step base_plus_archive_timestamp_restore \
  cargo test -q -p gpu_db_engine \
    relational_state_recovers_from_base_checkpoint_plus_wal_archive_timestamp_target \
    -- --color never

run_step checkpoint_pitr_window_retention \
  cargo test -q -p gpu_db_engine \
    checkpoint_window_archive_retention \
    -- --color never

run_step local_maintenance_cleanup_preflight \
  scripts/run_wal_archive_maintenance_preflight_smoke.sh

run_step object_bundle_backup_preflight \
  scripts/run_wal_archive_object_backup_preflight_smoke.sh

run_step mvcc_retention_boundary \
  cargo test -q -p gpu_db_engine checkpoint_vacuum -- --color never

printf 'backup_pitr_dr_drill=passed\n'
printf 'backup_pitr_dr_restore_targets=transaction,timestamp\n'
printf 'backup_pitr_dr_local_maintenance=archive_retention_plus_timeline_prune\n'
printf 'backup_pitr_dr_object_bundle=file_backed_export_restore_recover_corrupt_reject\n'
printf 'backup_pitr_dr_scope=local_checkpoint_control_wal_archive_timeline_registry_object_bundle\n'
printf 'backup_pitr_dr_gap_physical_page_image_backup=missing\n'
printf 'backup_pitr_dr_gap_production_object_storage=missing\n'
printf 'backup_pitr_dr_gap_live_background_scheduling=missing\n'
printf 'backup_pitr_dr_gap_production_timeline_failover=missing\n'
