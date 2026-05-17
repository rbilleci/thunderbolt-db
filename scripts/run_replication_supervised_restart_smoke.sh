#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo build -p gpu_db_replication --example operational_service_smoke --quiet
binary="$repo_root/target/debug/examples/operational_service_smoke"
if [[ ! -x "$binary" ]]; then
  printf 'service smoke binary is not executable: %s\n' "$binary" >&2
  exit 1
fi

output="$("$binary" --supervised-restart)"
printf '%s\n' "$output"

required_lines=(
  "operational_replication_supervised_restart_smoke=passed"
  "supervised_restart_scope=parent_leader_restarts_one_follower_service"
  "supervised_restart_transport=tcp_append_entries follower_services=2 restarted_follower=2 append_batches_sent=4 heartbeat_batches_sent=4 follower_acks_recorded=6"
  "service_follower id=2 commit=2 applied=2 caught_up=true read_after_apply=create table t(id int) | insert into t values (1)"
  "service_follower id=2 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)"
  "service_follower id=3 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)"
  "supervised_restart_replay=full_durable_prefix_after_restart"
  "service_shutdown=controlled follower_services=2"
  "deployment_gap_service_restart_supervision=implemented_bounded_local_smoke"
  "deployment_gap_production_supervision=missing"
  "deployment_gap_kubernetes_deployment=missing"
)

for required in "${required_lines[@]}"; do
  if ! grep -Fq "$required" <<<"$output"; then
    printf 'missing required supervised-restart operational evidence line: %s\n' "$required" >&2
    exit 1
  fi
done
