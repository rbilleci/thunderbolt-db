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

output="$("$binary")"
printf '%s\n' "$output"

required_lines=(
  "operational_replication_service_smoke=passed"
  "service_deployment_scope=parent_leader_two_long_running_follower_services"
  "service_transport=tcp_append_entries follower_services=2 append_batches_sent=4 heartbeat_batches_sent=4 follower_acks_recorded=6 requests_per_service=4"
  "service_follower id=2 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)"
  "service_follower id=3 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)"
  "service_shutdown=controlled follower_services=2"
  "deployment_gap_long_running_service=implemented"
  "deployment_gap_container_deployment=missing"
)

for required in "${required_lines[@]}"; do
  if ! grep -Fq "$required" <<<"$output"; then
    printf 'missing required service operational evidence line: %s\n' "$required" >&2
    exit 1
  fi
done
