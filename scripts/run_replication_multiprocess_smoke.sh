#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo build -p gpu_db_replication --example operational_multiprocess_smoke --quiet

binary="$repo_root/target/debug/examples/operational_multiprocess_smoke"
if [[ ! -x "$binary" ]]; then
  printf 'multiprocess smoke binary is not executable: %s\n' "$binary" >&2
  exit 1
fi

output="$("$binary")"
printf '%s\n' "$output"

required_lines=(
  "operational_replication_multiprocess_smoke=passed"
  "multiprocess_deployment_scope=parent_leader_two_follower_processes"
  "multiprocess_transport=tcp_append_entries child_processes=2 append_batches_sent=2 heartbeat_batches_sent=2 follower_acks_recorded=4"
  "multiprocess_follower id=2 commit=2 applied=2 caught_up=true read_after_apply=create table t(id int) | insert into t values (1)"
  "multiprocess_follower id=3 commit=2 applied=2 caught_up=true read_after_apply=create table t(id int) | insert into t values (1)"
  "deployment_gap_packaged_multiprocess_smoke=implemented"
  "deployment_gap_long_running_service=missing"
  "deployment_gap_container_deployment=missing"
)

for required in "${required_lines[@]}"; do
  if ! grep -Fqx "$required" <<<"$output"; then
    printf 'missing required multiprocess operational evidence line: %s\n' "$required" >&2
    exit 1
  fi
done
