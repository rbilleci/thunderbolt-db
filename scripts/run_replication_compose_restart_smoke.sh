#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

image="thunderbolt-db-replication-service-smoke:local"
project="thunderbolt-db-replication-compose-restart-smoke-$$"
compose_file="docker/replication-service/compose-smoke.yml"
build_context="$(mktemp -d "${TMPDIR:-/tmp}/thunderbolt-db-replication-compose-restart-smoke.XXXXXX")"

cleanup() {
  GPU_DB_REPLICATION_SERVICE_IMAGE="$image" docker compose -f "$compose_file" -p "$project" down -v --remove-orphans >/dev/null 2>&1 || true
  rm -rf "$build_context"
}
trap cleanup EXIT

cargo build -p gpu_db_replication --example operational_service_smoke --quiet
cp target/debug/examples/operational_service_smoke "$build_context/operational_service_smoke"
docker build -q -t "$image" -f docker/replication-service/Dockerfile "$build_context" >/dev/null
GPU_DB_REPLICATION_SERVICE_IMAGE="$image" docker compose -f "$compose_file" -p "$project" up -d >/dev/null

wait_for_ready() {
  local service="$1"
  local id="$2"
  local expected="service_follower_ready id=${id} addr=0.0.0.0:55432"
  for _ in {1..50}; do
    if GPU_DB_REPLICATION_SERVICE_IMAGE="$image" docker compose -f "$compose_file" -p "$project" logs --no-color "$service" 2>&1 | grep -Fq "$expected"; then
      return 0
    fi
    sleep 0.1
  done
  printf 'compose service %s did not become ready; logs follow:\n' "$service" >&2
  GPU_DB_REPLICATION_SERVICE_IMAGE="$image" docker compose -f "$compose_file" -p "$project" logs --no-color "$service" >&2 || true
  return 1
}

wait_for_ready follower-a 2
wait_for_ready follower-b 3

port_a="$(GPU_DB_REPLICATION_SERVICE_IMAGE="$image" docker compose -f "$compose_file" -p "$project" port follower-a 55432 | awk -F: '{print $NF}')"
port_b="$(GPU_DB_REPLICATION_SERVICE_IMAGE="$image" docker compose -f "$compose_file" -p "$project" port follower-b 55432 | awk -F: '{print $NF}')"

parent_output="$(
  target/debug/examples/operational_service_smoke \
    --compose-supervised-restart \
    --restarting-follower "2=127.0.0.1:${port_a}" \
    --stable-follower "3=127.0.0.1:${port_b}" \
    --compose-file "$compose_file" \
    --compose-project "$project" \
    --restart-service follower-a
)"

compose_logs="$(GPU_DB_REPLICATION_SERVICE_IMAGE="$image" docker compose -f "$compose_file" -p "$project" logs --no-color follower-a follower-b)"
output="$parent_output"$'\n'"$compose_logs"
printf '%s\n' "$output"

required_lines=(
  "operational_replication_compose_restart_smoke=host_parent_passed"
  "compose_restart_scope=host_leader_restarts_one_compose_follower_service"
  "compose_restart_transport=tcp_append_entries follower_services=2 restarted_follower=2 stable_follower=3 restart_service=follower-a append_batches_sent=4 heartbeat_batches_sent=4 follower_acks_recorded=6"
  "service_follower id=2 commit=2 applied=2 caught_up=true read_after_apply=create table t(id int) | insert into t values (1)"
  "service_follower id=2 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)"
  "service_follower id=3 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)"
  "compose_restart_replay=full_durable_prefix_after_restart"
  "deployment_gap_compose_restart_supervision=implemented_bounded_local_smoke"
  "deployment_gap_production_supervision=missing"
  "deployment_gap_kubernetes_deployment=missing"
)

for required in "${required_lines[@]}"; do
  if ! grep -Fq "$required" <<<"$output"; then
    printf 'missing required compose-restart operational evidence line: %s\n' "$required" >&2
    exit 1
  fi
done
