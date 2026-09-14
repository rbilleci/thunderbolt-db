#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

image="thunderbolt-db-replication-service-smoke:local"
network="thunderbolt-db-replication-smoke-$$"
container_a="thunderbolt-db-replication-smoke-a-$$"
container_b="thunderbolt-db-replication-smoke-b-$$"
build_context="$(mktemp -d "${TMPDIR:-/tmp}/thunderbolt-db-replication-container-smoke.XXXXXX")"

cleanup() {
  docker rm -f "$container_a" "$container_b" >/dev/null 2>&1 || true
  docker network rm "$network" >/dev/null 2>&1 || true
  rm -rf "$build_context"
}
trap cleanup EXIT

cargo build -p gpu_db_replication --example operational_service_smoke --quiet
cp target/debug/examples/operational_service_smoke "$build_context/operational_service_smoke"
docker build -q -t "$image" -f docker/replication-service/Dockerfile "$build_context" >/dev/null
docker network create "$network" >/dev/null

docker run -d --name "$container_a" --network "$network" -p 127.0.0.1::55432 \
  "$image" --follower-service --id 2 --expected-requests 4 --listen 0.0.0.0:55432 >/dev/null
docker run -d --name "$container_b" --network "$network" -p 127.0.0.1::55432 \
  "$image" --follower-service --id 3 --expected-requests 4 --listen 0.0.0.0:55432 >/dev/null

wait_for_ready() {
  local container="$1"
  local id="$2"
  local expected="service_follower_ready id=${id} addr=0.0.0.0:55432"
  for _ in {1..50}; do
    if docker logs "$container" 2>&1 | grep -Fq "$expected"; then
      return 0
    fi
    sleep 0.1
  done
  printf 'container %s did not become ready; logs follow:\n' "$container" >&2
  docker logs "$container" >&2 || true
  return 1
}

wait_for_ready "$container_a" 2
wait_for_ready "$container_b" 3

port_a="$(docker port "$container_a" 55432/tcp | awk -F: '{print $NF}')"
port_b="$(docker port "$container_b" 55432/tcp | awk -F: '{print $NF}')"

parent_output="$(
  target/debug/examples/operational_service_smoke \
    --external-follower "2=127.0.0.1:${port_a}" \
    --external-follower "3=127.0.0.1:${port_b}"
)"

docker wait "$container_a" "$container_b" >/dev/null
container_logs="$(docker logs "$container_a"; docker logs "$container_b")"
output="$parent_output"$'\n'"$container_logs"
printf '%s\n' "$output"

required_lines=(
  "operational_replication_container_smoke=host_parent_passed"
  "container_deployment_scope=host_leader_two_follower_service_containers"
  "service_transport=tcp_append_entries follower_services=2 append_batches_sent=4 heartbeat_batches_sent=4 follower_acks_recorded=6 requests_per_service=4"
  "service_follower id=2 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)"
  "service_follower id=3 commit=3 applied=3 caught_up=true read_after_apply=create table t(id int) | insert into t values (1) | insert into t values (2)"
  "service_shutdown=controlled follower_services=2"
  "deployment_gap_long_running_service=implemented"
  "deployment_gap_container_deployment=implemented"
)

for required in "${required_lines[@]}"; do
  if ! grep -Fq "$required" <<<"$output"; then
    printf 'missing required container operational evidence line: %s\n' "$required" >&2
    exit 1
  fi
done
