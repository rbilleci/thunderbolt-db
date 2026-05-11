#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo build -p gpu_db_replication --example operational_cluster_smoke --quiet

binary="$repo_root/target/debug/examples/operational_cluster_smoke"
if [[ ! -x "$binary" ]]; then
  printf 'packaged smoke binary is not executable: %s\n' "$binary" >&2
  exit 1
fi

output="$("$binary")"
printf '%s\n' "$output"

required_lines=(
  "operational_replication_smoke=passed"
  "operational_deployment_preflight=passed"
  "deployment_scope=packaged_local_three_node_raft_smoke"
  "deployment_transport=single_request_tcp_append_entries append_batches_sent=3 heartbeat_batches_sent=2 follower_acks_recorded=3"
  "deployment_election=deterministic_request_vote candidate_id=1 elected_term=2 votes_granted=3 quorum=2 elected=true"
  "deployment_package=local_cargo_example_binary entrypoint=crates/replication/examples/operational_cluster_smoke.rs smoke_script=scripts/run_replication_cluster_smoke.sh packaged_script=scripts/run_replication_packaged_smoke.sh reproducible=true"
  "deployment_gap_network_transport=implemented"
  "deployment_gap_automatic_election=implemented"
  "deployment_gap_packaged_deployment=implemented"
)

for required in "${required_lines[@]}"; do
  if ! grep -Fqx "$required" <<<"$output"; then
    printf 'missing required packaged operational evidence line: %s\n' "$required" >&2
    exit 1
  fi
done
