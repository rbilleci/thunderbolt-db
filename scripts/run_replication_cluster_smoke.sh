#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output="$(cargo run -p gpu_db_replication --example operational_cluster_smoke --quiet)"
printf '%s\n' "$output"

required_lines=(
  "operational_replication_smoke=passed"
  "operational_deployment_preflight=passed"
  "deployment_scope=in_process_three_node_raft_smoke"
  "deployment_gap_network_transport=missing"
  "deployment_gap_automatic_election=missing"
  "deployment_gap_packaged_deployment=missing"
)

for required in "${required_lines[@]}"; do
  if ! grep -Fqx "$required" <<<"$output"; then
    printf 'missing required operational evidence line: %s\n' "$required" >&2
    exit 1
  fi
done
