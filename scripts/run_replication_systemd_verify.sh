#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

unit="systemd/replication-follower/thunderbolt-db-replication-follower@.service"
env_a="systemd/replication-follower/replication-follower@2.env"
env_b="systemd/replication-follower/replication-follower@3.env"
migration="systemd/replication-follower/MIGRATION.md"
binary="target/debug/examples/operational_service_smoke"
verify_unit="$(mktemp "${TMPDIR:-/tmp}/thunderbolt-db-replication-follower-unit.XXXXXX.service")"

cleanup() {
  rm -f "$verify_unit"
}
trap cleanup EXIT

cargo build -p gpu_db_replication --example operational_service_smoke --quiet
sed "s#/usr/local/bin/operational_service_smoke#${repo_root}/${binary}#" "$unit" > "$verify_unit"
systemd-analyze verify "$verify_unit"

required_unit_lines=(
  "Description=Thunderbolt DB replication follower service %i"
  "Documentation=file:/usr/share/doc/thunderbolt-db/operations-runbooks.md"
  "EnvironmentFile=-/etc/thunderbolt-db/replication-follower@%i.env"
  "ExecStart=/usr/local/bin/operational_service_smoke --follower-service --id \${GPU_DB_REPLICATION_FOLLOWER_ID} --expected-requests \${GPU_DB_REPLICATION_EXPECTED_REQUESTS} --listen \${GPU_DB_REPLICATION_LISTEN_ADDR}"
  "Restart=on-failure"
  "NoNewPrivileges=true"
  "ProtectSystem=strict"
  "ReadWritePaths=/var/lib/thunderbolt-db"
)

for required in "${required_unit_lines[@]}"; do
  if ! grep -Fq "$required" "$unit"; then
    printf 'missing required systemd unit contract line: %s\n' "$required" >&2
    exit 1
  fi
done

for required in \
  "gpu-db-replication-follower@.service" \
  "thunderbolt-db-replication-follower@.service" \
  "/etc/gpu-db" \
  "/etc/thunderbolt-db" \
  "/var/lib/gpu-db" \
  "/var/lib/thunderbolt-db" \
  "Do not run the old and new units for the same follower ID concurrently."; do
  if ! grep -Fq "$required" "$migration"; then
    printf 'missing required systemd migration guidance: %s\n' "$required" >&2
    exit 1
  fi
done

for env_file in "$env_a" "$env_b"; do
  for required in \
    "GPU_DB_REPLICATION_FOLLOWER_ID=" \
    "GPU_DB_REPLICATION_EXPECTED_REQUESTS=" \
    "GPU_DB_REPLICATION_LISTEN_ADDR=0.0.0.0:55432"; do
    if ! grep -Fq "$required" "$env_file"; then
      printf 'missing required systemd environment contract line in %s: %s\n' "$env_file" "$required" >&2
      exit 1
    fi
  done
done

printf 'operational_replication_systemd_verify=passed\n'
printf 'systemd_unit=%s\n' "$unit"
printf 'systemd_env_files=%s,%s\n' "$env_a" "$env_b"
printf 'systemd_binary=%s\n' "$binary"
printf 'systemd_service_contract=follower_service id_expected_requests_listen\n'
printf 'systemd_rename_migration=old_unit_and_paths_to_thunderbolt_db_paths\n'
printf 'deployment_gap_production_service_manager=implemented_unit_syntax_and_command_contract\n'
printf 'deployment_gap_live_systemd_supervision=missing\n'
printf 'deployment_gap_kubernetes_deployment=missing\n'
