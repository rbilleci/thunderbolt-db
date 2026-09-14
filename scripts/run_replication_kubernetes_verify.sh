#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

manifest="k8s/replication-service/follower-services.yml"
binary="target/debug/examples/operational_service_smoke"

cargo build -p gpu_db_replication --example operational_service_smoke --quiet

python3 - "$manifest" <<'PY'
import sys

try:
    import yaml
except ImportError:
    yaml = None

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest_text = handle.read()

if yaml is None:
    if manifest_text.count("kind: Deployment") != 2 or manifest_text.count("kind: Service") != 2:
        raise SystemExit("manifest must contain two Deployments and two Services")
    for follower_id in ("2", "3"):
        required = [
            f"name: gpu-db-replication-follower-{follower_id}",
            f"gpu-db-follower-id: \"{follower_id}\"",
            "image: gpu-db-replication-service:local",
            "- --follower-service",
            "- $(GPU_DB_REPLICATION_FOLLOWER_ID)",
            "- $(GPU_DB_REPLICATION_EXPECTED_REQUESTS)",
            "- $(GPU_DB_REPLICATION_LISTEN_ADDR)",
            f"- name: GPU_DB_REPLICATION_FOLLOWER_ID\n              value: \"{follower_id}\"",
            "- name: GPU_DB_REPLICATION_EXPECTED_REQUESTS\n              value: \"4\"",
            "- name: GPU_DB_REPLICATION_LISTEN_ADDR\n              value: 0.0.0.0:55432",
            "containerPort: 55432",
            "targetPort: append",
        ]
        for snippet in required:
            if snippet not in manifest_text:
                raise SystemExit(f"missing Kubernetes manifest contract snippet: {snippet}")
    print("kubernetes_manifest_contract=passed")
    raise SystemExit(0)

docs = [doc for doc in yaml.safe_load_all(manifest_text) if doc is not None]

if len(docs) != 4:
    raise SystemExit(f"expected 4 Kubernetes resources, got {len(docs)}")

by_kind_name = {(doc.get("kind"), doc.get("metadata", {}).get("name")): doc for doc in docs}
expected_args = [
    "--follower-service",
    "--id",
    "$(GPU_DB_REPLICATION_FOLLOWER_ID)",
    "--expected-requests",
    "$(GPU_DB_REPLICATION_EXPECTED_REQUESTS)",
    "--listen",
    "$(GPU_DB_REPLICATION_LISTEN_ADDR)",
]

for follower_id in ("2", "3"):
    name = f"gpu-db-replication-follower-{follower_id}"
    deployment = by_kind_name.get(("Deployment", name))
    service = by_kind_name.get(("Service", name))
    if deployment is None or service is None:
        raise SystemExit(f"missing Deployment/Service pair for follower {follower_id}")

    labels = deployment["spec"]["selector"]["matchLabels"]
    if labels.get("gpu-db-follower-id") != follower_id:
        raise SystemExit(f"deployment selector does not pin follower {follower_id}")
    if deployment["spec"].get("replicas") != 1:
        raise SystemExit(f"deployment {name} must be single-replica")

    containers = deployment["spec"]["template"]["spec"]["containers"]
    if len(containers) != 1:
        raise SystemExit(f"deployment {name} must have exactly one container")
    container = containers[0]
    if container.get("image") != "gpu-db-replication-service:local":
        raise SystemExit(f"deployment {name} has unexpected image {container.get('image')}")
    if container.get("imagePullPolicy") != "IfNotPresent":
        raise SystemExit(f"deployment {name} must use IfNotPresent for local image smoke")
    if container.get("args") != expected_args:
        raise SystemExit(f"deployment {name} args drifted from follower-service contract")

    env = {entry["name"]: str(entry["value"]) for entry in container.get("env", [])}
    expected_env = {
        "GPU_DB_REPLICATION_FOLLOWER_ID": follower_id,
        "GPU_DB_REPLICATION_EXPECTED_REQUESTS": "4",
        "GPU_DB_REPLICATION_LISTEN_ADDR": "0.0.0.0:55432",
    }
    if env != expected_env:
        raise SystemExit(f"deployment {name} env drifted: {env}")

    ports = container.get("ports", [])
    if ports != [{"name": "append", "containerPort": 55432}]:
        raise SystemExit(f"deployment {name} port contract drifted: {ports}")

    selector = service["spec"]["selector"]
    if selector.get("gpu-db-follower-id") != follower_id:
        raise SystemExit(f"service selector does not pin follower {follower_id}")
    service_ports = service["spec"].get("ports", [])
    if service_ports != [{"name": "append", "port": 55432, "targetPort": "append"}]:
        raise SystemExit(f"service {name} port contract drifted: {service_ports}")

print("kubernetes_manifest_contract=passed")
PY

if command -v kubectl >/dev/null 2>&1; then
  kubectl apply --dry-run=client --validate=false -f "$manifest" >/dev/null
  kubectl_contract="client_dry_run_passed"
else
  kubectl_contract="not_available_local_manifest_contract_only"
fi

printf 'operational_replication_kubernetes_verify=passed\n'
printf 'kubernetes_manifest=%s\n' "$manifest"
printf 'kubernetes_binary=%s\n' "$binary"
printf 'kubernetes_service_contract=follower_service id_expected_requests_listen_append_port\n'
printf 'kubernetes_resources=deployments:2,services:2\n'
printf 'kubernetes_kubectl_validation=%s\n' "$kubectl_contract"
printf 'deployment_gap_kubernetes_deployment=implemented_manifest_contract\n'
printf 'deployment_gap_live_kubernetes_rollout=missing\n'
