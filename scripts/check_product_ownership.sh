#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
import subprocess
import sys

metadata = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--no-deps", "--format-version", "1"], text=True
))
listeners = [
    (package["name"], target["name"], target["src_path"])
    for package in metadata["packages"]
    for target in package["targets"]
    if "bin" in target["kind"]
    and target["name"].startswith("gpu-db")
]
expected = [("gpu_db_server", "gpu-db-engine-server")]
actual = [(package, target) for package, target, _ in listeners]
if actual != expected:
    print(f"expected exactly one product pgwire binary {expected}, found {listeners}", file=sys.stderr)
    sys.exit(1)
PY

test ! -e crates/protocol/src/bin/gpu-db-server.rs
test ! -d crates/protocol/src/bin/gpu-db-server
test ! -e crates/server/examples/p8_engine_pgwire_benchmark_endpoint.rs
test ! -d crates/server/examples/p8_engine_pgwire_benchmark_endpoint
test ! -e crates/server/examples/p8_engine_protocol_boundary_probe.rs

if rg -n \
  --glob '!scripts/check_product_ownership.sh' \
  --glob '!docs/**' \
  --glob '!target/**' \
  --glob '!Cargo.lock' \
  '(gpu-db-server|p8_engine_pgwire_benchmark_endpoint|p8_engine_protocol_boundary_probe)' \
  crates scripts tests; then
  echo 'superseded product-like protocol entry point remains live' >&2
  exit 1
fi

if ! rg -q '^\s*pub fn submit\(' crates/facade/src/lib.rs; then
  echo 'SharedEngine::submit facade boundary is missing' >&2
  exit 1
fi

if rg -q '^\s*pub fn (execute_text|enqueue_set_text|tick_batching|flush_admin|commit_mutation)' crates/facade/src; then
  echo 'superseded public facade mutation entry point remains' >&2
  exit 1
fi

if ! rg -q 'struct CommitPublicationCoordinator' crates/engine/src/engine_commit_coordinator.rs \
  || ! rg -q 'WalBuffer' crates/engine/src/engine_lifecycle.rs \
  || ! rg -q 'commit_mutex' crates/engine/src/engine_commit.rs; then
  echo 'canonical commit/WAL/publication authority evidence is incomplete' >&2
  exit 1
fi

printf 'product_ownership_guard=passed\n'
printf 'product_ownership_guard_server_target=gpu_db_server:gpu-db-engine-server\n'
printf 'product_ownership_guard_facade=SharedEngine::submit\n'
printf 'product_ownership_guard_commit_wal_publication=canonical\n'
