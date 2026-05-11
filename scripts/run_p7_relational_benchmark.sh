#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

device_info="nvidia-smi unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
  device_info="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | paste -sd ';' -)"
fi

mkdir -p target/bench
GPU_DB_BENCH_DEVICE_INFO="$device_info" \
  cargo run -p gpu_db_engine --example relational_workload_benchmark --quiet \
  | tee target/bench/p7-relational-benchmark.md
