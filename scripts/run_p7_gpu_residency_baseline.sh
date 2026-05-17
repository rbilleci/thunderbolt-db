#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

rows="${GPU_DB_BENCH_ROWS:-1000}"
lookups="${GPU_DB_BENCH_LOOKUPS:-16}"
git_sha="$(git rev-parse HEAD)"

device_info="nvidia-smi unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
  device_info="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' -)"
fi

mkdir -p target/bench
report="target/bench/p7-gpu-residency-baseline.md"

{
  echo "# P7 GPU Residency Baseline"
  echo
  echo "- git_sha: ${git_sha}"
  echo "- stream: benchmark"
  echo "- dataset_rows: ${rows}"
  echo "- lookup_count: ${lookups}"
  echo "- device_info: ${device_info}"
  echo "- reproduction: GPU_DB_BENCH_ROWS=${rows} GPU_DB_BENCH_LOOKUPS=${lookups} scripts/run_p7_gpu_residency_baseline.sh"
  echo
} >"$report"

GPU_DB_BENCH_ROWS="$rows" \
  GPU_DB_BENCH_LOOKUPS="$lookups" \
  GPU_DB_BENCH_DEVICE_INFO="$device_info" \
  cargo run -p gpu_db_engine --example relational_gpu_residency_baseline --quiet \
  | sed 's/^# /## /; s/^## /### /' >>"$report"

cat "$report"
