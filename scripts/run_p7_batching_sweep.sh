#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

rows="${GPU_DB_BENCH_ROWS:-1000}"
lookup_sizes="${GPU_DB_BENCH_LOOKUP_SIZES:-1 4 16 64}"
git_sha="$(git rev-parse HEAD)"

device_info="nvidia-smi unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
  device_info="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | paste -sd ';' -)"
fi

mkdir -p target/bench
report="target/bench/p7-batching-sweep.md"

{
  echo "# P7 Batching Sweep Benchmark"
  echo
  echo "- git_sha: ${git_sha}"
  echo "- stream: benchmark"
  echo "- dataset_rows: ${rows}"
  echo "- lookup_sizes: ${lookup_sizes}"
  echo "- device_info: ${device_info}"
  echo "- reproduction: GPU_DB_BENCH_LOOKUP_SIZES=\"${lookup_sizes}\" GPU_DB_BENCH_ROWS=${rows} scripts/run_p7_batching_sweep.sh"
  echo
} >"$report"

for lookup_count in $lookup_sizes; do
  {
    echo "## lookup_count=${lookup_count}"
    echo
  } >>"$report"

  GPU_DB_BENCH_ROWS="$rows" \
    GPU_DB_BENCH_LOOKUPS="$lookup_count" \
    GPU_DB_BENCH_DEVICE_INFO="$device_info" \
    cargo run -p gpu_db_engine --example relational_workload_benchmark --quiet \
    | sed 's/^# /### /; s/^## /#### /' >>"$report"

  echo >>"$report"
done

cat "$report"
