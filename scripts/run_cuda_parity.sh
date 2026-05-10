#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${CUDA_PARITY_OUT_DIR:-"$ROOT/target/cuda-parity"}"
REPORT="$OUT_DIR/environment.txt"
LOG="$OUT_DIR/cuda-parity.log"

mkdir -p "$OUT_DIR"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo "cargo=$(cargo --version)"
  echo "rustc=$(rustc --version)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia_smi_query="
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  else
    echo "nvidia_smi=missing"
  fi
} | tee "$REPORT"

{
  echo "== cuda runtime =="
  cargo test -p gpu_db_execution cuda_driver_runtime -- --include-ignored --nocapture
  echo "== cuda mvcc engine =="
  cargo test -p gpu_db_engine execute_mvcc_query_cuda_driver_runs -- --include-ignored --nocapture
} 2>&1 | tee "$LOG"

echo "environment_report=$REPORT"
echo "test_log=$LOG"
