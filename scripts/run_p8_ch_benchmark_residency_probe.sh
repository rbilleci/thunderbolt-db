#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

OUT_DIR="${GPU_DB_CH_BENCH_OUT_DIR:-target/p8-ch-benchmark-residency}"
ROWS="${GPU_DB_CH_BENCH_ROWS:-512}"
CONCURRENCY="${GPU_DB_CH_BENCH_CONCURRENCY:-1,10}"

usage() {
  cat <<'USAGE'
usage: scripts/run_p8_ch_benchmark_residency_probe.sh [--dry-run|--run-baseline|--run-25pct|--cleanup|--self-check]

Environment:
  GPU_DB_CH_BENCH_OUT_DIR       output directory, default target/p8-ch-benchmark-residency
  GPU_DB_CH_BENCH_ROWS          safe calibration rows, default 512
  GPU_DB_CH_BENCH_CONCURRENCY   logical request targets, default 1,10
  GPU_DB_CH_BENCH_MAX_ROWS      scheduled-run guardrail, default 10000
USAGE
}

write_25pct_blocker() {
  mkdir -p "$OUT_DIR"
  local retained_target_bytes=6442450944
  local retained_bytes_per_row=40
  local generated_bytes_per_row=96
  local estimated_rows=$(((retained_target_bytes + retained_bytes_per_row - 1) / retained_bytes_per_row))
  local generated_table_bytes=$((estimated_rows * generated_bytes_per_row))
  local wal_log_bytes=$((generated_table_bytes / 2))
  local report_bytes=$((2 * 1024 * 1024))
  local required_bytes=$((generated_table_bytes + wal_log_bytes + report_bytes))
  local available_bytes
  available_bytes=$(df -B1 "$OUT_DIR" | awk 'NR==2 {print $4}')
  cat >"$OUT_DIR/25pct-preflight.md" <<PREFLIGHT
# P8 CH-benCHmark 25% VRAM Preflight

- tier: 25pct
- retained_target_bytes: $retained_target_bytes
- estimated_order_line_rows: $estimated_rows
- generated_table_bytes: $generated_table_bytes
- wal_log_bytes: $wal_log_bytes
- report_bytes: $report_bytes
- required_disk_bytes: $required_bytes
- available_disk_bytes: $available_bytes
- disk_preflight: $(if [ "$available_bytes" -gt "$required_bytes" ]; then echo pass; else echo fail; fi)
- run_preflight: blocked

The current executable probe seeds the table through the in-memory MVCC engine
and builds a resident snapshot by collecting all decoded rows plus the device
payload in process memory before admission. It does not yet have a bounded
streaming/on-disk generator or chunked resident snapshot builder for the
estimated 161061274-row / 6 GiB retained tier.
PREFLIGHT
  cat >"$OUT_DIR/25pct-preflight.jsonl" <<PREFLIGHT_JSON
{"kind":"tier_preflight","tier":"25pct","retained_target_bytes":$retained_target_bytes,"estimated_rows":$estimated_rows,"generated_table_bytes":$generated_table_bytes,"wal_log_bytes":$wal_log_bytes,"report_bytes":$report_bytes,"required_disk_bytes":$required_bytes,"available_disk_bytes":$available_bytes,"disk_preflight":$(if [ "$available_bytes" -gt "$required_bytes" ]; then echo true; else echo false; fi),"run_preflight":"blocked","blocker":"missing_bounded_streaming_on_disk_generator_and_chunked_resident_snapshot_builder"}
PREFLIGHT_JSON
  cat "$OUT_DIR/25pct-preflight.md"
  echo "p8_ch_benchmark_25pct=blocked reason=missing_bounded_streaming_on_disk_generator_and_chunked_resident_snapshot_builder" >&2
  return 1
}

mode="${1:---dry-run}"
case "$mode" in
  --dry-run)
    mkdir -p "$OUT_DIR"
    cargo run -q -p gpu_db_engine --example p8_ch_benchmark_residency_probe -- \
      --estimate \
      --output-dir "$OUT_DIR" \
      --rows "$ROWS"
    test -s "$OUT_DIR/estimate.md"
    test -s "$OUT_DIR/estimate.jsonl"
    grep -q '"tier":"25pct"' "$OUT_DIR/estimate.jsonl"
    grep -q '"tier":"200pct"' "$OUT_DIR/estimate.jsonl"
    if grep -q '"tier":"400pct"' "$OUT_DIR/estimate.jsonl"; then
      echo "400% VRAM benchmark tier should not be configured" >&2
      exit 1
    fi
    grep -q 'concurrency_targets: \[1, 10, 100, 1000, 10000\]' "$OUT_DIR/estimate.md"
    cat "$OUT_DIR/estimate.md"
    ;;
  --run-baseline)
    mkdir -p "$OUT_DIR"
    cargo run -q -p gpu_db_engine --example p8_ch_benchmark_residency_probe -- \
      --run \
      --output-dir "$OUT_DIR" \
      --rows "$ROWS" \
      --concurrency "$CONCURRENCY"
    test -s "$OUT_DIR/baseline.md"
    test -s "$OUT_DIR/metrics.jsonl"
    grep -q '"kind":"metric"' "$OUT_DIR/metrics.jsonl"
    grep -q '"resident_route_zero_h2d":true' "$OUT_DIR/metrics.jsonl"
    if grep -q '"resident_route_accepted":false' "$OUT_DIR/metrics.jsonl"; then
      echo "p8 CH-benCHmark baseline included a rejected resident route" >&2
      exit 1
    fi
    grep -q '"kind":"over_residency_probe"' "$OUT_DIR/metrics.jsonl"
    cat "$OUT_DIR/baseline.md"
    ;;
  --run-25pct)
    write_25pct_blocker
    ;;
  --cleanup)
    rm -rf "$OUT_DIR"
    if [ -e "$OUT_DIR" ]; then
      echo "p8_ch_benchmark_cleanup=failed path=$OUT_DIR"
      exit 1
    fi
    echo "p8_ch_benchmark_cleanup=passed path=$OUT_DIR"
    ;;
  --self-check)
    tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/gpu-db-p8-ch-bench.XXXXXX")
    trap 'rm -rf "$tmp_dir"' EXIT
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_ROWS=16 GPU_DB_CH_BENCH_CONCURRENCY=1 \
      "$0" --dry-run >"$tmp_dir/dry-run.out"
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_ROWS=16 GPU_DB_CH_BENCH_CONCURRENCY=1 \
      "$0" --run-baseline >"$tmp_dir/run.out"
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" "$0" --cleanup >"$tmp_dir/cleanup.out"
    grep -q 'p8_ch_benchmark_cleanup=passed' "$tmp_dir/cleanup.out"
    echo "p8 ch benchmark residency probe self-check passed"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
