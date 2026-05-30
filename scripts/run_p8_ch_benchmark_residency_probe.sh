#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

OUT_DIR="${GPU_DB_CH_BENCH_OUT_DIR:-target/p8-ch-benchmark-residency}"
ROWS="${GPU_DB_CH_BENCH_ROWS:-512}"
CONCURRENCY="${GPU_DB_CH_BENCH_CONCURRENCY:-1,10}"

usage() {
  cat <<'USAGE'
usage: scripts/run_p8_ch_benchmark_residency_probe.sh [--dry-run|--run-baseline|--run-25pct|--pgsql-baseline-preflight|--streaming-self-check|--cleanup|--self-check]

Environment:
  GPU_DB_CH_BENCH_OUT_DIR       output directory, default target/p8-ch-benchmark-residency
  GPU_DB_CH_BENCH_ROWS          safe calibration rows, default 512
  GPU_DB_CH_BENCH_CONCURRENCY   logical request targets, default 1,10
  GPU_DB_CH_BENCH_MAX_ROWS      scheduled-run guardrail, default 10000
  GPU_DB_CH_BENCH_CHUNK_ROWS    streaming self-check chunk rows, default 16
  GPU_DB_CH_BENCH_PGSQL_URL     libpq connection string for PostgreSQL baseline
USAGE
}

write_pgsql_baseline_workload_sql() {
  local sql_path="$1"
  local rows="$2"
  cat >"$sql_path" <<'SQL'
\set ON_ERROR_STOP on
DROP TABLE IF EXISTS order_line;
CREATE TABLE order_line (
  ol_o_id INT,
  ol_i_id INT,
  ol_quantity INT,
  ol_amount INT,
  ol_dist_info TEXT
);
COPY order_line (ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info) FROM STDIN WITH (FORMAT csv);
SQL
  local id bucket item qty amount dist
  for ((id = 1; id <= rows; id++)); do
    bucket=$((id % 10))
    item=$(((id % 100000) + 1))
    qty=$(((id % 50) + 1))
    amount=$(((id * 17) % 100000))
    if ((id % 2 == 0)); then
      dist="alpha"
    else
      dist="omega"
    fi
    printf '%s,%s,%s,%s,%s%s\n' "$id" "$item" "$qty" "$amount" "$dist" "$bucket" >>"$sql_path"
  done
  cat >>"$sql_path" <<'SQL'
\.
ANALYZE order_line;
SELECT COUNT(*) AS order_line_count_all FROM order_line;
SELECT SUM(ol_amount) AS order_line_sum_amount FROM order_line;
SELECT AVG(ol_quantity) AS order_line_avg_quantity_between FROM order_line WHERE ol_quantity BETWEEN 10 AND 40;
SELECT MAX(ol_amount) AS order_line_max_amount_filter FROM order_line WHERE ol_amount >= :lower_bound;
EXPLAIN (ANALYZE, FORMAT JSON) SELECT COUNT(*) FROM order_line;
EXPLAIN (ANALYZE, FORMAT JSON) SELECT SUM(ol_amount) FROM order_line;
EXPLAIN (ANALYZE, FORMAT JSON) SELECT AVG(ol_quantity) FROM order_line WHERE ol_quantity BETWEEN 10 AND 40;
EXPLAIN (ANALYZE, FORMAT JSON) SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= :lower_bound;
SQL
}

write_pgsql_baseline_preflight() {
  mkdir -p "$OUT_DIR/pgsql-baseline"
  local rows="$ROWS"
  local lower_bound=$((rows / 4))
  if [ "$lower_bound" -lt 1 ]; then
    lower_bound=1
  fi
  local sql_path="$OUT_DIR/pgsql-baseline/workload.sql"
  local out_path="$OUT_DIR/pgsql-baseline/psql.out"
  local err_path="$OUT_DIR/pgsql-baseline/psql.err"
  local report_path="$OUT_DIR/pgsql-baseline/preflight.md"
  local json_path="$OUT_DIR/pgsql-baseline/preflight.jsonl"

  write_pgsql_baseline_workload_sql "$sql_path" "$rows"

  if ! command -v psql >/dev/null 2>&1; then
    cat >"$report_path" <<PREFLIGHT
# PostgreSQL Baseline Preflight

- rows: $rows
- workload_sql: $sql_path
- status: blocked
- blocker: missing_psql_client

The P8 benchmark must be baselined against PostgreSQL for the same generated
dataset and query set. The local psql client is not available, so no GPU DB
benchmark result should be treated as comparable yet.
PREFLIGHT
    printf '{"kind":"pgsql_baseline_preflight","rows":%s,"status":"blocked","blocker":"missing_psql_client"}\n' "$rows" >"$json_path"
    cat "$report_path"
    echo "p8_ch_benchmark_pgsql_baseline=blocked reason=missing_psql_client" >&2
    return 1
  fi

  if [ -z "${GPU_DB_CH_BENCH_PGSQL_URL:-}" ]; then
    cat >"$report_path" <<PREFLIGHT
# PostgreSQL Baseline Preflight

- rows: $rows
- workload_sql: $sql_path
- psql_version: $(psql --version)
- status: blocked
- blocker: missing_pgsql_baseline_connection

The P8 benchmark must be baselined against PostgreSQL for the same generated
dataset and query set before GPU DB numbers are interpreted. Set
\`GPU_DB_CH_BENCH_PGSQL_URL\` to a disposable PostgreSQL database connection
string, then rerun this preflight.
PREFLIGHT
    printf '{"kind":"pgsql_baseline_preflight","rows":%s,"status":"blocked","psql_client":true,"blocker":"missing_pgsql_baseline_connection"}\n' "$rows" >"$json_path"
    cat "$report_path"
    echo "p8_ch_benchmark_pgsql_baseline=blocked reason=missing_pgsql_baseline_connection" >&2
    return 1
  fi

  if psql "$GPU_DB_CH_BENCH_PGSQL_URL" -v lower_bound="$lower_bound" -X -f "$sql_path" >"$out_path" 2>"$err_path"; then
    cat >"$report_path" <<PREFLIGHT
# PostgreSQL Baseline Preflight

- rows: $rows
- workload_sql: $sql_path
- output: $out_path
- stderr: $err_path
- psql_version: $(psql --version)
- status: pass

The PostgreSQL baseline workload loaded the same deterministic \`order_line\`
dataset and ran the same aggregate query set plus \`EXPLAIN ANALYZE\` JSON
probes. GPU DB benchmark reports must include this PostgreSQL output or a
newer baseline artifact before claiming comparative performance.
PREFLIGHT
    printf '{"kind":"pgsql_baseline_preflight","rows":%s,"status":"pass","workload_sql":"%s","output":"%s"}\n' "$rows" "$sql_path" "$out_path" >"$json_path"
    cat "$report_path"
    return 0
  fi

  cat >"$report_path" <<PREFLIGHT
# PostgreSQL Baseline Preflight

- rows: $rows
- workload_sql: $sql_path
- output: $out_path
- stderr: $err_path
- psql_version: $(psql --version)
- status: blocked
- blocker: pgsql_baseline_workload_failed

The PostgreSQL baseline connection was configured, but the workload failed.
Inspect the stderr artifact before interpreting GPU DB benchmark results.
PREFLIGHT
  printf '{"kind":"pgsql_baseline_preflight","rows":%s,"status":"blocked","blocker":"pgsql_baseline_workload_failed","stderr":"%s"}\n' "$rows" "$err_path" >"$json_path"
  cat "$report_path"
  echo "p8_ch_benchmark_pgsql_baseline=blocked reason=pgsql_baseline_workload_failed" >&2
  return 1
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
- bounded_streaming_generator_probe: available via \`--streaming-self-check\`
- postgresql_baseline_required: \`scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-preflight\`
- chunked_resident_cache_install: blocked
- blocker: missing_relational_resident_cache_chunked_install_api

The current executable probe now has a small checked benchmark-only streaming
artifact generator, but the engine residency boundary still installs snapshots
through an in-memory \`RelationalResidencySnapshot\` with
\`resident_rows: Vec<Vec<SqlValue>>\` and one retained device payload. The 25%
tier cannot safely start until \`RelationalResidentCache\` or an equivalent
benchmark-only admission API can install column chunks without materializing
all generated rows and the whole payload in process memory.
PREFLIGHT
  cat >"$OUT_DIR/25pct-preflight.jsonl" <<PREFLIGHT_JSON
{"kind":"tier_preflight","tier":"25pct","retained_target_bytes":$retained_target_bytes,"estimated_rows":$estimated_rows,"generated_table_bytes":$generated_table_bytes,"wal_log_bytes":$wal_log_bytes,"report_bytes":$report_bytes,"required_disk_bytes":$required_bytes,"available_disk_bytes":$available_bytes,"disk_preflight":$(if [ "$available_bytes" -gt "$required_bytes" ]; then echo true; else echo false; fi),"run_preflight":"blocked","bounded_streaming_generator_probe":true,"postgresql_baseline_required":true,"chunked_resident_cache_install":false,"blocker":"missing_relational_resident_cache_chunked_install_api"}
PREFLIGHT_JSON
  cat "$OUT_DIR/25pct-preflight.md"
  echo "p8_ch_benchmark_25pct=blocked reason=missing_relational_resident_cache_chunked_install_api" >&2
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
  --pgsql-baseline-preflight)
    write_pgsql_baseline_preflight
    ;;
  --streaming-self-check)
    mkdir -p "$OUT_DIR"
    cargo run -q -p gpu_db_engine --example p8_ch_benchmark_residency_probe -- \
      --streaming-self-check \
      --output-dir "$OUT_DIR" \
      --rows "${GPU_DB_CH_BENCH_ROWS:-64}" \
      --chunk-rows "${GPU_DB_CH_BENCH_CHUNK_ROWS:-16}"
    test -s "$OUT_DIR/streaming-order-line/manifest.jsonl"
    test -s "$OUT_DIR/streaming-order-line/self-check.md"
    grep -q '"kind":"streaming_summary"' "$OUT_DIR/streaming-order-line/manifest.jsonl"
    grep -q '"blocker":"missing_relational_resident_cache_chunked_install_api"' "$OUT_DIR/streaming-order-line/manifest.jsonl"
    cat "$OUT_DIR/streaming-order-line/self-check.md"
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
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_ROWS=16 GPU_DB_CH_BENCH_CHUNK_ROWS=4 \
      "$0" --streaming-self-check >"$tmp_dir/streaming.out"
    if GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_ROWS=16 \
      "$0" --pgsql-baseline-preflight >"$tmp_dir/pgsql.out" 2>"$tmp_dir/pgsql.err"; then
      grep -q 'status: pass' "$tmp_dir/pgsql.out"
    else
      grep -Eq 'missing_pgsql_baseline_connection|missing_psql_client' "$tmp_dir/pgsql.err"
    fi
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" "$0" --cleanup >"$tmp_dir/cleanup.out"
    grep -q 'missing_relational_resident_cache_chunked_install_api' "$tmp_dir/streaming.out"
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
