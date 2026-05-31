#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

OUT_DIR="${GPU_DB_CH_BENCH_OUT_DIR:-target/p8-ch-benchmark-residency}"
ROWS="${GPU_DB_CH_BENCH_ROWS:-512}"
CONCURRENCY="${GPU_DB_CH_BENCH_CONCURRENCY:-1,10}"

usage() {
  cat <<'USAGE'
usage: scripts/run_p8_ch_benchmark_residency_probe.sh [--dry-run|--run-baseline|--run-25pct|--run-25pct-execute|--pgsql-baseline-preflight|--pgsql-baseline-25pct-latency|--pgsql-baseline-docker-up|--pgsql-baseline-docker-preflight|--pgsql-baseline-docker-down|--streaming-self-check|--chunked-install-self-check|--chunked-upload-self-check|--cleanup|--self-check]

Environment:
  GPU_DB_CH_BENCH_OUT_DIR       output directory, default target/p8-ch-benchmark-residency
  GPU_DB_CH_BENCH_ROWS          safe calibration rows, default 512
  GPU_DB_CH_BENCH_CONCURRENCY   logical request targets, default 1,10
  GPU_DB_CH_BENCH_MAX_ROWS      scheduled-run guardrail, default 10000
  GPU_DB_CH_BENCH_CHUNK_ROWS    streaming self-check chunk rows, default 16
  GPU_DB_CH_BENCH_EXECUTE_ROWS  guarded chunked execution rows, default 1024
  GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS  guarded execution chunk rows, default 256
  GPU_DB_CH_BENCH_ALLOW_FULL_25PCT  set to 1 to attempt all estimated 25pct rows
  GPU_DB_CH_BENCH_PGSQL_ROWS    PostgreSQL latency rows, default 1024 unless full guard is set
  GPU_DB_CH_BENCH_ALLOW_FULL_PGSQL_25PCT  set to 1 to load/query all estimated 25pct PostgreSQL rows
  GPU_DB_CH_BENCH_PGSQL_URL     libpq connection string for PostgreSQL baseline
  GPU_DB_CH_BENCH_PGSQL_DOCKER_NAME      default gpu-db-p8-pgsql-baseline-disposable
  GPU_DB_CH_BENCH_PGSQL_DOCKER_IMAGE     default postgres:16
  GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT      default 55434
  GPU_DB_CH_BENCH_PGSQL_DOCKER_PASSWORD  default gpu_db_p8_benchmark
USAGE
}

rows_25pct() {
  local retained_target_bytes=6442450944
  local retained_bytes_per_row=40
  echo $(((retained_target_bytes + retained_bytes_per_row - 1) / retained_bytes_per_row))
}

expected_amount_sum() {
  local rows="$1"
  local period=100000
  local full_periods=$((rows / period))
  local remainder=$((rows % period))
  local period_sum=4999950000
  local remainder_sum=0
  local id
  for ((id = 1; id <= remainder; id++)); do
    remainder_sum=$((remainder_sum + ((id * 17) % 100000)))
  done
  echo $((full_periods * period_sum + remainder_sum))
}

expected_quantity_between_avg() {
  local rows="$1"
  local period=50
  local full_periods=$((rows / period))
  local remainder=$((rows % period))
  local period_count=31
  local period_sum=775
  local count=$((full_periods * period_count))
  local sum=$((full_periods * period_sum))
  local id value
  for ((id = 1; id <= remainder; id++)); do
    value=$(((id % 50) + 1))
    if ((value >= 10 && value <= 40)); then
      count=$((count + 1))
      sum=$((sum + value))
    fi
  done
  local whole=$((sum / count))
  local rem=$((sum % count))
  local digits=()
  local digit idx carry
  for _ in $(seq 1 16); do
    rem=$((rem * 10))
    digit=$((rem / count))
    digits+=("$digit")
    rem=$((rem % count))
  done
  rem=$((rem * 10))
  digit=$((rem / count))
  if ((digit >= 5)); then
    carry=1
    for ((idx = 15; idx >= 0; idx--)); do
      if ((digits[idx] < 9)); then
        digits[idx]=$((digits[idx] + 1))
        carry=0
        break
      fi
      digits[idx]=0
    done
    if ((carry == 1)); then
      whole=$((whole + 1))
    fi
  fi
  local fractional=""
  for digit in "${digits[@]}"; do
    fractional+="$digit"
  done
  printf '%s.%s\n' "$whole" "$fractional"
}

expected_amount_max_filter() {
  local rows="$1"
  local lower="$2"
  local scan_rows="$rows"
  if ((scan_rows > 100000)); then
    scan_rows=100000
  fi
  local id value max=""
  for ((id = 1; id <= scan_rows; id++)); do
    value=$(((id * 17) % 100000))
    if ((value >= lower)) && { [ -z "$max" ] || ((value > max)); }; then
      max="$value"
    fi
  done
  if [ -z "$max" ]; then
    echo NULL
  else
    echo "$max"
  fi
}

write_pgsql_copy_stream() {
  local rows="$1"
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
    printf '%s,%s,%s,%s,%s%s\n' "$id" "$item" "$qty" "$amount" "$dist" "$bucket"
  done
}

pgsql_docker_name() {
  printf '%s\n' "${GPU_DB_CH_BENCH_PGSQL_DOCKER_NAME:-gpu-db-p8-pgsql-baseline-disposable}"
}

pgsql_docker_image() {
  printf '%s\n' "${GPU_DB_CH_BENCH_PGSQL_DOCKER_IMAGE:-postgres:16}"
}

pgsql_docker_port() {
  printf '%s\n' "${GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT:-55434}"
}

pgsql_docker_password() {
  printf '%s\n' "${GPU_DB_CH_BENCH_PGSQL_DOCKER_PASSWORD:-gpu_db_p8_benchmark}"
}

pgsql_docker_url() {
  printf 'postgresql://postgres:%s@127.0.0.1:%s/gpu_db_p8_baseline\n' "$(pgsql_docker_password)" "$(pgsql_docker_port)"
}

write_pgsql_docker_env() {
  mkdir -p "$OUT_DIR/pgsql-baseline"
  local env_path="$OUT_DIR/pgsql-baseline/docker.env"
  cat >"$env_path" <<ENV
GPU_DB_CH_BENCH_PGSQL_URL='$(pgsql_docker_url)'
GPU_DB_CH_BENCH_PGSQL_DOCKER_NAME='$(pgsql_docker_name)'
GPU_DB_CH_BENCH_PGSQL_DOCKER_IMAGE='$(pgsql_docker_image)'
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT='$(pgsql_docker_port)'
ENV
  chmod 600 "$env_path"
}

pgsql_baseline_docker_graceful_stop() {
  local name="$1"
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$name"; then
    return 0
  fi

  local pg_ctl
  pg_ctl="$(docker exec "$name" sh -lc 'command -v pg_ctl || find /usr/lib/postgresql -name pg_ctl -type f 2>/dev/null | head -1' 2>/dev/null || true)"
  if [ -z "$pg_ctl" ]; then
    return 1
  fi

  docker exec --user postgres "$name" sh -lc "'$pg_ctl' -D \"\$PGDATA\" -m fast -w stop" >/dev/null 2>&1
}

pgsql_baseline_docker_remove() {
  local name="$1"
  if ! docker ps -a --format '{{.Names}}' | grep -Fxq "$name"; then
    return 0
  fi

  pgsql_baseline_docker_graceful_stop "$name" || true
  docker rm "$name" >/dev/null 2>&1
}

pgsql_baseline_docker_up() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "p8_ch_benchmark_pgsql_docker=blocked reason=missing_docker_client" >&2
    return 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "p8_ch_benchmark_pgsql_docker=blocked reason=docker_daemon_unavailable" >&2
    return 1
  fi

  local name image port password
  name="$(pgsql_docker_name)"
  image="$(pgsql_docker_image)"
  port="$(pgsql_docker_port)"
  password="$(pgsql_docker_password)"

  if docker ps -a --format '{{.Names}}' | grep -Fxq "$name"; then
    if ! pgsql_baseline_docker_remove "$name"; then
      echo "p8_ch_benchmark_pgsql_docker=blocked reason=existing_container_cleanup_failed name=$name" >&2
      return 1
    fi
  fi
  docker run -d \
    --name "$name" \
    --label gpu-db.p8-benchmark=true \
    --label gpu-db.lifecycle=disposable \
    -e POSTGRES_PASSWORD="$password" \
    -e POSTGRES_DB=gpu_db_p8_baseline \
    -p "127.0.0.1:${port}:5432" \
    "$image" >/dev/null

  for _ in $(seq 1 60); do
    if pg_isready -h 127.0.0.1 -p "$port" -U postgres -d gpu_db_p8_baseline >/dev/null 2>&1; then
      write_pgsql_docker_env
      echo "p8_ch_benchmark_pgsql_docker=ready name=$name port=$port"
      return 0
    fi
    sleep 1
  done

  docker logs "$name" >&2 || true
  echo "p8_ch_benchmark_pgsql_docker=blocked reason=postgres_container_not_ready" >&2
  return 1
}

pgsql_baseline_docker_down() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "p8_ch_benchmark_pgsql_docker_cleanup=skipped reason=missing_docker_client"
    return 0
  fi
  local name
  name="$(pgsql_docker_name)"
  if ! pgsql_baseline_docker_remove "$name"; then
    echo "p8_ch_benchmark_pgsql_docker_cleanup=failed name=$name reason=docker_remove_failed" >&2
    return 1
  fi
  echo "p8_ch_benchmark_pgsql_docker_cleanup=passed name=$name"
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

write_pgsql_25pct_latency() {
  mkdir -p "$OUT_DIR/pgsql-latency"
  if ! command -v psql >/dev/null 2>&1; then
    echo "p8_ch_benchmark_pgsql_latency=blocked reason=missing_psql_client" >&2
    return 1
  fi
  if [ -z "${GPU_DB_CH_BENCH_PGSQL_URL:-}" ]; then
    echo "p8_ch_benchmark_pgsql_latency=blocked reason=missing_pgsql_baseline_connection" >&2
    return 1
  fi

  local estimated_rows rows row_tier status blocker full_command lower_bound
  estimated_rows="$(rows_25pct)"
  rows="${GPU_DB_CH_BENCH_PGSQL_ROWS:-1024}"
  row_tier="scaled"
  status="blocked"
  blocker="full_25pct_postgresql_latency_requires_operator_long_run"
  full_command="GPU_DB_CH_BENCH_PGSQL_URL='$(pgsql_docker_url)' GPU_DB_CH_BENCH_ALLOW_FULL_PGSQL_25PCT=1 GPU_DB_CH_BENCH_PGSQL_ROWS=$estimated_rows scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-25pct-latency"
  if [ "${GPU_DB_CH_BENCH_ALLOW_FULL_PGSQL_25PCT:-0}" = "1" ]; then
    rows="$estimated_rows"
    row_tier="25pct"
    status="pass"
    blocker="none"
  fi
  lower_bound=$((rows / 4))
  if [ "$lower_bound" -lt 1 ]; then
    lower_bound=1
  fi

  local expected_count expected_sum expected_avg expected_max
  expected_count="$rows"
  expected_sum="$(expected_amount_sum "$rows")"
  expected_avg="$(expected_quantity_between_avg "$rows")"
  expected_max="$(expected_amount_max_filter "$rows" "$lower_bound")"

  local out_path err_path validation_path report_path metrics_path timing_path
  out_path="$OUT_DIR/pgsql-latency/psql.out"
  err_path="$OUT_DIR/pgsql-latency/psql.err"
  validation_path="$OUT_DIR/pgsql-latency/validation.tsv"
  report_path="$OUT_DIR/pgsql-latency/latency.md"
  metrics_path="$OUT_DIR/pgsql-latency/metrics.jsonl"
  timing_path="$OUT_DIR/pgsql-latency/timing.tsv"
  rm -f "$out_path" "$err_path" "$validation_path" "$report_path" "$metrics_path" "$timing_path"

  local started ended total_runtime_ms
  started=$(date +%s%3N)
  {
    cat <<SQL
\\set ON_ERROR_STOP on
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
    write_pgsql_copy_stream "$rows"
    cat <<SQL
\\.
ANALYZE order_line;
\\pset tuples_only on
\\pset format unaligned
\\o $validation_path
SELECT 'order_line_count_all' || E'\\t' || COUNT(*)::text || E'\\t' || '$expected_count' FROM order_line;
SELECT 'order_line_sum_amount' || E'\\t' || COALESCE(SUM(ol_amount)::text, 'NULL') || E'\\t' || '$expected_sum' FROM order_line;
SELECT 'order_line_avg_quantity_between' || E'\\t' || COALESCE(TRUNC(AVG(ol_quantity), 16)::numeric(40,16)::text, 'NULL') || E'\\t' || '$expected_avg' FROM order_line WHERE ol_quantity BETWEEN 10 AND 40;
SELECT 'order_line_max_amount_filter' || E'\\t' || COALESCE(MAX(ol_amount)::text, 'NULL') || E'\\t' || '$expected_max' FROM order_line WHERE ol_amount >= $lower_bound;
\\o $OUT_DIR/pgsql-latency/order_line_count_all.explain.json
EXPLAIN (ANALYZE, FORMAT JSON) SELECT COUNT(*) FROM order_line;
\\o $OUT_DIR/pgsql-latency/order_line_sum_amount.explain.json
EXPLAIN (ANALYZE, FORMAT JSON) SELECT SUM(ol_amount) FROM order_line;
\\o $OUT_DIR/pgsql-latency/order_line_avg_quantity_between.explain.json
EXPLAIN (ANALYZE, FORMAT JSON) SELECT AVG(ol_quantity) FROM order_line WHERE ol_quantity BETWEEN 10 AND 40;
\\o $OUT_DIR/pgsql-latency/order_line_max_amount_filter.explain.json
EXPLAIN (ANALYZE, FORMAT JSON) SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= $lower_bound;
\\o
SQL
  } | psql "$GPU_DB_CH_BENCH_PGSQL_URL" -X >"$out_path" 2>"$err_path" || {
    cat >"$report_path" <<REPORT
# PostgreSQL 25% Latency Runner

- rows: $rows
- row_tier: $row_tier
- status: blocked
- blocker: pgsql_latency_workload_failed
- output: $out_path
- stderr: $err_path
- full_run_command: \`$full_command\`
REPORT
    cat "$report_path"
    echo "p8_ch_benchmark_pgsql_latency=blocked reason=pgsql_latency_workload_failed" >&2
    return 1
  }
  ended=$(date +%s%3N)
  total_runtime_ms=$((ended - started))

  local validation_status=pass
  while IFS=$'\t' read -r query actual expected; do
    if [ "$actual" != "$expected" ]; then
      validation_status=fail
    fi
  done <"$validation_path"

  local query explain execution_ms p50_us throughput
  : >"$metrics_path"
  : >"$timing_path"
  while IFS=$'\t' read -r query _actual _expected; do
    explain="$OUT_DIR/pgsql-latency/${query}.explain.json"
    execution_ms=$(awk -F': ' '/"Execution Time"/ {gsub(/[, ]/, "", $2); print $2; exit}' "$explain")
    if [ -z "$execution_ms" ]; then
      execution_ms=0
    fi
    p50_us=$(awk -v ms="$execution_ms" 'BEGIN { printf "%.0f", ms * 1000 }')
    if [ "$p50_us" = "0" ]; then
      throughput=0
    else
      throughput=$(awk -v us="$p50_us" 'BEGIN { printf "%.3f", 1000000 / us }')
    fi
    printf '%s\t%s\t%s\t%s\n' "$query" "$execution_ms" "$p50_us" "$throughput" >>"$timing_path"
    printf '{"kind":"pgsql_latency_metric","query":"%s","rows":%s,"row_tier":"%s","single_run":true,"p50_us":%s,"p95_us":%s,"p99_us":%s,"throughput_qps":%s,"total_runtime_ms":%s,"validation_status":"%s","artifact":"%s","explain_artifact":"%s"}\n' \
      "$query" "$rows" "$row_tier" "$p50_us" "$p50_us" "$p50_us" "$throughput" "$total_runtime_ms" "$validation_status" "$metrics_path" "$explain" >>"$metrics_path"
  done <"$validation_path"

  if [ "$validation_status" != pass ]; then
    status=blocked
    blocker=pgsql_latency_validation_failed
  fi

  cat >"$report_path" <<REPORT
# PostgreSQL 25% Latency Runner

- rows: $rows
- estimated_25pct_rows: $estimated_rows
- row_tier: $row_tier
- status: $status
- blocker: $blocker
- single_run_semantics: true
- total_runtime_ms: $total_runtime_ms
- psql_version: $(psql --version)
- validation: $validation_status
- validation_artifact: $validation_path
- metrics_artifact: $metrics_path
- output: $out_path
- stderr: $err_path
- full_run_command: \`$full_command\`
- cleanup_command: \`scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup && scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down\`

The runner streams deterministic \`order_line\` rows into PostgreSQL through
\`COPY FROM STDIN\` and does not write one enormous generated SQL file. Latency
is recorded from PostgreSQL \`EXPLAIN (ANALYZE, FORMAT JSON)\` output. Because
each query is executed once, p50/p95/p99 are intentionally the same single-run
latency value.

## Query Metrics

| workload/query id | row count / tier | side | p50 us | p95 us | p99 us | throughput qps | total runtime ms | validation | artifact |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
REPORT
  while IFS=$'\t' read -r query execution_ms p50_us throughput; do
    printf '| %s | %s / %s | PostgreSQL | %s | %s | %s | %s | %s | %s | `%s` |\n' \
      "$query" "$rows" "$row_tier" "$p50_us" "$p50_us" "$p50_us" "$throughput" "$total_runtime_ms" "$validation_status" "$metrics_path" >>"$report_path"
  done <"$timing_path"
  cat >>"$report_path" <<REPORT

## Validation

\`\`\`text
$(cat "$validation_path")
\`\`\`
REPORT
  cat "$report_path"

  if [ "$status" = pass ]; then
    echo "p8_ch_benchmark_pgsql_latency=passed rows=$rows"
    return 0
  fi
  echo "p8_ch_benchmark_pgsql_latency=blocked reason=$blocker rows=$rows" >&2
  return 1
}

write_25pct_preflight() {
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
  local disk_preflight
  if [ "$available_bytes" -gt "$required_bytes" ]; then
    disk_preflight=pass
  else
    disk_preflight=fail
  fi
  local baseline_preflight=blocked
  if [ -s "$OUT_DIR/pgsql-baseline/preflight.jsonl" ] && grep -q '"status":"pass"' "$OUT_DIR/pgsql-baseline/preflight.jsonl"; then
    baseline_preflight=pass
  fi
  local run_preflight=ready_to_start
  local blocker=null
  if [ "$disk_preflight" != pass ]; then
    run_preflight=blocked
    blocker='"insufficient_disk_for_25pct_tier"'
  elif [ "$baseline_preflight" != pass ]; then
    run_preflight=blocked
    blocker='"missing_passed_pgsql_baseline_preflight"'
  fi
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
- disk_preflight: $disk_preflight
- postgresql_baseline_preflight: $baseline_preflight
- run_preflight: $run_preflight
- bounded_streaming_generator_probe: available via \`--streaming-self-check\`
- postgresql_baseline_required: \`scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-preflight\`
- chunked_retained_device_memory_upload: available via \`--chunked-upload-self-check\`
- benchmark_chunked_resident_cache_admission: available via \`--chunked-install-self-check\`
- blocker: $(if [ "$blocker" = null ]; then echo none; else printf '%s' "$blocker" | tr -d '"'; fi)

The bounded generator, retained CUDA chunk upload, and benchmark-only
resident-cache admission prerequisites are now checked. This command is still a
preflight: it proves whether the 25% / 6 GiB long run is safe to start on this
host with a passed PostgreSQL comparator artifact; it does not by itself claim
completed GPU DB performance numbers.
PREFLIGHT
  cat >"$OUT_DIR/25pct-preflight.jsonl" <<PREFLIGHT_JSON
{"kind":"tier_preflight","tier":"25pct","retained_target_bytes":$retained_target_bytes,"estimated_rows":$estimated_rows,"generated_table_bytes":$generated_table_bytes,"wal_log_bytes":$wal_log_bytes,"report_bytes":$report_bytes,"required_disk_bytes":$required_bytes,"available_disk_bytes":$available_bytes,"disk_preflight":$(if [ "$disk_preflight" = pass ]; then echo true; else echo false; fi),"postgresql_baseline_preflight":"$baseline_preflight","run_preflight":"$run_preflight","bounded_streaming_generator_probe":true,"postgresql_baseline_required":true,"chunked_retained_device_memory_upload":true,"benchmark_chunked_resident_cache_admission":true,"blocker":$blocker}
PREFLIGHT_JSON
  cat "$OUT_DIR/25pct-preflight.md"
  if [ "$run_preflight" = ready_to_start ]; then
    echo "p8_ch_benchmark_25pct_preflight=passed status=ready_to_start"
    return 0
  fi
  echo "p8_ch_benchmark_25pct=blocked reason=$(printf '%s' "$blocker" | tr -d '\"')" >&2
  return 1
}

write_25pct_execution() {
  mkdir -p "$OUT_DIR"
  local retained_target_bytes=6442450944
  local retained_bytes_per_row=40
  local estimated_rows=$(((retained_target_bytes + retained_bytes_per_row - 1) / retained_bytes_per_row))
  local execute_rows="${GPU_DB_CH_BENCH_EXECUTE_ROWS:-1024}"
  local execute_chunk_rows="${GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS:-256}"
  local full_command="GPU_DB_CH_BENCH_ALLOW_FULL_25PCT=1 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=1048576 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute"

  pgsql_baseline_docker_up
  GPU_DB_CH_BENCH_PGSQL_URL="$(pgsql_docker_url)" write_pgsql_baseline_preflight
  write_25pct_preflight

  if [ "${GPU_DB_CH_BENCH_ALLOW_FULL_25PCT:-0}" = "1" ]; then
    execute_rows="$estimated_rows"
    execute_chunk_rows="${GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS:-1048576}"
  fi

  cargo run -q -p gpu_db_engine --example p8_ch_benchmark_residency_probe -- \
    --chunked-execute \
    --output-dir "$OUT_DIR" \
    --rows "$execute_rows" \
    --chunk-rows "$execute_chunk_rows" \
    --concurrency "$CONCURRENCY"
  test -s "$OUT_DIR/chunked-execute/execution.md"
  test -s "$OUT_DIR/chunked-execute/metrics.jsonl"
  grep -q '"kind":"chunked_metric"' "$OUT_DIR/chunked-execute/metrics.jsonl"
  grep -q '"resident_route_zero_h2d":true' "$OUT_DIR/chunked-execute/metrics.jsonl"
  if grep -q '"resident_route_accepted":false' "$OUT_DIR/chunked-execute/metrics.jsonl"; then
    echo "p8 CH-benCHmark chunked execution included a rejected resident route" >&2
    exit 1
  fi
  grep -q '"kind":"chunked_memory_pressure_probe"' "$OUT_DIR/chunked-execute/metrics.jsonl"

  local status blocker
  if [ "${GPU_DB_CH_BENCH_ALLOW_FULL_25PCT:-0}" = "1" ]; then
    status=completed
    blocker=none
  else
    status=blocked
    blocker=full_25pct_requires_operator_long_run_after_streaming_boundary
  fi

  local completion_note
  if [ "$status" = completed ]; then
    completion_note="With the full-tier guard enabled, this run executed the complete estimated 25% / 6 GiB GPU DB resident tier. A separate full-row PostgreSQL latency runner is still required before publishing speedup/regression ratios for the same row count."
  else
    completion_note="When not explicitly opted into the full 25% tier, this command stops after the guarded scaled execution. The resident upload/admission boundary now streams owned chunks through the engine/runtime path instead of retaining a full caller-owned chunk vector or text layout. The remaining full-tier blocker is run-window/capacity: attempting 161,061,274 rows in an unattended worker slice should be an explicit operator-approved long run. The exact full command is:"
  fi

  cat >"$OUT_DIR/25pct-execution.md" <<REPORT
# P8 CH-benCHmark 25% Chunked Execution

- tier: 25pct
- status: $status
- retained_target_bytes: $retained_target_bytes
- estimated_order_line_rows: $estimated_rows
- executed_rows: $execute_rows
- execute_chunk_rows: $execute_chunk_rows
- postgresql_baseline_artifact: $OUT_DIR/pgsql-baseline/preflight.md
- tier_preflight_artifact: $OUT_DIR/25pct-preflight.md
- gpu_db_execution_artifact: $OUT_DIR/chunked-execute/execution.md
- raw_metrics: $OUT_DIR/chunked-execute/metrics.jsonl
- cleanup_command: \`scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup && scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down\`
- blocker: $blocker

The checked execution path creates an empty \`order_line\` catalog table,
installs generated benchmark-only resident chunks without SQL-visible MVCC
inserts, executes the supported aggregate query set, validates deterministic
formula-backed answers, records p50/p95/p99/throughput/CUDA/H2D/D2H/zero-H2D route
metrics, and verifies memory-pressure fallback behavior.

$completion_note

\`\`\`bash
$full_command
\`\`\`
REPORT
  cat "$OUT_DIR/25pct-execution.md"
  if [ "$status" = completed ]; then
    echo "p8_ch_benchmark_25pct_execution=completed rows=$execute_rows"
  else
    echo "p8_ch_benchmark_25pct_execution=blocked reason=$blocker" >&2
    return 1
  fi
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
    grep -q '"tier":"125pct"' "$OUT_DIR/estimate.jsonl"
    if grep -Eq '"tier":"(50pct|100pct|200pct|400pct)"' "$OUT_DIR/estimate.jsonl"; then
      echo "retired 50/100/200/400% VRAM benchmark tiers should not be configured" >&2
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
    write_25pct_preflight
    ;;
  --run-25pct-execute)
    write_25pct_execution
    ;;
  --pgsql-baseline-preflight)
    write_pgsql_baseline_preflight
    ;;
  --pgsql-baseline-25pct-latency)
    write_pgsql_25pct_latency
    ;;
  --pgsql-baseline-docker-up)
    pgsql_baseline_docker_up
    ;;
  --pgsql-baseline-docker-preflight)
    pgsql_baseline_docker_up
    GPU_DB_CH_BENCH_PGSQL_URL="$(pgsql_docker_url)" write_pgsql_baseline_preflight
    ;;
  --pgsql-baseline-docker-down)
    pgsql_baseline_docker_down
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
    grep -q '"status":"pass"' "$OUT_DIR/streaming-order-line/manifest.jsonl"
    cat "$OUT_DIR/streaming-order-line/self-check.md"
    ;;
  --chunked-install-self-check)
    mkdir -p "$OUT_DIR"
    cargo run -q -p gpu_db_engine --example p8_ch_benchmark_residency_probe -- \
      --chunked-install-self-check \
      --output-dir "$OUT_DIR" \
      --rows "${GPU_DB_CH_BENCH_ROWS:-64}" \
      --chunk-rows "${GPU_DB_CH_BENCH_CHUNK_ROWS:-16}"
    test -s "$OUT_DIR/chunked-install-self-check/self-check.jsonl"
    test -s "$OUT_DIR/chunked-install-self-check/self-check.md"
    grep -q '"kind":"chunked_install_self_check"' "$OUT_DIR/chunked-install-self-check/self-check.jsonl"
    grep -q '"status":"pass"' "$OUT_DIR/chunked-install-self-check/self-check.jsonl"
    cat "$OUT_DIR/chunked-install-self-check/self-check.md"
    ;;
  --chunked-upload-self-check)
    mkdir -p "$OUT_DIR"
    cargo run -q -p gpu_db_engine --example p8_ch_benchmark_residency_probe -- \
      --chunked-upload-self-check \
      --output-dir "$OUT_DIR" \
      --rows "${GPU_DB_CH_BENCH_ROWS:-64}" \
      --chunk-rows "${GPU_DB_CH_BENCH_CHUNK_ROWS:-16}"
    test -s "$OUT_DIR/chunked-upload-self-check/self-check.jsonl"
    test -s "$OUT_DIR/chunked-upload-self-check/self-check.md"
    grep -q '"kind":"chunked_upload_self_check"' "$OUT_DIR/chunked-upload-self-check/self-check.jsonl"
    grep -q '"status":"pass"' "$OUT_DIR/chunked-upload-self-check/self-check.jsonl"
    cat "$OUT_DIR/chunked-upload-self-check/self-check.md"
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
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_ROWS=16 GPU_DB_CH_BENCH_CHUNK_ROWS=4 \
      "$0" --chunked-install-self-check >"$tmp_dir/chunked-install.out"
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_ROWS=16 GPU_DB_CH_BENCH_CHUNK_ROWS=4 \
      "$0" --chunked-upload-self-check >"$tmp_dir/chunked-upload.out"
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_EXECUTE_ROWS=16 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=4 GPU_DB_CH_BENCH_CONCURRENCY=1 \
      "$0" --run-25pct-execute >"$tmp_dir/chunked-execute.out" 2>"$tmp_dir/chunked-execute.err" || true
    grep -q 'full_25pct_requires_operator_long_run_after_streaming_boundary' "$tmp_dir/chunked-execute.err"
    grep -q 'expected_results: deterministic formulas' "$tmp_dir/out/chunked-execute/execution.md"
    if GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_ROWS=16 \
      "$0" --pgsql-baseline-preflight >"$tmp_dir/pgsql.out" 2>"$tmp_dir/pgsql.err"; then
      grep -q 'status: pass' "$tmp_dir/pgsql.out"
    else
      grep -Eq 'missing_pgsql_baseline_connection|missing_psql_client' "$tmp_dir/pgsql.err"
    fi
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" "$0" --cleanup >"$tmp_dir/cleanup.out"
    grep -q 'chunked_resident_cache_install_available: true' "$tmp_dir/streaming.out"
    grep -q 'benchmark_chunked_resident_cache_admission: pass' "$tmp_dir/chunked-install.out"
    grep -q 'chunked_retained_device_memory_upload: pass' "$tmp_dir/chunked-upload.out"
    grep -q 'p8_ch_benchmark_cleanup=passed' "$tmp_dir/cleanup.out"
    "$0" --pgsql-baseline-docker-down >"$tmp_dir/docker-down.out"
    grep -q 'p8_ch_benchmark_pgsql_docker_cleanup=passed' "$tmp_dir/docker-down.out"
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
