#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

OUT_DIR="${GPU_DB_CH_BENCH_OUT_DIR:-target/p8-ch-benchmark-residency}"
ROWS="${GPU_DB_CH_BENCH_ROWS:-512}"
CONCURRENCY="${GPU_DB_CH_BENCH_CONCURRENCY:-1,10}"

usage() {
  cat <<'USAGE'
usage: scripts/run_p8_ch_benchmark_residency_probe.sh [--dry-run|--run-baseline|--run-25pct|--run-25pct-execute|--run-125pct|--pgsql-fairness-audit|--gpu-db-protocol-benchmark-smoke|--engine-backed-pgwire-benchmark-smoke|--protocol-retained-route-bridge-report|--engine-backed-protocol-boundary-probe|--pgsql-baseline-preflight|--pgsql-baseline-25pct-latency|--pgsql-baseline-125pct-latency|--pgsql-baseline-docker-up|--pgsql-baseline-docker-preflight|--pgsql-baseline-docker-down|--streaming-self-check|--chunked-install-self-check|--chunked-upload-self-check|--cleanup|--self-check]

Environment:
  GPU_DB_CH_BENCH_OUT_DIR       output directory, default target/p8-ch-benchmark-residency
  GPU_DB_CH_BENCH_ROWS          safe calibration rows, default 512
  GPU_DB_CH_BENCH_CONCURRENCY   logical request targets, default 1,10
  GPU_DB_CH_BENCH_MAX_ROWS      scheduled-run guardrail, default 10000
  GPU_DB_CH_BENCH_CHUNK_ROWS    streaming self-check chunk rows, default 16
  GPU_DB_CH_BENCH_EXECUTE_ROWS  guarded chunked execution rows, default 1024
  GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS  guarded execution chunk rows, default 256
  GPU_DB_CH_BENCH_ACCEPT_SCALED_25PCT  set to 1 to treat guarded scaled --run-25pct-execute evidence as success
  GPU_DB_CH_BENCH_ALLOW_FULL_25PCT  set to 1 to attempt all estimated 25pct rows
  GPU_DB_CH_BENCH_ACCEPT_SCALED_125PCT set to 1 to treat guarded scaled --run-125pct evidence as success
  GPU_DB_CH_BENCH_ALLOW_FULL_125PCT set to 1 to permit a future full 125pct attempt after readiness is safe
  GPU_DB_CH_BENCH_PGSQL_ROWS    PostgreSQL latency rows, default 1024 unless full guard is set
  GPU_DB_CH_BENCH_PGSQL_AUDIT_ROWS    scaled fairness audit rows, default 2048
  GPU_DB_CH_BENCH_PGSQL_AUDIT_REPEATS repeated warm timings per query/profile, default 3
  GPU_DB_CH_BENCH_GPU_DB_PROTOCOL_ROWS scaled GPU DB protocol smoke rows, default 64
  GPU_DB_CH_BENCH_GPU_DB_PROTOCOL_PORT GPU DB protocol smoke listen port, default 55435
  GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS scaled engine-backed pgwire smoke rows, default 64
  GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT engine-backed pgwire smoke listen port, default 55437
  GPU_DB_CH_BENCH_PROTOCOL_BRIDGE_ROWS scaled bridge blocker rows, default 64
  GPU_DB_CH_BENCH_ENGINE_PROTOCOL_BOUNDARY_ROWS scaled boundary rows, default 64
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

rows_125pct() {
  local retained_target_bytes=32212254720
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

json_escape() {
  sed 's/\\/\\\\/g; s/"/\\"/g' <<<"$1"
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

write_pgsql_125pct_latency() {
  mkdir -p "$OUT_DIR/pgsql-125pct-latency"
  local estimated_rows
  estimated_rows="$(rows_125pct)"
  local report_path="$OUT_DIR/pgsql-125pct-latency/latency.md"
  local json_path="$OUT_DIR/pgsql-125pct-latency/status.jsonl"
  local full_command="GPU_DB_CH_BENCH_PGSQL_URL='$(pgsql_docker_url)' GPU_DB_CH_BENCH_ALLOW_FULL_PGSQL_125PCT=1 GPU_DB_CH_BENCH_PGSQL_ROWS=$estimated_rows scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-125pct-latency"

  cat >"$report_path" <<REPORT
# PostgreSQL 125% Latency Runner

- rows: $estimated_rows
- row_tier: 125pct
- status: blocked
- blocker: full_125pct_postgresql_latency_requires_operator_long_run_and_over_resident_gpu_path
- full_run_command: \`$full_command\`

The 125% PostgreSQL comparator is intentionally not executed by default. The
same-row-count PostgreSQL load would stream 805,306,368 deterministic rows and
only becomes useful after the GPU side has a partitioned or streamed
over-resident execution path to compare against.
REPORT
  printf '{"kind":"pgsql_latency_readiness","tier":"125pct","rows":%s,"status":"blocked","blocker":"full_125pct_postgresql_latency_requires_operator_long_run_and_over_resident_gpu_path"}\n' "$estimated_rows" >"$json_path"
  cat "$report_path"
  echo "p8_ch_benchmark_pgsql_125pct_latency=blocked reason=full_125pct_postgresql_latency_requires_operator_long_run_and_over_resident_gpu_path" >&2
  return 1
}

write_pgsql_fairness_audit() {
  mkdir -p "$OUT_DIR/pgsql-fairness-audit"
  local audit_dir="$OUT_DIR/pgsql-fairness-audit"
  local rows="${GPU_DB_CH_BENCH_PGSQL_AUDIT_ROWS:-2048}"
  local repeats="${GPU_DB_CH_BENCH_PGSQL_AUDIT_REPEATS:-3}"
  local report_path="$audit_dir/fairness-audit.md"
  local metrics_path="$audit_dir/metrics.jsonl"
  local settings_path="$audit_dir/postgresql-settings.tsv"
  local host_path="$audit_dir/host-facts.txt"
  local concurrency_path="$audit_dir/concurrency-curve-plan.csv"
  local blocker="gpu_db_protocol_benchmark_path_required"
  local status="blocked"
  local lower_bound=$((rows / 4))
  if [ "$lower_bound" -lt 1 ]; then
    lower_bound=1
  fi

  : >"$metrics_path"
  {
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname=$(uname -a)"
    command -v lscpu >/dev/null 2>&1 && lscpu || true
    command -v free >/dev/null 2>&1 && free -h || true
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true
    command -v docker >/dev/null 2>&1 && docker version --format 'docker_client={{.Client.Version}} docker_server={{.Server.Version}}' 2>/dev/null || true
  } >"$host_path"

  cat >"$concurrency_path" <<CSV
tier,profile,query,concurrency,status,blocker,metric_schema
CSV
  local profile query concurrency
  for profile in "default_postgresql" "tuned_postgresql" "gpu_db_retained_resident_path"; do
    for query in order_line_count_all order_line_sum_amount order_line_avg_quantity_between order_line_max_amount_filter; do
      for concurrency in 1 2 4 8 16 32 64 128; do
        printf '25pct,%s,%s,%s,blocked,%s,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"\n' \
          "$profile" "$query" "$concurrency" "$blocker" >>"$concurrency_path"
      done
    done
  done

  if ! command -v psql >/dev/null 2>&1; then
    cat >"$report_path" <<REPORT
# P8 PostgreSQL Fairness Audit

- rows: $rows
- repeats: $repeats
- status: blocked
- blocker: missing_psql_client
- host_facts: $host_path
- concurrency_plan: $concurrency_path

The fairness audit could not collect PostgreSQL evidence because the local
\`psql\` client is missing.
REPORT
    cat "$report_path"
    echo "p8_ch_benchmark_pgsql_fairness_audit=blocked reason=missing_psql_client" >&2
    return 0
  fi

  pgsql_baseline_docker_up
  local pgurl
  pgurl="$(pgsql_docker_url)"
  local docker_name docker_image
  docker_name="$(pgsql_docker_name)"
  docker_image="$(pgsql_docker_image)"

  psql "$pgurl" -X -v ON_ERROR_STOP=1 -Atc "SELECT name || E'\t' || setting FROM pg_settings WHERE name IN ('shared_buffers','work_mem','maintenance_work_mem','effective_cache_size','max_parallel_workers_per_gather','jit','max_parallel_workers','max_worker_processes','max_parallel_maintenance_workers') ORDER BY name" >"$settings_path"

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
SQL
  } | psql "$pgurl" -X >"$audit_dir/load.out" 2>"$audit_dir/load.err"

  run_pgsql_audit_profile "$pgurl" "$audit_dir" "$metrics_path" default_postgresql "$rows" "$repeats" "$lower_bound"
  psql "$pgurl" -X -v ON_ERROR_STOP=1 >"$audit_dir/tuned-ddl.out" 2>"$audit_dir/tuned-ddl.err" <<SQL
CREATE INDEX IF NOT EXISTS order_line_quantity_btree ON order_line (ol_quantity);
CREATE INDEX IF NOT EXISTS order_line_amount_btree ON order_line (ol_amount);
CREATE INDEX IF NOT EXISTS order_line_quantity_brin ON order_line USING brin (ol_quantity);
CREATE INDEX IF NOT EXISTS order_line_amount_brin ON order_line USING brin (ol_amount);
ANALYZE order_line;
SQL
  run_pgsql_audit_profile "$pgurl" "$audit_dir" "$metrics_path" tuned_postgresql "$rows" "$repeats" "$lower_bound"

  cat >>"$metrics_path" <<JSON
{"kind":"fairness_blocker","tier":"25pct","status":"blocked","blocker":"$blocker","reason":"GPU DB retained benchmark evidence currently uses Engine::new_local()/execute_relational_select rather than the same PostgreSQL-compatible client/protocol benchmark path used for PostgreSQL."}
{"kind":"concurrency_blocker","tier":"25pct","status":"blocked","blocker":"identical_pg_client_harness_required","concurrency_targets":[1,2,4,8,16,32,64,128],"artifact":"$concurrency_path"}
{"kind":"tier_blocker","tier":"125pct","status":"blocked","blocker":"missing_partitioned_over_resident_execution"}
JSON

  cat >"$report_path" <<REPORT
# P8 PostgreSQL Fairness Audit

- rows: $rows
- repeats: $repeats
- status: $status
- blocker: $blocker
- postgresql_profile_status: scaled_default_and_tuned_evidence_collected
- docker_image: $docker_image
- docker_name: $docker_name
- psql_version: $(psql --version)
- host_facts: $host_path
- postgresql_settings: $settings_path
- metrics_artifact: $metrics_path
- concurrency_plan: $concurrency_path
- default_plan_dir: $audit_dir/default_postgresql
- tuned_plan_dir: $audit_dir/tuned_postgresql

## Decision

The audit command now records scaled default and tuned PostgreSQL evidence, but
it does not admit a headline GPU DB-vs-PostgreSQL product comparison. The GPU DB
retained 25% benchmark evidence still enters through the Rust example and direct
engine calls rather than the same PostgreSQL-compatible benchmark client and
protocol stack used for PostgreSQL. Until a GPU DB protocol benchmark target is
available, retained GPU timings must be labeled \`engine_internal\`.

## Tuned PostgreSQL Profile

The scaled tuned comparator adds btree and BRIN indexes on \`ol_quantity\` and
\`ol_amount\`, then runs the same aggregate query shapes with repeated
\`EXPLAIN (ANALYZE, FORMAT JSON)\` samples. This is a smoke-sized audit, not a
full 161,061,274-row tuned PostgreSQL admission run.

## Concurrency Gate

True concurrent-client curves are blocked on the same protocol-parity gap. The
graph-ready plan enumerates targets \`1,2,4,8,16,32,64,128\` for
\`default PostgreSQL\`, \`tuned PostgreSQL\`, and \`GPU DB retained resident
path\`, with the required metric schema for wall-clock throughput, p50/p95/p99
latency, errors, correctness, and saturation notes.

## Required Follow-Up Command Shape

\`\`\`bash
# after a PostgreSQL-compatible GPU DB benchmark endpoint/target exists:
scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-fairness-audit
\`\`\`

The full 25% tuned PostgreSQL audit remains a separate operator-approved
long-run decision. The 125% tier remains blocked on
\`missing_partitioned_over_resident_execution\`.
REPORT
  cat "$report_path"
  echo "p8_ch_benchmark_pgsql_fairness_audit=blocked reason=$blocker artifact=$report_path"
  return 0
}

run_pgsql_audit_profile() {
  local pgurl="$1"
  local audit_dir="$2"
  local metrics_path="$3"
  local profile="$4"
  local rows="$5"
  local repeats="$6"
  local lower_bound="$7"
  local profile_dir="$audit_dir/$profile"
  mkdir -p "$profile_dir"

  local queries=(
    "order_line_count_all|SELECT COUNT(*) FROM order_line"
    "order_line_sum_amount|SELECT SUM(ol_amount) FROM order_line"
    "order_line_avg_quantity_between|SELECT AVG(ol_quantity) FROM order_line WHERE ol_quantity BETWEEN 10 AND 40"
    "order_line_max_amount_filter|SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= $lower_bound"
  )
  local entry query_name sql repeat explain execution_ms latency_us
  for entry in "${queries[@]}"; do
    query_name="${entry%%|*}"
    sql="${entry#*|}"
    for repeat in $(seq 1 "$repeats"); do
      explain="$profile_dir/${query_name}.repeat_${repeat}.explain.json"
      psql "$pgurl" -X -v ON_ERROR_STOP=1 -Atc "EXPLAIN (ANALYZE, FORMAT JSON) $sql" >"$explain"
      execution_ms=$(awk -F': ' '/"Execution Time"/ {gsub(/[, ]/, "", $2); print $2; exit}' "$explain")
      if [ -z "$execution_ms" ]; then
        execution_ms=0
      fi
      latency_us=$(awk -v ms="$execution_ms" 'BEGIN { printf "%.0f", ms * 1000 }')
      printf '{"kind":"pgsql_fairness_sample","profile":"%s","query":"%s","rows":%s,"repeat":%s,"latency_us":%s,"explain_artifact":"%s"}\n' \
        "$profile" "$query_name" "$rows" "$repeat" "$latency_us" "$explain" >>"$metrics_path"
    done
  done
}

gpu_db_protocol_query_metrics() {
  local url="$1"
  local metrics_path="$2"
  local query_id="$3"
  local route_classification="$4"
  local expected="$5"
  local sql="$6"
  local tmp_prefix="$7"
  local out_path="${tmp_prefix}-${query_id}.out"
  local err_path="${tmp_prefix}-${query_id}.err"
  local start_ns end_ns latency_us actual status error_count

  start_ns=$(date +%s%N)
  if psql "$url" -X -v ON_ERROR_STOP=1 -Atc "$sql" >"$out_path" 2>"$err_path"; then
    status="pass"
    error_count=0
  else
    status="error"
    error_count=1
  fi
  end_ns=$(date +%s%N)
  latency_us=$(((end_ns - start_ns) / 1000))
  actual="$(tr '\n' '|' <"$out_path" | sed 's/|$//')"
  if [ "$status" = "pass" ] && [ "$actual" != "$expected" ]; then
    status="wrong_result"
    error_count=1
  fi
  printf '{"kind":"gpu_db_protocol_smoke_metric","target":"gpu_db_protocol_endpoint","client_driver":"psql/libpq","query":"%s","concurrency":1,"p50_us":%s,"p95_us":%s,"p99_us":%s,"throughput_qps":%.6f,"error_count":%s,"correctness_status":"%s","route_classification":"%s","retained_gpu_route":false,"protocol_catalog_path":true,"expected":"%s","actual":"%s"}\n' \
    "$query_id" \
    "$latency_us" \
    "$latency_us" \
    "$latency_us" \
    "$(awk -v us="$latency_us" 'BEGIN { if (us > 0) printf "%.6f", 1000000 / us; else printf "0.000000" }')" \
    "$error_count" \
    "$status" \
    "$route_classification" \
    "$(json_escape "$expected")" \
    "$(json_escape "$actual")" >>"$metrics_path"
}

write_gpu_db_protocol_benchmark_smoke() {
  mkdir -p "$OUT_DIR/gpu-db-protocol-benchmark-smoke"
  local smoke_dir="$OUT_DIR/gpu-db-protocol-benchmark-smoke"
  local rows="${GPU_DB_CH_BENCH_GPU_DB_PROTOCOL_ROWS:-64}"
  local port="${GPU_DB_CH_BENCH_GPU_DB_PROTOCOL_PORT:-55435}"
  local listen="127.0.0.1:$port"
  local url="postgresql://postgres@127.0.0.1:$port/postgres?sslmode=disable"
  local report_path="$smoke_dir/protocol-benchmark-smoke.md"
  local metrics_path="$smoke_dir/metrics.jsonl"
  local load_path="$smoke_dir/load.sql"
  local server_log="$smoke_dir/gpu-db-server.log"
  local startup_facts="$smoke_dir/startup-facts.txt"
  local concurrency_path="$smoke_dir/concurrency-curve-plan.csv"
  local blocker="protocol_endpoint_uses_protocol_catalog_not_p8_resident_engine"
  : >"$metrics_path"

  if ! command -v psql >/dev/null 2>&1; then
    cat >"$report_path" <<REPORT
# P8 GPU DB Protocol Benchmark Smoke

- status: blocked
- blocker: missing_psql_client

The GPU DB protocol benchmark smoke requires the PostgreSQL \`psql\` client.
REPORT
    cat "$report_path"
    return 0
  fi

  cargo build -q -p gpu_db_protocol --bin gpu-db-server
  target/debug/gpu-db-server --listen "$listen" --shared-catalog >"$server_log" 2>&1 &
  local server_pid=$!
  trap 'kill "$server_pid" >/dev/null 2>&1 || true; wait "$server_pid" >/dev/null 2>&1 || true' RETURN

  local ready=0
  for _ in $(seq 1 120); do
    if psql "$url" -X -v ON_ERROR_STOP=1 -Atc "SELECT 1 AS one" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  if [ "$ready" -ne 1 ]; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
    trap - RETURN
    cat >"$report_path" <<REPORT
# P8 GPU DB Protocol Benchmark Smoke

- status: blocked
- blocker: gpu_db_protocol_endpoint_startup_failed
- command: \`target/debug/gpu-db-server --listen $listen --shared-catalog\`
- log: $server_log
REPORT
    cat "$report_path"
    return 0
  fi

  {
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "command=target/debug/gpu-db-server --listen $listen --shared-catalog"
    echo "host=127.0.0.1"
    echo "port=$port"
    echo "database=postgres"
    echo "user=postgres"
    echo "auth_profile=local-dev trust-style startup"
    echo "tls_profile=sslmode=disable"
    echo "client_driver=psql/libpq"
    echo "psql_version=$(psql --version)"
    echo "server_version=16.0"
  } >"$startup_facts"

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
SQL
  } >"$load_path"
  psql "$url" -X -f "$load_path" >"$smoke_dir/load.out" 2>"$smoke_dir/load.err"

  local lower_bound
  lower_bound=$((rows / 4))
  if [ "$lower_bound" -lt 1 ]; then
    lower_bound=1
  fi
  local expected_count expected_sum expected_avg expected_max lookup_key lookup_item lookup_qty lookup_amount lookup_dist composite_key composite_item composite_qty composite_amount composite_dist
  expected_count="$rows"
  expected_sum="$(expected_amount_sum "$rows")"
  expected_avg="$(expected_quantity_between_avg "$rows")"
  expected_max="$(expected_amount_max_filter "$rows" "$lower_bound")"
  lookup_key=$(((rows + 1) / 2))
  lookup_item=$(((lookup_key % 100000) + 1))
  lookup_qty=$(((lookup_key % 50) + 1))
  lookup_amount=$(((lookup_key * 17) % 100000))
  lookup_dist="$(order_line_dist_info_shell "$lookup_key")"
  composite_key="$lookup_key"
  composite_item="$lookup_item"
  composite_qty="$lookup_qty"
  composite_amount="$lookup_amount"
  composite_dist="$lookup_dist"

  local route_classification="protocol_shared_catalog_cpu_scan"
  local tmp_prefix="$smoke_dir/query"
  gpu_db_protocol_query_metrics "$url" "$metrics_path" order_line_count_all "$route_classification" "$expected_count" "SELECT COUNT(*) FROM order_line" "$tmp_prefix"
  gpu_db_protocol_query_metrics "$url" "$metrics_path" order_line_sum_amount "$route_classification" "$expected_sum" "SELECT SUM(ol_amount) FROM order_line" "$tmp_prefix"
  gpu_db_protocol_query_metrics "$url" "$metrics_path" order_line_avg_quantity_between "$route_classification" "$expected_avg" "SELECT AVG(ol_quantity) FROM order_line WHERE ol_quantity BETWEEN 10 AND 40" "$tmp_prefix"
  gpu_db_protocol_query_metrics "$url" "$metrics_path" order_line_max_amount_filter "$route_classification" "$expected_max" "SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= $lower_bound" "$tmp_prefix"
  gpu_db_protocol_query_metrics "$url" "$metrics_path" order_line_lookup_ol_o_id "$route_classification" "${lookup_key}|${lookup_item}|${lookup_qty}|${lookup_amount}" "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = $lookup_key" "$tmp_prefix"
  gpu_db_protocol_query_metrics "$url" "$metrics_path" order_line_lookup_composite "$route_classification" "${composite_key}|${composite_item}|${composite_qty}|${composite_amount}|${composite_dist}" "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = $composite_key AND ol_i_id = $composite_item" "$tmp_prefix"

  cat >"$concurrency_path" <<CSV
tier,target,client_driver,query,concurrency,status,blocker,route_classification,metric_schema
25pct,gpu_db_protocol_endpoint,psql/libpq,order_line_count_all,1,scaled_smoke,$blocker,protocol_shared_catalog_cpu_scan,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"
25pct,gpu_db_protocol_endpoint,psql/libpq,order_line_sum_amount,1,scaled_smoke,$blocker,protocol_shared_catalog_cpu_scan,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"
25pct,gpu_db_protocol_endpoint,psql/libpq,order_line_avg_quantity_between,1,scaled_smoke,$blocker,protocol_shared_catalog_cpu_scan,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"
25pct,gpu_db_protocol_endpoint,psql/libpq,order_line_max_amount_filter,1,scaled_smoke,$blocker,protocol_shared_catalog_cpu_scan,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"
25pct,gpu_db_protocol_endpoint,psql/libpq,order_line_lookup_ol_o_id,1,scaled_smoke,$blocker,protocol_shared_catalog_cpu_scan,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"
25pct,gpu_db_protocol_endpoint,psql/libpq,order_line_lookup_composite,1,scaled_smoke,$blocker,protocol_shared_catalog_cpu_scan,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"
CSV
  local query concurrency
  for query in order_line_count_all order_line_sum_amount order_line_avg_quantity_between order_line_max_amount_filter order_line_lookup_ol_o_id order_line_lookup_composite; do
    for concurrency in 2 4 8 16 32 64 128; do
      printf '25pct,gpu_db_protocol_endpoint,psql/libpq,%s,%s,blocked,true_concurrency_pg_client_runner_required,protocol_shared_catalog_cpu_scan,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"\n' \
        "$query" "$concurrency" >>"$concurrency_path"
    done
  done
  cat >>"$metrics_path" <<JSON
{"kind":"gpu_db_protocol_benchmark_blocker","tier":"25pct","status":"blocked","blocker":"$blocker","secondary_blocker":"gpu_db_protocol_seed_to_resident_cache_required","client_driver":"psql/libpq","endpoint_seed_path":"CREATE TABLE plus COPY FROM STDIN","retained_gpu_route":false,"reason":"gpu-db-server owns protocol-visible SharedCatalog/Table rows and does not expose a warm/admit path into Engine RelationalResidentCache for P8 retained-resident execution."}
{"kind":"gpu_db_protocol_concurrency_blocker","tier":"25pct","status":"blocked","blocker":"true_concurrency_pg_client_runner_required","available_scaled_concurrency":[1],"required_concurrency":[1,2,4,8,16,32,64,128],"artifact":"$concurrency_path"}
JSON

  cat >"$report_path" <<REPORT
# P8 GPU DB Protocol Benchmark Smoke

- rows: $rows
- status: blocked
- blocker: $blocker
- secondary_blocker: gpu_db_protocol_seed_to_resident_cache_required
- concurrency_blocker: true_concurrency_pg_client_runner_required
- startup_facts: $startup_facts
- metrics_artifact: $metrics_path
- concurrency_plan: $concurrency_path
- server_log: $server_log

## Result

The GPU DB PostgreSQL-compatible endpoint can be started and driven through the
same PostgreSQL client family used by the PostgreSQL comparator
(\`psql\`/libpq). This smoke seeds \`order_line\` through protocol-visible
\`CREATE TABLE\` plus \`COPY FROM STDIN\`, then runs the current aggregate
shapes and two key-equality lookup shapes through the protocol endpoint.

This does not close the P8 headline benchmark gap. Code inspection shows
\`crates/protocol/src/bin/gpu-db-server.rs\` stores protocol-visible rows in
\`Session\` / \`SharedCatalog\` tables and answers \`SELECT\` through
\`execute_select_result(...)\`. The checked P8 retained benchmark path still
uses \`Engine::new_local()\`, benchmark-only resident chunk admission, and
\`execute_relational_select(...)\` in
\`crates/engine/examples/p8_ch_benchmark_residency_probe.rs\`. There is no
checked protocol-visible warm/admit path that moves \`COPY\`-loaded
\`order_line\` rows into the retained \`RelationalResidentCache\` route.

## Endpoint Facts

- command: \`target/debug/gpu-db-server --listen $listen --shared-catalog\`
- host_port: $listen
- database: postgres
- user: postgres
- auth_tls_profile: local-dev trust-style startup, sslmode=disable
- server_version_string: captured in $startup_facts
- cleanup: server process killed by the smoke trap after report generation

## Route Classification

- aggregate queries: protocol_shared_catalog_cpu_scan
- key-equality lookup queries: protocol_shared_catalog_cpu_scan
- index_metadata_lookup: false
- retained_gpu_route: false
- protocol_catalog_scan: true

## Lookup Coverage

- single-key lookup: \`SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = \$1\` represented with a scaled literal value through \`psql\`
- composite lookup: \`SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = \$1 AND ol_i_id = \$2\` represented with scaled literal values through \`psql\`

## Decision

The next unblock trigger is an integration/design slice that lets the
PostgreSQL-compatible endpoint either use the P8 \`Engine\` retained-residency
machinery directly or admit protocol-visible table state into the retained
resident cache before benchmarking. Until then, GPU DB product latency and
concurrency claims must remain blocked by
\`protocol_endpoint_uses_protocol_catalog_not_p8_resident_engine\`.
REPORT
  kill "$server_pid" >/dev/null 2>&1 || true
  wait "$server_pid" >/dev/null 2>&1 || true
  trap - RETURN
  cat "$report_path"
  echo "p8_ch_benchmark_gpu_db_protocol_smoke=blocked reason=$blocker artifact=$report_path"
}

engine_pgwire_query_metrics() {
  local url="$1"
  local metrics_path="$2"
  local query_id="$3"
  local expected="$4"
  local sql="$5"
  local tmp_prefix="$6"
  local route_classification="$7"
  local retained_route="$8"
  local blocker="$9"
  local out_path="${tmp_prefix}-${query_id}.out"
  local err_path="${tmp_prefix}-${query_id}.err"
  local start_ns end_ns latency_us actual status error_count

  start_ns=$(date +%s%N)
  if psql "$url" -X -v ON_ERROR_STOP=1 -Atc "$sql" >"$out_path" 2>"$err_path"; then
    status="pass"
    error_count=0
  else
    status="error"
    error_count=1
  fi
  end_ns=$(date +%s%N)
  latency_us=$(((end_ns - start_ns) / 1000))
  actual="$(tr '\n' '|' <"$out_path" | sed 's/|$//')"
  if [ "$status" = "pass" ] && [ "$actual" != "$expected" ]; then
    status="wrong_result"
    error_count=1
  fi
  printf '{"kind":"engine_backed_pgwire_smoke_metric","target":"engine_backed_pgwire_endpoint","client_driver":"psql/libpq","query":"%s","concurrency":1,"p50_us":%s,"p95_us":%s,"p99_us":%s,"throughput_qps":%.6f,"error_count":%s,"correctness_status":"%s","route_classification":"%s","retained_gpu_route":%s,"protocol_catalog_path":false,"blocker":"%s","expected":"%s","actual":"%s"}\n' \
    "$query_id" \
    "$latency_us" \
    "$latency_us" \
    "$latency_us" \
    "$(awk -v us="$latency_us" 'BEGIN { if (us > 0) printf "%.6f", 1000000 / us; else printf "0.000000" }')" \
    "$error_count" \
    "$status" \
    "$route_classification" \
    "$retained_route" \
    "$blocker" \
    "$(json_escape "$expected")" \
    "$(json_escape "$actual")" >>"$metrics_path"
}

write_engine_backed_pgwire_benchmark_smoke() {
  mkdir -p "$OUT_DIR/engine-backed-pgwire-benchmark-smoke"
  local smoke_dir="$OUT_DIR/engine-backed-pgwire-benchmark-smoke"
  local rows="${GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS:-64}"
  local port="${GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT:-55437}"
  local listen="127.0.0.1:$port"
  local url="postgresql://postgres@127.0.0.1:$port/postgres?sslmode=disable"
  local report_path="$smoke_dir/engine-backed-pgwire-benchmark-smoke.md"
  local metrics_path="$smoke_dir/metrics.jsonl"
  local load_path="$smoke_dir/load.sql"
  local server_log="$smoke_dir/engine-pgwire-endpoint.log"
  local facts_path="$smoke_dir/endpoint-facts.txt"
  local concurrency_path="$smoke_dir/concurrency-curve-plan.csv"
  local lookup_blocker="retained_multi_column_projection_required"
  : >"$metrics_path"

  if ! command -v psql >/dev/null 2>&1; then
    cat >"$report_path" <<REPORT
# P8 Engine-Backed Pgwire Benchmark Smoke

- status: blocked
- blocker: missing_psql_client

The engine-backed pgwire benchmark smoke requires the PostgreSQL \`psql\` client.
REPORT
    cat "$report_path"
    return 0
  fi

  cargo build -q -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint
  GPU_DB_P8_ENGINE_PGWIRE_LISTEN="$listen" \
    GPU_DB_P8_ENGINE_PGWIRE_FACTS="$facts_path" \
    GPU_DB_P8_ENGINE_PGWIRE_MAX_SESSIONS=12 \
    target/debug/examples/p8_engine_pgwire_benchmark_endpoint >"$server_log" 2>&1 &
  local server_pid=$!
  trap 'kill "$server_pid" >/dev/null 2>&1 || true; wait "$server_pid" >/dev/null 2>&1 || true' RETURN

  {
    cat <<SQL
\set ON_ERROR_STOP on
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
\.
SQL
  } >"$load_path"

  local ready=0
  for _ in $(seq 1 120); do
    if psql "$url" -X -f "$load_path" >"$smoke_dir/load.out" 2>"$smoke_dir/load.err"; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  if [ "$ready" -ne 1 ]; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
    trap - RETURN
    cat >"$report_path" <<REPORT
# P8 Engine-Backed Pgwire Benchmark Smoke

- status: blocked
- blocker: engine_backed_pgwire_endpoint_startup_or_load_failed
- command: \`target/debug/examples/p8_engine_pgwire_benchmark_endpoint\`
- log: $server_log
- load_err: $smoke_dir/load.err
REPORT
    cat "$report_path"
    return 0
  fi

  local lower_bound expected_count lookup_key lookup_item lookup_qty lookup_amount lookup_dist
  lower_bound=$((rows / 4))
  if [ "$lower_bound" -lt 1 ]; then
    lower_bound=1
  fi
  expected_count="$rows"
  lookup_key=$(((rows + 1) / 2))
  lookup_item=$(((lookup_key % 100000) + 1))
  lookup_qty=$(((lookup_key % 50) + 1))
  lookup_amount=$(((lookup_key * 17) % 100000))
  lookup_dist="$(order_line_dist_info_shell "$lookup_key")"

  local tmp_prefix="$smoke_dir/query"
  engine_pgwire_query_metrics "$url" "$metrics_path" order_line_count_all "$expected_count" \
    "SELECT COUNT(*) FROM order_line" "$tmp_prefix" retained_engine_count_all true none
  engine_pgwire_query_metrics "$url" "$metrics_path" order_line_lookup_ol_o_id_retained_projection \
    "$lookup_key" \
    "SELECT ol_o_id FROM order_line WHERE ol_o_id = $lookup_key" \
    "$tmp_prefix" retained_engine_int4_equality_projection true none
  engine_pgwire_query_metrics "$url" "$metrics_path" order_line_lookup_ol_o_id_multi_column \
    "${lookup_key}|${lookup_item}|${lookup_qty}|${lookup_amount}" \
    "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount FROM order_line WHERE ol_o_id = $lookup_key" \
    "$tmp_prefix" engine_mvcc_cpu_fallback false "$lookup_blocker"
  engine_pgwire_query_metrics "$url" "$metrics_path" order_line_lookup_composite \
    "${lookup_key}|${lookup_item}|${lookup_qty}|${lookup_amount}|${lookup_dist}" \
    "SELECT ol_o_id, ol_i_id, ol_quantity, ol_amount, ol_dist_info FROM order_line WHERE ol_o_id = $lookup_key AND ol_i_id = $lookup_item" \
    "$tmp_prefix" engine_mvcc_cpu_fallback false "$lookup_blocker"

  cat >"$concurrency_path" <<CSV
tier,target,client_driver,query,concurrency,status,blocker,route_classification,metric_schema
25pct,engine_backed_pgwire_endpoint,psql/libpq,order_line_count_all,1,scaled_smoke,none,retained_engine_count_all,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"
25pct,engine_backed_pgwire_endpoint,psql/libpq,order_line_lookup_ol_o_id_retained_projection,1,scaled_smoke,none,retained_engine_int4_equality_projection,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"
25pct,engine_backed_pgwire_endpoint,psql/libpq,order_line_lookup_ol_o_id_multi_column,1,scaled_smoke,$lookup_blocker,engine_mvcc_cpu_fallback,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"
25pct,engine_backed_pgwire_endpoint,psql/libpq,order_line_lookup_composite,1,scaled_smoke,$lookup_blocker,engine_mvcc_cpu_fallback,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"
CSV
  local concurrency
  for concurrency in 2 4 8 16 32 64 128; do
    printf '25pct,engine_backed_pgwire_endpoint,psql/libpq,order_line_count_all,%s,blocked,true_concurrency_pg_client_runner_required,retained_engine_count_all,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"\n' "$concurrency" >>"$concurrency_path"
    printf '25pct,engine_backed_pgwire_endpoint,psql/libpq,order_line_lookup_ol_o_id_retained_projection,%s,blocked,true_concurrency_pg_client_runner_required,retained_engine_int4_equality_projection,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"\n' "$concurrency" >>"$concurrency_path"
    printf '25pct,engine_backed_pgwire_endpoint,psql/libpq,order_line_lookup_ol_o_id_multi_column,%s,blocked,%s,engine_mvcc_cpu_fallback,"wall_clock_throughput,p50_us,p95_us,p99_us,error_count,correctness_status,saturation_note"\n' "$concurrency" "$lookup_blocker" >>"$concurrency_path"
  done
  cat >>"$metrics_path" <<JSON
{"kind":"engine_backed_pgwire_endpoint_decision","tier":"25pct","status":"closed_with_blocker","endpoint":"engine_backed_pgwire_endpoint","client_driver":"psql/libpq","seed_path":"CREATE TABLE plus COPY FROM STDIN","count_retained_route":true,"count_zero_h2d":true,"lookup_retained_route":true,"lookup_retained_shape":"int4_equality_projection","multi_column_lookup_retained_route":false,"lookup_blocker":"$lookup_blocker","concurrency_blocker":"true_concurrency_pg_client_runner_required","facts":"$facts_path"}
JSON

  cat >"$report_path" <<REPORT
# P8 Engine-Backed Pgwire Benchmark Smoke

- rows: $rows
- status: closed_with_blocker
- endpoint: engine-backed PostgreSQL-compatible TCP benchmark target
- lookup_blocker: $lookup_blocker
- concurrency_blocker: true_concurrency_pg_client_runner_required
- endpoint_facts: $facts_path
- metrics_artifact: $metrics_path
- concurrency_plan: $concurrency_path
- server_log: $server_log

## Result

This smoke starts \`p8_engine_pgwire_benchmark_endpoint\`, connects through
\`psql\`/libpq, seeds \`order_line\` with SQL-visible \`CREATE TABLE\` plus
\`COPY FROM STDIN\`, warms the copied rows into \`RelationalResidentCache\`, and
runs client-visible \`SELECT\` traffic through the engine-owned endpoint.

\`SELECT COUNT(*) FROM order_line\` reaches the retained engine route with
accepted zero-H2D evidence in $facts_path. The narrow single-column
key-equality lookup \`SELECT ol_o_id FROM order_line WHERE ol_o_id = <literal>\`
also reaches a retained \`int4_equality_projection\` route with zero-H2D
evidence through the same client boundary. Multi-column and composite
key-equality lookups return correct rows, but still fall back to engine MVCC CPU
execution because retained row gathering / multi-column projection is not
implemented. That narrows the remaining point-lookup requirement to
\`$lookup_blocker\`.

## Boundary

- client boundary: PostgreSQL-compatible TCP startup and frontend frames
- client driver: \`psql\`/libpq
- seed path: \`CREATE TABLE\` plus \`COPY FROM STDIN WITH (FORMAT csv)\`
- state path: \`Engine\` WAL/MVCC table state
- retained path: \`Engine::warm_relational_residency_with_policy(...)\` plus
  retained \`COUNT(*)\` through \`Engine::execute_relational_select(...)\`
- non-goals: production server parity, prepared statements, portals, cursors,
  security/TLS, full concurrency curves, and 125% execution

## Next Blocker

The endpoint boundary is now available for a future identical-client harness
for retained \`COUNT(*)\` and same-column int4 equality-projection evidence. A
production-relevant multi-column or composite key-equality headline still needs
retained row gathering / multi-column projection, and true
1/2/4/8/16/32/64/128 client curves remain blocked until the scheduler targets
this endpoint.
REPORT

  kill "$server_pid" >/dev/null 2>&1 || true
  wait "$server_pid" >/dev/null 2>&1 || true
  trap - RETURN
  cat "$report_path"
  echo "p8_ch_benchmark_engine_backed_pgwire_smoke=closed_with_blocker retained_lookup_shape=int4_equality_projection lookup_blocker=$lookup_blocker artifact=$report_path"
}

write_protocol_retained_route_bridge_report() {
  mkdir -p "$OUT_DIR/protocol-retained-route-bridge"
  local bridge_dir="$OUT_DIR/protocol-retained-route-bridge"
  local rows="${GPU_DB_CH_BENCH_PROTOCOL_BRIDGE_ROWS:-64}"
  local report_path="$bridge_dir/protocol-retained-route-bridge.md"
  local metrics_path="$bridge_dir/metrics.jsonl"
  local architecture_path="$bridge_dir/architecture-facts.txt"
  local blocker="engine_backed_protocol_endpoint_required"
  local secondary_blocker="protocol_shared_catalog_to_engine_adapter_required"
  local seed_blocker="protocol_seed_to_resident_cache_admission_required"
  local lookup_blocker="primary_key_lookup_retained_route_required"

  {
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "crate_direction=gpu_db_engine_depends_on_gpu_db_protocol"
    echo "protocol_crate_depends_on_engine=false"
    echo "protocol_server_target=crates/protocol/src/bin/gpu-db-server.rs"
    echo "protocol_state=Session/SharedCatalog private Table rows"
    echo "protocol_select_executor=execute_select_result over protocol Table rows"
    echo "protocol_copy_admission=apply_copy_in_rows extends protocol Table rows and persists SharedCatalog snapshot"
    echo "retained_engine_target=crates/engine/examples/p8_ch_benchmark_residency_probe.rs"
    echo "retained_engine_entry=Engine::new_local plus install_benchmark_relational_residency_owned_chunks plus execute_relational_select"
    echo "retained_seed_state=benchmark-only generated chunks outside normal SQL/MVCC inserts"
    echo "safe_bridge_this_slice=false"
  } >"$architecture_path"

  cat >"$metrics_path" <<JSON
{"kind":"protocol_retained_route_bridge_decision","rows":$rows,"status":"blocked","blocker":"$blocker","secondary_blocker":"$secondary_blocker","seed_blocker":"$seed_blocker","lookup_blocker":"$lookup_blocker","protocol_client_family_required":"psql/libpq or PostgreSQL-compatible driver","protocol_endpoint_current_route":"protocol_shared_catalog_cpu_scan","retained_route_current_entry":"engine_internal","crate_dependency_cycle_risk":true,"safe_to_label_protocol_smoke_as_retained":false}
{"kind":"protocol_retained_route_required_change","status":"blocked","smallest_next_unblocker":"add an engine-owned PostgreSQL-compatible server/benchmark target, or split shared protocol session/catalog adapters into a crate that can be used by an engine-backed endpoint without making gpu_db_protocol depend on gpu_db_engine"}
JSON

  cat >"$report_path" <<REPORT
# P8 Protocol To Retained Route Bridge

- rows: $rows
- status: blocked
- blocker: $blocker
- secondary_blocker: $secondary_blocker
- seed_blocker: $seed_blocker
- lookup_blocker: $lookup_blocker
- architecture_facts: $architecture_path
- metrics_artifact: $metrics_path

## Result

The checked PostgreSQL-compatible smoke path proves \`psql\`/libpq can seed and
query the GPU DB endpoint, but this bridge cannot be safely closed by wiring a
small call across the current crates. The protocol endpoint is implemented as
\`gpu_db_protocol\`'s \`gpu-db-server\` binary and stores protocol-visible rows in
private \`Session\` / \`SharedCatalog\` \`Table\` state. Its \`COPY FROM STDIN\`
path appends parsed rows to that protocol table state, and its \`SELECT\` path
executes \`execute_select_result(...)\` over those rows.

The retained P8 benchmark route is owned by \`gpu_db_engine\`: it creates an
\`Engine::new_local()\`, installs benchmark-only generated chunks into
\`RelationalResidentCache\` through
\`install_benchmark_relational_residency_owned_chunks(...)\`, and executes
\`execute_relational_select(...)\`. The current crate direction is
\`gpu_db_engine -> gpu_db_protocol\`; making the protocol server directly depend
on \`gpu_db_engine\` would introduce the dependency cycle the benchmark contract
explicitly rejects.

## Decision

Do not relabel \`--gpu-db-protocol-benchmark-smoke\` as retained-route evidence.
The narrow next unblocker is an engine-backed PostgreSQL-compatible target or a
crate-boundary split/shared adapter that lets protocol-visible SQL/COPY table
state enter engine-owned WAL/MVCC state and retained-resident admission before
benchmarking. A benchmark-only resident admission bridge would also need an
explicit product decision because it would bypass normal SQL durability.

Until that target exists, aggregate and key-equality protocol smoke metrics stay
classified as \`protocol_shared_catalog_cpu_scan\`, the existing 25% retained
aggregate result stays \`engine_internal\`, and true concurrency curves remain
blocked behind a retained-route PostgreSQL-compatible target.
REPORT

  cat "$report_path"
  echo "p8_ch_benchmark_protocol_retained_route_bridge=blocked reason=$blocker artifact=$report_path"
}

write_engine_backed_protocol_boundary_probe() {
  mkdir -p "$OUT_DIR/engine-backed-protocol-boundary"
  local boundary_dir="$OUT_DIR/engine-backed-protocol-boundary"
  local rows="${GPU_DB_CH_BENCH_ENGINE_PROTOCOL_BOUNDARY_ROWS:-64}"
  local report_path="$boundary_dir/engine-backed-protocol-boundary.md"
  local facts_path="$boundary_dir/probe-facts.txt"
  local metrics_path="$boundary_dir/metrics.jsonl"
  local blocker="identical_pg_client_concurrency_harness_required"
  local secondary_blocker="true_concurrent_client_curves_required"
  local retained_blocker="closed"

  cargo run -q -p gpu_db_engine --example p8_engine_protocol_boundary_probe >"$facts_path"

  cat >"$metrics_path" <<JSON
{"kind":"engine_backed_protocol_boundary_probe","rows":$rows,"status":"closed","engine_owned_target":true,"protocol_parser_reused":true,"startup_packet_parser_reused":true,"frontend_message_parser_reused":true,"wire_session_api_available":true,"copy_parser_in_protocol_lib":true,"backend_writer_api_available":true,"ready_loop_state_available":true,"engine_owned_session_probe":true,"copy_stream_lifecycle_probe":true,"protocol_server_session_catalog_reusable":false,"create_table_into_engine_wal_mvcc":true,"copy_rows_visible_through_engine_select":true,"resident_admission_from_sql_visible_rows":true,"retained_route_zero_h2d":true,"post_mutation_residency_invalidated":true,"next_blocker":"$blocker","secondary_blocker":"$secondary_blocker","retained_blocker":"$retained_blocker"}
{"kind":"endpoint_boundary_decision","status":"closed","narrowest_safe_next_step":"run the identical PostgreSQL-compatible client harness and true concurrency curves against the retained engine route"}
JSON

  cat >"$report_path" <<REPORT
# P8 SQL-Visible Retained Admission Probe

- rows: $rows
- status: closed
- adapter_boundary: engine-owned startup/simple-query/COPY/select session probe with retained admission
- next_blocker: $blocker
- secondary_blocker: $secondary_blocker
- retained_blocker: $retained_blocker
- probe_facts: $facts_path
- metrics_artifact: $metrics_path

## Result

The checked \`p8_engine_protocol_boundary_probe\` example now proves the bounded
SQL-visible retained-route admission boundary. The probe owns
\`Engine::new_local()\`, reuses \`gpu_db_protocol\` startup-packet parsing,
frontend-message parsing, SQL parsing, COPY statement/row decoding, and backend
result writers, then routes a PostgreSQL-shaped startup + simple-query
\`CREATE TABLE\` + \`COPY FROM STDIN\` + \`SELECT COUNT(*)\` flow into engine
WAL/MVCC table state.

The same probe writes normal backend startup, \`CopyInResponse\`, command,
\`RowDescription\`, \`DataRow\`, and \`ReadyForQuery\` messages through
\`gpu_db_protocol::backend::BackendWriter\`. Rows loaded through COPY are visible
through \`Engine::execute_relational_select(...)\` from the engine-owned session
state. It then warms \`order_line\` through
\`Engine::warm_relational_residency_with_policy(...)\`, executes
\`SELECT COUNT(*)\` through the default \`Engine::execute_relational_select(...)\`
retained route, records accepted zero-H2D telemetry, and verifies a later engine
mutation invalidates the resident snapshot. This closes the retained-admission
blocker without making \`gpu_db_protocol\` depend on \`gpu_db_engine\`.

## Narrowest Viable Architecture

The next implementation slice should run the identical PostgreSQL-compatible
client harness and true concurrency curves against this retained engine route.
The current protocol server may still keep its private
\`Session\` / \`SharedCatalog\` path for the broader compatibility endpoint, but
the P8 benchmark target now has a bounded engine-owned path for
startup/simple-query/COPY/result writing plus retained admission from
SQL-visible rows.

## Rejected Alternatives

- Do not add \`gpu_db_engine\` as a dependency of \`gpu_db_protocol\`; the
  workspace already depends in the opposite direction.
- Do not relabel the current \`--gpu-db-protocol-benchmark-smoke\` metrics as
  retained-route evidence; those are protocol \`SharedCatalog\` CPU scans.
- Do not bypass WAL/MVCC by admitting COPY output directly to retained chunks;
  this probe warms from engine-visible table rows after COPY has committed.
- Do not collect true-concurrency product curves until the PostgreSQL-compatible
  target reaches the retained engine route.

## Benchmark Gate Impact

The existing 25% aggregate result remains provisional \`engine_internal\`
evidence. This slice proves SQL/COPY-loaded rows can enter engine WAL/MVCC,
be admitted to retained residency, and return through a PostgreSQL-shaped
result-writing boundary with accepted zero-H2D retained-route telemetry.
The benchmark trust gate still needs the same PostgreSQL-compatible client
harness for default PostgreSQL, tuned PostgreSQL, and GPU DB plus true
concurrent-client curves.
The 125% tier remains blocked on \`missing_partitioned_over_resident_execution\`.
REPORT

  cat "$report_path"
  echo "p8_ch_benchmark_sql_visible_retained_admission=closed next_blocker=$blocker artifact=$report_path"
}

order_line_dist_info_shell() {
  local id="$1"
  local bucket=$((id % 10))
  if ((id % 2 == 0)); then
    printf 'alpha%s\n' "$bucket"
  else
    printf 'omega%s\n' "$bucket"
  fi
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
  elif [ "${GPU_DB_CH_BENCH_ACCEPT_SCALED_25PCT:-0}" = "1" ]; then
    echo "p8_ch_benchmark_25pct_execution=scaled_pass rows=$execute_rows blocker=$blocker"
  else
    echo "p8_ch_benchmark_25pct_execution=blocked reason=$blocker" >&2
    return 1
  fi
}

write_125pct_readiness() {
  mkdir -p "$OUT_DIR"
  local retained_target_bytes=32212254720
  local retained_bytes_per_row=40
  local generated_bytes_per_row=96
  local estimated_rows
  estimated_rows="$(rows_125pct)"
  local generated_table_bytes=$((estimated_rows * generated_bytes_per_row))
  local retained_column_bytes=$((estimated_rows * retained_bytes_per_row))
  local wal_log_bytes=$((generated_table_bytes / 2))
  local report_bytes=$((2 * 1024 * 1024))
  local required_disk_bytes=$((generated_table_bytes + wal_log_bytes + report_bytes))
  local available_disk_bytes
  available_disk_bytes=$(df -B1 "$OUT_DIR" | awk 'NR==2 {print $4}')
  local disk_preflight=fail
  if [ "$available_disk_bytes" -gt "$required_disk_bytes" ]; then
    disk_preflight=pass
  fi

  local gpu_total_mib=unknown gpu_free_mib=unknown gpu_preflight=blocked
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_line
    gpu_line="$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)"
    if [ -n "$gpu_line" ]; then
      gpu_total_mib="$(printf '%s\n' "$gpu_line" | awk -F',' '{gsub(/ /, "", $1); print $1}')"
      gpu_free_mib="$(printf '%s\n' "$gpu_line" | awk -F',' '{gsub(/ /, "", $2); print $2}')"
      gpu_preflight=pass
    fi
  fi

  cargo run -q -p gpu_db_engine --example p8_ch_benchmark_residency_probe -- \
    --estimate \
    --output-dir "$OUT_DIR" \
    --rows "$ROWS"
  test -s "$OUT_DIR/estimate.jsonl"
  grep -q '"tier":"25pct"' "$OUT_DIR/estimate.jsonl"
  grep -q '"tier":"125pct"' "$OUT_DIR/estimate.jsonl"
  if grep -Eq '"tier":"(50pct|100pct|200pct|400pct)"' "$OUT_DIR/estimate.jsonl"; then
    echo "retired 50/100/200/400% VRAM benchmark tiers should not be configured" >&2
    exit 1
  fi

  local baseline_preflight=blocked
  if pgsql_baseline_docker_up >/tmp/gpu-db-p8-125pct-docker-up.out 2>/tmp/gpu-db-p8-125pct-docker-up.err; then
    if GPU_DB_CH_BENCH_PGSQL_URL="$(pgsql_docker_url)" write_pgsql_baseline_preflight >/tmp/gpu-db-p8-125pct-pgsql-preflight.out 2>/tmp/gpu-db-p8-125pct-pgsql-preflight.err; then
      baseline_preflight=pass
    fi
  fi

  local execute_rows="${GPU_DB_CH_BENCH_EXECUTE_ROWS:-1024}"
  local execute_chunk_rows="${GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS:-256}"
  cargo run -q -p gpu_db_engine --example p8_ch_benchmark_residency_probe -- \
    --chunked-execute \
    --output-dir "$OUT_DIR" \
    --rows "$execute_rows" \
    --chunk-rows "$execute_chunk_rows" \
    --concurrency "$CONCURRENCY"
  test -s "$OUT_DIR/chunked-execute/metrics.jsonl"
  grep -q '"kind":"chunked_metric"' "$OUT_DIR/chunked-execute/metrics.jsonl"
  grep -q '"resident_route_zero_h2d":true' "$OUT_DIR/chunked-execute/metrics.jsonl"
  grep -q '"kind":"chunked_memory_pressure_probe"' "$OUT_DIR/chunked-execute/metrics.jsonl"
  grep -q '"memory_pressure_route_accepted":false' "$OUT_DIR/chunked-execute/metrics.jsonl"

  local readiness_status=blocked
  local blocker=missing_partitioned_over_resident_execution
  if [ "$disk_preflight" != pass ]; then
    blocker=insufficient_disk_for_125pct_tier
  elif [ "$gpu_preflight" != pass ]; then
    blocker=missing_local_gpu_memory_facts
  elif [ "$baseline_preflight" != pass ]; then
    blocker=missing_passed_pgsql_baseline_preflight
  fi
  if [ "${GPU_DB_CH_BENCH_ALLOW_FULL_125PCT:-0}" = "1" ]; then
    blocker=full_125pct_still_blocked_without_partitioned_over_resident_execution
  fi

  local full_gpu_command="GPU_DB_CH_BENCH_ALLOW_FULL_125PCT=1 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=1048576 scripts/run_p8_ch_benchmark_residency_probe.sh --run-125pct"
  local full_pgsql_command="GPU_DB_CH_BENCH_PGSQL_URL='$(pgsql_docker_url)' GPU_DB_CH_BENCH_ALLOW_FULL_PGSQL_125PCT=1 GPU_DB_CH_BENCH_PGSQL_ROWS=$estimated_rows scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-125pct-latency"

  cat >"$OUT_DIR/125pct-readiness.md" <<REPORT
# P8 CH-benCHmark 125% Over-Resident Readiness

- tier: 125pct
- status: $readiness_status
- blocker: $blocker
- retained_target_bytes: $retained_target_bytes
- estimated_order_line_rows: $estimated_rows
- retained_column_bytes: $retained_column_bytes
- generated_table_bytes: $generated_table_bytes
- wal_log_bytes: $wal_log_bytes
- report_bytes: $report_bytes
- required_disk_bytes: $required_disk_bytes
- available_disk_bytes: $available_disk_bytes
- disk_preflight: $disk_preflight
- local_gpu_memory_total_mib: $gpu_total_mib
- local_gpu_memory_free_mib: $gpu_free_mib
- gpu_preflight: $gpu_preflight
- postgresql_baseline_preflight: $baseline_preflight
- retired_50_100_200_400pct_tiers_absent: true
- scaled_probe_rows: $execute_rows
- scaled_probe_chunk_rows: $execute_chunk_rows
- scaled_probe_artifact: $OUT_DIR/chunked-execute/execution.md
- raw_metrics: $OUT_DIR/chunked-execute/metrics.jsonl
- full_gpu_command: \`$full_gpu_command\`
- full_postgresql_command: \`$full_pgsql_command\`
- cleanup_command: \`scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup && scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down\`

The readiness command computes the 125% retained target and local capacity
facts, proves the retired 50/100/200/400% tiers are absent from dry-run output,
validates the disposable PostgreSQL comparator at preflight scale, and runs a
small deterministic chunked resident execution probe. The scaled probe verifies
accepted zero-H2D retained routes before explicitly marking GPU memory pressure;
the route is then rejected with the retained snapshot invalidated, which proves
the current fallback/rejection signal path is observable.

The full 125% tier is not safe to schedule yet. The current execution path
installs one retained CUDA resident layout, so the about 30 GiB 125% retained
target is larger than the local RTX 3090 memory envelope. A trustworthy full
125% PostgreSQL-vs-GPU run needs partitioned or streamed over-resident execution
and an explicit operator-approved long-run window before either guarded full
command can be promoted from blocked source truth.
REPORT

  cat >"$OUT_DIR/125pct-readiness.jsonl" <<JSON
{"kind":"tier_readiness","tier":"125pct","status":"$readiness_status","blocker":"$blocker","retained_target_bytes":$retained_target_bytes,"estimated_rows":$estimated_rows,"retained_column_bytes":$retained_column_bytes,"generated_table_bytes":$generated_table_bytes,"wal_log_bytes":$wal_log_bytes,"report_bytes":$report_bytes,"required_disk_bytes":$required_disk_bytes,"available_disk_bytes":$available_disk_bytes,"disk_preflight":"$disk_preflight","gpu_total_mib":"$gpu_total_mib","gpu_free_mib":"$gpu_free_mib","gpu_preflight":"$gpu_preflight","postgresql_baseline_preflight":"$baseline_preflight","retired_tiers_absent":true,"scaled_probe_rows":$execute_rows,"scaled_probe_chunk_rows":$execute_chunk_rows,"full_gpu_guard":"GPU_DB_CH_BENCH_ALLOW_FULL_125PCT=1","full_pgsql_guard":"GPU_DB_CH_BENCH_ALLOW_FULL_PGSQL_125PCT=1"}
JSON
  cat "$OUT_DIR/125pct-readiness.md"
  if [ "${GPU_DB_CH_BENCH_ACCEPT_SCALED_125PCT:-0}" = "1" ]; then
    echo "p8_ch_benchmark_125pct_readiness=scaled_pass blocker=$blocker"
    return 0
  fi
  echo "p8_ch_benchmark_125pct_readiness=blocked reason=$blocker" >&2
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
    grep -q '"tier":"125pct"' "$OUT_DIR/estimate.jsonl"
    if grep -Eq '"tier":"(50pct|100pct|200pct|400pct)"' "$OUT_DIR/estimate.jsonl"; then
      echo "retired 50/100/200/400% VRAM benchmark tiers should not be configured" >&2
      exit 1
    fi
    grep -q 'logical_request_targets: \[1, 10, 100, 1000, 10000\]' "$OUT_DIR/estimate.md"
    grep -q 'true_concurrency_targets: blocked_until_protocol_benchmark_harness' "$OUT_DIR/estimate.md"
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
  --run-125pct)
    write_125pct_readiness
    ;;
  --pgsql-baseline-preflight)
    write_pgsql_baseline_preflight
    ;;
  --pgsql-baseline-25pct-latency)
    write_pgsql_25pct_latency
    ;;
  --pgsql-baseline-125pct-latency)
    write_pgsql_125pct_latency
    ;;
  --pgsql-fairness-audit)
    write_pgsql_fairness_audit
    ;;
  --gpu-db-protocol-benchmark-smoke)
    write_gpu_db_protocol_benchmark_smoke
    ;;
  --engine-backed-pgwire-benchmark-smoke)
    write_engine_backed_pgwire_benchmark_smoke
    ;;
  --protocol-retained-route-bridge-report)
    write_protocol_retained_route_bridge_report
    ;;
  --engine-backed-protocol-boundary-probe)
    write_engine_backed_protocol_boundary_probe
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
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_EXECUTE_ROWS=16 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=4 GPU_DB_CH_BENCH_CONCURRENCY=1 \
      "$0" --run-125pct >"$tmp_dir/125pct.out" 2>"$tmp_dir/125pct.err" || true
    grep -q 'missing_partitioned_over_resident_execution' "$tmp_dir/125pct.err"
    grep -q 'retired_50_100_200_400pct_tiers_absent: true' "$tmp_dir/out/125pct-readiness.md"
    if GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_ROWS=16 \
      "$0" --pgsql-baseline-preflight >"$tmp_dir/pgsql.out" 2>"$tmp_dir/pgsql.err"; then
      grep -q 'status: pass' "$tmp_dir/pgsql.out"
    else
      grep -Eq 'missing_pgsql_baseline_connection|missing_psql_client' "$tmp_dir/pgsql.err"
    fi
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_PGSQL_AUDIT_ROWS=16 GPU_DB_CH_BENCH_PGSQL_AUDIT_REPEATS=1 \
      "$0" --pgsql-fairness-audit >"$tmp_dir/fairness.out"
    grep -q 'gpu_db_protocol_benchmark_path_required' "$tmp_dir/out/pgsql-fairness-audit/fairness-audit.md"
    grep -q 'concurrency_targets' "$tmp_dir/out/pgsql-fairness-audit/metrics.jsonl"
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_GPU_DB_PROTOCOL_ROWS=16 GPU_DB_CH_BENCH_GPU_DB_PROTOCOL_PORT=55436 \
      "$0" --gpu-db-protocol-benchmark-smoke >"$tmp_dir/gpu-db-protocol.out"
    grep -q 'protocol_endpoint_uses_protocol_catalog_not_p8_resident_engine' "$tmp_dir/out/gpu-db-protocol-benchmark-smoke/protocol-benchmark-smoke.md"
    grep -q '"query":"order_line_lookup_ol_o_id"' "$tmp_dir/out/gpu-db-protocol-benchmark-smoke/metrics.jsonl"
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_PROTOCOL_BRIDGE_ROWS=16 \
      "$0" --protocol-retained-route-bridge-report >"$tmp_dir/protocol-bridge.out"
    grep -q 'engine_backed_protocol_endpoint_required' "$tmp_dir/out/protocol-retained-route-bridge/protocol-retained-route-bridge.md"
    grep -q '"safe_to_label_protocol_smoke_as_retained":false' "$tmp_dir/out/protocol-retained-route-bridge/metrics.jsonl"
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_ENGINE_PROTOCOL_BOUNDARY_ROWS=16 \
      "$0" --engine-backed-protocol-boundary-probe >"$tmp_dir/engine-boundary.out"
    grep -q 'sql_visible_retained_admission_ready' "$tmp_dir/out/engine-backed-protocol-boundary/probe-facts.txt"
    grep -q 'resident_admission_from_sql_visible_rows=true' "$tmp_dir/out/engine-backed-protocol-boundary/probe-facts.txt"
    grep -q 'retained_route_zero_h2d=true' "$tmp_dir/out/engine-backed-protocol-boundary/probe-facts.txt"
    grep -q 'identical_pg_client_concurrency_harness_required' "$tmp_dir/out/engine-backed-protocol-boundary/engine-backed-protocol-boundary.md"
    grep -q 'protocol_parser_reused=true' "$tmp_dir/out/engine-backed-protocol-boundary/probe-facts.txt"
    GPU_DB_CH_BENCH_OUT_DIR="$tmp_dir/out" GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55438 \
      "$0" --engine-backed-pgwire-benchmark-smoke >"$tmp_dir/engine-pgwire.out"
    grep -q 'client_visible_select_retained_route_accepted=true' "$tmp_dir/out/engine-backed-pgwire-benchmark-smoke/endpoint-facts.txt"
    grep -q 'client_visible_select_retained_route_zero_h2d=true' "$tmp_dir/out/engine-backed-pgwire-benchmark-smoke/endpoint-facts.txt"
    grep -q 'int4_equality_projection' "$tmp_dir/out/engine-backed-pgwire-benchmark-smoke/endpoint-facts.txt"
    grep -q 'retained_multi_column_projection_required' "$tmp_dir/out/engine-backed-pgwire-benchmark-smoke/engine-backed-pgwire-benchmark-smoke.md"
    grep -q '"query":"order_line_lookup_ol_o_id_retained_projection"' "$tmp_dir/out/engine-backed-pgwire-benchmark-smoke/metrics.jsonl"
    grep -q '"query":"order_line_lookup_ol_o_id_multi_column"' "$tmp_dir/out/engine-backed-pgwire-benchmark-smoke/metrics.jsonl"
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
