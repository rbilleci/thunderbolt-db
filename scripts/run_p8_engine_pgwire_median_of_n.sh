#!/usr/bin/env bash
# Median-of-N noise-controlled baseline for the engine-backed pgwire concurrency
# smoke (prototype->production plan §5.7, minimal Phase-5 noise-control slice).
#
# Runs the EXISTING `--engine-backed-pgwire-concurrency-smoke` N times (plus one
# discarded warm-up run — warmup separated from measurement), each into its own
# output dir and port, then aggregates per (query, concurrency) cell into
# median + 95% CI + coefficient of variation via aggregate_concurrency_runs.py.
#
# It changes NOTHING on the measured path; it only repeats and post-processes the
# existing harness, so it is safe to run as a baseline before/after a refactor.
#
# Conditions are pinned to the M0 baseline: cache-off, batched read runtime on,
# concurrency targets 1,2,4,8,16,32,64 (override via the env vars below).
#
# Env:
#   GPU_DB_MEDIAN_RUNS      measured runs to aggregate (default 10)
#   GPU_DB_MEDIAN_WARMUP    1 to run one discarded warm-up first (default 1)
#   GPU_DB_MEDIAN_OUT_DIR   base output dir (default target/p1-m3-median-of-n)
#   GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS  default 1,2,4,8,16,32,64
#   GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT  base port (default 55437; run i uses base+i)
set -uo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

RUNS="${GPU_DB_MEDIAN_RUNS:-10}"
WARMUP="${GPU_DB_MEDIAN_WARMUP:-1}"
BASE="${GPU_DB_MEDIAN_OUT_DIR:-target/p1-m3-median-of-n}"
TARGETS="${GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS:-1,2,4,8,16,32,64}"
BASE_PORT="${GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT:-55437}"
PROBE="scripts/run_p8_ch_benchmark_residency_probe.sh"
AGG="scripts/aggregate_concurrency_runs.py"

rm -rf "$BASE"
mkdir -p "$BASE"

FACTS="$BASE/host-facts.txt"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo "cargo=$(cargo --version 2>/dev/null)"
  echo "rustc=$(rustc --version 2>/dev/null)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "measured_runs=$RUNS warmup=$WARMUP"
  echo "concurrency_targets=$TARGETS"
  echo "conditions=cache_off,retained_read_runtime_view_on (pinned to M0)"
  command -v nvidia-smi >/dev/null 2>&1 &&
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$FACTS"

# Monotonic port allocator: no two probe attempts in this batch share a port, so a
# prior run's listener lingering in TIME_WAIT can never collide with the next. The
# starting point is jittered per batch so back-to-back *batches* also avoid reusing
# ports still in TIME_WAIT (the retry below handles any residual collision).
PORT_NEXT=$((BASE_PORT + (RANDOM % 256)))
PORT=""
alloc_port() {
  PORT_NEXT=$((PORT_NEXT + 1))
  PORT="$PORT_NEXT"
}

settle() { sleep "${GPU_DB_MEDIAN_SETTLE_SECS:-1}"; }

# One probe invocation into a given dir/port. Pins M0 conditions explicitly.
run_probe() {
  local dir="$1" port="$2" log="$3"
  GPU_DB_CH_BENCH_OUT_DIR="$dir" \
  GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS="$TARGETS" \
  GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT="$port" \
  GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RESPONSE_CACHE=0 \
  GPU_DB_P8_ENGINE_PGWIRE_RETAINED_READ_RUNTIME_VIEW=1 \
    "$PROBE" --engine-backed-pgwire-concurrency-smoke >"$log" 2>&1
  return $?
}

# Run the probe with a fresh port each attempt; retry transient bind failures only.
# Returns 0 and leaves a non-empty metrics.jsonl on success.
attempt_probe() {
  local dir="$1" log="$2" tries="${GPU_DB_MEDIAN_TRIES:-3}" t rc
  local m="$dir/engine-backed-pgwire-concurrency-smoke/metrics.jsonl"
  for ((t = 1; t <= tries; t++)); do
    alloc_port
    run_probe "$dir" "$PORT" "$log"
    rc=$?
    if [ "$rc" -eq 0 ] && [ -s "$m" ]; then
      [ "$t" -gt 1 ] && echo "  recovered on attempt $t (port $PORT)"
      settle
      return 0
    fi
    # The bind error (AddrInUse) lands in the *endpoint* log; the probe's own stdout
    # ($log) instead carries the `..._startup_or_load_failed` blocker. Treat either as
    # a transient and retry on a fresh port.
    if grep -qiE 'addrinuse|address already in use|engine_backed_pgwire_endpoint_startup_or_load_failed' "$log" 2>/dev/null; then
      RETRIES=$((RETRIES + 1))
      echo "  attempt $t/$tries: startup/bind failed (port $PORT), retrying on fresh port" >&2
      settle
      continue
    fi
    echo "  attempt $t/$tries: rc=$rc metrics=$([ -s "$m" ] && echo present || echo empty) non-transient" >&2
    settle
    return 1
  done
  return 1
}

RETRIES=0  # transient bind/startup retries across the batch (recorded in host-facts)

if [ "$WARMUP" = "1" ]; then
  echo "== warm-up run (discarded) =="
  attempt_probe "$BASE/run-0-warmup" "$BASE/run-0-warmup.log" && echo "warmup ok (discarded)" || echo "warmup failed (discarded anyway)"
fi

METRICS=()
FAILED=0
for ((i = 1; i <= RUNS; i++)); do
  echo "== measured run $i/$RUNS =="
  dir="$BASE/run-$i"
  if attempt_probe "$dir" "$BASE/run-$i.log"; then
    METRICS+=("$dir/engine-backed-pgwire-concurrency-smoke/metrics.jsonl")
    echo "run $i: ok"
  else
    FAILED=$((FAILED + 1))
    echo "run $i: FAILED (see $BASE/run-$i.log)" >&2
  fi
done

echo "== aggregate =="
if [ "${#METRICS[@]}" -eq 0 ]; then
  echo "median_of_n=blocked reason=no_successful_runs failed=$FAILED" >&2
  exit 1
fi
python3 "$AGG" "$BASE/aggregate" "${METRICS[@]}"

echo "retries_total=$RETRIES" >>"$FACTS"
echo "median_of_n=done measured=${#METRICS[@]} failed=$FAILED retries=$RETRIES base=$BASE"
echo "summary=$BASE/aggregate/summary.md"
[ "$FAILED" -gt 0 ] && echo "WARNING: $FAILED run(s) failed and were excluded" >&2
exit 0
