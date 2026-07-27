#!/usr/bin/env bash
# Run the measurement-only FUA controller cadence gate. Each cargo invocation
# starts a fresh benchmark process; the Rust trace validates its own recovery,
# controls, amplification, and durability thresholds before returning success.
set -euo pipefail

trace_runs="${CONVEYOR_FUA_CONTROLLER_TRACE_RUNS:-3}"
trace_groups="${CONVEYOR_FUA_CONTROLLER_TRACE_GROUPS:-8000}"
seed_base="${CONVEYOR_FUA_CONTROLLER_TRACE_SEED_BASE:-1}"

if [[ "$trace_runs" != "3" ]]; then
    echo "controller trace gate requires exactly 3 fresh-process runs (got $trace_runs)" >&2
    exit 2
fi
if [[ "$trace_groups" != "8000" ]]; then
    echo "controller trace gate requires exactly 8000 production-cadence groups (got $trace_groups)" >&2
    exit 2
fi

for run in 1 2 3; do
    seed=$((seed_base + run - 1))
    echo "controller_trace_run=$run seed=$seed groups=$trace_groups"
    CONVEYOR_FUA_MODE=controller_trace \
        CONVEYOR_FUA_CONTROLLER_TRACE_GROUPS="$trace_groups" \
        CONVEYOR_FUA_CONTROLLER_TRACE_SEED="$seed" \
        CONVEYOR_FUA_CONTROLLER_TRACE_ENFORCE_GATE=1 \
        cargo run --release -p gpu_db_write_conveyor --example fua_frame_log_bench
done
