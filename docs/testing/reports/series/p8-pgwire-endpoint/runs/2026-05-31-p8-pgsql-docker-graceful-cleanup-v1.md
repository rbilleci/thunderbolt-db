# P8 PostgreSQL Docker Graceful Cleanup Report

- date: 2026-05-31
- stream: benchmark
- milestone: P8 PostgreSQL comparator lifecycle hardening
- status: pass
- blocker_resolved: `docker_cleanup_permission_denied_for_new_pgsql_baseline_container`

## Result

The local Docker daemon on this host can create PostgreSQL comparator containers,
but forced Docker stop/remove can fail with `permission denied`. The benchmark
harness no longer depends on the daemon kill path for normal comparator cleanup.

`scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`
now first asks PostgreSQL to stop itself from inside the container as the
`postgres` user, waits for the server process to exit, and then removes the
already-exited container. Existing comparator cleanup in
`--pgsql-baseline-docker-up` uses the same helper before starting a fresh
container.

## Evidence

- Both previously stuck comparator containers were gracefully stopped from
  inside the container and removed:
  - `gpu-db-p8-pgsql-baseline`
  - `gpu-db-p8-pgsql-baseline-disposable`
- A fresh Docker PostgreSQL baseline preflight passed.
- A repeated `--pgsql-baseline-docker-up` cleaned the existing comparator with
  the graceful path and started a fresh one.
- Final `--pgsql-baseline-docker-down` passed.
- No `gpu-db.p8-benchmark=true` disposable comparator containers remained after
  cleanup.

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
- repeated `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-up`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct`: passed with
  `run_preflight: ready_to_start`
- `git diff --check`: passed

## Remaining Scope

This fixes the benchmark harness lifecycle on hosts where PostgreSQL can shut
itself down but Docker daemon forced kill is denied. It does not claim to fix
general Docker daemon/container runtime kill behavior for arbitrary containers.
