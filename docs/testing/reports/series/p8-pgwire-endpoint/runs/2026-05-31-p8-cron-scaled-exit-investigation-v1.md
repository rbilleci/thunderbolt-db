# P8 Cron Scaled Exit Investigation

- date: 2026-05-31
- affected_job: `gpu-db-autoloop-until-blocker`
- affected_round: `2026-05-31-p8-between-avg-parallel-stats-v1`
- failure_surface: `Codex stopped before confirming the turn was complete`

## Finding

The failing cron runs reached the focused scaled execution probe:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-between-parallel-stats-1m \
GPU_DB_CH_BENCH_EXECUTE_ROWS=1048576 \
GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=262144 \
scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute
```

The local artifacts show that the scaled GPU DB execution itself passed and
wrote metrics under `target/p8-between-parallel-stats-1m/chunked-execute/`.
The script then returned non-zero by design because the full 25% long-run guard
was not enabled. In the isolated cron environment, that expected non-zero tool
result was reported as a Codex turn failure instead of a precise benchmark
blocker.

## Fix

`--run-25pct-execute` now supports:

```bash
GPU_DB_CH_BENCH_ACCEPT_SCALED_25PCT=1
```

When set, guarded scaled execution still records the full-run blocker in the
report, but exits zero after the scaled evidence passes. Full 25% execution
still requires `GPU_DB_CH_BENCH_ALLOW_FULL_25PCT=1`.

The active worker control file now tells workers to use
`GPU_DB_CH_BENCH_ACCEPT_SCALED_25PCT=1` for focused scaled probes, so expected
scaled evidence should not surface as a cron/runtime error again.

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`
- `GPU_DB_CH_BENCH_ACCEPT_SCALED_25PCT=1 GPU_DB_CH_BENCH_OUT_DIR=target/p8-scaled-exit-smoke GPU_DB_CH_BENCH_EXECUTE_ROWS=1024 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=256 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute`
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`
