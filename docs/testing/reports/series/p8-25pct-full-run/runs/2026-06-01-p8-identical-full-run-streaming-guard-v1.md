# P8 Identical Full-Run Streaming Guard

- date: 2026-06-01
- stream: benchmark
- milestone: identical pgwire full-run streaming load guard and readiness boundary
- status: closed_with_blocker
- readiness_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-full-readiness`
- readiness_artifact: `target/p8-identical-full-run-streaming-guard-v1-readiness/identical-pgwire-full-readiness/readiness.md`
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke`
- smoke_artifact: `target/p8-identical-full-run-streaming-guard-v1-smoke2/identical-pgwire-target-smoke/identical-pgwire-target-smoke.md`
- metrics_artifact: `target/p8-identical-full-run-streaming-guard-v1-smoke2/identical-pgwire-target-smoke/metrics.jsonl`
- curve_artifact: `target/p8-identical-full-run-streaming-guard-v1-smoke2/identical-pgwire-target-smoke/concurrency-curve.csv`
- next_blocker: `full_25pct_identical_curves_require_operator_long_run`

## Result

The identical pgwire target harness now streams deterministic setup and data
directly into each target through `psql` instead of materializing
`identical-pgwire-target-smoke/load.sql`. The load contract remains
SQL-visible: `CREATE TABLE` plus `COPY FROM STDIN` commits rows through Engine
WAL/MVCC state before GPU DB retained residency warmup.

The new readiness mode records the full 25% operator package without executing
it. For the current 25% tier it estimates `161,061,274` `order_line` rows,
about `15,461,882,304` generated-table bytes, `7,730,941,152` WAL bytes, and a
bounded raw-metric budget. It lists default PostgreSQL, tuned PostgreSQL, and
GPU DB retained endpoint targets with redacted/default local URLs, concurrency
targets, cleanup command, expected retained-route facts, and the remaining
125% blocker.

Full identical 25% execution is explicitly guarded:

`GPU_DB_CH_BENCH_ALLOW_FULL_IDENTICAL_PGWIRE_25PCT=1 GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=161061274 GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke`

The checked scaled run used 16 rows and concurrency targets `1,2`. It passed
across default PostgreSQL, tuned PostgreSQL, and the GPU DB retained endpoint,
and `load.sql` was absent from the smoke artifact directory.

## Remaining Boundary

This slice does not run the full 25% default PostgreSQL, tuned PostgreSQL, or
GPU DB retained curves. That remains blocked on an operator-approved long-run
window and artifact budget: `full_25pct_identical_curves_require_operator_long_run`.

Full 125% remains blocked by `missing_partitioned_over_resident_execution`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-full-run-streaming-guard-v1-readiness GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-full-readiness`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-full-run-streaming-guard-v1-smoke2 GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55504 GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55505 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke`: passed
- `test ! -e target/p8-identical-full-run-streaming-guard-v1-smoke2/identical-pgwire-target-smoke/load.sql`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
- `git diff --check`: passed
