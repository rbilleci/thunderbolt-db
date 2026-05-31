# P8 Retained BETWEEN AVG Parallel-Stats Report

- date: 2026-05-31
- stream: benchmark
- milestone: P8 retained `AVG ... BETWEEN` parallel stats narrowing
- status: pass
- source_regression_report: `docs/testing/reports/2026-05-31-p8-25pct-aggregate-refresh-v1.md`

## Result

The retained `AVG(int4) WHERE same_column BETWEEN lower AND upper` path no
longer launches the obvious one-block, one-thread full-row stats scan. The CUDA
retained-device-memory helper now launches a bounded parallel grid with
256-thread blocks, uses a strided per-thread scan over the retained int4 column,
and atomically reduces `count`, `sum`, `min`, and `max` into one scalar stats
buffer initialized on device before launch.

The route still preserves answer parity for `SUM`, `AVG`, `MIN`, `MAX`, and
empty ranges, keeps H2D at zero for retained execution, and keeps D2H to scalar
stats size for non-empty BETWEEN aggregates.

## Focused Evidence

- Focused retained-device-memory test passed for `SUM`, `AVG`, `MIN`, `MAX`, and
  empty range parity against the CPU path.
- Scaled `--run-25pct-execute` evidence over 1,024 rows preserved
  formula-backed correctness and zero-H2D resident routing.
- Scaled metrics recorded `order_line_avg_quantity_between` D2H as `32` bytes
  for one logical request and `320` bytes for ten logical requests.
- Scaled metrics recorded `order_line_avg_quantity_between` CUDA event time as
  `8 us` for one logical request and `80 us` across ten logical requests.
- The earlier scalar-stats evidence artifact on the same 1,024-row shape
  recorded `83 us` for one logical request and `835 us` across ten logical
  requests, so this bounded probe removes the prior single-thread kernel shape
  and materially narrows runtime at the focused scale.

## Scope Impact

This closes the code-reality blocker that retained `AVG ... BETWEEN` still used
a serial stats scan after the scalar D2H readback fix. It does not by itself
admit the 125% over-resident tier or claim the full 25% side-by-side regression
is resolved. The checked full 25% aggregate report still records the older
`AVG ... BETWEEN` timing, so the next honest admission decision is a refreshed
25% aggregate comparison using the parallel BETWEEN stats kernel, or an explicit
product decision if the refreshed full-tier evidence remains slower.

Do not run retired 50%, 100%, or 200% tiers from this report. Future P8 tiering
remains simplified to 25% and 125%, with 125% gated on the refreshed 25%
admission decision.

## Validation Gate

- `cargo test -p gpu_db_engine gpu_resident_device_memory_between_scalar_aggregate_probe_materializes_int4_results -- --nocapture`: passed
- `GPU_DB_CH_BENCH_ACCEPT_SCALED_25PCT=1 GPU_DB_CH_BENCH_OUT_DIR=target/p8-between-parallel-stats GPU_DB_CH_BENCH_EXECUTE_ROWS=1024 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=256 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute`: passed
- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_engine --example p8_ch_benchmark_residency_probe`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- `git diff --check`: passed
