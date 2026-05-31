# P8 Retained BETWEEN AVG Scalar-Stats Report

- date: 2026-05-31
- stream: benchmark
- milestone: P8 retained BETWEEN scalar aggregate efficiency
- status: pass
- source_regression_report: `docs/testing/reports/2026-05-31-p8-25pct-regression-triage-v1.md`

## Result

The retained `AVG ... BETWEEN` aggregate path no longer reports D2H transfer
proportional to qualifying row count. The CUDA retained-device-memory helper now
computes `count`, `sum`, `min`, and `max` for the inclusive range directly into
one scalar stats buffer, and the engine reports scalar-stats D2H telemetry for
retained BETWEEN scalar aggregate probes.

This closes the obvious readback-shape defect behind the 25% triage report's
`528,280,972` D2H bytes for `order_line_avg_quantity_between`. It does not claim
that the full 25% PostgreSQL-vs-GPU aggregate regression is fully resolved,
because the full 25% long run was intentionally not rerun in this bounded slice.

## Evidence

- Focused retained-device-memory test passed for `SUM`, `AVG`, `MIN`, `MAX`, and
  empty range parity against the CPU path.
- Scaled `--run-25pct-execute` evidence over 1,024 rows preserved
  formula-backed correctness and zero H2D resident routing.
- Scaled metrics recorded `order_line_avg_quantity_between` D2H as `32` bytes
  for one logical request and `320` bytes for ten logical requests.
- The same scaled run still reported the expected guarded blocker
  `full_25pct_requires_operator_long_run_after_streaming_boundary`; the full
  25% GPU DB and PostgreSQL artifacts from the prior reports were preserved.

## Scope Impact

`order_line_avg_quantity_between` is now narrowed from "implementation bottleneck
plus matched-value D2H readback" to "kernel/runtime timing needs full-tier
refresh before the regression can be reclassified." Retained `SUM` and
`MAX ... filter` regressions remain open from the prior triage report. The
50%, 100%, and 200% tiers remain gated until the retained aggregate regressions
are optimized further, explicitly accepted as current non-claims, or refreshed
with matching side-by-side evidence.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine gpu_resident_device_memory_between_scalar_aggregate_probe_materializes_int4_results -- --nocapture`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- `cargo check -p gpu_db_engine --example p8_ch_benchmark_residency_probe`: passed
- `git diff --check`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-between-scalar-stats scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute`: expected guarded blocker after scaled execution
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-between-scalar-stats scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
