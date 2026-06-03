# P8 Retained MAX Filter Domain-Pruning Probe

- date: 2026-05-31
- stream: benchmark
- milestone: P8 retained `MAX(int4)` same-column filter empty-domain pruning
- status: pass
- source_regression_report: `docs/testing/reports/series/p8-25pct-full-run/runs/2026-05-31-p8-25pct-regression-triage-v1.md`

## Result

The retained filtered `MAX(int4)` path can now avoid the pointless retained GPU
scan when resident int4 column metadata proves the same-column comparison
predicate is empty. Benchmark-only chunked resident admission carries explicit
min/max metadata for deterministic `order_line` int4 columns; the normal
resident snapshot path also records observed min/max for materialized int4
columns. This is a checked resident-snapshot/domain proof, not a broad planner
statistics claim.

For the full 25% benchmark shape, `order_line_max_amount_filter` is
`SELECT MAX(ol_amount) FROM order_line WHERE ol_amount >= rows / 4`. At
`161,061,274` rows, the literal is `40,265,318`, while deterministic
`ol_amount = (id * 17) % 100000` is bounded to `0..99,999`, so the retained path
can return the existing SQL NULL result shape without scanning the resident
column.

The retained filtered scalar aggregate path also now reports scalar-stats D2H
for normal non-empty filtered aggregates instead of count-proportional D2H
telemetry. Non-empty filtered `MAX` still routes through the retained kernel and
preserves CPU-equivalent results.

## Evidence

- Focused retained-device-memory test passed for non-empty `SUM`/`AVG`/`MIN`/`MAX`
  filtered aggregates plus an empty-domain `MAX(amount) WHERE amount >= 1000`
  case. The empty-domain case preserved CPU-equivalent NULL output, zero H2D,
  scalar 8-byte D2H accounting, and zero new retained kernel samples.
- Scaled non-empty P8 execution over 1,024 rows preserved formula-backed
  `order_line_max_amount_filter` correctness, zero H2D, retained route
  acceptance, 32 D2H bytes for one logical request, and 320 D2H bytes for ten
  logical requests.
- Scaled empty-domain P8 execution over 400,004 rows used a lower bound of
  `100,001`, proved the deterministic amount domain empty, preserved
  formula-backed correctness, kept H2D at zero, reported 8 D2H bytes for one
  logical request and 80 D2H bytes for ten logical requests, and recorded zero
  retained kernel samples for `order_line_max_amount_filter`.
- The scaled P8 execution commands intentionally stopped at the existing guarded
  `full_25pct_requires_operator_long_run_after_streaming_boundary` blocker; the
  prior full 25% GPU DB and PostgreSQL artifacts were preserved.

## Scope Impact

`order_line_max_amount_filter` is now narrowed from "retained GPU scan despite
deterministically empty predicate" to "domain-pruned empty predicate; full-tier
side-by-side aggregate reclassification still needs refreshed evidence." The
50%, 100%, and 200% tiers remain gated until retained aggregate regressions are
refreshed after the bounded improvements, accepted as current non-claims, or
narrowed by further instrumentation. This slice does not claim full
CH-benCHmark/BenchBase compatibility, joins, transaction mix, production cache
daemon behavior, durable GPU pages, or completed 50%+ PostgreSQL-vs-GPU tiers.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine gpu_resident_device_memory_filtered_scalar_aggregate_probe_materializes_int4_results -- --nocapture`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo check -p gpu_db_engine --example p8_ch_benchmark_residency_probe`: passed
- `git diff --check`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-max-filter-domain-pruning-nonempty GPU_DB_CH_BENCH_EXECUTE_ROWS=1024 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=256 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute`: expected guarded blocker after scaled non-empty execution
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-max-filter-domain-pruning-empty GPU_DB_CH_BENCH_EXECUTE_ROWS=400004 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=100001 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute`: expected guarded blocker after scaled empty-domain execution
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
