# P8 Retained SUM Parallel-Reduction Probe

- date: 2026-05-31
- stream: benchmark
- milestone: P8 retained unfiltered `SUM(int4)` aggregate kernel efficiency
- status: pass
- source_regression_report: `docs/testing/reports/2026-05-31-p8-25pct-regression-triage-v1.md`

## Result

The retained unfiltered `SUM(int4)` path no longer uses the original
one-thread full-row CUDA proof kernel. The runtime now launches a bounded
parallel grid with 256-thread blocks, each thread scans a strided slice of the
resident `int4` column and contributes its local `i64` sum through a device-side
atomic add into one scalar result.

CPU/GPU answer parity is preserved for normal, negative-value, and empty-table
inputs in the focused retained-device-memory test. The route still performs zero
H2D transfer and now reports the scalar `i64` D2H result explicitly: `8` bytes
per logical request.

## Evidence

- Focused retained-device-memory test passed for negative and empty inputs.
- Scaled guarded `--run-25pct-execute` evidence over 1,024 rows preserved
  formula-backed correctness and zero-H2D resident routing.
- Scaled metrics recorded `order_line_sum_amount` D2H as `8` bytes for one
  logical request and `80` bytes for ten logical requests.
- Scaled metrics recorded `order_line_sum_amount` CUDA event time as `8 us` for
  one logical request and `73 us` across ten logical requests.
- The same scaled run still reported the expected guarded blocker
  `full_25pct_requires_operator_long_run_after_streaming_boundary`; the full
  25% GPU DB and PostgreSQL artifacts from the prior reports were preserved.

## Scope Impact

`order_line_sum_amount` is now narrowed from "single-thread full-table retained
SUM proof" to "parallel retained SUM reduction needs refreshed full-tier
side-by-side evidence before the 25% regression can be reclassified." Retained
`MAX ... filter` remains open from the prior triage report. The 50%, 100%, and
200% tiers remain gated until the retained aggregate regressions are optimized
further, explicitly accepted as current non-claims, or refreshed with matching
side-by-side evidence.

## Validation Gate

- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine gpu_resident_device_memory_sum_probe_parallel_reduction_preserves_scalar_telemetry -- --nocapture`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`: passed
- `cargo check -p gpu_db_engine --example p8_ch_benchmark_residency_probe`: passed
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-sum-parallel-reduction GPU_DB_CH_BENCH_EXECUTE_ROWS=1024 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=256 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute`: expected guarded blocker after scaled execution
