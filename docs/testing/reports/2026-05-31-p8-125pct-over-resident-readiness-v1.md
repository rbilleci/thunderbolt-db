# P8 125% Over-Resident Readiness

- date: 2026-05-31
- stream: benchmark
- milestone: P8 125% / about 30 GiB over-resident readiness
- status: blocked
- blocker: missing_partitioned_over_resident_execution
- readiness_artifact: `target/p8-125pct-readiness/125pct-readiness.md`
- raw_metrics: `target/p8-125pct-readiness/chunked-execute/metrics.jsonl`

## Result

The 125% tier is now actionable, but it is not safe to schedule as a full
PostgreSQL-vs-GPU long run yet. The checked readiness command computes the
805,306,368-row retained target, records local disk/GPU/PostgreSQL comparator
facts, proves retired 50%, 100%, 200%, and 400% tiers remain absent from dry-run
output, and runs a bounded scaled execution probe that verifies zero-H2D
resident routes plus memory-pressure rejection.

The blocker is narrower than a missing benchmark command: the current GPU DB
execution path installs one retained CUDA resident layout. The 125% target is
about 30 GiB retained, while the local RTX 3090 reports 24,576 MiB total and
22,830 MiB free during the readiness run. A defensible full 125% tier therefore
requires partitioned or streamed over-resident execution before an
operator-approved long-run window can compare against a same-row-count
PostgreSQL baseline.

## Commands

```bash
bash -n scripts/run_p8_ch_benchmark_residency_probe.sh
GPU_DB_CH_BENCH_OUT_DIR=target/p8-125pct-readiness GPU_DB_CH_BENCH_EXECUTE_ROWS=1024 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=256 GPU_DB_CH_BENCH_ACCEPT_SCALED_125PCT=1 scripts/run_p8_ch_benchmark_residency_probe.sh --run-125pct
```

## Readiness Facts

- retained_target_bytes: 32,212,254,720
- estimated_order_line_rows: 805,306,368
- retained_column_bytes: 32,212,254,720
- generated_table_bytes: 77,309,411,328
- wal_log_bytes: 38,654,705,664
- report_bytes: 2,097,152
- required_disk_bytes: 115,966,214,144
- available_disk_bytes: 365,625,319,424
- disk_preflight: pass
- local_gpu_memory_total_mib: 24,576
- local_gpu_memory_free_mib: 22,830
- gpu_preflight: pass
- postgresql_baseline_preflight: pass
- retired_50_100_200_400pct_tiers_absent: true
- scaled_probe_rows: 1,024
- scaled_probe_chunk_rows: 256

## Guarded Full Commands

These commands are intentionally not admitted yet:

```bash
GPU_DB_CH_BENCH_ALLOW_FULL_125PCT=1 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=1048576 scripts/run_p8_ch_benchmark_residency_probe.sh --run-125pct
GPU_DB_CH_BENCH_PGSQL_URL='postgresql://postgres:gpu_db_p8_benchmark@127.0.0.1:55434/gpu_db_p8_baseline' GPU_DB_CH_BENCH_ALLOW_FULL_PGSQL_125PCT=1 GPU_DB_CH_BENCH_PGSQL_ROWS=805306368 scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-125pct-latency
```

## Decision

The 25% retained aggregate claim set remains admitted. The 125% gate is blocked
until a partitioned or streamed over-resident execution path exists for the GPU
side, followed by explicit operator approval for the long PostgreSQL and GPU DB
same-row-count runs. Do not revive the retired 50%, 100%, 200%, or 400% tiers.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- scaled `--run-125pct` readiness with `GPU_DB_CH_BENCH_ACCEPT_SCALED_125PCT=1`: passed
