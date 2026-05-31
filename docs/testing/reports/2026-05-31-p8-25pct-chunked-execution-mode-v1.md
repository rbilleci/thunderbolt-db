# P8 25% Chunked Execution Mode Report

- date: 2026-05-31
- stream: benchmark
- milestone: P8 CH-benCHmark-derived 25% / 6 GiB retained-residency chunked execution mode
- status: blocked
- narrowed_blocker: `missing_streaming_chunk_iterator_for_full_25pct_execution`
- environment_blocker: `pgsql_baseline_docker_cleanup_permission_denied`
- validation_gate: implementation gates passed through `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`; `scripts/run_p8_ch_benchmark_residency_probe.sh --streaming-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-upload-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-install-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`; `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct`; `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute` with the expected blocker; `cargo clippy --all-features --example p8_ch_benchmark_residency_probe -- -D warnings`; `cargo fmt --all -- --check`; `git diff --check`. Final cleanup attempted and now correctly reports failure because Docker refused to stop/remove the existing comparator container.
- validation_artifacts_before_cleanup: `target/p8-ch-benchmark-residency-report/25pct-execution.md`; `target/p8-ch-benchmark-residency-report/25pct-preflight.md`; `target/p8-ch-benchmark-residency-report/pgsql-baseline/preflight.md`; `target/p8-ch-benchmark-residency-report/chunked-execute/execution.md`; `target/p8-ch-benchmark-residency-report/chunked-execute/metrics.jsonl`

## Work Order

- active_lane: P8 CH-benCHmark-derived 25% / 6 GiB retained-residency chunked execution mode
- lane_classification: open
- falsifiable_claim: the harness can either run a checked 25% chunked execution command with PostgreSQL comparator evidence and formula-backed result validation, or narrow the remaining full-tier execution blocker
- evidence_required: 25% tier row derivation, empty catalog-table creation, generated benchmark-only resident chunk install, supported aggregate query execution, deterministic formula-backed answers, PostgreSQL Docker baseline, p95/p99/throughput/CUDA/zero-H2D/fallback metrics, and cleanup verification
- non_goals: 50/100/200% tiers, the removed 400% tier, joins, transaction mix, full CH-benCHmark/BenchBase compatibility, normal SQL durability for generated artifacts, durable GPU pages, production cache-daemon scheduling, external orchestration, and kernel/cache optimization
- minimum_meaningful_chunk: one checked execution command plus either a completed 25% report or a narrower blocker
- stop_rule: stop after a checked 25% report or one precise blocker

## Result

The repo now has a checked `--run-25pct-execute` command separate from the safe
`--run-25pct` startability preflight.

The command:

1. starts the disposable PostgreSQL Docker baseline and records the comparator
   preflight for the deterministic `order_line` dataset/query set;
2. reruns the 25% disk/comparator preflight and derives the 161,061,274-row
   target for the 6 GiB retained tier;
3. creates an empty `order_line` catalog table;
4. installs benchmark-only generated resident chunks through retained CUDA
   device memory without SQL-visible MVCC inserts and with
   `resident_rows_materialized: 0`;
5. executes the supported aggregate query set at the selected logical
   concurrency targets;
6. validates answers from deterministic formulas rather than a full CPU MVCC
   mirror;
7. emits raw JSONL and a human-readable execution report with p95/p99,
   throughput, H2D/D2H, CUDA-event timing where available, zero-H2D accepted
   route evidence, and memory-pressure fallback behavior.

By default the command runs a guarded scaled execution (`executed_rows: 1024`)
instead of attempting the full 161,061,274-row tier. That is intentional: the
current Rust example still builds and owns the full `Vec<ResidentUploadChunk>`
before admission, and the text-column layout builds offsets/bytes in caller
memory. Running the full 25% tier in the current 6-hour worker window would
risk turning a proven execution path into an unsafe host-memory/runtime test.

## Operator Summary

| item | status | evidence |
|---|---:|---|
| PostgreSQL Docker comparator | pass | `target/p8-ch-benchmark-residency-report/pgsql-baseline/preflight.md` |
| 25% startability preflight | pass | `target/p8-ch-benchmark-residency-report/25pct-preflight.md` |
| checked chunked execution command | pass, scaled | `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute` |
| formula-backed validation | pass | `expected_source:"formula"` in `metrics.jsonl` |
| zero-H2D resident route evidence | pass | all chunked metrics reported `resident_route_zero_h2d:true` |
| memory-pressure fallback probe | pass | `memory_pressure_route_accepted:false`, reason `resident snapshot is InvalidatedByMemoryPressure` |
| full 25% / 6 GiB execution | blocked | `missing_streaming_chunk_iterator_for_full_25pct_execution` |
| target artifact cleanup | pass | generated `target/p8-ch-benchmark-residency*` artifacts removed after validation |
| Docker baseline cleanup | blocked | Docker refused to stop/remove existing `gpu-db-p8-pgsql-baseline` with `permission denied` |

## Scaled Execution Evidence

- retained_target_bytes: 6442450944
- estimated_order_line_rows: 161061274
- generated_table_bytes: 15461882304
- wal_log_bytes: 7730941152
- required_disk_bytes: 23194920608
- available_disk_bytes: 377179029504
- executed_rows: 1024
- execute_chunk_rows: 256
- upload_chunks: 24
- resident_bytes: 30736
- resident_rows_materialized: 0
- PostgreSQL baseline artifact: `target/p8-ch-benchmark-residency-report/pgsql-baseline/preflight.md`
- raw metrics artifact: `target/p8-ch-benchmark-residency-report/chunked-execute/metrics.jsonl`

Selected metrics from the guarded run:

| query | logical_requests | p95_us | p99_us | throughput_qps | h2d_bytes_total | d2h_bytes_total | cuda_event_samples | zero_h2d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order_line_count_all | 1 | 262 | 262 | 3759.003 | 0 | 0 | 1 | true |
| order_line_sum_amount | 1 | 191 | 191 | 5149.437 | 0 | 0 | 1 | true |
| order_line_avg_quantity_between | 1 | 298 | 298 | 3326.802 | 0 | 3352 | 1 | true |
| order_line_max_amount_filter | 1 | 269 | 269 | 3688.281 | 0 | 4044 | 1 | true |
| order_line_count_all | 10 | 135 | 135 | 8026.268 | 0 | 0 | 10 | true |
| order_line_sum_amount | 10 | 166 | 166 | 6284.091 | 0 | 0 | 10 | true |
| order_line_avg_quantity_between | 10 | 285 | 285 | 3730.379 | 0 | 33520 | 10 | true |
| order_line_max_amount_filter | 10 | 267 | 267 | 3811.250 | 0 | 40440 | 10 | true |

## Exact Full-Run Command

The full 25% attempt is deliberately opt-in until the remaining blocker is
accepted for a longer operator window or fixed with a streaming upload boundary:

```bash
GPU_DB_CH_BENCH_ALLOW_FULL_25PCT=1 GPU_DB_CH_BENCH_EXECUTE_CHUNK_ROWS=1048576 scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct-execute
```

## Cleanup

Generated artifacts are bounded under `target/`. Target artifact cleanup passed:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-ch-benchmark-residency-report scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup
```

- p8_ch_benchmark_cleanup: passed

Docker baseline cleanup is blocked by the local Docker runtime:

```text
Error response from daemon: cannot stop container: gpu-db-p8-pgsql-baseline: permission denied
Error response from daemon: cannot remove container "gpu-db-p8-pgsql-baseline": could not kill container: permission denied
```

- container: `gpu-db-p8-pgsql-baseline`
- observed_state: running
- observed_pid: 784910
- started_at: 2026-05-30T20:05:11.980640452Z
- worker_attempted_cleanup: yes
- docker_cleanup_status: blocked
- blocker: `pgsql_baseline_docker_cleanup_permission_denied`
