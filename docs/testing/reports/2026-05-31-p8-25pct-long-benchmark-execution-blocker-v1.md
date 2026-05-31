# P8 25% Long Benchmark Execution Blocker

- date: 2026-05-31
- stream: benchmark
- milestone: P8 CH-benCHmark-derived 25% / 6 GiB retained-residency long benchmark execution
- status: blocked
- validation_gate: `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`; `scripts/run_p8_ch_benchmark_residency_probe.sh --self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --streaming-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-upload-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-install-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`; `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct`; `cargo clippy --all-features --example p8_ch_benchmark_residency_probe -- -D warnings`; `git diff --check`
- validation_artifacts_before_cleanup: `target/p8-ch-benchmark-residency/estimate.md`; `target/p8-ch-benchmark-residency/25pct-preflight.md`; `target/p8-ch-benchmark-residency/pgsql-baseline/preflight.md`; `target/p8-ch-benchmark-residency/chunked-install-self-check/self-check.md`; `target/p8-ch-benchmark-residency/chunked-upload-self-check/self-check.md`

## Work Order

- active_lane: P8 CH-benCHmark-derived 25% / 6 GiB retained-residency long benchmark execution with PostgreSQL Docker baseline evidence
- classification: blocked after startability preflight
- falsifiable_claim: the repo can convert the checked 25% startability decision into a real 6 GiB retained-residency benchmark report, or name the exact missing execution path
- evidence_required: PostgreSQL baseline artifact, 25% disk/startability preflight, retained CUDA chunk upload, benchmark-only chunked resident-cache admission, zero-H2D route evidence, p95/p99/throughput/CUDA/fallback metrics, and cleanup verification
- non_goals: 50/100/200% tiers, the removed 400% tier, full CH-benCHmark/BenchBase compatibility, joins, transaction mix, normal SQL durability for generated benchmark artifacts, production cache-daemon scheduling, durable GPU pages, external orchestration, and optimization iterations
- minimum_meaningful_chunk: one completed 25% report or one narrower blocker
- stop_rule: stop after the 25% tier is classified completed, blocked, or aborted-by-guardrail

## Result

The 25% / 6 GiB tier is still blocked, but the blocker is narrower than the
previous resident-cache admission gap.

The current `--run-25pct` command is a startability preflight only. It writes
`target/p8-ch-benchmark-residency/25pct-preflight.md`, confirms disk capacity
and PostgreSQL comparator readiness, and exits before running GPU DB queries.
It does not invoke the Rust example in a mode that installs the estimated
161,061,274 generated `order_line` rows and measures p95/p99/throughput.

The existing Rust execution modes do not yet provide a defensible long-tier
runner:

- `Mode::Run` is the old calibration path. It is guarded by
  `GPU_DB_CH_BENCH_MAX_ROWS` and still seeds SQL-visible MVCC rows through
  `seed_engine(args.rows)`, so it cannot represent the benchmark-only generated
  6 GiB retained tier.
- `Mode::ChunkedInstallSelfCheck` proves the benchmark-only resident-cache
  admission API for small generated chunks, but it is a self-check. It does not
  schedule the 25% tier, does not emit long-run metrics, and its layout builder
  still accumulates all upload chunks in host memory before admission.
- `Engine::install_benchmark_relational_residency_chunks(...)` can install
  caller-provided chunks into an empty catalog table and keep
  `resident_rows_materialized: 0`, but the benchmark harness still lacks a
  production-scale 25% runner that streams/generates chunks, installs the
  resident snapshot, computes formula-backed expected answers, and emits the
  required long-run report.

## Latest 25% Startability Evidence

- tier: 25pct
- retained_target_bytes: 6442450944
- estimated_order_line_rows: 161061274
- generated_table_bytes: 15461882304
- wal_log_bytes: 7730941152
- required_disk_bytes: 23194920608
- disk_preflight: pass
- postgresql_baseline_preflight: pass
- run_preflight: ready_to_start
- blocker: missing_25pct_chunked_benchmark_execution_mode

The disposable PostgreSQL Docker baseline remains the required comparator path
for this dataset/query set. No GPU DB performance number should be interpreted
without a passed PostgreSQL baseline artifact.

## Narrowed Blocker

`missing_25pct_chunked_benchmark_execution_mode`

The next implementable slice is to add a checked Rust/script mode, separate
from the preflight, that:

1. derives the 25% row count from the retained target;
2. creates an empty `order_line` catalog table;
3. builds/adopts a bounded chunk producer for the generated resident layout
   without SQL-visible MVCC inserts and without one full decoded row snapshot;
4. installs the benchmark-only resident snapshot with retained CUDA chunks;
5. runs the supported aggregate query set at the selected logical concurrency
   targets;
6. validates answers from deterministic formulas rather than a full CPU MVCC
   mirror;
7. records p95/p99, throughput, H2D/D2H, CUDA event timing where available,
   zero-H2D accepted-route evidence, memory-pressure fallback behavior, disk
   and VRAM usage, raw JSONL, and cleanup verification.

Until that mode exists, rerunning `--run-25pct` would only reproduce the same
`ready_to_start` preflight and would not produce the required benchmark report.

## Cleanup

Generated artifacts are bounded under `target/p8-ch-benchmark-residency/`.
Cleanup verification passed after validation:

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup
scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down
```

- p8_ch_benchmark_cleanup: passed
- p8_ch_benchmark_pgsql_docker_cleanup: passed

The Docker baseline container is disposable and was removed after this blocker
run. A later full benchmark attempt should recreate the comparator artifact
before reporting GPU DB performance numbers.
