# P8 Benchmark Chunked Resident-Cache Admission

- date: 2026-05-31
- stream: benchmark
- milestone: P8 CH-benCHmark-derived benchmark-only relational resident-cache chunked admission prerequisite
- validation_gate: `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`; `scripts/run_p8_ch_benchmark_residency_probe.sh --self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --streaming-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-upload-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --chunked-install-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-preflight`; `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct`; `cargo clippy --all-features --example p8_ch_benchmark_residency_probe -- -D warnings`; `git diff --check`
- report_artifacts: `target/p8-ch-benchmark-residency/chunked-install-self-check/self-check.md`; `target/p8-ch-benchmark-residency/25pct-preflight.md`; `target/p8-ch-benchmark-residency/pgsql-baseline/preflight.md`

## Work Order

- active_lane: P8 CH-benCHmark-derived benchmark-only relational resident-cache chunked admission path for the 25% / 6 GiB retained-residency tier
- classification: closed prerequisite
- falsifiable_claim: generated `order_line` chunks can be admitted into retained resident routes without whole-tier `resident_rows` materialization, while PostgreSQL baseline gating and SQL durability boundaries stay explicit
- evidence_required: focused chunked-admission self-check, PostgreSQL baseline gate, 25% preflight decision, cleanup verification, and focused Rust checks
- non_goals: normal SQL durability for generated artifacts, full CH-benCHmark/BenchBase compatibility, joins, transaction mix, production cache daemon behavior, durable GPU pages, and 50/100/200% tiers
- minimum_meaningful_chunk: prove benchmark-only chunked admission or produce a narrower blocker
- stop_rule: stop after the 25% tier is startable or a narrower blocker replaces `missing_benchmark_only_relational_resident_cache_chunked_admission_api`

## Result

The benchmark-only resident-cache admission blocker is closed for the checked
`order_line` layout.

`Engine::install_benchmark_relational_residency_chunks(...)` now installs
generated retained chunks into `RelationalResidentCache` for an empty catalog
table. The API is deliberately named and constrained as benchmark-only: it
requires no SQL-visible rows, validates catalog `int4`/`text` layout names,
copies caller-provided chunks through retained CUDA device memory, and creates a
resident snapshot with `resident_rows: Vec::new()`.

The focused self-check passed with:

- rows: 64
- chunk_rows: 16
- chunks: 4
- resident_rows_materialized: 0
- route_accepted: true
- zero_h2d_route: true

The self-check also executed the supported benchmark aggregate set through
resident routes and validated deterministic results for `COUNT(*)`,
`SUM(ol_amount)`, `AVG(ol_quantity) WHERE ol_quantity BETWEEN 10 AND 40`, and
`MAX(ol_amount) WHERE ol_amount >= 16`.

## 25% Preflight Decision

`scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct` now reports the
25% / 6 GiB tier as `ready_to_start` on this host when the PostgreSQL Docker
baseline has passed.

Latest preflight evidence:

- estimated_order_line_rows: 161061274
- retained_target_bytes: 6442450944
- required_disk_bytes: 23194920608
- available_disk_bytes: 377177317376
- disk_preflight: pass
- postgresql_baseline_preflight: pass
- blocker: none

This is a startability decision, not a completed 6 GiB benchmark result. No
50%, 100%, or 200% tier was attempted.

## PostgreSQL Baseline Boundary

The PostgreSQL comparator gate remains required. The disposable Docker
`postgres:16` path passed locally with PostgreSQL 16.14 and wrote the same
deterministic `order_line` workload SQL/output under
`target/p8-ch-benchmark-residency/pgsql-baseline/`.

No GPU DB performance number should be interpreted without a passed PostgreSQL
baseline artifact for the same generated dataset/query set.

## Next Defensible Slice

Run the operator-approved 25% / 6 GiB long benchmark tier with the checked
PostgreSQL baseline, record p95/p99/throughput/CUDA timing/zero-H2D/fallback
evidence, and verify cleanup. Keep 50%, 100%, and 200% gated behind the 25%
report.
