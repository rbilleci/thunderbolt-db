# P8 Streaming Generator Prerequisite Probe

- date: 2026-05-30
- stream: benchmark
- milestone: P8 CH-benCHmark-derived streaming/on-disk generation and chunked admission prerequisite
- validation_gate: `scripts/run_p8_ch_benchmark_residency_probe.sh --dry-run`; `scripts/run_p8_ch_benchmark_residency_probe.sh --self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --streaming-self-check`; `scripts/run_p8_ch_benchmark_residency_probe.sh --run-25pct`; `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`; `cargo clippy --all-features --example p8_ch_benchmark_residency_probe -- -D warnings`; `git diff --check`
- report_artifacts: `target/p8-ch-benchmark-residency/estimate.md`; `target/p8-ch-benchmark-residency/streaming-order-line/self-check.md`; `target/p8-ch-benchmark-residency/25pct-preflight.md`

## Result

This slice narrows the previous broad 25% / 6 GiB blocker.

The benchmark harness now has a checked benchmark-only streaming generator
self-check. It writes deterministic `order_line` rows directly to chunk files
under `target/p8-ch-benchmark-residency/streaming-order-line/` and records a
JSONL manifest without seeding the MVCC engine or collecting all generated rows
in memory.

The 25% tier still does not start. The remaining blocker is now the narrower
engine/cache boundary:

```text
missing_relational_resident_cache_chunked_install_api
```

`RelationalResidentCache` is currently populated through
`Engine::populate_relational_residency_snapshot(...)`, which scans MVCC state,
collects `resident_rows: Vec<Vec<SqlValue>>`, builds one contiguous device
payload, and then admits the snapshot. The generator side can be bounded, but
the cache install/admission side still needs a chunked API before the 6 GiB
tier can safely begin.

## PostgreSQL Baseline Requirement

P8 benchmark results must be compared against PostgreSQL for the same generated
dataset and query set before any GPU DB performance claim is accepted. The
harness now includes:

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-preflight
```

That preflight writes a deterministic PostgreSQL workload SQL file under
`target/p8-ch-benchmark-residency/pgsql-baseline/` and requires
`GPU_DB_CH_BENCH_PGSQL_URL` to point at a disposable PostgreSQL database. If no
PostgreSQL connection is configured, the preflight blocks the benchmark with
`missing_pgsql_baseline_connection` rather than allowing standalone GPU DB
numbers.

## Scope Boundary

The streaming artifact path is benchmark-only. It does not claim normal SQL
durability, WAL-before-visibility insertion, BenchBase compatibility, joins,
transaction mix, durable GPU pages, production cache-daemon behavior, or
completed 6 GiB residency evidence.

Because the generated artifacts are recoverable from deterministic generator
inputs and live only under `target/`, cleanup remains:

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup
```

## 25% Preflight Decision

The 25% preflight still estimates the tier as:

- retained_target_bytes: `6442450944`
- estimated_order_line_rows: `161061274`
- generated_table_bytes: `15461882304`
- wal_log_bytes: `7730941152`
- report_bytes: `2097152`

Disk preflight passes on this host, but run preflight is blocked until the
resident cache can install generated column chunks without materializing all
rows and the whole retained device payload in process memory.

## Next Defensible Slice

Add a narrow `RelationalResidentCache` / benchmark-only resident snapshot
builder that can install generated `int4` and `text` column chunks from the
streaming artifacts, preserve explicit benchmark-only durability boundaries,
record admission/budget/route telemetry, require a passed PostgreSQL baseline
artifact for the same dataset/query set, and execute the existing retained query
kernels without requiring `resident_rows: Vec<Vec<SqlValue>>` for the whole
tier.
