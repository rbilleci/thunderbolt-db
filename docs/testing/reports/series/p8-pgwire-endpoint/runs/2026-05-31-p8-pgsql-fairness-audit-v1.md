# P8 PostgreSQL Fairness Audit

- date: 2026-05-31
- stream: benchmark
- milestone: P8 25% PostgreSQL fairness and true-concurrency gate
- status: blocked
- blocker: gpu_db_protocol_benchmark_path_required
- secondary_blocker: identical_pg_client_harness_required

## Result

The fairness/concurrency gate is now actionable but not admitted. The harness
has a checked `--pgsql-fairness-audit` command that records scaled default and
tuned PostgreSQL evidence, including host/container facts, selected PostgreSQL
settings, repeated warm `EXPLAIN (ANALYZE, FORMAT JSON)` samples, btree/BRIN
tuned-profile DDL, machine-readable metrics, and a graph-ready true-concurrency
plan for 1, 2, 4, 8, 16, 32, 64, and 128 clients.

The current 25% GPU DB retained aggregate evidence remains `engine_internal`.
PostgreSQL measurements enter through a PostgreSQL client path, while the GPU
DB retained benchmark still enters through the Rust example with direct
`Engine::new_local()` / `execute_relational_select` calls. A headline
PostgreSQL-vs-GPU product latency or concurrency claim therefore requires a
PostgreSQL-compatible GPU DB benchmark target so the same driver/client harness
can run both sides with only target/profile parameters changed.

## Command

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-pgsql-fairness-audit \
  GPU_DB_CH_BENCH_PGSQL_AUDIT_ROWS=64 \
  GPU_DB_CH_BENCH_PGSQL_AUDIT_REPEATS=1 \
  scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-fairness-audit
```

## Artifacts

| artifact | path |
|---|---|
| audit report | `target/p8-pgsql-fairness-audit/pgsql-fairness-audit/fairness-audit.md` |
| raw metrics | `target/p8-pgsql-fairness-audit/pgsql-fairness-audit/metrics.jsonl` |
| host facts | `target/p8-pgsql-fairness-audit/pgsql-fairness-audit/host-facts.txt` |
| PostgreSQL settings | `target/p8-pgsql-fairness-audit/pgsql-fairness-audit/postgresql-settings.tsv` |
| concurrency plan | `target/p8-pgsql-fairness-audit/pgsql-fairness-audit/concurrency-curve-plan.csv` |

## Source-Truth Update

The 25% retained aggregate result from
`docs/testing/reports/series/p8-25pct-full-run/runs/2026-05-31-p8-25pct-aggregate-refresh-after-between-v1.md`
must be treated as provisionally admitted engine-internal evidence. It is not a
headline product/client latency or real-world throughput claim until the
protocol-parity blocker is closed and true concurrent-client curves are
collected through the same benchmark harness for default PostgreSQL, tuned
PostgreSQL, and GPU DB.

The 125% tier remains deferred on `missing_partitioned_over_resident_execution`.
