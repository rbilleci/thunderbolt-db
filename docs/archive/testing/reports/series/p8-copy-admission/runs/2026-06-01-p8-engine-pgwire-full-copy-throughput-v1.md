# P8 Engine Pgwire Full COPY Throughput

- date: 2026-06-01
- stream: benchmark
- milestone: engine-backed pgwire full-COPY admission throughput/lifecycle
- status: blocked
- narrowed_blocker: `engine_sql_visible_mvcc_bulk_copy_admission_required`
- previous_blocker: `gpu_db_endpoint_full_copy_admission_throughput_required`
- code_change: current-process COPY applies already-decoded rows directly while preserving durable INSERT WAL payloads for replay
- smoke_artifact: `target/p8-full-copy-throughput-fastpath-smoke/identical-pgwire-target-smoke/identical-pgwire-target-smoke.md`
- bounded_probe_artifact: `target/p8-full-copy-throughput-direct-65536/identical-pgwire-target-smoke/endpoint-facts.txt`
- stopped_probe_cleanup: PostgreSQL Docker cleanup passed; no endpoint or benchmark `psql` process remained

## Result

The endpoint no longer reparses the just-decoded COPY chunk in the current
process before applying it to Engine state. `Engine::execute_relational_copy_rows(...)`
now still writes a durable SQL `INSERT` payload to WAL for recovery/replay, but
the live process preflights the already-decoded `Insert` and applies that
structure directly after WAL flush/replication commit. This preserves the
current WAL-before-visibility boundary and avoids adding a second non-SQL
durability format for this benchmark endpoint.

The endpoint also records per-COPY-chunk facts:

- `copy_current_process_decoded_apply_fast_path=true`
- `copy_chunk_rows=<rows>`
- `copy_chunk_elapsed_ms=<milliseconds>`
- `copy_chunk_rows_per_sec=<integer rows/sec>`

The scaled 16-row identical pgwire smoke passed after the change with
`GPU_DB_P8_ENGINE_PGWIRE_COPY_CHUNK_ROWS=5`, preserving SQL-visible
`CREATE TABLE` plus `COPY FROM STDIN`, retained warmup, and retained lookup
queries through the real `psql`/libpq boundary.

## Bounded Throughput Probe

A bounded 65,536-row probe was started with the same endpoint path and the
default 8,192-row COPY chunk size:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-full-copy-throughput-direct-65536 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=65536 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55546 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55547 \
GPU_DB_P8_ENGINE_PGWIRE_COPY_CHUNK_ROWS=8192 \
timeout 900 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

The worker stopped the probe after five committed chunks because the measured
trend was already indefensible for the full 161,061,274-row load:

| committed chunk | rows | elapsed ms | rows/sec |
|---:|---:|---:|---:|
| 1 | 8,192 | 3,426 | 2,391 |
| 2 | 8,192 | 9,843 | 832 |
| 3 | 8,192 | 16,395 | 499 |
| 4 | 8,192 | 23,473 | 348 |
| 5 | 8,192 | 29,264 | 279 |

That proves the prior parser round-trip was not the only blocker. Even after
current-process decoded apply, SQL-visible per-row MVCC materialization,
per-column value-index maintenance, row-key allocation, and full SQL WAL payload
rendering still grow too expensive as visible row count increases.

## Narrowed Blocker

`engine_sql_visible_mvcc_bulk_copy_admission_required`: the full 25% identical
run needs a defended bulk COPY admission design for SQL-visible rows, not just
larger protocol chunks. A viable next slice must preserve WAL-before-visibility,
durable replay, MVCC visibility, resident warmup invalidation, and the
PostgreSQL-compatible `COPY FROM STDIN` boundary while avoiding the current
per-row/per-chunk Engine storage and SQL payload overhead that makes
161,061,274-row endpoint loading impractical.

The full approved benchmark should not be relaunched until that storage/WAL
admission boundary is implemented or narrowed further. Full 125% remains
blocked by `missing_partitioned_over_resident_execution`.

## Safe Continuation Command

After a bulk SQL-visible MVCC COPY admission slice exists, retry a bounded probe
before the full approved run:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-full-copy-throughput-next-probe \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=65536 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55550 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55551 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Only if that bounded probe shows stable defended admission throughput should a
later worker retry the approved full command with a fresh output directory:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-full-25pct-execution-v3 \
GPU_DB_CH_BENCH_ALLOW_FULL_IDENTICAL_PGWIRE_25PCT=1 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=161061274 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine relational_copy_rows_commit_through_engine_wal_mvcc -- --nocapture`: passed
- scaled endpoint smoke: passed
- bounded 65,536-row probe: deliberately stopped after five chunks with the
  narrowed blocker above
- PostgreSQL Docker cleanup: passed
