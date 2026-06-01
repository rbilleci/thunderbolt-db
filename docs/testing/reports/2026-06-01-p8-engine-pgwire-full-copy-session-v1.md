# P8 Engine Pgwire COPY/Session Readiness

- date: 2026-06-01
- stream: benchmark
- milestone: engine-backed pgwire full-run COPY/session readiness
- status: closed
- closed_blockers: `engine_pgwire_full_copy_streaming_required`, `engine_pgwire_max_sessions_below_full_curve`
- requested_rows: `161,061,274`
- requested_concurrency_targets: `1,2,4,8,16,32,64,128`
- scaled_smoke_artifact: `target/p8-engine-pgwire-full-copy-session-v1-smoke/identical-pgwire-target-smoke/identical-pgwire-target-smoke.md`
- endpoint_facts: `target/p8-engine-pgwire-full-copy-session-v1-smoke/identical-pgwire-target-smoke/endpoint-facts.txt`
- metrics_artifact: `target/p8-engine-pgwire-full-copy-session-v1-smoke/identical-pgwire-target-smoke/metrics.jsonl`
- curve_artifact: `target/p8-engine-pgwire-full-copy-session-v1-smoke/identical-pgwire-target-smoke/concurrency-curve.csv`
- cleanup_status: PostgreSQL Docker cleanup passed

## Result

The engine-backed retained pgwire endpoint no longer retains the whole decoded
COPY payload before Engine admission. `PendingCopy` keeps a bounded decoded row
buffer, controlled by `GPU_DB_P8_ENGINE_PGWIRE_COPY_CHUNK_ROWS` and defaulting
to `8192`, and sends each ready chunk through the owner-thread Engine scheduler.
Each chunk is committed through `Engine::execute_relational_copy_rows(...)`
before the final retained warmup. The endpoint then warms only after copied rows
are visible in Engine WAL/MVCC state.

The identical target harness now sizes `GPU_DB_P8_ENGINE_PGWIRE_MAX_SESSIONS`
from the requested GPU DB schedule: one load session plus three query families
times the sum of requested client counts. For the full requested curve
`1,2,4,8,16,32,64,128`, that is `766` endpoint sessions instead of the previous
fixed `64`.

## Scaled Evidence

Validation used a 16-row scaled identical pgwire smoke with concurrency `1,2`
and `GPU_DB_P8_ENGINE_PGWIRE_COPY_CHUNK_ROWS=5`.

Endpoint facts recorded:

- `max_sessions=10`
- `copy_chunk_rows_limit=5`
- `copy_streaming_bounded_chunks=true`
- `copy_committed_chunks=4`
- `copy_max_buffered_decoded_rows=5`
- `copy_rows_decoded_by_protocol=16`
- `sql_visible_resident_row_count=16`
- `sql_visible_resident_device_memory_retained=true`
- retained zero-H2D route facts for `COUNT(*)`, int4 lookup, and composite/text lookup
- device-side match-index compaction for retained equality lookup shapes

The smoke passed across default PostgreSQL, tuned PostgreSQL, and the GPU DB
retained endpoint with the same `psql`/libpq boundary, query text, concurrency
schedule, and graph-ready metric schema. No `load.sql` artifact was produced.

## Safe Continuation Command

The full 25% identical default/tuned/GPU retained run is approved and can be
started by a later worker/supervisor round with:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-full-25pct-execution-v1 \
GPU_DB_CH_BENCH_ALLOW_FULL_IDENTICAL_PGWIRE_25PCT=1 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=161061274 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Use explicit port overrides if another local run owns the default PostgreSQL or
GPU DB endpoint ports.

## Validation

- `cargo fmt --all -- --check`: passed
- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `GPU_DB_CH_BENCH_OUT_DIR=target/p8-engine-pgwire-full-copy-session-v1-smoke GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=16 GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2 GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55524 GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55525 GPU_DB_P8_ENGINE_PGWIRE_COPY_CHUNK_ROWS=5 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke`: passed
- `test ! -e target/p8-engine-pgwire-full-copy-session-v1-smoke/identical-pgwire-target-smoke/load.sql`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed

Full 125% remains blocked by `missing_partitioned_over_resident_execution`.
