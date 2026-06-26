# P8 Identical PG-Client Concurrency Runner V2

- date: 2026-06-01
- stream: benchmark
- milestone: P8 identical PostgreSQL-compatible true-concurrency runner
- status: blocked
- blocker: `engine_pgwire_endpoint_concurrency_unsafe`
- smallest_next_unblocker: `engine_pgwire_session_scheduler_required`
- smoke_command: `scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke`
- smoke_artifact: `target/p8-identical-pg-client-concurrency-runner-v2/engine-backed-pgwire-concurrency-smoke/engine-backed-pgwire-concurrency-smoke.md`
- metrics_artifact: `target/p8-identical-pg-client-concurrency-runner-v2/engine-backed-pgwire-concurrency-smoke/metrics.jsonl`
- curve_artifact: `target/p8-identical-pg-client-concurrency-runner-v2/engine-backed-pgwire-concurrency-smoke/concurrency-curve.csv`

## Result

This slice narrowed the identical-client concurrency runner blocker to the
engine-backed endpoint session model. The retained endpoint is still valid for
single-client `psql`/libpq smokes: SQL-visible `CREATE TABLE` plus
`COPY FROM STDIN` reaches Engine WAL/MVCC state, and retained `COUNT(*)` plus
bounded multi-column int4 lookup run through retained zero-H2D routes.

The endpoint cannot honestly be turned into a threaded concurrent client target
with the current state ownership. A direct implementation attempt using
`Arc<Mutex<EndpointState>>` and one client handler thread per accepted session
failed at compile time because `EndpointState` owns `Engine`, `Engine` owns
`RelationalResidentCache`, and retained CUDA memory contains a non-`Send`
`*mut c_void` pointer. That is the right safety boundary: marking CUDA resident
memory as thread-safe just to create a benchmark would be unsafe and would make
the resulting evidence suspect.

## Evidence

The focused command now writes a graph-ready blocked curve artifact for the GPU
DB retained endpoint:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-pg-client-concurrency-runner-v2 \
  GPU_DB_CH_BENCH_ENGINE_PGWIRE_ROWS=16 \
  GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55442 \
  GPU_DB_CH_BENCH_ENGINE_PGWIRE_CONCURRENCY_TARGETS=1,2 \
  scripts/run_p8_ch_benchmark_residency_probe.sh --engine-backed-pgwire-concurrency-smoke
```

The CSV records the same target/profile/client/query schema expected by the
future runner, with concurrency `1` labeled as already covered by the existing
single-client smoke and concurrency `2` blocked by
`engine_pgwire_endpoint_concurrency_unsafe`.

## Rejected Alternatives

- Do not force `Send`/`Sync` on retained CUDA pointers to make the benchmark
  compile.
- Do not run concurrent clients against the broader `gpu-db-server` and label
  it retained-route evidence; that path measures protocol `SharedCatalog`
  state, not the P8 retained `Engine` route.
- Do not compare PostgreSQL true-concurrency timings to a single-threaded GPU
  DB endpoint and call that an identical-client concurrency result.

## Next Unblock Trigger

Add a bounded engine pgwire session scheduler that keeps `Engine` and retained
CUDA memory on their owning thread while concurrent client sessions send parsed
simple-query/COPY work to that owner and receive backend responses. That should
enable the shared `psql`/libpq runner contract for default PostgreSQL, tuned
PostgreSQL, and GPU DB retained endpoint targets without violating CUDA memory
ownership.

Composite/text lookup remains blocked by
`retained_composite_or_text_lookup_required`. Full 125% remains blocked by
`missing_partitioned_over_resident_execution`.

## Validation Gate

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo check -p gpu_db_engine --example p8_engine_pgwire_benchmark_endpoint`: passed
- focused scaled `--engine-backed-pgwire-concurrency-smoke`: passed with
  blocker `engine_pgwire_endpoint_concurrency_unsafe`
