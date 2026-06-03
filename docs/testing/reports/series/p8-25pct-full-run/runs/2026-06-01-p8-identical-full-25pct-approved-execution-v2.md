# P8 Identical Full 25% Approved Execution V2

- date: 2026-06-01
- stream: benchmark
- milestone: operator-approved full 25% identical pgwire benchmark execution
- status: blocked
- blocker: `gpu_db_endpoint_full_copy_admission_throughput_required`
- prior_blockers_closed: `engine_pgwire_full_copy_streaming_required`, `engine_pgwire_max_sessions_below_full_curve`
- requested_rows: `161,061,274`
- requested_concurrency_targets: `1,2,4,8,16,32,64,128`
- attempted_command: `GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-full-25pct-execution-v2 GPU_DB_CH_BENCH_ALLOW_FULL_IDENTICAL_PGWIRE_25PCT=1 GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=161061274 GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke`
- run_log: `target/autoloop-logs/2026-06-01-p8-identical-full-25pct-approved-execution-v2.log`
- partial_artifact_dir: `target/p8-identical-full-25pct-execution-v2/identical-pgwire-target-smoke`
- cleanup_status: PostgreSQL Docker cleanup passed; endpoint process was stopped; `.autoloop.lock` released

## Result

The approved full-run command was started after the required preflight
inspection and `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh` gate.
It closed the external PostgreSQL load phase but exposed a smaller GPU DB
endpoint execution blocker before any comparable latency/throughput curves were
produced.

Default PostgreSQL accepted the full streamed load:

- `default-postgresql-load.out`: `COPY 161061274`
- `default-postgresql-analyze.out`: `ANALYZE`
- `default-postgresql-load.err`: empty
- `default-postgresql-analyze.err`: empty
- captured settings: `target/p8-identical-full-25pct-execution-v2/identical-pgwire-target-smoke/postgresql-settings.tsv`

The GPU DB retained endpoint then started with the requested full-curve session
sizing:

- `max_sessions=766`
- `copy_chunk_rows_limit=8192`
- `create_table_into_engine_wal_mvcc=true`
- `copy_parser_in_protocol_lib=true`
- `copy_rows_committed_to_engine_wal_mvcc=true`

However, the endpoint only recorded 12 committed COPY chunks before the worker
stopped the attempt as unsafe to continue inside the automation window:
`12 * 8192 = 98,304` rows. Those facts were written between the endpoint start
and `2026-06-01 16:18:33 Europe/Amsterdam`, while the full target is
`161,061,274` rows. At that observed full-run admission rate, completing the GPU
DB load would not be a practical operator-approved benchmark run.

No graph-ready curve rows were admitted for this attempt:

- `metrics.jsonl`: 0 rows
- `concurrency-curve.csv`: not produced
- `identical-pgwire-target-smoke.md`: blocked before query metrics

## Narrowed Blocker

`gpu_db_endpoint_full_copy_admission_throughput_required`: bounded COPY chunks
are now functionally correct, but full 25% SQL-visible pgwire loading through
the engine-backed retained endpoint is too slow for the approved benchmark. The
next implementation slice should add or precisely block a bulk/streaming Engine
COPY admission path that preserves WAL/MVCC visibility, retained warmup, and
bounded memory while avoiding per-small-chunk overhead that makes a
161,061,274-row load impractical.

This is narrower than the earlier COPY/session readiness blockers: the endpoint
does commit bounded chunks and it is sized for the full client curve, but it is
not yet a defensible full 25% load target.

## Safe Continuation Command

After the endpoint full-COPY admission throughput blocker is fixed or narrowed,
retry with the same approved command and a fresh output directory:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-full-25pct-execution-v3 \
GPU_DB_CH_BENCH_ALLOW_FULL_IDENTICAL_PGWIRE_25PCT=1 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=161061274 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Use explicit port overrides if another local run owns the default PostgreSQL or
GPU DB endpoint ports.

## Cleanup

- `docker ps -a --format '{{.Names}} {{.Status}}' | rg 'gpu-db-p8|postgres'`:
  no GPU DB benchmark PostgreSQL containers remained.
- `pgrep -af 'p8_engine_pgwire|run_p8_ch_benchmark|psql postgresql://postgres@127.0.0.1:55437'`:
  no benchmark endpoint, harness, or GPU endpoint `psql` process remained.
- `.autoloop.lock`: absent after exit.

The first lock acquisition found a dead prior PID and archived the stale lock
before this run started.

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- full approved command: blocked at GPU DB endpoint full-COPY admission
- `git diff --check`: passed

Full 125% remains blocked by `missing_partitioned_over_resident_execution`.
