# P8 Identical Full 25% Execution

- date: 2026-06-01
- stream: benchmark
- milestone: operator-approved full 25% identical pgwire benchmark execution
- status: blocked
- blocker: `engine_pgwire_full_copy_streaming_required`
- secondary_blocker: `engine_pgwire_max_sessions_below_full_curve`
- requested_rows: `161,061,274`
- requested_concurrency_targets: `1,2,4,8,16,32,64,128`
- targets: default PostgreSQL, tuned PostgreSQL, GPU DB retained endpoint
- intended_command: `GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-full-25pct-execution-v1 GPU_DB_CH_BENCH_ALLOW_FULL_IDENTICAL_PGWIRE_25PCT=1 GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=161061274 GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke`

## Work Order

- active lane: P8 full 25% identical pgwire benchmark execution
- lane classification: operator-approved long run
- falsifiable claim: the guarded streamed harness can either run the 25% curves for default PostgreSQL, tuned PostgreSQL, and GPU DB retained endpoint through the same `psql`/libpq path, or precisely block before wasting the long-run window
- evidence required: code/readiness inspection, preserved full command, blocker evidence, artifact/report path, cleanup status
- non-goals: 125%, retired tiers, broad endpoint rewrites, README restructuring, or speculative optimization
- minimum meaningful chunk: one durable full-run decision report
- validation gate: `git diff --check` for docs-only changes
- stop rule: stop after recording the full-run blocker and handoff

## Result

The operator approval closed the previous governance blocker, but inspection of
the guarded full-run path found a smaller implementation blocker before launch:
the GPU DB retained endpoint is not yet a full 25% COPY-streaming target.

`scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke`
does stream the generated `CREATE TABLE` plus `COPY FROM STDIN` text into
`psql`, so it avoids the old full `load.sql` artifact. The target endpoint then
decodes each `CopyData` frame into `PendingCopy.rows: Vec<Vec<SqlValue>>` in
`crates/engine/examples/p8_engine_pgwire_benchmark_endpoint.rs` and only calls
`Engine::execute_relational_copy_rows(...)` after `CopyDone`.

For the requested `161,061,274` rows, that means the GPU DB endpoint would
retain the full decoded COPY payload in host memory before committing it into
Engine WAL/MVCC and warming `RelationalResidentCache`. That violates the
full-run intent: the external load is streamed, but the retained endpoint still
has an internal full-payload buffering boundary.

The same full curve also requests concurrency targets
`1,2,4,8,16,32,64,128`, but the harness starts the endpoint with
`GPU_DB_P8_ENGINE_PGWIRE_MAX_SESSIONS=64`. The GPU DB phase needs one load
session plus one `psql` session per query/client for three query families across
the requested curve. That exceeds 64 sessions before completing the full GPU DB
target schedule, so even after the PostgreSQL targets completed, the retained
endpoint would stop accepting clients before the 1..128 curve finished.

## Narrowed Blocker

`engine_pgwire_full_copy_streaming_required`: the endpoint needs a bounded COPY
admission path for the 25% row count. Acceptable fixes include incremental COPY
commit chunks, a bounded spill/chunk path that preserves WAL-before-visibility,
or another Engine-owned streaming API that avoids retaining the full decoded
COPY payload before commit.

`engine_pgwire_max_sessions_below_full_curve`: after COPY admission is bounded,
the full identical harness must size `GPU_DB_P8_ENGINE_PGWIRE_MAX_SESSIONS` from
the requested query schedule or keep the endpoint alive until the benchmark
driver finishes all requested clients.

## Safe Continuation Command

After those two endpoint/harness blockers are fixed, retry the approved command:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-full-25pct-execution-v1 \
GPU_DB_CH_BENCH_ALLOW_FULL_IDENTICAL_PGWIRE_25PCT=1 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=161061274 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Use explicit port overrides if another local run owns the default PostgreSQL or
GPU DB endpoint ports.

## Cleanup

No long-run comparator or endpoint process was started for this report, so no
benchmark cleanup was required. The existing Docker cleanup command remains:

```bash
scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down
```

Full 125% remains blocked by `missing_partitioned_over_resident_execution`.
