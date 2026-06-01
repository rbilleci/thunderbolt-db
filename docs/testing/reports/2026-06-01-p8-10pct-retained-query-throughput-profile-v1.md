# P8 10% Retained Query Throughput Profile

- date: 2026-06-01
- stream: benchmark
- milestone: 10% retained endpoint query phase profiling and bounded fix
- status: closed_with_blocker
- previous_blocker: `gpu_db_10pct_retained_query_throughput_profile_required`
- narrowed_blocker: `gpu_db_10pct_copy_admission_below_30000_rows_per_sec_recheck_required`
- baseline_artifact: `target/p8-10pct-retained-query-throughput-profile-v1-probe/identical-pgwire-target-smoke/`
- after_fix_artifact: `target/p8-10pct-retained-query-throughput-profile-v1-after-fix/identical-pgwire-target-smoke/`
- cleanup_status: PostgreSQL Docker cleanup passed; no endpoint, benchmark `psql`, or comparator container process remained

## Result

The worker profiled the retained endpoint query path after the first 10% run
showed retained `COUNT(*)` and int4 lookup around `14s`, and composite/text
lookup around `190s`. Two bounded retained-query setup costs were removed:

- retained `COUNT(*)` and retained equality projection no longer clone the full
  `RelationalResidencySnapshot` and its SQL-visible `resident_rows` payload
  before using retained device-memory metadata.
- conjunctive equality access-path setup now intersects the existing
  relational value-index key sets instead of full-scanning MVCC rows before the
  retained match-index route executes.

This preserves SQL-visible `CREATE TABLE` plus PostgreSQL-compatible
`COPY FROM STDIN`, Engine WAL/MVCC row visibility, retained warmup from
SQL-visible rows, zero-H2D retained routes, device-side match-index compaction,
selected-row/result-sized D2H accounting, and owner-thread Engine/CUDA state.

## Commands

Baseline/current-process phase probe before the conjunction fix:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-retained-query-throughput-profile-v1-probe \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55564 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55565 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

After-fix proof:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-retained-query-throughput-profile-v1-after-fix \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55566 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55567 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

## Phase Evidence

All retained GPU DB query rows passed correctness with zero errors.

| shape | before p50 us | after p50 us | after engine us | after retained wall us | dominant after phase |
|---|---:|---:|---:|---:|---|
| retained `COUNT(*)` | 36,451 | 38,395 | 4,868 | 3,541 | retained header/count lookup, 3,509 us |
| retained multi-column int4 lookup | 33,750 | 35,273 | 1,258 | 1,099 | match-index, 925 us |
| retained composite/text lookup | 2,807,225 | 33,665 | 1,264 | 1,150 | match-index, 909 us |

Before the conjunction-index fix, the composite/text endpoint facts showed
`client_visible_select_engine_execute_micros=2770946` while retained match
index, selected projection, and materialization were only `795`, `70`, and
`16` us respectively. That narrowed the dominant phase to CPU-side
access-path setup before the retained route's device match/projection work.

After the fix, composite/text endpoint facts show:

- `client_visible_select_retained_route_zero_h2d=true`
- `client_visible_select_retained_route_d2h_delta=62`
- `client_visible_select_retained_route_kernel_delta=6`
- `client_visible_select_engine_execute_micros=1264`
- `client_visible_select_retained_wall_micros=1150`
- `client_visible_select_retained_match_index_micros=909`
- `client_visible_select_retained_selected_projection_micros=96`
- `client_visible_select_retained_result_materialization_micros=15`
- `client_visible_select_retained_matched_rows=1`

## Load Evidence

The after-fix 1,048,576-row probe still measured GPU DB endpoint load below
Richard's `>=30k rows/sec` benchmark target:

- default PostgreSQL load: `1,048,576` rows in `17,829 ms`, `58,812 rows/sec`
- GPU DB endpoint load wall time: `1,048,576` rows in `49,275 ms`,
  `21,280 rows/sec`
- GPU DB minimum met: `false`

The load result is not a new COPY optimization proof; it is evidence that the
retained-query fix does not close the already-known COPY target blocker.

## Decision

The retained-query profile/fix lane is closed for this bounded slice. The
worker found and removed the dominant measured retained query setup costs at
1,048,576 rows. The result is enough to justify a later 10% retry for retained
query throughput, but the overall 10% benchmark remains blocked because GPU DB
COPY admission still needs to meet or defensibly recheck the `>=30k rows/sec`
sustained target at the 64,424,510-row target.

Safe continuation command:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-10pct-execution-v4 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=64424510 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55562 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55563 \
timeout 21600 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Do not run 25% or 125% from this state. The 25% scope remains deferred by
Richard's 10% pivot, and 125% remains blocked on
`missing_partitioned_over_resident_execution`.

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine p8_resident_route_executes_same_column_equality_projection -- --nocapture`: passed
- 1,048,576-row retained-query baseline probe: passed
- 1,048,576-row retained-query after-fix probe: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
