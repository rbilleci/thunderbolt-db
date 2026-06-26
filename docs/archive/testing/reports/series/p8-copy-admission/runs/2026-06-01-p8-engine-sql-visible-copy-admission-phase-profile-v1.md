# P8 Engine SQL-Visible COPY Admission Phase Profile

- date: 2026-06-01
- stream: benchmark
- milestone: SQL-visible COPY admission phase profiling and bounded unblocker
- status: closed
- previous_blocker: `engine_sql_visible_copy_admission_storage_wal_profile_required`
- code_change: phase-profiled COPY admission and skipped the growing visible-row constraint scan for unconstrained INSERT/COPY tables
- before_fix_artifact: `target/p8-10pct-copy-path-storage-wal-profile-before-fix/identical-pgwire-target-smoke/endpoint-facts.txt`
- post_fix_artifact: `target/p8-10pct-copy-path-storage-wal-profile-probe/identical-pgwire-target-smoke/endpoint-facts.txt`
- next_lane: run or precisely block the 10pct single-load default/tuned/GPU retained curves

## Result

The worker added per-chunk COPY admission phase facts to the engine-backed
pgwire endpoint and used them to identify the decaying phase. The dominant
measured limiter was not protocol parsing, MVCC tuple insertion, value-index
append, SQL rendering, or residency invalidation. It was the generic
`preflight_unique_index_constraints(...)` `INSERT` branch scanning all visible
table rows for every COPY chunk even when the benchmark table had no unique
indexes, check constraints, foreign keys, or referencing foreign keys.

The fix keeps the pre-WAL validation boundary for column existence, value count,
type checks, and supported defaults. It only skips the all-visible-row
constraint candidate scan when there are no constraints to enforce. Tables with
unique indexes, check constraints, owned foreign keys, or referencing foreign
keys still use the existing preflight path before WAL commit.

## Phase Evidence

The before-fix profile used the same required 1,048,576-row probe. It completed
functionally but reproduced the blocker:

- committed chunks: `128`
- committed rows: `1,048,576`
- total measured GPU DB COPY chunk time: `255,549` ms
- average COPY throughput over committed chunks: about `4,103` rows/sec
- first chunk: `25,440` rows/sec
- latest/slowest chunk: `2,233` rows/sec
- latest-rate 10pct COPY-only projection: about `8.01` hours
- average-rate 10pct COPY-only projection: about `4.36` hours
- dominant measured phase: `copy_profile_unique_preflight_micros`
  - average: `1,687,066` us per chunk
  - latest: `3,370,585` us per chunk
- stable smaller phases:
  - SQL WAL payload rendering average: `24,322` us
  - WAL/commit/flush boundary average: `124,707` us
  - MVCC insertion average: `22,822` us
  - value-index append average: `120,401` us
  - residency invalidation average: about `1` us

After the guard, the required bounded probe completed with the same 128 bounded
COPY chunks and retained-route query evidence:

- committed chunks: `128`
- committed rows: `1,048,576`
- maximum buffered decoded rows: `8,192`
- retained warmup row count: `1,048,576`
- total measured GPU DB COPY chunk time: `40,591` ms
- average COPY throughput over committed chunks: about `25,833` rows/sec
- first chunk: `27,215` rows/sec
- latest chunk: `26,089` rows/sec
- slowest chunk: `22,382` rows/sec
- latest-rate 10pct COPY-only projection: about `0.69` hours for `64,424,510` rows
- average-rate 10pct COPY-only projection: about `0.69` hours

Post-fix phase facts:

- `copy_profile_unique_preflight_micros`
  - average: `9,573` us
  - latest: `9,340` us
- `copy_profile_wal_commit_flush_boundary_micros`
  - average: `129,698` us
  - latest: `127,733` us
- `copy_profile_value_index_append_micros`
  - average: `125,137` us
  - latest: `124,369` us
- `copy_profile_mvcc_insert_micros`
  - average: `23,330` us
  - latest: `22,505` us
- `copy_profile_render_sql_wal_payload_micros`
  - average: `19,883` us
  - latest: `18,823` us
- `copy_profile_row_prepare_micros`
  - average: `4,995` us
  - latest: `4,850` us
- `copy_profile_check_preflight_micros`: `0`
- `copy_profile_foreign_key_preflight_micros`: `0`
- `copy_profile_residency_invalidation_micros`: `0` average

## Correctness And Retained Route Evidence

The post-fix probe preserved the required product boundary:

- SQL-visible `COPY FROM STDIN` through real `psql`/libpq
- durable SQL `INSERT` WAL payloads for replay
- current-process decoded COPY apply
- reserved generated row-key MVCC insertion
- grouped relational value-index append
- normal MVCC `SELECT` visibility
- retained warmup from SQL-visible rows
- retained `COUNT(*)` accepted with zero H2D
- retained multi-column int4 lookup accepted with zero H2D and match-index compaction
- retained composite/text lookup accepted with zero H2D and match-index compaction

The focused engine test still covers COPY rows through WAL/MVCC, normal SELECT
visibility, equality-index access, duplicate-key rejection, and replay.

## Decision

`engine_sql_visible_copy_admission_storage_wal_profile_required` is closed for
this bounded slice. The measured dominant decaying phase was the unnecessary
constraint candidate scan for unconstrained benchmark COPY chunks, and the
smallest replay-preserving fix removes that scan without weakening supported
constraint/index semantics.

The full 10pct curves were not run in this worker slice. The safe next command
for the supervisor to consider is:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-10pct-execution-v3 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=64424510 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64,128 \
scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Do not run 25pct or 125pct curves from this report. The 25pct scope remains
deferred by Richard's 10pct pivot, and 125pct remains blocked on
`missing_partitioned_over_resident_execution`.

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine relational_copy_rows_commit_through_engine_wal_mvcc -- --nocapture`: passed
- required 1,048,576-row endpoint probe before fix: passed functionally and identified the dominant phase
- required 1,048,576-row endpoint probe after fix: passed with stable throughput and retained-route query evidence
