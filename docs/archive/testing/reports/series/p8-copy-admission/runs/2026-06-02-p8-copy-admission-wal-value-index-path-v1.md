# P8 COPY Admission WAL/Value-Index Path

- date: 2026-06-02
- stream: benchmark
- milestone: 10% GPU DB COPY admission WAL/value-index optimization and recheck
- status: blocked
- previous_blocker: `gpu_db_10pct_copy_admission_wal_value_index_path_required`
- narrowed_blocker: `gpu_db_10pct_copy_admission_wal_value_index_architecture_required`
- bounded_artifact: `target/p8-10pct-copy-admission-30000-recheck-v2/identical-pgwire-target-smoke/`
- metrics_artifact: `target/p8-10pct-copy-admission-30000-recheck-v2/identical-pgwire-target-smoke/metrics.jsonl`
- endpoint_facts: `target/p8-10pct-copy-admission-30000-recheck-v2/identical-pgwire-target-smoke/endpoint-facts.txt`
- cleanup_status: benchmark cleanup passed; PostgreSQL Docker cleanup passed

## Command

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-admission-30000-recheck-v2 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55568 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55569 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

## Result

The bounded proof preserved the product boundary but did not meet Richard's
`>=30,000 rows/sec` GPU DB COPY target.

Load metrics:

| target/profile | rows | elapsed ms | rows/sec | status |
|---|---:|---:|---:|---|
| default PostgreSQL load | 1,048,576 | 16,386 | 63,992 | pass |
| GPU DB retained endpoint load wall time | 1,048,576 | 48,544 | 21,600 | below target |
| GPU DB COPY chunk admission | 1,048,576 | measured over 128 chunks | 26,609 average | below target |

COPY chunk evidence:

- committed chunks: `128`
- committed rows: `1,048,576`
- chunk rows: `8,192`
- average chunk admission: `26,609 rows/sec`
- slowest chunk admission: `24,236 rows/sec`
- fastest chunk admission: `29,049 rows/sec`
- latest chunk admission: `25,761 rows/sec`
- `>=30,000 rows/sec` met: `false`

The retained-query health smoke remained healthy at this bounded size:

- retained `COUNT(*)`: p50 `34,616 us`, zero errors, retained route
- retained multi-column int4 lookup: p50 `35,425 us`, zero errors, retained route
- retained composite/text lookup: p50 `33,475 us`, zero errors, retained route
- composite/text endpoint facts: zero H2D, `62` D2H byte delta, match-index
  compaction true, matched rows `1`

## Phase Evidence

Average per-chunk phase timings from `endpoint-facts.txt`:

| phase | average us | latest us |
|---|---:|---:|
| durable SQL WAL payload render | 18,644 | 18,448 |
| WAL commit/flush/current boundary | 122,677 | 126,155 |
| row/default preparation | 4,656 | latest fact present in artifact |
| unique/index preflight | 8,924 | 9,534 |
| check preflight | 0 | 0 |
| foreign-key preflight | 0 | 0 |
| MVCC tuple insertion/materialization | 22,132 | 22,436 |
| relational value-index append | 127,081 | 130,616 |
| residency invalidation/post-commit | 0 | 0 |

The measured limiter is still the same two-part boundary:

- WAL/proposal/flush/current-apply boundary for a durable SQL `INSERT` payload.
- Relational value-index maintenance for every copied row and supported column.

A scoped local value-index grouping experiment was rejected during this round:
changing entry construction to column-major value grouping still produced only
`26,609 rows/sec` average chunk admission and left value-index append at about
`127 ms` per chunk, so it was not kept.

## Correctness And Product Boundaries

The bounded run preserved:

- SQL-visible `CREATE TABLE` plus PostgreSQL-compatible `COPY FROM STDIN`
  through real `psql`/libpq
- Engine WAL-before-visibility and durable SQL `INSERT` WAL payload replay
  boundary
- current-process decoded COPY apply
- normal MVCC `SELECT` visibility
- retained warmup from SQL-visible rows
- retained zero-H2D facts for accepted query rows
- owner-thread Engine/CUDA state and endpoint lifecycle

## Decision

This round is blocked. The smallest obvious value-index allocation/grouping
change did not move the measured path enough to justify a full 10% retry. The
remaining defensible continuation is broader than a local grouping tweak:
either a replay-preserving compact/batched COPY WAL representation for this
path, a different value-index physical representation/build strategy, or a
combined design that keeps equality access-path correctness without per-chunk
all-column BTreeMap maintenance in the admission hot path.

Do not run 25%, 125%, or concurrency `128` from this state. Do not run the full
10% retry until bounded evidence meets the `>=30k rows/sec` COPY target with
margin for warmup, query curves, cleanup, and reporting.

Smallest continuation command after a WAL/value-index architecture slice:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-admission-30000-recheck-v3 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55568 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55569 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed before changes
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine relational_copy_rows_commit_through_engine_wal_mvcc -- --nocapture`: passed during the rejected local experiment
- required 1,048,576-row endpoint probe: passed functionally, missed COPY target
- `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
