# P8 COPY Admission WAL/Value-Index Architecture

- date: 2026-06-02
- stream: benchmark
- milestone: 10% GPU DB COPY admission WAL/value-index architecture slice
- status: closed
- previous_blocker: `gpu_db_10pct_copy_admission_wal_value_index_architecture_required`
- code_change: skip generic KV state-machine clone/reparse for the current engine-applied COPY WAL entry
- bounded_artifact: `target/p8-10pct-copy-admission-30000-recheck-v3/identical-pgwire-target-smoke/`
- metrics_artifact: `target/p8-10pct-copy-admission-30000-recheck-v3/identical-pgwire-target-smoke/metrics.jsonl`
- endpoint_facts: `target/p8-10pct-copy-admission-30000-recheck-v3/identical-pgwire-target-smoke/endpoint-facts.txt`
- cleanup_status: benchmark cleanup passed; PostgreSQL Docker cleanup passed

## Architecture Direction

The smallest defensible slice was the WAL/current-apply boundary. The COPY fast
path already applies the current committed entry directly through
`Engine::apply_insert_with_profile(...)` inside
`commit_mutation_at_with_current_apply(...)`. Before this slice, the same
current entry was also passed through the generic `KvStateMachine::apply`, which
clones and attempts to parse the full multi-row SQL `INSERT` payload.

This change avoids that duplicate generic state-machine apply only for the
current entry handled by the caller-supplied Engine apply closure. Earlier
committed entries still go through the generic state machine plus normal MVCC
replay. Durable WAL records, local replication entries, commit ordering,
WAL-before-visibility, replay from `durable_wal_records()`, MVCC visibility,
value-index correctness, and retained warmup are preserved.

## Command

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-10pct-copy-admission-30000-recheck-v3 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=1048576 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55568 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55569 \
timeout 1800 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

## Result

The bounded proof met Richard's `>=30,000 rows/sec` GPU DB COPY target while
preserving the product boundary.

| target/profile | rows | elapsed ms | rows/sec | status |
|---|---:|---:|---:|---|
| default PostgreSQL load | 1,048,576 | 17,008 | 61,651 | pass |
| GPU DB retained endpoint load wall time | 1,048,576 | 33,746 | 31,072 | pass |
| GPU DB COPY chunk admission | 1,048,576 | measured over 128 chunks | 42,376 average | pass |

COPY chunk evidence:

- committed chunks: `128`
- committed rows: `1,048,576`
- chunk rows: `8,192`
- average chunk admission: `42,376 rows/sec`
- slowest chunk admission: `36,571 rows/sec`
- fastest chunk admission: `46,811 rows/sec`
- latest chunk admission: `42,226 rows/sec`
- `>=30,000 rows/sec` met: `true`

## Phase Evidence

| phase | previous average us/chunk | after average us/chunk |
|---|---:|---:|
| WAL commit/flush/current boundary | 122,677 | 421 |
| relational value-index append | 127,081 | 130,350 |
| durable SQL WAL payload render | 18,644 | 19,988 |
| MVCC tuple insertion/materialization | 22,132 | 23,714 |

The architecture slice removed the measured WAL/current boundary as a COPY
admission limiter. The remaining dominant phase is relational value-index
append at about `130 ms` per `8,192`-row chunk, but the bounded proof now clears
the explicit 30k rows/sec gate.

## Retained Query Health

The retained-query smoke remained healthy at concurrency `1`:

| query | p50 us | throughput qps | correctness | retained route |
|---|---:|---:|---|---|
| `COUNT(*)` | 37,013 | 20.152352 | pass | yes |
| multi-column int4 lookup | 33,162 | 21.586616 | pass | yes |
| composite/text lookup | 33,791 | 21.032263 | pass | yes |

Endpoint facts preserved retained-route execution, zero H2D for accepted query
rows, device-side match-index compaction, selected-row D2H readback, and owner
thread Engine/CUDA state.

## Correctness And Product Boundaries

This slice preserved:

- SQL-visible `CREATE TABLE` plus PostgreSQL-compatible `COPY FROM STDIN`
  through real `psql`/libpq
- bounded `8,192`-row decoded COPY chunks
- Engine WAL-before-visibility and durable replay from WAL records
- current-process decoded COPY apply
- normal MVCC `SELECT` visibility
- equality-filter correctness through the existing value-index path
- retained warmup from SQL-visible rows
- retained zero-H2D facts for accepted query rows
- owner-thread Engine/CUDA state and endpoint cleanup

## Decision

`gpu_db_10pct_copy_admission_wal_value_index_architecture_required` is closed
for this bounded slice. The 1,048,576-row proof met the 30k rows/sec gate, but
the full-load wall margin is narrow enough that this worker stopped at the
contract boundary instead of launching the full 10% retry in the same slice.

The safe next command is the 10% single-load retry capped at concurrency
`1,2,4,8,16,32,64`:

```bash
GPU_DB_CH_BENCH_OUT_DIR=target/p8-identical-10pct-execution-v4 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_ROWS=64424510 \
GPU_DB_CH_BENCH_IDENTICAL_PGWIRE_CONCURRENCY_TARGETS=1,2,4,8,16,32,64 \
GPU_DB_CH_BENCH_ENGINE_PGWIRE_PORT=55562 \
GPU_DB_CH_BENCH_PGSQL_DOCKER_PORT=55563 \
timeout 21600 scripts/run_p8_ch_benchmark_residency_probe.sh --identical-pgwire-target-smoke
```

Do not run 25%, 125%, or concurrency `128` from this state.

## Validation

- `bash -n scripts/run_p8_ch_benchmark_residency_probe.sh`: passed before changes
- `cargo fmt --all -- --check`: passed
- `cargo test -p gpu_db_engine relational_copy_rows_commit_through_engine_wal_mvcc -- --nocapture`: passed
- required 1,048,576-row endpoint proof: passed and met COPY target
- `scripts/run_p8_ch_benchmark_residency_probe.sh --cleanup`: passed
- `scripts/run_p8_ch_benchmark_residency_probe.sh --pgsql-baseline-docker-down`: passed
